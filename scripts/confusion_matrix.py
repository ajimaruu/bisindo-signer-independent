"""Confusion matrix per model from the test predictions at the run's confidence threshold (config_log section 27).

Source: the laptop test export of each run (runs/laptop_preds/<run>/preds_test_coco.json, scripts/collect_results.py),
the same predictions behind the paper's mAP and P/R/F1, with the run's conf_thr from results/runs.csv.

Per test image (exactly one ground-truth box per frame, CLAUDE.md "Aturan bounding box"), keeping only detections
with score >= conf_thr:
  - row = true class; column = class of the highest-scoring kept detection with IoU >= 0.5 to the ground-truth box,
    or "missed" when there is none;
  - every kept detection with IoU < 0.5 to the ground-truth box counts once in the "background" row under its class
    (a false detection away from the hands).
Kept detections that overlap the ground truth but are not the highest-scoring one (duplicates, second labels) are not
counted in the matrix; their number is reported.

LOSO: summed over the folds complete for the model (both seeds) x 2 seeds; P1: summed over the 2 seeds.
Output: results/confusion/confusion_<p1|loso>_<model>.csv (counts) and .png (rows normalised to the number of
ground-truth images of the class; the background row shows counts).

Usage: python scripts/confusion_matrix.py --protocol LOSO
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"yolo11m": "YOLO11m", "rtdetr-l": "RT-DETR-L", "frcnn-r50fpn": "Faster R-CNN R50-FPN"}
SEEDS = [0, 1]
FOLDS = ["s1", "s2", "s3", "s4"]
IOU = 0.5
CLASSES = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
NAMES = [CLASSES["names"][i] for i in range(CLASSES["nc"])]
OUT = ROOT / "results" / "confusion"
# sequential one-hue ramp (light -> dark blue)
CMAP = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def iou(a, b):
    """IoU of two COCO boxes [x, y, w, h]."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def run_matrix(split, dets, thr):
    """Counts [nc + 1 rows (classes, background), nc + 1 columns (classes, missed)] and the ignored count."""
    nc = len(NAMES)
    gt = json.loads((ROOT / "data" / "coco" / f"{split}_test.json").read_text(encoding="utf-8"))
    gt_box = defaultdict(list)
    for a in gt["annotations"]:
        gt_box[a["image_id"]].append(a)
    by_img = defaultdict(list)
    for d in dets:
        if d["score"] >= thr:
            by_img[d["image_id"]].append(d)
    m, ignored = np.zeros((nc + 1, nc + 1), dtype=int), 0
    for img in gt["images"]:
        anns = gt_box[img["id"]]
        if len(anns) != 1:
            raise SystemExit(f"{split} image {img['file_name']}: {len(anns)} ground-truth boxes, expected 1")
        g = anns[0]
        hit = sorted((d for d in by_img[img["id"]] if iou(d["bbox"], g["bbox"]) >= IOU), key=lambda d: -d["score"])
        m[g["category_id"] - 1, hit[0]["category_id"] - 1 if hit else nc] += 1
        ignored += max(0, len(hit) - 1)
        for d in by_img[img["id"]]:
            if iou(d["bbox"], g["bbox"]) < IOU:
                m[nc, d["category_id"] - 1] += 1
    return m, ignored


def plot(m, title, path):
    nc = len(NAMES)
    n_gt = m[:nc].sum(axis=1, keepdims=True)
    frac = np.vstack([m[:nc] / n_gt, np.zeros((1, nc + 1))])
    fig, ax = plt.subplots(figsize=(8.6, 7.6), dpi=200)
    ax.imshow(frac, cmap=CMAP, vmin=0, vmax=1)
    for i in range(nc + 1):
        for j in range(nc + 1):
            v = m[i, j]
            if v == 0 or (i == nc and j == nc):
                continue
            txt = f"{v}" if i == nc else f"{frac[i, j]:.2f}".lstrip("0") if frac[i, j] < 1 else "1"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if frac[i, j] > 0.55 else "#1f1f1f")
    ax.set_xticks(range(nc + 1), NAMES + ["missed"], rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(nc + 1), [f"{c} ({n})" for c, n in zip(NAMES, n_gt[:, 0])] + ["background"], fontsize=7.5)
    ax.set_xlabel("Predicted (highest-scoring detection with IoU ≥ 0.5)", fontsize=8.5)
    ax.set_ylabel("True class (test images × seeds)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    ax.axhline(nc - 0.5, color="#888888", lw=0.6)
    ax.axvline(nc - 0.5, color="#888888", lw=0.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="LOSO", choices=["P1", "LOSO"])
    args = ap.parse_args()
    with open(ROOT / "results" / "runs.csv", newline="", encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if r["protocol"] == args.protocol]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.protocol.lower()
    for model, label in MODELS.items():
        rr = [r for r in runs if r["model"] == model]
        if args.protocol == "LOSO":
            folds = [f for f in FOLDS if sorted(int(r["seed"]) for r in rr if r["fold"] == f) == SEEDS]
            rr = [r for r in rr if r["fold"] in folds]
            scope = f"folds {' '.join(folds)} ({len(folds)}/{len(FOLDS)}) × seeds {SEEDS}"
        else:
            folds = ["p1"] if sorted(int(r["seed"]) for r in rr) == SEEDS else []
            scope = f"seeds {SEEDS}"
        if not folds:
            print(f"{label}: no complete fold, skipped")
            continue
        total, ignored = 0, 0
        for r in rr:
            split = "p1" if args.protocol == "P1" else f"loso_{r['fold']}"
            name = f"{model}_{split}_seed{r['seed']}"
            dets = json.loads((ROOT / "runs" / "laptop_preds" / name / "preds_test_coco.json").read_text())
            m, ig = run_matrix(split, dets, float(r["conf_thr"]))
            total, ignored = total + m, ignored + ig
        nc = len(NAMES)
        with open(OUT / f"confusion_{tag}_{model}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true \\ predicted"] + NAMES + ["missed"])
            for i, row in enumerate(total):
                w.writerow([(NAMES + ["background"])[i]] + list(row))
        correct = int(np.trace(total[:nc, :nc]))
        n_gt = int(total[:nc].sum())
        prelim = " (preliminary)" if args.protocol == "LOSO" and len(folds) < len(FOLDS) else ""
        plot(total, f"{label}, {args.protocol}{prelim}: {scope}, conf ≥ run threshold", OUT / f"confusion_{tag}_{model}.png")
        print(f"{label:<22} {scope}: {n_gt} test images, correct {correct} ({correct / n_gt:.3f}), "
              f"confused {n_gt - correct - int(total[:nc, nc].sum())}, missed {int(total[:nc, nc].sum())}, "
              f"background detections {int(total[nc].sum())}, ignored overlapping {ignored}")
    print(f"-> {OUT.relative_to(ROOT).as_posix()}/confusion_{tag}_<model>.csv / .png")


if __name__ == "__main__":
    main()
