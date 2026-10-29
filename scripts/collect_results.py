"""Collect the finished Kaggle runs of the main plan into results/runs.csv and results/per_class.csv (CLAUDE.md,
"Format hasil").

Source: the downloaded session outputs in runs/kaggle/<timestamp>/ (scripts/kaggle_download.py). Every session's
output holds all runs so far, so each run is taken from the newest download that has its eval_test.json.

For every finished run the test predictions saved on Kaggle (preds_test_coco.json) are scored again here with the
common pycocotools evaluator (scripts/coco_eval.py) and must reproduce the Kaggle numbers in eval_test.json; the
laptop's values are the ones written. Both CSVs are rebuilt from scratch on every call (no hand edits).

Precision / recall / F1 (CLAUDE.md, decided 29 Oct 2026): IoU 0.5, macro-averaged over the 15 classes, from the
COCO-format predictions (pycocotools matching, scripts/coco_eval.py). The confidence threshold is chosen per run on
the validation split (the one that maximises macro-F1) and then applied to the Kaggle test predictions; it is
written to runs.csv (conf_thr, with the validation macro-F1). Validation predictions were not saved on Kaggle, so
best.pt predicts the validation split here on the laptop with the test settings (conf 0.001, NMS IoU 0.7, FP16;
same batch for YOLO/RT-DETR, Faster R-CNN loader batch FRCNN_PRED_BATCH instead of 32, which does not fit in 4 GB). As a check, the test split is predicted here too: its mAP must match Kaggle's within MAP_TOL.
Laptop predictions are cached in runs/laptop_preds/<run>/.

fps stays empty until the final FPS measurement (results/fps.csv). train_time_min = sum of per-epoch train +
validation wall time on the Kaggle T4.

Usage: python scripts/collect_results.py
"""
import csv
import json
from pathlib import Path

import yaml

from coco_eval import evaluate, match_iou50, predict_ultralytics, prf, save_dets, select_threshold
from detectors import ROOT, load_detector

PLAN = yaml.safe_load(open(ROOT / "configs" / "kaggle_plan.yaml", encoding="utf-8"))["main"]
CLASSES = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
GROUP = {CLASSES["names"][i]: g for i, g in CLASSES["groups"].items()}
IMGSZ = yaml.safe_load(open(ROOT / "configs" / "train.yaml", encoding="utf-8"))["imgsz"]
RUN_COLS = ["model", "protocol", "fold", "seed", "epochs", "batch", "imgsz", "map50", "map50_95", "precision",
            "recall", "f1", "conf_thr", "val_f1", "fps", "params_M", "train_time_min", "checkpoint_path"]
CLASS_COLS = ["model", "protocol", "fold", "seed", "class", "ap50", "ap50_95", "group"]
TOL = 1e-9
MAP_TOL = 2e-3       # laptop (RTX 3050) vs Kaggle (T4) test predictions: FP16 kernels differ slightly
LAPTOP_PREDS = ROOT / "runs" / "laptop_preds"
FRCNN_PRED_BATCH = 8   # images per Faster R-CNN val/test batch on the laptop (Kaggle: 2 x 16 = 32)


def laptop_preds(model, split, part, best, batch, name):
    """COCO-format predictions of best.pt on <split>_<part>, made on this laptop exactly as on Kaggle."""
    path = LAPTOP_PREDS / name / f"preds_{part}_coco.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if model == "frcnn-r50fpn":
        import torch
        import train_frcnn
        dl, _ = train_frcnn.loaders(split, FRCNN_PRED_BATCH // 2, probe=False, fraction=1.0)   # val/test: x 2, rect
        net = train_frcnn.build_model(len(CLASSES["names"]))
        net.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["model"])
        dets = train_frcnn.predict(net.to("cuda"), dl[part], f"{split}_{part}", torch.device("cuda"))
        del net
        torch.cuda.empty_cache()
    else:
        dets = predict_ultralytics(best, f"{split}_{part}", batch=batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dets(dets, path)
    return dets


def find_run(model, name):
    sub = "frcnn" if model == "frcnn-r50fpn" else "ultralytics"
    for dl in sorted((ROOT / "runs" / "kaggle").iterdir(), reverse=True):   # newest download first
        for rdir in dl.glob(f"**/runs/{sub}/{name}"):
            if (rdir / "eval_test.json").exists():
                return rdir
    return None


def epoch_rows(model, rdir):
    with open(rdir / ("results.csv" if model == "frcnn-r50fpn" else "gpu_log.csv"), newline="") as f:
        return list(csv.DictReader(f))


def main():
    runs, per_class, n_params = [], [], {}
    for r in PLAN:
        model, split, seed = r["model"], r["split"], r["seed"]
        name = r.get("name") or f"{model}_{split}_seed{seed}"
        rdir = find_run(model, name)
        if rdir is None:
            print(f"{name:<28} not finished yet")
            continue
        ev = json.loads((rdir / "eval_test.json").read_text(encoding="utf-8"))
        dets = json.loads((rdir / "preds_test_coco.json").read_text(encoding="utf-8"))
        coco = evaluate(f"{split}_test", dets)
        kag = ev["pycocotools"]
        diffs = [abs(coco[k] - kag[k]) for k in ("map50", "map50_95")]
        diffs += [abs(a - b) for c in coco["per_class"] for a, b in zip(coco["per_class"][c], kag["per_class"][c])]
        if max(diffs) > TOL:
            raise SystemExit(f"{name}: laptop pycocotools differs from Kaggle's eval_test.json by {max(diffs):.2e}")
        best = rdir / "weights" / "best.pt"
        # P / R / F1: threshold from the validation split, applied to Kaggle's test predictions
        val_dets = laptop_preds(model, split, "val", best, ev["batch"], name)
        thr, _, _, val_f1 = select_threshold(match_iou50(f"{split}_val", val_dets))
        prec, rec, f1 = (float(x[0]) for x in prf(match_iou50(f"{split}_test", dets), [thr]))
        lap = evaluate(f"{split}_test", laptop_preds(model, split, "test", best, ev["batch"], name))
        lap_diff = max(abs(lap[k] - coco[k]) for k in ("map50", "map50_95"))
        if lap_diff > MAP_TOL:
            raise SystemExit(f"{name}: laptop test mAP differs from Kaggle's by {lap_diff:.4f}")
        if model not in n_params:
            n_params[model] = load_detector(model, best, conf=0.001).n_params / 1e6
        ep = epoch_rows(model, rdir)
        protocol, fold = ("P1", "") if split == "p1" else ("LOSO", split.split("_")[1])
        runs.append({"model": model, "protocol": protocol, "fold": fold, "seed": seed, "epochs": len(ep),
                     "batch": ev["batch"], "imgsz": IMGSZ, "map50": round(coco["map50"], 5),
                     "map50_95": round(coco["map50_95"], 5), "precision": round(prec, 5), "recall": round(rec, 5),
                     "f1": round(f1, 5), "conf_thr": round(thr, 5), "val_f1": round(val_f1, 5), "fps": "",
                     "params_M": round(n_params[model], 3),
                     "train_time_min": round(sum(float(e["train_val_s"]) for e in ep) / 60, 1),
                     "checkpoint_path": best.relative_to(ROOT).as_posix()})
        for c, (ap50, ap) in coco["per_class"].items():
            per_class.append({"model": model, "protocol": protocol, "fold": fold, "seed": seed, "class": c,
                              "ap50": round(ap50, 5), "ap50_95": round(ap, 5), "group": GROUP[c]})
        print(f"{name:<28} epochs {len(ep):3d}  mAP50 {coco['map50']:.4f}  mAP50-95 {coco['map50_95']:.4f}  "
              f"(= Kaggle, max diff {max(diffs):.1e})  thr {thr:.3f} (val F1 {val_f1:.4f})  test P {prec:.4f} "
              f"R {rec:.4f} F1 {f1:.4f}  laptop test mAP diff {lap_diff:.1e}")
    for path, cols, rows in ((ROOT / "results" / "runs.csv", RUN_COLS, runs),
                             (ROOT / "results" / "per_class.csv", CLASS_COLS, per_class)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print(f"{len(runs)} runs -> results/runs.csv, {len(per_class)} rows -> results/per_class.csv")


if __name__ == "__main__":
    main()
