"""Summary tables for Chapter 3 from results/runs.csv and results/per_class.csv (scripts/collect_results.py).

Protocol 1 (--protocol P1):
  1. Per model: mean +- SD over the training seeds of mAP50, mAP50-95, precision, recall, F1 (IoU 0.5, macro,
     threshold from validation), parameters and training time.
  2. Per model: AP50-95 per class (mean over seeds), the mean over C_one and C_two, and
     Delta_two = mean AP50-95 (C_one) - mean AP50-95 (C_two) (manuscript eq. (21)), given as mean +- SD over seeds.
     Teman (the only class with hand-over-hand occlusion) is marked.
SD is the sample standard deviation (ddof = 1) over seeds. A model is summarised only when all its seeds are
finished; otherwise it is listed as incomplete.

Output: results/summary_p1.md (tables), results/summary_p1.csv, results/summary_p1_per_class.csv.
LOSO (mean over folds, Delta_abs, Delta_rel) is not implemented yet.

Usage: python scripts/summarise.py --protocol P1
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"yolo11m": "YOLO11m", "rtdetr-l": "RT-DETR-L", "frcnn-r50fpn": "Faster R-CNN R50-FPN"}
SEEDS = [0, 1]
METRICS = [("map50", "mAP50", 3), ("map50_95", "mAP50-95", 3), ("precision", "Precision", 3),
           ("recall", "Recall", 3), ("f1", "F1", 3), ("params_M", "Params (M)", 1),
           ("train_time_min", "Training time (min)", 0)]
CLASSES = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
NAMES = [CLASSES["names"][i] for i in range(CLASSES["nc"])]
GROUP = {CLASSES["names"][i]: g for i, g in CLASSES["groups"].items()}
OCCLUSION = CLASSES["names"][CLASSES["occlusion_class"]]


def read(name):
    with open(ROOT / "results" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def pm(x, nd):
    return f"{np.mean(x):.{nd}f} ± {sd(x):.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="P1", choices=["P1"])
    args = ap.parse_args()
    runs = [r for r in read("runs.csv") if r["protocol"] == args.protocol]
    pcs = [r for r in read("per_class.csv") if r["protocol"] == args.protocol]

    done, missing = [], []
    for m in MODELS:
        seeds = sorted(int(r["seed"]) for r in runs if r["model"] == m)
        (done if seeds == SEEDS else missing).append((m, seeds))
    models = [m for m, _ in done]

    md, rows_main, rows_cls = [f"# Summary, Protocol {args.protocol[1:]} (random frame split)", ""], [], []
    md += [f"Mean ± SD over training seeds {SEEDS} (sample SD). Test split; pycocotools. Precision, recall and F1 at "
           "IoU 0.5, macro-averaged over the 15 classes, at the confidence threshold chosen on the validation split.",
           ""]
    if missing:
        md += ["Incomplete (not summarised): " + ", ".join(f"{MODELS[m]} (seeds done: {s})" for m, s in missing), ""]

    # ---- table 1: overall metrics
    md += ["| Model | " + " | ".join(lbl for _, lbl, _ in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for m in models:
        rr = [r for r in runs if r["model"] == m]
        vals = {k: [float(r[k]) for r in rr] for k, _, _ in METRICS}
        md.append(f"| {MODELS[m]} | " + " | ".join(pm(vals[k], nd) for k, _, nd in METRICS) + " |")
        row = {"model": m, "protocol": args.protocol, "n_seeds": len(rr)}
        for k, _, _ in METRICS:
            row[f"{k}_mean"], row[f"{k}_sd"] = round(float(np.mean(vals[k])), 5), round(sd(vals[k]), 5)
        rows_main.append(row)

    # ---- table 2: AP50-95 per class, groups, Delta_two
    ap_ = {(r["model"], int(r["seed"]), r["class"]): float(r["ap50_95"]) for r in pcs}
    md += ["", "AP50-95 per class (mean over seeds). Groups: C_two = two-handed, C_one = one-handed. "
           "Δtwo = mean C_one − mean C_two (eq. 21); positive = two-handed glosses are harder.", "",
           "| Class | Group | " + " | ".join(MODELS[m] for m in models) + " |", "|---|---" + "|---" * len(models) + "|"]
    for c in NAMES:
        mark = " (occlusion)" if c == OCCLUSION else ""
        cells = []
        for m in models:
            v = [ap_[(m, s, c)] for s in SEEDS]
            cells.append(f"{np.mean(v):.3f}")
            rows_cls.append({"model": m, "protocol": args.protocol, "class": c, "group": GROUP[c],
                             "ap50_95_mean": round(float(np.mean(v)), 5), "ap50_95_sd": round(sd(v), 5)})
        md.append(f"| {c}{mark} | {GROUP[c]} | " + " | ".join(cells) + " |")
    summary = {}
    for m in models:
        per_seed = {g: [np.mean([ap_[(m, s, c)] for c in NAMES if GROUP[c] == g]) for s in SEEDS] for g in ("one", "two")}
        per_seed["delta"] = [o - t for o, t in zip(per_seed["one"], per_seed["two"])]
        summary[m] = per_seed
        for g, label in (("one", "mean C_one"), ("two", "mean C_two"), ("delta", "delta_two")):
            rows_cls.append({"model": m, "protocol": args.protocol, "class": label, "group": "",
                             "ap50_95_mean": round(float(np.mean(per_seed[g])), 5),
                             "ap50_95_sd": round(sd(per_seed[g]), 5)})
    for g, label in (("one", "**Mean C_one** (11)"), ("two", "**Mean C_two** (4)"), ("delta", "**Δtwo**")):
        md.append(f"| {label} | | " + " | ".join(
            (f"{np.mean(summary[m][g]):+.3f} ± {sd(summary[m][g]):.3f}" if g == "delta"
             else f"{np.mean(summary[m][g]):.3f} ± {sd(summary[m][g]):.3f}") for m in models) + " |")

    tag = args.protocol.lower()
    (ROOT / "results" / f"summary_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for name, rows in ((f"summary_{tag}.csv", rows_main), (f"summary_{tag}_per_class.csv", rows_cls)):
        if rows:
            with open(ROOT / "results" / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print("\n".join(md))
    print(f"\n-> results/summary_{tag}.md, summary_{tag}.csv, summary_{tag}_per_class.csv")


if __name__ == "__main__":
    main()
