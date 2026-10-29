"""Summary tables for Chapter 3 from results/runs.csv and results/per_class.csv (scripts/collect_results.py).
Aggregation rules: CLAUDE.md, "Format hasil" (locked 30 Oct 2026, config_log section 26).

Protocol 1 (--protocol P1):
  1. Per model: mean +- SD over the training seeds of mAP50, mAP50-95, precision, recall, F1 (IoU 0.5, macro,
     threshold from validation), parameters and training time.
  2. Per model: AP50-95 per class (mean over seeds), the mean over C_one and C_two, and
     Delta_two = mean AP50-95 (C_one) - mean AP50-95 (C_two) (manuscript eq. (21)), given as mean +- SD over seeds.
     Teman (the only class with hand-over-hand occlusion) is marked.

Protocol 2, leave-one-signer-out (--protocol LOSO; needs the finished P1 runs for mAP^rand):
  3.2 a. Per fold and model: mean +- SD over the 2 seeds.
      b. Per model: metric^SI (eq. 7) = mean over the folds of the per-fold seed means, +- SD over folds; the seed
         SD is reported separately as the mean over folds of the within-fold seed SD.
         Delta_abs = metric^rand - metric^SI and Delta_rel = Delta_abs / metric^rand x 100, with metric^rand the
         mean of the 2 P1 seeds; for mAP50, mAP50-95 and F1.
  3.3 AP50-95 per class = mean over folds x seeds (all runs of the complete folds), Delta_two from these class
      means; Delta_two per fold (from the per-fold seed means) to see whether its sign is consistent.
  3.4 Paired model comparison: per fold, the difference of the seed means (A - B); mean +- SD of the differences
      over folds and the number of folds whose order agrees with the order of the means. No significance test
      (n = 4 folds).
A fold counts for a model only when all its seeds are finished; the pairwise comparison uses the folds complete
for both models. Until all 4 folds are complete the tables are marked as preliminary.

SD is the sample standard deviation (ddof = 1); "–" when it is undefined (one value).

Output: results/summary_<p1|loso>.md (tables) and CSVs: summary_p1.csv, summary_p1_per_class.csv;
summary_loso.csv (per model, SI and Deltas), summary_loso_folds.csv, summary_loso_per_class.csv,
summary_loso_pairs.csv.

Usage: python scripts/summarise.py --protocol P1
       python scripts/summarise.py --protocol LOSO
"""
import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"yolo11m": "YOLO11m", "rtdetr-l": "RT-DETR-L", "frcnn-r50fpn": "Faster R-CNN R50-FPN"}
SEEDS = [0, 1]
FOLDS = ["s1", "s2", "s3", "s4"]
METRICS = [("map50", "mAP50", 3), ("map50_95", "mAP50-95", 3), ("precision", "Precision", 3),
           ("recall", "Recall", 3), ("f1", "F1", 3), ("params_M", "Params (M)", 1),
           ("train_time_min", "Training time (min)", 0)]
DET = METRICS[:5]                                   # detection metrics, reported per fold
GAP = [("map50", "mAP50"), ("map50_95", "mAP50-95"), ("f1", "F1")]   # Delta_abs / Delta_rel and pairwise
CLASSES = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
NAMES = [CLASSES["names"][i] for i in range(CLASSES["nc"])]
GROUP = {CLASSES["names"][i]: g for i, g in CLASSES["groups"].items()}
OCCLUSION = CLASSES["names"][CLASSES["occlusion_class"]]


def read(name):
    with open(ROOT / "results" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    if rows:
        with open(ROOT / "results" / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def fmt_sd(v, nd):
    return "–" if math.isnan(v) else f"{v:.{nd}f}"


def pm(x, nd):
    return f"{np.mean(x):.{nd}f} ± {fmt_sd(sd(x), nd)}"


def r5(v):
    return round(float(v), 5)


def group_means(ap_of_class):
    """Mean AP over C_one and C_two and Delta_two (eq. 21) from a {class: AP} mapping."""
    one = float(np.mean([ap_of_class[c] for c in NAMES if GROUP[c] == "one"]))
    two = float(np.mean([ap_of_class[c] for c in NAMES if GROUP[c] == "two"]))
    return one, two, one - two


def complete(runs, model, fold):
    seeds = sorted(int(r["seed"]) for r in runs if r["model"] == model and r["fold"] == fold)
    return seeds == SEEDS


def summarise_p1(runs, pcs):
    done, missing = [], []
    for m in MODELS:
        seeds = sorted(int(r["seed"]) for r in runs if r["model"] == m)
        (done if seeds == SEEDS else missing).append((m, seeds))
    models = [m for m, _ in done]

    md, rows_main, rows_cls = ["# Summary, Protocol 1 (random frame split)", ""], [], []
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
        row = {"model": m, "protocol": "P1", "n_seeds": len(rr)}
        for k, _, _ in METRICS:
            row[f"{k}_mean"], row[f"{k}_sd"] = r5(np.mean(vals[k])), r5(sd(vals[k]))
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
            rows_cls.append({"model": m, "protocol": "P1", "class": c, "group": GROUP[c],
                             "ap50_95_mean": r5(np.mean(v)), "ap50_95_sd": r5(sd(v))})
        md.append(f"| {c}{mark} | {GROUP[c]} | " + " | ".join(cells) + " |")
    summary = {}
    for m in models:
        per_seed = list(zip(*(group_means({c: ap_[(m, s, c)] for c in NAMES}) for s in SEEDS)))
        summary[m] = dict(zip(("one", "two", "delta"), per_seed))
        for g, label in (("one", "mean C_one"), ("two", "mean C_two"), ("delta", "delta_two")):
            rows_cls.append({"model": m, "protocol": "P1", "class": label, "group": "",
                             "ap50_95_mean": r5(np.mean(summary[m][g])), "ap50_95_sd": r5(sd(summary[m][g]))})
    for g, label in (("one", "**Mean C_one** (11)"), ("two", "**Mean C_two** (4)"), ("delta", "**Δtwo**")):
        md.append(f"| {label} | | " + " | ".join(
            (f"{np.mean(summary[m][g]):+.3f} ± {fmt_sd(sd(summary[m][g]), 3)}" if g == "delta"
             else pm(summary[m][g], 3)) for m in models) + " |")
    return md, {"summary_p1.csv": rows_main, "summary_p1_per_class.csv": rows_cls}


def summarise_loso(runs, pcs, p1_runs):
    p1_models = [m for m in MODELS if sorted(int(r["seed"]) for r in p1_runs if r["model"] == m) == SEEDS]
    folds = {m: [f for f in FOLDS if complete(runs, m, f)] for m in MODELS}
    models = [m for m in MODELS if folds[m] and m in p1_models]
    n_all = len(FOLDS)

    md = ["# Summary, Protocol 2 (signer-independent, leave-one-signer-out)", ""]
    md += [f"Per fold: mean ± SD over training seeds {SEEDS}. SI = mean over folds of the per-fold seed means, ± SD "
           "over folds; seed SD = mean over folds of the within-fold seed SD. rand = mean of the P1 seeds. "
           "Δabs = rand − SI, Δrel = Δabs / rand × 100. Test split of each fold; pycocotools; P, R, F1 at IoU 0.5, "
           "macro, threshold from the fold's validation split. SD: sample SD; – = undefined (one value).", ""]
    status = ", ".join(f"{MODELS[m]} {len(folds[m])}/{n_all} ({' '.join(folds[m]) or 'none'})" for m in MODELS)
    md += [f"Complete folds (both seeds): {status}."]
    if any(len(folds[m]) < n_all for m in MODELS):
        md += ["**Preliminary: not all folds are complete; SI and the comparisons cover only the folds listed.**"]
    md.append("")

    # per-fold seed means and SDs: fm[m][f][k] = list of seed values
    vals = {m: {f: {k: [float(r[k]) for r in runs if r["model"] == m and r["fold"] == f] for k, _, _ in DET}
                for f in folds[m]} for m in models}
    fold_mean = {m: {f: {k: float(np.mean(v)) for k, v in vals[m][f].items()} for f in folds[m]} for m in models}

    # ---- 3.2 a: per fold
    rows_fold = []
    md += ["## 3.2 Random split vs signer-independent", "", "### Per fold (mean ± SD over seeds)", "",
           "| Model | Fold (test signer) | " + " | ".join(lbl for _, lbl, _ in DET) + " |",
           "|---|---" + "|---" * len(DET) + "|"]
    for m in models:
        for f in folds[m]:
            md.append(f"| {MODELS[m]} | {f} | " + " | ".join(pm(vals[m][f][k], nd) for k, _, nd in DET) + " |")
            row = {"model": m, "fold": f, "n_seeds": len(SEEDS)}
            for k, _, _ in DET:
                row[f"{k}_mean"], row[f"{k}_sd"] = r5(np.mean(vals[m][f][k])), r5(sd(vals[m][f][k]))
            rows_fold.append(row)

    # ---- 3.2 b: SI, seed SD, rand, Delta_abs, Delta_rel
    rows_main = []
    md += ["", "### Signer-independent (eq. 7) vs random split", "",
           "| Model | Folds | Metric | SI (± SD over folds) | Seed SD (mean) | rand (P1, ± SD over seeds) | Δabs | Δrel (%) |",
           "|---|---|---|---|---|---|---|---|"]
    for m in models:
        p1 = [r for r in p1_runs if r["model"] == m]
        for k, lbl, nd in DET:
            fm = [fold_mean[m][f][k] for f in folds[m]]
            seed_sd = float(np.mean([sd(vals[m][f][k]) for f in folds[m]]))
            rand = [float(r[k]) for r in p1]
            si, rd = float(np.mean(fm)), float(np.mean(rand))
            dabs = rd - si
            drel = 100 * dabs / rd
            gap = k in dict(GAP)
            md.append(f"| {MODELS[m]} | {len(folds[m])}/{n_all} | {lbl} | {pm(fm, nd)} | {seed_sd:.{nd}f} | "
                      f"{pm(rand, nd)} | " + (f"{dabs:+.{nd}f} | {drel:+.1f} |" if gap else "| |"))
            rows_main.append({"model": m, "metric": k, "n_folds": len(folds[m]), "folds": " ".join(folds[m]),
                              "si_mean": r5(si), "si_sd_folds": r5(sd(fm)), "seed_sd_mean": r5(seed_sd),
                              "rand_mean": r5(rd), "rand_sd": r5(sd(rand)),
                              "delta_abs": r5(dabs) if gap else "", "delta_rel_pct": round(drel, 3) if gap else ""})

    # ---- 3.3: per class (folds x seeds), Delta_two pooled and per fold
    ap_ = {(r["model"], r["fold"], int(r["seed"]), r["class"]): float(r["ap50_95"]) for r in pcs}
    rows_cls = []
    md += ["", "## 3.3 Per-class AP50-95 and Δtwo (signer-independent)", "",
           "AP50-95 per class = mean over the complete folds × seeds. Δtwo = mean C_one − mean C_two (eq. 21) from "
           "these class means; positive = two-handed glosses are harder. P1 = mean over the P1 seeds.", "",
           "| Class | Group | " + " | ".join(f"{MODELS[m]} SI | P1" for m in models) + " |",
           "|---|---" + "|---|---" * len(models) + "|"]
    p1_ap = {(r["model"], int(r["seed"]), r["class"]): float(r["ap50_95"])
             for r in read("per_class.csv") if r["protocol"] == "P1"}
    pooled, p1_cls = {}, {}
    for m in models:
        pooled[m] = {c: float(np.mean([ap_[(m, f, s, c)] for f in folds[m] for s in SEEDS])) for c in NAMES}
        p1_cls[m] = {c: float(np.mean([p1_ap[(m, s, c)] for s in SEEDS])) for c in NAMES}
    for c in NAMES:
        mark = " (occlusion)" if c == OCCLUSION else ""
        md.append(f"| {c}{mark} | {GROUP[c]} | " +
                  " | ".join(f"{pooled[m][c]:.3f} | {p1_cls[m][c]:.3f}" for m in models) + " |")
        for m in models:
            v = [ap_[(m, f, s, c)] for f in folds[m] for s in SEEDS]
            rows_cls.append({"model": m, "class": c, "group": GROUP[c], "n_runs": len(v),
                             "ap50_95_mean": r5(np.mean(v)), "ap50_95_sd_runs": r5(sd(v)),
                             "ap50_95_p1": r5(p1_cls[m][c])})
    g_si = {m: group_means(pooled[m]) for m in models}
    g_p1 = {m: group_means(p1_cls[m]) for m in models}
    for i, label in enumerate(("**Mean C_one** (11)", "**Mean C_two** (4)", "**Δtwo**")):
        f_ = "{:+.3f}" if i == 2 else "{:.3f}"
        md.append(f"| {label} | | " + " | ".join(
            f"{f_.format(g_si[m][i])} | {f_.format(g_p1[m][i])}" for m in models) + " |")
        for m in models:
            rows_cls.append({"model": m, "class": ("mean C_one", "mean C_two", "delta_two")[i], "group": "",
                             "n_runs": len(folds[m]) * len(SEEDS), "ap50_95_mean": r5(g_si[m][i]),
                             "ap50_95_sd_runs": "", "ap50_95_p1": r5(g_p1[m][i])})

    md += ["", "### Δtwo per fold (from the per-fold seed means)", "",
           "| Model | " + " | ".join(FOLDS) + " | Folds with Δtwo > 0 |", "|---" * (len(FOLDS) + 2) + "|"]
    rows_dtwo = []
    for m in models:
        d = {}
        for f in folds[m]:
            d[f] = group_means({c: float(np.mean([ap_[(m, f, s, c)] for s in SEEDS])) for c in NAMES})[2]
            rows_dtwo.append({"model": m, "class": f"delta_two {f}", "group": "", "n_runs": len(SEEDS),
                              "ap50_95_mean": r5(d[f]), "ap50_95_sd_runs": "", "ap50_95_p1": ""})
        md.append(f"| {MODELS[m]} | " + " | ".join(f"{d[f]:+.3f}" if f in d else "–" for f in FOLDS) +
                  f" | {sum(v > 0 for v in d.values())}/{len(d)} |")
    rows_cls += rows_dtwo

    # ---- 3.4: paired comparison per fold
    rows_pair = []
    md += ["", "## 3.4 Paired model comparison (signer-independent)", "",
           "Per fold: difference of the seed means (A − B). Mean ± SD of the differences over the folds complete for "
           "both models; 'Same order' = folds in which A − B has the sign of the mean difference. "
           "No significance test (n = 4 folds).", "",
           "| A − B | Folds | " + " | ".join(f"{lbl} Δ (mean ± SD) | Same order" for _, lbl in GAP) + " |",
           "|---|---" + "|---|---" * len(GAP) + "|"]
    for a, b in itertools.combinations(models, 2):
        common = [f for f in FOLDS if f in folds[a] and f in folds[b]]
        if not common:
            continue
        cells = []
        row = {"model_a": a, "model_b": b, "n_folds": len(common), "folds": " ".join(common)}
        for k, _ in GAP:
            diff = [fold_mean[a][f][k] - fold_mean[b][f][k] for f in common]
            mean = float(np.mean(diff))
            same = sum(np.sign(x) == np.sign(mean) for x in diff)
            cells.append(f"{mean:+.3f} ± {fmt_sd(sd(diff), 3)} | {same}/{len(common)}")
            row.update({f"{k}_diff_mean": r5(mean), f"{k}_diff_sd": r5(sd(diff)), f"{k}_same_order": same,
                        f"{k}_per_fold": " ".join(f"{x:+.5f}" for x in diff)})
        md.append(f"| {MODELS[a]} − {MODELS[b]} | {len(common)}/{n_all} | " + " | ".join(cells) + " |")
        rows_pair.append(row)

    return md, {"summary_loso.csv": rows_main, "summary_loso_folds.csv": rows_fold,
                "summary_loso_per_class.csv": rows_cls, "summary_loso_pairs.csv": rows_pair}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="P1", choices=["P1", "LOSO"])
    args = ap.parse_args()
    runs, pcs = read("runs.csv"), read("per_class.csv")
    sel = lambda rows: [r for r in rows if r["protocol"] == args.protocol]   # noqa: E731
    if args.protocol == "P1":
        md, out = summarise_p1(sel(runs), sel(pcs))
    else:
        md, out = summarise_loso(sel(runs), sel(pcs), [r for r in runs if r["protocol"] == "P1"])

    tag = args.protocol.lower()
    (ROOT / "results" / f"summary_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for name, rows in out.items():
        write(name, rows)
    print("\n".join(md))
    print(f"\n-> results/summary_{tag}.md, " + ", ".join(out))


if __name__ == "__main__":
    main()
