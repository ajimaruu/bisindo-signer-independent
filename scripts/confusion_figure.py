"""Manuscript figure: the three LOSO confusion matrices side by side in one PNG (300 dpi).

Reads the count matrices written by scripts/confusion_matrix.py --protocol LOSO
(results/confusion/confusion_loso_<model>.csv: 15 true classes + background row, 15 predicted classes + missed),
so the definition is the one of config_log section 27 (∅ as in manuscript eq. (19): the missed column and the
background row): per test image the highest-scoring detection with IoU >= 0.5
at the run's conf_thr, 4 folds x 2 seeds summed. Rows of the 15 classes are normalised to the number of ground-truth
frames of the class; the background row shows counts (detections with IoU < 0.5), unshaded so that it is not read
as a fraction.

The figure is drawn at the manuscript's text width (15.5 cm), so the font sizes below are the printed sizes.
The caption is not drawn into the image; it goes into the manuscript (see captions_loso.csv for the duplicate counts).

Output: results/confusion/confusion_loso_combined.png
Usage:  python scripts/confusion_figure.py [--width-cm 15.5]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "confusion"
MODELS = {"yolo11m": "YOLO11m", "rtdetr-l": "RT-DETR-L", "frcnn-r50fpn": "Faster R-CNN R50-FPN"}
# same sequential one-hue ramp as scripts/confusion_matrix.py (light -> dark blue)
CMAP = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED, RULE = "#1f1f1f", "#555555", "#9a9a9a"
SHOW_MIN = 0.05      # off-diagonal fractions below this are shaded but not printed


def read(model):
    with open(DIR / f"confusion_loso_{model}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    cols = rows[0][1:]
    names = [r[0] for r in rows[1:]]
    return names, cols, np.array([[int(v) for v in r[1:]] for r in rows[1:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-cm", type=float, default=15.5)
    ap.add_argument("--out", type=Path, default=DIR / "confusion_loso_combined.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6})
    data = {m: read(m) for m in MODELS}
    names, cols, _ = data["yolo11m"]
    nc = len(names) - 1                                   # 15 classes; last row = background
    for m, (n, c, _) in data.items():
        assert n == names and c == cols, f"{m}: class order differs"
    n_gt = data["yolo11m"][2][:nc].sum(axis=1)
    for m, (_, _, cm) in data.items():
        assert (cm[:nc].sum(axis=1) == n_gt).all(), f"{m}: ground-truth counts differ"

    w = args.width_cm / 2.54
    fig = plt.figure(figsize=(w, w * 0.47), dpi=300)
    left, right, bottom, top, gap = 0.16, 0.935, 0.235, 0.93, 0.012
    pw = (right - left - 2 * gap) / 3
    for k, (m, label) in enumerate(MODELS.items()):
        cm = data[m][2]
        frac = cm[:nc] / n_gt[:, None]
        shade = np.vstack([frac, np.zeros(nc + 1)])       # background row: counts, unshaded
        ax = fig.add_axes((left + k * (pw + gap), bottom, pw, top - bottom))
        im = ax.imshow(shade, cmap=CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for i in range(nc + 1):
            for j in range(nc + 1):
                v = cm[i, j]
                if v == 0:
                    continue
                if i == nc:
                    txt = str(v)
                else:
                    f = frac[i, j]
                    if j != i and f < SHOW_MIN:
                        continue
                    txt = "1" if f >= 0.995 else f"{f:.2f}"[1:]       # .87 style, as in the per-model figures
                dark = shade[i, j] > 0.55
                ax.text(j, i, txt, ha="center", va="center", fontsize=3.9,
                        color="#ffffff" if dark else INK, fontweight="bold" if i == j else "normal")
        ax.axhline(nc - 0.5, color=RULE, lw=0.5)
        ax.axvline(nc - 0.5, color=RULE, lw=0.5)
        ax.set_xticks(range(nc + 1), cols[:nc] + ["∅ (missed)"], rotation=90, fontsize=5.6)
        if k == 0:
            ax.set_yticks(range(nc + 1), [f"{c} ({n})" for c, n in zip(names[:nc], n_gt)] + ["∅ (background)"],
                          fontsize=5.6)
            ax.set_ylabel("True class (test frames × seeds)", fontsize=6.2, color=MUTED, labelpad=2)
        else:
            ax.set_yticks(range(nc + 1), [""] * (nc + 1))
        ax.set_title(f"({'abc'[k]}) {label}", fontsize=6.8, color=INK, pad=3)
        ax.tick_params(length=0, pad=1.5, colors=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text((left + right) / 2, 0.012, "Predicted class (highest-scoring detection with IoU ≥ 0.5 at the validation-selected confidence threshold)",
             ha="center", va="bottom", fontsize=6.2, color=MUTED)
    cax = fig.add_axes((right + 0.012, bottom + 0.06, 0.011, top - bottom - 0.12))
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    cb.outline.set_visible(False)
    cb.set_label("Row fraction", fontsize=5.6, color=MUTED, labelpad=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"-> {args.out} ({w:.2f} x {w * 0.47:.2f} in, 300 dpi)")


if __name__ == "__main__":
    main()
