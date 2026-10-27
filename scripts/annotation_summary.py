"""Summarise the annotation: frames used / excluded per class, per signer and per reason, and how the
reviewed boxes relate to the pre-annotation (accepted unchanged, corrected, drawn manually).

Read-only on data/; writes results/annotation_summary.csv (long format: one row per group value).
A frame counts as "used" only when its label was reviewed in LabelImg (same rule as check_annotations.py).

Usage: python scripts/annotation_summary.py
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from annot_common import AUTO, EXCLUDED, FRAMES, LABELS, REVIEWED, ROOT, label_digest, label_lines, read_table
from check_annotations import FRAME_RE, load_configs, max_edge_shift_px, read_label

OUT = ROOT / "results" / "annotation_summary.csv"
COLS = ["used", "excl_a", "excl_b", "excl_c", "pending",
        "auto_unchanged", "auto_corrected", "manual_drawn", "auto_rejected"]


def frame_record(stem, excluded, reviewed):
    """Status and pre-annotation outcome of one frame."""
    auto = AUTO / f"{stem}.txt"
    if stem in excluded:
        # auto_rejected: the detector proposed a box but the frame was excluded
        return {f"excl_{excluded[stem]}": 1, "auto_rejected": int(auto.exists())}
    label = LABELS / f"{stem}.txt"
    rev = reviewed.get(stem)
    if not (label_lines(label) and rev and rev["sha1"] == label_digest(label)):
        return {"pending": 1}
    if not auto.exists():
        return {"used": 1, "manual_drawn": 1}
    a, b = read_label(auto)[0][0][1:], read_label(label)[0][0][1:]
    with Image.open(FRAMES / f"{stem}.jpg") as im:
        w, h = im.size
    # LabelImg rounds to whole pixels when it re-saves, so allow a 2 px tolerance (as check_annotations.py).
    key = "auto_unchanged" if max_edge_shift_px(a, b, w, h) <= 2 else "auto_corrected"
    return {"used": 1, key: 1}


def print_table(title, rows, keys):
    print(f"\n{title}")
    head = f"{'':<14}" + "".join(f"{c:>9}" for c in ["used", "excl_a", "excl_b", "excl_c", "pending", "excl_%"])
    print(head)
    for k in keys:
        r = rows[k]
        excl = r["excl_a"] + r["excl_b"] + r["excl_c"]
        total = sum(r[c] for c in ["used", "excl_a", "excl_b", "excl_c", "pending"])
        print(f"{str(k):<14}" + "".join(f"{r[c]:>9}" for c in ["used", "excl_a", "excl_b", "excl_c", "pending"])
              + f"{100 * excl / total:>8.1f}%")


def main():
    classes, _ = load_configs()
    names = {int(k): v for k, v in classes["names"].items()}
    slug_to_id = {v.lower().replace(" ", ""): k for k, v in names.items()}
    groups = {int(k): v for k, v in classes["groups"].items()}
    excluded = {k: v["reason"].strip().lower() for k, v in read_table(EXCLUDED).items()}
    reviewed = read_table(REVIEWED)

    by = {"class": defaultdict(Counter), "signer": defaultdict(Counter), "group": defaultdict(Counter),
          "total": defaultdict(Counter)}
    used_per_video = Counter()
    videos = set()
    for p in sorted(FRAMES.glob("*.jpg")):
        signer, slug, vid = FRAME_RE.match(p.stem).group(1, 2, 3)
        cid = slug_to_id[slug]
        rec = frame_record(p.stem, excluded, reviewed)
        videos.add((signer, slug, vid))
        used_per_video[(signer, slug, vid)] += rec.get("used", 0) + rec.get("pending", 0)
        for g, key in (("class", f"{cid:02d} {names[cid]}"), ("signer", f"s{signer}"),
                       ("group", f"C_{groups[cid]}"), ("total", "all")):
            by[g][key].update(rec)

    for g, title in (("class", "Per kelas"), ("signer", "Per signer"), ("group", "Per kelompok"),
                     ("total", "Total")):
        print_table(title, by[g], sorted(by[g]))

    t = by["total"]["all"]
    n_auto = t["auto_unchanged"] + t["auto_corrected"]
    print(f"\nBox yang dipakai: {t['used']}  = diterima utuh {t['auto_unchanged']}"
          f" ({100 * t['auto_unchanged'] / max(t['used'], 1):.1f}%)"
          f" + dikoreksi {t['auto_corrected']} ({100 * t['auto_corrected'] / max(t['used'], 1):.1f}%)"
          f" + digambar manual {t['manual_drawn']} ({100 * t['manual_drawn'] / max(t['used'], 1):.1f}%)")
    print(f"Box otomatis yang ada: {n_auto + t['auto_rejected']}  -> dipakai {n_auto},"
          f" frame-nya dibuang {t['auto_rejected']}")

    empty = sorted(v for v in videos if used_per_video[v] == 0)
    print(f"\nVideo tanpa frame tersisa: {len(empty)}"
          + "".join(f"\n  s{s}_{g}_v{v}" for s, g, v in empty))
    hist = Counter(used_per_video[v] for v in videos)
    print("Frame dipakai per video: " + ", ".join(f"{k}: {hist[k]} video" for k in sorted(hist)))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["by", "key"] + COLS)
        for g in ("class", "signer", "group", "total"):
            for k in sorted(by[g]):
                w.writerow([g, k] + [by[g][k][c] for c in COLS])
    print(f"\nwritten: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
