"""Check annotation coverage: every extracted frame must be labelled or explicitly excluded.

Reads data/frames/, data/labels_yolo/ (LabelImg YOLO .txt) and data/excluded_frames.csv.
Writes nothing except the report results/annotation_check.csv. Exit code 1 if any issue.

Issue types:
  missed          frame has no label (or an empty label file) and is not in the exclusion list
  conflict        frame has a box but is also in the exclusion list
  multi_box       more than one box (rule: one box per frame)
  class_mismatch  box class id differs from the gloss in the file name
  bad_box         malformed line, or box outside [0, 1] / zero size
  orphan_label    label file without a matching frame
  excl_unknown    exclusion entry for a file that does not exist in data/frames/
  excl_reason     exclusion reason not in a/b/c
  excl_duplicate  file listed more than once in the exclusion list
  classes_txt     data/labels_yolo/classes.txt does not match configs/classes.yaml
  frame_count     a video has a number of frames different from frames_per_video

Usage: python scripts/check_annotations.py [--show 20]
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
LABELS = ROOT / "data" / "labels_yolo"
REPORT = ROOT / "results" / "annotation_check.csv"
FRAME_RE = re.compile(r"^s(\d+)_([a-z]+)_v(\d{2})_f(\d{4})$")


def load_configs():
    classes = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
    prep = yaml.safe_load(open(ROOT / "configs" / "preprocess.yaml", encoding="utf-8"))
    return classes, prep


def read_label(path):
    """Return (boxes, errors); boxes are (class_id, xc, yc, w, h)."""
    boxes, errors = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            cid, xc, yc, w, h = int(parts[0]), *map(float, parts[1:5])
            if len(parts) != 5:
                raise ValueError
        except (ValueError, IndexError):
            errors.append(f"line {n}: malformed '{line.strip()}'")
            continue
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= xc - w / 2 + 1e-6 and xc + w / 2 - 1e-6 <= 1
                and 0 <= yc - h / 2 + 1e-6 and yc + h / 2 - 1e-6 <= 1):
            errors.append(f"line {n}: box outside image or zero size '{line.strip()}'")
        boxes.append((cid, xc, yc, w, h))
    return boxes, errors


def read_exclusions(path, valid_reasons, issues):
    excluded = {}
    if not path.exists():
        return excluded
    with open(path, newline="", encoding="utf-8-sig") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            name = (row.get("file") or "").strip()
            reason = (row.get("reason") or "").strip().lower()
            if not name:
                continue
            stem = Path(name).stem
            if reason not in valid_reasons:
                issues.append(("excl_reason", stem, f"row {n}: reason '{reason}' not in {sorted(valid_reasons)}"))
            if stem in excluded:
                issues.append(("excl_duplicate", stem, f"row {n}: already listed"))
            excluded[stem] = reason
    return excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=20, help="examples printed per issue type")
    args = ap.parse_args()

    classes, prep = load_configs()
    slug_to_id = {name.lower().replace(" ", ""): int(k) for k, name in classes["names"].items()}
    valid_reasons = set(prep["exclusion"]["reasons"])
    per_video = prep["sampling"]["frames_per_video"]
    issues = []

    frames = sorted(p.stem for p in FRAMES.glob("*.jpg"))
    frame_set = set(frames)
    excluded = read_exclusions(ROOT / prep["exclusion"]["list_file"], valid_reasons, issues)

    classes_txt = LABELS / "classes.txt"
    if classes_txt.exists():
        listed = [l.strip() for l in classes_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        expected = [classes["names"][i] for i in range(classes["nc"])]
        if listed != expected:
            issues.append(("classes_txt", "classes.txt", f"order {listed} != {expected}"))

    status = {}  # frame stem -> labelled / excluded / missed / conflict
    videos = defaultdict(list)
    for stem in frames:
        m = FRAME_RE.match(stem)
        if not m or m.group(2) not in slug_to_id:
            issues.append(("bad_box", stem, "frame name does not follow s{signer}_{gloss}_v{nn}_f{nnnn}"))
            continue
        signer, slug, sample, _ = m.groups()
        videos[(signer, slug, sample)].append(stem)
        lab = LABELS / f"{stem}.txt"
        boxes, errors = read_label(lab) if lab.exists() else ([], [])
        for e in errors:
            issues.append(("bad_box", stem, e))
        if boxes and stem in excluded:
            status[stem] = "conflict"
            issues.append(("conflict", stem, f"{len(boxes)} box(es) but excluded (reason {excluded[stem]})"))
        elif boxes:
            status[stem] = "labelled"
            if len(boxes) > 1:
                issues.append(("multi_box", stem, f"{len(boxes)} boxes"))
            wrong = {b[0] for b in boxes} - {slug_to_id[slug]}
            if wrong:
                issues.append(("class_mismatch", stem, f"class id {sorted(wrong)} but gloss '{slug}' = {slug_to_id[slug]}"))
        elif stem in excluded:
            status[stem] = "excluded"
        else:
            status[stem] = "missed"
            issues.append(("missed", stem, "empty label file" if lab.exists() else "no label file"))

    for stem in sorted(set(excluded) - frame_set):
        issues.append(("excl_unknown", stem, "not found in data/frames/"))
    for p in sorted(LABELS.glob("*.txt")):
        if p.name != "classes.txt" and p.stem not in frame_set:
            issues.append(("orphan_label", p.stem, "no matching frame"))
    for key, stems in sorted(videos.items()):
        if len(stems) != per_video:
            issues.append(("frame_count", f"s{key[0]}_{key[1]}_v{key[2]}", f"{len(stems)} frames, expected {per_video}"))

    # Report
    REPORT.parent.mkdir(exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue", "item", "detail"])
        w.writerows(issues)

    counts = Counter(status.values())
    print(f"frames: {len(frames)}  videos: {len(videos)}")
    print(f"  labelled {counts['labelled']}  excluded {counts['excluded']}  "
          f"missed {counts['missed']}  conflict {counts['conflict']}")
    reasons = Counter(excluded[s] for s, st in status.items() if st == "excluded")
    print("  excluded by reason:", {r: reasons.get(r, 0) for r in sorted(valid_reasons)})
    done = counts["labelled"] + counts["excluded"]
    print(f"  progress: {done}/{len(frames)} frames decided ({done / max(len(frames), 1):.1%})")

    if counts["missed"] == 0 and counts["conflict"] == 0:
        empty = [f"s{k[0]}_{k[1]}_v{k[2]}" for k, st in videos.items()
                 if not any(status.get(s) == "labelled" for s in st)]
        kept = Counter()
        for s, st in status.items():
            if st == "labelled":
                m = FRAME_RE.match(s)
                kept[(m.group(2), "s" + m.group(1))] += 1
        print(f"  videos with no kept frame: {len(empty)} {empty[:args.show]}")
        print("  kept frames per gloss x signer:")
        signers = sorted({k[1] for k in kept})
        for slug in sorted(slug_to_id, key=slug_to_id.get):
            print(f"    {slug_to_id[slug]:2d} {slug:<13}" + " ".join(f"{sg}={kept[(slug, sg)]:3d}" for sg in signers))

    by_type = defaultdict(list)
    for t, item, detail in issues:
        by_type[t].append((item, detail))
    print("\nissues:" if issues else "\nissues: none")
    for t, rows in by_type.items():
        print(f"  {t}: {len(rows)}")
        for item, detail in rows[:args.show]:
            print(f"    {item}  {detail}")
        if len(rows) > args.show:
            print(f"    ... {len(rows) - args.show} more in {REPORT.relative_to(ROOT)}")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
