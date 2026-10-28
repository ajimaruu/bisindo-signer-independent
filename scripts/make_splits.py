"""Create the Protocol 1 split and the 4 LOSO folds from the kept (reviewed, not excluded) frames.

Rules (CLAUDE.md, split rules):
  - P1: random frame-level split 70/15/15, stratified per class, seed 0. Frames of one video may land in
    different partitions on purpose (that leakage is what P1 measures).
  - LOSO fold k (k = 1..4): test = all kept frames of signer k. Validation = 15% of the *videos* of the
    training signers (signer0 included), stratified per class, seed 0; every frame of a video stays in one
    partition. The test signer is never used for validation.
Per class, partition sizes are round(fraction * n); the remainder goes to the last partition.

Output: splits/p1_{train,val,test}.txt, splits/loso_s{k}_{train,val,test}.txt (frame file names, one per
line, sorted) and results/split_counts.csv (frames per class per partition).
Refuses to run unless scripts/check_annotations.py passes.

Usage: python scripts/make_splits.py
"""
import csv
import random
import subprocess
import sys
from collections import Counter, defaultdict

from annot_common import EXCLUDED, FRAMES, LABELS, ROOT, label_lines, read_table
from check_annotations import FRAME_RE, load_configs

SPLITS = ROOT / "splits"
COUNTS = ROOT / "results" / "split_counts.csv"
SEED = 0
P1_FRACTIONS = (0.70, 0.15)       # train, val; test = remainder
LOSO_VAL_FRACTION = 0.15
TEST_SIGNERS = ("1", "2", "3", "4")


def kept_frames(slug_to_id):
    """{file name: (signer, class id, video key)} for every kept frame."""
    excluded = read_table(EXCLUDED)
    kept = {}
    for p in sorted(FRAMES.glob("*.jpg")):
        if p.stem in excluded or not label_lines(LABELS / f"{p.stem}.txt"):
            continue
        signer, slug, vid = FRAME_RE.match(p.stem).group(1, 2, 3)
        kept[p.name] = (signer, slug_to_id[slug], f"s{signer}_{slug}_v{vid}")
    return kept


def split_p1(kept):
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for f, (_, cid, _) in sorted(kept.items()):
        by_class[cid].append(f)
    parts = {"train": [], "val": [], "test": []}
    for cid in sorted(by_class):
        files = by_class[cid]
        rng.shuffle(files)
        n_tr = round(P1_FRACTIONS[0] * len(files))
        n_va = round(P1_FRACTIONS[1] * len(files))
        parts["train"] += files[:n_tr]
        parts["val"] += files[n_tr:n_tr + n_va]
        parts["test"] += files[n_tr + n_va:]
    return parts


def split_loso(kept, test_signer):
    rng = random.Random(SEED)
    videos = defaultdict(set)     # class id -> videos of the training signers that have kept frames
    for signer, cid, video in kept.values():
        if signer != test_signer:
            videos[cid].add(video)
    val_videos = set()
    for cid in sorted(videos):
        vids = sorted(videos[cid])
        rng.shuffle(vids)
        val_videos.update(vids[:round(LOSO_VAL_FRACTION * len(vids))])
    parts = {"train": [], "val": [], "test": []}
    for f, (signer, _, video) in sorted(kept.items()):
        part = "test" if signer == test_signer else "val" if video in val_videos else "train"
        parts[part].append(f)
    return parts


def verify(name, parts, kept, test_signer=None):
    sets = {k: set(v) for k, v in parts.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), name
    assert set().union(*sets.values()) == set(kept), f"{name}: partitions do not cover the kept frames"
    if test_signer is not None:
        assert all(kept[f][0] == test_signer for f in sets["test"]), name
        assert all(kept[f][0] != test_signer for f in sets["train"] | sets["val"]), name
        video_part = {}
        for part, files in sets.items():
            for f in files:
                assert video_part.setdefault(kept[f][2], part) == part, f"{name}: video {kept[f][2]} split"


def main():
    if subprocess.run([sys.executable, str(ROOT / "scripts" / "check_annotations.py")],
                      capture_output=True).returncode != 0:
        sys.exit("check_annotations.py does not pass; finish the annotation first.")
    classes, _ = load_configs()
    names = {int(k): v for k, v in classes["names"].items()}
    slug_to_id = {v.lower().replace(" ", ""): k for k, v in names.items()}
    kept = kept_frames(slug_to_id)

    splits = {"p1": split_p1(kept)}
    verify("p1", splits["p1"], kept)
    for s in TEST_SIGNERS:
        splits[f"loso_s{s}"] = split_loso(kept, s)
        verify(f"loso_s{s}", splits[f"loso_s{s}"], kept, s)

    SPLITS.mkdir(exist_ok=True)
    rows = []
    for name, parts in splits.items():
        for part, files in parts.items():
            (SPLITS / f"{name}_{part}.txt").write_text("".join(f"{f}\n" for f in sorted(files)), encoding="utf-8")
            n = Counter(kept[f][1] for f in files)
            n_videos = len({kept[f][2] for f in files})
            rows.append([name, part, len(files), n_videos] + [n[c] for c in sorted(names)])
    with open(COUNTS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "partition", "frames", "videos"] + [names[c] for c in sorted(names)])
        w.writerows(rows)

    print(f"kept frames: {len(kept)}  seed: {SEED}")
    for r in rows:
        print(f"{r[0]:<9}{r[1]:<6} frames {r[2]:>5}  videos {r[3]:>4}  per class: {r[4:]}")
    print(f"written: {SPLITS.relative_to(ROOT)}/ (15 files), {COUNTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
