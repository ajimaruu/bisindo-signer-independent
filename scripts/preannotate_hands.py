"""Hand-detector pre-annotation: proposes one YOLO box per frame for manual correction in LabelImg.

Runs in the separate env .venv-mediapipe (MediaPipe pins its own numpy/protobuf).
Detector: MediaPipe HandLandmarker (Tasks API), model data/models/hand_landmarker.task.

Box rule (CLAUDE.md, annotation rules):
  - each detected hand -> tight box around its 21 landmarks, expanded by MARGIN of its longer side
  - C_two gloss -> union of the (up to two) hand boxes; one detected hand is flagged
  - C_one gloss -> only the highest raised hand (smallest landmark-centroid y)
  - no hand detected -> no label; frame listed as an exclusion *candidate* (user decides)
Class id comes from the gloss in the file name (configs/classes.yaml order = classes.txt).

Usage:
  python scripts/preannotate_hands.py --pilot      # 20 frames -> results/preannot_pilot/
  python scripts/preannotate_hands.py --all        # all frames -> data/labels_auto/ (+ copy to labels_yolo)
"""
import argparse
import csv
import random
import re
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
LABELS = ROOT / "data" / "labels_yolo"
AUTO = ROOT / "data" / "labels_auto"
MODEL = ROOT / "data" / "models" / "hand_landmarker.task"
PILOT = ROOT / "results" / "preannot_pilot"
CANDIDATES = ROOT / "data" / "exclusion_candidates.csv"
REPORT = ROOT / "data" / "preannot_report.csv"
FRAME_RE = re.compile(r"^s(\d+)_([a-z]+)_v(\d{2})_f(\d{4})$")

MARGIN = 0.08          # fraction of each hand box's longer side added on every side
MIN_DET_CONF = 0.5     # MediaPipe defaults
MIN_PRESENCE_CONF = 0.5


def load_classes():
    c = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
    slug_to_id = {name.lower().replace(" ", ""): int(k) for k, name in c["names"].items()}
    groups = {int(k): v for k, v in c["groups"].items()}
    return c, slug_to_id, groups


def make_detector():
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=MIN_DET_CONF, min_hand_presence_confidence=MIN_PRESENCE_CONF)
    return vision.HandLandmarker.create_from_options(opts)


def detect_hands(detector, img_bgr):
    """Return list of dicts: box (x1,y1,x2,y2 px, with margin), centroid_y, score, handedness, points."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    hands = []
    for lms, hd in zip(res.hand_landmarks, res.handedness):
        pts = np.array([(lm.x * w, lm.y * h) for lm in lms])
        x1, y1 = pts.min(0)
        x2, y2 = pts.max(0)
        m = MARGIN * max(x2 - x1, y2 - y1)
        box = (max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m))
        hands.append({"box": box, "centroid_y": float(pts[:, 1].mean()), "score": hd[0].score,
                      "handedness": hd[0].category_name, "points": pts})
    return hands


def propose(hands, group):
    """Apply the box rule. Returns (box or None, chosen hand indices, flag)."""
    if not hands:
        return None, [], "no_hand"
    if group == "two":
        boxes = np.array([h["box"] for h in hands])
        box = (*boxes[:, :2].min(0), *boxes[:, 2:].max(0))
        return box, list(range(len(hands))), "two_class_one_hand" if len(hands) == 1 else ""
    i = int(np.argmin([h["centroid_y"] for h in hands]))
    return hands[i]["box"], [i], ""


def to_yolo(cid, box, w, h):
    x1, y1, x2, y2 = box
    return f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def draw(img, hands, chosen, box, title, flag):
    out = img.copy()
    for i, hd in enumerate(hands):
        x1, y1, x2, y2 = map(int, hd["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255) if i in chosen else (160, 160, 160), 2)
        for x, y in hd["points"].astype(int):
            cv2.circle(out, (x, y), 2, (0, 200, 255) if i in chosen else (160, 160, 160), -1)
        cv2.putText(out, f"{hd['handedness'][0]} {hd['score']:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255) if i in chosen else (160, 160, 160), 2)
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 4)
    bar = np.full((44, out.shape[1], 3), 255, np.uint8)
    color = (0, 0, 220) if flag else (0, 0, 0)
    cv2.putText(bar, f"{title}  {flag or ''}", (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return np.vstack([bar, out])


def pilot_selection(slug_to_id, groups, seed=0):
    """20 frames: each C_two gloss x2, each C_one gloss x1, one extra Teman (occlusion class);
    random signer/video, cycling through all five frame positions."""
    rng = random.Random(seed)
    by_video = {}
    for p in sorted(FRAMES.glob("*.jpg")):
        m = FRAME_RE.match(p.stem)
        by_video.setdefault((m.group(2), m.group(1), m.group(3)), []).append(p)
    two = [s for s, i in sorted(slug_to_id.items(), key=lambda x: x[1]) if groups[i] == "two"]
    one = [s for s, i in sorted(slug_to_id.items(), key=lambda x: x[1]) if groups[i] == "one"]
    plan = [g for g in two for _ in range(2)] + one + ["teman"]
    picks = []
    for k, slug in enumerate(plan):
        vids = sorted(v for v in by_video if v[0] == slug)
        frames = sorted(by_video[rng.choice(vids)])
        picks.append(frames[k % len(frames)])  # cycle through positions 20%..80%
    return picks


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--all", action="store_true")
    args = ap.parse_args()

    classes, slug_to_id, groups = load_classes()
    detector = make_detector()

    if args.pilot:
        frames = pilot_selection(slug_to_id, groups)
        out_labels = PILOT / "labels"
        (PILOT / "overlays").mkdir(parents=True, exist_ok=True)
    else:
        frames = sorted(FRAMES.glob("*.jpg"))
        out_labels = AUTO
    out_labels.mkdir(parents=True, exist_ok=True)

    excluded = set()
    excl_path = ROOT / "data" / "excluded_frames.csv"
    if excl_path.exists():
        excluded = {Path(r["file"]).stem for r in csv.DictReader(open(excl_path, encoding="utf-8-sig"))}

    report, tiles, copied, kept_existing = [], [], 0, 0
    for n, p in enumerate(frames, 1):
        m = FRAME_RE.match(p.stem)
        cid = slug_to_id[m.group(2)]
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        hands = detect_hands(detector, img)
        box, chosen, flag = propose(hands, groups[cid])
        row = {"file": p.name, "class_id": cid, "group": groups[cid], "n_hands": len(hands),
               "scores": " ".join(f"{x['score']:.2f}" for x in hands), "flag": flag}
        if box is not None:
            (out_labels / f"{p.stem}.txt").write_text(to_yolo(cid, box, w, h) + "\n", encoding="utf-8")
            if args.all and p.stem not in excluded:
                dst = LABELS / f"{p.stem}.txt"
                if dst.exists() and dst.read_text(encoding="utf-8").strip():
                    kept_existing += 1   # never overwrite a label that already has content
                else:
                    shutil.copyfile(out_labels / f"{p.stem}.txt", dst)
                    copied += 1
        report.append(row)
        if args.pilot:
            title = f"{p.stem}  [{classes['names'][cid]}, {groups[cid]}]  hands={len(hands)}"
            ov = draw(img, hands, chosen, box, title, flag)
            cv2.imwrite(str(PILOT / "overlays" / f"{p.stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
            tiles.append(cv2.resize(ov, (640, int(round(640 * ov.shape[0] / ov.shape[1]))),
                                    interpolation=cv2.INTER_AREA))
        if n % 250 == 0 or args.pilot:
            print(f"[{n}/{len(frames)}] {p.name} hands={len(hands)} {flag}")

    report_path = PILOT / "report.csv" if args.pilot else REPORT
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(report[0]))
        wr.writeheader()
        wr.writerows(report)
    cand_path = PILOT / "exclusion_candidates.csv" if args.pilot else CANDIDATES
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "hint"])
        wr.writerows([r["file"], r["flag"]] for r in report if r["flag"] == "no_hand")

    if args.pilot:
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(PILOT / "pilot_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    flags = {}
    for r in report:
        flags[r["flag"] or "ok"] = flags.get(r["flag"] or "ok", 0) + 1
    print(f"done: {len(report)} frames  {flags}")
    if args.all:
        print(f"copied to labels_yolo: {copied}  existing labels kept: {kept_existing}  "
              f"excluded skipped: {sum(1 for r in report if Path(r['file']).stem in excluded)}")


if __name__ == "__main__":
    main()
