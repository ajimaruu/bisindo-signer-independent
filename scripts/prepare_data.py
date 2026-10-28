"""Convert the reviewed YOLO labels to COCO per split, and build the Ultralytics view of the data.

  - data/coco/<split>.json for every splits/<split>.txt (15 files). Boxes in pixels of the original frame,
    category_id = class id + 1 (0 is background in torchvision). image_id is fixed per frame across splits
    (1-based index in the sorted list of all kept frames).
  - data/yolo/images and data/yolo/labels: directory junctions (symlinks on Linux/Kaggle) to data/frames and data/labels_yolo
    (Ultralytics finds a label by replacing /images/ with /labels/ in the image path; nothing is copied).
  - data/yolo/<split>.txt (image lists) and data/yolo/<p1|loso_sN>.yaml (Ultralytics data files).
  - Check: every COCO box is converted back and compared with its YOLO line, and the COCO boxes of 10 random
    frames are drawn to results/coco_check/ for visual inspection.

Usage: python scripts/prepare_data.py
"""
import json
import os
import random
import subprocess
from pathlib import Path

import cv2
import yaml

from annot_common import FRAMES, LABELS, ROOT
from check_annotations import load_configs, read_label

SPLITS = ROOT / "splits"
COCO = ROOT / "data" / "coco"
YOLO = ROOT / "data" / "yolo"
CHECK = ROOT / "results" / "coco_check"
GROUPS = ["p1"] + [f"loso_s{k}" for k in range(1, 5)]


def junction(link, target):
    if link.exists():
        assert link.resolve() == target.resolve(), f"{link} points elsewhere"
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:   # Kaggle / Linux
        os.symlink(target.resolve(), link, target_is_directory=True)


def main():
    classes, _ = load_configs()
    names = {int(k): v for k, v in classes["names"].items()}
    groups = {int(k): v for k, v in classes["groups"].items()}
    split_files = {p.stem: p.read_text(encoding="utf-8").split() for p in sorted(SPLITS.glob("*.txt"))}
    all_frames = sorted(set().union(*split_files.values()))
    image_id = {f: i for i, f in enumerate(all_frames, 1)}

    # one COCO image + annotation per kept frame
    images, anns = {}, {}
    for f in all_frames:
        h, w = cv2.imread(str(FRAMES / f)).shape[:2]
        boxes, errors = read_label(LABELS / f"{Path(f).stem}.txt")
        assert len(boxes) == 1 and not errors, f
        cid, xc, yc, bw, bh = boxes[0]
        x, y = (xc - bw / 2) * w, (yc - bh / 2) * h
        images[f] = {"id": image_id[f], "file_name": f, "width": w, "height": h}
        anns[f] = {"id": image_id[f], "image_id": image_id[f], "category_id": cid + 1,
                   "bbox": [round(x, 3), round(y, 3), round(bw * w, 3), round(bh * h, 3)],
                   "area": round(bw * w * bh * h, 3), "iscrowd": 0}
    categories = [{"id": k + 1, "name": names[k], "supercategory": groups[k]} for k in sorted(names)]

    COCO.mkdir(parents=True, exist_ok=True)
    for split, files in split_files.items():
        doc = {"info": {"description": f"WL-BISINDO 15 glosses, split {split}"},
               "images": [images[f] for f in sorted(files)], "annotations": [anns[f] for f in sorted(files)],
               "categories": categories}
        (COCO / f"{split}.json").write_text(json.dumps(doc), encoding="utf-8")

    # numeric round trip COCO -> YOLO for every frame
    worst = 0.0
    for f in all_frames:
        a, im = anns[f], images[f]
        x, y, bw, bh = a["bbox"]
        back = ((x + bw / 2) / im["width"], (y + bh / 2) / im["height"], bw / im["width"], bh / im["height"])
        cid, *orig = read_label(LABELS / f"{Path(f).stem}.txt")[0][0]
        assert a["category_id"] == cid + 1
        worst = max(worst, max(abs(p - q) * s for p, q, s in
                               zip(back, orig, (im["width"], im["height"], im["width"], im["height"]))))

    # Ultralytics view
    YOLO.mkdir(parents=True, exist_ok=True)
    junction(YOLO / "images", FRAMES)
    junction(YOLO / "labels", LABELS)
    for split, files in split_files.items():
        (YOLO / f"{split}.txt").write_text("".join(f"./images/{f}\n" for f in sorted(files)), encoding="utf-8")
    for g in GROUPS:
        data = {"path": str(YOLO), "train": f"{g}_train.txt", "val": f"{g}_val.txt", "test": f"{g}_test.txt",
                "names": names}
        (YOLO / f"{g}.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # visual check: draw the COCO boxes of 10 random frames
    CHECK.mkdir(parents=True, exist_ok=True)
    gt = json.loads((COCO / "p1_test.json").read_text(encoding="utf-8"))
    by_img = {a["image_id"]: a for a in gt["annotations"]}
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    tiles = []
    for im in random.Random(0).sample(gt["images"], 10):
        img = cv2.imread(str(FRAMES / im["file_name"]))
        a = by_img[im["id"]]
        x, y, bw, bh = map(round, a["bbox"])
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 220, 0), 3)
        cv2.putText(img, f"{im['file_name']}  {cats[a['category_id']]}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(CHECK / im["file_name"]), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    sheet = cv2.vconcat([cv2.hconcat(tiles[i:i + 2]) for i in range(0, 10, 2)])
    cv2.imwrite(str(CHECK / "coco_check_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"frames: {len(all_frames)}  COCO files: {len(split_files)}  "
          f"max round-trip error: {worst:.4f} px")
    for split, files in split_files.items():
        print(f"  {split:<16} images {len(files):>5}")
    print(f"Ultralytics data: {', '.join(f'data/yolo/{g}.yaml' for g in GROUPS)}")
    print(f"visual check: {CHECK.relative_to(ROOT)}/coco_check_sheet.jpg")


if __name__ == "__main__":
    main()
