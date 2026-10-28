"""Common evaluator for all three models (CLAUDE.md rule 6): predictions exported to COCO format and scored
with pycocotools (mAP@0.5, mAP@[.50:.95], 101-point interpolation, maxDets 100, area 'all').

Ground truth: data/coco/<split>.json (scripts/prepare_data.py). category_id = class id + 1.
"""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "coco"
FRAMES = ROOT / "data" / "frames"

# Inference settings for the exported predictions (same for every model).
CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 100


def load_gt(split):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(COCO_DIR / f"{split}.json"))


def evaluate(split, dets):
    """dets: list of {image_id, category_id, bbox [x,y,w,h] px, score}. Returns overall and per-class AP."""
    gt = load_gt(split)
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    if not dets:
        return {"map50": 0.0, "map50_95": 0.0, "per_class": {n: (0.0, 0.0) for n in names.values()}}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval["precision"]          # [T=10 IoU, R=101 recall, K classes, A=4 areas, M=3 maxDets]
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, 2]
        ap5095 = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
        ap50 = float(p[0][p[0] > -1].mean()) if (p[0] > -1).any() else float("nan")
        per_class[names[cid]] = (ap50, ap5095)
    return {"map50": float(ev.stats[1]), "map50_95": float(ev.stats[0]), "per_class": per_class}


def predict_ultralytics(weights, split, batch=8):
    """Run an Ultralytics model (YOLO or RT-DETR) on a split and return COCO-format detections."""
    from ultralytics import RTDETR, YOLO

    gt = load_gt(split)
    ids = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    model = (RTDETR if "rtdetr" in str(weights).lower() else YOLO)(str(weights))
    files = [str(FRAMES / f) for f in sorted(ids)]
    dets = []
    for i in range(0, len(files), batch):
        for r in model.predict(files[i:i + batch], imgsz=640, conf=CONF, iou=NMS_IOU, max_det=MAX_DET,
                               half=True, verbose=False):
            b = r.boxes
            for (x1, y1, x2, y2), c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist()):
                dets.append({"image_id": ids[Path(r.path).name], "category_id": int(c) + 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return dets


def save_dets(dets, path):
    Path(path).write_text(json.dumps(dets), encoding="utf-8")


def format_result(res):
    worst = sorted(res["per_class"].items(), key=lambda kv: kv[1][1])[:3]
    return (f"mAP50 {res['map50']:.4f}  mAP50-95 {res['map50_95']:.4f}  lowest AP50-95: "
            + ", ".join(f"{n} {v[1]:.3f}" for n, v in worst))


def nan_mean(xs):
    return float(np.nanmean(xs)) if len(xs) else float("nan")
