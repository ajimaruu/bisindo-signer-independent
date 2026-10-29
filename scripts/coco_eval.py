"""Common evaluator for all three models (CLAUDE.md rule 6): predictions exported to COCO format and scored
with pycocotools (mAP@0.5, mAP@[.50:.95], 101-point interpolation, maxDets 100, area 'all').

Ground truth: data/coco/<split>.json (scripts/prepare_data.py). category_id = class id + 1.

Labels per box in the exported detections (decided 29 Oct 2026, config_log section 25): every model may give one box
several class labels, as the standard COCO-style evaluation does.
  - YOLO11m: predict() preprocessing, but NMS with multi_label=True (every class above CONF per candidate box, then
    class-wise NMS), as in Ultralytics' validator; predict()'s default keeps only the top class per box.
  - RT-DETR-L: no NMS; the 300 highest query-class scores over all decoder queries and classes (unchanged).
  - Faster R-CNN: one class-specific box per class above CONF for every proposal, then class-wise NMS (torchvision).
The desktop application and the FPS measurement keep the plain predict() (one label per box).
"""
import contextlib
import functools
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


def match_iou50(split, dets):
    """Per class: detection scores, true-positive flags and number of ground-truth boxes, with pycocotools'
    own matching (greedy by score, IoU >= 0.5, at most MAX_DET detections per image, area 'all')."""
    gt = load_gt(split)
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets) if dets else COCO(), "bbox")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [MAX_DET]
        ev.params.areaRng, ev.params.areaRngLbl = [ev.params.areaRng[0]], ["all"]
        ev.evaluate()
    for cid, name in names.items():
        scores, tp, n_gt = [], [], 0
        for e in ev.evalImgs:
            if e is None or e["category_id"] != cid:
                continue
            keep = ~np.asarray(e["dtIgnore"][0], bool)
            scores += list(np.asarray(e["dtScores"])[keep])
            tp += list(np.asarray(e["dtMatches"][0])[keep] > 0)
            n_gt += int((~np.asarray(e["gtIgnore"], bool)).sum())
        out[name] = (np.array(scores, float), np.array(tp, bool), n_gt)
    return out


def prf(matched, thresholds):
    """Macro-averaged precision, recall and F1 over classes at IoU 0.5 for each confidence threshold
    (detections with score >= threshold are kept). Per class: P = TP / (TP + FP), 0 when nothing is kept;
    R = TP / n_gt; F1 = 2PR / (P + R), 0 when P + R = 0. Macro-F1 = mean of the per-class F1."""
    thresholds = np.atleast_1d(np.asarray(thresholds, float))
    P, R, F = [], [], []
    for scores, tp, n_gt in matched.values():
        if n_gt == 0:
            raise ValueError("class without ground truth in this split")
        order = np.argsort(scores)                      # ascending
        s, cum_tp = scores[order], np.cumsum(tp[order][::-1])[::-1]   # cum_tp[i] = TP among s >= s[i]
        idx = np.searchsorted(s, thresholds, side="left")            # first kept detection
        kept = len(s) - idx
        tps = np.where(idx < len(s), cum_tp[np.minimum(idx, len(s) - 1)] if len(s) else 0, 0)
        p = np.divide(tps, kept, out=np.zeros(len(thresholds)), where=kept > 0)
        r = tps / n_gt
        f = np.divide(2 * p * r, p + r, out=np.zeros(len(thresholds)), where=(p + r) > 0)
        P.append(p), R.append(r), F.append(f)
    return np.mean(P, 0), np.mean(R, 0), np.mean(F, 0)


def select_threshold(matched):
    """Confidence threshold that maximises macro-F1 (IoU 0.5); every detection score is a candidate.
    Ties: the highest threshold. Returns (threshold, precision, recall, f1) on these detections."""
    cands = np.unique(np.concatenate([m[0] for m in matched.values()] + [np.array([CONF])]))
    P, R, F = prf(matched, cands)
    i = len(F) - 1 - int(np.argmax(F[::-1]))
    return float(cands[i]), float(P[i]), float(R[i]), float(F[i])


@contextlib.contextmanager
def multi_label_nms():
    """Inside this block Ultralytics' NMS keeps every class above conf for each box (multi_label=True)."""
    from ultralytics.utils import nms
    orig = nms.non_max_suppression
    nms.non_max_suppression = functools.partial(orig, multi_label=True)
    try:
        yield
    finally:
        nms.non_max_suppression = orig


def predict_ultralytics(weights, split, batch=8, multi_label=True):
    """Run an Ultralytics model (YOLO or RT-DETR) on a split and return COCO-format detections.
    multi_label: YOLO NMS keeps every class above CONF per box (the paper export); False = predict() default,
    used only to compare with predictions exported before 29 Oct 2026. No effect on RT-DETR (no NMS)."""
    from ultralytics import RTDETR, YOLO

    gt = load_gt(split)
    ids = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    model = (RTDETR if "rtdetr" in str(weights).lower() else YOLO)(str(weights))
    files = [str(FRAMES / f) for f in sorted(ids)]
    dets = []
    for i in range(0, len(files), batch):
        with multi_label_nms() if multi_label else contextlib.nullcontext():
            results = model.predict(files[i:i + batch], imgsz=640, conf=CONF, iou=NMS_IOU, max_det=MAX_DET,
                                    half=True, verbose=False)
        for r in results:
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
