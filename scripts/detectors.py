"""One interface for the three trained detectors, used by the desktop app (scripts/app_detect.py) and the
paper FPS benchmark (scripts/measure_fps.py).

    det = load_detector("yolo11m" | "rtdetr-l" | "frcnn-r50fpn", weights, conf=..., iou=..., max_det=...)
    dets, ms = det(frame_bgr)   # dets: [(x1, y1, x2, y2, score, class_id)] in frame pixels
                                # ms:   {"preprocess", "inference", "postprocess"} in milliseconds

Every stage is timed with Ultralytics' ops.Profile, which synchronises CUDA before reading the clock.
  - YOLO11m / RT-DETR-L: Ultralytics' own predictor and its own stage timings.
      preprocess  = LetterBox to 640 (minimal padding, stride 32), BGR->RGB, tensor, FP16, /255
      inference   = network forward
      postprocess = NMS (YOLO) or score/top-k selection (RT-DETR, NMS-free) + rescale to the frame
  - Faster R-CNN: the same LetterBox and the same timer.
      preprocess  = LetterBox, BGR->RGB, tensor, /255
      inference   = torchvision forward under FP16 autocast; this already contains torchvision's
                    normalisation, the RPN and the per-class NMS of the ROI heads
      postprocess = rescale to the frame
"""
from pathlib import Path

import run_guards  # noqa: I001  (before torch: disables torch.compile)

import numpy as np
import torch
import yaml
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import Profile, scale_boxes

ROOT = Path(__file__).resolve().parents[1]
run_guards.apply()   # same RoIAlign kernel and no torch.compile, as in training
MODELS = ("yolo11m", "rtdetr-l", "frcnn-r50fpn")
LABELS = {"yolo11m": "YOLO11m", "rtdetr-l": "RT-DETR-L", "frcnn-r50fpn": "Faster R-CNN R50-FPN"}


def class_names():
    c = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
    return {int(k): v for k, v in c["names"].items()}


class UltralyticsDetector:
    def __init__(self, kind, weights, conf, iou, max_det, half=True, imgsz=640):
        from ultralytics import RTDETR, YOLO
        self.model = (RTDETR if kind == "rtdetr-l" else YOLO)(str(weights))
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, half=half, verbose=False)
        self.n_params = sum(p.numel() for p in self.model.model.parameters())

    def set_conf(self, conf):
        self.kw["conf"] = conf

    def __call__(self, frame):
        r = self.model.predict(frame, **self.kw)[0]
        b = r.boxes
        dets = [(*xyxy, s, int(c)) for xyxy, s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]
        return dets, dict(r.speed)


class FRCNNDetector:
    def __init__(self, weights, conf, iou, max_det, half=True, imgsz=640):
        from train_frcnn import build_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(len(class_names()))
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model"])
        self.model.to(self.device).eval()
        self.set_conf(conf)
        self.model.roi_heads.nms_thresh = iou
        self.model.roi_heads.detections_per_img = max_det
        self.half = half and self.device.type == "cuda"
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)   # as Ultralytics predict
        self.prof = [Profile(device=self.device) for _ in range(3)]
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def set_conf(self, conf):
        self.model.roi_heads.score_thresh = conf

    @torch.no_grad()
    def __call__(self, frame):
        with self.prof[0]:
            img = self.letterbox(image=frame)
            t = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))).to(self.device)
            t = t.float() / 255
        with self.prof[1]:
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
                out = self.model([t])[0]
        with self.prof[2]:
            boxes = scale_boxes(t.shape[1:], out["boxes"].float(), frame.shape[:2])
            dets = [(*xyxy, s, int(c) - 1) for xyxy, s, c in
                    zip(boxes.tolist(), out["scores"].float().tolist(), out["labels"].tolist())]
        ms = {k: p.dt * 1e3 for k, p in zip(("preprocess", "inference", "postprocess"), self.prof)}
        return dets, ms


def load_detector(kind, weights, conf, iou=0.7, max_det=100, half=True):
    if kind == "frcnn-r50fpn":
        return FRCNNDetector(weights, conf, iou, max_det, half)
    return UltralyticsDetector(kind, weights, conf, iou, max_det, half)


def default_weights():
    """Weights per model from configs/app.yaml (paths relative to the repo root)."""
    cfg = yaml.safe_load(open(ROOT / "configs" / "app.yaml", encoding="utf-8"))
    return {k: ROOT / v for k, v in cfg["weights"].items() if v}
