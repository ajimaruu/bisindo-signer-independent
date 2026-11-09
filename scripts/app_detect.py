"""Desktop application (manuscript section 2.6): reads a webcam, runs a trained detector and draws the boxes
with the predicted gloss on the displayed frame.

Two speeds are shown and kept apart:
  - Display FPS: the whole application loop (frame acquisition + prediction + drawing + window update),
    averaged over the last second. Bounded by the camera's own frame rate (often 30 FPS).
  - Model latency: the prediction stage alone (preprocess / inference / postprocess, CUDA-synchronised),
    the same boundary as equation (18). The FPS reported in the paper comes from scripts/measure_fps.py,
    not from this window.

Frames are shown as the camera delivers them, without mirroring: the training frames were never flipped
(fliplr=0), so a mirrored input would swap the dominant and non-dominant hand.

Keys: 1 / 2 / 3 = YOLO11m / RT-DETR-L / Faster R-CNN,  + / - = confidence threshold,  s = save screenshot,
      q or Esc = quit.

Default confidence threshold per model: `conf` in configs/app.yaml (the conf_thr of the run whose checkpoint is
used, chosen on the validation split); switching the model switches to its default. --conf overrides it for all
models.

Usage:
  python scripts/app_detect.py                          # webcam 0, YOLO11m, weights from configs/app.yaml
  python scripts/app_detect.py --model rtdetr-l --camera 1
  python scripts/app_detect.py --source video.mp4 --headless --save runs/app/out.mp4 --max-frames 150
  python scripts/app_detect.py --live-test              # per-sign test mode (Table 8), see scripts/live_test.py
"""
import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from detectors import LABELS, MODELS, ROOT, class_names, default_conf, default_weights, load_detector

KEYS = {ord("1"): "yolo11m", ord("2"): "rtdetr-l", ord("3"): "frcnn-r50fpn"}
GROUP_COLOUR = {"two": (0, 140, 255), "one": (60, 200, 60)}   # BGR: C_two orange, C_one green
SHOTS = ROOT / "runs" / "app"


def draw(frame, dets, names, groups, conf, model, ms, display_fps, max_show):
    for x1, y1, x2, y2, score, cid in sorted(dets, key=lambda d: -d[4])[:max_show]:
        colour = GROUP_COLOUR[groups[cid]]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, colour, 3)
        text = f"{names[cid]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        y = max(p1[1], th + 12)
        cv2.rectangle(frame, (p1[0], y - th - 12), (p1[0] + tw + 10, y), colour, -1)
        cv2.putText(frame, text, (p1[0] + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    model_ms = sum(ms.values())
    lines = [f"{LABELS[model]}   conf >= {conf:.2f}",
             f"Display FPS: {display_fps:5.1f}  (camera + model + drawing)",
             f"Model latency: {model_ms:5.1f} ms = pre {ms['preprocess']:.1f} + inf {ms['inference']:.1f} "
             f"+ post {ms['postprocess']:.1f}  ({1000 / max(model_ms, 1e-6):.0f} FPS)",
             "1/2/3 model   +/- conf   s screenshot   q quit"]
    pad = 8
    h = 26 * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (700, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class LatestFrame:
    """Reads the camera in a background thread and keeps only the newest frame, so the prediction never
    waits for the next camera frame (camera wait and model time would otherwise add up)."""

    def __init__(self, cap):
        self.cap, self.frame, self.ok, self.seq = cap, None, True, 0
        self.lock, self.new = threading.Lock(), threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            with self.new:
                self.ok, self.frame, self.seq = ok, frame, self.seq + 1
                self.new.notify_all()

    def read(self, last_seq):
        with self.new:   # wait for a frame newer than the one already processed
            self.new.wait_for(lambda: self.seq > last_seq or not self.ok, timeout=2.0)
            return self.ok and self.frame is not None, self.frame, self.seq

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.ok = False
        self.cap.release()


def open_source(args):
    if args.source:
        return cv2.VideoCapture(args.source)
    # On this laptop DirectShow ignores the MJPG request and delivers 1280x720 YUY2 at 10 FPS, while
    # Media Foundation (MSMF) delivers 30 FPS; try MSMF first, DirectShow as fallback.
    for backend in (getattr(cv2, "CAP_MSMF", None), getattr(cv2, "CAP_DSHOW", None), cv2.CAP_ANY):
        if backend is None:
            continue
        cap = cv2.VideoCapture(args.camera, backend)
        if cap.isOpened():
            break
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return LatestFrame(cap)


def main():
    if "--live-test" in sys.argv:   # separate mode; the normal application below is unchanged
        import live_test
        return live_test.main([a for a in sys.argv[1:] if a != "--live-test"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="yolo11m")
    ap.add_argument("--weights", type=Path, help="override the weights of --model (default: configs/app.yaml)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="video file instead of the webcam (for testing)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=None,
                    help="display threshold for every model (default: per model from configs/app.yaml, else 0.5)")
    ap.add_argument("--max-show", type=int, default=1, help="boxes drawn per frame (the dataset has one per frame)")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--save", type=Path, help="write the annotated frames to this video")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    weights = default_weights()
    if args.weights:
        weights[args.model] = args.weights
    model_conf = {m: args.conf if args.conf is not None else default_conf().get(m, 0.5) for m in MODELS}
    names = class_names()
    import yaml
    groups = {int(k): v for k, v in yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))["groups"].items()}

    loaded = {}

    def get(model):
        if model not in loaded:
            print(f"loading {LABELS[model]}: {weights[model]}")
            loaded[model] = load_detector(model, weights[model], conf=model_conf[model], max_det=10)
        loaded[model].set_conf(conf)
        return loaded[model]

    model = args.model
    conf = model_conf[model]
    det = get(model)
    cap = open_source(args)
    if not cap.isOpened():
        raise SystemExit("cannot open the camera/source")
    writer, stamps, n, seq = None, deque(maxlen=60), 0, 0
    win = "BISINDO detector"
    while True:
        if isinstance(cap, LatestFrame):
            ok, frame, seq = cap.read(seq)
        else:
            ok, frame = cap.read()
        if not ok:
            break
        dets, ms = det(frame)
        stamps.append(time.perf_counter())
        display_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        out = draw(frame, dets, names, groups, conf, model, ms, display_fps, args.max_show)
        if args.save:
            if writer is None:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), 30, out.shape[1::-1])
            writer.write(out)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        if args.headless:
            continue
        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in KEYS and KEYS[key] in weights:
            model_conf[model] = conf   # keep a +/- change for this model
            model = KEYS[key]
            conf = model_conf[model]
            det = get(model)
            stamps.clear()
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
            det.set_conf(conf)
        elif key == ord("-"):
            conf = max(0.05, conf - 0.05)
            det.set_conf(conf)
        elif key == ord("s"):
            SHOTS.mkdir(parents=True, exist_ok=True)
            path = SHOTS / f"shot_{time.strftime('%Y%m%d_%H%M%S')}_{model}.jpg"
            cv2.imwrite(str(path), out)
            print(f"saved {path}")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"frames: {n}, last display FPS {display_fps:.1f}, last model latency {sum(ms.values()):.1f} ms")


if __name__ == "__main__":
    main()
