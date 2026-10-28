"""Inference speed for the paper (CLAUDE.md rule 7, manuscript eq. (18) and section 2.6).

Boundary: the model prediction stage only. Frames are read from disk into memory *before* timing, nothing is
drawn, batch 1, warm-up first, then the mean over many frames. Settings are those of the mAP evaluation
(FP16, conf 0.001, NMS IoU 0.7, at most 100 detections), so speed and accuracy describe the same model call.

The three stages are timed separately (CUDA-synchronised, see scripts/detectors.py) and all are written to
results/fps.csv:
  t_pred  = preprocess + inference + postprocess   -> fps     = 1000 / mean(t_pred)  (eq. 18)
  t_infer = network forward only                    -> fps_net = 1000 / mean(t_infer) (reference)
Final-measurement rules (CLAUDE.md, "Pengukuran FPS final"): final Kaggle checkpoints; laptop on the charger,
high-performance power mode, cooling pad, Chrome and Discord closed; before every measurement the script waits
until the GPU is back near its idle temperature (idle at start + --cool-margin); each model is measured
--repeats times (default 3) and reported as mean +- SD with the GPU temperatures; FP16 for all three models.
GPU temperature and SM clock are logged before and after each repeat, because the laptop GPU throttles.

Usage: python scripts/measure_fps.py [--models yolo11m rtdetr-l frcnn-r50fpn] [--repeats 3] [--note "..."]
"""
import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from coco_eval import CONF, MAX_DET, NMS_IOU
from detectors import LABELS, MODELS, ROOT, default_weights, load_detector

OUT = ROOT / "results" / "fps.csv"
COLS = ["timestamp", "model", "repeat", "weights", "gpu", "precision", "imgsz", "batch", "n_frames", "warmup", "iters",
        "pre_ms", "infer_ms", "post_ms", "pred_ms", "pred_ms_median", "pred_ms_p90", "fps", "fps_net",
        "params_M", "idle_temp_c", "wait_cool_s", "temp_start_c", "temp_end_c", "sm_clock_start_mhz",
        "sm_clock_end_mhz", "note"]


def gpu_state():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        t, c = r.stdout.strip().splitlines()[0].split(",")
        return float(t), float(c)
    except (ValueError, IndexError):
        return float("nan"), float("nan")


def wait_cool(target, max_wait):
    """Idle until the GPU temperature is at most `target` (or max_wait seconds have passed)."""
    t0 = time.time()
    while gpu_state()[0] > target and time.time() - t0 < max_wait:
        time.sleep(5)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--split", default="p1_test", help="frames of this split are used as input")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cool-margin", type=float, default=5.0, help="wait until GPU temp <= idle temp + margin")
    ap.add_argument("--cool-max-wait", type=int, default=900, help="seconds")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", type=Path, default=OUT, help="CSV to append to (tests: not results/fps.csv)")
    args = ap.parse_args()

    files = (ROOT / "splits" / f"{args.split}.txt").read_text(encoding="utf-8").split()[:args.frames]
    frames = [cv2.imread(str(ROOT / "data" / "frames" / f)) for f in files]   # read before any timing
    weights = default_weights()
    gpu = torch.cuda.get_device_name(0)
    idle = gpu_state()[0]
    print(f"GPU {gpu}, idle temperature at start {idle:.0f} C; cooling target {idle + args.cool_margin:.0f} C")
    rows = []
    for model in args.models:
        det = load_detector(model, weights[model], conf=CONF, iou=NMS_IOU, max_det=MAX_DET, half=True)
        for rep in range(1, args.repeats + 1):
            waited = wait_cool(idle + args.cool_margin, args.cool_max_wait)
            for i in range(args.warmup):
                det(frames[i % len(frames)])
            temp0, clk0 = gpu_state()
            stages = []
            for i in range(args.iters):
                _, ms = det(frames[i % len(frames)])
                stages.append((ms["preprocess"], ms["inference"], ms["postprocess"]))
            temp1, clk1 = gpu_state()
            s = np.array(stages)
            pred = s.sum(1)
            row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": model, "repeat": rep,
                   "weights": str(Path(weights[model]).relative_to(ROOT)), "gpu": gpu, "precision": "fp16",
                   "imgsz": 640, "batch": 1, "n_frames": len(frames), "warmup": args.warmup, "iters": args.iters,
                   "pre_ms": s[:, 0].mean(), "infer_ms": s[:, 1].mean(), "post_ms": s[:, 2].mean(),
                   "pred_ms": pred.mean(), "pred_ms_median": np.median(pred),
                   "pred_ms_p90": np.percentile(pred, 90), "fps": 1000 / pred.mean(),
                   "fps_net": 1000 / s[:, 1].mean(), "params_M": det.n_params / 1e6, "idle_temp_c": idle,
                   "wait_cool_s": waited, "temp_start_c": temp0, "temp_end_c": temp1,
                   "sm_clock_start_mhz": clk0, "sm_clock_end_mhz": clk1, "note": args.note}
            rows.append(row)
            print(f"{LABELS[model]:<22} #{rep} pre {row['pre_ms']:5.1f}  inf {row['infer_ms']:5.1f}  "
                  f"post {row['post_ms']:5.1f} = {row['pred_ms']:5.1f} ms -> {row['fps']:5.1f} FPS "
                  f"(net only {row['fps_net']:5.1f})  cooled {waited:.0f} s  GPU {temp0:.0f}->{temp1:.0f} C, "
                  f"{clk0:.0f}->{clk1:.0f} MHz")
        del det
        torch.cuda.empty_cache()

    print("\nmean +- SD over repeats (FP16, batch 1):")
    for model in args.models:
        r = [x for x in rows if x["model"] == model]
        fps, pred = np.array([x["fps"] for x in r]), np.array([x["pred_ms"] for x in r])
        sd = (lambda a: a.std(ddof=1) if len(a) > 1 else 0.0)
        print(f"  {LABELS[model]:<22} {fps.mean():5.1f} +- {sd(fps):4.1f} FPS   {pred.mean():5.1f} +- {sd(pred):4.1f} ms   "
              f"GPU {min(x['temp_start_c'] for x in r):.0f}-{max(x['temp_end_c'] for x in r):.0f} C   "
              f"params {r[0]['params_M']:.1f} M")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    print(f"appended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
