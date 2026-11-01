"""Display FPS of the desktop application and the resting-hand check (manuscript section 3.6).

Runs the application loop of scripts/app_detect.py unchanged (MSMF webcam 1280x720 read in a background thread
that keeps the newest frame, the model's predict() with the application's settings, drawing the box and the
overlay, window update), with the checkpoints and default thresholds of configs/app.yaml.

Part A, display FPS (not the paper's model FPS, which is scripts/measure_fps.py):
  per model --repeats times (default 3): wait until the GPU is back near its idle temperature (as measure_fps.py),
  --warmup seconds not counted, then --seconds of the loop (default 60).
  display FPS = (frames shown - 1) / (time of last frame - time of first frame).
  The camera's own delivery rate in the same window is logged too, because display FPS cannot exceed it.
Part B, resting hands (--rest-seconds per model, default 20; 0 skips it):
  the person keeps both hands in the rest position; the model runs with a low threshold (0.05) so the top score of
  every frame is logged, and the share of frames whose top score reaches the application's threshold (= a label
  would be shown) is reported. The frame with the highest score is saved with the box drawn.

The window title tells the person in front of the camera what to do.
Outputs: results/app_fps.csv (one row per repeat), results/app_rest_check.csv (one row per model),
         runs/app/final/frames.csv (per frame), runs/app/final/rest_<model>.jpg
Usage:   python scripts/app_fps.py [--repeats 3] [--seconds 60] [--rest-seconds 20] [--note "..."]
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from app_detect import draw, open_source
from detectors import LABELS, MODELS, ROOT, class_names, default_conf, default_weights, load_detector
from measure_fps import gpu_state

OUT_DIR = ROOT / "runs" / "app" / "final"
WIN = "BISINDO detector"
REST_CONF = 0.05


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def loop(cap, det, model, conf, names, groups, seconds, title, rows, tag, keep_best=False):
    """The application loop for `seconds`; returns per-frame records (time, camera seq, model ms, top det)."""
    cv2.setWindowTitle(WIN, title)
    stamps, seq, t_end, best = [], 0, time.perf_counter() + seconds, None
    seq0 = None
    while time.perf_counter() < t_end:
        ok, frame, seq = cap.read(seq)
        if not ok:
            raise SystemExit("camera stopped delivering frames")
        if seq0 is None:
            seq0 = seq
        dets, ms = det(frame)
        top = max(dets, key=lambda d: d[4]) if dets else None
        shown = [d for d in dets if d[4] >= conf]
        fps_now = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        out = draw(frame, shown, names, groups, conf, model, ms, fps_now, 1)
        cv2.imshow(WIN, out)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise SystemExit("stopped by the user")
        t = time.perf_counter()
        stamps.append(t)
        rec = {"tag": tag, "model": model, "t": t, "cam_seq": seq, "pred_ms": sum(ms.values()),
               "top_score": top[4] if top else 0.0, "top_class": names[top[5]] if top else ""}
        rows.append(rec)
        if keep_best and top and (best is None or top[4] > best[0]):
            best = (top[4], out.copy())
    return stamps, seq - seq0, best


def cool_with_window(cap, target, max_wait, what):
    """As measure_fps.wait_cool, but keeps showing the plain camera image (no model) so the window stays alive."""
    t0, seq, last = time.time(), 0, 0.0
    while time.time() - t0 < max_wait:
        if time.time() - last > 5:
            temp, last = gpu_state()[0], time.time()
            if temp <= target:
                break
            cv2.setWindowTitle(WIN, f"{what}: istirahat, menunggu GPU dingin ({temp:.0f} -> {target:.0f} C)")
        ok, frame, seq = cap.read(seq)
        cv2.imshow(WIN, frame)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            raise SystemExit("stopped by the user")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--rest-seconds", type=float, default=20)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cool-margin", type=float, default=5.0)
    ap.add_argument("--cool-max-wait", type=int, default=180)
    ap.add_argument("--note", default="")
    ap.add_argument("--out-dir", type=Path, default=None, help="tests: write all outputs here instead")
    args = ap.parse_args()
    global OUT_DIR
    res_dir = args.out_dir or ROOT / "results"
    OUT_DIR = args.out_dir or OUT_DIR
    args.source, args.width, args.height = None, 1280, 720

    weights, confs, names = default_weights(), default_conf(), class_names()
    groups = {int(k): v for k, v in
              yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))["groups"].items()}
    dets = {m: load_detector(m, weights[m], conf=confs[m], max_det=10) for m in args.models}   # app settings
    gpu = torch.cuda.get_device_name(0)
    idle = gpu_state()[0]
    print(f"GPU {gpu}, idle {idle:.0f} C; cooling target {idle + args.cool_margin:.0f} C")
    cap = open_source(args)
    if not cap.isOpened():
        raise SystemExit("cannot open the camera")
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, fps_rows, rest_rows = [], [], []

    for m in args.models:   # part A
        for rep in range(1, args.repeats + 1):
            waited = cool_with_window(cap, idle + args.cool_margin, args.cool_max_wait, f"{LABELS[m]} #{rep}")
            loop(cap, dets[m], m, confs[m], names, groups, args.warmup,
                 f"{LABELS[m]} #{rep}: pemanasan - duduk di depan kamera, isyarat seperti biasa", [], "warmup")
            temp0, clk0 = gpu_state()
            stamps, cam_frames, _ = loop(cap, dets[m], m, confs[m], names, groups, args.seconds,
                                         f"{LABELS[m]} #{rep}: MENGUKUR {args.seconds:.0f} s - isyarat seperti biasa",
                                         frames, f"fps_{rep}")
            temp1, clk1 = gpu_state()
            span = stamps[-1] - stamps[0]
            pred = np.array([r["pred_ms"] for r in frames if r["model"] == m and r["tag"] == f"fps_{rep}"])
            row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": m, "repeat": rep,
                   "weights": str(Path(weights[m]).relative_to(ROOT)), "conf": confs[m], "gpu": gpu,
                   "seconds": round(span, 2), "frames": len(stamps), "display_fps": (len(stamps) - 1) / span,
                   "camera_fps": cam_frames / span, "pred_ms": pred.mean(), "idle_temp_c": idle,
                   "wait_cool_s": waited, "temp_start_c": temp0, "temp_end_c": temp1,
                   "sm_clock_start_mhz": clk0, "sm_clock_end_mhz": clk1, "note": args.note}
            fps_rows.append(row)
            print(f"{LABELS[m]:<22} #{rep} display {row['display_fps']:5.1f} FPS  camera {row['camera_fps']:5.1f}  "
                  f"model {row['pred_ms']:5.1f} ms  cooled {waited:.0f} s  GPU {temp0:.0f}->{temp1:.0f} C")

    if args.rest_seconds > 0:   # part B
        for m in args.models:
            dets[m].set_conf(REST_CONF)
            for left in range(5, 0, -1):
                loop(cap, dets[m], m, confs[m], names, groups, 1,
                     f"{LABELS[m]}: TURUNKAN KEDUA TANGAN ke posisi istirahat dan diam - mulai {left} s", [], "prep")
            _, _, best = loop(cap, dets[m], m, confs[m], names, groups, args.rest_seconds,
                              f"{LABELS[m]}: TANGAN ISTIRAHAT, jangan bergerak ({args.rest_seconds:.0f} s)",
                              frames, "rest", keep_best=True)
            dets[m].set_conf(confs[m])
            r = [x for x in frames if x["model"] == m and x["tag"] == "rest"]
            s = np.array([x["top_score"] for x in r])
            above = [x for x in r if x["top_score"] >= confs[m]]
            cls = {}
            for x in above:
                cls[x["top_class"]] = cls.get(x["top_class"], 0) + 1
            if best:
                cv2.imwrite(str(OUT_DIR / f"rest_{m}.jpg"), best[1])
            row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": m, "conf": confs[m],
                   "frames": len(r), "labelled_frames": len(above), "labelled_share": len(above) / len(r),
                   "score_median": float(np.median(s)), "score_p90": float(np.percentile(s, 90)),
                   "score_max": float(s.max()),
                   "labelled_classes": "; ".join(f"{k} {v}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1])),
                   "note": args.note}
            rest_rows.append(row)
            print(f"{LABELS[m]:<22} rest: {row['labelled_frames']}/{row['frames']} frames labelled "
                  f"(score >= {confs[m]:.3f}); top score median {row['score_median']:.2f}, "
                  f"p90 {row['score_p90']:.2f}, max {row['score_max']:.2f}; {row['labelled_classes']}")
    cap.release()
    cv2.destroyAllWindows()

    print("\ndisplay FPS, mean +- SD over repeats:")
    for m in args.models:
        r = [x for x in fps_rows if x["model"] == m]
        d, c = [x["display_fps"] for x in r], [x["camera_fps"] for x in r]
        print(f"  {LABELS[m]:<22} {np.mean(d):5.1f} +- {sd(d):4.1f}   (camera {np.mean(c):5.1f} +- {sd(c):4.1f})")

    for path, rows in ((res_dir / "app_fps.csv", fps_rows), (res_dir / "app_rest_check.csv", rest_rows),
                       (OUT_DIR / "frames.csv", frames)):
        if not rows:
            continue
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            for x in rows:
                w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in x.items()})
        print(f"appended {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
