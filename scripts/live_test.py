"""Live per-sign test mode of the desktop application (manuscript Table 8: RT-DETR-L on new signers,
10 trials per sign). Started with:  python scripts/app_detect.py --live-test   (or python scripts/live_test.py)

Fixed (decided 5 Nov 2026, config_log section 34):
  - model RT-DETR-L, Protocol 1 seed-0 best.pt from configs/app.yaml; threshold = that run's conf_thr in
    results/runs.csv (checked against configs/app.yaml; the script stops if they differ);
  - FP16, Ultralytics predict() exactly as the application (imgsz 640; RT-DETR's predictor resizes the frame to a
    640x640 tensor without padding (scale_fill); the measured tensor shape is written to the session file);
  - camera frames are not mirrored;
  - during the 3-second recording only the camera image and "MEREKAM" are shown (no boxes or labels), so the
    signer cannot adapt to the model's output; the result is shown after the trial.

One trial: target gloss shown with "posisi netral" -> SPACE -> 3 s countdown (model not run) -> 3 s recording
(model on every frame; per frame the top label at or above the threshold with its score, else "none")
-> SUCCESS when the target is the label on >= 5 consecutive frames (scripts/live_rule.py), else MISS with the
most frequent other label -> result shown -> next trial.
Keys: SPACE start, R repeat the last finished trial (its row stays, flagged repeated=1), Q quit (every trial is
written when it finishes).

Order: 15 signs x 10 trials, shuffled with a random seed written to the session file (or sequential).
Resume: the same participant ID continues with the first trial without a valid (not repeated) result, in the
order stored in session_{ID}.json.

Outputs (results/live_test/ or --out-dir): trials_{ID}.csv, frames_{ID}.csv, session_{ID}.json.
No video is stored.
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors import ROOT, class_names, default_conf, default_weights, load_detector  # noqa: E402
from live_rule import NEED, NONE, judge  # noqa: E402

MODEL = "rtdetr-l"
OUT = ROOT / "results" / "live_test"
COUNTDOWN_S, RECORD_S, RESULT_S = 3.0, 3.0, 1.8
WIN = "BISINDO - mode uji per isyarat"
TRIAL_COLS = ["participant", "signer_status", "distance_m", "lighting", "trial_no", "gloss_id", "gloss", "hands",
              "result", "max_consecutive_target", "frames_in_window", "label_on_miss", "mean_fps", "threshold",
              "checkpoint_path", "timestamp", "repeated"]
FRAME_COLS = ["trial_no", "attempt", "frame_idx", "pred_label", "score"]


# ---------------------------------------------------------------- setup
def locked_model():
    """Weights and threshold of the application's RT-DETR-L, checked against results/runs.csv (P1 seed 0)."""
    weights, conf = default_weights()[MODEL], default_conf()[MODEL]
    run = [r for r in csv.DictReader(open(ROOT / "results" / "runs.csv", encoding="utf-8"))
           if r["model"] == MODEL and r["protocol"] == "P1" and r["seed"] == "0"]
    if len(run) != 1:
        raise SystemExit("runs.csv: RT-DETR-L P1 seed 0 not found")
    run = run[0]
    if Path(weights).resolve() != (ROOT / run["checkpoint_path"]).resolve():
        raise SystemExit(f"configs/app.yaml weights {weights} != runs.csv checkpoint {run['checkpoint_path']}")
    if abs(conf - float(run["conf_thr"])) > 1e-9:
        raise SystemExit(f"configs/app.yaml conf {conf} != runs.csv conf_thr {run['conf_thr']}")
    return Path(weights), float(run["conf_thr"]), run["checkpoint_path"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version():
    def git(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "script": "scripts/live_test.py", "script_sha256": sha256(__file__)}


def form(defaults):
    """Session form (tkinter). Returns a dict or None when cancelled."""
    import tkinter as tk
    from tkinter import messagebox, ttk
    root = tk.Tk()
    root.title("Mode uji per isyarat - data sesi")
    vals, out = {}, {}
    rows = [("participant", "ID peserta (mis. P01)", "entry"), ("signer_status", "Status pemeraga", ("Tuli", "dengar")),
            ("distance_m", "Jarak ke kamera (m)", "entry"), ("lighting", "Kondisi cahaya", "entry"),
            ("notes", "Catatan", "entry"), ("order_mode", "Urutan percobaan", ("acak", "berurutan"))]
    for i, (key, label, kind) in enumerate(rows):
        ttk.Label(root, text=label).grid(row=i, column=0, sticky="w", padx=8, pady=4)
        v = tk.StringVar(value=str(defaults.get(key, "")))
        w = ttk.Entry(root, textvariable=v, width=36) if kind == "entry" else \
            ttk.Combobox(root, textvariable=v, values=kind, state="readonly", width=33)
        w.grid(row=i, column=1, padx=8, pady=4)
        vals[key] = v
    ttk.Label(root, text="ID yang sudah ada: sesi dilanjutkan dengan urutan yang tersimpan.",
              foreground="#555").grid(row=len(rows), column=0, columnspan=2, padx=8)

    def ok():
        d = {k: v.get().strip() for k, v in vals.items()}
        if not d["participant"] or not d["participant"].replace("_", "").replace("-", "").isalnum():
            return messagebox.showerror("Data sesi", "ID peserta wajib diisi (huruf/angka, mis. P01).")
        if d["signer_status"] not in ("Tuli", "dengar"):
            return messagebox.showerror("Data sesi", "Pilih status pemeraga.")
        try:
            float(d["distance_m"].replace(",", "."))
        except ValueError:
            return messagebox.showerror("Data sesi", "Jarak harus angka (meter), mis. 1.5")
        d["distance_m"] = d["distance_m"].replace(",", ".")
        out.update(d)
        root.destroy()

    ttk.Button(root, text="Mulai", command=ok).grid(row=len(rows) + 1, column=1, sticky="e", padx=8, pady=8)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return out or None


def tensor_shape(det, frame):
    """The input tensor shape the predictor actually builds for this frame size (also the warm-up prediction)."""
    det(frame)   # the predictor is created on the first call
    p = det.model.predictor
    orig, seen = p.preprocess, []
    p.preprocess = lambda im: (lambda t: (seen.append(list(t.shape)), t)[1])(orig(im))
    det(frame)
    p.preprocess = orig
    return seen[0]


# ---------------------------------------------------------------- files
def read_trials(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_trials(path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def append_frames(path, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_COLS)
        if new:
            w.writeheader()
        w.writerows(rows)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- screens
def banner(frame, lines, colour=(255, 255, 255), bg=(0, 0, 0), big=0):
    """Text lines over a dark band at the top; line `big` in large type."""
    out = frame.copy()
    h = 70 + 46 * (len(lines) - 1) + 30
    band = out.copy()
    cv2.rectangle(band, (0, 0), (out.shape[1], h), bg, -1)
    cv2.addWeighted(band, 0.65, out, 0.35, 0, out)
    y = 62
    for i, text in enumerate(lines):
        scale, thick = (1.9, 4) if i == big else (0.9, 2)
        cv2.putText(out, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)
        y += 70 if i == big else 46
    return out


def show(img):
    cv2.imshow(WIN, img)
    return cv2.waitKey(1) & 0xFF


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT, help="tests/simulation: write here instead")
    ap.add_argument("--gestures", type=int, nargs="+", default=None, help="tests: only these gloss ids")
    ap.add_argument("--reps", type=int, default=10, help="trials per sign (paper: 10)")
    ap.add_argument("--participant", help="skip the form (tests); with --status/--distance/--lighting/--order")
    ap.add_argument("--status", default="dengar")
    ap.add_argument("--distance", default="1.5")
    ap.add_argument("--lighting", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--order", choices=("acak", "berurutan"), default="acak")
    ap.add_argument("--auto", action="store_true", help="tests: start every trial without SPACE")
    args = ap.parse_args(argv)

    weights, thr, ckpt_rel = locked_model()
    names = class_names()
    groups = {int(k): v for k, v in
              yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))["groups"].items()}
    gloss_ids = args.gestures if args.gestures is not None else sorted(names)
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.participant:
        meta = {"participant": args.participant, "signer_status": args.status, "distance_m": args.distance,
                "lighting": args.lighting, "notes": args.notes, "order_mode": args.order}
    else:
        last = sorted(out_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime)
        prev = json.loads(last[-1].read_text(encoding="utf-8")) if last else {}
        meta = form({"participant": "", "signer_status": prev.get("signer_status", "Tuli"),
                     "distance_m": prev.get("distance_m", ""), "lighting": prev.get("lighting", ""),
                     "notes": "", "order_mode": "acak"})
        if meta is None:
            print("cancelled")
            return
    pid = meta["participant"]
    f_session, f_trials, f_frames = (out_dir / f"session_{pid}.json", out_dir / f"trials_{pid}.csv",
                                     out_dir / f"frames_{pid}.csv")

    det = load_detector(MODEL, weights, conf=0.01, max_det=10)   # threshold applied below; top det unchanged
    now = datetime.now().isoformat(timespec="seconds")
    if f_session.exists():
        session = json.loads(f_session.read_text(encoding="utf-8"))
        if session["threshold"] != thr or session["checkpoint_path"] != ckpt_rel:
            raise SystemExit("session file was made with another checkpoint/threshold")
        order = session["order"]
        print(f"resuming {pid}: order seed {session['order_seed']}, {len(order)} trials")
    else:
        order = [g for g in gloss_ids for _ in range(args.reps)]
        seed = None
        if meta["order_mode"] == "acak":
            seed = random.SystemRandom().randrange(2 ** 31)
            random.Random(seed).shuffle(order)
        import torch
        import ultralytics
        session = {"participant": pid, "created": now, "model": MODEL, "checkpoint_path": ckpt_rel,
                   "checkpoint_sha256": sha256(weights), "threshold": thr,
                   "threshold_source": "results/runs.csv conf_thr (rtdetr-l, P1, seed 0), selected on validation",
                   "rule": f">= {NEED} consecutive frames with the target as top label >= threshold",
                   "countdown_s": COUNTDOWN_S, "record_s": RECORD_S, "precision": "fp16", "imgsz": 640,
                   "mirrored": False, "labels_shown_during_recording": False,
                   "order_mode": meta["order_mode"], "order_seed": seed, "reps_per_sign": args.reps,
                   "gloss_ids": gloss_ids, "order": order, "code": code_version(),
                   "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                                "ultralytics": ultralytics.__version__, "opencv": cv2.__version__},
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "sessions": []}
    session.update({k: meta[k] for k in ("signer_status", "distance_m", "lighting", "notes")})

    from app_detect import open_source
    cam_args = argparse.Namespace(source=None, camera=args.camera, width=1280, height=720)
    cap = open_source(cam_args)
    if not cap.isOpened():
        raise SystemExit("cannot open the camera")
    seq = 0
    for _ in range(100):   # first frame
        ok, frame, seq = cap.read(seq)
        if ok:
            break
    session["sessions"].append({"start": now, **meta, "code": code_version()})
    session["camera"] = {"index": args.camera, "backend": cap.cap.getBackendName(),
                         "resolution": [int(frame.shape[1]), int(frame.shape[0])]}
    session["input_tensor_shape"] = tensor_shape(det, frame)
    session["input_note"] = ("Ultralytics RT-DETR predictor (scale_fill=True): the camera frame is resized to "
                             "640x640 without padding, as in the application, the mAP evaluation and the FPS runs")
    save_json(f_session, session)
    print(f"{pid}: threshold {thr}, checkpoint {ckpt_rel}, camera {session['camera']}, "
          f"tensor {session['input_tensor_shape']}")

    trials = read_trials(f_trials)
    attempts = {}   # attempt number per trial; from frames_{ID}.csv too, so an attempt is never numbered twice
    for r in trials:
        attempts[int(r["trial_no"])] = attempts.get(int(r["trial_no"]), 0) + 1
    if f_frames.exists():
        with open(f_frames, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                attempts[int(r["trial_no"])] = max(attempts.get(int(r["trial_no"]), 0), int(r["attempt"]))

    def next_trial():
        done = {int(r["trial_no"]) for r in trials if r["repeated"] == "0"}
        return next((i for i in range(1, len(order) + 1) if i not in done), None)

    def live(seconds, draw_fn, run_model=False):
        """Camera loop for `seconds`; returns (key pressed or None, [(t, label, score)])."""
        nonlocal seq
        t0, recs = time.perf_counter(), []
        while time.perf_counter() - t0 < seconds:
            ok, frame, seq = cap.read(seq)
            if not ok:
                raise SystemExit("camera stopped")
            if run_model:
                dets, _ = det(frame)
                top = max(dets, key=lambda d: d[4]) if dets else None
                score = float(top[4]) if top else 0.0
                label = names[int(top[5])] if top and score >= thr else NONE
                recs.append((time.perf_counter(), label, score))
            k = show(draw_fn(frame, time.perf_counter() - t0))
            if k in (ord("q"), ord("Q"), 27):
                return "q", recs
        return None, recs

    last_done = None
    msg = ""
    while True:
        tno = next_trial()
        if tno is None:
            show(banner(frame, [f"{pid}: semua {len(order)} percobaan selesai", "Tekan tombol apa saja"], big=-1))
            cv2.waitKey(0 if not args.auto else 500)
            break
        gid = order[tno - 1]
        gloss = names[gid]
        head = f"{pid}   percobaan {tno}/{len(order)}"
        # waiting screen
        if not args.auto:
            key = None
            while key is None:
                ok, frame, seq = cap.read(seq)
                k = show(banner(frame, [head, gloss, "Posisi netral.  SPASI = mulai   R = ulangi terakhir   Q = keluar",
                                        msg], big=1))
                if k == ord(" "):
                    key = "start"
                elif k in (ord("q"), ord("Q"), 27):
                    key = "q"
                elif k in (ord("r"), ord("R")) and last_done is not None:
                    key = "r"
            if key == "q":
                break
            if key == "r":
                for r in trials:
                    if int(r["trial_no"]) == last_done and r["repeated"] == "0":
                        r["repeated"] = "1"
                write_trials(f_trials, trials)
                msg = f"percobaan {last_done} akan diulang"
                last_done = None
                continue
        msg = ""
        k, _ = live(COUNTDOWN_S, lambda f, t: banner(f, [head, gloss, f"Mulai dalam {COUNTDOWN_S - t:.0f}"], big=1))
        if k == "q":
            break
        k, recs = live(RECORD_S, lambda f, t: banner(f, ["MEREKAM", f"{RECORD_S - t:.1f} s"], colour=(80, 80, 255),
                                                     big=0), run_model=True)
        if k == "q":
            break   # an interrupted recording is not stored
        preds = [lab for _, lab, _ in recs]
        success, mc, miss_label = judge(preds, gloss)
        span = recs[-1][0] - recs[0][0] if len(recs) > 1 else float("nan")
        attempts[tno] = attempts.get(tno, 0) + 1
        row = {"participant": pid, "signer_status": meta["signer_status"], "distance_m": meta["distance_m"],
               "lighting": meta["lighting"], "trial_no": tno, "gloss_id": gid, "gloss": gloss, "hands": groups[gid],
               "result": "success" if success else "miss", "max_consecutive_target": mc,
               "frames_in_window": len(recs), "label_on_miss": miss_label,
               "mean_fps": round((len(recs) - 1) / span, 2) if len(recs) > 1 else "",
               "threshold": thr, "checkpoint_path": ckpt_rel,
               "timestamp": datetime.now().isoformat(timespec="seconds"), "repeated": "0"}
        trials.append(row)
        write_trials(f_trials, trials)
        append_frames(f_frames, [{"trial_no": tno, "attempt": attempts[tno], "frame_idx": i, "pred_label": lab,
                                  "score": round(s, 4)} for i, (_, lab, s) in enumerate(recs)])
        last_done = tno
        res = ["BERHASIL", f"{gloss}: {mc} frame berturut-turut"] if success else \
              ["MISS", f"label terbanyak: {miss_label}   (target maks. {mc} berturut-turut)"]
        colour = (60, 200, 60) if success else (60, 60, 230)
        k, _ = live(RESULT_S, lambda f, t: banner(f, [head, *res], colour=colour, big=1))
        print(f"{tno:>3} {gloss:<13} {row['result']:<7} run {mc:>2}  frames {len(recs):>3}  "
              f"fps {row['mean_fps']}  {miss_label}")
        if k == "q":
            break
    cap.release()
    cv2.destroyAllWindows()
    valid = [r for r in trials if r["repeated"] == "0"]
    print(f"{pid}: {len(valid)}/{len(order)} trials done, "
          f"{sum(r['result'] == 'success' for r in valid)} success -> {f_trials}")


if __name__ == "__main__":
    main()
