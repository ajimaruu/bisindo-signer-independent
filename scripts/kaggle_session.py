"""One Kaggle notebook session: install pinned packages, rebuild the working tree, restore earlier runs,
probe the batch size once, run a plan of training runs and write a report.

Stages (all run in order; each is idempotent):
  env      install torch/torchvision at the version in requirements_freeze.txt (CUDA build: the laptop's
           first, then cu128, cu126 - the first that works on the Kaggle driver) and the other pinned
           packages; differences from the laptop -> runs/env_kaggle.json
  setup    /kaggle/working/bisindo: scripts, configs, splits, frames/labels (symlinks), weights; checks the
           sha256 manifest, regenerates data/coco + data/yolo with scripts/prepare_data.py and checks that the
           COCO files are byte-identical to the laptop's
  restore  copies runs/ from an attached output of an earlier session (checkpoints, logs)
  probe    once: the same batch for all three models, trying 16, 12, 8 -> runs/kaggle_batch.json
  plan     runs the plan (configs/kaggle_plan.yaml) in order: finished runs (eval_test.json) are skipped,
           unfinished ones resume from last.pt; stops before the session deadline (exit code 3 of the
           training scripts = stopped after a saved epoch, resumable)
  report   runs/session_report.json, and one row per session in runs/sessions.csv (GPU hours)

This file must not import torch or other pinned packages at module level: they are installed by 'env'.
Usage (Kaggle notebook): python -u /kaggle/input/<dataset>/scripts/kaggle_session.py --plan pilot
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1]           # /kaggle/input/<dataset>
WORK = Path("/kaggle/working/bisindo")
RUNS = WORK / "runs"
PY = sys.executable
T_START = time.time()
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
CUDA_FALLBACKS = ["cu128", "cu126"]
OTHER_PINS = ["ultralytics", "ultralytics-thop", "pycocotools", "numpy", "opencv-python", "pillow", "PyYAML",
              "polars", "matplotlib", "psutil", "requests"]
BATCH_CANDIDATES = [16, 12, 8]
MIN_START_S = 20 * 60          # do not start or resume a run with less than 20 min left
DEADLINE_EXIT = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T_START) / 60:5.1f} min] {msg}", flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, text=True, **kw)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pins():
    out = {}
    for line in (DATASET / "requirements_freeze.txt").read_text().splitlines():
        if "==" in line:
            name, ver = line.strip().split("==")
            out[name.lower()] = ver
    return out


# ---------------------------------------------------------------- env
def cuda_works():
    r = sh([PY, "-c", "import torch; assert torch.cuda.is_available(); "
                      "x = torch.ones(8, device='cuda'); print(torch.__version__, torch.version.cuda, (x * 2).sum().item())"],
           capture_output=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:]


def stage_env():
    p = pins()
    torch_ver, laptop_cuda = p["torch"].split("+")
    tv_ver = p["torchvision"].split("+")[0]
    chosen = None
    for cu in [laptop_cuda] + [c for c in CUDA_FALLBACKS if c != laptop_cuda]:
        log(f"installing torch=={torch_ver} torchvision=={tv_ver} ({cu})")
        r = sh([PY, "-m", "pip", "install", "-q", f"torch=={torch_ver}", f"torchvision=={tv_ver}",
                "--index-url", TORCH_INDEX.format(cu)], capture_output=True)
        ok, msg = cuda_works() if r.returncode == 0 else (False, r.stderr.strip().splitlines()[-1:])
        log(f"  {cu}: {'OK' if ok else 'not usable'} {msg}")
        if ok:
            chosen = cu
            break
    if chosen is None:
        sys.exit("no torch build of the pinned version works on this GPU/driver")
    others = [f"{n}=={p[n.lower()]}" for n in OTHER_PINS if n.lower() in p]
    log("installing " + " ".join(others))
    sh([PY, "-m", "pip", "install", "-q", *others], check=True)

    freeze = {}
    for line in sh([PY, "-m", "pip", "freeze"], capture_output=True).stdout.splitlines():
        if "==" in line:
            n, v = line.split("==")
            freeze[n.lower()] = v
    diffs = [{"package": n, "laptop": v, "kaggle": freeze.get(n, "missing")}
             for n, v in sorted(p.items()) if freeze.get(n) != v]
    info = json.loads(sh([PY, "-c", (
        "import json, platform, torch, torchvision, ultralytics, numpy, cv2;"
        "print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__,"
        "'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),"
        "'torchvision': torchvision.__version__, 'ultralytics': ultralytics.__version__,"
        "'numpy': numpy.__version__, 'opencv': cv2.__version__,"
        "'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")],
        capture_output=True, check=True).stdout)
    smi = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
             capture_output=True).stdout.strip().splitlines()
    env = {**info, "nvidia_smi": smi, "torch_build_chosen": chosen, "laptop_torch_build": laptop_cuda,
           "differences_from_laptop": diffs, "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "env_kaggle.json").write_text(json.dumps(env, indent=1))
    log(f"GPU: {smi}  torch {info['torch']} (CUDA {info['torch_cuda']})")
    log(f"differences from laptop ({len(diffs)}): "
        + ", ".join(f"{d['package']} {d['laptop']} -> {d['kaggle']}" for d in diffs[:12])
        + (" ..." if len(diffs) > 12 else ""))
    return env


# ---------------------------------------------------------------- setup
def stage_setup():
    for d in ("scripts", "configs", "splits"):
        shutil.copytree(DATASET / d, WORK / d, dirs_exist_ok=True)
    (WORK / "data" / "models").mkdir(parents=True, exist_ok=True)
    for name, target in (("frames", DATASET / "frames"), ("labels_yolo", DATASET / "labels")):
        link = WORK / "data" / name
        if not link.exists():
            os.symlink(target, link, target_is_directory=True)
    for w in ("yolo11m.pt", "rtdetr-l.pt"):
        if not (WORK / "data" / "models" / w).exists():
            shutil.copy2(DATASET / "weights" / w, WORK / "data" / "models" / w)
    hub = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    for f in (DATASET / "weights").glob("fasterrcnn_*.pth"):
        if not (hub / f.name).exists():
            shutil.copy2(f, hub / f.name)

    bad = []
    for line in (DATASET / "manifest_sha256.txt").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if sha256(DATASET / rel) != digest:
            bad.append(rel)
    if bad:
        sys.exit(f"dataset files differ from the laptop manifest: {bad[:5]}")
    log("dataset manifest OK (frames, labels, weights)")

    r = sh([PY, str(WORK / "scripts" / "prepare_data.py")], cwd=WORK, capture_output=True)
    if r.returncode:
        sys.exit(f"prepare_data.py failed:\n{r.stdout}\n{r.stderr}")
    expected = dict(line.split("  ")[::-1] for line in (DATASET / "coco_sha256.txt").read_text().splitlines())
    got = {f"data/coco/{p.name}": sha256(p) for p in sorted((WORK / "data" / "coco").glob("*.json"))}
    if got != expected:
        sys.exit("regenerated COCO files differ from the laptop's")
    log(f"COCO ground truth identical to the laptop ({len(got)} files); Ultralytics data in data/yolo/")


# ---------------------------------------------------------------- restore
def stage_restore():
    """Copy runs/ of the newest attached earlier session (Kaggle mounts notebook outputs under /kaggle/input)."""
    found = [Path(p) for p in glob.glob("/kaggle/input/**/bisindo/runs", recursive=True)]
    found = [p for p in found if (p / "sessions.csv").exists()]
    if not found:
        log("restore: no earlier session attached; starting fresh")
        return
    newest = max(found, key=lambda p: (p / "sessions.csv").read_text().strip().splitlines()[-1])
    log(f"restore: copying {newest}")
    shutil.copytree(newest, RUNS, dirs_exist_ok=True)


# ---------------------------------------------------------------- training calls
def train_cmd(model, split, seed, batch, epochs=None, patience=None, name=None, probe=False):
    if model == "frcnn-r50fpn":
        cmd = [PY, "-u", "scripts/train_frcnn.py", "--split", split]
    else:
        cmd = [PY, "-u", "scripts/train_ultralytics.py", "--model", model, "--split", split]
    cmd += ["--seed", str(seed), "--batch", str(batch)]
    if probe:
        return cmd + ["--probe"]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    if patience:
        cmd += ["--patience", str(patience)]
    if name:
        cmd += ["--name", name]
    return cmd


def run_logged(cmd, logfile, deadline):
    env = {**os.environ, "BISINDO_DEADLINE": str(deadline), "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        for line in proc.stdout:
            f.write(line)
            # echo epoch summaries and errors, not progress bars
            if not any(s in line for s in ("━", "──", "\r", "it/s", "s/it")):
                print(line, end="", flush=True)
        return proc.wait()


def run_dir(model, name):
    return RUNS / ("frcnn" if model == "frcnn-r50fpn" else "ultralytics") / name


def epoch_log(model, rdir):
    path = rdir / ("results.csv" if model == "frcnn-r50fpn" else "gpu_log.csv")
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- probe
def gpu_name(env):
    return env["gpus"][0] if env.get("gpus") else "unknown"


def stage_probe(deadline, env):
    out = RUNS / "kaggle_batch.json"
    if out.exists():
        res = json.loads(out.read_text())
        if res["gpu"] != gpu_name(env):   # all 30 runs must use one GPU type
            sys.exit(f"this session got {gpu_name(env)}, but the batch was probed on {res['gpu']}; "
                     "choose the same accelerator as before")
        log(f"probe: batch {res['chosen']} on {res['gpu']} (from an earlier session)")
        return res["chosen"]
    tried = []
    for b in BATCH_CANDIDATES:
        results = {}
        for model in ("yolo11m", "rtdetr-l", "frcnn-r50fpn"):
            log(f"probe {model} batch {b}")
            code = run_logged(train_cmd(model, "p1", 0, b, probe=True), RUNS / "logs" / f"probe_{model}_b{b}.log",
                              deadline)
            rows = epoch_log(model, RUNS / "probe" / (f"frcnn_b{b}" if model == "frcnn-r50fpn" else f"{model}_b{b}"))
            peak = float(rows[-1]["peak_reserved_gib"]) if rows and code == 0 else None
            results[model] = {"ok": code == 0, "peak_reserved_gib": peak,
                              "train_s_295_images": float(rows[-1]["train_s"]) if rows and code == 0 else None}
            log(f"  {model} b{b}: {'fits' if code == 0 else f'FAILED (exit {code})'} peak {peak} GiB")
            if code != 0:
                break
        tried.append({"batch": b, "results": results})
        if all(r["ok"] for r in results.values()) and len(results) == 3:
            res = {"chosen": b, "gpu": gpu_name(env), "tried": tried,
                   "rule": "largest batch in 16/12/8 that fits all three models"}
            out.write_text(json.dumps(res, indent=1))
            log(f"probe: batch {b} fits all three models")
            return b
    sys.exit("no common batch in 16/12/8 fits all three models; ask before choosing per-model batches")


# ---------------------------------------------------------------- plan
def stage_plan(plan_name, batch, deadline):
    import yaml   # installed / pinned by stage_env
    plan = yaml.safe_load(open(WORK / "configs" / "kaggle_plan.yaml", encoding="utf-8"))[plan_name]
    touched = []
    for r in plan:
        name = r.get("name") or f"{r['model']}_{r['split']}_seed{r['seed']}"
        rdir = run_dir(r["model"], name)
        if (rdir / "eval_test.json").exists():
            log(f"{name}: finished earlier, skipped")
            continue
        if deadline - time.time() < MIN_START_S:
            log(f"{name}: not started, less than {MIN_START_S // 60} min left in this session")
            break
        log(f"{name}: {'resuming' if (rdir / 'weights' / 'last.pt').exists() else 'starting'}")
        code = run_logged(train_cmd(r["model"], r["split"], r["seed"], batch, r.get("epochs"), r.get("patience"),
                                    name), RUNS / "logs" / f"{name}.log", deadline)
        touched.append(name)
        if code == DEADLINE_EXIT:
            log(f"{name}: paused before the session deadline; attach this output to the next session to resume")
            break
        log(f"{name}: {'finished' if code == 0 else f'FAILED (exit {code}), see runs/logs/{name}.log'}")
    return plan, touched


# ---------------------------------------------------------------- report
def stage_report(plan_name, plan, batch, touched, env):
    rows = []
    for r in plan or []:
        name = r.get("name") or f"{r['model']}_{r['split']}_seed{r['seed']}"
        rdir = run_dir(r["model"], name)
        ep = epoch_log(r["model"], rdir)
        ev = json.loads((rdir / "eval_test.json").read_text()) if (rdir / "eval_test.json").exists() else None
        times = [float(e["train_val_s"]) for e in ep]
        if r["model"] == "frcnn-r50fpn":
            fit = [float(e["val_map50_95"]) for e in ep]
        else:
            fit = [float(e["fitness"]) for e in ep]
        best = max(range(len(fit)), key=lambda i: fit[i]) + 1 if fit else None
        rows.append({"run": name, "model": r["model"], "epochs_done": len(ep), "epochs_max": r.get("epochs"),
                     "patience": r.get("patience"), "best_epoch": best,
                     "best_val_map50_95": max(fit) if fit else None,
                     "mean_epoch_s": sum(times) / len(times) if times else None,
                     "total_train_h": sum(times) / 3600, "finished": ev is not None,
                     "test_map50_95_pycocotools": ev["pycocotools"]["map50_95"] if ev else None,
                     "test_map50_95_ultralytics": ev.get("ultralytics", {}).get("map50_95") if ev else None})
    gpu_h = (time.time() - T_START) / 3600
    report = {"plan": plan_name, "gpu": env.get("nvidia_smi"), "batch": batch, "session_hours": gpu_h,
              "runs": rows, "touched_this_session": touched}
    (RUNS / "session_report.json").write_text(json.dumps(report, indent=1))
    new = not (RUNS / "sessions.csv").exists()
    with open(RUNS / "sessions.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["start_utc", "end_utc", "hours", "gpu", "plan", "batch", "runs_touched"])
        w.writerow([datetime.fromtimestamp(T_START, timezone.utc).isoformat(timespec="seconds"),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"), round(gpu_h, 3),
                    "; ".join(env.get("nvidia_smi", [])), plan_name, batch, " ".join(touched)])
    log(f"session: {gpu_h:.2f} h, GPU {env.get('nvidia_smi')}, batch {batch}")
    for r in rows:
        log(f"  {r['run']}: epochs {r['epochs_done']}/{r['epochs_max']}, best epoch {r['best_epoch']}, "
            f"{(r['mean_epoch_s'] or 0) / 60:.1f} min/epoch, finished {r['finished']}, "
            f"test mAP50-95 coco {r['test_map50_95_pycocotools']} / ultralytics {r['test_map50_95_ultralytics']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="pilot", help="plan name in configs/kaggle_plan.yaml")
    ap.add_argument("--budget-hours", type=float, default=11.0, help="stop before this (Kaggle limit is 12 h)")
    args = ap.parse_args()
    deadline = T_START + args.budget_hours * 3600
    WORK.mkdir(parents=True, exist_ok=True)

    env = stage_env()
    stage_setup()
    stage_restore()
    (RUNS / "env_kaggle.json").write_text(json.dumps(env, indent=1))   # this session's environment
    batch = stage_probe(deadline, env)
    plan, touched = stage_plan(args.plan, batch, deadline)
    stage_report(args.plan, plan, batch, touched, env)
    # data/ is rebuilt from the dataset every session (symlinks, weight copies, COCO/YOLO views): keep it out
    # of the saved notebook output, which then holds only runs/ (checkpoints, logs, reports).
    shutil.rmtree(WORK / "data", ignore_errors=True)


if __name__ == "__main__":
    main()
