"""One Kaggle notebook session: install pinned packages, rebuild the working tree, restore earlier runs,
probe the batch size once, run a plan of training runs and write a report.

Stages (all run in order; each is idempotent):
  env      install torch/torchvision at the version in requirements_freeze.txt (CUDA build: the laptop's
           first, then cu128, cu126 - the first that works on the Kaggle driver) and the other pinned
           packages; differences from the laptop -> runs/env_<platform>.json
  setup    /kaggle/working/bisindo: scripts, configs, splits, frames/labels (symlinks), weights; checks the
           sha256 manifest, regenerates data/coco + data/yolo with scripts/prepare_data.py and checks that the
           COCO files are byte-identical to the laptop's
  restore  copies runs/ from an attached output of an earlier session (checkpoints, logs)
  probe    once: the same batch for all three models, trying 16, 12, 8 -> runs/kaggle_batch.json
  plan     runs the plan (configs/kaggle_plan.yaml) in order: finished runs (eval_test.json) are skipped,
           unfinished ones resume from last.pt; stops before the session deadline (--max-hours) (exit code 3 of the
           training scripts = stopped after a saved epoch, resumable)
  report   runs/session_report.json, and one row per session in runs/sessions.csv (GPU hours)

This file must not import torch or other pinned packages at module level: they are installed by 'env'.
Usage (Kaggle notebook): python -u /kaggle/input/<dataset>/scripts/kaggle_session.py --plan main --max-hours 10.8

Google Colab (fold s4, decided 31 Oct 2026; scripts/colab_notebook.ipynb): the same script and dataset zip with
BISINDO_PLATFORM=colab. Then the environment is written to runs/env_colab.json, --max-hours may go up to
MAX_HOURS_CAP_COLAB (Colab's own runtime limit is 24 h), and runs/ is a folder on Google Drive that persists between
sessions (the notebook links it into BISINDO_WORK), so there is nothing to restore. Everything else (pinned
packages, albumentations removed, guards, the Tesla T4 check, checkpoints every epoch, resume) is unchanged.

Local rehearsal (laptop, no package installs; see results/config_log.md): set BISINDO_DATASET (the built dataset
folder), BISINDO_WORK (a scratch folder), BISINDO_INPUT (folder searched for an earlier session's runs/),
BISINDO_SKIP_INSTALL=1 and optionally BISINDO_PROBE_BATCHES (e.g. "2").
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

DATASET = Path(os.environ.get("BISINDO_DATASET", Path(__file__).resolve().parents[1]))   # /kaggle/input/<dataset>
WORK = Path(os.environ.get("BISINDO_WORK", "/kaggle/working/bisindo"))
INPUT = Path(os.environ.get("BISINDO_INPUT", "/kaggle/input"))
SKIP_INSTALL = os.environ.get("BISINDO_SKIP_INSTALL") == "1"
PLATFORM = os.environ.get("BISINDO_PLATFORM", "kaggle")   # kaggle | colab
RUNS = WORK / "runs"
PY = sys.executable
T_START = time.time()
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
CUDA_FALLBACKS = ["cu128", "cu126"]
OTHER_PINS = ["ultralytics", "ultralytics-thop", "pycocotools", "numpy", "opencv-python", "pillow", "PyYAML",
              "polars", "matplotlib", "psutil", "requests"]
BATCH_CANDIDATES = [int(b) for b in os.environ.get("BISINDO_PROBE_BATCHES", "16 12 8").split()]
DEFAULT_MAX_HOURS = 10.8      # notebook cell MAX_HOURS; Kaggle's own limit is 12 h per session
MAX_HOURS_CAP = 11.5
MAX_HOURS_CAP_COLAB = 23.5
MIN_START_S = 20 * 60          # do not start or resume a run with less than 20 min left
DEADLINE_EXIT = 3
FORBIDDEN_EXIT = 4             # run_logged: a forbidden line appeared in a run's output
FORBIDDEN_LINES = ("albumentations:",)   # Ultralytics prints this only when albumentations transforms are active
HASH_THREADS = 32              # /kaggle/input is a network mount: hashing is latency-bound, so read in parallel
JSON_MARK = "@@BISINDO_JSON@@"   # child processes may print other things (e.g. Ultralytics' first-run settings notice)


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


def child_json(code):
    """Run `code` in a fresh interpreter; it prints JSON_MARK + json.dumps(...). Only that line is parsed, so any
    other output on stdout (package notices, warnings) is harmless. On failure the full output is shown."""
    r = sh([PY, "-c", code], capture_output=True)
    marked = [line[len(JSON_MARK):] for line in r.stdout.splitlines() if line.startswith(JSON_MARK)]
    try:
        if r.returncode == 0 and len(marked) == 1:
            return json.loads(marked[0])
        problem = f"exit code {r.returncode}, {len(marked)} marked line(s)"
    except json.JSONDecodeError as e:
        problem = f"invalid JSON: {e}"
    sys.exit(f"child python failed ({problem})\n--- code ---\n{code}\n--- stdout ---\n{r.stdout}"
             f"\n--- stderr ---\n{r.stderr}")


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
    if SKIP_INSTALL:   # local rehearsal: use the interpreter's own packages
        log("BISINDO_SKIP_INSTALL=1: no package installs")
        chosen = "skipped"
    for cu in [] if SKIP_INSTALL else [laptop_cuda] + [c for c in CUDA_FALLBACKS if c != laptop_cuda]:
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
    if not SKIP_INSTALL:
        log("installing " + " ".join(others))
        sh([PY, "-m", "pip", "install", "-q", *others], check=True)
        # preinstalled on Kaggle, absent on the laptop; Ultralytics would add Blur/MedianBlur/ToGray/CLAHE
        log("uninstalling albumentations (not part of the locked augmentation)")
        sh([PY, "-m", "pip", "uninstall", "-y", "-q", "albumentations"])

    freeze = {}
    for line in sh([PY, "-m", "pip", "freeze"], capture_output=True).stdout.splitlines():
        if "==" in line:
            n, v = line.split("==")
            freeze[n.lower()] = v
    diffs = [{"package": n, "laptop": v, PLATFORM: freeze.get(n, "missing")}
             for n, v in sorted(p.items()) if freeze.get(n) != v]
    info = child_json(
        "import json, platform, importlib.util, torch, torchvision, ultralytics, numpy, cv2;"
        "from torch._dynamo.utils import is_compile_supported;"
        f"print({JSON_MARK!r} + json.dumps({{'python': platform.python_version(), 'torch': torch.__version__,"
        "'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),"
        "'torchvision': torchvision.__version__, 'ultralytics': ultralytics.__version__,"
        "'numpy': numpy.__version__, 'opencv': cv2.__version__,"
        "'albumentations_importable': importlib.util.find_spec('albumentations') is not None,"
        "'triton_importable': importlib.util.find_spec('triton') is not None,"
        "'torch_compile_supported_cuda': is_compile_supported('cuda'),"
        "'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")
    smi = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
             capture_output=True).stdout.strip().splitlines()
    if info["albumentations_importable"]:
        sys.exit("albumentations is still importable after uninstalling it")
    env = {**info, "platform": PLATFORM, "nvidia_smi": smi, "torch_build_chosen": chosen,
           "laptop_torch_build": laptop_cuda,
           "differences_from_laptop": diffs, "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"env_{PLATFORM}.json").write_text(json.dumps(env, indent=1))
    log(f"GPU: {smi}  torch {info['torch']} (CUDA {info['torch_cuda']})")
    log(f"differences from laptop ({len(diffs)}): "
        + ", ".join(f"{d['package']} {d['laptop']} -> {d[PLATFORM]}" for d in diffs[:12])
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
            if os.name == "nt":   # local rehearsal on Windows: junction (symlinks need admin rights)
                sh(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
            else:
                os.symlink(target, link, target_is_directory=True)
    for w in ("yolo11m.pt", "rtdetr-l.pt"):
        if not (WORK / "data" / "models" / w).exists():
            shutil.copy2(DATASET / "weights" / w, WORK / "data" / "models" / w)
    hub = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    for f in (DATASET / "weights").glob("fasterrcnn_*.pth"):
        if not (hub / f.name).exists():
            shutil.copy2(f, hub / f.name)

    entries = [line.split("  ", 1) for line in (DATASET / "manifest_sha256.txt").read_text().splitlines()]
    t0 = time.time()
    with ThreadPoolExecutor(HASH_THREADS) as pool:   # every file still fully hashed with sha256
        got = list(pool.map(lambda e: sha256(DATASET / e[1]), entries))
    bad = [rel for (digest, rel), h in zip(entries, got) if h != digest]
    if bad:
        sys.exit(f"dataset files differ from the laptop manifest: {bad[:5]}")
    log(f"dataset manifest OK ({len(entries)} files: frames, labels, weights; {time.time() - t0:.0f} s)")

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
    """Copy runs/ of the newest attached earlier session (Kaggle mounts notebook outputs under /kaggle/input).
    On Colab runs/ lives on Google Drive and already holds the earlier sessions."""
    if (RUNS / "sessions.csv").exists():
        log(f"restore: runs/ already holds earlier sessions ({PLATFORM}, persistent storage); nothing to copy")
        return
    found = [Path(p) for p in glob.glob(str(INPUT / "**" / "bisindo" / "runs"), recursive=True)]
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
            if any(s in line for s in FORBIDDEN_LINES):
                proc.kill()
                proc.wait()
                f.write(f"\n===== stopped by kaggle_session.py: forbidden line {line.strip()!r}\n")
                log(f"STOPPED: forbidden line in the run output: {line.strip()}")
                return FORBIDDEN_EXIT
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
    import yaml   # installed / pinned by stage_env
    cfg = yaml.safe_load(open(WORK / "configs" / "train.yaml", encoding="utf-8"))
    locked = set(cfg["batch"].values())
    if None not in locked and len(locked) == 1:   # batch locked after the pilot: no new probe
        if gpu_name(env) != cfg["kaggle_gpu"]:
            sys.exit(f"this session got {gpu_name(env)}, but the batch is locked for {cfg['kaggle_gpu']}; "
                     "choose the same accelerator as before")
        log(f"batch {locked.pop()} locked in configs/train.yaml for {cfg['kaggle_gpu']} (no probe)")
        return cfg["batch"]["yolo11m"]
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
            if code == FORBIDDEN_EXIT:
                sys.exit("probe stopped: forbidden line in the output (see the log above)")
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
        if code == FORBIDDEN_EXIT:
            log(f"{name}: stopped, forbidden line in the output; the remaining runs are not started")
            break
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
    report = {"plan": plan_name, "platform": PLATFORM, "gpu": env.get("nvidia_smi"), "batch": batch, "session_hours": gpu_h,
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
    ap.add_argument("--max-hours", "--budget-hours", dest="max_hours", type=float, default=DEFAULT_MAX_HOURS,
                    help="session length limit from the start of this script; no epoch is started that would end "
                         "after it (Kaggle stops a session at 12 h; lower it when less weekly GPU quota is left)")
    args = ap.parse_args()
    if PLATFORM not in ("kaggle", "colab"):
        sys.exit(f"BISINDO_PLATFORM must be kaggle or colab, got {PLATFORM!r}")
    cap = MAX_HOURS_CAP_COLAB if PLATFORM == "colab" else MAX_HOURS_CAP
    if not 0 < args.max_hours <= cap:
        sys.exit(f"--max-hours must be in (0, {cap}] on {PLATFORM}, got {args.max_hours}")
    deadline = T_START + args.max_hours * 3600
    log(f"session limit {args.max_hours:.2f} h: training stops by "
        f"{datetime.fromtimestamp(deadline, timezone.utc).strftime('%H:%M')} UTC, then report + output save")
    WORK.mkdir(parents=True, exist_ok=True)

    env = stage_env()
    stage_setup()
    stage_restore()
    (RUNS / f"env_{PLATFORM}.json").write_text(json.dumps(env, indent=1))   # this session's environment
    batch = stage_probe(deadline, env)
    plan, touched = stage_plan(args.plan, batch, deadline)
    stage_report(args.plan, plan, batch, touched, env)
    # data/ is rebuilt from the dataset every session (symlinks, weight copies, COCO/YOLO views): keep it out
    # of the saved notebook output, which then holds only runs/ (checkpoints, logs, reports).
    shutil.rmtree(WORK / "data", ignore_errors=True)


if __name__ == "__main__":
    main()
