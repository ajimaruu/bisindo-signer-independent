"""Bring the output of a Kaggle session to the laptop.

Everything (checkpoints included) goes to runs/kaggle/<timestamp>/ (untracked); the small result files
(*.csv, eval_test.json, args.json, session_report.json, env_kaggle.json, kaggle_batch.json) are also copied to
results/kaggle/ (the CSVs there are tracked by git). Existing files in results/kaggle/ are replaced by the
newer session's copies; checkpoints are never deleted.

Usage:
  python scripts/kaggle_download.py --kernel <user>/<notebook-slug>   # Kaggle CLI (.venv-kaggle), latest version
  python scripts/kaggle_download.py --kernel <user>/<notebook-slug>/4 # a given notebook version (e.g. while a
                                                                      # newer version is still running)
  python scripts/kaggle_download.py --zip path/to/output.zip         # 'Download' from the notebook's Output tab
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv-kaggle" / "Scripts" / "kaggle.exe"
SMALL = {"eval_test.json", "args.json", "session_report.json", "env_kaggle.json", "kaggle_batch.json"}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kernel", help="<user>/<notebook-slug>")
    src.add_argument("--zip", type=Path)
    args = ap.parse_args()

    dest = ROOT / "runs" / "kaggle" / datetime.now().strftime("%Y%m%d-%H%M")
    dest.mkdir(parents=True, exist_ok=False)
    if args.kernel:
        r = subprocess.run([str(KAGGLE), "kernels", "output", args.kernel, "-p", str(dest)])
        if r.returncode:
            sys.exit("kaggle CLI failed (token in %USERPROFILE%\\.kaggle\\kaggle.json? notebook slug right?)")
        if not any(dest.iterdir()):
            dest.rmdir()
            sys.exit("nothing downloaded: that version has no output yet (still running?). "
                     "Pass the finished version as <user>/<notebook-slug>/<version>.")
    else:
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(dest)

    runs = next((p for p in dest.rglob("runs") if (p / "sessions.csv").exists()), None)
    if runs is None:
        sys.exit(f"no runs/sessions.csv in {dest}; is this the output of the session notebook?")
    out = ROOT / "results" / "kaggle"
    n = 0
    for p in runs.rglob("*"):
        if p.is_file() and (p.suffix == ".csv" or p.name in SMALL):
            target = out / p.relative_to(runs)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            n += 1
    ckpts = sorted(runs.rglob("*.pt"))
    print(f"downloaded to {dest.relative_to(ROOT)}")
    print(f"copied {n} result files to {out.relative_to(ROOT)}")
    print(f"checkpoints: {len(ckpts)} ({sum(p.stat().st_size for p in ckpts) / 2**30:.2f} GiB), e.g.")
    for p in [c for c in ckpts if c.name == "best.pt"][:10]:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
