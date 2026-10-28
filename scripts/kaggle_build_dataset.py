"""Build the private Kaggle dataset (laptop side). Read-only on data/; writes kaggle_upload/ (untracked).

kaggle_upload/bisindo-wl15/
  frames/   the 2,108 kept frames (union of splits/*.txt)        labels/   their YOLO labels + classes.txt
  splits/  configs/  scripts/   requirements_freeze.txt
  weights/  yolo11m.pt, rtdetr-l.pt, fasterrcnn_resnet50_fpn_coco-258fb6c6.pth (the exact files used on the laptop)
  manifest_sha256.txt   sha256 of every frame, label and weight file (checked on Kaggle)
  coco_sha256.txt       sha256 of data/coco/*.json (Kaggle regenerates them and must get identical files)
  dataset-metadata.json for the Kaggle CLI
kaggle_upload/bisindo-wl15.zip  the same folder zipped, for the web uploader.

Usage: python scripts/kaggle_build_dataset.py [--owner <kaggle-username>]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_upload" / "bisindo-wl15"
SLUG = "bisindo-wl15"
TORCH_HUB = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
FRCNN = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="YOUR_KAGGLE_USERNAME",
                    help="Kaggle username (only used in dataset-metadata.json for the Kaggle CLI)")
    args = ap.parse_args()

    # the COCO files must be current: regenerate them from the labels first
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py")], check=True, capture_output=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    frames = sorted(set().union(*(p.read_text(encoding="utf-8").split() for p in (ROOT / "splits").glob("*.txt"))))
    (OUT / "frames").mkdir(parents=True)
    (OUT / "labels").mkdir()
    for f in frames:
        shutil.copy2(ROOT / "data" / "frames" / f, OUT / "frames" / f)
        shutil.copy2(ROOT / "data" / "labels_yolo" / f"{Path(f).stem}.txt", OUT / "labels" / f"{Path(f).stem}.txt")
    shutil.copy2(ROOT / "data" / "labels_yolo" / "classes.txt", OUT / "labels" / "classes.txt")
    shutil.copytree(ROOT / "splits", OUT / "splits")
    shutil.copytree(ROOT / "configs", OUT / "configs")
    (OUT / "scripts").mkdir()
    for p in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(p, OUT / "scripts" / p.name)
    shutil.copy2(ROOT / "results" / "requirements_freeze.txt", OUT / "requirements_freeze.txt")
    (OUT / "weights").mkdir()
    for src in (ROOT / "data" / "models" / "yolo11m.pt", ROOT / "data" / "models" / "rtdetr-l.pt", TORCH_HUB / FRCNN):
        shutil.copy2(src, OUT / "weights" / src.name)

    manifest = [f"{sha256(p)}  {p.relative_to(OUT).as_posix()}"
                for d in ("frames", "labels", "weights") for p in sorted((OUT / d).iterdir())]
    (OUT / "manifest_sha256.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    coco = [f"{sha256(p)}  data/coco/{p.name}" for p in sorted((ROOT / "data" / "coco").glob("*.json"))]
    (OUT / "coco_sha256.txt").write_text("\n".join(coco) + "\n", encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "BISINDO WL15 frames", "id": f"{args.owner}/{SLUG}", "licenses": [{"name": "other"}],
        "description": "Private. 2,108 annotated frames of 15 glosses from WL-BISINDO (Kaggle glennleonali/wl-bisindo, "
                       "CC BY-NC 4.0), labels, splits, configs, scripts and COCO-pretrained weights for the "
                       "signer-independent detector comparison."}, indent=1), encoding="utf-8")

    zip_path = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 2**20
    print(f"frames {len(frames)}  labels {len(frames)}  weights 3  manifest {len(manifest)} files  "
          f"COCO {len(coco)} files")
    print(f"folder: {OUT.relative_to(ROOT)} ({size:.0f} MB)")
    print(f"zip:    {Path(zip_path).relative_to(ROOT)} ({Path(zip_path).stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
