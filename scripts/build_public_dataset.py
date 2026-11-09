"""Build the PUBLIC dataset release (Kaggle, CC BY-NC 4.0). Read-only on data/; writes kaggle_upload/<slug>/
(untracked). Decided by the user on 9 Nov 2026 (config_log section 35): a new public dataset, separate from the
private training dataset bisindo-wl15, without pretrained weights.

kaggle_upload/bisindo-wl15-detection/
  frames/      the 2,108 annotated frames used in the experiments (union of splits/*.txt)
  labels/      their YOLO labels (one box per frame) + classes.txt
  coco/        COCO json per split and partition (data/coco/*.json, as used by pycocotools)
  splits/      Protocol 1 (random frame split) and the 4 LOSO folds (file lists)
  excluded_frames.csv   extracted frames dropped during annotation, with reason a/b/c
  classes.yaml, README.md, LICENSE.txt, manifest_sha256.txt, dataset-metadata.json

The frames and labels are checked against the sha256 manifest of the private training dataset
(kaggle_upload/bisindo-wl15/manifest_sha256.txt), so the release holds exactly the files that were trained on.

Usage: python scripts/build_public_dataset.py [--owner ajimaru12]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "bisindo-wl15-detection"
OUT = ROOT / "kaggle_upload" / SLUG
PRIVATE_MANIFEST = ROOT / "kaggle_upload" / "bisindo-wl15" / "manifest_sha256.txt"
REPO = "https://github.com/ajimaruu/bisindo-signer-independent"

README = """# WL-BISINDO 15-gloss hand detection frames

Annotated frames for **hand-region gesture detection** in Indonesian Sign Language (BISINDO), derived from
[WL-BISINDO](https://www.kaggle.com/datasets/glennleonali/wl-bisindo) (Glenn Leonali, CC BY-NC 4.0).
This is the data of the paper *Signer-Independent Evaluation of Real-Time Detectors for Indonesian Sign Language*
(Aji Bayu Seno, Universitas Dian Nuswantoro). Code, training configuration and all results: {repo}

## Contents

| Folder / file | Description |
|---|---|
| `frames/` | {n} JPEG frames, long side at most 1280 px |
| `labels/` | YOLO labels, one box per frame (`class cx cy w h`, normalised) and `classes.txt` |
| `coco/` | COCO detection json per split and partition (`category_id` = class id + 1) |
| `splits/` | file lists: `p1_{{train,val,test}}.txt` (random frame split 70/15/15) and `loso_s{{1-4}}_{{train,val,test}}.txt` (leave-one-signer-out, test signer 1–4) |
| `excluded_frames.csv` | extracted frames dropped during annotation and why (a: hands at rest, b: motion blur, c: hands mostly out of frame) |
| `classes.yaml` | class ids, original WL-BISINDO label, one-/two-handed group |
| `manifest_sha256.txt` | sha256 of every frame and label |

File names: `s{{signer}}_{{gloss}}_v{{sample:02d}}_f{{frame:04d}}.jpg` (signer 0–4, frame = decoded frame index).

## How the frames were made

- 15 of the 32 WL-BISINDO glosses, 750 videos: {classes}.
- 5 frames per video at 20, 35, 50, 65 and 80 % of the duration; HLG videos tone-mapped to SDR BT.709 (mobius); long side downscaled to 1280.
- One box per frame tightly around the signing hand(s); for two-handed signs one box around both hands; the face is not included.
- Boxes were pre-annotated with MediaPipe and every box was checked and corrected by hand. 3,750 frames were extracted, {n} were kept.
- Two-handed group (C_two): Motor, Bagaimana, Teman, Rumah; the other 11 are one-handed (C_one).

## Splits

- **Protocol 1 (random):** per-frame split stratified by class, seed 0. Frames of the same video can be in different partitions on purpose; the leakage is what the protocol measures.
- **Signer-independent (LOSO):** 4 folds, test signer 1, 2, 3 or 4. Validation = 15 % of the training signers' videos (all frames of a video in the same partition), stratified by class, seed 0. Signer 0 is only used for training and validation, because it has no videos for 6 of the 15 glosses.

## License and attribution

CC BY-NC 4.0, the license of the source dataset. Non-commercial use only; credit both
WL-BISINDO (glennleonali/wl-bisindo) and this release. The frames show the WL-BISINDO signers; use them for sign
language research only.
"""

LICENSE = """This dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0): https://creativecommons.org/licenses/by-nc/4.0/legalcode

It is an adapted work of WL-BISINDO (https://www.kaggle.com/datasets/glennleonali/wl-bisindo), licensed CC BY-NC 4.0.
Changes: 15 of 32 glosses selected; 5 frames extracted per video, tone-mapped and downscaled; frames dropped by
fixed criteria; hand bounding boxes, class labels and splits added.

Attribution: WL-BISINDO (Glenn Leonali) and Aji Bayu Seno, "WL-BISINDO 15-gloss hand detection frames",
Universitas Dian Nuswantoro, 2026. Code: {repo} (AGPL-3.0).
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="ajimaru12")
    args = ap.parse_args()
    import yaml
    classes = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
    names = [classes["names"][i] for i in range(classes["nc"])]

    frames = sorted({ln.strip() for p in (ROOT / "splits").glob("*.txt") for ln in p.read_text().split() if ln.strip()})
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("frames", "labels", "coco", "splits"):
        (OUT / d).mkdir(parents=True)
    for f in frames:
        shutil.copy2(ROOT / "data" / "frames" / f, OUT / "frames" / f)
        shutil.copy2(ROOT / "data" / "labels_yolo" / (Path(f).stem + ".txt"), OUT / "labels" / (Path(f).stem + ".txt"))
    (OUT / "labels" / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    for p in sorted((ROOT / "data" / "coco").glob("*.json")):
        shutil.copy2(p, OUT / "coco" / p.name)
    for p in sorted((ROOT / "splits").glob("*.txt")):
        shutil.copy2(p, OUT / "splits" / p.name)
    shutil.copy2(ROOT / "data" / "excluded_frames.csv", OUT / "excluded_frames.csv")
    shutil.copy2(ROOT / "configs" / "classes.yaml", OUT / "classes.yaml")

    manifest = {f"{d}/{p.name}": sha256(p) for d in ("frames", "labels") for p in sorted((OUT / d).iterdir())}
    private = dict(reversed(ln.split("  ", 1)) for ln in PRIVATE_MANIFEST.read_text(encoding="utf-8").splitlines()
                   if "  " in ln)
    checked = 0
    for path, h in manifest.items():
        if path in private:
            if private[path] != h:
                raise SystemExit(f"{path} differs from the trained-on file")
            checked += 1
    n_frames = sum(1 for k in manifest if k.startswith("frames/"))
    if checked < 2 * n_frames:
        raise SystemExit(f"only {checked} of {2 * n_frames} frame/label files found in the private manifest")
    (OUT / "manifest_sha256.txt").write_text("".join(f"{h}  {k}\n" for k, h in manifest.items()), encoding="utf-8")
    (OUT / "README.md").write_text(README.format(repo=REPO, n=f"{n_frames:,}", classes=", ".join(names)),
                                   encoding="utf-8")
    (OUT / "LICENSE.txt").write_text(LICENSE.format(repo=REPO), encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "WL-BISINDO 15-Gloss Hand Detection Frames",
        "subtitle": "Annotated BISINDO sign frames with random and leave-one-signer-out splits",
        "id": f"{args.owner}/{SLUG}",
        "licenses": [{"name": "CC-BY-NC-4.0"}],
        "keywords": ["computer vision", "object detection", "languages"],
        "description": README.format(repo=REPO, n=f"{n_frames:,}", classes=", ".join(names))}, indent=1),
        encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 2**20
    print(f"frames {n_frames}, labels {n_frames} (+classes.txt), coco {len(list((OUT / 'coco').iterdir()))}, "
          f"splits {len(list((OUT / 'splits').iterdir()))}; {checked} files identical to the trained-on dataset")
    print(f"-> {OUT.relative_to(ROOT)} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
