"""Extract frames from the 15 selected glosses following configs/preprocess.yaml.

Rules (CLAUDE.md): 5 frames per video, equally spaced over the middle 60% of the
decoded frames, HLG tone-mapped to SDR with mobius, long side downscaled to 1280.
Reads data/raw/ only; never modifies it.

Usage:
  python scripts/extract_frames.py --preview   # 15 videos -> results/preview_extraction/
  python scripts/extract_frames.py --all       # 750 videos -> data/frames/ (skips finished videos)
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FRAMES = ROOT / "data" / "frames"
PREVIEW = ROOT / "results" / "preview_extraction"
WINGET_BIN = Path.home() / ("AppData/Local/Microsoft/WinGet/Packages/"
                            "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin")
NAME_RE = re.compile(r"signer(\d+)_label(\d+)_sample(\d+)\.mp4$")


def find_tool(name, given):
    if given:
        return given
    return shutil.which(name) or str(WINGET_BIN / f"{name}.exe")


def load_configs():
    classes = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
    prep = yaml.safe_load(open(ROOT / "configs" / "preprocess.yaml", encoding="utf-8"))
    return classes, prep


def list_videos(classes):
    """All raw videos of the selected glosses, as dicts."""
    orig_to_new = {int(k): int(v) for k, v in classes["orig_to_new"].items()}
    videos = []
    for p in sorted(RAW.glob("*.mp4")):
        m = NAME_RE.match(p.name)
        if not m or int(m.group(2)) not in orig_to_new:
            continue
        signer, label, sample = map(int, m.groups())
        new_id = orig_to_new[label]
        gloss = classes["names"][new_id]
        videos.append({"path": p, "signer": signer, "orig_label": label, "new_id": new_id,
                       "gloss": gloss, "slug": gloss.lower().replace(" ", ""), "sample": sample})
    return videos


def probe(ffprobe, path):
    # Count decoded frames, not packets: some clips carry MP4 edit lists, so packets > shown frames.
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=width,height,color_transfer,nb_read_frames", "-of", "json", str(path)]
    st = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)["streams"][0]
    return {"width": st["width"], "height": st["height"], "n_frames": int(st["nb_read_frames"]),
            "hlg": st.get("color_transfer") == "arib-std-b67"}


def frame_indices(n_frames, prep):
    s = prep["sampling"]
    pos = np.linspace(s["start_fraction"], s["end_fraction"], s["frames_per_video"])
    idx = [int(round(p * (n_frames - 1))) for p in pos]
    assert len(set(idx)) == len(idx), f"duplicate frame indices {idx}"
    return pos.tolist(), idx


def frame_name(v, idx):
    return f"s{v['signer']}_{v['slug']}_v{v['sample']:02d}_f{idx:04d}.jpg"


def extract(ffmpeg, v, info, idx, prep, out_dir):
    """Decode once, keep only the selected frame indices, colour-convert, resize, save as JPEG."""
    select = "select='" + "+".join(f"eq(n,{i})" for i in idx) + "'"
    colour = prep["colour"]["hlg_filter"] if info["hlg"] else prep["colour"]["sdr_filter"]
    L = prep["resize"]["long_side"]
    scale = (f"scale='if(gte(iw,ih),min({L},iw),-2)':'if(gte(iw,ih),-2,min({L},ih))':flags=area")
    vf = ",".join([select, colour, scale])
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        cmd = [ffmpeg, "-v", "error", "-y", "-i", str(v["path"]), "-vf", vf, "-fps_mode", "passthrough",
               "-q:v", str(prep["output"]["jpeg_qscale"]), str(Path(tmp) / "%02d.jpg")]
        subprocess.run(cmd, check=True)
        got = sorted(Path(tmp).glob("*.jpg"))
        if len(got) != len(idx):
            raise RuntimeError(f"{v['path'].name}: expected {len(idx)} frames, got {len(got)}")
        outs = []
        for src, i in zip(got, idx):
            dst = out_dir / frame_name(v, i)
            src.replace(dst)
            outs.append(dst)
    return outs


def contact_sheet(v, info, pos, idx, paths, out_path):
    """5 frames side by side, with a caption bar on top and under each frame."""
    tiles = []
    for p, i, fp in zip(pos, idx, paths):
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        img = cv2.resize(img, (384, int(round(384 * h / w))), interpolation=cv2.INTER_AREA)
        bar = np.full((26, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"f{i:04d}  ({p:.0%})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([img, bar]))
        out_wh = (w, h)
    row = np.hstack([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255) for t in tiles])
    colour = "HLG -> SDR (mobius)" if info["hlg"] else "SDR"
    title = (f"{v['new_id']:2d} {v['gloss']}  |  signer{v['signer']} sample{v['sample']}  |  {colour}  |  "
             f"{info['width']}x{info['height']} -> {out_wh[0]}x{out_wh[1]}  |  {info['n_frames']} frames")
    head = np.full((34, row.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([head, row]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def preview_selection(videos, n_classes):
    """One video per gloss, rotating through the signers that have that gloss."""
    chosen = []
    for k in range(n_classes):
        signers = sorted({v["signer"] for v in videos if v["new_id"] == k})
        signer = signers[k % len(signers)]
        sample = k % 10 + 1
        chosen.append(next(v for v in videos
                           if v["new_id"] == k and v["signer"] == signer and v["sample"] == sample))
    return chosen


def write_index(rows, path):
    items = "\n".join(f'<h3>{r["sheet"]}</h3><img src="{r["sheet"]}" style="max-width:100%">' for r in rows)
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>Extraction preview</title>"
                    f"<body style='font-family:sans-serif;background:#eee'>{items}</body>", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--all", action="store_true")
    ap.add_argument("--ffmpeg")
    ap.add_argument("--ffprobe")
    args = ap.parse_args()
    ffmpeg, ffprobe = find_tool("ffmpeg", args.ffmpeg), find_tool("ffprobe", args.ffprobe)

    classes, prep = load_configs()
    videos = list_videos(classes)

    if args.preview:
        out_dir = PREVIEW / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = preview_selection(videos, classes["nc"])
    else:
        out_dir = FRAMES
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = videos

    manifest = []
    for n, v in enumerate(targets, 1):
        info = probe(ffprobe, v["path"])
        pos, idx = frame_indices(info["n_frames"], prep)
        expected = [out_dir / frame_name(v, i) for i in idx]
        if args.all and all(p.exists() for p in expected):
            paths = expected  # resume: this video is already done
        else:
            paths = extract(ffmpeg, v, info, idx, prep, out_dir)
        row = {"video": v["path"].name, "signer": v["signer"], "new_id": v["new_id"], "gloss": v["gloss"],
               "sample": v["sample"], "hlg": int(info["hlg"]), "n_frames": info["n_frames"],
               "frame_indices": " ".join(map(str, idx)), "files": " ".join(p.name for p in paths)}
        if args.preview:
            sheet = f"{v['new_id']:02d}_{v['slug']}_s{v['signer']}_v{v['sample']:02d}.jpg"
            contact_sheet(v, info, pos, idx, paths, PREVIEW / sheet)
            row["sheet"] = sheet
        manifest.append(row)
        print(f"[{n}/{len(targets)}] {v['path'].name}  hlg={int(info['hlg'])}  frames={idx}")

    man_path = (PREVIEW if args.preview else FRAMES) / "manifest.csv"
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    if args.preview:
        write_index(manifest, PREVIEW / "index.html")
    print(f"done: {len(manifest)} videos, manifest -> {man_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
