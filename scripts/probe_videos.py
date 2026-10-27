"""Read-only ffprobe of the raw videos for the 15 selected glosses.

Writes one row per video to results/video_probe.csv and prints a summary
(resolution, fps, duration min/mean/max). Never modifies data/raw/.

Usage: python scripts/probe_videos.py --ffprobe PATH_TO_FFPROBE
"""
import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "video_probe.csv"
WINGET_BIN = Path.home() / ("AppData/Local/Microsoft/WinGet/Packages/"
                            "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin")
NAME_RE = re.compile(r"signer(\d+)_label(\d+)_sample(\d+)\.mp4$")


def probe(ffprobe, path):
    # n_frames = decoded frames; n_packets can be larger when the MP4 has an edit list (trimmed clip).
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-count_frames",
           "-show_entries",
           "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,nb_read_frames,codec_name,color_transfer",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    st = info["streams"][0]
    num, den = map(int, st["avg_frame_rate"].split("/"))
    return {
        "width": st["width"], "height": st["height"], "codec": st["codec_name"],
        "color_transfer": st.get("color_transfer", ""),
        "r_frame_rate": st["r_frame_rate"], "fps": round(num / den, 3) if den else 0.0,
        "n_frames": int(st["nb_read_frames"]), "n_packets": int(st["nb_read_packets"]),
        "duration_s": float(info["format"]["duration"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffprobe")
    args = ap.parse_args()
    ffprobe = args.ffprobe or shutil.which("ffprobe") or str(WINGET_BIN / "ffprobe.exe")

    cfg = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
    orig_to_new = {int(k): int(v) for k, v in cfg["orig_to_new"].items()}

    rows = []
    for p in sorted(RAW.glob("*.mp4")):
        m = NAME_RE.match(p.name)
        if not m or int(m.group(2)) not in orig_to_new:
            continue
        signer, label, sample = map(int, m.groups())
        new_id = orig_to_new[label]
        rows.append({"file": p.name, "signer": signer, "orig_label": label, "new_id": new_id,
                     "gloss": cfg["names"][new_id], "sample": sample, **probe(ffprobe, p)})

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    durs = [r["duration_s"] for r in rows]
    frames = [r["n_frames"] for r in rows]
    print(f"videos probed: {len(rows)} -> {OUT.relative_to(ROOT)}")
    print("resolution:", dict(Counter(f"{r['width']}x{r['height']}" for r in rows)))
    print("fps:", dict(Counter(r["fps"] for r in rows)))
    print("codec:", dict(Counter(r["codec"] for r in rows)))
    print(f"duration s: min {min(durs):.2f}  mean {statistics.mean(durs):.2f}  "
          f"max {max(durs):.2f}  sd {statistics.stdev(durs):.2f}")
    print(f"frames:     min {min(frames)}  mean {statistics.mean(frames):.1f}  max {max(frames)}  "
          f"total {sum(frames)}")
    print("shortest:", min(rows, key=lambda r: r["duration_s"])["file"],
          "| longest:", max(rows, key=lambda r: r["duration_s"])["file"])
    print("\nper gloss (n videos, mean dur s, min-max frames):")
    for nid in sorted(set(r["new_id"] for r in rows)):
        g = [r for r in rows if r["new_id"] == nid]
        print(f"  {nid:2d} {cfg['names'][nid]:<13} n={len(g):3d} mean={statistics.mean(r['duration_s'] for r in g):.2f}"
              f"  frames {min(r['n_frames'] for r in g)}-{max(r['n_frames'] for r in g)}")
    print("\nper signer (n videos, mean dur s):")
    for s in sorted(set(r["signer"] for r in rows)):
        g = [r for r in rows if r["signer"] == s]
        print(f"  signer{s}: n={len(g):3d} mean={statistics.mean(r['duration_s'] for r in g):.2f}")


if __name__ == "__main__":
    main()
