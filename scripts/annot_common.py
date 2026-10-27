"""Shared helpers for the annotation bookkeeping (stdlib only; must run on Python 3.9 and 3.11).

Used by scripts/labelimg_launch.py (LabelImg env) and scripts/check_annotations.py (.venv).
"""
import csv
import hashlib
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
LABELS = ROOT / "data" / "labels_yolo"
AUTO = ROOT / "data" / "labels_auto"
EXCLUDED = ROOT / "data" / "excluded_frames.csv"         # file,reason   (reason a/b/c)
REVIEWED = ROOT / "data" / "reviewed_labels.csv"         # file,sha1,reviewed_at
CANDIDATES = ROOT / "data" / "exclusion_candidates.csv"  # file,hint     (from pre-annotation)


def label_lines(path):
    """Non-empty, whitespace-normalised lines of a YOLO label file ([] if missing)."""
    if not Path(path).exists():
        return []
    text = Path(path).read_text(encoding="utf-8")
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def label_digest(path):
    """sha1 of the normalised label content; independent of CRLF/LF and trailing spaces."""
    return hashlib.sha1("\n".join(label_lines(path)).encode("utf-8")).hexdigest()


def read_table(path, key="file"):
    """CSV -> {stem of key column: row dict}. Missing file -> {}."""
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {Path(r[key].strip()).stem: r for r in csv.DictReader(f) if (r.get(key) or "").strip()}


def write_table(path, header, rows):
    """Write rows (list of dicts) atomically, so a crash never leaves a half-written list."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
