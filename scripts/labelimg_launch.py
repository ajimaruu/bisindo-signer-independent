"""Open LabelImg on data/frames with the project classes, YOLO format and data/labels_yolo as save dir.

Runs in the separate LabelImg env (.venv-labelimg, Python 3.9): LabelImg 1.8.6 crashes on
Python >= 3.10 (PyQt5 rejects float arguments), so it is not installed in .venv.

Usage: .venv-labelimg\\Scripts\\python.exe scripts\\labelimg_launch.py   (or scripts\\labelimg.bat)
"""
import sys
from pathlib import Path

from labelImg.labelImg import get_main_app
from libs.constants import FORMAT_YOLO

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
LABELS = ROOT / "data" / "labels_yolo"
CLASSES = LABELS / "classes.txt"


def main():
    for p in (FRAMES, LABELS, CLASSES):
        if not p.exists():
            sys.exit(f"missing: {p}")
    app, win = get_main_app([sys.argv[0], str(FRAMES), str(CLASSES), str(LABELS)])
    # Same as clicking the format button until it reads "YOLO"; LabelImg has no CLI flag for this.
    win.set_format(FORMAT_YOLO)
    win.statusBar().showMessage(f"Format YOLO | save dir: {LABELS}")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
