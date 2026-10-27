"""Open LabelImg on data/frames with the project classes, YOLO format and data/labels_yolo as save dir,
plus review/exclusion bookkeeping.

Runs in the separate LabelImg env (.venv-labelimg, Python 3.9): LabelImg 1.8.6 crashes on
Python >= 3.10 (PyQt5 rejects float arguments), so it is not installed in .venv.

Additions on top of LabelImg:
  - Every save of a frame from LabelImg (Ctrl+S, Space = verify, auto-save) marks its current label as
    reviewed in data/reviewed_labels.csv (with a digest of the label content).
  - Ctrl+1 / Ctrl+2 / Ctrl+3 exclude the frame with reason a / b / c (data/excluded_frames.csv),
    remove its label from data/labels_yolo and go to the next frame. Ctrl+0 undoes the exclusion
    and restores the pre-annotation box from data/labels_auto if there is one.
  - The window title shows the status of the current frame and overall progress.

Usage: .venv-labelimg\\Scripts\\python.exe scripts\\labelimg_launch.py   (or scripts\\labelimg.bat)
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QAction
from labelImg.labelImg import get_main_app
from libs.constants import FORMAT_YOLO

from annot_common import (AUTO, CANDIDATES, EXCLUDED, FRAMES, LABELS, REVIEWED, label_digest,
                          label_lines, read_table, write_table)

CLASSES = LABELS / "classes.txt"
REASONS = {"a": "tangan istirahat / belum naik", "b": "bentuk tangan kabur", "c": "tangan keluar frame"}


class Bookkeeper:
    def __init__(self, win):
        self.win = win
        self.excluded = {k: v["reason"] for k, v in read_table(EXCLUDED).items()}
        self.reviewed = {k: v for k, v in read_table(REVIEWED).items()}
        self.candidates = {k: v.get("hint", "") for k, v in read_table(CANDIDATES).items()}
        self.n_frames = len(list(FRAMES.glob("*.jpg")))

    # ---------- persistence ----------
    def save_lists(self):
        write_table(EXCLUDED, ["file", "reason"],
                    [{"file": f"{s}.jpg", "reason": r} for s, r in sorted(self.excluded.items())])
        write_table(REVIEWED, ["file", "sha1", "reviewed_at"],
                    [{"file": f"{s}.jpg", "sha1": v["sha1"], "reviewed_at": v["reviewed_at"]}
                     for s, v in sorted(self.reviewed.items())])

    def current_stem(self):
        return Path(self.win.file_path).stem if self.win.file_path else None

    # ---------- status ----------
    def status(self, stem):
        label = LABELS / f"{stem}.txt"
        if stem in self.excluded:
            r = self.excluded[stem]
            return f"DIBUANG ({r}: {REASONS.get(r, '?')})"
        if label_lines(label):
            rev = self.reviewed.get(stem)
            if rev and rev["sha1"] == label_digest(label):
                return "DIPERIKSA"
            return "BELUM DIPERIKSA (box otomatis / diubah di luar LabelImg)"
        if stem in self.candidates:
            return "TANPA LABEL - kandidat buang (tangan tidak terdeteksi)"
        return "TANPA LABEL"

    def refresh_title(self):
        stem = self.current_stem()
        if stem is None:
            return
        done = sum(1 for s in self.reviewed if s not in self.excluded) + len(self.excluded)
        self.win.setWindowTitle(f"{stem}.jpg  |  {self.status(stem)}  |  "
                                f"selesai {done}/{self.n_frames}  |  Ctrl+1/2/3 buang a/b/c, Ctrl+0 batal")

    # ---------- hooks ----------
    def after_save(self):
        stem = self.current_stem()
        label = LABELS / f"{stem}.txt"
        if label_lines(label):
            self.reviewed[stem] = {"sha1": label_digest(label),
                                   "reviewed_at": datetime.now().isoformat(timespec="seconds")}
            if self.excluded.pop(stem, None):
                self.win.statusBar().showMessage("Box disimpan: frame dikeluarkan dari daftar buang.")
        else:
            self.reviewed.pop(stem, None)
        self.save_lists()
        self.refresh_title()

    def exclude(self, reason):
        stem = self.current_stem()
        if stem is None:
            return
        self.excluded[stem] = reason
        self.reviewed.pop(stem, None)
        label = LABELS / f"{stem}.txt"
        if label.exists():
            label.unlink()          # the pre-annotation stays in data/labels_auto
        self.save_lists()
        self.win.set_clean()
        self.win.load_file(self.win.file_path)
        self.win.statusBar().showMessage(f"{stem}.jpg dibuang, alasan {reason}: {REASONS[reason]}")
        self.win.open_next_image()

    def unexclude(self):
        stem = self.current_stem()
        if stem is None or stem not in self.excluded:
            return
        del self.excluded[stem]
        auto, label = AUTO / f"{stem}.txt", LABELS / f"{stem}.txt"
        if auto.exists() and not label_lines(label):
            shutil.copyfile(auto, label)
        self.save_lists()
        self.win.set_clean()
        self.win.load_file(self.win.file_path)
        self.win.statusBar().showMessage(f"{stem}.jpg tidak jadi dibuang.")


def install(win):
    bk = Bookkeeper(win)

    orig_save, orig_load = win._save_file, win.load_file

    def _save_file(annotation_file_path):
        orig_save(annotation_file_path)
        if not win.dirty:           # LabelImg marks the window clean only when the save succeeded
            bk.after_save()

    def load_file(file_path=None):
        ok = orig_load(file_path)
        bk.refresh_title()
        return ok

    win._save_file, win.load_file = _save_file, load_file

    menu = win.menuBar().addMenu("&Buang")
    for key, reason in (("Ctrl+1", "a"), ("Ctrl+2", "b"), ("Ctrl+3", "c")):
        act = QAction(f"Buang ({reason}) {REASONS[reason]}", win)
        act.setShortcut(key)
        act.triggered.connect(lambda _=False, r=reason: bk.exclude(r))
        menu.addAction(act)
    act = QAction("Batal buang", win)
    act.setShortcut("Ctrl+0")
    act.triggered.connect(lambda _=False: bk.unexclude())
    menu.addAction(act)
    return bk


def main(argv=None):
    for p in (FRAMES, LABELS, CLASSES):
        if not p.exists():
            sys.exit(f"missing: {p}")
    app, win = get_main_app([sys.argv[0], str(FRAMES), str(CLASSES), str(LABELS)])
    # Same as clicking the format button until it reads "YOLO"; LabelImg has no CLI flag for this.
    win.set_format(FORMAT_YOLO)
    win.bookkeeper = install(win)
    win.bookkeeper.refresh_title()
    win.statusBar().showMessage(f"Format YOLO | save dir: {LABELS}")
    return app, win


if __name__ == "__main__":
    app, _win = main()
    sys.exit(app.exec_())
