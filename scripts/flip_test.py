"""Supplementary analysis (not part of the 30-run protocol): how the trained detectors respond to mirrored input.

Question (1 Nov 2026): a webcam that shows a mirror image, or a left-handed signer, presents the signs flipped left to
right. Training never flipped frames (fliplr=0), so the models are not expected to be invariant to it.

Procedure: the Protocol 1 test frames are flipped horizontally and written with flipped ground truth
(x -> W - x - w) to runs/flip_test/ (the original frames and annotations are untouched). The P1 best.pt of both
seeds of each model is evaluated on them exactly as for the paper (laptop export, multi-label, conf 0.001,
NMS 0.7, max 100, FP16, pycocotools); P/R/F1 at IoU 0.5 use the run's own conf_thr from runs.csv.
The original-orientation numbers are recomputed from the paper's laptop predictions (runs/laptop_preds/).

Outputs: results/flip_test.csv (one row per run and orientation), results/flip_test_per_class.csv
Usage:   python scripts/flip_test.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_guards  # noqa: E402,I001
import coco_eval  # noqa: E402
import train_frcnn  # noqa: E402
import collect_results as cr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "flip_test"
CLASSES = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
GROUP = {CLASSES["names"][i]: g for i, g in CLASSES["groups"].items()}
SPLIT = "p1_test"
MODELS = ("yolo11m", "rtdetr-l", "frcnn-r50fpn")


def build_flipped():
    gt = json.loads((ROOT / "data" / "coco" / f"{SPLIT}.json").read_text(encoding="utf-8"))
    (OUT / "coco").mkdir(parents=True, exist_ok=True)
    img_dir, lab_dir = OUT / "yolo" / "images", OUT / "yolo" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    size = {}
    for im in gt["images"]:
        src = cv2.imread(str(ROOT / "data" / "frames" / im["file_name"]))
        assert src.shape[1] == im["width"] and src.shape[0] == im["height"], im["file_name"]
        cv2.imwrite(str(img_dir / im["file_name"]), cv2.flip(src, 1), [cv2.IMWRITE_JPEG_QUALITY, 100])
        size[im["id"]] = (im["file_name"], im["width"], im["height"])
    labels = {}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        name, W, H = size[a["image_id"]]
        a["bbox"] = [W - x - w, y, w, h]
        labels.setdefault(name, []).append(
            f"{a['category_id'] - 1} {(W - x - w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    for name, _, _ in size.values():
        (lab_dir / (Path(name).stem + ".txt")).write_text("\n".join(labels.get(name, [])) + "\n", encoding="utf-8")
    (OUT / "coco" / f"{SPLIT}.json").write_text(json.dumps(gt), encoding="utf-8")
    (OUT / "yolo" / "test.txt").write_text("".join(f"./images/{n}\n" for n, _, _ in sorted(size.values())),
                                           encoding="utf-8")
    cfg = {"path": str(OUT / "yolo"), "train": "test.txt", "val": "test.txt", "test": "test.txt",
           "names": CLASSES["names"]}
    (OUT / "yolo" / "flip.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")


def predict_flipped(model, best):
    if model == "frcnn-r50fpn":
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset
        cfg = get_cfg(overrides={**train_frcnn.CFG["augment"], "imgsz": train_frcnn.CFG["imgsz"]})
        data = check_det_dataset(str(OUT / "yolo" / "flip.yaml"))
        b = cr.FRCNN_PRED_BATCH   # same images per batch as the paper's laptop export
        ds = build_yolo_dataset(cfg, data["test"], b, data, mode="val", rect=True)
        dl = build_dataloader(ds, b, train_frcnn.CFG["workers"], shuffle=False)
        net = train_frcnn.build_model(len(CLASSES["names"]))
        net.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["model"])
        dets = train_frcnn.predict(net.to("cuda"), dl, SPLIT, torch.device("cuda"))
        del net
        torch.cuda.empty_cache()
        return dets
    return coco_eval.predict_ultralytics(best, SPLIT, batch=8, multi_label=True)


def scores(dets, thr):
    res = coco_eval.evaluate(SPLIT, dets)
    p, r, f = coco_eval.prf(coco_eval.match_iou50(SPLIT, dets), [thr])
    return res, float(p[0]), float(r[0]), float(f[0])


def main():
    runs = [r for r in csv.DictReader(open(ROOT / "results" / "runs.csv", encoding="utf-8")) if r["protocol"] == "P1"]
    orig_dirs = (coco_eval.COCO_DIR, coco_eval.FRAMES)
    if not (OUT / "coco" / f"{SPLIT}.json").exists():
        build_flipped()
    rows, per_class = [], []
    for model in MODELS:
        for r in sorted((r for r in runs if r["model"] == model), key=lambda r: r["seed"]):
            name, thr, best = f"{model}_p1_seed{r['seed']}", float(r["conf_thr"]), ROOT / r["checkpoint_path"]
            coco_eval.COCO_DIR, coco_eval.FRAMES = orig_dirs
            orig = json.loads((cr.LAPTOP_PREDS / name / "preds_test_coco.json").read_text(encoding="utf-8"))
            out = {"original": scores(orig, thr)}
            assert abs(out["original"][0]["map50_95"] - float(r["map50_95"])) < 1e-4, f"{name}: paper number differs"
            coco_eval.COCO_DIR, coco_eval.FRAMES = OUT / "coco", OUT / "yolo" / "images"
            path = OUT / "preds" / f"{name}.json"
            if path.exists():
                flipped = json.loads(path.read_text(encoding="utf-8"))
            else:
                flipped = predict_flipped(model, best)
                path.parent.mkdir(parents=True, exist_ok=True)
                coco_eval.save_dets(flipped, path)
            out["mirrored"] = scores(flipped, thr)
            for orient, (res, p, rc, f) in out.items():
                rows.append({"model": model, "seed": r["seed"], "orientation": orient, "map50": res["map50"],
                             "map50_95": res["map50_95"], "precision": p, "recall": rc, "f1": f, "conf_thr": thr})
                for c, (a50, a5095) in res["per_class"].items():
                    per_class.append({"model": model, "seed": r["seed"], "orientation": orient, "class": c,
                                      "ap50": a50, "ap50_95": a5095, "group": GROUP[c]})
                print(f"{name:<22} {orient:<9} mAP50 {res['map50']:.4f}  mAP50-95 {res['map50_95']:.4f}  "
                      f"P {p:.3f} R {rc:.3f} F1 {f:.3f} (thr {thr:.3f})")
    coco_eval.COCO_DIR, coco_eval.FRAMES = orig_dirs
    for path, data in ((ROOT / "results" / "flip_test.csv", rows), (ROOT / "results" / "flip_test_per_class.csv", per_class)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            for x in data:
                w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in x.items()})
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
