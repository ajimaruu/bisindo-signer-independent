"""Train Faster R-CNN ResNet-50 FPN v1 (torchvision) with the locked configuration in configs/train.yaml.

Data pipeline = Ultralytics' own (build_yolo_dataset / build_dataloader with the same augmentation values),
so the training transforms are literally the same classes as for YOLO11 and RT-DETR:
RandomPerspective (translate, scale, letterbox to 640) -> RandomHSV; evaluation uses LetterBox with
rectangular batches, as Ultralytics' validator. Batches are only re-packed into torchvision targets
(xyxy pixel boxes, label = class id + 1).

  - lr = lr_base * batch / lr_ref_batch, SGD; linear warmup and linear decay as Ultralytics
    (warmup_epochs, lrf in configs/train.yaml); AMP; gradient clipping at norm 10 (as Ultralytics).
  - Session deadline (env BISINDO_DEADLINE, unix time): after an epoch's checkpoint is saved, exit with
    code 3 if the next epoch would not finish in time; the next session resumes from last.pt.
  - Each epoch: validation mAP@[.50:.95] with pycocotools (scripts/coco_eval.py) -> best.pt, early stopping.
  - Resumable: last.pt holds model, optimizer, scaler, epoch, best score and RNG states.
  - After training: best.pt scored on the test split (pycocotools) -> <run>/eval_test.json.
  - --probe: one epoch on --fraction of the training set, no validation, to find the largest batch.

Usage:
  python scripts/train_frcnn.py --split p1 --seed 0 --epochs 10 --batch 4
  python scripts/train_frcnn.py --split p1 --probe --batch 4
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_guards  # noqa: E402,I001  (before torch: disables torch.compile)

import numpy as np
import torch
import yaml
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.torch_utils import init_seeds

from coco_eval import CONF, MAX_DET, NMS_IOU, evaluate, format_result, load_gt, save_dets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "train.yaml", encoding="utf-8"))
M = CFG["models"]["frcnn-r50fpn"]
WARMUP_EPOCHS = M["warmup_epochs"]   # linear warmup from 0, at least 100 iterations
LRF = M["lrf"]                       # final lr = lr * LRF, linear decay
GRAD_CLIP = 10.0                     # Ultralytics clips the gradient norm at 10
DEADLINE_EXIT = 3                    # exit code: stopped before the session deadline, resumable


def build_model(nc):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                    min_size=M["min_size"], max_size=M["max_size"],
                                    box_score_thresh=CONF, box_nms_thresh=NMS_IOU, box_detections_per_img=MAX_DET)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc + 1)   # + background
    return model


def loaders(split, batch, probe, fraction):
    cfg = get_cfg(overrides={**CFG["augment"], "imgsz": CFG["imgsz"], "fraction": fraction if probe else 1.0})
    data = check_det_dataset(str(ROOT / "data" / "yolo" / f"{split}.yaml"))
    train_ds = build_yolo_dataset(cfg, data["train"], batch, data, mode="train")
    out = {"train": build_dataloader(train_ds, batch, CFG["workers"], shuffle=True)}
    for part in ("val", "test"):
        ds = build_yolo_dataset(cfg, data[part], batch * 2, data, mode="val", rect=True)
        out[part] = build_dataloader(ds, batch * 2, CFG["workers"], shuffle=False)
    return out, data


def to_torchvision(batch, device):
    imgs = batch["img"].to(device, non_blocking=True).float() / 255
    h, w = imgs.shape[2:]
    targets = []
    for i in range(imgs.shape[0]):
        m = batch["batch_idx"] == i
        xc, yc, bw, bh = batch["bboxes"][m].unbind(1)
        boxes = torch.stack([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], 1)
        targets.append({"boxes": boxes.to(device), "labels": batch["cls"][m].view(-1).long().to(device) + 1})
    return list(imgs), targets


@torch.no_grad()
def predict(model, loader, split, device):
    model.eval()
    ids = {im["file_name"]: im["id"] for im in load_gt(split).dataset["images"]}
    dets = []
    for batch in loader:
        imgs = list(batch["img"].to(device).float() / 255)
        with torch.autocast("cuda", dtype=torch.float16, enabled=CFG["amp"]):
            out = model(imgs)
        shape = batch["img"].shape[2:]
        for i, o in enumerate(out):
            boxes = scale_boxes(shape, o["boxes"].float().clone(), batch["ori_shape"][i],
                                ratio_pad=batch["ratio_pad"][i])
            for (x1, y1, x2, y2), c, s in zip(boxes.tolist(), o["labels"].tolist(), o["scores"].tolist()):
                dets.append({"image_id": ids[Path(batch["im_file"][i]).name], "category_id": int(c),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return dets


def lr_at(it, epoch, nb, epochs, lr):
    """Ultralytics schedule: linear decay over epochs; linear warmup from 0 over the first iterations."""
    lf = (1 - epoch / epochs) * (1.0 - LRF) + LRF
    nw = max(round(WARMUP_EPOCHS * nb), 100)
    target = lr * lf
    return float(np.interp(it, [0, nw], [0.0, target])) if it <= nw else target


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="p1 or loso_s1..loso_s4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fraction", type=float, default=0.2)
    args = ap.parse_args()

    batch = args.batch or CFG["batch"]["frcnn-r50fpn"]
    epochs = 1 if args.probe else args.epochs or CFG["early_stopping"]["max_epochs"]
    patience = args.patience or CFG["early_stopping"]["patience"] or epochs
    lr = M["lr_base"] * batch / M["lr_ref_batch"]
    run_dir = (ROOT / "runs" / "probe" / f"frcnn_b{batch}" if args.probe else
               ROOT / "runs" / "frcnn" / (args.name or f"frcnn-r50fpn_{args.split}_seed{args.seed}"))
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print(f"guards: {run_guards.apply()}", flush=True)
    init_seeds(args.seed, deterministic=True)          # same seeding as Ultralytics
    dl, data = loaders(args.split, batch, args.probe, args.fraction)
    print(f"train transforms: {run_guards.check_transforms(dl['train'].dataset)}", flush=True)
    model = build_model(len(data["names"])).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=M["momentum"], weight_decay=M["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=CFG["amp"])
    start, best, best_epoch = 0, -1.0, -1

    last = run_dir / "weights" / "last.pt"
    if last.exists() and not args.probe:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("finished"):
            print(f"{run_dir.name}: training already finished")
            epochs = start = ck["epoch"]
        else:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            set_rng_state(ck["rng"])
        start, best, best_epoch = ck["epoch"], ck["best"], ck["best_epoch"]
        print(f"resuming {last} after epoch {start} (best mAP50-95 {best:.4f} at epoch {best_epoch})")
    (run_dir / "args.json").write_text(json.dumps({**vars(args), "batch": batch, "epochs": epochs,
                                                   "patience": patience, "lr": lr}, indent=1))

    nb = len(dl["train"])
    log = run_dir / "results.csv"
    for epoch in range(start, epochs):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        t0, losses = time.time(), []
        for i, b in enumerate(dl["train"]):
            it = i + nb * epoch
            for g in opt.param_groups:
                g["lr"] = lr_at(it, epoch, nb, epochs, lr)
            imgs, targets = to_torchvision(b, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=CFG["amp"]):
                loss = sum(model(imgs, targets).values())
            if not math.isfinite(loss.item()):
                sys.exit(f"non-finite loss at epoch {epoch + 1}, iteration {i}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if i % 20 == 0:
                print(f"epoch {epoch + 1}/{epochs} it {i}/{nb} loss {np.mean(losses):.4f} "
                      f"lr {opt.param_groups[0]['lr']:.5f} mem {torch.cuda.max_memory_reserved() / 2**30:.2f}G",
                      flush=True)
        train_s, mem = time.time() - t0, torch.cuda.max_memory_reserved() / 2**30

        val = {"map50": float("nan"), "map50_95": float("nan")}
        if not args.probe:
            val = evaluate(f"{args.split}_val", predict(model, dl["val"], f"{args.split}_val", device))
            if val["map50_95"] > best:
                best, best_epoch = val["map50_95"], epoch + 1
                torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val": val}, run_dir / "weights" / "best.pt")
        stop = (epoch + 1 - best_epoch) >= patience and not args.probe
        done = stop or epoch + 1 == epochs
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "best": best, "best_epoch": best_epoch, "rng": rng_state(),
                    "finished": done}, last)
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "train_loss", "lr", "val_map50", "val_map50_95", "train_s", "train_val_s",
                            "peak_reserved_gib", "batch", "n_train"])
            w.writerow([epoch + 1, round(float(np.mean(losses)), 5), round(opt.param_groups[0]["lr"], 6),
                        round(val["map50"], 5), round(val["map50_95"], 5), round(train_s, 1),
                        round(time.time() - t0, 1), round(mem, 3), batch, len(dl["train"].dataset)])
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.4f}  val mAP50 {val['map50']:.4f}  "
              f"mAP50-95 {val['map50_95']:.4f}  train {train_s:.0f}s  peak {mem:.2f} GiB", flush=True)
        if stop:
            print(f"early stopping: no improvement for {patience} epochs (best epoch {best_epoch})")
            break
        epoch_s = time.time() - t0
        if not done and time.time() + 1.25 * epoch_s > float(os.environ.get("BISINDO_DEADLINE", "inf")):
            print(f"session deadline: stopping after epoch {epoch + 1}, resumable from last.pt", flush=True)
            os._exit(DEADLINE_EXIT)
    if args.probe:
        return

    model.load_state_dict(torch.load(run_dir / "weights" / "best.pt", map_location="cpu", weights_only=False)["model"])
    dets = predict(model, dl["test"], f"{args.split}_test", device)
    save_dets(dets, run_dir / "preds_test_coco.json")
    res = evaluate(f"{args.split}_test", dets)
    (run_dir / "eval_test.json").write_text(json.dumps(
        {"model": "frcnn-r50fpn", "split": args.split, "seed": args.seed, "batch": batch, "best_epoch": best_epoch,
         "epochs_max": epochs, "patience": patience, "lr": lr,
         "selection": "best validation mAP50-95 (pycocotools)",
         "eval": {"conf": CONF, "nms_iou": NMS_IOU, "max_det_coco": MAX_DET}, "pycocotools": res}, indent=1))
    print(f"test pycocotools: {format_result(res)}")


if __name__ == "__main__":
    main()
