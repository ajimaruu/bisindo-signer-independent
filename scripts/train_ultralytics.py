"""Train YOLO11m or RT-DETR-L (Ultralytics) with the locked configuration in configs/train.yaml.

  - Resumable: if runs/ultralytics/<name>/weights/last.pt exists and training has not finished, it resumes.
  - Logs peak GPU memory (reserved) and wall time per epoch to <run>/gpu_log.csv.
  - After training: best.pt is scored on the test split with the common pycocotools evaluator
    (scripts/coco_eval.py) and, for comparison, with Ultralytics' own model.val(). Both go to <run>/eval_test.json.
  - --probe: one short epoch on a fraction of the training set, no checkpoint selection, only to find the
    largest batch that fits (runs/probe/).

Usage:
  python scripts/train_ultralytics.py --model yolo11m --split p1 --seed 0 --epochs 10 --batch 8
  python scripts/train_ultralytics.py --model yolo11m --split p1 --probe --batch 16
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from ultralytics import RTDETR, YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval import (CONF, MAX_DET, NMS_IOU, evaluate, format_result, predict_ultralytics,  # noqa: E402
                       save_dets)

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "train.yaml", encoding="utf-8"))


DEADLINE_EXIT = 3        # exit code: stopped before the session deadline, resumable


def gpu_logger(run_dir, batch):
    state = {}
    deadline = float(os.environ.get("BISINDO_DEADLINE", "inf"))

    def on_train_start(trainer):
        # Resume across sessions: Ultralytics restores best_fitness but starts a fresh EarlyStopping, which
        # would restart the patience count. Rebuild its state from results.csv (fitness = mAP50-95).
        csv_path = Path(trainer.save_dir) / "results.csv"
        if trainer.start_epoch > 0 and csv_path.exists():
            with open(csv_path, newline="") as f:
                rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
            best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
            trainer.stopper.best_epoch = int(float(best["epoch"]))
            trainer.stopper.best_fitness = float(best["metrics/mAP50-95(B)"])
            print(f"early stopping restored: best epoch {trainer.stopper.best_epoch}, "
                  f"fitness {trainer.stopper.best_fitness:.5f}")

    def on_train_epoch_start(trainer):
        if trainer.batch_size != batch:   # Ultralytics halves the batch silently after an OOM in epoch 1
            raise RuntimeError(f"batch changed from {batch} to {trainer.batch_size} (out of memory); "
                               "the locked batch does not fit")
        torch.cuda.reset_peak_memory_stats()
        state["t0"] = time.time()

    def on_train_epoch_end(trainer):
        state["train_s"] = time.time() - state["t0"]
        state["mem"] = torch.cuda.max_memory_reserved() / 2**30
        state["pending"] = True

    def on_fit_epoch_end(trainer):
        if not state.pop("pending", False):   # the final evaluation of best.pt fires this hook again
            return
        path = Path(trainer.save_dir) / "gpu_log.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "train_s", "train_val_s", "peak_reserved_gib", "batch", "n_train",
                            "fitness"])
            w.writerow([trainer.epoch + 1, round(state["train_s"], 1), round(time.time() - state["t0"], 1),
                        round(state["mem"], 3), trainer.batch_size, len(trainer.train_loader.dataset),
                        round(float(trainer.fitness or 0), 5)])
        # last.pt of this epoch is already saved (Ultralytics saves before on_fit_epoch_end). Stop here if
        # the next epoch would not finish before the session deadline; a later session resumes from last.pt.
        epoch_s = time.time() - state["t0"]
        if not trainer.stop and time.time() + 1.25 * epoch_s > deadline:
            print(f"session deadline: stopping after epoch {trainer.epoch + 1}, resumable from last.pt", flush=True)
            os._exit(DEADLINE_EXIT)

    return {"on_train_start": on_train_start, "on_train_epoch_start": on_train_epoch_start,
            "on_train_epoch_end": on_train_epoch_end, "on_fit_epoch_end": on_fit_epoch_end}


def build(model_key, weights):
    return (RTDETR if model_key.startswith("rtdetr") else YOLO)(str(weights))


def finished(last):
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    return ckpt.get("optimizer") is None      # Ultralytics strips the optimizer when training ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo11m", "rtdetr-l"])
    ap.add_argument("--split", required=True, help="p1 or loso_s1..loso_s4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--probe", action="store_true", help="batch-size probe: 1 epoch on --fraction of train")
    ap.add_argument("--fraction", type=float, default=0.2)
    args = ap.parse_args()

    m = CFG["models"][args.model]
    batch = args.batch or CFG["batch"][args.model]
    epochs = 1 if args.probe else args.epochs or CFG["early_stopping"]["max_epochs"]
    patience = args.patience or CFG["early_stopping"]["patience"] or epochs
    data = ROOT / "data" / "yolo" / f"{args.split}.yaml"
    if args.probe:
        project, name = ROOT / "runs" / "probe", f"{args.model}_b{batch}"
    else:
        project = ROOT / "runs" / "ultralytics"
        name = args.name or f"{args.model}_{args.split}_seed{args.seed}"
    run_dir = project / name
    last = run_dir / "weights" / "last.pt"

    callbacks = gpu_logger(run_dir, batch)
    if last.exists() and not args.probe and not finished(last):
        print(f"resuming {last}")
        model = build(args.model, last)
        for k, fn in callbacks.items():
            model.add_callback(k, fn)
        model.train(resume=True)
    elif not (last.exists() and not args.probe):
        model = build(args.model, ROOT / m["weights"])
        for k, fn in callbacks.items():
            model.add_callback(k, fn)
        opt = {k: m[k] for k in ("optimizer", "lr0", "momentum", "weight_decay", "warmup_bias_lr") if k in m}
        model.train(data=str(data), epochs=epochs, patience=patience, batch=batch, imgsz=CFG["imgsz"],
                    seed=args.seed, deterministic=True, amp=CFG["amp"], workers=CFG["workers"],
                    project=str(project), name=name, exist_ok=True, plots=False, verbose=True,
                    fraction=args.fraction if args.probe else 1.0, **opt, **CFG["augment"])
    if args.probe:
        return

    # ---- test evaluation: common pycocotools evaluator vs Ultralytics' own metric ----
    best = run_dir / "weights" / "best.pt"
    dets = predict_ultralytics(best, f"{args.split}_test", batch=batch)
    save_dets(dets, run_dir / "preds_test_coco.json")
    coco = evaluate(f"{args.split}_test", dets)
    uv = build(args.model, best).val(data=str(data), split="test", imgsz=CFG["imgsz"], batch=batch,
                                     conf=CONF, iou=NMS_IOU, half=True, plots=False, verbose=False,
                                     project=str(run_dir), name="val_test", exist_ok=True)
    names = uv.names
    ultra = {"map50": float(uv.box.map50), "map50_95": float(uv.box.map),
             "per_class": {names[int(c)]: (float(uv.box.ap50[i]), float(uv.box.ap[i]))
                           for i, c in enumerate(uv.box.ap_class_index)}}
    out = {"model": args.model, "split": args.split, "seed": args.seed, "batch": batch, "epochs_max": epochs,
           "eval": {"conf": CONF, "nms_iou": NMS_IOU, "max_det_coco": MAX_DET},
           "pycocotools": coco, "ultralytics": ultra}
    (run_dir / "eval_test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"pycocotools : {format_result(coco)}")
    print(f"ultralytics : {format_result(ultra)}")


if __name__ == "__main__":
    main()
