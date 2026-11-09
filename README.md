# Signer-Independent Evaluation of Real-Time Detectors for Indonesian Sign Language

Experiment code for the paper *Signer-Independent Evaluation of Real-Time Detectors for Indonesian Sign Language*
(Aji Bayu Seno, Informatics Engineering, Universitas Dian Nuswantoro).

The paper compares **RT-DETR-L**, **YOLO11m** (Ultralytics) and **Faster R-CNN ResNet-50 FPN** (torchvision) on
15 glosses of [WL-BISINDO](https://www.kaggle.com/datasets/glennleonali/wl-bisindo). It uses two evaluation
protocols:

- a random frame-level split (Protocol 1);
- a leave-one-signer-out (LOSO) protocol with 4 folds.

Each configuration was trained with 2 seeds, 30 runs in all. All three models share the same settings: input 640,
identical augmentation with horizontal flip off, and one evaluator (pycocotools).

- **Dataset (frames, labels, splits):** <https://www.kaggle.com/datasets/ajimaru12/bisindo-wl15-detection> (CC BY-NC 4.0)
- **Full experiment log** (every decision, deviation and check, dated): [`results/config_log.md`](results/config_log.md)

## Main results

| Model | mAP50-95, random split | mAP50-95, signer-independent (± SD folds) | Δrel | FPS (RTX 3050 Laptop, FP16) |
|---|---|---|---|---|
| YOLO11m | 0.863 | 0.683 ± 0.091 | 20.8 % | 66.2 |
| RT-DETR-L | 0.886 | 0.729 ± 0.087 | 17.8 % | 30.3 |
| Faster R-CNN R50-FPN | 0.832 | 0.662 ± 0.084 | 20.5 % | 24.2 |

- Each value is the mean over 2 training seeds; the signer-independent value is the mean over 4 LOSO folds.
- Δrel = relative drop from the random split to the signer-independent protocol.
- FPS covers the model prediction stage only, batch 1.

Tables: `results/summary_p1.md`, `results/summary_loso.md`; per-run rows: `results/runs.csv`; per-class AP:
`results/per_class.csv`.

## Repository layout

```
configs/    classes, frame extraction, training, Kaggle plan, desktop app
scripts/    every step: extraction, annotation checks, splits, training, evaluation, summaries, FPS, app
splits/     file lists of Protocol 1 and the 4 LOSO folds (made once, seed 0)
data/       annotation records: labels_yolo/ (checked boxes), labels_auto/ (raw MediaPipe pre-annotation),
            excluded_frames.csv, reviewed_labels.csv
results/    config_log.md and all result CSVs (runs, per class, confusion matrices, FPS, live test)
```

Videos, frames, checkpoints and run folders are not in this repository. Frames and labels are in the Kaggle dataset
above; the videos are in WL-BISINDO.

## Reproducing

1. Python 3.11 with the package versions in `results/requirements_freeze.txt` (PyTorch with CUDA, Ultralytics 8.4.190, pycocotools).
2. Put the dataset's `frames/` in `data/frames/`, or rebuild the frames from WL-BISINDO videos with `scripts/extract_frames.py`. The labels are in `data/labels_yolo/` and the splits in `splits/`.
3. `python scripts/prepare_data.py` builds the YOLO and COCO files.
4. `scripts/train_ultralytics.py` and `scripts/train_frcnn.py` train one run each; the paper's runs used `scripts/kaggle_session.py` on a Tesla T4 (`configs/kaggle_plan.yaml`).
5. `scripts/collect_results.py` re-exports and scores all runs, then `scripts/summarise.py` and `scripts/confusion_matrix.py` build the tables and matrices.
6. `scripts/measure_fps.py` measures speed; `scripts/app_detect.py` runs the webcam application, and `--live-test` runs the per-sign live test.

## License

- **Code:** [AGPL-3.0](LICENSE), the license of Ultralytics, which the code uses.
- **Data:** the frames and annotations are CC BY-NC 4.0, as the source dataset WL-BISINDO.

When using the data, credit both WL-BISINDO (Glenn Leonali) and this work.
