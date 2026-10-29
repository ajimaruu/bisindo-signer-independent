# Configuration log

Started 27 Oct 2026. One section per decision or check; newest at the bottom.

## 1. Environment (27 Oct 2026)

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language 10.0.26300 |
| CPU / RAM | Intel Core i5-11400H / 16 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4.00 GiB, compute capability 8.6 |
| NVIDIA driver | 621.01 (CUDA UMD 13.5) |
| Python | 3.11.9 (venv at `.venv/`) |
| PyTorch | 2.14.1+cu130 (CUDA build 13.0, cuDNN 9.24.0) |
| torchvision | 0.29.1+cu130 |
| Ultralytics | 8.4.190 |
| pycocotools | 2.0.11 |
| NumPy / OpenCV / PyYAML | 2.4.6 / 5.0.0 / 6.0.3 |
| FFmpeg / ffprobe | 9.0.2 full_build (gyan.dev, via winget) |

- `torch.cuda.is_available()` = **True**. An fp16 autocast matmul on the GPU ran without error, so AMP works.
- Wheel choice: the driver supports CUDA ≤ 13.5, so I used the newest stable wheel on the official cu130 index.
- Python 3.11 was chosen over the system default 3.14 for wider package compatibility.
- Full package list: `results/requirements_freeze.txt`.
- ffprobe path (not on PATH in the current shell until it restarts):
  `C:\Users\ajise\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe`

## 2. Raw dataset layout (`data/raw/`, read-only)

- 1,600 `.mp4` files in one flat folder, no subfolders and no split files.
- File name pattern: `signer{0-4}_label{0-31}_sample{n}.mp4`. Signer IDs are **0-based (0–4)**.
- The README's "32 gloss × 10 samples × 5 signers" does **not** hold per signer:

| Original labels | signer0 | signer1 | signer2 | signer3 | signer4 |
|---|---|---|---|---|---|
| 0–11 | 10 | 10 | 10 | 10 | 10 |
| 12–31 | **0** | 15 | 15 | 10 | 10 |

  Every label still has 50 videos in total (1,600 overall).
- For the 15 selected glosses (750 videos):
  - signer0: 90 (labels 2–10 only)
  - signer1: 180
  - signer2: 180
  - signer3: 150
  - signer4: 150
  - **signer0 has no videos for Bagaimana, Berangkat, Teman, Keluarga, Rumah or Pagi.** This includes 2 of the 4 C_two classes and the occlusion class Teman.
- The label → gloss mapping in the official repo matches `configs/classes.yaml`.

## 3. Official splits (github.com/AceKinnn/WL-BISINDO, `data_structuring/`)

- `SD_split_metadata.json`: signer-dependent, 70/30 per gloss.
- `SI_split_metadata.json`: **one file only**. Train = signers 0, 1, 2, 3; test = **signer4**.
- The README describes 4 SI configurations ("Signer 1" to "Signer 4" in its results table), but `SI_1` to `SI_4` metadata files are not in the repo.
- Inference, not confirmed: the four configurations probably hold out signers 1–4. signer0 lacks labels 12–31, so it cannot be a full test signer.
- This project still runs its own 5-fold LOSO (CLAUDE.md). The signer0 fold needs a decision (see section 5).

## 4. Video properties (ffprobe, 750 videos of the 15 glosses; `results/video_probe.csv`)

- Resolution:
  - 1920×1080: 573 videos
  - 1280×720: 177 videos (signer0: 87; signer2: 90)
- No rotation metadata found, so all videos are landscape as stored.
- Codec:
  - HEVC: 663 videos
  - H.264: 87 videos (signer0)
- Colour:
  - **HDR HLG (BT.2020, 10-bit)**: signer1 (180), signer3 (150), signer4 (150) and signer2 (60)
  - SDR BT.709: signer0 (90) and signer2 (120)
- Frame rate:
  - Nominal rate: 30 fps
  - 477 videos average exactly 30.00 fps
  - The rest are variable frame rate, averaging 24.9–30.0 fps (signer0, signer1 and signer2 only)
- Duration (s), all 750 videos:
  - min 1.40 (`signer3_label5_sample7`)
  - mean 2.20 (sd 0.44)
  - max 5.85 (`signer2_label2_sample10`)
- Frames per video (**decoded** frames; corrected 27 Oct 2026, see section 8):
  - min 42, mean 65.8, max 173
  - total 49,332
  - The first count (mean 71.7, max 178, total 53,764) used packets and overstated the number of frames.
- Mean duration per signer (s):
  - s0 2.27
  - s1 2.56
  - s2 2.39
  - s3 1.80
  - s4 1.89

## 5. Open decisions (to be settled with the user before frame extraction)

- ~~How to handle the signer0 LOSO fold~~ → decided in section 7.
- How many frames to extract per video.
- ~~How to convert HDR HLG to SDR~~ → decided in section 6.
- Whether to keep source resolution or resize at extraction time. Training is at 640×640 either way.

## 6. HDR → SDR decision (27 Oct 2026)

- **Decided by the user:** every HLG video is tone-mapped to SDR BT.709 with ffmpeg at extraction time. All videos go through the same ffmpeg pipeline.
- Candidate filter chain:
  `zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=<OP>:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p`
- Pilot on one frame (t = 1 s) per video, grey-level mean / sd at 640 px:

| Video | raw (no tone-map) | hable | mobius | reinhard |
|---|---|---|---|---|
| s1 Teman v1 (HLG) | 201.6 / 55.9 | 204.0 / 57.4 | 225.1 / 53.5 | 220.7 / 52.7 |
| s2 Teman v1 (HLG) | 159.1 / 50.2 | **103.5 / 31.6** | 153.5 / 46.0 | 141.6 / 40.5 |
| s3 Teman v1 (HLG) | 186.3 / 68.8 | 186.2 / 71.3 | 206.9 / 67.1 | 202.6 / 66.7 |
| s4 Teman v1 (HLG) | 193.8 / 64.7 | 194.8 / 67.0 | 215.1 / 63.4 | 210.8 / 63.0 |
| SDR refs: s0 Terima kasih v1 / s0 Cari v1 / s2 Terima kasih v1 | 166.9 / 167.6 / 155.9 | – | – | – |

- **hable** darkens signer2's indoor HLG clip strongly, to a dim grey image. That would create a signer-specific shift in appearance, which is a risk for LOSO.
- **mobius** stays close to the raw image for signer2, and is about 20 grey levels brighter for s1/s3/s4.
- **Decided by the user: mobius.** Reason: it keeps signers visually consistent and avoids the signer2 darkening seen with hable. Recipe stored in `configs/preprocess.yaml`.

## 7. LOSO folds (27 Oct 2026)

- **Decided by the user: 4 folds, test signer = signer1, signer2, signer3, signer4** (0-based file IDs).
- signer0 is used only in the train/validation pool. It has no videos for 6 of the 15 classes, so a signer0 test fold could not evaluate all classes or Δtwo comparably.
- This agrees with the presumed official SI_1–SI_4 configuration, though that configuration is not confirmed (see section 3).
- Total runs: 3 models × (1 random split + 4 folds) × 2 seeds = **30**. CLAUDE.md updated accordingly.
- Split file names will use the 0-based ID, for example `loso_s1_test.txt` = test on signer1.

## 8. Frame extraction and annotation rules (locked by the user, 27 Oct 2026)

- **Extraction rules:**
  - 5 frames per video, equally spaced over the middle 60% of the video, endpoints included: positions 0.20, 0.35, 0.50, 0.65 and 0.80.
  - Position p maps to decoded frame index `round(p × (n_frames − 1))`.
  - HLG videos are tone-mapped with mobius.
  - Long side downscaled to 1280, downscale only. All outputs are 1280×720.
  - JPEG `-q:v 2`.
- **Expected output:** 750 videos × 5 = 3,750 frames.
- **Implementation:** `scripts/extract_frames.py`, with parameters in `configs/preprocess.yaml`.
- **Annotation rules:**
  - One box per frame, tightly enclosing the signing hand(s), labelled with the gloss.
  - Two-handed gestures get one box around both hands.
  - The face is never inside the box.
- **Packets vs decoded frames:**
  - 276 of the 750 videos (signer0 86, signer1 100, signer2 90) contain more packets than decoded frames.
  - The extra packets have negative timestamps. They are the start of the clip that was trimmed on the phone and is never shown (MP4 edit list).
  - The frame count therefore uses decoded frames (`ffprobe -count_frames`). ffmpeg's `select=eq(n,…)` counts the same frames.
  - This was found when the preview extraction got 4 of 5 frames for `signer2_label4_sample3` (106 packets, 73 frames).
  - `scripts/probe_videos.py` now records both `n_packets` and `n_frames`.
- **Preview** (`results/preview_extraction/`): one video per gloss, rotating through the signers, 75 frames.
  - All 75 frames extracted at 1280×720, and colour looks consistent across signers.
  - **Some sampled frames contain no sign:**
    - At **20%**, the hands are often still at rest or moving up, with motion blur. Seen in Cari, Lagi, Maaf, Makan, Motor, Saya, Bagaimana, Berangkat, Teman, Keluarga, Rumah and Pagi.
    - At **80%**, the hands are sometimes already back at rest. Seen in Cari, Hari, Ingat and Motor.
    - Berangkat (signer3) starts signing only at about 50%, so its 20% and 35% frames are both at rest.
    - By eye, 17 of 75 preview frames (12 at 20%, 1 at 35%, 4 at 80%) show the hands at rest or in transition rather than the sign.
  - Pending a decision with the user on how to treat these frames.

## 9. Exclusion criteria and full extraction (27 Oct 2026)

- **Decided by the user:** keep 5 frames per video and exclude frames without a visible sign during annotation.
- **Exclusion reasons:**
  - (a) hands still at rest, or not yet raised into the signing space
  - (b) handshape blurred by motion
  - (c) hands mostly outside the frame
- **How the criteria are applied:**
  - The same criteria apply to every gloss and signer.
  - Each frame is decided during annotation, before any training. The list is never changed after seeing detection results.
- **Recording exclusions:**
  - Excluded frames are listed in `data/excluded_frames.csv` (`file,reason`). If several reasons apply, record the first in the order a → b → c.
  - Excluded frames stay in `data/frames/`.
- **Coverage check:** `scripts/check_annotations.py` must report zero issues before splits are made.
  - It flags frames that are neither labelled nor excluded, conflicts, multi-box labels, class-id mismatches, bad boxes, orphan labels and malformed exclusion rows.
  - It was tested on a synthetic set with one case of each issue, and all were caught.
- **LabelImg class order:** `data/labels_yolo/classes.txt` is written from `configs/classes.yaml` so that LabelImg class ids match.
- **Full extraction:** `scripts/extract_frames.py --all`, 14:51 → 15:04 (≈13.5 min).
  - 750 videos, 540 of them HLG tone-mapped.
  - **3,750 frames**, all 1280×720, 162 MB.
  - Per signer: s0 450, s1 900, s2 900, s3 750, s4 750.
  - 250 frames per gloss.
  - No errors. Manifest: `data/frames/manifest.csv`. Log: `runs/extract_frames.log`.

## 10. Annotation tool (27 Oct 2026)

- **Version and environment:** LabelImg 1.8.6 (the last release; the project is archived), with PyQt5 5.15.11 / Qt 5.15.2.
  - It is installed in a **separate env**, `.venv-labelimg`, on Python 3.9.25 (via uv), not tracked by git.
  - Reason: LabelImg 1.8.6 crashes on Python ≥ 3.10 because PyQt5 rejects float arguments, and `.venv` is Python 3.11. Keeping it separate also keeps PyQt5 out of the experiment env.
- **Launch:**
  - `scripts/labelimg.bat` runs `scripts/labelimg_launch.py`. Desktop shortcut: "LabelImg BISINDO".
  - It opens `data/frames` with class file `data/labels_yolo/classes.txt` and save dir `data/labels_yolo`.
  - It sets the YOLO format through LabelImg's own `set_format`, because there is no CLI flag for it.
- **Offscreen test:**
  - 3,750 images loaded, 15 classes in project order, format YOLO, save dir correct.
  - One box saved through LabelImg on a scratch copy gave `11 0.468750 0.555556 0.156250 0.277778` for Teman. This is the correct class id and normalised coordinates.
- **Behaviour to know:**
  - LabelImg rewrites `classes.txt` on every save, with the same content and order but CRLF line endings. `check_annotations.py` verifies the order.
  - LabelImg remembers the last opened image in `~/.labelImgSettings.pkl`, so a new session resumes where the last one stopped.

## 11. Pre-annotation and manual verification (27 Oct 2026)

- **All boxes are verified by hand.** The initial boxes come from automatic pre-annotation, and **every box is verified manually** in LabelImg before it counts as annotated. No pre-annotated box enters training unreviewed. `check_annotations.py` fails while any label is still unreviewed.
- **Detector:**
  - MediaPipe 1.1.1, HandLandmarker (Tasks API), `hand_landmarker.task` float16, 7,819,105 bytes.
  - Model SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`, published 26 Apr 2023.
  - Settings: IMAGE mode, `num_hands=2`, detection and presence confidence 0.5 (defaults).
  - It runs on the CPU in a separate env, `.venv-mediapipe` (Python 3.11), so that MediaPipe's dependencies stay out of `.venv`.
- **Box rule:**
  - Each detected hand gets a tight box around its 21 landmarks, expanded by 8% of the hand box's longer side on every side.
  - C_two: the union of the detected hand boxes. A single detected hand is flagged `two_class_one_hand`.
  - C_one: only the hand with the highest landmark centroid.
  - Class id comes from the gloss in the file name.
- **Frames with no detected hand:** these get no label. They are listed in `data/exclusion_candidates.csv` only. The exclusion decision stays manual.
- **Bookkeeping files:**
  - `data/labels_auto/`: raw pre-annotation, never edited.
  - `data/reviewed_labels.csv`: the sha1 of each label at the time it was saved from LabelImg.
  - `data/excluded_frames.csv`: written by LabelImg hotkeys Ctrl+1/2/3 (reason a/b/c), undone with Ctrl+0.
- **Correction statistics:** `check_annotations.py` reports how much the review changed the pre-annotation. It counts unchanged boxes (every edge within 2 px; LabelImg rounds to whole pixels when it re-saves), the IoU distribution, manually drawn boxes and excluded frames that had an automatic box. These numbers go in the paper's annotation description.
- **Pilot (20 frames):** each C_two gloss ×2, each C_one gloss ×1, plus one extra Teman; all five frame positions. Output in `results/preannot_pilot/`.
  - 16 frames got a box, 3 had no hand detected and 1 was flagged `two_class_one_hand`.
  - No hand detected:
    - `s2_motor_v05_f0016` and `s0_saya_v09_f0054`: hands at rest.
    - `s3_lagi_v08_f0017`: handshape blurred.
    - All three would fall under exclusion criteria a/b.
  - `s3_rumah_v02_f0018`: the second hand was missed and the box covers only one hand.
  - `s3_teman_v03_f0010`: both hands are at rest at the waist but were detected and boxed. **The detector does not apply criterion (a)**, so rest frames with visible hands must be excluded manually.
  - `s4_motor_v09_f0023`: the motion-blurred right hand is slightly cut off.
  - Otherwise the boxes are tight and on the correct hand.
- **Workflow test:** an end-to-end test on a scratch copy of the project passed. It covered Space-approve, Ctrl+1 exclude, edit and save, exclude then Ctrl+0, and an external edit reverting a label to unreviewed.

## 12. Margin decision and full pre-annotation run (27 Oct 2026)

- **Decided by the user: keep the 8% margin.**
  - The box rule is "tightly enclosing". Tight boxes keep mAP@[.50:.95] consistent across frames.
  - A hand blurred badly enough to be cut off usually meets exclusion criterion (b).
  - If the blur is mild and the frame is kept, the box is widened by hand. This is rare, and quicker than narrowing almost every box if the margin were 12%.
- **Review guidance from the pilot (user):**
  - Detected hands at rest still get a box (for example Teman s3 f0010). Exclude these with Ctrl+1 (a); never approve with Space just because a box exists.
  - The last frame of each video (80% position) is the most likely to be back at rest (for example Saya s0 f0054).
  - C_two frames flagged `two_class_one_hand` in `data/preannot_report.csv` need the box widened to cover both hands (for example Rumah s3 f0018).
  - Frames with no box because of blur (for example Lagi s3 f0017) are excluded with Ctrl+2 (b).
- **Full run:** `scripts/preannotate_hands.py --all`, 15:55 → 15:57 (≈2.5 min, CPU).
  - 3,750 frames: 2,941 got a box, **670 had no hand detected** (exclusion candidates) and **139 were C_two frames with only one hand detected**.
  - 3,080 labels were written to `data/labels_auto/` and copied to `data/labels_yolo/`. No labels existed before, so none were overwritten.
  - No hand detected, by frame position (20/35/50/65/80%): 328 / 80 / 32 / 16 / 214. The misses concentrate at the start and end of the video, as expected for rest poses.
  - No hand detected, by signer: s0 89/450, s1 93/900, s2 115/900, s3 119/750, **s4 254/750**.
  - signer4 stands farther from the camera, so their hands are small. Some mid-video frames with a visible hand were missed (for example `s4_cari_v01_f0022`). These need a box drawn by hand.
  - Videos with no hand detected in any of the 5 frames: `s4_cari_v02`, `s4_cari_v05`, `s4_makan_v03`.
  - C_one frames with two hands detected (the highest one was used): 101.
- **Rejected second pass:** detection on the central half of the frame, upscaled 2×.
  - It recovered 136 of the 670 no-hand frames, but most of the recovered boxes were resting hands at the waist, and in several frames it boxed the resting hand instead of the moving one.
  - It would add boxes to delete rather than save work, so it was not applied. Pre-annotation stays single-pass as piloted.

## 13. Annotation result (completed 28 Oct 2026)
- `scripts/check_annotations.py` passes: 3,750/3,750 frames decided. There are no missed frames, no unreviewed labels, no conflicts, no multi-box labels and no class mismatches.
- Breakdown per class, signer and reason: `scripts/annotation_summary.py` → `results/annotation_summary.csv`.
- **Kept 2,108 frames, excluded 1,642 (43.8%)**: reason a 876, b 766, c 0.
- By signer (kept / excluded %):
  - s0 256 / 43.1%
  - s1 526 / 41.6%
  - s2 587 / 34.8%
  - s3 336 / 55.2%
  - s4 403 / 46.3%

  The LOSO test fold for signer3 is the smallest.
- By class (kept): Cari 108, Hari 105, Ingat 147, Lagi 141, Maaf 147, Makan 119, Motor 144, Saya 147, Terima kasih 156, Bagaimana 154, Berangkat 134, Teman 143, Keluarga 147, Rumah 155, Pagi 161.
- By group (kept / excluded %): C_one 1,512 / 45.0%, C_two 596 / 40.4%.
- **Pre-annotation vs manual review** (2,108 kept boxes):
  - **2,055 (97.5%) accepted unchanged** (every edge within 2 px).
  - 41 (1.9%) corrected.
  - 12 (0.6%) drawn manually on frames where no hand was detected.
  - Median IoU between the automatic and the final box is 0.990.
  - Another 984 automatic boxes belonged to frames that were excluded.
  - The final boxes are therefore essentially MediaPipe landmark boxes with an 8% margin, each one checked by eye. This should be stated in the paper.
- **Videos with no kept frame (12)**, reported as required by the exclusion rules:
  - s1: `rumah_v07`
  - s2: `cari_v05`
  - s3: `cari_v08`, `hari_v05`, `hari_v10`, `keluarga_v04`, `lagi_v10`, `makan_v06`, `makan_v10`, `saya_v04`
  - s4: `cari_v02`, `motor_v06`
- Kept frames per video: 0 → 12 videos, 1 → 48, 2 → 184, 3 → 352, 4 → 134, 5 → 20.

## 14. Splits (28 Oct 2026)
- **Rules** (decided by the user, also in CLAUDE.md):
  - **P1**: random frame-level split, stratified per class, 70/15/15, split seed 0. Frames of the same video may fall into different partitions on purpose, because that leakage is what P1 measures.
  - **LOSO validation**: 15% of the videos of the training signers (signer0 included), stratified per class, seed 0. All frames of a video stay in one partition, and the test signer is never used for validation.
  - Per-class sizes are `round(fraction × n)`; the remainder goes to test (P1) or train (LOSO).
- **Script:** `scripts/make_splits.py`, which runs `check_annotations.py` first and refuses to run if it fails.
- **Input:** only the 2,108 kept frames.
- **Checks passed:**
  - partitions are disjoint and together cover all kept frames;
  - the test signer is absent from train/val;
  - no LOSO video is split across partitions.
- **Output:** `splits/*.txt` (frame file names) and `results/split_counts.csv` (frames per class per partition).

| Split | Train | Val | Test |
|---|---|---|---|
| P1 | 1,477 | 315 | 316 |
| LOSO s1 | 1,340 (84 val videos) | 242 | 526 |
| LOSO s2 | 1,286 | 235 | 587 |
| LOSO s3 | 1,495 | 277 | 336 |
| LOSO s4 | 1,432 | 273 | 403 |

- **Smallest classes in each test partition:**
  - P1: Hari 15, Cari 16.
  - s1: Cari 21, Hari 23.
  - s2: Cari 16, Hari 20.
  - **s3: Hari 12**, Cari 16, Makan 17, Berangkat 17.
  - s4: Cari 14, Keluarga 21.

  Per-class AP on small test sets (12–16 frames) is noisy. This is worth a sentence in the paper.

## 15. Training configuration (locked 28 Oct 2026)
- **Source of truth:** `configs/train.yaml`, also summarised in CLAUDE.md.
- **Models:** all COCO-pretrained.
  - RT-DETR-L and YOLO11m (Ultralytics 8.4.190).
  - Faster R-CNN ResNet-50 FPN v1 (torchvision 0.29.1, `COCO_V1` weights).
- **Input:** 640. Faster R-CNN uses `min_size=max_size=640`.
- **Augmentation, identical for all models:**
  - On: hsv_h 0.015, hsv_s 0.4, hsv_v 0.3, translate 0.1, scale 0.2.
  - Off: fliplr, flipud, mosaic, mixup, cutmix, copy_paste, degrees, shear, perspective, erasing, bgr.
  - Faster R-CNN is trained through the same Ultralytics transform classes (RandomPerspective with letterbox, RandomHSV) inside its torchvision training loop.
  - Albumentations is not installed, so Ultralytics' Albumentations step is a no-op.
  - RT-DETR uses the same letterbox pipeline as YOLO in this Ultralytics version (`RTDETRDataset` does not override `build_transforms`).
- **Optimisers (explicit, not `auto`):**
  - YOLO11m: SGD, lr0 0.01, momentum 0.937, weight decay 5e-4.
  - RT-DETR-L: AdamW, lr0 1e-4, weight decay 1e-4.
  - Faster R-CNN: SGD, lr 0.02 × batch/16, momentum 0.9, weight decay 1e-4.
  - All models use warmup and AMP.
- **Seeds:** training seeds 0 and 1, independent of the split seed 0.
- **Epochs:** early stopping on validation mAP@[.50:.95]. The maximum number of epochs and the patience will be set from the pilot and locked before the main runs.

## 16. Conversion and pilot (28 Oct 2026)

### Conversion
- `scripts/prepare_data.py` writes `data/coco/<split>.json` for all 15 split lists:
  - pixel boxes, `category_id` = class + 1, fixed `image_id` per frame;
  - the YOLO → COCO → YOLO round trip has a maximum error of 0.0005 px.
- The COCO boxes of 10 random P1-test frames were redrawn (`results/coco_check/`) and all sit tightly on the signing hands with the right gloss.
- For Ultralytics, the same script builds `data/yolo/` with directory junctions to `data/frames` and `data/labels_yolo` (no copies), the list files and `<split>.yaml`.

### Common evaluator
- `scripts/coco_eval.py` handles all models with the same export settings: conf 0.001, NMS IoU 0.7 and at most 100 detections per image.
- Faster R-CNN uses `box_score_thresh=0.001`, `box_nms_thresh=0.7` and `box_detections_per_img=100`.

### Batch size (largest without spilling)
- On Windows the driver spills into shared system memory instead of raising out-of-memory. A batch therefore "fits" only when peak reserved VRAM stays below 4 GiB **and** time per image does not grow.
- The display uses another 0.2–0.9 GB.
- Probes ran 1 epoch on 20% of P1-train (295 images).

| Model | Batch | Peak reserved (GiB) | Train time for 295 images | Verdict |
|---|---|---|---|---|
| YOLO11m | 4 | 2.17 | 76 s | fits |
| YOLO11m | **6** | 3.15 | 78 s | **chosen** |
| YOLO11m | 8 | 4.09 | 92 s | spills |
| YOLO11m | 16 | 7.88 | 263 s | spills |
| RT-DETR-L | 2 | 2.23 | 201 s | fits |
| RT-DETR-L | **4** | 3.65 | 190 s | **chosen**, close to the limit |
| Faster R-CNN | 4 | 2.46 | 176 s | fits |
| Faster R-CNN | **6** | 3.28 | 178 s | **chosen**, lr = 0.02 × 6/16 = 0.0075 |
| Faster R-CNN | 8 | 4.25 | 349 s | spills |

### Speed limits
- The data loader delivers 150 images/s, so the GPU is the bottleneck.
- The GPU reaches 87–94 °C under load and reports thermal clock slowdown (throttle reason 0x20).
- Deterministic mode is not the cause of the low speed.

### Pilot YOLO11m, P1, seed 0, 10 epochs, batch 6
- Train+val took 379–534 s per epoch (mean ≈ 470 s), 78 min in total.
- Validation mAP50-95 (Ultralytics, used as fitness), epochs 1–10: 0.629, 0.688, 0.679, 0.700, 0.728, 0.748, 0.776, 0.782, 0.801, 0.834. It was still rising at epoch 10, so the model has not converged.
- **Test (best.pt):**

| Evaluator | mAP50 | mAP50-95 |
|---|---|---|
| pycocotools | 0.9945 | 0.8255 |
| Ultralytics `val` | 0.9902 | 0.8214 |

- The differences are 0.004 overall and at most 0.018 for any class (Bagaimana).
- Ultralytics caps AP50 at 0.995 because of its interpolation.
- The conversion and the export are therefore consistent.
- Lowest AP50-95: Hari 0.632, Cari 0.734, Keluarga 0.764.

### Pipeline test RT-DETR-L, P1, 1 epoch, batch 4
- 846 s train, 910 s train+val.
- Test mAP50-95: pycocotools 0.0071, Ultralytics 0.0072.
- The whole epoch is inside the 3-epoch warmup, so a near-zero mAP is expected. The loss decreased (cls 70 → 56, as summed over decoder layers and denoising) and did not diverge.

### Pipeline test Faster R-CNN, P1, 1 epoch, batch 6
- The run was stopped by the host at iteration 240/247 because system RAM ran low. This was not an error in the script.
- Training was stable: loss 3.40 → 0.52, peak 3.28 GiB.
- The validation and test path (prediction → `scale_boxes` → pycocotools) is **not yet verified** and must be rerun.
- Estimated train time from the probe: ≈ 890 s per epoch.

## 17. Decisions after the laptop pilot; training moved to Kaggle (28 Oct 2026)

### Decisions (user)
- **RT-DETR-L:** `warmup_bias_lr=0` and AdamW beta1 (`momentum`) 0.9. The Ultralytics default bias warmup of 0.1 is meant for SGD; Ultralytics itself sets it to 0 when it picks AdamW automatically.
- **Gradient accumulation (for the paper):** each framework keeps its own default.
  - Ultralytics accumulates gradients up to a nominal batch of `nbs=64` and scales weight decay by batch × accumulate / 64.
  - The torchvision loop for Faster R-CNN does not accumulate; instead it scales lr by batch/16.
- **Faster R-CNN lr schedule:** linear warmup over 3 epochs (at least 100 iterations) from 0, then linear decay to 1% of lr at the last epoch, the same as Ultralytics (`warmup_epochs`, `lrf` in `configs/train.yaml`).
- **All 30 training runs on Kaggle:**
  - one GPU type, a single GPU, no DDP;
  - the same batch for all three models if it fits (target 16; Faster R-CNN lr = 0.02 × batch/16);
  - the laptop is used only for the final pycocotools evaluation and FPS.
  - The laptop batch values in §16 (6/4/6) no longer apply to the main runs.

### Kaggle workflow
- **Laptop → dataset:** `scripts/kaggle_build_dataset.py` writes `kaggle_upload/bisindo-wl15/` and a zip (348 MB, untracked).
  - Contents: 2,108 kept frames, their labels, splits, configs, scripts, `requirements_freeze.txt` and the three exact pretrained weight files (`yolo11m.pt` sha256 d5ffc1a6…, `rtdetr-l.pt` 6de60b10…, `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`).
  - Also a sha256 manifest of 4,220 files and the sha256 of the 15 COCO files.
- **Kaggle notebook:** `scripts/kaggle_notebook.ipynb` runs `scripts/kaggle_session.py --plan <name>` (plans in `configs/kaggle_plan.yaml`), which:
  - installs torch 2.14.1 / torchvision 0.29.1, using the laptop's CUDA build cu130 if it works on the Kaggle driver, otherwise cu128, then cu126;
  - installs the other pinned packages, and records every difference from the laptop in `runs/env_kaggle.json`;
  - checks the manifest, regenerates the COCO/YOLO views and requires byte-identical COCO files;
  - restores `runs/` from an attached earlier output;
  - probes the batch once (16 → 12 → 8 for all three models; the GPU type is fixed from then on);
  - runs the plan and writes `runs/session_report.json` and `runs/sessions.csv` (GPU hours).
- **Resuming across 12 h sessions:**
  - Checkpoints are saved every epoch.
  - After a saved epoch, the training scripts exit with code 3 if the next epoch would end after the 11 h budget.
  - The next session resumes from `last.pt` and restores the early-stopping state: Ultralytics would otherwise restart the patience count, so it is rebuilt from `results.csv`.
- **Batch guard:** Ultralytics 8.4.190 silently halves the batch after an out-of-memory error in epoch 1. The script stops the run instead, so the locked batch is never changed unnoticed.
- **Kaggle → laptop:** `scripts/kaggle_download.py --kernel <user>/<slug>` (Kaggle CLI 2.2.5 in `.venv-kaggle`) or `--zip <output.zip>`.
  - Everything goes to `runs/kaggle/<timestamp>/`.
  - The CSV/JSON results are copied to `results/kaggle/`.

### Tests on the laptop (temporary 60/30/30-image split, removed afterwards)
- **YOLO11m:**
  - with the deadline set to "now", it stopped after epoch 1 (exit 3);
  - rerun, it resumed at epoch 2 with the early-stopping state restored (best epoch 1), finished epoch 3 and scored the test split with both evaluators.
- **Faster R-CNN:**
  - the same stop/resume behaviour;
  - this also verifies the validation and test path that the interrupted laptop test (§16) had not reached;
  - predictions are mapped back to the original 1280×720 frames (max x2 = 1127; median best IoU with the ground truth 0.73 after 3 tiny epochs).
- The test runs remain in `runs/ultralytics/zztest_yolo` and `runs/frcnn/zztest_frcnn`.

## 18. Desktop application and FPS measurement (28 Oct 2026)
- **Manuscript section 2.6** (`BISINDO_JAIC2027.docx`) defines `t_inf` in eq. (18) as the model prediction stage alone, excluding frame acquisition and interface rendering. CLAUDE.md rule 7 says the same.

### Scripts
- **`scripts/detectors.py`:** one interface for YOLO11m, RT-DETR-L and Faster R-CNN.
  - Each stage (preprocess, inference, postprocess) is timed with Ultralytics' `ops.Profile`, which synchronises CUDA.
  - Ultralytics models use their own predictor (LetterBox 640, minimal padding).
  - Faster R-CNN uses the same LetterBox. Its "inference" includes torchvision's normalisation, RPN and ROI NMS; its "postprocess" is the rescale only.
- **`scripts/app_detect.py`:** the desktop app (Python + OpenCV).
  - Reads the webcam and draws the box with gloss and score; C_two boxes are orange, C_one green.
  - Shows **Display FPS** (the whole loop) and **Model latency** (the three stages) separately. Keys 1/2/3 switch model.
  - Frames are not mirrored, because the training frames were never flipped.
  - Weights come from `configs/app.yaml`. They are temporary for now: the YOLO11m laptop pilot, and RT-DETR-L and Faster R-CNN test checkpoints that are not trained.
- **`scripts/measure_fps.py`:** the paper FPS.
  - P1-test frames are loaded into memory before timing; nothing is drawn; batch 1; FP16.
  - Same settings as the mAP evaluation: conf 0.001, NMS IoU 0.7, max 100 detections.
  - 50 warm-up iterations, then the mean over 500 iterations.
  - `fps = 1000 / mean(pre + inf + post)` (eq. 18), `fps_net` = network forward only.
  - Also records parameter count, GPU temperature and SM clock before and after, with a cooldown between models.
  - Appends to `results/fps.csv`; tests go to `runs/fps_test.csv`.

### Webcam findings
- With DirectShow the camera ignores the MJPG request and delivers 1280×720 YUY2 at 10 FPS. Media Foundation (MSMF) delivers 29.7–30.7 FPS, so the app uses MSMF.
- Camera reading runs in a background thread that keeps only the newest frame, so camera wait and model time do not add up. Before this change RT-DETR-L showed 9.3 display FPS; after it, 22.5.

### Test values with temporary weights (not for the paper)

| Model | pre (ms) | inf (ms) | post (ms) | pred (ms) | FPS | Net-only FPS | Params | GPU |
|---|---|---|---|---|---|---|---|---|
| YOLO11m | 0.9 | 13.7 | 1.6 | 16.3 | 61.5 | 72.8 | 20.1 M | 73→80 °C |
| RT-DETR-L | 2.3 | 33.2 | 0.8 | 36.4 | 27.5 | 30.1 | 32.8 M | 76→85 °C |
| Faster R-CNN | 1.3 | 42.7 | 0.6 | 44.6 | 22.4 | 23.4 | 41.4 M | 79→86 °C |

- These used 200 iterations per model.
- App display FPS on the webcam: YOLO11m ≈ 30 (camera-bound), RT-DETR-L 22.5, Faster R-CNN 15.2.
- A transient drop of YOLO11m to 15 FPS in one run was not reproduced (29.9 on rerun). The camera's auto-exposure and GPU throttling make display FPS unstable, which is why it is not used for the paper.
- With the model trained on non-transition frames only, the app labels hands moving to rest with a gloss (for example "Teman" 0.78 on a lowered hand in `signer0_label9_sample1`). Excluded transition frames were never seen in training. This is relevant for the discussion of the application.
- **Final FPS rules (user, 28 Oct 2026; also in CLAUDE.md):**
  - Measure after all training, with the final Kaggle checkpoints.
  - Laptop on the charger, high-performance power mode, cooling pad, Chrome and Discord closed.
  - Cool down until the GPU is back to normal (≤ idle temperature at start + 5 °C) before every measurement.
  - 3 repeats per model, reported as mean ± SD with GPU temperatures.
  - FP16 for all three models.
  - `measure_fps.py` implements the cooldown wait and the repeats. A test with 2 repeats gave YOLO11m 57.6 ± 0.3 FPS, then at 67→79 °C (temporary weights).
- **Rest-hand detections:** recorded as discussion and limitation material only; the training data are not changed.

## 19. First Kaggle pilot attempt failed in stage_env; fix (28 Oct 2026)
- **Failure:** `stage_env` stopped after about 3 min with a `JSONDecodeError`. The torch build (cu130) had installed fine.
  - Cause: on the first import, Ultralytics writes a three-line "Creating new Ultralytics Settings…" notice to stdout. The child that reports the environment printed its JSON after that notice, so the whole stdout was not valid JSON.
  - Reproduced on the laptop with an empty `YOLO_CONFIG_DIR`. No training ran, so no results are affected.
- **Fix in `kaggle_session.py`:** `child_json()` prints the JSON on one line prefixed with `@@BISINDO_JSON@@` and parses only that line. On any failure it stops with the code, the full stdout and the full stderr.
- **Other subprocess reads checked, none of them parse JSON from stdout:**
  - `cuda_works`: uses the exit code.
  - `pip freeze`: reads lines containing `==`.
  - `nvidia-smi`: CSV output.
  - `prepare_data`: uses the exit code.
  - Training runs: piped to a log; the results come from CSV files.
- **Duplicated log lines:** the notebook now pipes the session's output and prints it line by line. A child that writes straight to the kernel's stdout appears more than once in the saved Kaggle log.
- **Local rehearsal hooks:**
  - `BISINDO_DATASET`, `BISINDO_WORK`, `BISINDO_INPUT` point the script at local folders.
  - `BISINDO_SKIP_INSTALL=1` skips the package installs.
  - `BISINDO_PROBE_BATCHES` sets the probe batches.
  - On Windows, junctions replace symlinks.
  - Kaggle runs with none of these set, so it behaves exactly as before.
- **Rehearsal on the laptop:**
  - Setup: the built dataset folder, an empty work folder, an empty `YOLO_CONFIG_DIR` (the settings notice did appear) and an empty `TORCH_HOME`; probe batch 2.
  - Every stage passed: env (0 differences) → manifest → COCO identical → restore (fresh) → probe (YOLO 1.26 / RT-DETR 2.23 / FRCNN 1.48 GiB) → `pilot_yolo11m_p1_seed0` started training. It was stopped at epoch 1, iteration 36.
- **Dataset:** uploaded as a new version of `ajimaru12/bisindo-wl15`. Kaggle's copy of `kaggle_session.py` is byte-identical to the repo's.

## 20. Kaggle pilot-2 stopped in the batch probe; laptop and Kaggle code paths aligned (28 Oct 2026)
**Pilot-2 (T4 ×2, only GPU 0 used):** env OK.
- Python is 3.13.15 on Kaggle vs 3.11 on the laptop. The image's Python cannot be pinned, so this is recorded as a difference.
- Probe results:
  - YOLO11m fits at batch 16/12/8 (peak 7.84/5.94/4.07 GiB).
  - RT-DETR-L fits at batch 16/12/8 (peak 12.44/9.65/6.53 GiB).
  - Faster R-CNN ran out of memory at all three batches, already in the first iteration (backward pass).

**1. Faster R-CNN OOM: the cause was torch.compile inside torchvision, not the resize.**
- `min_size = max_size = 640` comes from `configs/train.yaml` on both machines, and the data loader is the same.
- `init_seeds(deterministic=True)` turns on `torch.use_deterministic_algorithms`. With that on, torchvision 0.29's `roi_align` switches to a pure-Python `_roi_align` wrapped in `torch.compile`, but only when `is_compile_supported('cuda')` is true:
  - Kaggle (Linux, triton): true.
  - Laptop (Windows, no triton): false, so the laptop always used the C++/CUDA RoIAlign kernel.
- The compiled backward (`aten.index_put_` with accumulate) needed about 12 GB at batch 8.
- The deterministic mode itself is the same on both machines; only this RoIAlign branch differed.

**Fix: `scripts/run_guards.py`, imported first by `train_ultralytics.py`, `train_frcnn.py` and `detectors.py`.**
- Dynamo is disabled (`TORCHDYNAMO_DISABLE=1`, `torch._dynamo.config.disable`), so `torch.compile` returns the function unchanged in every model. Ultralytics also gets `compile=False` explicitly.
- torchvision's `is_compile_supported` check is answered "not supported", so RoIAlign uses the C++/CUDA kernel on both machines.
  - Consequence: the RoIAlign backward is not bit-deterministic (atomicAdd). PyTorch warns because `warn_only=True`. This was already the case in every laptop run.
  - The other deterministic settings are unchanged.
- **Tested on the laptop:**
  - With `is_compile_supported` forced to true (as on Kaggle), the compiled Python path is taken without the guard and the CUDA kernel with it.
  - `torch.compile(f) is f`.
  - Faster R-CNN at batch 6 peaks at 3.27 GiB, the same as the earlier laptop test (3.3 GB).

**2. Albumentations was active on Kaggle in all three models:** Blur, MedianBlur, ToGray and CLAHE, p = 0.01 each. Kaggle has the package preinstalled, and Ultralytics adds these transforms whenever the package can be imported.
- They are not in the locked augmentation and are now removed everywhere:
  - `kaggle_session.py` uninstalls `albumentations` after the pinned installs. The session stops if the package can still be imported, which `env_kaggle.json` records.
  - `run_guards.apply()` stops every training run if the package can be imported (laptop as well).
  - `run_guards.check_transforms()` checks the training dataset's transform pipeline at the start of every run and stops if an Albumentations step is active.
  - `run_logged` stops any run whose output contains `albumentations:`, which Ultralytics prints only when those transforms are active. The probe or plan then stops (exit 4).
- **Laptop pilot not affected:**
  - albumentations is not installed in `.venv`, so Ultralytics' Albumentations step was a no-op (§15); it is not in `requirements_freeze.txt`.
  - triton is not installed either.
- Only the Kaggle pilot-2 probe epochs (never used for results) ran with albumentations.

**3. Manifest check took 8 min per session on Kaggle.** The cause is per-file latency on the network-mounted `/kaggle/input`.
- Files are now hashed in parallel (32 threads). Every file is still fully hashed with sha256 against the laptop manifest, so the integrity guarantee is unchanged.

**Rehearsal on the laptop** (empty settings and torch hub, probe batch 2): all stages passed up to the start of the YOLO pilot.
- Every run log shows `guards: torch.compile off, RoIAlign = torchvision CUDA kernel, albumentations absent` and the transform list.
- Batch rule unchanged: the same batch for all three models, tried 16 → 12 → 8.

## 21. Kaggle pilot-3 and locked main configuration (28 Oct 2026)
**Pilot-3 (Tesla T4, GPU 0 only, batch 16; session 1.44 h):**
- Manifest check took 2 s (was 8 min); the guards were active in all 6 runs; no `albumentations:` line.
- Probe at batch 16, peak reserved: YOLO11m 7.83 GiB, RT-DETR-L 12.45 GiB, Faster R-CNN 9.03 GiB. All fit, so 16 → 12 → 8 stopped at 16.

| Run | Epochs | Best epoch | s/epoch (train+val) | Best val mAP50-95 | Test mAP50-95 pycocotools / Ultralytics |
|---|---|---|---|---|---|
| YOLO11m P1 seed 0, max 100, patience 20 | 100 (ran to the cap) | 95 | 43.3 | 0.8706 | 0.8589 / 0.8619 |
| RT-DETR-L P1 seed 0, 2 epochs | 2 | 2 | 98.9 | 0.0913 | 0.0903 / 0.0904 |
| Faster R-CNN P1 seed 0, 2 epochs | 2 | 2 | 122.4 | 0.2608 | 0.2672 / – |

**Locked for the main runs (user decision):**
- **Batch:** 16 for all three models; Faster R-CNN lr = 0.02 × 16/16 = 0.02. `configs/train.yaml` holds `batch` and `kaggle_gpu: Tesla T4`; `kaggle_session.py` takes the locked batch (no new probe) and stops if the GPU is not a Tesla T4.
- **Epochs:** max 100, patience 20, for all three models.
- **Best checkpoint and early stopping:** pure validation mAP50-95 for all three models. Both trainers stop when (epoch − best epoch) ≥ patience, and an epoch counts as better only with a strictly higher value.
  - Ultralytics 8.4.190: fitness = (P, R, mAP50, mAP50-95) · (0, 0, 0, 1) (`Metric.fitness`), so it is not changed.
  - The 0.1 mAP50 + 0.9 mAP50-95 form is from older Ultralytics versions. In 8.4.190 it survives only in the faster-coco-eval branch for COCO/LVIS JSON outside training, which our runs never take.
  - Pilot evidence: over all 100 epochs, |fitness − val mAP50-95| = 0 (the 0.9/0.1 mix would differ by up to 0.029); the best epoch (95) is the argmax of val mAP50-95.
  - The rule is enforced and recorded:
    - `train_ultralytics.py` checks the weights at start-up (stops otherwise).
    - Every epoch it checks that fitness equals val mAP50-95 (stops otherwise).
    - The rule is printed in the run log and stored in `eval_test.json` (`selection`, `patience`, `epochs_max`).
    - Faster R-CNN: pycocotools mAP50-95 on the validation split; `eval_test.json` records the same fields.
  - Remaining difference (for the manuscript): the per-epoch validation mAP50-95 of YOLO11m and RT-DETR-L comes from Ultralytics' validator, and that of Faster R-CNN from pycocotools. Both are COCO-style AP over IoU 0.50:0.95 with 101-point interpolation; on the pilot test split they differ by ~0.003. All reported test numbers use pycocotools for all three models (rule 6).
- **Main plan** (`configs/kaggle_plan.yaml`, 30 runs):
  - Order: P1 seed 0, P1 seed 1, then loso_s1 seed 0/1, s2, s3, s4.
  - Within each block: YOLO11m → RT-DETR-L → Faster R-CNN.
  - Names: `<model>_<split>_seed<seed>`.
  - The pilot runs are not reused as main runs; YOLO11m P1 seed 0 is trained again under the main plan.
- **Sessions:** budget ≤ 11 h. A run is paused after a saved epoch when the next epoch would not finish in time, and no run is started with < 20 min left. Each later session must attach the previous session's output, and restore then resumes the unfinished run from `last.pt`, with early-stopping state.

**Estimate:**
- Epoch time scales with each fold's train/val size (P1 1477/315; s1 1340/242; s2 1286/235; s3 1495/277; s4 1432/273); about 2 min per run for test evaluation; about 10.5 usable hours per session.

| Scenario (epochs actually run) | YOLO11m | RT-DETR-L | Faster R-CNN | Total GPU h | Sessions | Weeks at 30 h |
|---|---|---|---|---|---|---|
| all 100 (upper bound) | 11.8 | 25.5 | 31.9 | 69.1 | 7 | 2.3 |
| 100 / 80 / 50 | 11.8 | 20.4 | 16.1 | 48.3 | 5 | 1.6 |
| 90 / 60 / 35 | 10.6 | 15.4 | 11.4 | 37.4 | 4 | 1.2 |

- The P1 block alone (6 runs) is at most ~14.7 h, i.e. 1–2 sessions.
- Note: YOLO11m reached the 100-epoch cap in the pilot (best epoch 95), so some YOLO runs may end at the cap rather than by patience. The cap stays as decided; how many runs end at the cap will be reported.

## 22. Main session 1 (notebook version 4, 28–29 Oct 2026) and session length limit

- **Session:** 10.98 h on a Tesla T4, batch 16, plan `main`. Five P1 runs finished; `frcnn-r50fpn_p1_seed1` paused after epoch 11 (session deadline) and resumes in session 2 from `last.pt`.
- **Downloaded** with `kaggle_download.py --kernel ajimaru12/bisindo-wl15-training/4` (version 5, session 2, was already running, so the latest version had no output yet) → `runs/kaggle/20261029-0906/`.

| Run | Epochs | Best epoch | Val mAP50-95 | s/epoch | Train h | Test mAP50-95 (pycocotools) | Ended by |
|---|---|---|---|---|---|---|---|
| yolo11m_p1_seed0 | 100 | 95 | 0.8706 | 41.7 | 1.16 | 0.8589 | cap (100) |
| rtdetr-l_p1_seed0 | 95 | 75 | 0.8842 | 91.8 | 2.42 | 0.8831 | patience |
| frcnn-r50fpn_p1_seed0 | 100 | 88 | 0.8306 | 115.8 | 3.22 | 0.8320 | cap (100) |
| yolo11m_p1_seed1 | 100 | 85 | 0.8706 | 41.1 | 1.14 | 0.8593 | cap (100) |
| rtdetr-l_p1_seed1 | 100 | 99 | 0.8777 | 92.1 | 2.56 | 0.8894 | cap (100) |

- **Reproducibility check:** the main run `yolo11m_p1_seed0` reproduces pilot-3 (`pilot_yolo11m_p1_seed0`, notebook version 3) exactly on the same GPU type and code: all 649 tensors of `best.pt` are equal, `preds_test_coco.json` is byte-identical, `results.csv` differs only in the wall-time column, and the test mAP (pycocotools and Ultralytics) and all per-class APs are equal. Training is therefore deterministic for a given seed on the Kaggle T4.
- **100-epoch cap:** four of the five finished runs ended at the cap rather than by patience (only `rtdetr-l_p1_seed0` stopped by patience). The cap stays as decided (§21); it is reported per run. The session time estimate is therefore closer to the upper bound of §21 (~69 GPU h, 7 sessions).
- **`results/runs.csv`, `results/per_class.csv`:** built by `scripts/collect_results.py` from the downloaded outputs. It re-scores each run's saved test predictions with the common pycocotools evaluator on the laptop and stops if the result differs from Kaggle's `eval_test.json` (all five: identical). `fps` stays empty until the final FPS measurement; `precision`, `recall`, `f1` stay empty until their definition is decided.
- **Session length limit:** `kaggle_session.py --max-hours` (notebook cell `MAX_HOURS`, default 10.8 h; allowed (0, 11.5]). It is counted from the start of the session script; no epoch is started that would end after it, and no run is started with < 20 min left. The report and the saving of the notebook output take a few more minutes. Lower it when less than ~11 h of the weekly Kaggle GPU quota is left, so the quota never cuts a session off before its output is saved.

## 23. Precision, recall and F1: definition and threshold from validation (29 Oct 2026)

Decided by the user:
- **Definition:** precision, recall and F1 come from the same COCO-format predictions for all three models, at IoU 0.5, macro-averaged over the 15 classes.
- **Threshold:** the confidence threshold is chosen per run on the validation split (the one that maximises macro-F1), then applied to the test split.
- **Recording:** the chosen threshold is recorded per run in `runs.csv`. The same threshold will later be the desktop application's default for the checkpoint it uses.

Implementation (`scripts/coco_eval.py`: `match_iou50`, `prf`, `select_threshold`; `scripts/collect_results.py`):
- **Matching:** pycocotools' own matching (greedy by score, IoU ≥ 0.5, at most 100 detections per image, area "all"). It is the same matching as the AP50, run with `iouThrs = [0.5]`.
- **Per class, at threshold t** (detections with score ≥ t are kept):
  - P = TP / (TP + FP), or 0 when no detection is kept;
  - R = TP / n_gt;
  - F1 = 2PR / (P + R), or 0 when P + R = 0.
- **Macro:** P, R and F1 are means over the 15 classes. Macro-F1 is the mean of the per-class F1, not the harmonic mean of macro-P and macro-R.
- **Threshold search:** every detection score on the validation split is a candidate (exact maximisation). On ties the highest threshold is taken.
- **Checked** against an independent brute-force matcher on `rtdetr-l_p1_seed0` test predictions at t = 0.001, 0.25 and 0.5: identical to 6 decimals.
- **Validation predictions:** Kaggle saved only the test predictions, so `best.pt` predicts the validation split on the laptop (RTX 3050). Settings are the same as the test predictions: conf 0.001, NMS IoU 0.7, FP16. YOLO11m and RT-DETR-L use batch 16. Faster R-CNN uses a loader batch of 8 instead of 32, because 32 does not fit in 4 GB. Cached in `runs/laptop_preds/<run>/`.
- **Check:** the test split is also predicted on the laptop.
  - Test mAP vs Kaggle: max difference 4.4 × 10⁻⁴ (limit 2 × 10⁻³).
  - Test P/R/F1 at the chosen threshold: identical to 5 decimals with laptop or Kaggle test predictions.
  - The values written use Kaggle's test predictions, which are the same ones behind the mAP.
- **`runs.csv`:** new columns `conf_thr` and `val_f1` (macro-F1 on validation at that threshold).

| Run | conf_thr | val macro-F1 | test P | test R | test F1 |
|---|---|---|---|---|---|
| yolo11m_p1_seed0 | 0.811 | 0.9868 | 0.9897 | 0.9623 | 0.9751 |
| rtdetr-l_p1_seed0 | 0.417 | 0.9938 | 0.9903 | 0.9936 | 0.9918 |
| frcnn-r50fpn_p1_seed0 | 0.753 | 0.9899 | 0.9970 | 0.9928 | 0.9947 |
| yolo11m_p1_seed1 | 0.813 | 0.9874 | 0.9917 | 0.9797 | 0.9852 |
| rtdetr-l_p1_seed1 | 0.274 | 0.9873 | 0.9813 | 0.9958 | 0.9883 |
