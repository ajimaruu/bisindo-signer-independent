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

## 24. Main session 2 (notebook version 5) and the YOLO single-label export finding (29 Oct 2026)

**Session 2:** 10.99 h on a Tesla T4.
- Finished: `frcnn-r50fpn_p1_seed1` (resumed from epoch 11), `yolo11m_loso_s1_seed0`, `rtdetr-l_loso_s1_seed0`, `frcnn-r50fpn_loso_s1_seed0`, `yolo11m_loso_s1_seed1`.
- Paused: `rtdetr-l_loso_s1_seed1` at epoch 44.
- Downloaded to `runs/kaggle/20261029-2046/`. `runs.csv` / `per_class.csv` now hold 10 runs.
- P1 summary: `scripts/summarise.py --protocol P1` → `results/summary_p1.md`, `.csv`, `_per_class.csv`.

**Finding (investigated, nothing changed yet):** on `yolo11m_loso_s1_seed0` the test mAP50-95 is 0.718 with pycocotools and 0.736 with Ultralytics' `model.val()` (gap 0.018; in P1 it was below 0.004). Checked on the laptop with the Kaggle `best.pt`:

| Step (test, loso_s1) | mAP50-95 |
|---|---|
| Our exported detections (`predict()`, NMS `multi_label=False`), pycocotools | 0.7181 |
| Same input, NMS `multi_label=True` (as Ultralytics' validator), pycocotools | 0.7364 |
| The validator's own detections (rect input + 0.5-stride padding, multi-label), pycocotools | 0.7370 |
| Same detections, Ultralytics' AP computation | 0.7363 |

- **Cause:** Ultralytics' predictor gives each box only its top class (`multi_label=False`). Its validator, used for the per-epoch val metric and `model.val()`, keeps every class above conf for each box (`multi_label=True`).
  - On the unseen signer the model often ranks the true class second, for example `s1_cari_v08_f0055`: Berangkat 0.545, Cari second.
  - The single-label export then drops the true-class detection entirely, which caps recall. Cari has AP50 0.21 single-label against 0.49 multi-label.
  - In P1 the top class is almost always right, hence the small gap there (0.8576 vs 0.8618 on p1_test).
- **Not the cause:** the AP formula (−0.0007) or input padding (+0.0006). `max_det` 100 vs 300 makes no difference either, because YOLO keeps only ~2 detections per image after NMS at conf 0.001.
- **Other models:**
  - RT-DETR-L uses the same postprocess (one label per query) in `predict()` and in its validator, so it is not affected (0.790 vs 0.789).
  - torchvision Faster R-CNN emits a detection for every class above `box_score_thresh` per proposal, so it is effectively multi-label.
  - YOLO11m is therefore the only model exported single-label.
- **Effect on checkpoint selection:** YOLO/RT-DETR select on Ultralytics' validator (multi-label for YOLO). On validation splits the single-label vs multi-label offset is small (p1_val 0.8679 vs 0.8715; loso_s1_val 0.8447 vs 0.8462), but it can change the ranking of near-tied epochs.
  - `yolo11m_p1_seed1`: best.pt (epoch 85) against last.pt (epoch 100).
    - Validator: 0.8706 > 0.8705.
    - Pycocotools multi-label: 0.8731 > 0.8722.
    - Pycocotools single-label: 0.8669 < 0.8694, i.e. it would have preferred epoch 100.
  - `yolo11m_p1_seed0`: all three agree.
  - So YOLO's selection is consistent with a multi-label definition, not with the single-label test export. RT-DETR's selection is consistent with its export.
- Reported to the user for a decision before anything is changed.

## 25. Multi-label export for YOLO11m; all paper numbers from the laptop export (29 Oct 2026)

**Decided by the user after the P1 and LOSO-s1 results had been seen** (§24).

**What changed:** YOLO11m's exported predictions (test, and validation for the P/R/F1 threshold) now use NMS with `multi_label=True`. Every class above conf 0.001 is kept for each candidate box, then class-wise NMS is applied.

Implementation:
- `coco_eval.predict_ultralytics(..., multi_label=True)`: `predict()` preprocessing is unchanged; Ultralytics' NMS is patched only for the duration of the call.
- Applied to all YOLO runs, P1 and LOSO, test and validation.
- No retraining; checkpoints and checkpoint selection are unchanged.

**Why:**
1. **Uniform across models.** Faster R-CNN (torchvision) already emits one class-specific box per class above the threshold for each proposal. RT-DETR-L's head already ranks all query–class pairs (300 queries × 15 classes) and the exported top 100 can contain several labels for one query. Only YOLO11m was exported with one label per box.
2. **Consistent with checkpoint selection.** YOLO11m's best checkpoint and early stopping use Ultralytics' validator, which is multi-label. With the single-label export the test metric did not match the definition used to select the checkpoint (§24, `yolo11m_p1_seed1`).

**Source of the paper numbers (chosen):**
- All paper numbers for all three models (mAP, per-class AP, threshold, P/R/F1) come from predictions exported on the laptop (RTX 3050) from the Kaggle `best.pt`.
- The Kaggle code and dataset are not changed mid-plan. Main session 3 was already running with the old code, and changing it would mix two export versions. One code version and one machine thus serve all 30 runs.
- Kaggle's `preds_test_coco.json` and `eval_test.json` remain a cross-check in `collect_results.py`:
  - they are re-scored exactly;
  - the laptop export made the way Kaggle made it (single-label YOLO for runs without the `yolo_multi_label` field) must match Kaggle's test mAP within 0.002. Measured: ≤ 4.4 × 10⁻⁴.
- `train_ultralytics.py` now also exports multi-label and writes `yolo_multi_label: true`, in case the dataset is ever updated. `collect_results.py` handles both cases.
- Faster R-CNN on the laptop uses a loader batch of 8 (Kaggle: 32).

**Old numbers kept:** `results/runs_singlelabel.csv`, `per_class_singlelabel.csv`, `summary_p1_singlelabel.csv` and `summary_p1_per_class_singlelabel.csv`. These are the Kaggle test predictions, single-label for YOLO. Cached single-label laptop predictions are in `runs/laptop_preds/yolo11m_*/preds_*_coco_singlelabel.json`.

**Before → after (test):**

| Run | mAP50 | mAP50-95 (Δ) | P / R / F1, conf_thr |
|---|---|---|---|
| yolo11m P1 s0 | 0.9818 → 0.9884 | 0.8589 → 0.8631 (+0.0041) | unchanged (thr 0.811) |
| yolo11m P1 s1 | 0.9890 → 0.9930 | 0.8593 → 0.8624 (+0.0031) | unchanged (thr 0.813) |
| yolo11m LOSO-s1 s0 | 0.9201 → 0.9484 | 0.7184 → 0.7367 (+0.0183) | P 0.9195 → 0.9183, F1 0.8872 → 0.8865 (thr 0.518) |
| yolo11m LOSO-s1 s1 | 0.9163 → 0.9342 | 0.7393 → 0.7508 (+0.0115) | unchanged (thr 0.646) |
| rtdetr-l P1 s0 / s1 | ≤ 0.0002 | −0.0001 / −0.0004 | unchanged |
| rtdetr-l LOSO-s1 s0 | −0.0002 | +0.0000 | unchanged |
| frcnn P1 s0 / s1 | 0 | +0.0004 / +0.0001 | unchanged |
| frcnn LOSO-s1 s0 | 0 | −0.0002 | unchanged |

- The RT-DETR-L and Faster R-CNN changes come only from the switch to the laptop export. Their per-class AP50-95 changes by at most 0.0085.
- **YOLO per-class:** the largest gains are Cari (LOSO-s1 +0.174 and +0.116) and Berangkat (LOSO-s1 s0 +0.056; P1 s1 +0.046).
- **YOLO thresholds:** the chosen thresholds are unchanged, because the extra labels have low scores.
- **P1 summary (mean ± SD), YOLO11m:**
  - mAP50: 0.985 ± 0.005 → 0.991 ± 0.003
  - mAP50-95: 0.859 ± 0.000 → 0.863 ± 0.000
  - Δtwo: −0.009 ± 0.004 → −0.008 ± 0.008
- **P1 summary, RT-DETR-L:** Δtwo −0.018 → −0.017. **Faster R-CNN:** −0.011 → −0.011.

**Manuscript sentence (methods, export):** "For evaluation, the predictions of every model were exported at a confidence threshold of 0.001 with at most 100 detections per image, and a box could carry more than one class label: YOLO11m kept every class whose score exceeded the threshold for each candidate box before class-wise NMS (IoU 0.7), Faster R-CNN produced one class-specific box for every class above the threshold for each proposal before class-wise NMS (IoU 0.7, set explicitly instead of the torchvision default of 0.5), and RT-DETR-L, which uses no NMS, kept the 100 highest query–class scores over its 300 decoder queries and 15 classes."
  - Checked against the code (29 Oct 2026):
    - Faster R-CNN: `build_model` sets `box_nms_thresh = 0.7`, `box_score_thresh = 0.001`, `box_detections_per_img = 100` (torchvision default NMS 0.5 not used); the RPN NMS stays at its default 0.7.
    - RT-DETR-L: the decoder returns its 300 best query–class pairs sorted by score, and `predict()` keeps those above 0.001, then the first `max_det = 100`. Exported files have exactly 100 detections per image, so pycocotools' maxDets 100 cuts nothing further.
    - YOLO11m and Faster R-CNN exports stay well below 100 per image (at most 20 and 36).

**Unchanged:** the desktop application (`app_detect.py`) and the FPS measurement (`measure_fps.py`, via `detectors.py`) keep the plain `predict()`, with one label per box.

## 26. Main session 3 (notebook version 6), LOSO fold s1 complete, and the LOSO aggregation rules (30 Oct 2026)

**Session 3:** 5.48 h on a Tesla T4, with `MAX_HOURS = 5.5` set in the notebook cell (§22) because less weekly quota was left. Downloaded with `kaggle_download.py --kernel ajimaru12/bisindo-wl15-training/6` to `runs/kaggle/20261030-0222/`.

| Run | Epochs | Best epoch | Val mAP50-95 | s/epoch | Train h | Test mAP50-95 (laptop export) | Ended by |
|---|---|---|---|---|---|---|---|
| rtdetr-l_loso_s1_seed1 | 62 | 42 | 0.8565 | 81.8 | 1.41 | 0.7731 | patience (resumed from epoch 44) |
| frcnn-r50fpn_loso_s1_seed1 | 100 | 89 | 0.8069 | 102.5 | 2.85 | 0.7320 | cap (100) |
| yolo11m_loso_s2_seed0 | 100 | 91 | 0.8264 | 34.8 | 0.97 | 0.7596 | cap (100) |
| rtdetr-l_loso_s2_seed0 | 52 so far | 39 | 0.8505 | 77.3 | 1.12 | – | paused at the session deadline |

- `runs.csv` / `per_class.csv`: 13 runs. LOSO fold s1 is complete (3 models × 2 seeds).
- **Cross-checks (`collect_results.py`):**
  - All 13 runs: Kaggle's test predictions re-scored on the laptop reproduce `eval_test.json` exactly.
  - The laptop export made the way Kaggle exported each run matches Kaggle's test mAP within 5.4 × 10⁻⁴ (limit 2 × 10⁻³).
  - The P1 rows are unchanged; only `checkpoint_path` now points to the newest download.
- **yolo11m_loso_s2_seed0, export:**
  - It was scored on Kaggle with the single-label export, because the Kaggle code is deliberately unchanged mid-plan (§25): 0.7430.
  - The paper number is the laptop multi-label export: 0.7596 (+0.0166).
- **100-epoch cap:** `rtdetr-l_loso_s1_seed1` is the first run that stopped well before the cap by patience. Of the 12 runs finished up to fold s1, 8 ended at the cap (100 epochs) and 4 by patience: `rtdetr-l_p1_seed0` (95), `rtdetr-l_loso_s1_seed0` (96), `yolo11m_loso_s1_seed1` (99, best 79) and `rtdetr-l_loso_s1_seed1` (62). Most runs still reaching the cap is reported, as decided (§22).

**Fold s1 against P1** (mean of the 2 seeds; SI over one fold only, so preliminary):

| Model | mAP50-95 s1 | mAP50-95 P1 | Δabs (Δrel) | F1 s1 | F1 P1 |
|---|---|---|---|---|---|
| YOLO11m | 0.744 ± 0.010 | 0.863 ± 0.000 | 0.119 (13.8 %) | 0.891 ± 0.007 | 0.980 ± 0.007 |
| RT-DETR-L | 0.782 ± 0.012 | 0.886 ± 0.004 | 0.104 (11.8 %) | 0.930 ± 0.027 | 0.990 ± 0.002 |
| Faster R-CNN | 0.732 ± 0.001 | 0.832 ± 0.001 | 0.100 (12.0 %) | 0.918 ± 0.006 | 0.992 ± 0.003 |

- **mAP50 drops less** (3.4–5.0 %). The drop is mainly in box precision on the unseen signer.
- **Weakest classes in s1, all three models:** Cari and Hari (one-handed).
- **Δtwo in s1:**
  - YOLO11m +0.008, seed SD 0.048, so its sign is not interpretable;
  - RT-DETR-L −0.052;
  - Faster R-CNN −0.050.
  - The P1 values are −0.008, −0.017 and −0.011.

**LOSO aggregation rules (decided by the user on 30 Oct 2026; fold s1 had been seen, s2–s4 not finished).** Recorded in CLAUDE.md ("Format hasil") and implemented in `scripts/summarise.py --protocol LOSO`:
1. **Per fold:** mean of the 2 seeds ± SD over seeds.
2. **mAP^SI (eq. 7):** the mean over the 4 folds of the per-fold seed means, ± SD over folds. The seed SD is reported separately, as the mean over folds of the within-fold seed SD.
3. **Δabs and Δrel:**
   - Δabs = mAP^rand − mAP^SI.
   - Δrel = Δabs / mAP^rand × 100.
   - mAP^rand is the mean of the 2 P1 seeds.
   - Computed for mAP50, mAP50-95 and F1. P and R are given with their SI and rand values only.
4. **Per-class AP (LOSO):** the mean over 4 folds × 2 seeds. Δtwo comes from these class means. Δtwo is also given per fold (from the per-fold seed means), with the number of folds where it is positive, to show whether its sign is consistent.
5. **Model comparison:**
   - Per fold, the paired difference of the seed means (A − B).
   - Reported as the mean ± SD of the differences, plus the number of folds (of 4) whose order agrees with the order of the means.
   - No formal significance test, because n = 4.

Implementation notes:
- **Fold completeness:** a fold counts for a model only when both seeds are finished. The pairwise comparison uses the folds complete for both models. Until all 4 folds are complete, the output is marked "Preliminary" and the number of folds used is shown.
- **Output tables, by chapter section:**
  - 3.2: per-fold table; SI vs rand with Δabs and Δrel.
  - 3.3: per-class SI and P1, C_one, C_two, Δtwo, and Δtwo per fold.
  - 3.4: paired differences.
- **Output files:**
  - `results/summary_loso.md` (gitignored, like `summary_p1.md`);
  - `summary_loso.csv`, `summary_loso_folds.csv`, `summary_loso_per_class.csv`, `summary_loso_pairs.csv`.
- **Tested:**
  1. On the real data (fold s1): equal to an independent hand computation.
  2. On a synthetic 4-fold copy of s1, with the YOLO metrics shifted by −0.02, +0.01 and +0.03, in memory only: SI, SD over folds, paired differences and the agreeing-fold counts are as expected, including a 3/4 case. Δtwo per fold is unchanged under a uniform shift.
  3. The P1 output is byte-identical to before the change.

## 27. Chapter 3 section mapping and confusion matrix (30 Oct 2026)

**Section mapping (given by the user).** `summarise.py --protocol LOSO` now follows the manuscript's numbering. The numbers are unchanged from §26; only the order of the tables changed, and Teman per fold was added.

| Section | Content |
|---|---|
| 3.2 | LOSO per fold; SI ± SD over folds with the mean seed SD (mAP50, mAP50-95, P, R, F1); paired model comparison |
| 3.3 | Δabs and Δrel, P1 vs SI (mAP50, mAP50-95, F1) |
| 3.4 | Per-class AP50-95 (SI and P1); C_one, C_two and Δtwo; Δtwo per fold; Teman per fold; confusion matrix |
| 3.5 | FPS (placeholder until the final measurement) |

**Confusion matrix:** `scripts/confusion_matrix.py --protocol LOSO` (or `P1`).
- **Requested by the user:** per model, all folds combined, from the test predictions at the run's `conf_thr`.
- **Details proposed by me and confirmed by the user on 30 Oct 2026 (all three: column rule, background row and uncounted duplicates, summing the seeds):**
  - **Predictions:** the laptop test export (`runs/laptop_preds/<run>/preds_test_coco.json`), the same predictions behind the paper's mAP and P/R/F1. Detections with score ≥ the run's `conf_thr` are kept.
  - **Rows and columns:** each test frame has exactly one ground-truth box (the script checks this).
    - Row = true class.
    - Column = the class of the highest-scoring kept detection with IoU ≥ 0.5 to that box, or "missed" when there is none.
  - **Background row:** kept detections with IoU < 0.5 to the ground-truth box are counted there, under their class.
  - **Not counted:** further kept detections that overlap the ground truth (duplicates, second labels of the multi-label export). Their number is printed: LOSO s1 14 / 0 / 14, P1 2 / 6 / 0 (YOLO / RT-DETR / FRCNN).
  - **Pooling:** summed over the complete folds × 2 seeds (LOSO) or over the 2 seeds (P1), so each test image is counted once per seed.
  - **Figure:** rows normalised to the number of ground-truth images of the class; the background row shows counts.
- **Output:** `results/confusion/confusion_<p1|loso>_<model>.csv` (counts, tracked) and `.png` (figure; not tracked under the repo's results policy, regenerated by the script).

**First result (LOSO fold s1 only, 1052 image × seed pairs per model):**

| Model | Correct | Confused | Missed | Background detections |
|---|---|---|---|---|
| YOLO11m | 969 (0.921) | 34 | 49 | 29 |
| RT-DETR-L | 975 (0.927) | 37 | 40 | 8 |
| Faster R-CNN | 979 (0.931) | 39 | 34 | 24 |

- **YOLO11m, Cari (42):** labelled Keluarga in 40 % and Berangkat in 21 % of cases, missed in 26 %, correct in only 12 %. This matches its AP drop.
- **P1 (2 seeds, 632 pairs):** correct 0.973 / 0.992 / 0.991.
- **Figure caption (requested by the user, 30 Oct 2026):** every confusion-matrix figure carries a caption below the matrix. It names:
  - the folds × seeds combined (4 folds × 2 seeds when complete; until then the folds used, marked preliminary);
  - the matching rule (highest-scoring detection with IoU ≥ 0.5, at each run's validation-chosen `conf_thr`);
  - the row normalisation;
  - the number of duplicate detections not counted.
  The same text is in `results/confusion/captions_<p1|loso>.csv`, with the columns `folds`, `n_pairs` and `duplicates_not_counted`, ready for the manuscript.

## 28. Fold s4 moves to Google Colab; new Kaggle dataset version; FPS checkpoints (31 Oct 2026)

**Decided by the user (31 Oct 2026, for time: the user's exams start next week):**
- **Kaggle:** main-4 (notebook version 7) was running when this was decided. From main-5 on, Kaggle trains only P1 and folds s2–s3.
- **Colab:** fold s4 (6 runs) trains on Google Colab, on a Tesla T4 (the user's Google AI Plus subscription, 50 compute units).
- **FPS:** measured now, on the laptop, with the P1 seed-0 `best.pt` of all three models, instead of after all training.

**Plans** (`configs/kaggle_plan.yaml`):
- `main` (Kaggle) now holds the 24 runs of P1 and folds s1–s3.
- `colab_s4` holds the 6 runs of fold s4, in the same order (YOLO11m, RT-DETR-L, Faster R-CNN; seed 0, then seed 1).
- The 30 runs of the paper are `main` + `colab_s4`; `collect_results.py` reads both.
- Main-4 still runs the old plan from the previous dataset version; within its session limit it cannot reach fold s4.

**Kaggle dataset, new version** (`ajimaru12/bisindo-wl15`, uploaded 31 Oct 2026):
- Zip sha256: `6d7d7dedaebdc9d02b326376702a346d831e7c081c4ec347a02b025af398a77f`.
- **Data unchanged:** frames, labels, splits and weights (`manifest_sha256.txt`, 4,220 files) and the regenerated COCO ground truth (`coco_sha256.txt`) are byte-identical to the previous version.
- **Changed:** scripts and configs only.
- **Training path unchanged:** compared with the previous version's scripts:
  - In `coco_eval.py`, `load_gt`, `evaluate`, `save_dets` and the constants are unchanged (AST comparison). These are what `train_frcnn.py` uses for per-epoch validation and checkpoint selection.
  - `train_ultralytics.py` differs only in the `"yolo_multi_label": True` field of `eval_test.json`.
- **Test export after training:** Kaggle (from main-5) and Colab now export YOLO11m's test predictions multi-label (§25), flagged in `eval_test.json`. The laptop cross-check in `collect_results.py` already follows this flag. Paper numbers still come from the laptop export.
- **Notebook:** for main-5 the Kaggle notebook keeps `PLAN = "main"`; select the new dataset version and attach the main-4 output.

**Colab notebook** (`scripts/colab_notebook.ipynb`):
- **Checks:**
  - The GPU must be a Tesla T4 (`nvidia-smi`, first cell), and there must be at least 2 CPUs. Ultralytics uses min(CPUs, 2) data-loader workers, so 2 CPUs give the same 2 workers as on Kaggle.
  - The dataset zip on Google Drive must have the sha256 above, i.e. the same bundle as the Kaggle dataset version.
- **Run:** the unzipped dataset is local (`/content`). It runs the same `kaggle_session.py` with `BISINDO_PLATFORM=colab --plan colab_s4`. The session then:
  - installs the same pinned packages (torch build cu130 first, then cu128 / cu126, as on Kaggle);
  - uninstalls albumentations and stops if it is still importable;
  - checks the sha256 manifest and the COCO ground truth;
  - stops if the GPU is not the `kaggle_gpu` of `train.yaml` (Tesla T4);
  - runs with the same guards (`run_guards.py`, forbidden log lines) and the locked batch 16;
  - writes `runs/env_colab.json` (environment and differences from the laptop).
- **Persistence:** `runs/` is a link to `MyDrive/bisindo-colab/runs`. Checkpoints are written to Google Drive every epoch, and a new Colab session resumes from `last.pt` (early-stopping state restored, as on Kaggle).
- **Session limit:** `--max-hours` may go up to 23.5 h on Colab (11.5 h on Kaggle).
- **Export:** the last cell zips `runs/` without `last.pt` to `MyDrive/bisindo-colab/export/`. On the laptop: `kaggle_download.py --platform colab --zip <zip>` → `runs/colab/<timestamp>/`, with the CSVs copied to `results/colab/`.
- **Rehearsed on the laptop:**
  - Setup: `BISINDO_PLATFORM=colab`, no installs, `runs/` linked to a mock Drive folder holding a `sessions.csv`.
  - Checks passed: manifest OK, COCO identical, "runs/ already holds earlier sessions; nothing to copy", and `env_colab.json` written.
  - Stops: the session stopped at the GPU check ("got NVIDIA GeForce RTX 3050 Laptop GPU, but the batch is locked for Tesla T4").
  - `--max-hours` limits enforced: 24 is refused on Colab, 12 on Kaggle.
- **Differences to expect:**
  - **Environment:** the Colab driver, Python version and CPU count may differ from Kaggle's; they are recorded in `env_colab.json` and reported here when the first session has run.
  - **Results:** training is deterministic per platform (§22), but bit-identical results across Kaggle and Colab are not expected. Same GPU type, same code and package versions; the driver and CUDA build may differ.
  - **Training time:** the training times of fold s4 come from Colab, so they are not compared directly with the Kaggle runs. The P1 training-time table is all Kaggle.
- **Compute estimate:**
  - Fold s1 on the Kaggle T4 took 677 min of training and validation for its 6 runs, about 11.7 h with setup and test evaluation.
  - Colab bills a T4 at roughly 2 compute units per hour (the exact rate is shown in Colab's Resources panel), so about 24 of the 50 units if Colab is as fast as Kaggle.

**`runs.csv`:** a new column `platform` (kaggle / colab) after `seed`, taken from where the run was downloaded (`runs/kaggle/` or `runs/colab/`). All 13 finished runs are `kaggle`; no other value changed.

**Application and FPS checkpoints** (`configs/app.yaml`):
- **Weights:** the P1 seed-0 `best.pt` of the three models (Kaggle notebook version 6 download).
- **Default thresholds** (`conf`) = the runs' `conf_thr`: YOLO11m 0.81055, RT-DETR-L 0.41675, Faster R-CNN 0.75297.
- **`app_detect.py`:** starts with the model's own threshold and switches threshold with the model (keys 1/2/3); `--conf` overrides all three.
- **Tested:** headless on one test frame, all three models loaded.

**FPS result** (`scripts/measure_fps.py`, 31 Oct 2026, `results/fps.csv`, 9 rows):
- **Setup:** RTX 3050 Laptop GPU on AC power, Ultimate Performance plan, Chrome and Discord closed (checked), on the cooling pad (confirmed by the user).
- **Protocol:** 300 P1 test frames in memory, batch 1, FP16, conf 0.001, NMS IoU 0.7, max 100 detections; 50 warm-up and 500 timed iterations, 3 repeats per model.
- **Cooling:** idle temperature at start 57 °C. Before each repeat the script waited until the GPU was ≤ 62 °C (0–51 s); the warm-up then heated it to the start temperatures shown.

| Model | t_pred (ms) | FPS (eq. 18) | Network only (FPS) | pre / inf / post (ms) | GPU temperature |
|---|---|---|---|---|---|
| YOLO11m | 15.1 ± 0.9 | 66.2 ± 4.1 | 78.1 | 0.8 / 12.8 / 1.5 | 66–81 °C |
| RT-DETR-L | 33.0 ± 0.3 | 30.3 ± 0.3 | 32.9 | 1.9 / 30.4 / 0.6 | 73–87 °C |
| Faster R-CNN | 41.3 ± 0.1 | 24.2 ± 0.0 | 25.1 | 1.0 / 39.9 / 0.4 | 74–87 °C |

- **YOLO11m spread:** its larger SD comes from repeat 1 (61.9 FPS, against 69.9 and 66.7). Repeat 1 started right after loading, at the lowest temperature, so the cause is not heat.
- **Throttling:** the SM clock fell during each repeat, to 1,875 MHz for YOLO11m, 1,815 MHz for RT-DETR-L and 1,650 MHz for Faster R-CNN. The longest-running model throttles most.

## 29. Main session 4 (notebook version 7), LOSO fold s2 complete (31 Oct 2026)

**Session:** 10.79 h on a Tesla T4. It used the previous dataset version and the old plan, so YOLO11m's Kaggle test export was single-label. Downloaded to `runs/kaggle/20261031-1931/`.

**Runs:**
- **Finished** (epochs, ended by):
  - `rtdetr-l_loso_s2_seed0`: 59, patience;
  - `frcnn-r50fpn_loso_s2_seed0`: 77, patience;
  - `yolo11m_loso_s2_seed1`: 100, cap;
  - `rtdetr-l_loso_s2_seed1`: 63, patience;
  - `frcnn-r50fpn_loso_s2_seed1`: 94, patience;
  - `yolo11m_loso_s3_seed0`: 100, cap.
- **Paused:** `rtdetr-l_loso_s3_seed0` after epoch 61 (best epoch 53).
- **Fold s2 is now complete.** `runs.csv` holds 19 runs.

**Cross-checks** (`collect_results.py`): Kaggle's saved test predictions reproduce `eval_test.json` exactly for all 19 runs. The laptop export made the way Kaggle exported each run matches Kaggle's test mAP within 7.7 × 10⁻⁴.

**Kaggle dataset check before main-5:** `configs/kaggle_plan.yaml` was downloaded from the live Kaggle dataset version and is identical to the repo: `main` = 24 runs (P1, s1–s3), with no s4.

**Fold s2, mean of 2 seeds:**

| Model | mAP50 | mAP50-95 | P | R | F1 |
|---|---|---|---|---|---|
| YOLO11m | 0.948 ± 0.014 | 0.763 ± 0.005 | 0.905 ± 0.033 | 0.883 ± 0.016 | 0.879 ± 0.002 |
| RT-DETR-L | 0.975 ± 0.004 | 0.814 ± 0.017 | 0.912 ± 0.010 | 0.923 ± 0.001 | 0.904 ± 0.001 |
| Faster R-CNN | 0.936 ± 0.013 | 0.720 ± 0.002 | 0.914 ± 0.058 | 0.821 ± 0.011 | 0.841 ± 0.008 |

- **Faster R-CNN recall:** its validation-chosen thresholds in s2 are high (0.853 and 0.937), which gives the lowest recall.
- **Δtwo s2:** YOLO11m +0.068, RT-DETR-L +0.005, Faster R-CNN +0.029. In s1 they were +0.008, −0.052 and −0.050, so the sign is not yet consistent across folds for RT-DETR-L and Faster R-CNN.

## 30. Main session 5 (notebook version 8): all Kaggle runs finished, LOSO fold s3 complete (1 Nov 2026)

**Session:** 10.18 h on a Tesla T4, with the 31 Oct 2026 dataset version (plan `main` = P1 + s1–s3). Downloaded to `runs/kaggle/20261101-0806/`.
- **Runs finished** (epochs, ended by):
  - `rtdetr-l_loso_s3_seed0`: 82, patience (resumed from epoch 61);
  - `frcnn-r50fpn_loso_s3_seed0`: 100, cap;
  - `yolo11m_loso_s3_seed1`: 100, cap;
  - `rtdetr-l_loso_s3_seed1`: 87, patience;
  - `frcnn-r50fpn_loso_s3_seed1`: 100, cap.
- **All 24 Kaggle runs are finished.** `runs.csv` holds 24 runs; fold s4 (6 runs) is on Colab.
- **Kaggle export:** `yolo11m_loso_s3_seed1` is the first run scored on Kaggle with the multi-label YOLO export (`yolo_multi_label` in `eval_test.json`). Its Kaggle pycocotools and Ultralytics test mAP50-95 are close (0.6483 vs 0.6465), as expected.
- **Cross-checks:** all 24 runs pass. The largest laptop-vs-Kaggle test mAP difference is 9.1 × 10⁻⁴.

**Fold s3 (test = signer3), mean of 2 seeds:**

| Model | mAP50 | mAP50-95 | P | R | F1 |
|---|---|---|---|---|---|
| YOLO11m | 0.872 ± 0.021 | 0.662 ± 0.019 | 0.788 ± 0.087 | 0.800 ± 0.004 | 0.755 ± 0.040 |
| RT-DETR-L | 0.900 ± 0.034 | 0.699 ± 0.019 | 0.855 ± 0.005 | 0.840 ± 0.064 | 0.829 ± 0.037 |
| Faster R-CNN | 0.880 ± 0.016 | 0.643 ± 0.016 | 0.737 ± 0.051 | 0.788 ± 0.063 | 0.737 ± 0.010 |

- **The hardest fold so far for all three models.**
- **One-handed glosses:** they drop most. Keluarga has AP50-95 0.18 / 0.27 / 0.15 (YOLO / RT-DETR / FRCNN); its frames are mostly missed or labelled Cari or Berangkat. Makan is 0.36–0.43. Saya (YOLO 0.32) is labelled Terima kasih (YOLO) or Pagi (FRCNN).
- **Two-handed glosses** stay at 0.72–0.92.
- **Δtwo s3:** −0.237 / −0.212 / −0.172, so two-handed glosses are easier in this fold.
- **Thresholds:** the validation-chosen thresholds vary widely between seeds (YOLO 0.419 / 0.225, RT-DETR 0.895 / 0.308, FRCNN 0.464 / 0.821). This explains the large seed SD of P and F1 in s3.

## 31. Colab session (fold s4): all 30 runs finished; final LOSO summary (1 Nov 2026)

**Session:** Google Colab, one Tesla T4 (driver 580.82.07), plan `colab_s4`, 13.39 h (31 Oct 14:55 – 1 Nov 04:19 UTC), batch 16, one session without interruption. The export zip `bisindo_colab_runs_20261101-0419.zip` was imported with `kaggle_download.py --platform colab --zip` to `runs/colab/20261101-1150/`; small files are in `results/colab/`.

**Platform per run** (column `platform` in `runs.csv`):
- **Kaggle, 24 runs:** P1 (6) and LOSO folds s1, s2, s3 (6 each).
- **Colab, 6 runs:** LOSO fold s4.
- Training times of s4 come from Colab and are not compared directly with the Kaggle runs.

**Colab environment** (`results/colab/env_colab.json`):
- **Same as Kaggle:** Python 3.13.16 (Kaggle 3.13.15), torch 2.14.1+cu130, torchvision 0.29.1+cu130, Ultralytics 8.4.190, NumPy 2.4.6; albumentations not importable; same dataset zip (sha256 checked).
- **Different, deviation from the pinned set:** `cv2.__version__` in the Colab training process was **4.14.0**, while the laptop freeze and Kaggle use **5.0.0** (`opencv-python==5.0.0.93`). `kaggle_session.py` did install `opencv-python==5.0.0.93`, but the OpenCV package preinstalled on Colab was the one imported. The difference check compares package names in `pip freeze` only, so it did not flag this.
  - OpenCV is used in training for JPEG decoding, resizing (LetterBox) and the HSV / affine augmentation.
  - The evaluation is not affected: all test and validation numbers are the laptop re-export (OpenCV 5.0.0).
  - Reported in the manuscript as an environment difference of fold s4.
- **Minor helper packages differing from the laptop:** charset-normalizer, contourpy, filelock, fsspec, markupsafe, networkx, python-dateutil, ultralytics-platform, urllib3.

**Checks:**
- **Cross-checks:** all 30 runs pass. The largest laptop-vs-platform test mAP50-95 difference is 1.0 × 10⁻³ (`frcnn-r50fpn_loso_s4_seed0`); limit 0.002.
- **YOLO export:** both YOLO s4 runs were scored on Colab with the multi-label export (`yolo_multi_label: true` in `eval_test.json`). The laptop multi-label export gives 0.5597 / 0.5663 against Colab's 0.5601 / 0.5664.

**Fold s4 runs** (epochs, best epoch, ended by; patience = epoch − best ≥ 20):
- `yolo11m_loso_s4_seed0`: 100, 96, cap;
- `rtdetr-l_loso_s4_seed0`: 66, 46, patience;
- `frcnn-r50fpn_loso_s4_seed0`: 95, 75, patience;
- `yolo11m_loso_s4_seed1`: 100, 94, cap;
- `rtdetr-l_loso_s4_seed1`: 100, 86, cap;
- `frcnn-r50fpn_loso_s4_seed1`: 100, 96, cap.

**Stopping over all 30 runs:**

| Model | Ended at the 100-epoch cap | Ended by patience |
|---|---|---|
| YOLO11m | 9 | 1 |
| RT-DETR-L | 2 | 8 |
| Faster R-CNN | 7 | 3 |
| **All** | **18** | **12** |

**Fold s4 (test = signer4), mean ± SD of 2 seeds:**

| Model | mAP50 | mAP50-95 | P | R | F1 |
|---|---|---|---|---|---|
| YOLO11m | 0.814 ± 0.006 | 0.563 ± 0.005 | 0.855 ± 0.046 | 0.739 ± 0.024 | 0.772 ± 0.002 |
| RT-DETR-L | 0.853 ± 0.006 | 0.621 ± 0.005 | 0.820 ± 0.080 | 0.817 ± 0.057 | 0.799 ± 0.004 |
| Faster R-CNN | 0.840 ± 0.007 | 0.550 ± 0.010 | 0.834 ± 0.025 | 0.784 ± 0.029 | 0.791 ± 0.006 |

- **The hardest fold for all three models.**
- **Two-handed glosses drop most in s4.** Teman has AP50-95 0.28 / 0.37 / 0.31 (YOLO / RT-DETR / FRCNN) and AP50 0.42 / 0.51 / 0.45; 48–64 % of its test frames have no detection with IoU ≥ 0.5 at conf_thr. Motor drops for YOLO (0.32) and is labelled Bagaimana in 19–23 % of frames. Bagaimana is labelled Rumah in about 20 %.
- **Δtwo s4:** +0.187 / +0.098 / +0.086, the opposite sign of s3.

**Final LOSO summary** (4/4 folds; `summarise.py --protocol LOSO`, no longer preliminary):
- **SI:** mAP50-95 0.683 ± 0.091 (YOLO), 0.729 ± 0.087 (RT-DETR), 0.662 ± 0.084 (FRCNN); F1 0.824 / 0.865 / 0.822.
- **Δabs:** mAP50-95 0.180 / 0.157 / 0.170 (Δrel 20.8 / 17.8 / 20.5 %).
- **Δtwo:** overall +0.007 / −0.040 / −0.027; per fold > 0 in 3/4, 2/4 and 2/4 folds.
- Confusion matrices for LOSO and P1 were regenerated.

**FPS in the summary:** `summarise.py` now writes section 3.5 from the final rows of `results/fps.csv` (mean ± SD over the 3 repeats, the three stages, network-only FPS, GPU temperature range) and `results/summary_fps.csv`. The values are those of section 28.

## 32. Application display FPS and resting-hand check with the final checkpoints (1 Nov 2026, section 3.6)
- **`configs/app.yaml`:** weight paths now point to the download folder that `runs.csv` names (`runs/kaggle/20261101-0806/`, P1 seed 0). The files are byte-identical to the notebook version 6 download used before (same SHA-256), so the section 3.5 FPS stands. Thresholds unchanged (= `conf_thr`: 0.81055 / 0.41675 / 0.75297).
- **`scripts/app_fps.py` (new):** runs the `app_detect.py` loop unchanged (MSMF 1280×720, newest-frame thread, `predict()` with app settings, drawing, window). Per model 3 × 60 s after 5 s warm-up; display FPS = (frames − 1) / time span; the camera's delivery rate in the same window is logged. Before each repeat it waits for the GPU to cool to idle + 5 °C, capped at 180 s (the first attempt, with a 600 s cap, was killed by the 10 min limit of the background shell after YOLO11m #1; nothing was written). Outputs: `results/app_fps.csv`, `results/app_rest_check.csv`, `runs/app/final/`.
- **Conditions:** charger on, Ultimate Performance plan, Roblox and Chrome closed; **Discord was still running** (noted in the CSV rows). Idle GPU 62 °C at start (target 67 °C, often not reached within 180 s); GPU 68–87 °C during the runs.
- **Display FPS, mean ± SD of 3 repeats (camera rate in brackets):** YOLO11m 19.8 ± 7.7 (19.8 ± 7.8), RT-DETR-L 19.0 ± 3.1 (25.4 ± 8.0), Faster R-CNN 17.8 ± 0.3 (29.1 ± 1.6).
  - The camera dropped from 30 to about 15 FPS in YOLO11m #2–#3 and RT-DETR-L #1 (auto-exposure in dim light), and display FPS followed it. With the camera at 30 FPS: YOLO11m 28.7 (camera-bound), RT-DETR-L 20.8 and 20.8, Faster R-CNN 17.5–18.0 (model-bound).
- **Resting hands, 20 s per model at threshold 0.05:** labelled frames at the app threshold: YOLO11m 0/599 (max score 0.59, "Makan"), RT-DETR-L 2/384 (max 0.70, "Makan", a small box on the chin), Faster R-CNN 0/323 (no detection ≥ 0.05).
  - **Limit:** with this camera framing the hands at rest were below the image edge, so this checks hands out of view, not lowered hands visible at waist height as in the 28 Oct finding ("Teman" 0.78 on `signer0_label9_sample1`). That finding is not contradicted.
  - Share of signing-phase frames with a label shown: YOLO11m 0.01, RT-DETR-L 0.55, Faster R-CNN 0.62 (YOLO11m's threshold 0.81 is rarely reached on webcam frames).

## 33. Supplementary: mirrored input (1 Nov 2026; not part of the 30-run protocol)
- **Question (user):** would a mirrored webcam image (or a left-handed signer) give the same result? The webcam itself is not mirrored: a raised right hand appears on the image's left, as in WL-BISINDO, where one-hand boxes lie left of centre in 57–88 % of each signer's frames.
- **`scripts/flip_test.py`:** the 316 P1 test frames flipped horizontally with flipped ground truth (`runs/flip_test/`, originals untouched); the P1 `best.pt` of both seeds evaluated exactly as the paper export (multi-label, conf 0.001, NMS 0.7, FP16, pycocotools); P/R/F1 at each run's `conf_thr`. The original-orientation rows reproduce `runs.csv` (asserted). Outputs `results/flip_test.csv`, `results/flip_test_per_class.csv`.
- **mAP50-95 original → mirrored (seed 0 / seed 1):** YOLO11m 0.863 → 0.212 / 0.862 → 0.237; RT-DETR-L 0.883 → 0.725 / 0.889 → 0.737; Faster R-CNN 0.832 → 0.361 / 0.832 → 0.336. F1 at conf_thr: YOLO11m 0.19 / 0.21, RT-DETR-L 0.89 / 0.90, Faster R-CNN 0.45 / 0.38.
- **Class-agnostic mAP50 (localisation only) mirrored:** YOLO11m 0.34 / 0.41, RT-DETR-L 0.95 / 0.97, Faster R-CNN 0.64 / 0.60, so YOLO11m and Faster R-CNN mostly fail to find the hand, not only to name it.
- **By group (AP50-95, mean of seeds), original → mirrored:** C_one YOLO11m 0.86 → 0.05, RT-DETR-L 0.88 → 0.70, Faster R-CNN 0.83 → 0.24; C_two 0.87 → 0.69, 0.90 → 0.83, 0.84 → 0.66. Two-hand signs (one box around both hands, roughly symmetric) suffer much less.
- Limitation: P1 test frames (signers seen in training); a mirrored LOSO test was not run.

### 32a. Rerun with brighter light and hands visible (1 Nov 2026, 20:38–21:00)
- **Conditions:** brighter light; Roblox and Chrome closed, **Discord still running**; idle GPU 61 °C; GPU 66–88 °C during the runs. The first launch attempt did not start Python (bad quoting in the detached command; empty logs, nothing written); relaunched from a script file.
- **Display FPS, 3 × 60 s:** YOLO11m 28.5 / 30.0 / 30.0 = **29.5 ± 0.9** (camera 29.5 ± 0.8, camera-bound); RT-DETR-L 21.4 / 19.5 / 21.4 = **20.8 ± 1.1** (camera 30.0). Faster R-CNN was not rerun: its first-session repeats had the camera at 27–30 FPS, **17.8 ± 0.3**.
- **Values for section 3.6:** YOLO11m 29.5 ± 0.9, RT-DETR-L 20.8 ± 1.1 (rerun), Faster R-CNN 17.8 ± 0.3 (first session). The first-session YOLO11m and RT-DETR-L rows stay in `results/app_fps.csv` (camera at 15–16 FPS), marked by their `note`.
- **Resting hands, 20 s per model** (person standing, arms hanging at the sides, hands at hip height in view; face above the image edge): YOLO11m 0/602 frames labelled (max 0.32); RT-DETR-L **196/408 (48 %)**, "Bagaimana" 110 and "Keluarga" 86, median top score 0.36, p90 0.87, max 0.95; Faster R-CNN **338/346 (98 %)**, all "Bagaimana", median 0.98, max 1.00. The box covers both hands and the hips. Examples: `runs/app/final/rest_<model>.jpg`.
- This reproduces the 28 Oct finding for RT-DETR-L and Faster R-CNN: hands at rest are labelled with a gloss (here the two-hand C_two sign Bagaimana), because rest/transition frames were excluded from training (criterion a). YOLO11m stays silent mainly because its threshold (0.81) is high; it also labels only 1 % of signing frames on the webcam. Training data are not changed (CLAUDE.md).

## 34. Live per-sign test mode for Table 8 (5 Nov 2026)
- **Purpose (user):** a live test of RT-DETR-L on new signers, 15 signs × 10 trials per participant, for Table 8.
- **New files:**
  - `scripts/live_test.py`, started with `python scripts/app_detect.py --live-test`; the normal application is unchanged (only this flag was added);
  - `scripts/live_rule.py` (success rule) and `scripts/test_live_rule.py` (7 unit tests, all pass);
  - `scripts/summarise_live_test.py`.
- **Locked settings:**
  - RT-DETR-L P1 seed-0 `best.pt` from `configs/app.yaml`; the script stops if it differs from the `runs.csv` checkpoint;
  - threshold **0.41675** = `conf_thr` of that run (selected on validation), also checked against `app.yaml`;
  - FP16; camera not mirrored; MSMF 1280×720.
- **Input size (user decision B, 5 Nov 2026):** `predict()` exactly as the application, mAP evaluation and FPS runs (imgsz 640).
  - The measured tensor for a 1280×720 frame is **1×3×640×640**: Ultralytics' RT-DETR predictor (8.4.190) uses `scale_fill=True`, so the frame is resized to 640×640 without padding and without keeping the aspect ratio.
  - Minimal letterbox (e.g. 640×384) applies to YOLO11m only; the plan message to the user had said 640×384 for RT-DETR-L and was corrected.
  - The shape is written to `session_{ID}.json`.
- **Display (user decision A, 5 Nov 2026):** during the 3 s recording only the camera image and "MEREKAM" are shown, with no boxes or labels; the result is shown afterwards.
- **Trial procedure:**
  - target gloss with "posisi netral" → SPACE → 3 s countdown (model not run) → 3 s recording with the model on every frame;
  - per frame: the top detection's label if its score ≥ threshold, else "none", plus the score. The model runs at conf 0.01 and the threshold is applied afterwards; RT-DETR's top detection does not depend on conf, so the score of below-threshold frames is kept for audit.
- **Rule:**
  - SUCCESS when the target is the label on ≥ 5 consecutive frames; otherwise MISS;
  - label_on_miss = the most frequent non-target label (tie: first to appear), or "none" when no other label appeared. This includes misses where only the target appeared, on fewer than 5 consecutive frames; `max_consecutive_target` shows those.
- **Order:** 150 trials, shuffled with a random seed stored in the session file, or sequential.
- **Repeat (R):** the last finished trial's row is kept with `repeated=1` and the trial is run again; `frames_{ID}.csv` has an extra `attempt` column so repeated attempts stay separate.
- **Resume:** the same ID continues with the first trial without a valid row, using the stored order.
- **Saving:** every trial is written when it finishes. No video is stored.
- **Outputs:** `results/live_test/` with `trials_{ID}.csv`, `frames_{ID}.csv` and `session_{ID}.json`. The session file holds:
  - git commit and dirty flag, script SHA-256, order seed and order;
  - threshold, checkpoint and its SHA-256;
  - camera backend, index and resolution, tensor shape, and package versions.
- **Simulation** (`runs/live_test_sim/`, not in `results/`):
  - SIM: Cari × 2, automatic start, no signing, both misses;
  - SIM2: 2 signs × 2, sequential, used to check resume (a missing trial is redone) and the R flag (the flagged trial is redone and its old row kept);
  - in SIM2, Roblox and Chrome were running and the GPU was at 84 °C, and the recording reached only 8–14 FPS instead of about 19.
- **Note for the paper:** the rule counts frames, so its time length depends on the frame rate: 5 frames ≈ 0.25 s at 20 FPS but ≈ 0.6 s at 8 FPS. `mean_fps` is stored per trial. Real sessions should run with other GPU programs closed.

### 34a. P01 session and the first-attempt rule (5 Nov 2026)
- **P01:** hearing signer, 1 m, lighting "TERANG", random order (seed 358786899), one session, 150 trials (10 per sign); recording 16.7–22.3 FPS (median 19.1); threshold 0.41675; tensor 1×3×640×640.
- **Repeats:**
  - 18 trials were repeated with key R (26 repeated attempts in all); 22 of the 26 repeated attempts were misses;
  - no reason is recorded, so procedural repeats cannot be told apart from repeats after a model failure.
- **Decision (user, 5 Nov 2026):** Table 8 reports the **first attempt of every trial only** (the first row of each `trial_no` in file order, whatever its flag); repeated attempts are not counted. `summarise_live_test.py` changed accordingly; it also prints, for transparency only, the last-attempt count (149/150, not used).
- **Table 8, P01 (first attempts):**
  - **135/150 successful (90.0 %)**; all 15 misses showed a wrong label, none was "not detected";
  - most missed: Saya 6/10 (shown as "Terima kasih" 4×), Bagaimana 7/10 ("Motor" 3×), Berangkat 7/10 ("Hari" 2×, "Keluarga" 1×);
  - two-hand signs 37/40, one-hand signs 98/110.
- Output: `results/live_test/table8.csv`, `table8.md`.
