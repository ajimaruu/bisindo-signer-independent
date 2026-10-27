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
  - signer4 stands farther from the camera, so her hands are small. Some mid-video frames with a visible hand were missed (for example `s4_cari_v01_f0022`). These need a box drawn by hand.
  - Videos with no hand detected in any of the 5 frames: `s4_cari_v02`, `s4_cari_v05`, `s4_makan_v03`.
  - C_one frames with two hands detected (the highest one was used): 101.
- **Rejected second pass:** detection on the central half of the frame, upscaled 2×.
  - It recovered 136 of the 670 no-hand frames, but most of the recovered boxes were resting hands at the waist, and in several frames it boxed the resting hand instead of the moving one.
  - It would add boxes to delete rather than save work, so it was not applied. Pre-annotation stays single-pass as piloted.
