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
