# Multi-part car damage + per-part severity (6 parts × high/low)

Parts: front, rear, door, corner, window, other. One image can damage several parts, each with its own severity.
Target per image: 6 values in {none, low, high}. Model = shared backbone + 6×3 output, masked focal loss.

## Pipeline
1. Get data (below).      2. `build_labels.py` → `labels.csv` (-1 = unknown, 0 none, 1 low, 2 high).
3. Merge CSVs, set a `group` column (same car / near-duplicate → same group) to avoid leakage.
4. `train.py` (main model + baselines + ablations).      5. `predict_video.py` for video demos.

## Datasets
| Dataset | Gives you | Use |
|---|---|---|
| CarDD (USTC, 4k imgs, 6 damage types, masks) | damage masks, no parts | main real-image source; parts via part-segmenter |
| Car Parts & Car Damages (Humans in the Loop, Kaggle) | part polygons (998 imgs) + damage polygons (814 imgs) — different images | train part segmenter; damage polygons for extra data |
| VehiDE (Kaggle/UIT) | damage types, polygons | extra damage data; no parts |
| SYNDCAR (Mendeley, synthetic) | 28 parts + 4 damage classes, YOLO format, same images | only source with parts and damage together (synthetic) |
| CrashCar101 (synthetic, 101k) | part + damage masks | optional pre-training |
| CAD 100K (arXiv 2604.09023) | multi-task incl. severity, per its abstract | check whether released and its license |

Video: no public labelled car-damage video dataset was found. Use frames only for qualitative tests,
or hand-label a small set. Check licence/terms before using web or YouTube footage.

## Things that would sink a paper (fix these)
- Part-only datasets (HITL parts) show undamaged parts. Do not label them "damaged"; use them only to train the part segmenter.
- Severity is not annotated in these sources. The rules in `build_labels.py` are heuristics: hand-label ≥300 images, report agreement (Cohen's κ), and run a threshold-sensitivity table.
- Pseudo-label noise: report part-segmenter quality (mIoU) and test on a hand-verified subset.
- Synthetic ≠ real: report real-only test results separately.
- Split by group, report 3 seeds with mean±std, and set thresholds on val only.
- Mapping to 6 classes is a design choice: list every source label → class in an appendix, and check `unmapped_names()`.

## Metrics
Per-part presence F1 / AP, per-part severity macro-F1 (on truly damaged parts), high-severity recall, exact match (part set; part+severity).

## Suggested experiments
Baselines ResNet-50 / EfficientNet-B1 / ConvNeXt-S; ablate `--loss ce|focal`, `--sampler on|off`, real-only vs +synthetic; optional frozen VLM linear probe.

## Setup & smoke test
```bash
pip install -r requirements.txt
python smoke_test.py
```

## Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deba75/Car-Damage/blob/main/colab.ipynb)

Open [`colab.ipynb`](colab.ipynb) in Google Colab (either via the badge above after pushing to GitHub, or by uploading `colab.ipynb` in Colab via **File → Upload notebook**), select a **GPU** runtime (**Runtime → Change runtime type → T4 GPU**), and run the cells top-to-bottom. Checkpoints and metrics are saved to Google Drive (`/content/drive/MyDrive/cardamage6/runs`).
