# Model card: FetalAI HC18 U-Net

> **Academic research prototype, not a medical device.** Not clinically validated. Must not be used
> for diagnosis, screening decisions or pregnancy care.

## Overview

| | |
|---|---|
| Task | Binary segmentation of the fetal head in 2-D ultrasound, then head-circumference (HC) measurement |
| Architecture | U-Net, 4 down/up levels, base 32 channels, BatchNorm, 7,762,465 parameters (`ml/model.py`) |
| Input | Grayscale image resized to 256 × 256, intensities / 255 |
| Output | Per-pixel head probability; mask = p ≥ 0.5, largest connected component |
| HC method | Mask resized back to original resolution → `cv2.fitEllipse` on its contour → perimeter by Ramanujan's second approximation → × pixel spacing (mm/px) |
| Checkpoint | `outputs/hc18_unet_best.pth` (epoch 50), not committed; regenerate with `python -m ml.train` |
| Owners | Nachiketha K G (training and evaluation), Pujari Varshitha (data pipeline) |

## Data

HC18 grand-challenge training set (van den Heuvel et al., *PLoS ONE* 13(8): e0200412, 2018;
doi:10.5281/zenodo.1327317): 999 images with ellipse annotations and reference HC. See
[`DATASET.md`](DATASET.md).

Split (`data/splits/hc18_split.csv`, seed 42): **by examination group** (filename prefix, 806 groups),
so repeat scans of the same fetus never cross splits; HC-stratified.

| Split | Images | Groups | Used for |
|---|---|---|---|
| train | 696 | 565 | weight updates |
| val | 152 | 120 | early stopping, LR schedule, checkpoint selection |
| test | 151 | 121 | **final evaluation only** (results below) |

The official HC18 test set (335 images) has no public labels and is used only for qualitative demos.

## Training

Adam (lr 1e-3, halved on a validation-Dice plateau of 5 epochs), BCE + soft Dice loss, batch 16, mixed
precision (AMP), augmentation (horizontal flip, rotation ±15°, scale 0.9–1.1, brightness/contrast),
early stopping with patience 15. NVIDIA RTX 4050 Laptop GPU (6 GB): 65 epochs, **11.4 min**.
Best validation Dice **0.9760** (IoU 0.9539) at epoch 50. Curves: `docs/screenshots/training_curves.png`.

## Results on the held-out test split (n = 151)

Produced by `python -m ml.evaluate` → `outputs/test_metrics.json`, `outputs/test_predictions.csv`.
Segmentation metrics are computed at the **original** image resolution against the filled reference ellipse.

| Metric | Mean ± SD | Median | Range |
|---|---|---|---|
| Dice | 0.973 ± 0.028 | 0.979 | 0.731 – 0.992 |
| IoU | 0.948 ± 0.046 | 0.960 | 0.576 – 0.984 |
| HC error, predicted − reference (mm) | −0.10 ± 2.95 | −0.33 | −7.67 – +11.58 |

* **HC mean absolute error: 2.18 mm** (mean absolute percentage error 1.48 %).
* Absolute HC error percentiles: P50 1.80 mm, P90 4.66 mm, P95 5.93 mm.
* Bland–Altman bias −0.10 mm, 95 % limits of agreement −5.89 to +5.69 mm.
* Ellipse-fit failures: 0 / 151.

Figures: `docs/screenshots/test_hc_agreement.png` (scatter + Bland–Altman),
`docs/screenshots/test_examples.png` (worst / median / best),
`docs/screenshots/official_test_set_qualitative.png`.

### Measurement-code check

Fitting ellipses to the 999 **reference** annotations reproduces the CSV reference HC with an MAE of
0.009 mm (filled-mask route: 0.078 mm). The HC error above therefore comes from the segmentation, not
from the geometry code (`tests/test_geometry.py`).

## Limitations

* **Small, early-gestation heads are the weakest case.** The two worst test images (`022_2HC`,
  `022_HC`, HC ≈ 65 mm) leak into surrounding tissue (Dice 0.73 / 0.86, HC error +11.6 / −5.2 mm).
* Trained and tested on one public dataset (HC18) only. No external validation; performance on other
  machines, sites, settings or populations is unknown.
* The model assumes a correct trans-thalamic plane. It does **not** check the plane; a standard-plane
  classifier is Phase-2 work.
* Pixel spacing must be supplied (HC18 CSV or scanner DICOM metadata). A wrong spacing scales HC proportionally.
* Quality indicators in the API (mean confidence, uncertain-boundary fraction, border contact, ellipse-fit
  ratio) are **heuristics**. On the test split they correlate only weakly with HC error (Spearman ≈ 0.1);
  the uncertain-boundary flag (> 5 % of pixels with 0.3 < p < 0.7) fired on 3 / 151 images, including the
  worst case. Thresholds were set before the test evaluation. They are not calibrated uncertainty
  (calibration / MC-dropout is Phase-2).
* Image-level resize to a square distorts the aspect ratio; this is undone before measurement.

## Bug history

An earlier version reported "Dice 1.0". That number was **invalid**: every training mask was empty
because of a resize-before-fill bug. See [`evidence/README.md`](evidence/README.md).
