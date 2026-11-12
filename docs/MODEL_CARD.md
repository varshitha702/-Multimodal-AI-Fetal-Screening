# Model cards: HC18 U-Net and standard-plane gate

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
* The U-Net itself assumes a trans-thalamic plane. Since Feature A, a separate standard-plane classifier
  checks this first (see the second model card below); the U-Net still segments whatever it is given.
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

---

# Model card: standard-plane quality gate (Feature A)

> Same disclaimer: research prototype, not a medical device.

## Why

HC is defined on the **trans-thalamic (TT)** plane. Commercial machines measure HC automatically on whatever
image they are given; FetalAI first checks *which plane the image is* and warns (and withholds the
measurement) when it is not a confirmed TT plane. Owners: Pujari Varshitha (data), Nachiketha K G (model),
Vishwas K (API integration), Rakshitha V (UI).

## Overview

| | |
|---|---|
| Task | (1) plane: Fetal brain / abdomen / femur / thorax / Maternal cervix / Other; (2) brain sub-plane: Trans-thalamic / Trans-cerebellum / Trans-ventricular / Other |
| Architecture | ImageNet EfficientNet-B0 backbone (torchvision) + two linear heads, dropout 0.3 (`ml/planes.py`) |
| Input | Grayscale, letterboxed to **320 × 320**, replicated to 3 channels, ImageNet normalisation |
| Gate score | calibrated P(brain) × P(TT \| brain); **pass if ≥ 0.138** |
| Calibration | One temperature per head fitted on validation: T = 1.62 (plane), 2.28 (sub-plane) |
| Threshold rule | Largest threshold with validation TT recall ≥ 0.95 (fixed **before** test evaluation) |
| Checkpoint | `outputs/plane_classifier_best.pth` (epoch 16), not committed; `python -m ml.train_planes --size 320 --batch-size 32` |
| Latency (one 800×540 image, laptop) | 31 ms CPU, 61 ms RTX 4050 (segmentation: 81 ms CPU, 61 ms GPU) |

## Data and training

FETAL_PLANES_DB (Burgos-Artizzu et al., *Sci Rep* 2020): official test set kept untouched (5,271 images, 896
patients); official training set split by `Patient_num` into train 6,091 / val 1,038 images (762 / 134
patients). See [`DATASET.md`](DATASET.md). Loss = cross-entropy (plane) + cross-entropy (sub-plane, brain
images only). AdamW lr 3e-4, one-cycle schedule, batch 32, AMP, GPU augmentation, early stopping on validation
mean(plane macro-F1, sub-plane macro-F1): 22 epochs, **16.5 min** on the RTX 4050; best epoch 16 (validation
plane macro-F1 0.954, sub-plane macro-F1 0.785).

**Model selection (ablation).** Two input sizes were trained with the same recipe; the one with the better
**validation** score was kept, before either was evaluated on test. For completeness both test results:

| Input | Val score | Test plane acc / macro-F1 | Test sub-plane acc / macro-F1 | TT accepted | non-TT rejected | HC18 accepted |
|---|---|---|---|---|---|---|
| 224 px | 0.860 | 0.935 / 0.930 | 0.816 / 0.719 | 95.3 % | 93.2 % | 63.9 % |
| **320 px (deployed)** | **0.869** | **0.940 / 0.934** | **0.827 / 0.721** | **95.7 %** | **93.8 %** | **60.3 %** |

(224 px run: `outputs/plane_ablation_224px/`.)

## Results on the official FETAL_PLANES_DB test set

`python -m ml.evaluate_planes` → `outputs/plane_metrics.json`, `outputs/plane_test_predictions.csv`.

**Plane (n = 5,271): accuracy 0.940, macro-F1 0.934.** The dataset paper's best CNN (DenseNet-169) reports
a 6.2 % top-1 error on the same six classes, i.e. this model is in the same range.

| Plane | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Fetal brain | 0.987 | 0.991 | 0.989 | 1,472 |
| Fetal abdomen | 0.852 | 0.961 | 0.903 | 358 |
| Fetal femur | 0.870 | 0.891 | 0.880 | 524 |
| Fetal thorax | 0.907 | 0.956 | 0.931 | 660 |
| Maternal cervix | 0.995 | 0.995 | 0.995 | 645 |
| Other | 0.933 | 0.875 | 0.903 | 1,612 |

**Brain sub-plane (test brain images, n = 1,472): accuracy 0.827, macro-F1 0.721.**

| Sub-plane | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Trans-thalamic | 0.835 | 0.881 | 0.858 | 765 |
| Trans-cerebellum | 0.870 | 0.885 | 0.877 | 339 |
| Trans-ventricular | 0.822 | 0.732 | 0.774 | 302 |
| Other (brain) | 0.431 | 0.333 | 0.376 | 66 |

The paper reports a 25.0 % error for its best CNN on the three standard brain planes; our head also has an
"Other" class, so the numbers are not directly comparable. The main confusion is TV → TT (72 of 302 TV images).

**Calibration (temperature scaling, fitted on validation, measured on test):**

| Head | ECE before | ECE after | NLL before | NLL after |
|---|---|---|---|---|
| Plane | 0.035 | **0.014** | 0.245 | 0.197 |
| Brain sub-plane | 0.120 | **0.033** | 0.780 | 0.498 |

**Gate (threshold 0.138):**

| Test category | n | Passed |
|---|---|---|
| Brain: Trans-thalamic | 765 | **95.7 %** |
| Brain: Trans-ventricular | 302 | 52.3 % |
| Brain: Trans-cerebellum | 339 | 22.4 % |
| Brain: Other | 66 | 57.6 % |
| Abdomen / femur / thorax / cervix / other | 3,799 | 0.0–0.4 % (non-brain rejected 99.8 %) |

All non-TT images: 93.8 % rejected. The gate reliably stops **wrong-organ** images, but lets through about half
of the trans-ventricular and "other" brain planes, the cases closest to TT.

Figures: `docs/screenshots/plane_confusion.png`, `plane_reliability.png`, `plane_gate_scores.png`,
`plane_hc18_rejected.png`.

## HC18 acceptance (reported honestly)

All 1,334 HC18 images (999 training + 335 official test) are meant to be HC-measurement images, so ideally
all would pass. **60.3 % pass** (training set 60.9 %, test set 58.8 %). 94.5 % are recognised as fetal brain;
the loss is mostly TT vs TV (535 called TV vs 558 TT) and early gestation:

| HC18 reference HC (training set) | n | Recognised as brain | Accepted |
|---|---|---|---|
| < 100 mm (≈ ≤ 16 weeks) | 184 | 72.3 % | **31.5 %** |
| 100–150 mm | 61 | 96.7 % | 85.2 % |
| 150–200 mm | 538 | 99.8 % | 57.3 % |
| 200–250 mm | 67 | 100 % | 73.1 % |
| ≥ 250 mm | 149 | 98.7 % | 94.6 % |

Reasons: (1) FETAL_PLANES_DB covers 18–40 weeks only, while HC18 has many smaller, earlier heads;
(2) different machines and country, and HC18's black anonymisation boxes; (3) TT vs TV is a fine distinction
even inside FETAL_PLANES_DB. The threshold was **not** re-tuned on HC18. For demos on HC18 the API's
`override_plane_check` (UI: "Measure anyway") measures anyway and keeps the warning on the report.

## Limitations of the gate

* Trained on one dataset from two hospitals; domain shift is visible on HC18 (above). External validation on
  local hospital data is Phase-2 work.
* It classifies the plane; it does not score image quality within the plane (e.g. symmetry, calliper
  landmarks). It does not see the HC18 annotations.
* Calibration was fitted and checked in-distribution; calibrated probabilities on other scanners may be off.
