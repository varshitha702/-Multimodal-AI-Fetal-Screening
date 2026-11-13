# Dataset cards: HC18 and FETAL_PLANES_DB

Owner: Pujari Varshitha (dataset and preprocessing), for both datasets.

# Part 1: HC18 (head segmentation and HC)

## Source

**HC18 grand challenge** (automated measurement of fetal head circumference).
T. L. A. van den Heuvel, D. de Bruijn, C. L. de Korte, B. van Ginneken, "Automated measurement of fetal head
circumference using 2D ultrasound images", *PLoS ONE* 13(8): e0200412, 2018.
Data: Zenodo, doi:10.5281/zenodo.1327317 (CC BY 4.0). The data is **not committed** to this repository
(`data/raw/` is gitignored); see `data/README.md` for the expected layout.

## Contents (as verified by `scripts/check_hc18.py`)

| Part | Files | Labels |
|---|---|---|
| `training_set/` | 999 images `NNN_HC.png` / `NNN_kHC.png` + 999 `*_Annotation.png` | Ellipse outline (1–2 px) per image |
| `training_set_pixel_size_and_HC.csv` | 999 rows: `filename, pixel size(mm), head circumference (mm)` | Reference HC in mm |
| `test_set/` | 335 images | **No public labels** |
| `test_set_pixel_size.csv` | 335 rows: `filename, pixel size(mm)` | none |

* Images are 8-bit grayscale, mostly 800 × 540 px (a few between 738 and 800 px wide, 539–563 px high).
* Pixel spacing differs per image (it is stored in the CSV and needed for mm measurements).
* Reference HC in the training CSV spans 44.3–346.4 mm.

## Findings during this project

1. **Annotations are outlines, not filled masks.** They must be filled before training (see the bug
   below and `ml/masks.py`).
2. **Repeat scans of the same fetus.** Files sharing a numeric prefix (`014_HC`, `014_2HC`, `014_3HC`) are
   repeat scans of the same fetus with nearly identical HC. 999 images → **806 groups**
   (642 single, 141 ×2, 17 ×3, 6 ×4). The split is therefore done by group.
3. **One exact duplicate.** `training_set/392_HC.png` and `392_2HC.png` are byte-identical (same pixel
   size and HC). They are in the same group, so they cannot leak across splits.
4. **Test-set filenames reuse training-set names.** `test_set/000_HC.png` is a different image from
   `training_set/000_HC.png`. Anything keyed by filename alone is ambiguous; the frontend's pixel-size lookup
   is keyed by a SHA-256 of the file bytes instead.
5. **Fitting the reference outlines reproduces the CSV HC** to an MAE of 0.009 mm (ellipse fit + Ramanujan
   perimeter), which confirms how the reference HC was defined.

## Preprocessing (`ml/masks.py`, `ml/dataset.py`)

1. Annotation outline → `cv2.fitEllipse` on all outline pixels **at original resolution** → filled ellipse.
2. Filled mask resized to 256 × 256 (area interpolation, threshold 0.5); all 999 masks non-empty
   (foreground 3.6–57.6 %, mean 30.1 %).
3. Image converted to grayscale, resized to 256 × 256 (area interpolation), scaled to [0, 1].
4. Training-only augmentation: horizontal flip, rotation ±15°, scale 0.9–1.1, brightness/contrast jitter.

## Split (`ml/split.py` → `data/splits/hc18_split.csv`, committed)

Seed 42, 70/15/15 **by examination group**, stratified by HC quintile. HC18 has no patient IDs, so the group
prefix is the best available proxy; this is documented as an image-group-level split, not a verified
patient-level split.

| Split | Images | Groups | HC range (mm) | Mean HC (mm) |
|---|---|---|---|---|
| train | 696 | 565 | 44.3 – 346.4 | 175.3 |
| val | 152 | 120 | 56.8 – 335.0 | 172.7 |
| test | 151 | 121 | 60.3 – 323.3 | 171.7 |

The official 335-image test set has no public labels and is used only for qualitative demos
(`docs/screenshots/official_test_set_qualitative.png`).

## The empty-mask bug (found and fixed)

The original trainer resized the outline to 128 × 128 with nearest-neighbour **before** filling it, which broke
the 1-px outline so that **999 / 999 masks came out empty**; the epsilon-smoothed Dice then reported a fake
1.0. Evidence and fix: `docs/evidence/README.md`, `docs/screenshots/mask_bug_before_after.png`.

## Ethics and privacy

HC18 is a public, de-identified research dataset. The application does not store uploaded images.
Do not upload identifiable patient data to demos.

# Part 2: FETAL_PLANES_DB (standard-plane quality gate, Feature A)

## Source

X. P. Burgos-Artizzu, D. Coronado-Gutiérrez, B. Valenzuela-Alcaraz, E. Bonet-Carne, E. Eixarch, F. Crispi,
E. Gratacós, "Evaluation of deep convolutional neural networks for automatic classification of common
maternal fetal ultrasound planes", *Scientific Reports* 10, 10200 (2020). Data: Zenodo record 3904280
(doi:10.5281/zenodo.3904280, CC BY 4.0). Not committed (`data/raw/` is gitignored); layout in `data/README.md`.

## Contents (verified after download)

* 12,400 PNG images from **1,792 patients**, 18–40 weeks of gestation, routine screening at two Barcelona hospitals.
  Machines: Voluson E6 (5,862), Aloka (5,114), Voluson S10 (1,082), other (342).
* `FETAL_PLANES_DB_data.csv` (`;`-separated, last column name `Train ` has a trailing space):
  `Plane` (6 classes), `Brain_plane` (Trans-thalamic / Trans-cerebellum / Trans-ventricular / Other /
  Not A Brain), `Patient_num`, and the **official** `Train` flag (1 = train, 0 = test).
* The official train and test sets share **no patients** (896 / 896).

## Split (`ml/train_planes.py` → `data/splits/fetal_planes_split.csv`, committed)

* **test** = official `Train == 0` rows, untouched (final evaluation only).
* official `Train == 1` rows → **train / val by `Patient_num`** (15 % of patients to val, seed 42), so no
  patient has images in both. The script asserts this.

| Plane | train | val | test | total |
|---|---|---|---|---|
| Fetal brain | 1,402 | 218 | 1,472 | 3,092 |
| Fetal abdomen | 309 | 44 | 358 | 711 |
| Fetal femur | 454 | 62 | 524 | 1,040 |
| Fetal thorax | 934 | 124 | 660 | 1,718 |
| Maternal cervix | 848 | 133 | 645 | 1,626 |
| Other | 2,144 | 457 | 1,612 | 4,213 |
| **Images (patients)** | **6,091 (762)** | **1,038 (134)** | **5,271 (896)** | **12,400 (1,792)** |

Brain sub-planes (train / val / test): Trans-thalamic 755 / 118 / 765, Trans-cerebellum 319 / 56 / 339,
Trans-ventricular 263 / 32 / 302, Other 65 / 12 / 66.

## Preprocessing

Grayscale → **letterbox** to a square (aspect ratio kept, black padding, which matches the ultrasound
background) → resize → replicate to 3 channels → ImageNet normalisation (`ml/planes.py`, shared by training
and inference). Training-only augmentation on the GPU: horizontal flip, rotation ±12°, scale 0.85–1.15,
shift ±6 %, contrast/brightness/gamma jitter.

## Known domain gap with HC18

FETAL_PLANES_DB images span 18–40 weeks of gestation (two BCNatal sites, Hospital Clínic and Hospital Sant Joan de Déu, Barcelona; paper, Methods). HC18 (Netherlands) contains
many **early-gestation** images (reference HC < 100 mm, about ≤ 16 weeks), and every HC18 image has black
anonymisation boxes. Both affect how often the gate accepts HC18 images (see `docs/MODEL_CARD.md`).

# Part 3: UCI Cardiotocography (CTG branch, Feature C)

## Source

D. Ayres-de-Campos, J. Bernardes, A. Garrido, J. Marques-de-Sá, L. Pereira-Leite, "SisPorto 2.0: a program for
automated analysis of cardiotocograms", *J Matern Fetal Med* 9(5):311-318, 2000. UCI Machine Learning
Repository, dataset 193 "Cardiotocography" (CC BY 4.0), file `CTG.xls`. Not committed (`data/raw/` is
gitignored); download: `https://archive.ics.uci.edu/static/public/193/cardiotocography.zip` into
`data/raw/CTG/`.

## Contents (verified by `ml/ctg.py` and `tests/test_ctg.py`)

* 2,126 CTG records; NSP labels **Normal 1,655 / Suspect 295 / Pathological 176**, as documented by UCI.
* The 21 features are read from the **`Data`** sheet, where AC, FM, UC, DL, DS and DP are **per-second rates**
  (e.g. record CTG0002: 4 accelerations / 627 s = 0.00638). The `Raw Data` sheet holds counts; the loader checks
  that both sheets are row-aligned (columns `b`, `e`, `NSP`) and uses `SegFile` (CTG0001…) as the record id.
* No missing values. **10 groups of identical feature vectors (22 records), each group with one label**; they
  are kept together in one split so no duplicate can leak between train and test.

## Split (`data/splits/ctg_split.csv`, committed)

70/15/15 stratified by NSP, seed 42, by duplicate group: train 1,488, val 317, test 321 (class counts in
`docs/MODEL_CARD.md`). The UI's "load example" records come from the **test** split.

## Different patients

The CTG records and the ultrasound images (HC18, FETAL_PLANES_DB) come from different hospitals, countries
and patients. Nothing links a CTG record to an image; the screening summary is therefore rule-based and
illustrative (see the model card).
