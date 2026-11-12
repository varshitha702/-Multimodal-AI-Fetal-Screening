# Dataset card: HC18

Owner: Pujari Varshitha (dataset and preprocessing).

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
