# Testing

Owner: Yashwanth Gowda M N (testing), with each module owner writing the tests for their module.
All results below are from **actual runs on 12 Nov 2026** (Windows 11, Python 3.11.9, RTX 4050 Laptop GPU).
Raw logs: `docs/evidence/pytest_run.txt`, `docs/evidence/frontend_smoke_run.txt`, `docs/evidence/ui_e2e_run.txt`.

## Summary

| Suite | Tool | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|---|
| Backend API (T01–T32) | pytest + FastAPI TestClient | 38 (T18 and T24 have 4 inputs each) | 38 | 0 | 0 |
| ML unit / integration (U01–U32) | pytest | 32 | 32 | 0 | 0 |
| Plane quality gate (P01–P13) | pytest | 13 | 13 | 0 | 0 |
| INTERGROWTH-21st verification (G01–G08) | pytest | 14 (incl. all 189 published table cells) | 14 | 0 | 0 |
| Frontend smoke (S1–S3) | `node --test` against `next start` | 3 | 3 | 0 | 0 |
| End-to-end UI (E01–E14) | Playwright + Chrome, live backend and GPU models | 14 | 14 | 0 | 0 |
| **Total** | | **114** | **114** | **0** | **0** |

Feature A (standard-plane gate) added T26–T32, P01–P13 and the plane steps in E03, E04, E12 and E13;
the Python suite is **97 passed** locally (`docs/evidence/pytest_run.txt`).

Static checks: `ruff check .` (Python) clean; `npm run lint`, `npm run typecheck`, `npm run build` clean.

Screenshot of the pytest run: `docs/screenshots/test_run_pytest.png`.

**GitHub Actions CI** (run 43239985081): Python job **73 passed, 4 skipped** (51 + 4 skipped before step 5, run
43235961106), ruff clean; frontend job lint + build + smoke passed. The dataset and trained checkpoint are not in the repository, so the four tests
that need them (U05, U14, U27, U32, marked *data* below) are skipped in CI and run locally. The API tests never
need them: they use a randomly initialised checkpoint created in a temp folder.

## How to run

```powershell
python -m pytest -v                     # all Python tests (T + U)
python -m pytest tests/backend -v       # API tests only
cd frontend; npm run build; npm run test:smoke
# E2E (backend on :8000 and `next start` on :3000 running):
python scripts/ui_e2e.py
```

## Backend API test cases (`tests/backend/test_api.py`)

| ID | Type | Input | Expected | Result |
|---|---|---|---|---|
| T01 | Normal | `GET /health` with model loaded | 200, `status=ok`, `model_loaded=true` | Pass |
| T02 | Normal | `GET /model-info` | 200, input size, epoch, disclaimer | Pass |
| T03 | Normal | `GET /docs`, `/openapi.json` | 200; all 3 endpoints documented | Pass |
| T04 | Normal | Valid 320×240 PNG + pixel size 0.1 | 200; mask and overlay PNGs at original size; quality block; disclaimer | Pass |
| T05 | Normal | Valid RGB JPEG | 200 (converted to grayscale) | Pass |
| T06 | Normal | Valid PNG, no pixel size | 200; HC and area are `null`; warning flag | Pass |
| T07 | Normal | Valid PNG + pixel size 0.2 | Semi-axis mm = px × 0.2; HC > 0 | Pass |
| T08 | Boundary | 64×64 image (minimum side) | 200 | Pass |
| T09 | Boundary/negative | 32×32 image | 400 `invalid_image` | Pass |
| T10 | Boundary | File exactly at the size limit | 200 | Pass |
| T11 | Boundary | File 1 byte over the limit | 413 `file_too_large` | Pass |
| T12 | Negative | 11 MB upload (limit 10 MB) | 413 | Pass |
| T13 | Negative | Truncated (corrupted) PNG | 400 `invalid_image` | Pass |
| T14 | Negative | Random bytes declared as PNG | 400 | Pass |
| T15 | Negative | Empty file | 400 | Pass |
| T16 | Negative | PDF (`application/pdf`) | 415 `unsupported_media_type` | Pass |
| T17 | Negative | Real GIF renamed / declared as PNG | 415 (actual format checked) | Pass |
| T18 | Negative | `pixel_size_mm` = −0.1, 0, 5, `abc` | 422 `validation_error` (×4) | Pass |
| T19 | Negative | No file field | 422 | Pass |
| T20 | Negative | Checkpoint missing | `/health` → `degraded`; `/predict` and `/model-info` → 503 `model_not_loaded` | Pass |
| T21 | Normal | PNG + pixel size + GA 22.0 weeks | `growth` block; standard median 196.3 mm (published value at 22 weeks) | Pass |
| T22 | Normal | No GA | `growth` is `null` | Pass |
| T23 | Normal | GA but no pixel size | `growth` is `null` and a flag explains why | Pass |
| T24 | Boundary/negative | GA = 13.9, 40.5, −1, `abc` | 422 (×4) | Pass |
| T25 | Normal | `GET /reference/hc-centiles` | 53 points 14–40 weeks; 50th centile at 14 weeks = 97.9 mm (published) | Pass |
| T26 | Normal | Plane checkpoint present | `/health` → `plane_model_loaded=true`; `/model-info` has `plane_classifier` | Pass |
| T27 | Normal | Gate passes (threshold 0) | `plane_check.status=pass`, probabilities sum to 1, HC reported | Pass |
| T28 | Normal | Gate warns (threshold > 1), pixel size + GA | `status=warn`, "please re-acquire" message, `measurement_withheld=true`; HC, area and centile `null`; overlay still returned | Pass |
| T29 | Normal | Gate warns + `override_plane_check=true` | `overridden=true`, "Measured anyway", HC reported | Pass |
| T30 | Boundary | No plane checkpoint | `status=unavailable`, measurement not withheld | Pass |
| T31 | Normal/negative | `POST /plane-check` | 200 block (warn / unavailable); PDF → 415; junk bytes → 400 | Pass |
| T32 | Negative | `override_plane_check=maybe` | 422 | Pass |

The API tests use random-weight checkpoints written to a temp folder, so the gate outcome is forced through the
stored threshold (0 → always pass, 1.01 → always warn). They test the contract, not classifier accuracy.

## INTERGROWTH-21st verification (`tests/test_intergrowth.py`)

Equations from Papageorghiou et al., *Lancet* 2014;384:869-79, Table 2; reference values from the official
INTERGROWTH-21st table (`tests/data/intergrowth21_fetal_hc_zscores.csv`, transcribed from
intergrowth21.com `grow_fetal-zs_hc_table.pdf`).

| ID | Case | Result |
|---|---|---|
| G01 | All 189 published cells (14–40 weeks × z −3…+3) reproduced within 0.06 mm (185 identical after rounding to 0.1 mm; 4 differ by one rounding unit at .x5 boundaries) | Pass |
| G02 | Published medians at 14, 20, 30, 40 weeks (×4) | Pass |
| G03 | HC at the median → z = 0, 50th centile | Pass |
| G04 | z ±1.88 ↔ 3rd / 97th centile | Pass |
| G05 | HC at z = −2.5 is flagged outside the 3rd–97th range | Pass |
| G06 | GA 13.9, 40.1, 0, −5 rejected (×4) | Pass |
| G07 | SD increases with gestational age | Pass |
| G08 | Centile curves cover 14–40 weeks and are ordered | Pass |

## ML test cases (`tests/test_*.py`)

| ID | File | Case | Result |
|---|---|---|---|
| U01 | test_masks | Filling a 1-px synthetic outline gives the expected ellipse area (±2 %) and centre | Pass |
| U02 | test_masks | Fixed pipeline: mask still non-empty after downsizing to 128 px | Pass |
| U03 | test_masks | **Bug regression:** old resize-then-fill pipeline produces an (almost) empty mask | Pass |
| U04 | test_masks | Empty annotation is rejected | Pass |
| U05 | test_masks | *data*: all 999 real HC18 masks are non-empty (1–90 % foreground) | Pass |
| U06 | test_metrics | Empty target + empty prediction is NaN, **not 1.0** | Pass |
| U07 | test_metrics | Perfect prediction: Dice = IoU = 1 | Pass |
| U08 | test_metrics | Empty prediction on non-empty target: Dice = 0 | Pass |
| U09 | test_metrics | Half overlap: Dice 0.5, IoU 1/3 | Pass |
| U10 | test_metrics | Soft Dice loss ≈ 0 for a correct prediction | Pass |
| U11 | test_split | Group ID parsing (`014_3HC.png` → `014`), bad names rejected | Pass |
| U12 | test_split | Split is deterministic and no group appears in two splits | Pass |
| U13 | test_split | Different seed gives a different split | Pass |
| U14 | test_split | *data*: committed split == regenerated split, 70/15/15 ± 3 %, no leakage | Pass |
| U15 | test_dataset | Preprocessing shape/dtype/range | Pass |
| U16 | test_dataset | Augmentation keeps the mask binary and aligned with the image | Pass |
| U17 | test_dataset | Augmentation is reproducible for a fixed seed | Pass |
| U18 | test_geometry | Circle perimeter exact (2πr) | Pass |
| U19 | test_geometry | Ellipse a=10, b=5 perimeter = 48.4422411 (rel. 1e-8) | Pass |
| U20 | test_geometry | Degenerate ellipse (b=0) → 4a | Pass |
| U21 | test_geometry | Negative axis rejected | Pass |
| U22 | test_geometry | Fit recovers a drawn ellipse (axes ±1.5 px, angle ±1°) | Pass |
| U23 | test_geometry | Angle refers to the major axis for a tall ellipse | Pass |
| U24 | test_geometry | Empty mask → no fit | Pass |
| U25 | test_geometry | Largest component removes specks | Pass |
| U26 | test_geometry | HC is correct after the anisotropic 800×540 → 256×256 → 800×540 round trip (±1 %) | Pass |
| U27 | test_geometry | *data*: reference outlines reproduce the CSV HC within 0.1 mm (100 images) | Pass |
| U28 | test_inference | Full-resolution Dice/IoU helper | Pass |
| U29 | test_inference | Prediction returned at original resolution, probabilities in [0, 1] | Pass |
| U30 | test_inference | Non-grayscale input rejected | Pass |
| U31 | test_inference | No pixel size → no HC | Pass |
| U32 | test_inference | *data + checkpoint*: trained model within 10 mm on 5 held-out test images | Pass |

## Plane quality gate test cases (`tests/test_planes.py`)

| ID | Case | Result |
|---|---|---|
| P01 | Letterbox keeps the aspect ratio and pads with black | Pass |
| P02 | ImageNet normalisation gives the same result for uint8 and float input | Pass |
| P03 | Temperature scaling recovers a known over-confidence factor (T ≈ 3) | Pass |
| P04 | ECE is 0 for calibrated and 0.24 for over-confident toy predictions | Pass |
| P05 | Gate score = P(brain) × P(TT \| brain) | Pass |
| P06 | Threshold rule meets the target TT recall and is the largest such threshold | Pass |
| P07 | GPU augmentation keeps shape and range | Pass |
| P08 | Official CSV parsing (`;`, trailing-space column), labels and sub-labels | Pass |
| P09 | Split: official test untouched, train/val patient-disjoint, deterministic | Pass |
| P10 | `PlaneClassifier.check` probabilities, pass decision and sub-plane only for brain | Pass |
| P11 | Colour (3-D) input rejected | Pass |
| P12 | Unknown plane label rejected | Pass |
| P13 | Split file round trip | Pass |

## Frontend smoke test (`frontend/tests/smoke.test.mjs`)

| ID | Case | Result |
|---|---|---|
| S1 | Home page renders the analyser, upload area, pixel-spacing input and disclaimer | Pass |
| S2 | `/model` page renders | Pass |
| S3 | Unknown route returns 404 | Pass |

## End-to-end UI check (`scripts/ui_e2e.py`)

Real Chrome, real backend, trained models on the GPU. Demo image `006_HC.png` is from the held-out test split
(reference HC 67.84 mm, an early-gestation image that the plane gate does **not** accept; see `MODEL_CARD.md`).

| ID | Case | Result |
|---|---|---|
| E01 | Empty state, disclaimer and "Model ready" status visible | Pass |
| E02 | Uploading an HC18 image auto-fills its pixel spacing (0.06560565 mm/px) | Pass |
| E03 | Analyse → plane-check WARN badge shown before the result; HC withheld | Pass |
| E04 | "Measure anyway (override)" → HC 67.3 mm vs reference 67.84 mm (asserted within 5 mm), warning kept | Pass |
| E05 | Print report renders (screenshot + PDF) | Pass |
| E06 | Dark theme toggles | Pass |
| E07 | Corrupted upload shows the API's 400 message | Pass |
| E08 | Wrong file type is rejected in the browser | Pass |
| E09 | Model page shows held-out test metrics (segmentation and plane gate) | Pass |
| E10 | Mobile (390 px): results render, no horizontal scroll | Pass |
| E11 | Gestational age outside 14–40 weeks rejected in the form (Analyse disabled) | Pass |
| E12 | GA 20+0 (example input; HC18 has no GA) on `544_HC.png` → plane check PASS, centile card (42nd) and chart | Pass |
| E13 | FETAL_PLANES_DB test femur image → plane check WARN ("femur"), HC withheld | Pass |
| E14 | Swagger UI lists the endpoints | Pass |

The Phase-1 E01–E12 (before the plane gate) also passed against the **Docker Compose** stack:
`docs/evidence/ui_e2e_docker_run.txt`.

## Model evaluation

Held-out test metrics (Dice 0.973, HC MAE 2.18 mm; plane gate accuracy 94.0 %, TT acceptance 95.7 %) are
documented in [`MODEL_CARD.md`](MODEL_CARD.md).

## Known issues

* `npm audit` reports advisories in **build-time tooling only**: PostCSS 8.4.31 bundled inside `next`,
  and minimatch inside the ESLint TypeScript parser. Neither is shipped to the browser, and neither can be
  fixed without breaking upgrades of `next` / `eslint-config-next`. Re-check when upgrading.
