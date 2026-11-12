# HANDOFF: FetalAI Phase-1 (for rewriting the report and slides)

State on **12 Nov 2026**, for the Phase-1 review on **14 Nov 2026**. Every number below was produced by code in
this repository; the source file is given for each. Nothing here is a placeholder.

> Academic research prototype, **not a medical device**. This sentence must stay in the report, slides and UI.

## 1. One-paragraph summary

FetalAI segments the fetal head in 2-D ultrasound images with a U-Net trained on the public HC18 dataset, fits
an ellipse to the segmentation, and reports head circumference (HC) in millimetres. Optionally, it compares the
HC with the INTERGROWTH-21st fetal growth standard for a user-entered gestational age. It is served by a FastAPI
backend and used through a Next.js web app (plus a Streamlit research dashboard). On a held-out test split
(151 images, split by examination so no fetus is in both training and test) it reaches **Dice 0.973** and an
**HC mean absolute error of 2.18 mm**. An earlier version reported "Dice 1.0"; we found that number was invalid
(every training mask was empty), fixed the root cause, and documented it.

## 2. Final architecture and stack

```text
Browser ── Next.js 15.5 + TypeScript + Tailwind + shadcn/ui (frontend/, :3000)
              │  multipart POST /api/v1/predict, GET /model-info, /health, /reference/hc-centiles
              ▼
         FastAPI + Pydantic v2 + Uvicorn (backend/app, :8000)
         api/ (routes, deps) · core/ (config, errors, JSON logging) · schemas/ · services/
              │
              ▼
         ml/inference.Segmenter ──► U-Net (PyTorch 2.5, CUDA 12.1 / CPU) ──► ml/geometry (ellipse, HC)
                                                                         └► ml/intergrowth (centile)
Streamlit dashboard (dashboard/) ──► same ml/inference.Segmenter
Offline: ml/masks → ml/split → ml/train (GPU) → ml/evaluate
```

Diagrams (architecture, flowchart, UML use-case / class / sequence, DFD 0 and 1, all Mermaid and validated):
[`ARCHITECTURE.md`](ARCHITECTURE.md). Decisions: [`adr/`](adr/) (5 ADRs).

| Part | Technology |
|---|---|
| Model | U-Net, 4 levels, 32 base channels, 7,762,465 parameters, input 256 × 256 grayscale |
| Training | PyTorch 2.5.1 + CUDA 12.1, AMP, Adam, BCE + soft Dice, NVIDIA RTX 4050 Laptop 6 GB |
| Backend | FastAPI 0.136, Pydantic 2.13, pydantic-settings, Uvicorn |
| Deployment | Docker Compose (python:3.11-slim + CPU torch; node:22-alpine) |
| Frontend | Next.js 15.5.29, React 19.2.9, TypeScript, Tailwind 3.4, shadcn/ui (new-york), lucide icons, next-themes |
| Tests / CI | pytest 9, FastAPI TestClient, `node --test`, Playwright + Chrome, ruff, GitHub Actions |
| Fallback UI | Streamlit 1.52 |

## 3. Real metrics

### Dataset and split (`outputs/split_summary.json`, `docs/DATASET.md`)

HC18 training set: 999 labelled images in **806 examination groups**. Split 70/15/15 by group, seed 42,
HC-stratified: train 696 images / 565 groups, val 152 / 120, test 151 / 121.

### Training (`outputs/training_summary.json`, `outputs/training_history.csv`)

65 epochs (early stopping, patience 15), **11.4 min** on the RTX 4050. Best **validation Dice 0.9760**,
IoU 0.9539, at epoch 50.

### Held-out test split, n = 151 (`outputs/test_metrics.json`, `docs/MODEL_CARD.md`)

| Metric | Value |
|---|---|
| Dice | 0.973 ± 0.028 (median 0.979, min 0.731) |
| IoU | 0.948 ± 0.046 |
| HC mean absolute error | **2.18 mm** (1.48 % of HC) |
| HC error (pred − ref) | −0.10 ± 2.95 mm (bias −0.10 mm) |
| Bland–Altman 95 % limits of agreement | −5.89 to +5.69 mm |
| Absolute HC error P50 / P90 / P95 | 1.80 / 4.66 / 5.93 mm |
| Ellipse-fit failures | 0 / 151 |
| Inference time (GPU, one 800 × 540 image, via API) | about 60–170 ms |

Measurement-code check: applying the same ellipse + Ramanujan code to the 999 *reference* annotations reproduces
the CSV HC with MAE **0.009 mm**, so the 2.18 mm error is the model's, not the geometry's.

### INTERGROWTH-21st centile verification (`tests/test_intergrowth.py`)

Equations from Papageorghiou et al., *Lancet* 2014;384:869-79, Table 2. All **189** cells of the official
INTERGROWTH-21st fetal HC table (14–40 weeks × z −3…+3) are reproduced within **0.06 mm**.

## 4. Test counts (`docs/TESTING.md`)

| Suite | Cases | Result |
|---|---|---|
| Backend API, T01–T25 | 31 | 31 pass |
| ML unit/integration, U01–U32 | 32 | 32 pass |
| INTERGROWTH-21st, G01–G08 | 14 | 14 pass |
| Frontend smoke, S1–S3 | 3 | 3 pass |
| End-to-end UI in Chrome with the real model, E01–E12 | 12 | 12 pass |
| **Total** | **92** | **92 pass, 0 fail** |

GitHub Actions CI (ruff + pytest + frontend lint/build/smoke): green; Python job 73 passed, 4 skipped (the 4
need the dataset/checkpoint, which are not in the repository).

## 5. Screenshots and figures (`docs/screenshots/`)

| File | Shows |
|---|---|
| `mask_bug_before_after.png` | **Bug evidence:** image / outline / old (empty) mask / fixed mask |
| `mask_sanity_grid.png` | 12 fixed masks overlaid on HC18 images |
| `training_curves.png` | Loss and Dice/IoU per epoch |
| `test_hc_agreement.png` | Predicted vs reference HC scatter + Bland–Altman plot |
| `test_examples.png` | Worst / 2nd worst / median / best test cases |
| `official_test_set_qualitative.png` | Predictions on the unlabelled official HC18 test set |
| `ui_upload_empty.png` | Web app, empty state |
| `ui_upload_ready.png` | Image uploaded, pixel spacing auto-filled |
| `ui_results.png` | Results: original / overlay / mask, metric cards, quality checks |
| `ui_results_dark.png` | Same, dark theme |
| `ui_results_centile.png` | Results with gestational age: centile card and INTERGROWTH-21st chart |
| `ui_report_print.png`, `ui_report_centile_print.png`, `ui_report_example.pdf` | Printable report |
| `ui_error_corrupted.png` | Error state (corrupted file → API 400 message) |
| `ui_mobile.png` | Mobile layout (390 px) |
| `ui_model_page.png` | Model page with held-out metrics |
| `swagger_docs.png` | FastAPI Swagger UI (`/docs`) |
| `test_run_pytest.png` | pytest run (77 Python tests passing) |

## 6. Features: done vs moved to Phase 2

| Brief item | Status |
|---|---|
| 1. Data pipeline fix + non-empty-mask test + metric fix + sanity grid | **Done** |
| 2. Fixed-seed 70/15/15 split | **Done**, by examination group instead of single images (see §8) |
| 3. GPU training (256 px, AMP, Adam, BCE + Dice, augmentation, early stopping, logs, checkpoint) | **Done** |
| 4. Test-split evaluation: Dice, IoU, HC error in mm; model card | **Done** |
| 5. Clinical input: HC-for-GA centile from a cited, verified standard | **Done** (INTERGROWTH-21st). GA *estimation* from HC → Phase 2 |
| 6. FastAPI backend (3 endpoints, lifespan, Pydantic v2, settings, CORS, logging, 400/413/415/422/503) | **Done** (+ `/reference/hc-centiles`) |
| 7. Next.js frontend (upload, auto-fill, results, report, disclaimer, states, responsive, light/dark); Streamlit kept | **Done** |
| 8. Tests (numbered T-cases, unit, smoke) + GitHub Actions CI | **Done** (+ end-to-end UI check) |
| 9. Docker Compose | **Done**: `docker-compose.yml`, `backend/Dockerfile` (CPU torch), `frontend/Dockerfile`; all 12 end-to-end checks pass against the containers (`docs/evidence/ui_e2e_docker_run.txt`); CPU container gives the same HC (67.34 mm) as the GPU run |
| Docs (README, ARCHITECTURE, API, DATASET, MODEL_CARD, TESTING, ROADMAP, ADRs, CHANGELOG, CONTRIBUTING, HANDOFF) | **Done** |

Phase 2 ([`ROADMAP.md`](ROADMAP.md)): FETAL_PLANES_DB plane classifier as a quality gate; CTG branch with
decision-level fusion; uncertainty (TTA / MC dropout) and calibration; pretrained-encoder U-Net and
cross-validation; GA estimation from HC; ONNX/TensorRT; PostgreSQL history, JWT auth, audit log; cloud
deployment; clinician usability study.

## 7. Bugs found: root cause and fix

| # | Bug | Root cause | Fix | Evidence |
|---|---|---|---|---|
| 1 | **"Dice 1.0" after 2 epochs with loss ≈ 1.0; no mask in the app** | Annotation outline (1 px) resized to 128 px with nearest-neighbour *before* filling → outline broke → `findContours` + fill gave **empty masks for 999/999 images** | Fit ellipse to the outline at original resolution, fill, *then* resize (`ml/masks.py`) → 0/999 empty | `docs/evidence/README.md`, `mask_bug_before_after.png`, `outputs/mask_pipeline_check.json`, test U03 |
| 2 | Empty prediction scored as perfect | Dice `(2I + ε)/(P + T + ε)` = 1 when both are empty | Hard Dice/IoU return NaN for empty targets (`ml/metrics.py`) | test U06 |
| 3 | Trainer/dataset check could not find the data | Stale paths (`training_set/training_set`) and CSV column `pixel size` vs `pixel size(mm)` | Corrected paths and column | PR #1, #2 |
| 4 | Dashboard and API would give different HC | Dashboard had its own preprocessing (bilinear resize) and a pixel-contour perimeter | Dashboard uses the shared `Segmenter` → identical HC (67.34 mm on `006_HC.png`) | PR #7 |
| 5 | Dashboard reported 806 instead of 999 image–mask pairs | Suffix matching missed `NNN_2HC` names | Match on full stem | PR #7 |
| 6 | Fitted ellipse drawn with the wrong orientation (caught before release) | OpenCV's angle refers to the *first* axis; code sorted axes by size | Normalise so the angle refers to the major axis | test U23 |
| 7 | HC18 pixel-size lookup ambiguous | Official test set reuses training filenames (`000_HC.png` exists in both) | Key lookup by SHA-256 of file bytes | `docs/DATASET.md` |
| 8 | Possible leakage between splits | Repeat scans of one fetus (`014_HC`, `014_2HC`, …); one byte-identical pair (`392_HC` = `392_2HC`) | Group-level split | ADR-0002, test U14 |

## 8. Deviations from the original plan

* **Split by examination group** (806 groups), not single images: the brief said "image-level", but repeat
  scans would leak across splits. Documented as group-level, not verified patient-level.
* **HC measured at original resolution** after resizing the mask back (the 256 × 256 input is anisotropic).
* **Clinical input = INTERGROWTH-21st HC-for-GA centile only**; GA estimation from HC was not implemented
  because only the centile equations were verified against a published table in time.
* **shadcn/ui components written into the repo by hand** (same source as the shadcn CLI) because the network was
  too slow for the CLI; `components.json` is included for future `shadcn add`.
* **Next.js 15.5.29 / React 19.2.9 pinned** (the scaffold's 15.1.6 had known security advisories).
  `npm audit` still lists two build-time-only advisories (PostCSS inside `next`, minimatch inside the ESLint
  parser), not fixable without breaking upgrades; documented in `TESTING.md`.
* **End-to-end UI check uses Playwright + the installed Chrome** (`scripts/ui_e2e.py`).
* Extra beyond the brief: `/api/v1/reference/hc-centiles`, `/model` page, JSON/overlay downloads, API status pill.

## 9. Module mapping (team)

| Member | Module | Main files | PRs |
|---|---|---|---|
| Pujari Varshitha | Dataset and preprocessing | `ml/masks.py`, `ml/split.py`, `ml/dataset.py`, `docs/DATASET.md`, `scripts/check_hc18.py`, `scripts/mask_sanity.py` | #1, #3 |
| Nachiketha K G | Model training and evaluation | `ml/model.py`, `ml/train.py`, `ml/evaluate.py`, `ml/geometry.py`, `docs/MODEL_CARD.md` | #4, #5 |
| Vishwas K | FastAPI backend and integration | `backend/`, `ml/inference.py`, `ml/intergrowth.py`, `docs/API.md`, Docker | #6, #10, #11 |
| Rakshitha V | Frontend | `frontend/`, `scripts/export_pixel_sizes.py` | #7, #10 |
| Yashwanth Gowda M N | Requirements, design docs, testing | `tests/`, `scripts/ui_e2e.py`, `.github/workflows/ci.yml`, `docs/TESTING.md`, `docs/ARCHITECTURE.md`, `docs/adr/` | #8, #9 |

## 10. How to demo (5 minutes)

0. Alternative to steps 1–2: `docker compose up` (CPU) serves both on the same ports.
1. `scripts\run_backend.bat` → open <http://localhost:8000/docs> (Swagger).
2. `cd frontend && npm run start` (after `npm run build`) or `npm run dev` → <http://localhost:3000>.
3. Upload `data/raw/HC18/training_set/006_HC.png` (held-out test image): spacing auto-fills; HC ≈ 67.3 mm vs
   reference 67.84 mm.
4. Upload `544_HC.png`, enter GA 20 + 0 (example value; HC18 has no GA): 42nd centile on the INTERGROWTH-21st chart.
5. Show "Print report", the dark theme, a corrupted file (error message), and the `/model` page.
6. Show `mask_bug_before_after.png` for the bug story, then `test_hc_agreement.png` for the results.

## 11. Pull requests

#1 mask bug · #2 restructure · #3 split · #4 training · #5 evaluation · #6 backend · #7 frontend + CI ·
#8 testing docs · #9 project docs · #10 GA centiles · #11 Docker Compose + handoff.
