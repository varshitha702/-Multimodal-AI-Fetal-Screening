# HANDOFF: FetalAI Phase 1 + brief-2 features (for rewriting the report and slides)

State on **13 Nov 2026** (feature freeze 13 Nov, 8 PM), for the review on **14 Nov 2026**. Every number below
was produced by code in this repository; the source file is given for each. Nothing here is a placeholder.

> Academic research prototype, **not a medical device**. This sentence must stay in the report, slides and UI.

## 1. One-paragraph summary

FetalAI is a multimodal fetal screening prototype:
- **Plane check:** a calibrated EfficientNet-B0 classifier confirms that an ultrasound image is the standard
  **trans-thalamic plane** before anything is measured.
- **HC measurement:** a U-Net trained on HC18 segments the fetal head, an ellipse is fitted, and HC is
  reported in millimetres with an **uncertainty (HC ± SD from test-time augmentation)** and a "re-scan" flag.
- **Growth context:** the HC is placed on the INTERGROWTH-21st growth standard when the gestational age is
  given.
- **CTG:** a CTG record is classified as Normal / Suspect / Pathological by XGBoost, with SHAP explanations.
- **Summary:** all of this feeds a rule-based **screening summary** (Routine / Review recommended / Urgent
  review).

Held-out results:
- HC: Dice **0.973**, HC MAE **2.18 mm**.
- Plane classifier: accuracy **94.0 %**, calibration error 0.035 → **0.014**.
- Uncertainty: the SD correlates with the real HC error at Spearman **0.27**.
- CTG: macro-F1 **0.963**, pathological recall **27/27**.

The weak spots are reported, not hidden:
- The plane gate accepts only **60 %** of HC18 images (domain shift, early gestation).
- The uncertainty signal is modest.
- The ultrasound and CTG data come from **different patients**, so the fusion is illustrative.

An earlier version reported "Dice 1.0"; we found that number was invalid, fixed the cause and documented it.

## 2. What makes FetalAI different (with evidence)

Commercial ultrasound machines already measure HC automatically. FetalAI adds three things around that
measurement, and each claim below is backed by a held-out evaluation in this repository:

1. **It checks the scan plane before it measures.** HC is only defined on the trans-thalamic plane; FetalAI
   classifies the plane first and withholds the HC (with a plain "please re-acquire" message and an explicit
   override) when the plane is not confirmed. *Evidence:* official FETAL_PLANES_DB test set (5,271 images,
   unseen patients): plane accuracy 94.0 %, 95.7 % of trans-thalamic images accepted, 99.8 % of non-brain images
   rejected; probabilities temperature-calibrated (ECE 0.035 → 0.014). `outputs/plane_metrics.json`, ADR-0006.
2. **It says how much to trust each measurement.** Eight test-time-augmented passes give HC ± SD, an
   uncertainty map and a "low reliability, re-scan" flag with thresholds chosen on validation data.
   *Evidence:* HC18 test split (n = 151): SD vs actual error Spearman ρ 0.27 (95 % CI 0.11–0.41); flagged images
   have MAE 3.68 mm vs 1.99 mm for the rest. `outputs/uncertainty_metrics.json`, ADR-0007.
3. **It combines two modalities into one explained summary.** A CTG classifier with per-prediction SHAP
   factors and a documented rule set (a pathological CTG always escalates; missing inputs are listed, never
   assumed normal). *Evidence:* UCI CTG test split (n = 321): XGBoost macro-F1 0.963, pathological recall 27/27;
   every rule has a unit test (F01–F14). `outputs/ctg_metrics.json`, ADR-0008.

Underneath, the HC measurement itself is validated (test MAE 2.18 mm; the geometry code reproduces the
reference HC to 0.009 mm) and placed on a verified growth standard (all 189 INTERGROWTH-21st table cells within
0.06 mm). What we do **not** claim: clinical validation, a calibrated combined risk, or performance on other
hospitals' machines (see §9).

## 3. Final architecture and stack

```text
Browser ── Next.js 15.5 + TypeScript + Tailwind + shadcn/ui (frontend/, :3000)
              │  POST /predict (multipart), /plane-check, /ctg/predict, /screening/summary (JSON);
              │  GET /health, /model-info, /ctg/examples, /reference/hc-centiles
              ▼
         FastAPI + Pydantic v2 + Uvicorn (backend/app, :8000)
         api/ (routes, ctg_routes, deps) · core/ · schemas/ · services/ (model_service, ctg_service, image_service)
              │
              ├─► ml/planes.PlaneClassifier  (EfficientNet-B0, 2 heads, temperature-scaled)  ── plane gate
              ├─► ml/inference.Segmenter     (U-Net) ─► ml/geometry (ellipse, HC) ─► ml/intergrowth (centile)
              ├─► ml/uncertainty.tta_predict (8 augmented passes of the same U-Net)
              ├─► ml/ctg.CTGModel            (XGBoost Booster + SHAP TreeExplainer)
              └─► ml/fusion.summarize        (rules R1–R6)
Streamlit dashboard (dashboard/) ──► same ml/inference.Segmenter
Offline: ml/masks → split → train → evaluate · ml/train_planes → evaluate_planes · ml/evaluate_uncertainty ·
         ml/train_ctg
```

Diagrams (architecture, flowchart, UML use-case / class / two sequence diagrams, DFD 0 and 1; these 8 plus the
README diagram all render, checked with `scripts/check_mermaid.py`): [`ARCHITECTURE.md`](ARCHITECTURE.md). Decisions:
[`adr/`](adr/) (8 ADRs).

| Part | Technology |
|---|---|
| Segmentation | U-Net, 4 levels, 32 base channels, 7,762,465 parameters, 256 × 256 grayscale input |
| Plane gate | torchvision EfficientNet-B0 (ImageNet), plane head (6) + brain sub-plane head (4), 320 × 320 input |
| CTG | XGBoost 2.1 (`multi:softprob`), SHAP TreeExplainer (0.51 locally, 0.49 in the Docker image); scikit-learn for the baselines |
| Training | PyTorch 2.5.1 + CUDA 12.1, AMP, NVIDIA RTX 4050 Laptop 6 GB |
| Backend | FastAPI, Pydantic 2, pydantic-settings, Uvicorn |
| Deployment | Docker Compose (python:3.11-slim + CPU torch; node:22-alpine) |
| Frontend | Next.js 15.5.29, React 19.2.9, TypeScript, Tailwind 3.4, shadcn/ui, lucide, next-themes |
| Tests / CI | pytest 9, FastAPI TestClient, `node --test`, Playwright + Chrome, ruff, GitHub Actions |

## 4. Real metrics

### 4.1 HC segmentation and measurement (Phase 1; `outputs/test_metrics.json`, `docs/MODEL_CARD.md`)

HC18 training set: 999 labelled images in **806 examination groups**; split 70/15/15 by group, seed 42:
train 696 / val 152 / test 151 images. Training: 65 epochs, 11.4 min, best validation Dice 0.9760 (epoch 50).

| Held-out test, n = 151 | Value |
|---|---|
| Dice | 0.973 ± 0.028 (median 0.979, min 0.731) |
| IoU | 0.948 ± 0.046 |
| HC mean absolute error | **2.18 mm** (1.48 % of HC) |
| HC bias, Bland–Altman 95 % limits | −0.10 mm, −5.89 to +5.69 mm |
| Absolute HC error P50 / P90 / P95 | 1.80 / 4.66 / 5.93 mm |
| Ellipse-fit failures | 0 / 151 |

Geometry check: the ellipse + Ramanujan code reproduces the CSV HC from the reference annotations with MAE
**0.009 mm**. INTERGROWTH-21st: all **189** official table cells reproduced within **0.06 mm**.

### 4.2 Feature A: standard-plane quality gate (`outputs/plane_metrics.json`)

FETAL_PLANES_DB (Burgos-Artizzu et al., *Sci Rep* 2020): 12,400 images, 1,792 patients. Official test set kept
untouched (5,271 images, 896 patients); official train set split by patient into train 6,091 / val 1,038.
320 px chosen over 224 px on the **validation** score (0.869 vs 0.860). Training 16.5 min.

| Official test set | Value |
|---|---|
| Plane accuracy / macro-F1 (6 classes) | **94.0 %** / 0.934 (dataset paper's best CNN: 6.2 % top-1 error) |
| Brain sub-plane accuracy / macro-F1 (TT, TC, TV, other) | 82.7 % / 0.721 |
| ECE before → after temperature scaling | plane 0.035 → **0.014**; sub-plane 0.120 → 0.033 |
| Gate threshold (P(TT) ≥ 0.138, rule: validation TT recall ≥ 95 %) | TT accepted **95.7 %**; non-brain rejected **99.8 %**; neighbouring brain planes rejected only 42–78 % |
| **HC18 images accepted** (all 1,334) | **60.3 %**; HC < 100 mm (≈ ≤ 16 weeks): 31.5 %; HC ≥ 250 mm: 94.6 % |
| Added latency | 31 ms CPU / 61 ms GPU per image |

### 4.3 Feature B: measurement uncertainty (`outputs/uncertainty_metrics.json`)

8 passes (flip, zoom 0.9/1.1, gamma 0.8/1.25, flip + zoom), each undone before combining. Thresholds fixed on
**validation** (90th percentile): CV 1.05 %, disagreement 0.100. The single-pass HC stays the reported value
(lower validation MAE: 1.843 vs 1.861 mm).

| HC18 held-out test, n = 151 | Value |
|---|---|
| Spearman ρ(HC SD, \|HC error\|) | **0.27** (95 % CI 0.11–0.41, p < 0.001) |
| Spearman ρ(disagreement, \|HC error\|) | 0.15 (CI −0.02 to 0.31): not significant |
| Flagged "low reliability" | 17 / 151 (11 %); MAE flagged **3.68 mm** vs unflagged **1.99 mm** |
| Largest 10 % errors that were flagged | 6 / 16 |
| Round trip, Docker CPU backend | 0.23 s without TTA, **1.0 s** with TTA |

### 4.4 Feature C: CTG classifier and screening summary (`outputs/ctg_metrics.json`)

UCI Cardiotocography (Ayres-de-Campos et al., 2000): 2,126 records (Normal 1,655 / Suspect 295 / Pathological
176); 70/15/15 stratified split with duplicate records kept together (train 1,488 / val 317 / test 321);
class weights from train only; hyper-parameters fixed in advance.

| Held-out test, n = 321 | Macro-F1 (95 % CI) | ROC-AUC | Pathological recall |
|---|---|---|---|
| **XGBoost (deployed)** | **0.963** (0.936–0.986) | 0.987 | **27/27** (Wilson CI 0.875–1.000) |
| Random forest | 0.910 | 0.988 | 0.963 |
| Logistic regression | 0.778 | 0.960 | 0.889 |

XGBoost validation macro-F1 was 0.918 (the test figure may be optimistic); Suspect is the weakest class (recall
0.884). SHAP: Median, ASTV, ALTV and Mean FHR drive "Pathological". About 30 ms per record (Docker CPU).
Screening summary rules R1–R6 are in `ml/fusion.py` and `docs/MODEL_CARD.md`.

## 5. Test counts (`docs/TESTING.md`)

| Suite | Cases | Result |
|---|---|---|
| Backend API, T01–T45 | 62 | 62 pass |
| ML unit/integration, U01–U32 | 32 | 32 pass |
| Plane gate, P01–P13 | 13 | 13 pass |
| TTA uncertainty, Q01–Q09 | 14 | 14 pass |
| CTG, C01–C08 | 8 | 8 pass |
| Screening summary rules, F01–F14 | 16 | 16 pass |
| INTERGROWTH-21st, G01–G08 | 14 | 14 pass |
| Frontend smoke, S1–S3 | 3 | 3 pass |
| End-to-end UI in Chrome with the real models, E01–E21 | 21 | 21 pass |
| **Total** | **183** | **183 pass, 0 fail** |

Local Python run: **159 passed** (`docs/evidence/pytest_run.txt`). GitHub Actions CI on PR #14: Python job
**153 passed, 6 skipped** (the 6 need the raw HC18 / CTG data or the trained segmentation checkpoint, which are
not in the repository); ruff clean; frontend lint + build + smoke passed. UI run: `docs/evidence/ui_e2e_run.txt`.

## 6. Screenshots and figures (`docs/screenshots/`)

| File | Shows |
|---|---|
| `mask_bug_before_after.png` | **Bug evidence:** image / outline / old (empty) mask / fixed mask |
| `mask_sanity_grid.png` | 12 fixed masks overlaid on HC18 images |
| `training_curves.png` | U-Net loss and Dice/IoU per epoch |
| `test_hc_agreement.png` | Predicted vs reference HC scatter + Bland–Altman plot |
| `test_examples.png` | Worst / 2nd worst / median / best test cases |
| `official_test_set_qualitative.png` | Predictions on the unlabelled official HC18 test set |
| `plane_confusion.png` | **A:** plane and brain sub-plane confusion matrices (official test) |
| `plane_reliability.png` | **A:** reliability diagram before / after temperature scaling |
| `plane_gate_scores.png` | **A:** P(TT) distributions (TT vs other planes vs HC18) and the threshold |
| `plane_hc18_rejected.png` | **A:** HC18 images the gate rejects (mostly small, early-gestation heads) |
| `uncertainty_vs_error.png` | **B:** TTA SD / CV vs actual HC error, flagged cases in red |
| `uncertainty_examples.png` | **B:** uncertainty maps from lowest to highest SD |
| `ctg_confusion.png` | **C:** confusion matrices of the three CTG models |
| `ctg_shap_pathological.png` | **C:** SHAP summary toward "Pathological" |
| `ui_upload_empty.png`, `ui_upload_ready.png` | Web app: empty state; image uploaded, pixel spacing auto-filled |
| `ui_results.png` | Results after override: plane badge, images, HC ± SD, reliability, quality checks |
| `ui_plane_check_pass.png`, `ui_plane_check_warn.png`, `ui_plane_check_override.png` | **A:** PASS badge; femur image with HC withheld; override |
| `ui_uncertainty_map.png`, `ui_low_reliability.png` | **B:** uncertainty overlay; "Low: re-scan" case (`794_HC`) |
| `ui_ctg_result.png`, `ui_screening_summary_urgent.png`, `ui_ctg_and_summary.png` | **C:** CTG result with SHAP; summary escalated to Urgent review; full page |
| `ui_results_centile.png` | GA input: centile card and INTERGROWTH-21st chart |
| `ui_report_print.png`, `ui_report_centile_print.png`, `ui_report_screening_print.png`, `ui_report_example.pdf` | Printable report (incl. plane check, HC ± SD, CTG and summary) |
| `ui_results_dark.png`, `ui_mobile.png`, `ui_error_corrupted.png` | Dark theme, 390 px layout, error state |
| `ui_model_page.png` | Model page: segmentation, plane gate, uncertainty and CTG metrics |
| `swagger_docs.png` | FastAPI Swagger UI (`/docs`) with all endpoints |
| `test_run_pytest.png` | pytest run from Phase 1 (77 tests; current log: `docs/evidence/pytest_run.txt`) |

## 7. Features: done, cut, moved to Phase 2

| Item | Status |
|---|---|
| Phase-1 items 1–9 (mask fix, split, training, evaluation, GA centile, backend, frontend, tests + CI, Docker) | **Done** (PRs #1–#11) |
| **A. Standard-plane quality gate**: data, two-head model, official-test evaluation, calibration, gate in `/predict` with override, `/plane-check`, UI badge | **Done** (PR #12) |
| **B. Measurement uncertainty**: TTA, HC ± SD, uncertainty map, validation-chosen flag, Spearman evaluation, latency, API + UI | **Done** (PR #13) |
| **C. CTG branch + screening summary**: XGBoost vs LR/RF, SHAP, `/ctg/predict` (422 on bad input), rule-based summary, CTG form + summary card | **Done** (PR #14) |
| Cut | **Nothing was cut.** Not re-verified after brief 2: the full Docker Compose stack *with the frontend container* (the backend container was verified for A, B and C: `docs/evidence/docker_plane_gate_check.txt`, `docker_tta_latency.txt`, `docker_ctg_check.txt`; the frontend container last passed the UI check in Phase 1) |
| Docs (README, ARCHITECTURE, API, DATASET, MODEL_CARD, TESTING, ROADMAP, CHANGELOG, ADRs 0006–0008, HANDOFF) | **Done** |

Phase 2 ([`ROADMAP.md`](ROADMAP.md)): clinical validation with paired hospital data and ethics approval;
learned (feature-level) fusion and calibration of the combined output; plane gate fine-tuned on local images;
better uncertainty (MC dropout / ensembles, prediction intervals); GA estimation from HC; external validation;
ONNX/TensorRT; PostgreSQL history, JWT, audit log; cloud deployment; clinician usability study.

## 8. Bugs found: root cause and fix

| # | Bug | Root cause | Fix | Evidence |
|---|---|---|---|---|
| 1 | **"Dice 1.0" after 2 epochs with loss ≈ 1.0; no mask in the app** | Annotation outline (1 px) resized to 128 px with nearest-neighbour *before* filling → **empty masks for 999/999 images** | Fit ellipse to the outline at original resolution, fill, *then* resize (`ml/masks.py`) | `docs/evidence/README.md`, `mask_bug_before_after.png`, test U03 |
| 2 | Empty prediction scored as perfect | Dice `(2I + ε)/(P + T + ε)` = 1 when both are empty | Hard Dice/IoU return NaN for empty targets | test U06 |
| 3 | Trainer/dataset check could not find the data | Stale paths and CSV column name | Corrected | PR #1, #2 |
| 4 | Dashboard and API would give different HC | Dashboard had its own preprocessing | Shared `Segmenter` | PR #7 |
| 5 | Dashboard reported 806 instead of 999 pairs | Suffix matching missed `NNN_2HC` | Match on full stem | PR #7 |
| 6 | Ellipse drawn with the wrong orientation | OpenCV angle refers to the first axis | Normalise to the major axis | test U23 |
| 7 | HC18 pixel-size lookup ambiguous | Official test set reuses training filenames | SHA-256 lookup | `docs/DATASET.md` |
| 8 | Possible leakage between splits | Repeat scans / duplicates in HC18; 22 duplicate CTG records | Group-level splits | ADR-0002, `ml/ctg.py`, tests U14, C02 |
| 9 | Results title said "Fetal head segmented" on a rejected femur image | UI ignored `measurement_withheld` | "Segmentation preview (measurement withheld)" | PR #12 |
| 10 | TTA on CPU took ~6 s | A batch of 8 at 256 px was ~5× slower per image than single passes on this CPU | Passes run one at a time on CPU | `ml/uncertainty.py`, PR #13 |
| 11 | Response PNG encoding slow | `optimize=True` (~7× slower for 5 % smaller files) | Default compression | PR #13 |
| 12 | **CTG model did not load in Docker** (`ctg_model_loaded: false`) | Image resolved scikit-learn 1.9.1; the xgboost 2.1 scikit-learn wrapper cannot load models under it | Load a plain Booster saved at the best iteration; pin scikit-learn < 1.8 | `docs/evidence/docker_ctg_check.txt`, PR #14 |

## 9. Limitations and deviations (say these before the evaluators ask)

* **Plane gate on HC18: 60.3 % accepted.** FETAL_PLANES_DB covers 18–40 weeks (two Barcelona hospitals); HC18
  has many earlier, smaller heads (31.5 % accepted below HC 100 mm), different machines and anonymisation boxes,
  and the model confuses trans-thalamic with trans-ventricular. The threshold was **not** tuned on HC18; the
  override keeps HC18 demos usable. Demo image `006_HC.png` is one of the rejected early-gestation images.
* **Uncertainty is a modest signal** (ρ 0.27); 10 of the 16 largest errors are not flagged.
* **Fusion is illustrative.** Ultrasound (HC18, FETAL_PLANES_DB) and CTG (UCI) come from different patients;
  the summary is rule-based with no measured accuracy. Stated in the API response, UI and report.
* **CTG test set is small** (27 pathological records) and the CTG data are from one hospital in the 1990s.
* Phase-1 deviations still apply: split by examination group (not single images); HC measured at original
  resolution; GA estimation from HC not implemented (only verified centile equations); shadcn/ui components
  written by hand; Next.js pinned to 15.5.29; two build-time-only `npm audit` advisories (see `TESTING.md`).
* Latency numbers measured on a Windows laptop that throttles background processes; the Docker CPU figures are
  the reliable ones.

## 10. Module mapping (team)

| Member | Module | Main files | PRs |
|---|---|---|---|
| Pujari Varshitha | Dataset and preprocessing; FETAL_PLANES_DB data (A) | `ml/masks.py`, `ml/split.py`, `ml/dataset.py`, `docs/DATASET.md`, `data/splits/fetal_planes_split.csv` | #1, #3, #12 |
| Nachiketha K G | Model training and evaluation; plane classifier (A); TTA uncertainty (B) | `ml/model.py`, `ml/train.py`, `ml/evaluate.py`, `ml/planes.py`, `ml/train_planes.py`, `ml/evaluate_planes.py`, `ml/uncertainty.py`, `ml/evaluate_uncertainty.py`, `docs/MODEL_CARD.md` | #4, #5, #12, #13 |
| Vishwas K | Backend and integration; gate API (A), uncertainty API (B), CTG + fusion endpoints (C) | `backend/`, `ml/inference.py`, `ml/intergrowth.py`, `ml/fusion.py`, `docs/API.md`, Docker | #6, #10, #11, #12, #13, #14 |
| Rakshitha V | Frontend; plane badge (A), HC ± SD and uncertainty toggle (B), CTG form and summary card (C) | `frontend/` | #7, #10, #12, #13, #14 |
| Yashwanth Gowda M N | Testing, design docs; CTG model and tests (C) | `tests/`, `scripts/ui_e2e.py`, `scripts/check_mermaid.py`, `.github/workflows/ci.yml`, `docs/TESTING.md`, `docs/ARCHITECTURE.md`, `docs/adr/`, `ml/ctg.py`, `ml/train_ctg.py` | #8, #9, #14 |

## 11. How to demo (7 minutes)

0. `docker compose up` (CPU) serves both on the same ports (frontend container last verified in Phase 1; see §7), or: `scripts\run_backend.bat` and
   `cd frontend && npm run build && npm run start`. API docs: <http://localhost:8000/docs>.
1. **Plane gate + override:** upload `data/raw/HC18/training_set/006_HC.png` (held-out test). The badge says
   WARN (early-gestation image; looks like trans-cerebellum), HC is withheld; click "Measure anyway" →
   67.3 mm vs reference 67.84 mm, warning kept on the report.
2. **Wrong organ:** upload a FETAL_PLANES_DB femur image (e.g. `Patient00647_Plane5_1_of_1.png`) → "fetal femur
   (100 %) … please re-acquire".
3. **PASS + centile + uncertainty:** `544_HC.png`, GA 20 + 0 (example; HC18 has no GA) → PASS badge,
   HC 170.9 ± 1.6 mm, 42nd centile, reliability OK; toggle the uncertainty map.
4. **Low reliability:** `794_HC.png` → PASS badge but "Low: re-scan" (HC 301.4 ± 14.2 mm vs reference 308.5).
5. **CTG + summary:** on the same page, "Load example: Pathological" → Classify → PATHOLOGICAL with SHAP
   factors; the summary escalates to **Urgent review (R1)**. Point at the different-patients caveat.
6. Print report, `/model` page, and the bug story (`mask_bug_before_after.png`).

## 12. Pull requests

#1 mask bug · #2 restructure · #3 split · #4 training · #5 evaluation · #6 backend · #7 frontend + CI ·
#8 testing docs · #9 project docs · #10 GA centiles · #11 Docker Compose + handoff · #12 standard-plane gate ·
#13 TTA uncertainty · #14 CTG branch + screening summary.
