# Changelog

All notable changes are listed here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - brief 2 (differentiating features)

### Added
- **Standard-plane quality gate (Feature A):** two-head EfficientNet-B0 (plane + brain sub-plane) trained on
  FETAL_PLANES_DB with a patient-level split; temperature scaling per head; gate threshold chosen on validation
  (TT recall ≥ 95 %). Official test: plane accuracy 94.0 % (macro-F1 0.934), sub-plane accuracy 82.7 %
  (macro-F1 0.721), ECE 0.035 → 0.014; 95.7 % of TT images accepted, 99.8 % of non-brain images rejected;
  60.3 % of HC18 images accepted (reported, not tuned). 224 px run kept as an ablation.
- API: `plane_check` block and `measurement_withheld` in `/predict`, `override_plane_check` form field,
  new `POST /api/v1/plane-check`, `plane_model_loaded` in `/health`, `plane_classifier` in `/model-info`.
- UI: plane-check badge (PASS / WARN / N/A, calibrated confidence, P(TT) vs threshold) before the HC result,
  "Measure anyway (override)" button, plane check on the printed report, plane-gate section on the model page.
- Tests: T26–T32, P01–P13, E03/E04/E12/E13 (114 total). ADR-0006. Docker mounts the plane checkpoint.
- **Measurement uncertainty (Feature B):** 8-pass test-time augmentation with transforms undone before
  combining; HC ± SD, uncertainty map, mask disagreement; "low reliability, re-scan" flag with thresholds at the
  validation 90th percentile (CV 1.05 %, disagreement 0.100). Held-out test: Spearman ρ(SD, |error|) 0.27
  (95 % CI 0.11–0.41); flagged 11 %, MAE 3.68 vs 1.99 mm. API `uncertainty` field and block,
  `uncertainty_png_base64`; UI HC ± SD, reliability card, overlay toggle; T33–T38, Q01–Q09, E13–E15
  (137 total); ADR-0007.
- **CTG branch and screening summary (Feature C):** UCI Cardiotocography loader (per-second features from the
  `Data` sheet, duplicate-safe stratified 70/15/15 split); XGBoost with balanced class weights from train only,
  compared with logistic regression and random forest. Held-out test: XGBoost macro-F1 0.963 (95 % CI
  0.936–0.986), ROC-AUC 0.987, Pathological recall 27/27; SHAP top factors. Rule-based screening summary
  (R1–R6; a pathological CTG always escalates; inputs not given are "not assessed"; different-patients caveat).
  API `/ctg/predict` (422 on invalid features), `/ctg/examples`, `/screening/summary`, `ctg_model_loaded`,
  `ctg_classifier`; UI CTG form with held-out examples, summary card, report section, model-page section;
  T39–T45, C01–C08, F01–F14, E15–E18 (183 total); ADR-0008; `scripts/check_mermaid.py`.

### Changed
- Response PNGs are encoded without `optimize` (about 7× faster, about 5 % larger).
- New runtime dependencies: scikit-learn (< 1.8), scipy, xgboost, shap, xlrd.

### Fixed
- CTG model failed to load in the Docker image (scikit-learn 1.9 vs the xgboost 2.1 wrapper); it is now loaded
  as a plain Booster truncated to the best iteration.

## [1.0.0] - 2026-11-12 (Phase-1 review build)

### Fixed
- **HC18 empty-mask bug**: annotations were resized before being filled, so 999/999 training masks were empty
  and the reported "Dice 1.0" was invalid. Masks are now filled at original resolution (#1).
- Dice/IoU no longer score empty-vs-empty as 1.0 (#1).
- Stale dataset paths and CSV column name in the trainer and dataset check (#1, #2).
- Streamlit dashboard: inference now matches the API; dataset audit matched 806 instead of 999 pairs (#7).

### Added
- Leakage-safe 70/15/15 split by examination group, committed as CSV (#3).
- GPU training at 256 px with AMP, augmentation, LR schedule and early stopping; best val Dice 0.976 (#4).
- Held-out test evaluation: Dice 0.973, IoU 0.948, HC MAE 2.18 mm; model card (#5).
- Ellipse geometry (Ramanujan perimeter) and a shared `Segmenter` for all consumers (#5).
- FastAPI backend: `/api/v1/health`, `/api/v1/model-info`, `/api/v1/predict`, with validation and
  structured logging (#6).
- Next.js frontend: upload, auto-filled HC18 pixel spacing, results, print report, model page, light/dark (#7).
- Tests: 23 API, 32 ML, 3 frontend smoke, 10 end-to-end UI checks; GitHub Actions CI (#6, #7, #8).
- Documentation: architecture with Mermaid diagrams, API, dataset card, model card, testing, roadmap, ADRs,
  handoff.

- Gestational-age input and HC-for-GA centile from the INTERGROWTH-21st fetal standard (Papageorghiou et al.,
  Lancet 2014), verified against all 189 cells of the official table; centile chart and report section;
  `/api/v1/reference/hc-centiles` (#10).

- Docker Compose for backend (CPU) + frontend, verified with the end-to-end UI check (#11).

### Changed
- Repository reorganised into `ml/`, `backend/`, `frontend/`, `dashboard/`, `scripts/`, `tests/`, `docs/` (#2).

### Removed
- Unused ultrasound + ECG experiment, original app and API stub moved to `legacy/` (#2).
