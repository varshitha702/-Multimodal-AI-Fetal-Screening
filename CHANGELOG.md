# Changelog

All notable changes are listed here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

### Changed
- Repository reorganised into `ml/`, `backend/`, `frontend/`, `dashboard/`, `scripts/`, `tests/`, `docs/` (#2).

### Removed
- Unused ultrasound + ECG experiment, original app and API stub moved to `legacy/` (#2).
