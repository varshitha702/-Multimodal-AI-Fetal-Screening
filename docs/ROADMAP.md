# Roadmap

## Phase 1 (70 %, review 14 Nov 2026): done

| Item | Status | Evidence |
|---|---|---|
| Fix the HC18 empty-mask bug and the fake Dice metric | Done | `docs/evidence/README.md` |
| Leakage-safe group split (70/15/15, seed 42) | Done | `data/splits/hc18_split.csv` |
| GPU training (256 px, AMP, augmentation, early stopping) | Done | val Dice 0.976 |
| Held-out evaluation incl. HC error in mm | Done | test Dice 0.973, HC MAE 2.18 mm |
| FastAPI backend (health, model-info, predict) | Done | `docs/API.md` |
| Next.js frontend with report, light/dark, responsive | Done | `docs/screenshots/ui_*.png` |
| Tests + CI (pytest, smoke, end-to-end, GitHub Actions) | Done | `docs/TESTING.md` |
| Documentation (architecture, model card, dataset card, ADRs) | Done | `docs/` |

Items 5 (gestational-age centiles) and 9 (Docker Compose) of the Phase-1 brief were optional "if time
allows"; their final status is recorded in `docs/HANDOFF.md`.

## Phase 2 (remaining 30 %)

1. **Standard-plane quality gate.** Train a classifier on FETAL_PLANES_DB to check that an image is a
   trans-thalamic head plane before measuring; reject or warn otherwise.
2. **CTG (cardiotocography) branch** with confidence-aware decision-level fusion with the ultrasound branch
   (the "multimodal" part of the project title).
3. **Uncertainty and calibration.** Test-time augmentation and/or MC dropout; calibrate the confidence shown
   to users; replace the heuristic quality flags with calibrated ones.
4. **Stronger model and validation.** Pretrained-encoder U-Net (e.g. ResNet/EfficientNet encoder), k-fold
   cross-validation, external validation on another dataset.
5. **Clinical context (if verified).** Gestational-age estimation / HC-for-GA centiles from a published
   standard (e.g. INTERGROWTH-21st), implemented only with formulas taken from the cited source and unit-tested
   against its published tables.
6. **Optimisation.** ONNX export, TensorRT / ONNX Runtime inference, CPU-only deployment profile.
7. **Persistence and security.** PostgreSQL history of analyses, JWT authentication, audit log.
8. **Deployment.** Docker Compose for backend + frontend (if not completed in Phase 1), then cloud deployment.
9. **Usability study** with clinicians (task-based evaluation of the UI and report).
