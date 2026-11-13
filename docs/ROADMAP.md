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
| Docker Compose (backend CPU + frontend), verified end to end | Done | `docker-compose.yml` |
| HC-for-gestational-age centile (INTERGROWTH-21st), verified against the published table | Done | `ml/intergrowth.py`, `tests/test_intergrowth.py` |

## Phase-1 extension (brief 2, 13 Nov 2026 feature freeze)

| Item | Status | Evidence |
|---|---|---|
| **A. Standard-plane quality gate** (FETAL_PLANES_DB, two-head EfficientNet-B0, temperature scaling, gate in `/predict`, UI badge, override) | Done | `docs/MODEL_CARD.md` (plane acc 94.0 %, ECE 0.014, TT accepted 95.7 %, HC18 accepted 60.3 %), ADR-0006 |
| **B. Measurement uncertainty** (8-pass test-time augmentation, HC ± SD, uncertainty map, validation-chosen low-reliability flag) | Done | `docs/MODEL_CARD.md` addendum (Spearman ρ 0.27, flagged MAE 3.68 vs 1.99 mm), ADR-0007 |
| **C. CTG branch + rule-based screening summary** (XGBoost vs LR/RF, SHAP, `/ctg/predict`, `/screening/summary`, CTG form, summary card) | Done | `docs/MODEL_CARD.md` (CTG macro-F1 0.963, Pathological recall 27/27), ADR-0008 |


## Phase 2 (remaining 30 %)

1. **Plane gate on local data.** The gate (done, see above) accepts only 60 % of HC18 images because of
   domain shift and early gestation; fine-tune and re-validate it on paired local hospital images, and add
   within-plane quality scoring (landmarks, symmetry).
2. **Learned multimodal fusion.** The CTG branch and a rule-based summary are done (see above); with paired
   ultrasound + CTG data from the same pregnancies (ethics approval needed), learn feature-level fusion and
   calibrate the combined risk; validate the CTG model on external tracings.
3. **Better uncertainty.** TTA is done (modest correlation with error, ρ 0.27); next: MC dropout or deep
   ensembles, and calibrating the SD into a prediction interval on paired local data.
4. **Stronger model and validation.** Pretrained-encoder U-Net (e.g. ResNet/EfficientNet encoder), k-fold
   cross-validation, external validation on another dataset.
5. **Gestational-age estimation from HC** (the reverse direction; HC-for-GA centiles are already done).
   Only with an equation taken from its cited source and unit-tested against that source's published values.
6. **Optimisation.** ONNX export, TensorRT / ONNX Runtime inference, CPU-only deployment profile.
7. **Persistence and security.** PostgreSQL history of analyses, JWT authentication, audit log.
8. **Deployment.** Cloud deployment of the Docker Compose stack (GPU image optional).
9. **Usability study** with clinicians (task-based evaluation of the UI and report).
