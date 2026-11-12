# legacy/

Files kept for history only. **Nothing here is used by the current application.**

| File | What it was | Why it is retired |
|---|---|---|
| `train_multimodal_ecg.py`, `config.py`, `DATASET_GUIDE_ecg.md` | Ultrasound + ECG view-classification experiment | Imports `MultimodalFetalModel`, which never existed; no paired ultrasound/ECG data exists. The CTG branch is planned for Phase-2 (see `docs/ROADMAP.md`). |
| `app_original.py` | First Streamlit prototype | Replaced by `dashboard/app.py`. |
| `run_api.bat` | Stub stating there was no backend | Replaced by the FastAPI service in `backend/`. |
| `UPGRADE_NOTES.md` | Notes from an earlier dashboard rework | Superseded by `CHANGELOG.md`. |
