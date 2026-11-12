# FetalAI: Multimodal AI Fetal Screening

B.Tech Major Project (REVA University). FetalAI segments the fetal head in 2-D ultrasound
images (HC18 dataset) with a U-Net and measures head circumference (HC) in millimetres.

> **Academic research prototype, not a medical device.** It is not clinically validated and
> must not be used for diagnosis, reassurance, triage or pregnancy-care decisions.

## Repository layout

```text
Multimodal-AI-Fetal-Screening/
├── ml/            # data pipeline, U-Net, training and evaluation (Python package)
├── backend/       # FastAPI inference service (api / services / schemas / core)
├── frontend/      # Next.js + TypeScript + Tailwind web client
├── dashboard/     # Streamlit research dashboard (fallback UI)
├── scripts/       # utilities: dataset check, mask sanity figures, launchers
├── tests/         # pytest suite
├── docs/          # documentation, screenshots, evidence
├── data/          # datasets (gitignored; see data/README.md)
├── outputs/       # generated metrics and plots (checkpoints are gitignored)
├── legacy/        # retired experiments, kept for history only
├── pyproject.toml # pytest and ruff configuration
└── requirements.txt
```

## Quickstart (Windows, from the repository root)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts\check_hc18.py      # verify the dataset (see data/README.md)
python -m pytest                  # run the tests
python -m ml.train                # train the U-Net
streamlit run dashboard\app.py    # research dashboard
```

## Status

Phase-1 is in progress. The HC18 empty-mask bug has been found and fixed; see
[`docs/evidence/README.md`](docs/evidence/README.md).

## Team

| Member | Module |
|---|---|
| Pujari Varshitha | Dataset and preprocessing |
| Nachiketha K G | Model training and evaluation |
| Vishwas K | FastAPI backend and integration |
| Rakshitha V | Frontend |
| Yashwanth Gowda M N | Requirements, design docs, testing |
