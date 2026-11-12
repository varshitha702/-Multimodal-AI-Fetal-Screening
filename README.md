# FetalAI: Multimodal AI Fetal Screening

[![CI](https://github.com/varshitha702/-Multimodal-AI-Fetal-Screening/actions/workflows/ci.yml/badge.svg)](https://github.com/varshitha702/-Multimodal-AI-Fetal-Screening/actions/workflows/ci.yml)

B.Tech Major Project, REVA University. FetalAI segments the fetal head in a 2-D ultrasound image with a
U-Net, fits an ellipse to it and reports the **head circumference (HC) in millimetres**, through a FastAPI
backend and a Next.js web app.

> **Academic research prototype, not a medical device.** It is not clinically validated and must not be used
> for diagnosis, reassurance, triage or pregnancy-care decisions.

![Results screen](docs/screenshots/ui_results.png)

## Results (held-out test split, 151 images)

| Metric | Value |
|---|---|
| Dice | **0.973** ± 0.028 |
| IoU | 0.948 ± 0.046 |
| HC mean absolute error | **2.18 mm** (1.48 %) |
| HC bias, 95 % limits of agreement | −0.10 mm, [−5.89, +5.69] mm |

Trained on HC18 (van den Heuvel et al., *PLoS ONE* 2018) with a group-level split so no fetus is in both
training and test. Details and limitations: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Features

* **Segmentation + measurement:** U-Net → largest component → ellipse fit → Ramanujan perimeter × pixel spacing.
* **REST API** (FastAPI): `/api/v1/health`, `/api/v1/model-info`, `/api/v1/predict`; validation with clear
  400/413/415/422/503 errors; OpenAPI docs at `/docs`.
* **Web app** (Next.js + Tailwind + shadcn/ui): drag-and-drop upload, pixel spacing auto-filled for HC18
  images, original / overlay / mask views, metric cards, quality warnings, printable report, JSON download,
  model page, light/dark, mobile layout.
* **HC-for-gestational-age centile** (optional GA input) from the INTERGROWTH-21st fetal standard
  (Papageorghiou et al., *Lancet* 2014), verified against every cell of the official published table, with a
  centile chart. A comparison with a population standard, not a diagnosis.
* **Research dashboard** (Streamlit) using the same inference code.
* **Reproducible pipeline:** fixed-seed split, GPU training, evaluation script, 92 automated tests, CI.

## Architecture

```mermaid
flowchart LR
    U([User]) --> FE[Next.js frontend :3000]
    FE -->|/api/v1/predict| API[FastAPI backend :8000]
    API --> INF[ml.inference.Segmenter]
    DB[Streamlit dashboard] --> INF
    INF --> M[(U-Net checkpoint)]
    INF --> GEO[Ellipse fit + HC mm]
```

More diagrams (flowchart, UML use-case/class/sequence, DFD level 0/1): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quickstart (Windows, from the repository root)

Prerequisites: Python 3.11, Node 22, an NVIDIA GPU for training (inference also runs on CPU).

```powershell
# 1. Python environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dev.txt

# 2. Dataset: download HC18 (doi:10.5281/zenodo.1327317) into data/raw/HC18 (see data/README.md)
python scripts\check_hc18.py

# 3. Train and evaluate (about 12 min on an RTX 4050 Laptop GPU)
python -m ml.train
python -m ml.evaluate

# 4. Backend on http://localhost:8000 (docs at /docs)
scripts\run_backend.bat

# 5. Frontend on http://localhost:3000 (second terminal)
cd frontend
npm install
npm run dev
```

Optional research dashboard: `streamlit run dashboard\app.py`. Tests: `python -m pytest` and see
[`docs/TESTING.md`](docs/TESTING.md).

The trained checkpoint (`outputs/hc18_unet_best.pth`, 31 MB) is not committed; step 3 regenerates it.

## Screenshots

| | |
|---|---|
| Upload | ![](docs/screenshots/ui_upload_ready.png) |
| Results (dark) | ![](docs/screenshots/ui_results_dark.png) |
| GA centile | ![](docs/screenshots/ui_results_centile.png) |
| Printable report | ![](docs/screenshots/ui_report_print.png) |
| Model page | ![](docs/screenshots/ui_model_page.png) |
| API docs | ![](docs/screenshots/swagger_docs.png) |
| Training curves | ![](docs/screenshots/training_curves.png) |
| Bug found: empty masks (before / after) | ![](docs/screenshots/mask_bug_before_after.png) |

## Repository layout

```text
ml/          data pipeline, U-Net, training, evaluation, geometry, inference (Python package)
backend/     FastAPI service (api / core / schemas / services)
frontend/    Next.js + TypeScript + Tailwind web app
dashboard/   Streamlit research dashboard (fallback UI)
scripts/     dataset check, figures, launchers, end-to-end UI check
tests/       pytest (ML + backend API)
docs/        architecture, API, dataset/model cards, testing, roadmap, ADRs, handoff, screenshots
data/        datasets (gitignored) + committed split CSV
outputs/     metrics, logs and plots (checkpoint gitignored)
legacy/      retired experiments, kept for history
```

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [API](docs/API.md) · [Dataset card](docs/DATASET.md) ·
[Model card](docs/MODEL_CARD.md) · [Testing](docs/TESTING.md) · [Roadmap](docs/ROADMAP.md) ·
[ADRs](docs/adr/) · [Handoff](docs/HANDOFF.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Team

| Member | Module |
|---|---|
| Pujari Varshitha | Dataset and preprocessing |
| Nachiketha K G | Model training and evaluation |
| Vishwas K | FastAPI backend and integration |
| Rakshitha V | Frontend |
| Yashwanth Gowda M N | Requirements, design docs, testing |

## Citation and licence of data

HC18: T. L. A. van den Heuvel, D. de Bruijn, C. L. de Korte, B. van Ginneken, "Automated measurement of fetal
head circumference using 2D ultrasound images", *PLoS ONE* 13(8): e0200412, 2018. Data: Zenodo
doi:10.5281/zenodo.1327317.

**Disclaimer:** research and teaching use only. Not a medical device.
