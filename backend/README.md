# backend/: FastAPI inference service

Owner: **Vishwas K** (FastAPI backend and integration). Full endpoint reference: [`docs/API.md`](../docs/API.md).

## Run

From the repository root (the trained checkpoint must exist at `outputs/hc18_unet_best.pth`):

```powershell
python -m uvicorn backend.app.main:app --port 8000 --no-access-log
# or: scripts\run_backend.bat
```

* Swagger UI: <http://localhost:8000/docs> · ReDoc: <http://localhost:8000/redoc>
* Settings: environment variables with the `FETALAI_` prefix or a `.env` file (see `.env.example`).

## Layout

```text
backend/app/
├── main.py                 # app factory: lifespan (loads model once), CORS, request logging, routers
├── api/
│   ├── deps.py             # dependencies: settings, model service, require_model (503)
│   └── v1/routes.py        # GET /health, GET /model-info, POST /predict
├── core/
│   ├── config.py           # Settings (pydantic-settings, env prefix FETALAI_)
│   ├── errors.py           # APIError subclasses -> 400/413/415/503, 422 handler, one error format
│   └── logging.py          # JSON-lines structured logging
├── schemas/prediction.py   # Pydantic v2 response models (also the OpenAPI schema)
└── services/
    ├── image_service.py    # chunked upload with size limit, type check, PIL verify, decode, PNG/base64
    └── model_service.py    # wraps ml.inference.Segmenter; builds response, quality flags, overlay
```

The model code itself lives in `ml/` (`ml/inference.py`), so the dashboard, evaluation and API all run
exactly the same preprocessing and post-processing.

## Tests

```powershell
python -m pytest tests/backend -v     # T01-T20, uses a random-weight checkpoint (no GPU needed)
```
