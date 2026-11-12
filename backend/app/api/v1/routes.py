"""Version 1 HTTP endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app.api.deps import get_model_service, get_settings_dep, require_model
from backend.app.core.config import Settings
from backend.app.schemas.prediction import (
    CentileCurves,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)
from backend.app.services import image_service
from backend.app.services.model_service import ModelService
from ml import intergrowth

router = APIRouter(prefix="/api/v1")

ERRORS = {
    400: {"model": ErrorResponse, "description": "Corrupted, empty, too small or undecodable image"},
    413: {"model": ErrorResponse, "description": "File larger than the upload limit (10 MB by default)"},
    415: {"model": ErrorResponse, "description": "Not a PNG or JPEG image"},
    422: {"model": ErrorResponse, "description": "Invalid parameters (e.g. pixel size out of range)"},
    503: {"model": ErrorResponse, "description": "Model checkpoint not loaded"},
}


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: Annotated[ModelService, Depends(get_model_service)],
           settings: Annotated[Settings, Depends(get_settings_dep)]) -> HealthResponse:
    """Liveness and readiness: ``degraded`` means the API is up but the model is not loaded."""
    return HealthResponse(status="ok" if service.ready else "degraded", model_loaded=service.ready,
                          version=settings.version)


@router.get("/model-info", response_model=ModelInfo, tags=["model"], responses={503: ERRORS[503]})
def model_info(service: Annotated[ModelService, Depends(require_model)]) -> ModelInfo:
    """Model architecture, checkpoint details and held-out test metrics."""
    return service.info()


@router.post("/predict", response_model=PredictionResponse, tags=["model"], responses=ERRORS)
async def predict(
    service: Annotated[ModelService, Depends(require_model)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    file: Annotated[UploadFile, File(description="Ultrasound image (PNG or JPEG, max 10 MB)")],
    pixel_size_mm: Annotated[float | None, Form(
        gt=0, le=1.0, description="Pixel spacing in mm/pixel (HC18 images: about 0.05-0.3). "
                                  "Required to report head circumference in mm.")] = None,
    gestational_age_weeks: Annotated[float | None, Form(
        ge=intergrowth.GA_MIN_WEEKS, le=intergrowth.GA_MAX_WEEKS,
        description="Exact gestational age in weeks (14-40), e.g. 22.43 for 22+3. "
                    "If given (with pixel spacing), HC is compared with the INTERGROWTH-21st standard.")] = None,
) -> PredictionResponse:
    """Segment the fetal head, fit an ellipse and measure head circumference.

    Steps: check type -> read with size limit -> verify and decode -> U-Net -> ellipse fit -> HC
    -> (optional) INTERGROWTH-21st centile for the given gestational age.
    """
    image_service.check_declared_type(file, settings)
    data = await image_service.read_limited(file, settings.max_upload_bytes)
    gray = image_service.decode_grayscale(data, settings)
    # Inference is CPU/GPU bound; run it in a worker thread so the event loop stays responsive.
    return await run_in_threadpool(service.predict, gray, pixel_size_mm, gestational_age_weeks)


@router.get("/reference/hc-centiles", response_model=CentileCurves, tags=["reference"])
def hc_centiles() -> CentileCurves:
    """INTERGROWTH-21st fetal HC centile curves (3rd, 10th, 50th, 90th, 97th), 14-40 weeks, for charts."""
    return CentileCurves(reference=intergrowth.REFERENCE, points=intergrowth.centile_curves(step_weeks=0.5))
