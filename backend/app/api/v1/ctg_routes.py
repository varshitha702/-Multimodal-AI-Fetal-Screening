"""CTG and screening-summary endpoints (Feature C)."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from backend.app.api.deps import require_ctg
from backend.app.schemas.ctg import (
    CTGExample,
    CTGFeatures,
    CTGPrediction,
    ScreeningRequest,
    ScreeningSummaryResponse,
)
from backend.app.schemas.prediction import ErrorResponse
from backend.app.services.ctg_service import CTGService

router = APIRouter(prefix="/api/v1")
ERRORS = {
    422: {"model": ErrorResponse, "description": "Missing, extra, non-numeric or out-of-range fields"},
    503: {"model": ErrorResponse, "description": "CTG model not loaded"},
}


@router.post("/ctg/predict", response_model=CTGPrediction, tags=["ctg"], responses=ERRORS)
async def ctg_predict(features: CTGFeatures,  # type: ignore[valid-type]
                      service: Annotated[CTGService, Depends(require_ctg)]) -> CTGPrediction:
    """Classify one CTG record (21 SisPorto features) as Normal / Suspect / Pathological (NSP).

    Returns XGBoost class probabilities and the five largest SHAP contributions toward the predicted class.
    Trained on UCI Cardiotocography, i.e. on different patients from the ultrasound models.
    """
    return await run_in_threadpool(service.predict, features.model_dump())


@router.get("/ctg/examples", response_model=list[CTGExample], tags=["ctg"])
def ctg_examples() -> list[CTGExample]:
    """Example records from the held-out CTG test split (for the UI's "load example" button)."""
    return CTGService.examples()


@router.post("/screening/summary", response_model=ScreeningSummaryResponse, tags=["screening"],
             responses={422: ERRORS[422]})
def screening_summary(req: ScreeningRequest) -> ScreeningSummaryResponse:
    """Combine the plane check, HC centile, HC reliability and CTG class into one rule-based summary.

    Levels: Routine / Review recommended / Urgent review (a pathological CTG always escalates). The ultrasound
    and CTG datasets come from different patients, so the rules are illustrative and uncalibrated.
    """
    return CTGService.summary(req)
