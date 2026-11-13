"""Pydantic models for the CTG branch and the screening summary (Feature C)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from backend.app.schemas.prediction import DISCLAIMER
from ml.ctg import FEATURES
from ml.fusion import CAVEATS

# 21 required features, each range-checked (422 on missing, extra, non-numeric, NaN/inf or out-of-range values).
CTGFeatures = create_model(
    "CTGFeatures",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"description": "21 SisPorto CTG features (UCI order)"}),
    **{f.name: (float, Field(ge=f.lo, le=f.hi, allow_inf_nan=False,
                             description=f"{f.description}" + (f" ({f.unit})" if f.unit else "")))
       for f in FEATURES},
)

DIFFERENT_PATIENTS = ("The CTG model (UCI Cardiotocography) and the ultrasound models (HC18, FETAL_PLANES_DB) were "
                      "trained on different patients.")


class CTGFactor(BaseModel):
    feature: str
    description: str
    unit: str
    value: float
    shap: float = Field(description="SHAP contribution (log-odds) toward the predicted class")
    direction: Literal["towards", "away from"]


class CTGPrediction(BaseModel):
    predicted_class: Literal["Normal", "Suspect", "Pathological"]
    probabilities: dict[str, float]
    top_factors: list[CTGFactor] = Field(description="Largest SHAP contributions toward the predicted class")
    model_version: str
    note: str = DIFFERENT_PATIENTS
    disclaimer: str = DISCLAIMER


class CTGExample(BaseModel):
    record: str = Field(description="UCI record id (SegFile)")
    label: str = Field(description="Expert NSP label")
    split: str
    features: dict[str, float]


class ScreeningRequest(BaseModel):
    """Results of the other endpoints; every field is optional (missing inputs are reported as not assessed)."""

    model_config = ConfigDict(extra="forbid")

    plane_status: Literal["pass", "warn", "unavailable"] | None = None
    plane_overridden: bool = False
    measurement_withheld: bool = False
    hc_mm: float | None = Field(default=None, gt=0, le=500)
    hc_z_score: float | None = Field(default=None, ge=-10, le=10)
    hc_centile: float | None = Field(default=None, ge=0, le=100)
    low_reliability: bool | None = None
    ctg_class: Literal["Normal", "Suspect", "Pathological"] | None = None
    ctg_probability: float | None = Field(default=None, ge=0, le=1)


class ScreeningDriver(BaseModel):
    input: str
    finding: str
    level: str
    rule: str


class ScreeningSummaryResponse(BaseModel):
    level: Literal["Routine", "Review recommended", "Urgent review", "Insufficient data"]
    drivers: list[ScreeningDriver] = Field(description="Inputs that raised the level, most severe first")
    unremarkable: list[ScreeningDriver]
    not_assessed: list[str]
    rules_version: str
    rules: list[str]
    caveats: list[str] = Field(default_factory=lambda: list(CAVEATS))
    disclaimer: str = DISCLAIMER
