"""Pydantic v2 response models (they also define the OpenAPI schema at /docs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "Academic research prototype, not a medical device. Not clinically validated; "
    "must not be used for diagnosis or pregnancy-care decisions."
)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    plane_model_loaded: bool = Field(description="Standard-plane classifier loaded (quality gate active)")
    ctg_model_loaded: bool = Field(description="CTG classifier loaded (CTG endpoints available)")
    version: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    architecture: str
    input_size: int = Field(description="Network input is input_size x input_size grayscale")
    device: str
    checkpoint_epoch: int | None
    validation_dice: float | None
    test_metrics: dict | None = Field(description="Held-out test metrics from ml/evaluate.py, if available")
    training_data: str
    intended_use: str
    plane_classifier: dict | None = Field(
        default=None, description="Standard-plane gate: checkpoint details and official-test metrics, if loaded")
    uncertainty: dict | None = Field(
        default=None, description="TTA reliability thresholds (validation) and held-out test evaluation, if available")
    ctg_classifier: dict | None = Field(default=None, description="CTG XGBoost model and held-out test metrics")
    disclaimer: str = DISCLAIMER


class EllipseParams(BaseModel):
    center_x_px: float
    center_y_px: float
    semi_major_px: float
    semi_minor_px: float
    angle_deg: float
    semi_major_mm: float | None = None
    semi_minor_mm: float | None = None


class Quality(BaseModel):
    mean_confidence: float = Field(description="Mean foreground probability inside the predicted mask (0-1)")
    uncertain_fraction: float = Field(description="Share of pixels with probability 0.3-0.7 around the head")
    mask_area_fraction: float = Field(description="Predicted mask area / image area")
    touches_border: bool = Field(description="Head mask touches the image border (head may be cut off)")
    ellipse_fit_ratio: float | None = Field(description="Mask area / fitted ellipse area (1.0 = perfect ellipse)")
    flags: list[str] = Field(description="Human-readable warnings; empty when nothing looks unusual")


class PlaneCheck(BaseModel):
    """Standard-plane quality gate. HC is defined on the trans-thalamic (TT) plane."""

    status: Literal["pass", "warn", "unavailable"] = Field(
        description="pass: TT plane confirmed; warn: not TT or not confident enough; unavailable: no classifier")
    plane: str | None = Field(default=None, description="Most likely plane (FETAL_PLANES_DB classes)")
    plane_confidence: float | None = Field(default=None, description="Calibrated probability of that plane")
    brain_subplane: str | None = Field(default=None, description="Most likely brain plane, only for brain images")
    brain_subplane_confidence: float | None = None
    tt_probability: float | None = Field(
        default=None, description="Calibrated P(brain) x P(trans-thalamic | brain); the gate score")
    threshold: float | None = Field(default=None, description="Gate threshold, chosen on the validation set")
    plane_probabilities: dict[str, float] = Field(default_factory=dict)
    subplane_probabilities: dict[str, float] = Field(default_factory=dict)
    overridden: bool = Field(default=False, description="Measurement was forced despite a warning")
    message: str
    model_version: str | None = None


class Uncertainty(BaseModel):
    """Test-time augmentation: N augmented passes, each transform undone before combining."""

    n_passes: int
    hc_mean_mm: float | None = Field(description="Mean HC over the passes (mm); null without pixel spacing")
    hc_sd_mm: float | None = Field(description="SD of HC over the passes (mm); reported as HC +- SD")
    hc_cv_percent: float | None = Field(description="SD / mean x 100")
    hc_passes_mm: list[float | None] = Field(default_factory=list, description="HC of each pass (mm)")
    disagreement_fraction: float = Field(description="Pixels where the pass masks disagree / their union")
    low_reliability: bool = Field(description="True if CV or disagreement exceeds its validation-chosen threshold")
    reasons: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] | None = Field(default=None, description="From outputs/uncertainty_calibration.json")
    tta_ms: float


class GrowthAssessment(BaseModel):
    """Measured HC compared with the INTERGROWTH-21st fetal standard (not a diagnosis)."""

    gestational_age_weeks: float
    z_score: float
    centile: float = Field(description="0-100, from the standard normal CDF of the z-score")
    median_mm: float = Field(description="Standard's 50th centile HC at this gestational age")
    p3_mm: float
    p97_mm: float
    within_p3_p97: bool
    reference: str


class CentilePoint(BaseModel):
    ga_weeks: float
    p3: float
    p10: float
    p50: float
    p90: float
    p97: float


class CentileCurves(BaseModel):
    reference: str
    unit: str = "mm"
    points: list[CentilePoint]


class PredictionResponse(BaseModel):
    image_width: int
    image_height: int
    pixel_size_mm: float | None
    plane_check: PlaneCheck = Field(description="Standard-plane quality gate, run before measurement")
    measurement_withheld: bool = Field(
        default=False, description="True when the plane check warned and override_plane_check was not set: "
                                   "HC, area in mm2 and the centile are then not reported")
    head_detected: bool
    head_circumference_mm: float | None = Field(description="Ellipse perimeter (Ramanujan) x pixel size")
    head_area_mm2: float | None
    head_area_px: int
    ellipse: EllipseParams | None
    quality: Quality
    growth: GrowthAssessment | None = Field(
        default=None, description="Only when gestational_age_weeks was given and HC could be measured")
    uncertainty: Uncertainty | None = Field(
        default=None, description="Test-time-augmentation uncertainty; null when uncertainty=false")
    uncertainty_png_base64: str | None = Field(
        default=None, description="Per-pixel uncertainty (SD of the pass probabilities) as a heat-map overlay, PNG")
    mask_png_base64: str = Field(description="Binary mask, PNG, original resolution, base64")
    overlay_png_base64: str = Field(description="Image with mask and fitted ellipse drawn, PNG, base64")
    inference_ms: float
    model_version: str
    disclaimer: str = DISCLAIMER


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
