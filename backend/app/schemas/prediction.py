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


class PredictionResponse(BaseModel):
    image_width: int
    image_height: int
    pixel_size_mm: float | None
    head_detected: bool
    head_circumference_mm: float | None = Field(description="Ellipse perimeter (Ramanujan) x pixel size")
    head_area_mm2: float | None
    head_area_px: int
    ellipse: EllipseParams | None
    quality: Quality
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
