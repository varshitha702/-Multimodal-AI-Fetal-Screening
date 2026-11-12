"""Wraps the U-Net segmenter: loaded once at startup, turns a prediction into an API response."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from backend.app.core.config import REPO_ROOT, Settings
from backend.app.schemas.prediction import EllipseParams, ModelInfo, PredictionResponse, Quality
from backend.app.services.image_service import png_base64
from ml.inference import Prediction, Segmenter

log = logging.getLogger("fetalai.model")
TEST_METRICS_PATH = REPO_ROOT / "outputs" / "test_metrics.json"

# Quality-flag thresholds (heuristics for user feedback, not clinical criteria)
LOW_CONFIDENCE = 0.85
HIGH_UNCERTAINTY = 0.05
MIN_AREA_FRACTION = 0.01
POOR_ELLIPSE_FIT = 0.10  # |mask area / ellipse area - 1|


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.segmenter: Segmenter | None = None
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.segmenter is not None

    def load(self) -> None:
        path = Path(self.settings.checkpoint_path)
        try:
            t0 = time.perf_counter()
            self.segmenter = Segmenter(path, device=self.settings.device)
            log.info("model loaded", extra={"checkpoint": str(path), "device": str(self.segmenter.device),
                                            "seconds": round(time.perf_counter() - t0, 2)})
        except FileNotFoundError:
            self.load_error = f"Checkpoint not found: {path}"
            log.error("model not loaded", extra={"reason": self.load_error})
        except Exception as exc:  # corrupt checkpoint, CUDA problem, ...
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model not loaded")

    def info(self) -> ModelInfo:
        seg = self.segmenter
        test_metrics = None
        if TEST_METRICS_PATH.is_file():
            test_metrics = json.loads(TEST_METRICS_PATH.read_text())
        return ModelInfo(
            name="FetalAI HC18 U-Net",
            architecture=seg.meta.get("architecture", "UNet"),
            input_size=seg.image_size,
            device=str(seg.device),
            checkpoint_epoch=seg.meta.get("epoch"),
            validation_dice=seg.meta.get("best_val_dice"),
            test_metrics=test_metrics,
            training_data="HC18 grand challenge training set (van den Heuvel et al., 2018), group-level split",
            intended_use="Fetal head segmentation and head-circumference measurement on 2-D ultrasound "
                         "trans-thalamic plane images, for research and teaching only.",
        )

    def predict(self, gray: np.ndarray, pixel_size_mm: float | None) -> PredictionResponse:
        t0 = time.perf_counter()
        pred = self.segmenter.predict(gray, pixel_size_mm)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        h, w = gray.shape
        area_px = int(pred.mask.sum())
        e = pred.ellipse
        ellipse = None
        if e is not None:
            ellipse = EllipseParams(
                center_x_px=round(e.cx, 2), center_y_px=round(e.cy, 2),
                semi_major_px=round(e.semi_major, 2), semi_minor_px=round(e.semi_minor, 2),
                angle_deg=round(e.angle_deg, 2),
                semi_major_mm=round(e.semi_major * pixel_size_mm, 2) if pixel_size_mm else None,
                semi_minor_mm=round(e.semi_minor * pixel_size_mm, 2) if pixel_size_mm else None,
            )
        response = PredictionResponse(
            image_width=w, image_height=h, pixel_size_mm=pixel_size_mm,
            head_detected=e is not None,
            head_circumference_mm=round(pred.hc_mm, 2) if pred.hc_mm is not None else None,
            head_area_mm2=round(area_px * pixel_size_mm ** 2, 1) if (pixel_size_mm and area_px) else None,
            head_area_px=area_px,
            ellipse=ellipse,
            quality=self._quality(pred, pixel_size_mm),
            mask_png_base64=png_base64((pred.mask * 255).astype(np.uint8)),
            overlay_png_base64=png_base64(self._overlay(gray, pred)),
            inference_ms=round(elapsed_ms, 1),
            model_version=f"hc18-unet-epoch{self.segmenter.meta.get('epoch')}",
        )
        log.info("prediction", extra={"width": w, "height": h, "head_detected": response.head_detected,
                                      "hc_mm": response.head_circumference_mm, "ms": response.inference_ms})
        return response

    @staticmethod
    def _quality(pred: Prediction, pixel_size_mm: float | None) -> Quality:
        mask = pred.mask
        h, w = mask.shape
        area = int(mask.sum())
        touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        fit_ratio = None
        if pred.ellipse is not None:
            ellipse_area = math.pi * pred.ellipse.semi_major * pred.ellipse.semi_minor
            fit_ratio = round(area / ellipse_area, 3) if ellipse_area > 0 else None
        flags = []
        if pred.ellipse is None:
            flags.append("No fetal head region was detected.")
        else:
            if area / (h * w) < MIN_AREA_FRACTION:
                flags.append("Detected head region is very small; check that the image shows a fetal head.")
            if pred.mean_confidence < LOW_CONFIDENCE:
                flags.append("Low model confidence; the segmentation may be unreliable.")
            if pred.uncertain_fraction > HIGH_UNCERTAINTY:
                flags.append("Uncertain boundary; the head outline may be unclear.")
            if touches:
                flags.append("Head touches the image border; it may be cut off and HC underestimated.")
            if fit_ratio is not None and abs(fit_ratio - 1) > POOR_ELLIPSE_FIT:
                flags.append("Segmented shape is not elliptical; HC may be inaccurate.")
        if pixel_size_mm is None:
            flags.append("No pixel spacing given; HC is reported only when pixel size (mm) is provided.")
        return Quality(mean_confidence=round(pred.mean_confidence, 4),
                       uncertain_fraction=round(pred.uncertain_fraction, 4),
                       mask_area_fraction=round(area / (h * w), 4), touches_border=touches,
                       ellipse_fit_ratio=fit_ratio, flags=flags)

    @staticmethod
    def _overlay(gray: np.ndarray, pred: Prediction) -> np.ndarray:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        tint = rgb.copy()
        tint[pred.mask > 0] = (16, 185, 129)
        rgb = cv2.addWeighted(rgb, 0.65, tint, 0.35, 0)
        if pred.ellipse is not None:
            e = pred.ellipse
            thickness = max(2, round(max(gray.shape) / 300))
            cv2.ellipse(rgb, ((e.cx, e.cy), (2 * e.semi_major, 2 * e.semi_minor), e.angle_deg),
                        (250, 204, 21), thickness, cv2.LINE_AA)
        return rgb
