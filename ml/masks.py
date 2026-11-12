"""HC18 annotation -> filled binary mask.

HC18 annotations are a thin (about 1-2 px) ellipse *outline* drawn on an image the
same size as the ultrasound frame (van den Heuvel et al., PLoS ONE 2018).

Bug that this module fixes: the original trainer resized the outline to 128x128 with
NEAREST interpolation *before* filling it. Nearest-neighbour downsampling drops most
pixels of a 1-px line, so the outline breaks into fragments, ``findContours`` finds no
closed region, and the "filled" mask is empty. The model then learned to predict
nothing and an epsilon-smoothed Dice scored empty-vs-empty as 1.0.

Fix: fit an ellipse to the outline pixels at the ORIGINAL resolution, draw it filled,
and only then resize the filled mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Ellipse:
    """Ellipse in pixel coordinates (OpenCV ``fitEllipse`` convention)."""

    cx: float
    cy: float
    axis_a: float  # full length of first axis (diameter, px)
    axis_b: float  # full length of second axis (diameter, px)
    angle_deg: float

    def as_cv2(self):
        return (self.cx, self.cy), (self.axis_a, self.axis_b), self.angle_deg


def load_annotation(path: str | Path) -> np.ndarray:
    """Load an HC18 annotation PNG as a uint8 array (0 / 255)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def fit_outline_ellipse(outline: np.ndarray, min_points: int = 5) -> Ellipse:
    """Fit an ellipse to all non-zero pixels of an outline image."""
    ys, xs = np.nonzero(outline > 0)
    if xs.size < min_points:
        raise ValueError(f"Annotation has only {xs.size} outline pixels; cannot fit an ellipse")
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    (cx, cy), (a, b), angle = cv2.fitEllipse(points)
    return Ellipse(float(cx), float(cy), float(a), float(b), float(angle))


def fill_annotation(outline: np.ndarray) -> tuple[np.ndarray, Ellipse]:
    """Return a filled 0/1 uint8 mask at the outline's own resolution, plus the ellipse."""
    ellipse = fit_outline_ellipse(outline)
    filled = np.zeros(outline.shape[:2], dtype=np.uint8)
    cv2.ellipse(filled, ellipse.as_cv2(), color=1, thickness=cv2.FILLED)
    return filled, ellipse


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize a filled 0/1 mask to ``size x size`` (area-averaged, then thresholded at 0.5)."""
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (resized >= 0.5).astype(np.uint8)


def annotation_to_mask(path: str | Path, size: int) -> np.ndarray:
    """Full fixed pipeline: load outline -> fill at original resolution -> resize."""
    filled, _ = fill_annotation(load_annotation(path))
    mask = resize_mask(filled, size)
    if mask.sum() == 0:
        raise ValueError(f"Filled mask is empty after resize: {path}")
    return mask
