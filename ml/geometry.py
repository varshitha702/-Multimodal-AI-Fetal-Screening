"""Head-circumference geometry: mask -> ellipse -> perimeter in millimetres.

HC is the perimeter of the ellipse fitted to the fetal skull (the HC18 reference values were
measured this way; van den Heuvel et al., PLoS ONE 2018, doi:10.1371/journal.pone.0200412).

Perimeter: Ramanujan's second approximation (S. Ramanujan, "Modular equations and
approximations to pi", Quart. J. Math. 45, 1914, pp. 350-372):

    P ~= pi * (a + b) * (1 + 3h / (10 + sqrt(4 - 3h))),   h = (a - b)^2 / (a + b)^2

with semi-axes a, b. Its relative error is below 1e-9 for the axis ratios seen in fetal heads.

Resize handling: the network sees a ``size x size`` image that was resized with different x and
y scale factors, so a predicted mask is first resized back to the ORIGINAL image size; the
ellipse is fitted there, where HC18 pixel spacing (mm/px) is isotropic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def ramanujan_perimeter(a: float, b: float) -> float:
    """Ellipse perimeter from semi-axes ``a`` and ``b`` (Ramanujan II)."""
    if a < 0 or b < 0:
        raise ValueError("Semi-axes must be non-negative")
    if a + b == 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


@dataclass(frozen=True)
class EllipseFit:
    cx: float
    cy: float
    semi_major: float  # px
    semi_minor: float  # px
    angle_deg: float   # rotation of the MAJOR axis in degrees [0, 180), OpenCV image convention
    perimeter_px: float
    area_px: float      # pixels in the mask region used for fitting

    def hc_mm(self, pixel_size_mm: float) -> float:
        return self.perimeter_px * pixel_size_mm


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component (removes small false positives)."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def fit_ellipse_to_mask(mask: np.ndarray) -> EllipseFit | None:
    """Fit an ellipse to the boundary of the largest component. ``None`` if the mask is too small."""
    comp = largest_component(mask)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
    if d1 < d2:  # make the angle refer to the major axis: (d1, d2, t) == (d2, d1, t + 90)
        d1, d2, angle = d2, d1, angle + 90.0
    a, b = d1 / 2, d2 / 2
    return EllipseFit(float(cx), float(cy), float(a), float(b), float(angle % 180.0),
                      ramanujan_perimeter(a, b), float(comp.sum()))


def resize_to_original(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a (network-resolution) 0/1 mask back to the original image size."""
    up = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return (up >= 0.5).astype(np.uint8)


def hc_from_mask(mask: np.ndarray, original_wh: tuple[int, int], pixel_size_mm: float):
    """Network mask -> (EllipseFit at original resolution, HC in mm) or (None, None)."""
    full = resize_to_original(mask, *original_wh)
    fit = fit_ellipse_to_mask(full)
    if fit is None:
        return None, None
    return fit, fit.hc_mm(pixel_size_mm)
