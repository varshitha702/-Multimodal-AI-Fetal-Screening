"""Measurement uncertainty by test-time augmentation (TTA), Feature B.

Each image is passed through the U-Net N times under small, label-preserving transforms (horizontal flip, zoom
0.9 / 1.1, gamma 0.8 / 1.25). Every transform is UNDONE on the probability map before combining, so all passes
are compared in the original image geometry. From the N passes we report:

* HC per pass -> HC mean +- SD (mm) and coefficient of variation (CV, %);
* a per-pixel uncertainty map = SD of the N probability maps;
* the disagreement fraction = pixels where the N binary masks do not all agree / pixels in their union.

The "low reliability, re-scan" flag uses thresholds chosen on the VALIDATION split
(ml/evaluate_uncertainty.py -> outputs/uncertainty_calibration.json).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.dataset import preprocess
from ml.geometry import EllipseFit, fit_ellipse_to_mask, largest_component, resize_to_original

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "outputs" / "uncertainty_calibration.json"


@dataclass(frozen=True)
class Transform:
    name: str
    hflip: bool = False
    scale: float = 1.0
    gamma: float = 1.0


TRANSFORMS: tuple[Transform, ...] = (
    Transform("identity"),
    Transform("hflip", hflip=True),
    Transform("zoom_out_0.9", scale=0.9),
    Transform("zoom_in_1.1", scale=1.1),
    Transform("gamma_0.8", gamma=0.8),
    Transform("gamma_1.25", gamma=1.25),
    Transform("hflip_zoom_0.9", hflip=True, scale=0.9),
    Transform("hflip_zoom_1.1", hflip=True, scale=1.1),
)


def _zoom(img: np.ndarray, scale: float) -> np.ndarray:
    """Zoom about the image centre (scale > 1 enlarges); pixels coming from outside are 0."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), 0, scale)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def apply(x: np.ndarray, t: Transform) -> np.ndarray:
    """Forward transform of a float [0, 1] network-size image: gamma, then zoom, then flip."""
    if t.gamma != 1.0:
        x = np.power(x, t.gamma)
    if t.scale != 1.0:
        x = _zoom(x, t.scale)
    if t.hflip:
        x = x[:, ::-1]
    return np.ascontiguousarray(x, dtype=np.float32)


def undo(p: np.ndarray, t: Transform) -> tuple[np.ndarray, np.ndarray]:
    """Inverse geometric transform of a probability map. Returns (map, valid) where ``valid`` marks pixels that
    were inside the transformed view (a zoom-in loses the border, which then carries no information)."""
    if t.hflip:
        p = p[:, ::-1]
    p = np.ascontiguousarray(p, dtype=np.float32)
    valid = np.ones_like(p)
    if t.scale != 1.0:
        p = _zoom(p, 1.0 / t.scale)
        valid = _zoom(valid, 1.0 / t.scale)
    return p, valid > 0.99


@dataclass
class TTAResult:
    hc_passes_mm: list[float | None]       # one per transform (None: no ellipse or no pixel size)
    hc_mean_mm: float | None
    hc_sd_mm: float | None                 # sample SD over passes with an ellipse
    hc_cv_percent: float | None
    disagreement_fraction: float           # pixels where pass masks disagree / union of pass masks
    std_map: np.ndarray                    # float32 HxW at original resolution: SD of probabilities
    mean_probability: np.ndarray           # float32 HxW at original resolution
    ellipse_mean: EllipseFit | None        # ellipse fitted to the mean-probability mask
    n_passes: int


@torch.no_grad()
def tta_predict(segmenter, gray: np.ndarray, pixel_size_mm: float | None,
                transforms: tuple[Transform, ...] = TRANSFORMS) -> TTAResult:
    """Run all TTA passes in one batch; undo each transform; combine."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected a 2-D uint8 grayscale image")
    h, w = gray.shape
    size = segmenter.image_size
    x = preprocess(gray, size)
    batch = torch.from_numpy(np.stack([apply(x, t) for t in transforms]))[:, None].to(segmenter.device)
    probs = torch.sigmoid(segmenter.model(batch))[:, 0].float().cpu().numpy()
    identity = probs[0]
    maps = []
    for p, t in zip(probs, transforms, strict=True):
        p, valid = undo(p, t)
        maps.append(np.where(valid, p, identity))   # outside the view: fall back to the identity pass
    maps = np.stack(maps)
    masks = maps >= 0.5
    union, inter = masks.any(0), masks.all(0)
    disagreement = float((union & ~inter).sum() / union.sum()) if union.any() else 0.0

    hcs: list[float | None] = []
    for m in masks:
        full = largest_component(resize_to_original(m.astype(np.uint8), w, h))
        e = fit_ellipse_to_mask(full)
        hcs.append(e.hc_mm(pixel_size_mm) if (e is not None and pixel_size_mm) else None)
    valid_hc = np.array([v for v in hcs if v is not None], dtype=float)
    mean = float(valid_hc.mean()) if len(valid_hc) else None
    sd = float(valid_hc.std(ddof=1)) if len(valid_hc) > 1 else None
    cv = 100 * sd / mean if (sd is not None and mean) else None

    mean_small = maps.mean(0)
    mean_mask = largest_component(resize_to_original((mean_small >= 0.5).astype(np.uint8), w, h))
    return TTAResult(
        hc_passes_mm=hcs, hc_mean_mm=mean, hc_sd_mm=sd, hc_cv_percent=cv, disagreement_fraction=disagreement,
        std_map=cv2.resize(maps.std(0), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32),
        mean_probability=cv2.resize(mean_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32),
        ellipse_mean=fit_ellipse_to_mask(mean_mask), n_passes=len(transforms),
    )


def load_calibration(path: Path = CALIBRATION_PATH) -> dict | None:
    """Validation-chosen thresholds, or None if ml/evaluate_uncertainty.py has not been run."""
    return json.loads(path.read_text()) if path.is_file() else None


def reliability_reasons(r: TTAResult, cal: dict | None) -> list[str]:
    """Plain-language reasons for a 'low reliability' flag (empty list = reliable or not assessable)."""
    if cal is None:
        return []
    reasons = []
    t_cv, t_dis = cal["thresholds"]["hc_cv_percent"], cal["thresholds"]["disagreement_fraction"]
    if r.hc_cv_percent is not None and r.hc_cv_percent > t_cv:
        reasons.append(f"HC varies across {r.n_passes} augmented passes (CV {r.hc_cv_percent:.2f} % > {t_cv:.2f} %).")
    if r.disagreement_fraction > t_dis:
        reasons.append(f"Augmented passes disagree on {r.disagreement_fraction:.1%} of the head region "
                       f"(> {t_dis:.1%}).")
    if r.hc_passes_mm and any(v is None for v in r.hc_passes_mm) and r.hc_mean_mm is not None:
        reasons.append("The head outline could not be fitted in some augmented passes.")
    return reasons


def uncertainty_overlay(gray: np.ndarray, std_map: np.ndarray, vmax: float = 0.25) -> np.ndarray:
    """RGB image with the per-pixel SD as a heat map (transparent where SD is ~0)."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    norm = np.clip(std_map / vmax, 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    alpha = (np.clip(norm * 1.5, 0, 0.85))[..., None]
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
