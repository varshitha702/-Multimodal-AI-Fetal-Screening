"""Single-image inference shared by evaluation, the FastAPI backend and the dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.dataset import preprocess
from ml.geometry import EllipseFit, fit_ellipse_to_mask, largest_component, resize_to_original
from ml.model import UNet


@dataclass
class Prediction:
    probability: np.ndarray      # float32 HxW at ORIGINAL resolution, values in [0, 1]
    mask: np.ndarray             # uint8 HxW at original resolution, largest component only
    ellipse: EllipseFit | None   # fitted at original resolution
    hc_mm: float | None          # None when no pixel size was given or no ellipse could be fitted
    mean_confidence: float       # mean probability inside the predicted mask
    uncertain_fraction: float    # share of pixels with 0.3 < p < 0.7 inside the ellipse bounding area


class Segmenter:
    """Loads the U-Net checkpoint once and runs inference on grayscale uint8 images."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.image_size = int(ckpt["image_size"])
        self.meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        self.model = UNet().to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])

    @torch.no_grad()
    def probability_map(self, gray: np.ndarray) -> np.ndarray:
        """Foreground probability at network resolution (size x size)."""
        x = torch.from_numpy(preprocess(gray, self.image_size))[None, None].to(self.device)
        return torch.sigmoid(self.model(x))[0, 0].float().cpu().numpy()

    def predict(self, gray: np.ndarray, pixel_size_mm: float | None = None) -> Prediction:
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Expected a 2-D uint8 grayscale image")
        h, w = gray.shape
        prob_small = self.probability_map(gray)
        prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = largest_component(resize_to_original((prob_small >= 0.5).astype(np.uint8), w, h))
        ellipse = fit_ellipse_to_mask(mask)
        hc = ellipse.hc_mm(pixel_size_mm) if (ellipse is not None and pixel_size_mm) else None
        conf = float(prob[mask > 0].mean()) if mask.any() else 0.0
        if mask.any():
            ys, xs = np.nonzero(mask)
            box = prob[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            uncertain = float(((box > 0.3) & (box < 0.7)).mean())
        else:
            uncertain = 0.0
        return Prediction(prob.astype(np.float32), mask, ellipse, hc, conf, uncertain)
