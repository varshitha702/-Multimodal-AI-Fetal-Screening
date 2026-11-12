"""Standard-plane quality gate: a two-head classifier trained on FETAL_PLANES_DB.

Head 1 (plane):    Fetal brain / Fetal abdomen / Fetal femur / Fetal thorax / Maternal cervix / Other
Head 2 (sub-plane, meaningful for brain images only):
                   Trans-thalamic / Trans-cerebellum / Trans-ventricular / Other (brain, non-standard)

Both heads share one ImageNet-pretrained EfficientNet-B0 backbone. Each head has its own temperature
(fitted on the validation set) so the reported probabilities are calibrated.

The gate score is P(trans-thalamic) = P(plane = brain) * P(sub-plane = TT | brain), which is the plane on
which head circumference is defined. The image passes when that score is >= the threshold chosen on the
validation set (see ml/train_planes.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PLANES = ["Fetal brain", "Fetal abdomen", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"]
SUBPLANES = ["Trans-thalamic", "Trans-cerebellum", "Trans-ventricular", "Other"]
BRAIN = PLANES.index("Fetal brain")
TT = SUBPLANES.index("Trans-thalamic")
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PlaneNet(nn.Module):
    """EfficientNet-B0 backbone with two linear heads (plane, brain sub-plane)."""

    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.plane_head = nn.Linear(1280, len(PLANES))
        self.subplane_head = nn.Linear(1280, len(SUBPLANES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.dropout(torch.flatten(self.pool(self.features(x)), 1))
        return self.plane_head(z), self.subplane_head(z)


def letterbox(gray: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """uint8 HxW -> uint8 size x size: keep aspect ratio, pad with black (ultrasound background)."""
    h, w = gray.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = small
    return out


def normalize(batch: torch.Tensor) -> torch.Tensor:
    """(N, H, W) grayscale (uint8, or float in [0, 1]) -> (N, 3, H, W) float, ImageNet-normalised.

    Shared by training and inference.
    """
    x = batch.float().div(255.0) if batch.dtype == torch.uint8 else batch.float()
    x = x.unsqueeze(1).expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# --- calibration ----------------------------------------------------------------------------------

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): the T > 0 that minimises NLL of softmax(logits / T)."""
    logits, labels = logits.detach().float().cpu(), labels.detach().long().cpu()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def tt_score(plane_probs: np.ndarray, sub_probs: np.ndarray) -> np.ndarray:
    """P(trans-thalamic) = P(brain) * P(TT | brain)."""
    return plane_probs[..., BRAIN] * sub_probs[..., TT]


# --- inference -----------------------------------------------------------------------------------

@dataclass
class PlaneResult:
    plane: str
    plane_confidence: float
    plane_probabilities: dict[str, float]
    brain_subplane: str | None              # only when the predicted plane is "Fetal brain"
    brain_subplane_confidence: float | None
    subplane_probabilities: dict[str, float] = field(default_factory=dict)
    tt_probability: float = 0.0
    threshold: float = 0.5
    passed: bool = False


class PlaneClassifier:
    """Loads the plane checkpoint once; ``check(gray)`` returns calibrated probabilities and the gate decision."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        self.input_size = int(ckpt.get("input_size", INPUT_SIZE))
        self.t_plane = float(ckpt.get("temperature_plane", 1.0))
        self.t_sub = float(ckpt.get("temperature_subplane", 1.0))
        self.threshold = float(ckpt.get("tt_threshold", 0.5))
        self.model = PlaneNet(pretrained=False).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])

    @torch.no_grad()
    def probabilities(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(letterbox(gray, self.input_size))[None].to(self.device)
        lp, ls = self.model(normalize(x))
        return (F.softmax(lp / self.t_plane, 1)[0].cpu().numpy(),
                F.softmax(ls / self.t_sub, 1)[0].cpu().numpy())

    def check(self, gray: np.ndarray) -> PlaneResult:
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Expected a 2-D uint8 grayscale image")
        pp, sp = self.probabilities(gray)
        k = int(pp.argmax())
        is_brain = k == BRAIN
        j = int(sp.argmax())
        score = float(tt_score(pp, sp))
        return PlaneResult(
            plane=PLANES[k], plane_confidence=float(pp[k]),
            plane_probabilities={n: float(p) for n, p in zip(PLANES, pp, strict=True)},
            brain_subplane=SUBPLANES[j] if is_brain else None,
            brain_subplane_confidence=float(sp[j]) if is_brain else None,
            subplane_probabilities={n: float(p) for n, p in zip(SUBPLANES, sp, strict=True)},
            tt_probability=score, threshold=self.threshold, passed=score >= self.threshold,
        )
