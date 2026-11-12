"""Segmentation metrics that do NOT reward empty-vs-empty predictions.

The previous ``dice_metric`` used ``(2*I + eps) / (P + T + eps)``, which evaluates to 1.0
when both prediction and target are empty. With the broken (empty) masks this reported a
perfect Dice for a model that predicts nothing. Here an empty target gives NaN
(undefined) and is excluded from the mean; the dataset additionally guarantees that no
target is empty.
"""
from __future__ import annotations

import torch


def _binarize(pred: torch.Tensor, threshold: float = 0.5, from_logits: bool = True) -> torch.Tensor:
    probs = torch.sigmoid(pred) if from_logits else pred
    return (probs > threshold).float()


def dice_per_sample(pred: torch.Tensor, target: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """Hard Dice per sample (N,). NaN where the target is empty."""
    p = _binarize(pred, from_logits=from_logits).flatten(1)
    t = (target > 0.5).float().flatten(1)
    inter = (p * t).sum(1)
    denom = p.sum(1) + t.sum(1)
    dice = 2.0 * inter / denom.clamp(min=1.0)
    return torch.where(t.sum(1) > 0, dice, torch.full_like(dice, float("nan")))


def iou_per_sample(pred: torch.Tensor, target: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """Hard IoU per sample (N,). NaN where the target is empty."""
    p = _binarize(pred, from_logits=from_logits).flatten(1)
    t = (target > 0.5).float().flatten(1)
    inter = (p * t).sum(1)
    union = ((p + t) > 0).float().sum(1)
    iou = inter / union.clamp(min=1.0)
    return torch.where(t.sum(1) > 0, iou, torch.full_like(iou, float("nan")))


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Differentiable Dice loss (training only; smoothing is fine for a loss)."""
    probs = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (probs * t).sum(1)
    dice = (2.0 * inter + eps) / (probs.sum(1) + t.sum(1) + eps)
    return 1.0 - dice.mean()
