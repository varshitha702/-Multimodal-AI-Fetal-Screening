"""Tests for Dice/IoU metrics (ml/metrics.py)."""
import math

import torch

from ml.metrics import dice_per_sample, iou_per_sample, soft_dice_loss

BIG = 20.0  # logit -> prob ~1


def _mask(fill_rows):
    m = torch.zeros(1, 1, 8, 8)
    m[..., :fill_rows, :] = 1
    return m


def test_empty_target_and_empty_prediction_is_not_perfect():
    pred = torch.full((1, 1, 8, 8), -BIG)
    target = torch.zeros(1, 1, 8, 8)
    assert math.isnan(dice_per_sample(pred, target).item())
    assert math.isnan(iou_per_sample(pred, target).item())


def test_perfect_prediction_scores_one():
    t = _mask(4)
    pred = (t * 2 - 1) * BIG
    assert dice_per_sample(pred, t).item() == 1.0
    assert iou_per_sample(pred, t).item() == 1.0


def test_empty_prediction_on_non_empty_target_scores_zero():
    t = _mask(4)
    pred = torch.full_like(t, -BIG)
    assert dice_per_sample(pred, t).item() == 0.0


def test_half_overlap_values():
    t = _mask(4)  # rows 0-3
    p = torch.zeros_like(t)
    p[..., 2:6, :] = 1  # rows 2-5 -> overlap 2 rows
    pred = (p * 2 - 1) * BIG
    assert abs(dice_per_sample(pred, t).item() - 0.5) < 1e-6
    assert abs(iou_per_sample(pred, t).item() - 1 / 3) < 1e-6


def test_soft_dice_loss_low_for_good_prediction():
    t = _mask(4)
    assert soft_dice_loss((t * 2 - 1) * BIG, t).item() < 0.01
