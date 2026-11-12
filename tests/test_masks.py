"""Tests for the HC18 mask-filling fix (ml/masks.py)."""
import cv2
import numpy as np
import pytest

from ml.masks import annotation_to_mask, fill_annotation, fit_outline_ellipse, resize_mask
from tests.conftest import HC18_TRAIN


def synthetic_outline(h=540, w=800, center=(400, 270), axes=(150, 110), angle=20):
    """1-px ellipse outline like an HC18 annotation (axes are semi-axes here)."""
    img = np.zeros((h, w), np.uint8)
    cv2.ellipse(img, center, axes, angle, 0, 360, 255, thickness=1)
    return img


def test_fill_produces_solid_region_with_expected_area():
    filled, ell = fill_annotation(synthetic_outline())
    expected = np.pi * 150 * 110
    assert abs(filled.sum() - expected) / expected < 0.02
    assert abs(ell.cx - 400) < 1 and abs(ell.cy - 270) < 1
    assert sorted([ell.axis_a, ell.axis_b]) == pytest.approx([220, 300], abs=2)


def test_fixed_pipeline_mask_is_non_empty_after_downsizing():
    filled, _ = fill_annotation(synthetic_outline())
    small = resize_mask(filled, 128)
    assert small.shape == (128, 128)
    assert small.mean() > 0.05


def test_old_resize_then_fill_pipeline_loses_the_mask():
    """Documents the original bug: NEAREST-resizing a thin outline breaks it."""
    outline = cv2.resize(synthetic_outline(), (128, 128), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(outline)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 0:
            cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
    assert filled.mean() / 255 < 0.01  # broken outline -> (nearly) empty mask


def test_fit_rejects_empty_annotation():
    with pytest.raises(ValueError):
        fit_outline_ellipse(np.zeros((50, 50), np.uint8))


@pytest.mark.skipif(not HC18_TRAIN.is_dir(), reason="HC18 dataset not downloaded")
def test_all_real_hc18_masks_are_non_empty():
    paths = sorted(HC18_TRAIN.glob("*_Annotation.png"))
    assert len(paths) == 999
    for p in paths:
        m = annotation_to_mask(p, 256)
        assert 0.01 < m.mean() < 0.9, p.name
