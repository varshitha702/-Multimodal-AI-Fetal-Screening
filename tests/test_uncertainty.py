"""Tests for test-time-augmentation uncertainty (ml/uncertainty.py). No dataset or trained weights needed."""
import cv2
import numpy as np
import pytest
import torch

from ml.uncertainty import (
    TRANSFORMS,
    Transform,
    TTAResult,
    apply,
    reliability_reasons,
    tta_predict,
    uncertainty_overlay,
    undo,
)


def smooth_image(size=64):
    yy, xx = np.mgrid[:size, :size] / size
    return (0.5 + 0.4 * np.sin(3 * xx) * np.cos(2 * yy)).astype(np.float32)


@pytest.mark.parametrize("t", [t for t in TRANSFORMS if t.gamma == 1.0], ids=lambda t: t.name)
def test_undo_inverts_geometry(t):
    x = smooth_image()
    back, valid = undo(apply(x, t), t)
    assert valid.mean() > 0.7
    inner = valid.copy()
    inner[:3], inner[-3:], inner[:, :3], inner[:, -3:] = False, False, False, False   # interpolation at edges
    assert np.abs(back - x)[inner].max() < 0.05


def test_flip_is_exact_and_zoom_in_marks_lost_border():
    x = smooth_image()
    back, valid = undo(apply(x, Transform("f", hflip=True)), Transform("f", hflip=True))
    assert np.array_equal(back, x) and valid.all()
    _, valid = undo(apply(x, Transform("z", scale=1.1)), Transform("z", scale=1.1))
    assert not valid[0].any() and valid[32, 32]


def test_gamma_changes_intensity_only():
    x = smooth_image()
    y = apply(x, Transform("g", gamma=0.8))
    assert y.shape == x.shape and (y >= x - 1e-6).all()       # gamma < 1 brightens
    assert np.array_equal(undo(y, Transform("g", gamma=0.8))[0], y)


class ThresholdNet(torch.nn.Module):
    """Stand-in for the U-Net: logit = 20 * (intensity - 0.5), i.e. it segments bright pixels."""

    def forward(self, x):
        return 20 * (x - 0.5)


class FakeSegmenter:
    image_size = 128
    device = torch.device("cpu")
    model = ThresholdNet()


def disk_image(h=270, w=400, r=90):
    img = np.zeros((h, w), np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (r, int(r * 0.8)), 0, 0, 360, 255, -1)
    return img


def test_tta_on_clean_disk_is_consistent():
    r = tta_predict(FakeSegmenter(), disk_image(), pixel_size_mm=0.2)
    assert r.n_passes == len(TRANSFORMS) and len(r.hc_passes_mm) == len(TRANSFORMS)
    assert all(v is not None for v in r.hc_passes_mm)
    a, b = 90 * 0.2, 72 * 0.2
    expected = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    assert r.hc_mean_mm == pytest.approx(expected, rel=0.03)
    assert r.hc_cv_percent < 1.0 and r.disagreement_fraction < 0.1
    assert r.std_map.shape == (270, 400) and r.mean_probability.shape == (270, 400)
    assert r.std_map.max() <= 0.5 + 1e-6


def test_tta_without_pixel_size_gives_no_hc():
    r = tta_predict(FakeSegmenter(), disk_image(), pixel_size_mm=None)
    assert r.hc_mean_mm is None and r.hc_sd_mm is None and r.hc_cv_percent is None
    assert r.ellipse_mean is not None


def test_tta_on_blank_image():
    r = tta_predict(FakeSegmenter(), np.zeros((100, 120), np.uint8), pixel_size_mm=0.2)
    assert r.hc_mean_mm is None and r.disagreement_fraction == 0.0


def test_tta_rejects_colour():
    with pytest.raises(ValueError):
        tta_predict(FakeSegmenter(), np.zeros((64, 64, 3), np.uint8), 0.2)


def _result(cv, dis, passes=None):
    return TTAResult(hc_passes_mm=passes or [100.0, 101.0], hc_mean_mm=100.5, hc_sd_mm=0.7, hc_cv_percent=cv,
                     disagreement_fraction=dis, std_map=np.zeros((2, 2), np.float32),
                     mean_probability=np.zeros((2, 2), np.float32), ellipse_mean=None, n_passes=8)


def test_reliability_reasons_use_both_thresholds():
    cal = {"thresholds": {"hc_cv_percent": 1.0, "disagreement_fraction": 0.1}}
    assert reliability_reasons(_result(0.5, 0.05), cal) == []
    assert len(reliability_reasons(_result(1.5, 0.05), cal)) == 1
    assert len(reliability_reasons(_result(1.5, 0.2), cal)) == 2
    assert "could not be fitted" in reliability_reasons(_result(0.5, 0.05, [100.0, None]), cal)[0]
    assert reliability_reasons(_result(9.0, 0.9), None) == []      # no calibration file -> no flag


def test_uncertainty_overlay_shape():
    gray = np.full((50, 60), 100, np.uint8)
    std = np.zeros((50, 60), np.float32)
    std[10:20, 10:20] = 0.3
    out = uncertainty_overlay(gray, std)
    assert out.shape == (50, 60, 3) and out.dtype == np.uint8
    assert (out[0, 0] == 100).all() and not (out[15, 15] == 100).all()
