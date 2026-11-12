"""Tests for HC geometry (ml/geometry.py)."""
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from ml.geometry import fit_ellipse_to_mask, hc_from_mask, largest_component, ramanujan_perimeter
from ml.masks import fit_outline_ellipse, load_annotation
from tests.conftest import HC18_TRAIN

CSV = HC18_TRAIN.parent / "training_set_pixel_size_and_HC.csv"


def test_circle_perimeter_is_exact():
    assert ramanujan_perimeter(10, 10) == pytest.approx(2 * math.pi * 10, rel=1e-12)


def test_known_ellipse_perimeter():
    # a=10, b=5: exact perimeter 48.44224110... (complete elliptic integral of the 2nd kind)
    assert ramanujan_perimeter(10, 5) == pytest.approx(48.4422411, rel=1e-8)


def test_degenerate_ellipse_is_a_line():
    assert ramanujan_perimeter(10, 0) == pytest.approx(40.0, rel=1e-3)


def test_negative_axis_rejected():
    with pytest.raises(ValueError):
        ramanujan_perimeter(-1, 2)


def test_fit_recovers_drawn_ellipse():
    m = np.zeros((540, 800), np.uint8)
    cv2.ellipse(m, (400, 270), (200, 150), 30, 0, 360, 1, thickness=cv2.FILLED)
    fit = fit_ellipse_to_mask(m)
    assert fit.semi_major == pytest.approx(200, abs=1.5)
    assert fit.semi_minor == pytest.approx(150, abs=1.5)
    assert fit.angle_deg == pytest.approx(30, abs=1.0)


def test_angle_refers_to_major_axis_when_ellipse_is_tall():
    m = np.zeros((540, 800), np.uint8)
    cv2.ellipse(m, (400, 270), (100, 200), 0, 0, 360, 1, thickness=cv2.FILLED)  # major axis vertical
    fit = fit_ellipse_to_mask(m)
    assert fit.semi_major == pytest.approx(200, abs=1.5)
    assert fit.angle_deg == pytest.approx(90, abs=1.0)


def test_empty_mask_returns_none():
    assert fit_ellipse_to_mask(np.zeros((64, 64), np.uint8)) is None


def test_largest_component_drops_specks():
    m = np.zeros((100, 100), np.uint8)
    m[10:60, 10:60] = 1
    m[90:92, 90:92] = 1
    assert largest_component(m).sum() == 50 * 50


def test_hc_from_mask_accounts_for_anisotropic_resize():
    """Ellipse drawn at 800x540, downsized to 256x256, measured back at 800x540."""
    m = np.zeros((540, 800), np.uint8)
    cv2.ellipse(m, (400, 270), (200, 150), 0, 0, 360, 1, thickness=cv2.FILLED)
    small = (cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
    _, hc = hc_from_mask(small, (800, 540), pixel_size_mm=0.1)
    assert hc == pytest.approx(ramanujan_perimeter(200, 150) * 0.1, rel=0.01)


@pytest.mark.skipif(not CSV.is_file(), reason="HC18 dataset not downloaded")
def test_ground_truth_annotations_reproduce_csv_hc():
    """Fitting the reference outlines must give the published HC within 0.1 mm."""
    df = pd.read_csv(CSV).iloc[::10]  # 100 images
    for _, r in df.iterrows():
        el = fit_outline_ellipse(load_annotation(HC18_TRAIN / r.filename.replace(".png", "_Annotation.png")))
        hc = ramanujan_perimeter(el.axis_a / 2, el.axis_b / 2) * r["pixel size(mm)"]
        assert hc == pytest.approx(r["head circumference (mm)"], abs=0.1), r.filename
