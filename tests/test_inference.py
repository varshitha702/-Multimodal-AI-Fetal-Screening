"""Tests for evaluation helpers and the inference wrapper (ml/evaluate.py, ml/inference.py)."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ml.evaluate import dice_iou
from ml.inference import Segmenter
from ml.model import UNet
from ml.train import CHECKPOINT_PATH
from tests.conftest import HC18_TRAIN, ROOT


def test_dice_iou_full_resolution():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[:5] = 1
    b[2:7] = 1
    d, i = dice_iou(a, b)
    assert d == pytest.approx(0.6) and i == pytest.approx(3 / 7)


@pytest.fixture(scope="module")
def random_segmenter(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("m") / "rand.pth"
    torch.save({"model_state_dict": UNet().state_dict(), "image_size": 128, "epoch": 0}, path)
    return Segmenter(path, device="cpu")


def test_prediction_is_at_original_resolution(random_segmenter):
    gray = np.random.default_rng(0).integers(0, 255, (300, 451), dtype=np.uint8)
    pred = random_segmenter.predict(gray, pixel_size_mm=0.1)
    assert pred.probability.shape == pred.mask.shape == (300, 451)
    assert 0.0 <= pred.probability.min() and pred.probability.max() <= 1.0


def test_predict_rejects_non_grayscale(random_segmenter):
    with pytest.raises(ValueError):
        random_segmenter.predict(np.zeros((64, 64, 3), np.uint8))


def test_no_hc_without_pixel_size(random_segmenter):
    pred = random_segmenter.predict(np.full((128, 128), 128, np.uint8))
    assert pred.hc_mm is None


@pytest.mark.skipif(not (CHECKPOINT_PATH.is_file() and HC18_TRAIN.is_dir()),
                    reason="trained checkpoint or dataset not available")
def test_trained_model_regression_on_test_split_images():
    """The trained model must stay within 10 mm on a few held-out test images (guards against regressions)."""
    seg = Segmenter(CHECKPOINT_PATH)
    split = pd.read_csv(ROOT / "data" / "splits" / "hc18_split.csv")
    rows = split[split.split == "test"].head(5)
    for _, r in rows.iterrows():
        gray = np.asarray(Image.open(HC18_TRAIN / r.filename).convert("L"), dtype=np.uint8)
        pred = seg.predict(gray, pixel_size_mm=float(r["pixel size(mm)"]))
        assert pred.hc_mm == pytest.approx(r["head circumference (mm)"], abs=10.0), r.filename
