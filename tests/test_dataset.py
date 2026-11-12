"""Tests for preprocessing and augmentation (ml/dataset.py)."""
import cv2
import numpy as np

from ml.dataset import augment, preprocess


def _pair():
    img = np.zeros((256, 256), np.float32)
    mask = np.zeros((256, 256), np.float32)
    cv2.ellipse(img, (128, 128), (60, 40), 0, 0, 360, 0.8, thickness=cv2.FILLED)
    cv2.ellipse(mask, (128, 128), (60, 40), 0, 0, 360, 1.0, thickness=cv2.FILLED)
    return img, mask


def test_preprocess_shape_and_range():
    out = preprocess(np.full((540, 800), 255, np.uint8), 256)
    assert out.shape == (256, 256) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_keeps_mask_binary_and_aligned():
    img, mask = _pair()
    for seed in range(20):
        a_img, a_mask = augment(img, mask, np.random.default_rng(seed))
        assert a_img.shape == a_mask.shape == (256, 256)
        assert set(np.unique(a_mask)) <= {0.0, 1.0}
        assert 0.0 <= a_img.min() and a_img.max() <= 1.0
        # the bright ellipse and the mask must still coincide after geometric transforms
        inside = a_img[a_mask > 0].mean()
        outside = a_img[a_mask == 0].mean()
        assert inside > outside + 0.4


def test_augment_is_reproducible_for_same_seed():
    img, mask = _pair()
    a = augment(img, mask, np.random.default_rng(7))
    b = augment(img, mask, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
