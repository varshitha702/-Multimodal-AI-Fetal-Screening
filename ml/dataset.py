"""HC18 PyTorch dataset: preloads resized images + fixed masks, optional augmentation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.masks import annotation_to_mask

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "raw" / "HC18" / "training_set"


def preprocess(gray: np.ndarray, size: int) -> np.ndarray:
    """uint8 HxW grayscale -> float32 size x size in [0, 1]. Shared by training and inference."""
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def load_image(path: str | Path, size: int) -> np.ndarray:
    """Grayscale image resized to ``size x size``, float32 in [0, 1]."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return preprocess(arr, size)


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal flip, small rotation/scale, brightness/contrast jitter."""
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.7:
        h, w = image.shape
        angle = rng.uniform(-15, 15)
        scale = rng.uniform(0.9, 1.1)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        image = cv2.warpAffine(np.ascontiguousarray(image), m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), m, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.7:
        contrast = rng.uniform(0.8, 1.2)
        brightness = rng.uniform(-0.1, 0.1)
        image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)


class HC18Dataset(Dataset):
    """Rows of the split CSV -> (image[1,S,S], mask[1,S,S]) tensors."""

    def __init__(self, rows: pd.DataFrame, size: int = 256, train: bool = False,
                 image_dir: Path = IMAGE_DIR, seed: int = 0):
        self.filenames = rows["filename"].tolist()
        self.size = size
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.images = np.stack([load_image(image_dir / f, size) for f in self.filenames])
        self.masks = np.stack([
            annotation_to_mask(image_dir / f.replace(".png", "_Annotation.png"), size).astype(np.float32)
            for f in self.filenames
        ])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        image, mask = self.images[i], self.masks[i]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))  # reproducible per epoch and sample
            image, mask = augment(image, mask, rng)
        return torch.from_numpy(image)[None], torch.from_numpy(mask)[None]
