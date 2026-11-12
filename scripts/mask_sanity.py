"""Before/after evidence for the HC18 mask bug, plus a visual sanity grid.

Usage:  python scripts/mask_sanity.py
Writes: outputs/mask_pipeline_check.json, docs/screenshots/mask_bug_before_after.png,
        docs/screenshots/mask_sanity_grid.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.masks import annotation_to_mask, fill_annotation, load_annotation  # noqa: E402

IMAGE_DIR = ROOT / "data" / "raw" / "HC18" / "training_set"
OLD_SIZE, NEW_SIZE = 128, 256


def legacy_mask(path: Path, size: int = OLD_SIZE) -> np.ndarray:
    """Exact replica of the old train_hc18.py logic (resize outline, THEN fill)."""
    with Image.open(path) as f:
        m = np.asarray(f.convert("L").resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8)
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 0:
            cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
    return (filled > 0).astype(np.uint8)


def main() -> None:
    ann_paths = sorted(IMAGE_DIR.glob("*_Annotation.png"))
    old_fg, new_fg, old_empty, new_empty = [], [], [], []
    for p in ann_paths:
        o = legacy_mask(p)
        n = annotation_to_mask(p, NEW_SIZE)
        old_fg.append(o.mean())
        new_fg.append(n.mean())
        old_empty.append(int(o.sum() == 0))
        new_empty.append(int(n.sum() == 0))
    old_fg, new_fg = np.array(old_fg), np.array(new_fg)
    report = {
        "annotations_checked": len(ann_paths),
        "before_fix": {
            "pipeline": "resize outline to 128 (NEAREST) -> findContours -> FILLED",
            "empty_masks": int(sum(old_empty)),
            "nearly_empty_masks_lt_1pct": int((old_fg < 0.01).sum()),
            "mean_foreground_fraction": float(old_fg.mean()),
            "median_foreground_fraction": float(np.median(old_fg)),
        },
        "after_fix": {
            "pipeline": "fitEllipse on outline at original resolution -> filled ellipse -> resize to 256",
            "empty_masks": int(sum(new_empty)),
            "nearly_empty_masks_lt_1pct": int((new_fg < 0.01).sum()),
            "mean_foreground_fraction": float(new_fg.mean()),
            "min_foreground_fraction": float(new_fg.min()),
            "max_foreground_fraction": float(new_fg.max()),
        },
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "mask_pipeline_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    shots = ROOT / "docs" / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    # Before/after figure: image | raw outline | old mask | new mask
    sample = [ann_paths[i] for i in (0, 150, 400, 700)]
    fig, axes = plt.subplots(len(sample), 4, figsize=(12, 3 * len(sample)))
    for r, ap in enumerate(sample):
        img = np.asarray(Image.open(str(ap).replace("_Annotation", "")).convert("L"))
        outline = load_annotation(ap)
        old = legacy_mask(ap)
        new = annotation_to_mask(ap, NEW_SIZE)
        panels = [
            (img, f"{ap.name.replace('_Annotation', '')}"),
            (outline, "annotation (outline)"),
            (old, f"BEFORE: 128px, fg={old.mean():.3f}"),
            (new, f"AFTER: 256px, fg={new.mean():.3f}"),
        ]
        for c, (arr, title) in enumerate(panels):
            axes[r, c].imshow(arr, cmap="gray")
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("HC18 mask bug: resize-then-fill (before) vs fill-then-resize (after)")
    fig.tight_layout()
    fig.savefig(shots / "mask_bug_before_after.png", dpi=110)
    plt.close(fig)

    # Sanity grid: image with filled mask overlay
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ann_paths), 12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for ax, i in zip(axes.flat, pick, strict=True):
        ap = ann_paths[i]
        img = np.asarray(Image.open(str(ap).replace("_Annotation", "")).convert("L"))
        filled, _ = fill_annotation(load_annotation(ap))
        rgb = np.stack([img] * 3, -1).astype(np.float32) / 255
        rgb[filled > 0] = 0.6 * rgb[filled > 0] + 0.4 * np.array([0.1, 0.9, 0.3])
        ax.imshow(rgb)
        ax.set_title(ap.name.replace("_Annotation", ""), fontsize=9)
        ax.axis("off")
    fig.suptitle("Fixed masks overlaid on HC18 images (green = filled head ellipse)")
    fig.tight_layout()
    fig.savefig(shots / "mask_sanity_grid.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
