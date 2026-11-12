"""Evaluate the trained U-Net on the held-out TEST split (never used for training or model selection).

    python -m ml.evaluate

Metrics, all at the ORIGINAL image resolution:
  * Dice and IoU of the predicted mask vs the filled reference ellipse
  * HC error (mm) = predicted HC - reference HC from the HC18 CSV
Writes outputs/test_metrics.json, outputs/test_predictions.csv and figures in docs/screenshots/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from ml.dataset import IMAGE_DIR  # noqa: E402
from ml.inference import Segmenter  # noqa: E402
from ml.masks import fill_annotation, load_annotation  # noqa: E402
from ml.split import HC_COL, PX_COL, load_split  # noqa: E402
from ml.train import CHECKPOINT_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SHOTS = ROOT / "docs" / "screenshots"
OFFICIAL_TEST_DIR = ROOT / "data" / "raw" / "HC18" / "test_set"


def dice_iou(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    p, r = pred > 0, ref > 0
    inter = np.logical_and(p, r).sum()
    return 2 * inter / (p.sum() + r.sum()), inter / np.logical_or(p, r).sum()


def overlay(gray: np.ndarray, mask: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    import cv2

    rgb = np.stack([gray] * 3, -1).astype(np.uint8).copy()
    tint = rgb.copy()
    tint[mask > 0] = (0, 200, 120)
    rgb = cv2.addWeighted(rgb, 0.65, tint, 0.35, 0)
    if ref is not None:
        contours, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
    return rgb


def describe(errors: np.ndarray) -> dict:
    return {"mean": round(float(errors.mean()), 3), "sd": round(float(errors.std(ddof=1)), 3),
            "median": round(float(np.median(errors)), 3), "min": round(float(errors.min()), 3),
            "max": round(float(errors.max()), 3)}


def main() -> None:
    seg = Segmenter(CHECKPOINT_PATH)
    split = load_split()
    test = split[split.split == "test"].reset_index(drop=True)
    rows, cache = [], {}
    for _, r in test.iterrows():
        gray = np.asarray(Image.open(IMAGE_DIR / r.filename).convert("L"), dtype=np.uint8)
        ref, _ = fill_annotation(load_annotation(IMAGE_DIR / r.filename.replace(".png", "_Annotation.png")))
        pred = seg.predict(gray, pixel_size_mm=float(r[PX_COL]))
        d, i = dice_iou(pred.mask, ref)
        rows.append({"filename": r.filename, "dice": d, "iou": i, "hc_ref_mm": r[HC_COL],
                     "hc_pred_mm": pred.hc_mm, "pixel_size_mm": r[PX_COL],
                     "mean_confidence": pred.mean_confidence, "uncertain_fraction": pred.uncertain_fraction})
        cache[r.filename] = (gray, pred.mask, ref)
    df = pd.DataFrame(rows)
    df["hc_error_mm"] = df.hc_pred_mm - df.hc_ref_mm
    df["hc_abs_error_mm"] = df.hc_error_mm.abs()
    df.to_csv(OUT / "test_predictions.csv", index=False)

    fitted = df.dropna(subset=["hc_pred_mm"])
    err = fitted.hc_error_mm.to_numpy()
    metrics = {
        "split": "held-out test (group-level, seed 42)", "test_images": len(df),
        "checkpoint_epoch": seg.meta.get("epoch"), "checkpoint_val_dice": round(seg.meta.get("best_val_dice"), 4),
        "resolution": "original image resolution",
        "dice": describe(df.dice.to_numpy()), "iou": describe(df.iou.to_numpy()),
        "ellipse_fit_failures": int(df.hc_pred_mm.isna().sum()),
        "hc_error_mm": describe(err),
        "hc_mae_mm": round(float(np.abs(err).mean()), 3),
        "hc_abs_error_percent_mean": round(float((np.abs(err) / fitted.hc_ref_mm).mean() * 100), 3),
        "hc_abs_error_mm_percentiles": {f"p{q}": round(float(np.percentile(np.abs(err), q)), 3)
                                        for q in (50, 90, 95)},
        "bland_altman_limits_of_agreement_mm": [round(float(err.mean() - 1.96 * err.std(ddof=1)), 3),
                                                round(float(err.mean() + 1.96 * err.std(ddof=1)), 3)],
    }
    (OUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    SHOTS.mkdir(parents=True, exist_ok=True)
    # Scatter + Bland-Altman
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    a.scatter(fitted.hc_ref_mm, fitted.hc_pred_mm, s=12, alpha=0.7)
    lim = [fitted.hc_ref_mm.min() - 10, fitted.hc_ref_mm.max() + 10]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel("reference HC (mm)")
    a.set_ylabel("predicted HC (mm)")
    a.set_title(f"Test split (n={len(fitted)}): MAE {metrics['hc_mae_mm']:.2f} mm")
    mean_hc = (fitted.hc_ref_mm + fitted.hc_pred_mm) / 2
    b.scatter(mean_hc, err, s=12, alpha=0.7)
    lo, hi = metrics["bland_altman_limits_of_agreement_mm"]
    for y, style in [(err.mean(), "-"), (lo, "--"), (hi, "--")]:
        b.axhline(y, color="red", ls=style, lw=1)
    b.set_xlabel("mean of predicted and reference HC (mm)")
    b.set_ylabel("predicted - reference (mm)")
    b.set_title(f"Bland-Altman: bias {err.mean():+.2f} mm, LoA [{lo:.2f}, {hi:.2f}] mm")
    fig.tight_layout()
    fig.savefig(SHOTS / "test_hc_agreement.png", dpi=110)
    plt.close(fig)

    # Best / median / worst examples by Dice
    order = df.sort_values("dice").filename.tolist()
    picks = [("worst", order[0]), ("2nd worst", order[1]), ("median", order[len(order) // 2]),
             ("best", order[-1])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (label, name) in zip(axes, picks, strict=True):
        gray, mask, ref = cache[name]
        r = df[df.filename == name].iloc[0]
        ax.imshow(overlay(gray, mask, ref))
        hc = f"{r.hc_pred_mm:.1f}" if pd.notna(r.hc_pred_mm) else "n/a"
        ax.set_title(f"{label}: {name}\nDice {r.dice:.3f} | HC {hc} vs {r.hc_ref_mm:.1f} mm", fontsize=9)
        ax.axis("off")
    fig.suptitle("Held-out test split: green = prediction, red = reference outline")
    fig.tight_layout()
    fig.savefig(SHOTS / "test_examples.png", dpi=100)
    plt.close(fig)

    # Qualitative: official HC18 test set (no public labels)
    official = sorted(OFFICIAL_TEST_DIR.glob("*.png"))[:: max(1, len(list(OFFICIAL_TEST_DIR.glob("*.png"))) // 8)][:8]
    if official:
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
        for ax, p in zip(axes.flat, official, strict=False):
            gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            pred = seg.predict(gray)
            ax.imshow(overlay(gray, pred.mask))
            ax.set_title(f"{p.name} (confidence {pred.mean_confidence:.2f})", fontsize=9)
            ax.axis("off")
        fig.suptitle("Official HC18 test set (no public labels): qualitative only")
        fig.tight_layout()
        fig.savefig(SHOTS / "official_test_set_qualitative.png", dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    main()
