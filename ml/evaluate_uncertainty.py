"""Calibrate (validation) and evaluate (held-out test) the TTA uncertainty of Feature B.

    python -m ml.evaluate_uncertainty

VALIDATION split (rules fixed before running):
  * thresholds of the "low reliability" flag = 90th percentile of HC CV (%) and of the disagreement fraction;
  * primary HC = single pass or TTA mean, whichever has the lower validation MAE.
TEST split (n = 151): Spearman correlation of the uncertainty with the actual HC error (bootstrap 95 % CI and
p-value), error of flagged vs unflagged images, and latency of a single pass vs TTA.
Writes outputs/uncertainty_calibration.json, outputs/uncertainty_metrics.json, outputs/uncertainty_predictions.csv
and docs/screenshots/uncertainty_*.png.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from ml.dataset import IMAGE_DIR  # noqa: E402
from ml.inference import Segmenter  # noqa: E402
from ml.split import HC_COL, PX_COL, load_split  # noqa: E402
from ml.train import CHECKPOINT_PATH  # noqa: E402
from ml.uncertainty import CALIBRATION_PATH, TRANSFORMS, tta_predict, uncertainty_overlay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SHOTS = ROOT / "docs" / "screenshots"
FLAG_PERCENTILE = 90


def run(seg: Segmenter, rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        gray = np.asarray(Image.open(IMAGE_DIR / r.filename).convert("L"), dtype=np.uint8)
        px = float(r[PX_COL])
        single = seg.predict(gray, px)
        t = tta_predict(seg, gray, px)
        out.append({"filename": r.filename, "split": r.split, "hc_ref_mm": r[HC_COL], "hc_single_mm": single.hc_mm,
                    "hc_tta_mean_mm": t.hc_mean_mm, "hc_tta_sd_mm": t.hc_sd_mm, "hc_tta_cv_percent": t.hc_cv_percent,
                    "disagreement_fraction": t.disagreement_fraction,
                    "passes_without_ellipse": sum(v is None for v in t.hc_passes_mm)})
    df = pd.DataFrame(out)
    for k in ("single", "tta_mean"):
        df[f"abs_err_{k}_mm"] = (df[f"hc_{k}_mm"] - df.hc_ref_mm).abs()
    return df


def spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"rho": round(float(rho), 3), "p_value": float(f"{p:.3g}"),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "n": int(len(x))}


def latency(seg: Segmenter, names: list[str], repeats: int = 3) -> dict:
    grays = [np.asarray(Image.open(IMAGE_DIR / n).convert("L"), dtype=np.uint8) for n in names]
    seg.predict(grays[0], 0.1)
    tta_predict(seg, grays[0], 0.1)
    res = {}
    for name, fn in (("single_pass_ms", lambda g: seg.predict(g, 0.1)), ("tta_ms", lambda g: tta_predict(seg, g, 0.1))):
        t0 = time.perf_counter()
        for _ in range(repeats):
            for g in grays:
                fn(g)
        res[name] = round((time.perf_counter() - t0) * 1000 / (repeats * len(grays)), 1)
    return res


def cpu_latency(names: list[str]) -> dict:
    """CPU latency in a fresh CPU-only subprocess (a CUDA-initialised process and long runs distort CPU timing)."""
    code = ("import json, sys; from ml.inference import Segmenter; from ml.train import CHECKPOINT_PATH; "
            "from ml.evaluate_uncertainty import latency; "
            "print(json.dumps(latency(Segmenter(CHECKPOINT_PATH, device='cpu'), sys.argv[1:], repeats=1)))")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", code, *names], capture_output=True, text=True, env=env, cwd=ROOT,
                         check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    seg = Segmenter(CHECKPOINT_PATH)
    split = load_split()
    df = run(seg, split[split.split.isin(["val", "test"])].reset_index(drop=True))
    df.to_csv(OUT / "uncertainty_predictions.csv", index=False)
    val, test = df[df.split == "val"], df[df.split == "test"]

    # ---- validation: thresholds and primary HC ----
    t_cv = float(np.nanpercentile(val.hc_tta_cv_percent, FLAG_PERCENTILE))
    t_dis = float(np.nanpercentile(val.disagreement_fraction, FLAG_PERCENTILE))
    val_mae = {k: float(val[f"abs_err_{k}_mm"].mean()) for k in ("single", "tta_mean")}
    primary = min(val_mae, key=val_mae.get)
    calibration = {
        "rule": f"thresholds = {FLAG_PERCENTILE}th percentile on the validation split; flag if either is exceeded",
        "thresholds": {"hc_cv_percent": round(t_cv, 4), "disagreement_fraction": round(t_dis, 4)},
        "primary_hc": primary,
        "primary_rule": "lower validation MAE of single pass vs TTA mean",
        "validation_mae_mm": {k: round(v, 3) for k, v in val_mae.items()},
        "validation_n": int(len(val)),
        "transforms": [t.name for t in TRANSFORMS],
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2))

    # ---- test ----
    test = test.dropna(subset=["hc_tta_sd_mm", f"hc_{primary}_mm"]).copy()
    err = test[f"abs_err_{primary}_mm"].to_numpy()
    rel_err = 100 * err / test.hc_ref_mm.to_numpy()
    flagged = ((test.hc_tta_cv_percent > t_cv) | (test.disagreement_fraction > t_dis)).to_numpy()
    big = err >= np.percentile(err, 90)
    metrics = {
        "split": "held-out test (group-level, seed 42)", "n": int(len(test)), "n_passes": len(TRANSFORMS),
        "primary_hc": primary,
        "hc_mae_mm": {"single": round(float(test.abs_err_single_mm.mean()), 3),
                      "tta_mean": round(float(test.abs_err_tta_mean_mm.mean()), 3)},
        "hc_sd_mm": {"median": round(float(test.hc_tta_sd_mm.median()), 3),
                     "p90": round(float(test.hc_tta_sd_mm.quantile(0.9)), 3),
                     "max": round(float(test.hc_tta_sd_mm.max()), 3)},
        "spearman": {
            "sd_mm_vs_abs_error_mm": spearman(test.hc_tta_sd_mm.to_numpy(), err),
            "cv_percent_vs_abs_error_percent": spearman(test.hc_tta_cv_percent.to_numpy(), rel_err),
            "disagreement_vs_abs_error_mm": spearman(test.disagreement_fraction.to_numpy(), err),
        },
        "flag": {
            "thresholds": calibration["thresholds"],
            "flagged": int(flagged.sum()), "flagged_rate": round(float(flagged.mean()), 4),
            "mae_flagged_mm": round(float(err[flagged].mean()), 3) if flagged.any() else None,
            "mae_unflagged_mm": round(float(err[~flagged].mean()), 3),
            "largest_10pct_errors_flagged": f"{int((flagged & big).sum())} / {int(big.sum())}",
            "largest_10pct_errors_threshold_mm": round(float(np.percentile(err, 90)), 3),
        },
    }
    device = str(seg.device)
    metrics["latency"] = {device: latency(seg, test.filename.head(20).tolist())}
    if seg.device.type == "cuda":
        metrics["latency"]["cpu"] = cpu_latency(test.filename.head(3).tolist())
    metrics["latency"]["note"] = (
        "per image, 800x540, laptop (RTX 4050 / CPU), model only (no HTTP, no PNG encoding). CPU measured in a "
        "fresh CPU-only process over a short burst: under sustained load this laptop throttles the process and "
        "both numbers grow ~8x.")
    (OUT / "uncertainty_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(calibration, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---- figures ----
    SHOTS.mkdir(parents=True, exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    s = metrics["spearman"]["sd_mm_vs_abs_error_mm"]
    a.scatter(test.hc_tta_sd_mm, err, s=14, c=np.where(flagged, "tab:red", "tab:blue"), alpha=0.75)
    a.set_xlabel("TTA HC SD (mm)")
    a.set_ylabel(f"|HC error| (mm), {primary.replace('_', ' ')}")
    a.set_title(f"Test n={len(test)}: Spearman rho {s['rho']:.2f} (95% CI {s['ci95'][0]:.2f} to {s['ci95'][1]:.2f})",
                fontsize=10)
    c = metrics["spearman"]["cv_percent_vs_abs_error_percent"]
    b.scatter(test.hc_tta_cv_percent, rel_err, s=14, c=np.where(flagged, "tab:red", "tab:blue"), alpha=0.75)
    b.axvline(t_cv, color="red", ls="--", lw=1, label=f"CV threshold {t_cv:.2f}% (val P{FLAG_PERCENTILE})")
    b.set_xlabel("TTA HC CV (%)")
    b.set_ylabel("|HC error| (% of reference)")
    b.set_title(f"Relative: Spearman rho {c['rho']:.2f} (95% CI {c['ci95'][0]:.2f} to {c['ci95'][1]:.2f})",
                fontsize=10)
    b.legend(fontsize=8)
    fig.suptitle("Red = flagged 'low reliability' (thresholds from the validation split)")
    fig.tight_layout()
    fig.savefig(SHOTS / "uncertainty_vs_error.png", dpi=120)
    plt.close(fig)

    picks = test.sort_values("hc_tta_sd_mm")
    picks = [("lowest SD", picks.iloc[0]), ("median SD", picks.iloc[len(picks) // 2]),
             ("highest SD", picks.iloc[-1]), ("2nd highest SD", picks.iloc[-2])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (label, r) in zip(axes, picks, strict=True):
        gray = np.asarray(Image.open(IMAGE_DIR / r.filename).convert("L"), dtype=np.uint8)
        t = tta_predict(seg, gray, float(split.set_index("filename").loc[r.filename, PX_COL]))
        ax.imshow(uncertainty_overlay(gray, t.std_map))
        ax.set_title(f"{label}: {r.filename}\nHC {r.hc_tta_mean_mm:.1f} ± {r.hc_tta_sd_mm:.2f} mm, "
                     f"ref {r.hc_ref_mm:.1f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Per-pixel TTA uncertainty (SD of 8 probability maps), held-out test images")
    fig.tight_layout()
    fig.savefig(SHOTS / "uncertainty_examples.png", dpi=100)
    plt.close(fig)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
