"""Fixed-seed, leakage-safe train/val/test split of the 999 labelled HC18 images.

HC18 has no patient IDs, but filenames share a numeric prefix when several images come
from the same examination (``014_HC.png``, ``014_2HC.png``, ``014_3HC.png``; their
reference HCs differ by only a few mm). Splitting single images at random would put near-duplicates of
the same fetus in both train and test, so we split by **prefix group** (806 groups).
Groups are stratified by their mean HC (quintiles) so every split covers the whole
gestational range. The official HC18 test set has no public labels and is used only for
qualitative demos.

Usage:  python -m ml.split          -> writes data/splits/hc18_split.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "HC18" / "training_set_pixel_size_and_HC.csv"
SPLIT_PATH = ROOT / "data" / "splits" / "hc18_split.csv"
SUMMARY_PATH = ROOT / "outputs" / "split_summary.json"

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
HC_COL = "head circumference (mm)"
PX_COL = "pixel size(mm)"
_GROUP_RE = re.compile(r"^(\d+)_")


def group_id(filename: str) -> str:
    """'014_3HC.png' -> '014'."""
    m = _GROUP_RE.match(filename)
    if not m:
        raise ValueError(f"Unexpected HC18 filename: {filename}")
    return m.group(1)


def make_split(table: pd.DataFrame, seed: int = SEED, n_bins: int = 5) -> pd.DataFrame:
    """Return ``table`` with added ``group`` and ``split`` columns."""
    df = table.copy()
    df["group"] = df["filename"].map(group_id)
    groups = df.groupby("group")[HC_COL].mean().rename("hc").reset_index()
    groups["bin"] = pd.qcut(groups["hc"], q=n_bins, labels=False, duplicates="drop")

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for _, members in groups.groupby("bin"):
        ids = members["group"].to_numpy()
        rng.shuffle(ids)
        n_train = round(len(ids) * FRACTIONS["train"])
        n_val = round(len(ids) * FRACTIONS["val"])
        for i, g in enumerate(ids):
            assignment[g] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    df["split"] = df["group"].map(assignment)
    return df


def summarize(df: pd.DataFrame) -> dict:
    out = {"seed": SEED, "unit": "examination group (filename prefix)", "total_images": len(df),
           "total_groups": int(df["group"].nunique()), "splits": {}}
    for name in FRACTIONS:
        part = df[df["split"] == name]
        out["splits"][name] = {
            "images": len(part),
            "groups": int(part["group"].nunique()),
            "image_fraction": round(len(part) / len(df), 4),
            "hc_mm_mean": round(float(part[HC_COL].mean()), 2),
            "hc_mm_min": round(float(part[HC_COL].min()), 2),
            "hc_mm_max": round(float(part[HC_COL].max()), 2),
        }
    return out


def load_split(path: Path = SPLIT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"group": str})


def main() -> None:
    df = make_split(pd.read_csv(CSV_PATH))
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["filename", "group", "split", PX_COL, HC_COL]].to_csv(SPLIT_PATH, index=False)
    summary = summarize(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {SPLIT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
