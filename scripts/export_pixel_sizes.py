"""Export HC18 pixel sizes (mm/px) as JSON so the frontend can auto-fill spacing for demo images.

Keys are the first 16 hex chars of the SHA-256 of the image FILE BYTES, because the official
test set reuses the training-set filenames (000_HC.png exists in both, as different images).
The browser hashes the uploaded file with crypto.subtle and looks it up.

    python scripts/export_pixel_sizes.py  ->  frontend/src/lib/hc18-pixel-sizes.json

Only metadata (filename -> pixel size) is exported; reference HC values are deliberately NOT
included so the demo cannot show ground truth for the held-out test split.
Source: HC18, van den Heuvel et al., PLoS ONE 2018, doi:10.5281/zenodo.1327317 (CC BY 4.0).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HC18 = ROOT / "data" / "raw" / "HC18"
OUT = ROOT / "frontend" / "src" / "lib" / "hc18-pixel-sizes.json"


def main() -> None:
    sizes: dict[str, dict] = {}
    for folder, csv in (("training_set", "training_set_pixel_size_and_HC.csv"),
                        ("test_set", "test_set_pixel_size.csv")):
        df = pd.read_csv(HC18 / csv)
        col = next(c for c in df.columns if c.startswith("pixel size"))
        for _, r in df.iterrows():
            key = hashlib.sha256((HC18 / folder / r.filename).read_bytes()).hexdigest()[:16]
            px = round(float(r[col]), 8)
            if key in sizes:
                # HC18 contains one byte-identical pair (training_set 392_HC.png == 392_2HC.png, same px)
                if sizes[key]["px"] != px:
                    raise RuntimeError(f"Same image, different pixel size: {folder}/{r.filename}")
                continue
            sizes[key] = {"file": f"{folder}/{r.filename}", "px": px}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sizes, separators=(",", ":"), sort_keys=True))
    print(f"Wrote {len(sizes)} entries to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
