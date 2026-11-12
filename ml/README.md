# ml/ — data pipeline, model, training and evaluation

Owners: Pujari Varshitha (dataset and preprocessing), Nachiketha K G (training and evaluation).

| Module | Purpose |
|---|---|
| `masks.py` | HC18 outline annotation -> filled mask (fill at original resolution, then resize) |
| `metrics.py` | Dice / IoU that never score empty-vs-empty as perfect; soft Dice loss |
| `split.py` | Fixed-seed 70/15/15 split by examination group, HC-stratified: `python -m ml.split` |
| `model.py` | U-Net architecture |
| `train.py` | Training entry point: `python -m ml.train` (run from the repository root) |
