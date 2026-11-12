# ml/ — data pipeline, model, training and evaluation

Owners: Pujari Varshitha (dataset and preprocessing), Nachiketha K G (training and evaluation).

| Module | Purpose |
|---|---|
| `masks.py` | HC18 outline annotation -> filled mask (fill at original resolution, then resize) |
| `metrics.py` | Dice / IoU that never score empty-vs-empty as perfect; soft Dice loss |
| `split.py` | Fixed-seed 70/15/15 split by examination group, HC-stratified: `python -m ml.split` |
| `model.py` | U-Net architecture |
| `dataset.py` | Preloaded HC18 dataset with augmentation (flip, rotation, scale, brightness/contrast) |
| `train.py` | GPU training (AMP, Adam, BCE + Dice, early stopping): `python -m ml.train` |

## Reproducing the trained model

The checkpoint `outputs/hc18_unet_best.pth` (31 MB) is **not committed** (`*.pth` is gitignored).
Regenerate it on a CUDA GPU (about 12 minutes on an RTX 4050 Laptop 6 GB):

```powershell
python -m ml.split   # optional: the split is already committed in data/splits/
python -m ml.train   # writes outputs/hc18_unet_best.pth, training_history.csv, training_curves.png
```

Seeds are fixed (42), but cuDNN/AMP kernels can still give tiny run-to-run differences.
