"""Train the HC18 U-Net on the GPU.

    python -m ml.train [--epochs 100] [--batch-size 16] [--size 256]

Loss BCE + soft Dice, Adam, AMP, ReduceLROnPlateau, early stopping on validation Dice.
Writes (outputs/): hc18_unet_best.pth (gitignored), training_history.csv,
training_summary.json, training_curves.png.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ml.dataset import HC18Dataset  # noqa: E402
from ml.metrics import dice_per_sample, iou_per_sample, soft_dice_loss  # noqa: E402
from ml.model import UNet  # noqa: E402
from ml.split import SPLIT_PATH, load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHECKPOINT_PATH = OUT / "hc18_unet_best.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    """One pass over ``loader``. Trains if an optimizer is given. Returns (loss, dice, iou)."""
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()
    total_loss, dices, ious = 0.0, [], []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = bce(logits, masks) + soft_dice_loss(logits.float(), masks)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(images)
        dices.append(dice_per_sample(logits.detach().float(), masks).cpu())
        ious.append(iou_per_sample(logits.detach().float(), masks).cpu())
    n = sum(len(d) for d in dices)
    return (total_loss / n,
            float(torch.nanmean(torch.cat(dices))),
            float(torch.nanmean(torch.cat(ious))))


def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.plot(history.epoch, history.train_loss, label="train")
    a.plot(history.epoch, history.val_loss, label="validation")
    a.set_title("Loss (BCE + Dice)")
    a.set_xlabel("epoch")
    a.legend()
    b.plot(history.epoch, history.train_dice, label="train Dice")
    b.plot(history.epoch, history.val_dice, label="validation Dice")
    b.plot(history.epoch, history.val_iou, label="validation IoU", linestyle="--")
    best = history.val_dice.idxmax()
    b.scatter([history.epoch[best]], [history.val_dice[best]], color="red", zorder=5,
              label=f"best val Dice {history.val_dice[best]:.4f} (epoch {history.epoch[best]})")
    b.set_title("Segmentation metrics")
    b.set_xlabel("epoch")
    b.set_ylim(0, 1)
    b.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({device_name})")

    split = load_split()
    t0 = time.time()
    train_ds = HC18Dataset(split[split.split == "train"], args.size, train=True, seed=args.seed)
    val_ds = HC18Dataset(split[split.split == "val"], args.size)
    n_test = int((split.split == "test").sum())
    print(f"Loaded train={len(train_ds)} val={len(val_ds)} in {time.time() - t0:.1f}s "
          f"(test split of {n_test} images is held out)")

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    OUT.mkdir(exist_ok=True)
    history, best_dice, best_epoch, stale = [], -1.0, 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        e0 = time.time()
        train_ds.set_epoch(epoch)
        tr_loss, tr_dice, _ = run_epoch(model, train_loader, device, optimizer, scaler)
        with torch.no_grad():
            va_loss, va_dice, va_iou = run_epoch(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_dice)
        history.append({"epoch": epoch, "lr": lr, "train_loss": tr_loss, "train_dice": tr_dice,
                        "val_loss": va_loss, "val_dice": va_dice, "val_iou": va_iou,
                        "seconds": round(time.time() - e0, 1)})
        improved = va_dice > best_dice
        if improved:
            best_dice, best_epoch, stale = va_dice, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "architecture": "UNet(base=32, in=1, out=1)",
                        "image_size": args.size, "normalization": "grayscale / 255", "epoch": epoch,
                        "best_val_dice": va_dice, "seed": args.seed, "split_file": SPLIT_PATH.name},
                       CHECKPOINT_PATH)
        else:
            stale += 1
        print(f"epoch {epoch:3d} | lr {lr:.1e} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
              f"val loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f} | {time.time() - e0:.0f}s"
              + ("  *best*" if improved else ""), flush=True)
        pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
        if stale >= args.patience:
            print(f"Early stopping: no val Dice improvement for {args.patience} epochs.")
            break

    hist = pd.DataFrame(history)
    plot_history(hist, OUT / "training_curves.png")
    summary = {
        "model": "U-Net (base 32 channels)", "parameters": n_params, "image_size": args.size,
        "batch_size": args.batch_size, "optimizer": "Adam", "initial_lr": args.lr,
        "lr_schedule": "ReduceLROnPlateau(max val Dice, factor 0.5, patience 5)", "loss": "BCE + soft Dice",
        "amp": device.type == "cuda",
        "augmentation": "hflip, rotation +-15 deg, scale 0.9-1.1, brightness/contrast",
        "early_stopping_patience": args.patience, "seed": args.seed, "device": device_name,
        "train_images": len(train_ds), "val_images": len(val_ds), "epochs_run": len(hist),
        "best_epoch": best_epoch, "best_val_dice": round(best_dice, 4),
        "best_val_iou": round(float(hist.loc[hist.val_dice.idxmax(), "val_iou"]), 4),
        "training_minutes": round((time.time() - start) / 60, 1),
        "note": "Validation metrics only. Held-out test results are produced by ml/evaluate.py.",
    }
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
