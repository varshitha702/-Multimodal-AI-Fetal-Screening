
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ml.masks import annotation_to_mask
from ml.metrics import dice_per_sample
from ml.model import UNet


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "raw" / "HC18"
IMAGE_DIR = DATA_ROOT / "training_set"
CSV_PATH = DATA_ROOT / "training_set_pixel_size_and_HC.csv"

OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "hc18_unet_best.pth"
HISTORY_PATH = OUTPUT_DIR / "hc18_training_history.csv"
METRICS_PATH = OUTPUT_DIR / "hc18_metrics.json"


# ============================================================
# 2. LOW-MEMORY TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.20
SEED = 42

CPU_THREADS = 2
LOG_EVERY = 50


# ============================================================
# 3. REPRODUCIBILITY AND RUNTIME SETTINGS
# ============================================================

def configure_runtime():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.set_num_threads(CPU_THREADS)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# ============================================================
# 4. HC18 DATASET
# ============================================================

class HC18Dataset(Dataset):

    def __init__(self, augment=False):

        self.augment = augment

        if not CSV_PATH.is_file():
            raise FileNotFoundError(
                f"HC18 CSV file was not found:\n{CSV_PATH}"
            )

        if not IMAGE_DIR.is_dir():
            raise FileNotFoundError(
                f"HC18 image directory was not found:\n{IMAGE_DIR}"
            )

        table = pd.read_csv(CSV_PATH)

        required_columns = {
            "filename",
            "pixel size(mm)",
            "head circumference (mm)",
        }

        missing_columns = required_columns - set(table.columns)

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing_columns)}"
            )

        self.items = []
        missing_pairs = []

        for _, row in table.iterrows():

            filename = str(row["filename"])

            image_path = IMAGE_DIR / filename

            mask_path = IMAGE_DIR / (
                Path(filename).stem + "_Annotation.png"
            )

            if image_path.is_file() and mask_path.is_file():

                self.items.append(
                    (image_path, mask_path)
                )

            else:
                missing_pairs.append(filename)

        if not self.items:
            raise RuntimeError(
                f"No matching image-mask pairs found in {IMAGE_DIR}"
            )

        if missing_pairs:
            print(
                f"Warning: {len(missing_pairs)} pairs are missing."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        image_path, mask_path = self.items[index]

        # Load ultrasound image
        with Image.open(image_path) as image_file:

            image = image_file.convert("L")

            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # Fill the ellipse outline at ORIGINAL resolution, then resize
        # (resizing the 1-px outline first produced empty masks; see ml/masks.py).
        filled_mask = annotation_to_mask(mask_path, IMAGE_SIZE)

        # Convert to tensors
        image_tensor = torch.from_numpy(
            image_array.copy()
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            (filled_mask > 0).astype(np.float32)
        ).unsqueeze(0)

        # Training augmentation only
        if self.augment and random.random() < 0.5:

            image_tensor = torch.flip(
                image_tensor,
                dims=[2]
            )

            mask_tensor = torch.flip(
                mask_tensor,
                dims=[2]
            )

        return image_tensor, mask_tensor


# ============================================================
# 5. DICE LOSS
# ============================================================

def dice_loss(
    logits,
    target,
    epsilon=1e-7
):

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * target
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
    )

    dice_score = (
        (2.0 * intersection + epsilon)
        / (denominator + epsilon)
    )

    return 1.0 - dice_score.mean()


# ============================================================
# 6. DICE METRIC
# ============================================================

def dice_metric(logits, target):
    """Mean hard Dice; empty targets are undefined (NaN) and never count as 1.0."""
    return float(torch.nanmean(dice_per_sample(logits, target)).item())


# ============================================================
# 7. TRAINING OR VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
    epoch,
    total_epochs
):

    if training:

        model.train()
        phase = "TRAIN"

    else:

        model.eval()
        phase = "VALIDATION"

    bce_loss = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    total_batches = len(loader)

    start_time = time.time()

    for batch_index, (images, masks) in enumerate(
        loader,
        start=1
    ):

        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = model(images)

            loss = (
                bce_loss(logits, masks)
                + dice_loss(logits, masks)
            )

            if training:

                loss.backward()

                optimizer.step()

        current_batch_size = images.size(0)

        total_loss += (
            loss.item() * current_batch_size
        )

        batch_dice = dice_metric(
            logits.detach(),
            masks
        )

        total_dice += (
            batch_dice * current_batch_size
        )

        total_samples += current_batch_size

        del images
        del masks
        del logits
        del loss

        if (
            batch_index % LOG_EVERY == 0
            or batch_index == total_batches
        ):

            average_loss = (
                total_loss / max(total_samples, 1)
            )

            average_dice = (
                total_dice / max(total_samples, 1)
            )

            print(
                f"\rEpoch {epoch}/{total_epochs} "
                f"[{phase}] "
                f"Batch {batch_index}/{total_batches} "
                f"Loss: {average_loss:.4f} "
                f"Dice: {average_dice:.4f}",
                end="\n" if batch_index == total_batches else "",
                flush=True
            )

    average_loss = (
        total_loss / max(total_samples, 1)
    )

    average_dice = (
        total_dice / max(total_samples, 1)
    )

    elapsed_seconds = time.time() - start_time

    print(
        f"{phase} completed in "
        f"{elapsed_seconds / 60:.1f} minutes."
    )

    return average_loss, average_dice


# ============================================================
# 8. CREATE TRAINING AND VALIDATION LOADERS
# ============================================================

def create_dataloaders():

    base_dataset = HC18Dataset(augment=False)

    dataset_size = len(base_dataset)

    validation_size = max(
        1,
        int(dataset_size * VALIDATION_FRACTION)
    )

    training_size = dataset_size - validation_size

    if training_size < 1:

        raise ValueError(
            "Not enough samples for training and validation."
        )

    generator = torch.Generator().manual_seed(SEED)

    shuffled_indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    validation_indices = shuffled_indices[:validation_size]

    training_indices = shuffled_indices[validation_size:]

    # Separate datasets prevent validation augmentation.
    training_dataset = HC18Dataset(augment=True)

    validation_dataset = HC18Dataset(augment=False)

    training_subset = Subset(
        training_dataset,
        training_indices
    )

    validation_subset = Subset(
        validation_dataset,
        validation_indices
    )

    training_loader = DataLoader(
        training_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return (
        training_loader,
        validation_loader,
        training_size,
        validation_size
    )


# ============================================================
# 9. SAVE BEST CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    epoch,
    validation_dice
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_data = {

        "model_state_dict": model.state_dict(),

        "image_size": IMAGE_SIZE,

        "best_val_dice": validation_dice,

        "epoch": epoch,

        "architecture": "HC18 U-Net"
    }

    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")

    torch.save(
        checkpoint_data,
        temporary_path
    )

    temporary_path.replace(CHECKPOINT_PATH)

    print("\nBest checkpoint saved:")
    print(CHECKPOINT_PATH)


# ============================================================
# 10. MAIN TRAINING FUNCTION
# ============================================================

def main():

    configure_runtime()

    print("\n" + "=" * 58)

    print("       FetalAI HC18 U-Net Training")

    print("=" * 58)

    print("Project directory:", ROOT)

    print("Dataset directory:", IMAGE_DIR)

    print("CSV file:", CSV_PATH)

    print("Image size:", IMAGE_SIZE)

    print("Training batch size:", BATCH_SIZE)

    print("Validation batch size: 1")

    print("Epochs:", EPOCHS)

    print("Learning rate:", LEARNING_RATE)

    print("CPU threads:", CPU_THREADS)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    if device.type == "cpu":

        print("Low-memory CPU mode: enabled")

    # Verify dataset paths
    if not CSV_PATH.is_file():

        raise FileNotFoundError(
            f"HC18 CSV not found: {CSV_PATH}"
        )

    if not IMAGE_DIR.is_dir():

        raise FileNotFoundError(
            f"HC18 image folder not found: {IMAGE_DIR}"
        )

    # Load dataset
    (
        training_loader,
        validation_loader,
        training_count,
        validation_count
    ) = create_dataloaders()

    print("\nDataset successfully loaded.")

    print("Training samples:", training_count)

    print("Validation samples:", validation_count)

    print(
        "Total usable image-mask pairs:",
        training_count + validation_count
    )

    # Initialize original U-Net architecture
    print("\nInitializing U-Net model...")

    model = UNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_validation_dice = -1.0

    history = []

    print("\nStarting training...\n")

    # Training loop
    for epoch in range(1, EPOCHS + 1):

        print("\n" + "-" * 58)

        print(f"Starting epoch {epoch}/{EPOCHS}")

        print("-" * 58)

        # Training
        train_loss, train_dice = run_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            device=device,
            training=True,
            epoch=epoch,
            total_epochs=EPOCHS
        )

        # Validation
        val_loss, val_dice = run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=optimizer,
            device=device,
            training=False,
            epoch=epoch,
            total_epochs=EPOCHS
        )

        print("\nEpoch summary")

        print(f"Training loss:     {train_loss:.4f}")

        print(f"Training Dice:     {train_dice:.4f}")

        print(f"Validation loss:   {val_loss:.4f}")

        print(f"Validation Dice:   {val_dice:.4f}")

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_dice": train_dice,

            "validation_loss": val_loss,

            "validation_dice": val_dice

        })

        # Save best model
        if val_dice > best_validation_dice:

            best_validation_dice = val_dice

            save_checkpoint(
                model=model,
                epoch=epoch,
                validation_dice=best_validation_dice
            )

        # Save training history
        pd.DataFrame(history).to_csv(
            HISTORY_PATH,
            index=False
        )

        # Save evaluation metrics
        metrics = {

            "architecture": "HC18 U-Net",

            "dataset": "HC18",

            "image_size": IMAGE_SIZE,

            "batch_size": BATCH_SIZE,

            "epochs_completed": epoch,

            "best_validation_dice": best_validation_dice,

            "latest_train_loss": train_loss,

            "latest_train_dice": train_dice,

            "latest_validation_loss": val_loss,

            "latest_validation_dice": val_dice,

            "training_samples": training_count,

            "validation_samples": validation_count,

            "device": str(device),

            "split_strategy": "random_image_level",

            "clinical_validity": (
                "Not established. Research prototype only."
            )
        }

        METRICS_PATH.write_text(
            json.dumps(metrics, indent=4),
            encoding="utf-8"
        )

        gc.collect()

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # Final results
    print("\n" + "=" * 58)

    print("             TRAINING FINISHED")

    print("=" * 58)

    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.4f}"
    )

    print("Best model:", CHECKPOINT_PATH)

    print("Training history:", HISTORY_PATH)

    print("Metrics:", METRICS_PATH)

    print(
        "\nThis is an academic research prototype. "
        "It is not clinically validated and must not be used "
        "for diagnosis or pregnancy-care decisions."
    )


# ============================================================
# 11. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
