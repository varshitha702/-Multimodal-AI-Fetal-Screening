
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "raw" / "HC18"
IMAGE_DIR = DATA_ROOT / "training_set" / "training_set"
CSV_FILE = DATA_ROOT / "training_set_pixel_size_and_HC.csv"

print("\n--- HC18 DATASET CHECK ---")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

# Include both HC and 2HC naming formats
images = [
    p for p in IMAGE_DIR.glob("*.png")
    if not p.stem.endswith("_Annotation")
]

masks = list(IMAGE_DIR.glob("*_Annotation.png"))

image_ids = {p.stem for p in images}
mask_ids = {
    p.stem.removesuffix("_Annotation")
    for p in masks
}

print("Ultrasound images:", len(images))
print("Annotation masks:", len(masks))
print("Matching pairs:", len(image_ids & mask_ids))
print("Images without masks:", len(image_ids - mask_ids))
print("Masks without images:", len(mask_ids - image_ids))

if CSV_FILE.exists():
    df = pd.read_csv(CSV_FILE)

    print("\nCSV records:", len(df))
    print("CSV columns:", df.columns.tolist())

    csv_names = set(df["filename"].astype(str))
    image_names = {p.name for p in images}

    print("CSV filenames matched:", len(csv_names & image_names))
    print("CSV filenames missing:", len(csv_names - image_names))
    print("Images missing from CSV:", len(image_names - csv_names))

else:
    print("CSV file not found!")
