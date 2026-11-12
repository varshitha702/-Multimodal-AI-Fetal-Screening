from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
MODEL_DIR = ROOT / "models"
WEIGHTS_PATH = MODEL_DIR / "multimodal_fetal_model.pth"
METRICS_PATH = ROOT / "outputs" / "test_metrics.json"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = 224
ECG_LENGTH = 2048
CLASS_NAMES = ["Abdomen", "Brain", "Femur", "Thorax", "Cervix", "Other"]
MAX_UPLOAD_MB = int(os.getenv("FETALVISION_MAX_UPLOAD_MB", "10"))
MAX_IMAGE_PIXELS = 20_000_000
MODEL_VERSION = os.getenv("FETALVISION_MODEL_VERSION", "research-prototype-v1")
