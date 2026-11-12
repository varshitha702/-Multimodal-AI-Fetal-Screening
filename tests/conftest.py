import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HC18_TRAIN = ROOT / "data" / "raw" / "HC18" / "training_set"
