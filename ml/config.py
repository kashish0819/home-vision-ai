from pathlib import Path
import torch

# ======================================================
# Project Root
# ======================================================

# config.py is inside ml/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "saved_models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================================================
# Dataset
# ======================================================

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "valid.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

IMAGE_DIR = DATA_DIR / "eye_image"

# ======================================================
# Training
# ======================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_EPOCHS = 25

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
PATIENCE = 5

# ======================================================
# Device
# ======================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PROJECT_ROOT :", PROJECT_ROOT)
print("TRAIN_CSV    :", TRAIN_CSV)
print("VAL_CSV      :", VAL_CSV)
print("IMAGE_DIR    :", IMAGE_DIR)
print("Device       :", DEVICE)
print("=" * 60)