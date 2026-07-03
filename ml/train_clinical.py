from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from ml.datasets.clinical_dataset import ClinicalDataset
from ml.models.clinical_model import ClinicalModel

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "clinical" / "anemia.csv"

MODEL_DIR = PROJECT_ROOT / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# SETTINGS
# =====================================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

# =====================================================
# INFO
# =====================================================

print("=" * 60)
print("Clinical Model Training")
print("=" * 60)
print("Device :", DEVICE)
print("Dataset:", DATA_FILE)
print("=" * 60)

# =====================================================
# DATASETS
# =====================================================

train_dataset = ClinicalDataset(
    csv_file=DATA_FILE,
    train=True
)

val_dataset = ClinicalDataset(
    csv_file=DATA_FILE,
    train=False
)

print("Training Samples  :", len(train_dataset))
print("Validation Samples:", len(val_dataset))

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =====================================================
# MODEL
# =====================================================

model = ClinicalModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

print("\nModel Loaded Successfully")
print("Loss Function :", criterion)
print("Optimizer : Adam")

# =====================================================
# TRAINING
# =====================================================

print("\nStarting Training...\n")

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(NUM_EPOCHS):

    print("=" * 60)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    print("=" * 60)

    # -----------------------------
    # TRAIN
    # -----------------------------

    model.train()

    train_loss = 0.0
    train_preds = []
    train_labels = []

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)

    train_acc = accuracy_score(train_labels, train_preds)

    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

    val_loss = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)

    val_acc = accuracy_score(val_labels, val_preds)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")

    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------

    if val_acc > best_accuracy:

        best_accuracy = val_acc
        best_weights = copy.deepcopy(model.state_dict())

        torch.save(
            best_weights,
            MODEL_DIR / "clinical_model.pth"
        )

        print("✅ Best Model Saved")

print("\n" + "=" * 60)
print("Training Finished!")
print(f"Best Validation Accuracy : {best_accuracy:.4f}")
print("=" * 60)