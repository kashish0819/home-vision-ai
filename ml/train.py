from pathlib import Path
import time
import copy

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from ml.datasets.image_dataset import ImageDataset
from ml.models.cnn_model import CNNModel
from ml.losses import get_loss_function
from ml.utils import (
    calculate_metrics,
    save_checkpoint,
    plot_history
)

# ======================================================
# IMPORT CONFIG
# ======================================================

from ml.config import (
    TRAIN_CSV,
    VAL_CSV,
    IMAGE_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    NUM_EPOCHS
)

print("=" * 70)
print("Anemia FusionNet Training")
print("=" * 70)
print("Device :", DEVICE)
print("=" * 70)

print("TRAIN_CSV :", TRAIN_CSV)
print("VAL_CSV   :", VAL_CSV)
print("IMAGE_DIR :", IMAGE_DIR)

# ======================================================
# DATASETS
# ======================================================

train_dataset = ImageDataset(
    csv_file=TRAIN_CSV,
    image_root=IMAGE_DIR
)

val_dataset = ImageDataset(
    csv_file=VAL_CSV,
    image_root=IMAGE_DIR
)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")

# ======================================================
# DATALOADERS
# ======================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("\nDataLoaders Ready")

# ======================================================
# MODEL
# ======================================================

model = CNNModel().to(DEVICE)

print("\nModel Loaded Successfully")

# ======================================================
# LOSS
# ======================================================

criterion = get_loss_function()

print("Loss Function :", criterion)

# ======================================================
# OPTIMIZER
# ======================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

print("Optimizer : AdamW")

# ======================================================
# LR SCHEDULER
# ======================================================

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

print("Scheduler : CosineAnnealingLR")

# ======================================================
# TRAINING
# ======================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_acc = 0.0

print("\nStarting Training...\n")

for epoch in range(NUM_EPOCHS):

    print("=" * 60)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    print("=" * 60)

    # -------------------------------
    # TRAIN
    # -------------------------------

    model.train()

    running_loss = 0.0

    train_preds = []
    train_labels = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.cpu().numpy())

        train_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)

    train_acc = accuracy_score(train_labels, train_preds)

    # -------------------------------
    # VALIDATION
    # -------------------------------

    model.eval()

    val_loss = 0.0

    val_preds = []

    val_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())

            val_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)

    val_acc = accuracy_score(val_labels, val_preds)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")

    # -------------------------------
    # SAVE BEST MODEL
    # -------------------------------

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_DIR / "best_model.pth"
        )

        print("Best Model Saved")

print("\nTraining Finished!")

print(f"Best Validation Accuracy : {best_acc:.4f}")