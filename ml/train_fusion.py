from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from ml.config import *
from ml.datasets.fusion_dataset import FusionDataset
from ml.models.fusion_model import FusionModel


# ============================================================
# Configuration
# ============================================================

print("=" * 70)
print("FusionNet Training")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

FUSION_CSV = PROCESSED_DIR / "fusion_data.csv"

print("Fusion CSV :", FUSION_CSV)


# ============================================================
# Dataset
# ============================================================

dataset = FusionDataset(
    csv_file=FUSION_CSV,
    image_root=IMAGE_DIR
)

print("Total Samples :", len(dataset))


# ============================================================
# Train / Validation Split
# ============================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training Samples  :", len(train_dataset))
print("Validation Samples:", len(val_dataset))


# ============================================================
# DataLoaders
# ============================================================

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


# ============================================================
# Model
# ============================================================

model = FusionModel().to(device)

print("Fusion Model Loaded Successfully")


# ============================================================
# Loss Function
# ============================================================

criterion = nn.CrossEntropyLoss()

print("Loss :", criterion)


# ============================================================
# Optimizer
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("Optimizer :", optimizer.__class__.__name__)


# ============================================================
# Learning Rate Scheduler
# ============================================================

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

print("Scheduler :", scheduler.__class__.__name__)


# ============================================================
# Training History
# ============================================================

train_losses = []
val_losses = []

train_accs = []
val_accs = []

best_acc = 0.0
best_model = copy.deepcopy(model.state_dict())

# ============================================================
# Training Function
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, clinical, geo, labels in train_loader:

        images = images.to(device)
        clinical = clinical.to(device)
        geo = geo.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            images,
            clinical,
            geo
        )

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# ============================================================
# Validation Function
# ============================================================

def validate_one_epoch():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, clinical, geo, labels in val_loader:

            images = images.to(device)
            clinical = clinical.to(device)
            geo = geo.to(device)
            labels = labels.to(device)

            outputs = model(
                images,
                clinical,
                geo
            )

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)

    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

# ============================================================
# Start Training
# ============================================================

print("\nStarting FusionNet Training...")

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    print("=" * 60)

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate_one_epoch()

    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.2f}%")

    if val_acc > best_acc:

        best_acc = val_acc

        best_model = copy.deepcopy(model.state_dict())

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        save_path = MODEL_DIR / "fusion_best.pth"

        torch.save(best_model, save_path)

        print("\nBest Model Saved")

        print(save_path)


# ============================================================
# Training Finished
# ============================================================

print("\n" + "=" * 70)
print("Training Completed")
print("=" * 70)

print(f"Best Validation Accuracy : {best_acc:.2f}%")

print("\nModel Saved At")

print(MODEL_DIR / "fusion_best.pth")


# ============================================================
# Save Last Model
# ============================================================

torch.save(

    model.state_dict(),

    MODEL_DIR / "fusion_last.pth"

)

print("\nLast Model Saved")

print(MODEL_DIR / "fusion_last.pth")


# ============================================================
# Training History
# ============================================================

print("\n" + "=" * 70)
print("Training History")
print("=" * 70)

for i in range(NUM_EPOCHS):

    print(
        f"Epoch {i+1:02d} | "
        f"Train Loss {train_losses[i]:.4f} | "
        f"Train Acc {train_accs[i]:.2f}% | "
        f"Val Loss {val_losses[i]:.4f} | "
        f"Val Acc {val_accs[i]:.2f}%"
    )


print("\n" + "=" * 70)
print("FusionNet Training Finished Successfully")
print("=" * 70)