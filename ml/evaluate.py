from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from ml.datasets.image_dataset import ImageDataset
from ml.models.cnn_model import CNNModel
from ml.config import (
    TEST_CSV,
    IMAGE_DIR,
    MODEL_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS
)

# ======================================================
# LOAD TEST DATASET
# ======================================================

test_dataset = ImageDataset(
    csv_file=TEST_CSV,
    image_root=IMAGE_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("=" * 60)
print("Test Images :", len(test_dataset))
print("=" * 60)

# ======================================================
# LOAD MODEL
# ======================================================

model = CNNModel()

model.load_state_dict(
    torch.load(
        MODEL_DIR / "best_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Best Model Loaded Successfully")

# ======================================================
# EVALUATION
# ======================================================

true_labels = []
pred_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.numpy())

        pred_labels.extend(preds.cpu().numpy())

# ======================================================
# METRICS
# ======================================================

accuracy = accuracy_score(true_labels, pred_labels)

precision = precision_score(
    true_labels,
    pred_labels
)

recall = recall_score(
    true_labels,
    pred_labels
)

f1 = f1_score(
    true_labels,
    pred_labels
)

cm = confusion_matrix(
    true_labels,
    pred_labels
)

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        pred_labels
    )
)

print("=" * 60)