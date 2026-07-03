from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ml.datasets.clinical_dataset import ClinicalDataset
from ml.models.clinical_model import ClinicalModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "clinical" / "anemia.csv"

MODEL_PATH = PROJECT_ROOT / "saved_models" / "clinical_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Clinical Model Evaluation")
print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)

# Dataset
test_dataset = ClinicalDataset(
    csv_file=DATA_FILE,
    train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Test Samples :", len(test_dataset))

# Model
model = ClinicalModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Best Clinical Model Loaded Successfully")

predictions = []
targets = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(DEVICE)

        outputs = model(features)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.numpy())

accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions)
recall = recall_score(targets, predictions)
f1 = f1_score(targets, predictions)

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(targets, predictions))

print("\nClassification Report\n")
print(classification_report(targets, predictions))