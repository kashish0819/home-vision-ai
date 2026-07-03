from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

    print(f"\n✅ Checkpoint saved -> {path}")


def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint.
    """

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    print(f"\n✅ Checkpoint loaded from {path}")

    return model, optimizer, epoch


def plot_history(history, save_dir):
    """
    Plot training history.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loss Plot
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_dir / "loss_curve.png")

    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_dir / "accuracy_curve.png")

    plt.close()


def save_confusion_matrix(y_true, y_pred, save_path):
    """
    Save confusion matrix image.
    """

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(ax=ax)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()