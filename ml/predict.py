from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from ml.models.cnn_model import CNNModel
from ml.models.clinical_model import ClinicalModel

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_model.pth"
CLINICAL_MODEL_PATH = PROJECT_ROOT / "saved_models" / "clinical_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD MODELS
# =====================================================

image_model = CNNModel().to(DEVICE)
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
image_model.eval()

clinical_model = ClinicalModel().to(DEVICE)
clinical_model.load_state_dict(torch.load(CLINICAL_MODEL_PATH, map_location=DEVICE))
clinical_model.eval()

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================================================
# IMAGE PREDICTION
# =====================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = image_model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)

    return {
        "prediction": int(prediction.item()),
        "confidence": float(confidence.item())
    }

# =====================================================
# CLINICAL PREDICTION
# =====================================================

def predict_clinical(gender, hb, mch, mchc, mcv):

    features = np.array([[gender, hb, mch, mchc, mcv]], dtype=np.float32)

    features = torch.tensor(features).to(DEVICE)

    with torch.no_grad():
        outputs = clinical_model(features)
        probs = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)

    return {
        "prediction": int(prediction.item()),
        "confidence": float(confidence.item())
    }

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Prediction Module Loaded Successfully")
    print("=" * 60)

    print("Models Loaded Successfully")