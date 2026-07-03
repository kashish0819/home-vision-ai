from PIL import Image
import torch
import torchvision.transforms as transforms

from ml.config import DEVICE, MODEL_DIR
from ml.models.fusion_model import FusionModel


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = FusionModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "fusion_best.pth",
        map_location=DEVICE
    )
)

model.eval()

# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_fusion(
    image_file,
    gender,
    hemoglobin,
    mch,
    mchc,
    mcv,
    geo
):
    
   

    print("="*60)
    print("predict_fusion() CALLED")
    print(image_file)
    print(gender)
    print(hemoglobin)
    print(mch)
    print(mchc)
    print(mcv)
    print(geo)
    print("="*60)

    # uploaded file from Streamlit
    image = Image.open(image_file).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    clinical = torch.tensor(
        [[
            gender,
            hemoglobin,
            mch,
            mchc,
            mcv
        ]],
        dtype=torch.float32
    ).to(DEVICE)

    geo_tensor = torch.tensor(
        [[geo]],
        dtype=torch.float32
    ).to(DEVICE)

    print("Image Loaded Successfully")
    print("Clinical Tensor Created")

    with torch.no_grad():

        output = model(
            image,
            clinical,
            geo_tensor
        )

        probs = torch.softmax(output, dim=1)

        prediction = torch.argmax(probs, dim=1).item()

        confidence = probs.max().item()

    label = "Anemia" if prediction == 1 else "Normal"

    print("Prediction =", prediction)
    print("Confidence =", confidence)
    print("Label =", label)

    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence
    }