import torch
import torch.nn as nn

from ml.models.cnn_model import CNNModel
from ml.models.clinical_model import ClinicalModel
from ml.models.geo_model import GeoModel


class FusionModel(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # -----------------------------
        # Feature Extractors
        # -----------------------------
        self.image_model = CNNModel(pretrained=False)
        self.clinical_model = ClinicalModel()
        self.geo_model = GeoModel()

        # -----------------------------
        # Fusion Classifier
        # 128 + 16 + 8 = 152 features
        # -----------------------------
        self.fusion = nn.Sequential(

            nn.Linear(152, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.40),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, num_classes)

        )

    def forward(self, image, clinical, geo):

        # -----------------------------
        # Extract Features
        # -----------------------------
        image_features = self.image_model(
            image,
            return_features=True
        )

        clinical_features = self.clinical_model(
            clinical,
            return_features=True
        )

        geo_features = self.geo_model(geo)

        # -----------------------------
        # Feature Fusion
        # -----------------------------
        fused_features = torch.cat(

            [
                image_features,
                clinical_features,
                geo_features
            ],

            dim=1

        )

        # -----------------------------
        # Final Prediction
        # -----------------------------
        output = self.fusion(fused_features)

        return output


if __name__ == "__main__":

    model = FusionModel()

    image = torch.randn(8, 3, 224, 224)
    clinical = torch.randn(8, 5)
    geo = torch.randn(8, 1)

    output = model(image, clinical, geo)

    print("=" * 60)
    print("Fusion Model Loaded Successfully")
    print("=" * 60)
    print("Image Input      :", image.shape)
    print("Clinical Input   :", clinical.shape)
    print("Geo Input        :", geo.shape)
    print("Output Shape     :", output.shape)