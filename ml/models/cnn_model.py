import torch
import torch.nn as nn
import timm


class CNNModel(nn.Module):

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # EfficientNet-B0 Backbone
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        # Feature Extractor
        self.fc1 = nn.Linear(feature_dim, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.30)

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.20)

        # Final Classifier
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):

        # EfficientNet Features
        x = self.backbone(x)

        # Dense Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        # Dense Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        # 128-D Feature Vector
        features = x

        # Final Prediction
        output = self.fc3(features)

        if return_features:
            return features

        return output


if __name__ == "__main__":

    model = CNNModel()

    x = torch.randn(8, 3, 224, 224)

    y = model(x)

    features = model(x, return_features=True)

    print("=" * 60)
    print("CNN Model Loaded Successfully")
    print("=" * 60)
    print("Input Shape    :", x.shape)
    print("Feature Shape  :", features.shape)
    print("Output Shape   :", y.shape)