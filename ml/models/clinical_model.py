import torch
import torch.nn as nn


class ClinicalModel(nn.Module):

    def __init__(self, input_dim=5, num_classes=2):
        super().__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.30)

        # Layer 2
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.30)

        # Feature Layer
        self.fc3 = nn.Linear(32, 16)

        # Classifier
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x, return_features=False):

        # Layer 1
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.drop1(x)

        # Layer 2
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.drop2(x)

        # Feature Layer (16 Features)
        features = self.fc3(x)
        features = torch.relu(features)

        # Classification
        output = self.fc4(features)

        if return_features:
            return features

        return output


if __name__ == "__main__":

    model = ClinicalModel()

    x = torch.randn(8, 5)

    y = model(x)

    features = model(x, return_features=True)

    print("=" * 60)
    print("Clinical Model Loaded Successfully")
    print("=" * 60)
    print("Input Shape    :", x.shape)
    print("Feature Shape  :", features.shape)
    print("Output Shape   :", y.shape)