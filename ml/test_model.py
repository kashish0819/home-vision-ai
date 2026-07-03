import torch

from ml.models.cnn_model import CNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel()

model = model.to(device)

dummy = torch.randn(8, 3, 224, 224).to(device)

output = model(dummy)

print("=" * 50)
print("Model Loaded Successfully")
print("=" * 50)

print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)

print(model)