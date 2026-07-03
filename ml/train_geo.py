import torch

from ml.models.geo_model import GeoModel

print("=" * 60)
print("Geo Feature Model")
print("=" * 60)

model = GeoModel()

dummy = torch.rand(8, 1)

output = model(dummy)

print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)

print("\nGeo Model Loaded Successfully")