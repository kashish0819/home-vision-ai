import torch

from ml.models.geo_model import GeoModel

print("=" * 60)
print("Geo Model Evaluation")
print("=" * 60)

model = GeoModel()

dummy = torch.rand(4, 1)

output = model(dummy)

print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)

print("\nGeo Model Working Successfully")