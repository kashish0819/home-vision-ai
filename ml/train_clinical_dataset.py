from torch.utils.data import DataLoader

from ml.datasets.clinical_dataset import ClinicalDataset

dataset = ClinicalDataset(
    "data/clinical/anemia.csv",
    train=True
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

x, y = next(iter(loader))

print("="*50)
print("Clinical Dataset Loaded")
print("="*50)

print(x.shape)
print(y.shape)
print(y)