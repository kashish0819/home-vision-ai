from pathlib import Path

from torch.utils.data import DataLoader

from ml.datasets.image_dataset import ImageDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

dataset = ImageDataset(

    csv_file=PROJECT_ROOT/"data"/"processed"/"train.csv",

    image_root=PROJECT_ROOT/"data"/"eye_image"

)

loader = DataLoader(

    dataset,

    batch_size=16,

    shuffle=True

)

images, labels = next(iter(loader))

print("="*50)

print("Dataset Loaded Successfully")

print("="*50)

print(images.shape)

print(labels.shape)

print(labels)