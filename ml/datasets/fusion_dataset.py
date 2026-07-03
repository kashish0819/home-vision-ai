from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image, ImageFile, UnidentifiedImageError

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FusionDataset(Dataset):

    def __init__(self, csv_file, image_root):

        self.data = pd.read_csv(csv_file)

        self.image_root = Path(image_root)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = self.image_root / row["image_path"]

        # ----------------------------------------
        # Safe Image Loading
        # ----------------------------------------
        try:

            image = Image.open(image_path)

            image = image.convert("RGB")

            image = self.transform(image)

        except (UnidentifiedImageError, OSError, ValueError, EOFError) as e:

            print(f"\nSkipping bad image: {image_path}")
            print(e)

            return self.__getitem__((idx + 1) % len(self.data))

        # ----------------------------------------
        # Clinical Features
        # ----------------------------------------
        clinical = torch.tensor([
            float(row["Gender"]),
            float(row["Hemoglobin"]),
            float(row["MCH"]),
            float(row["MCHC"]),
            float(row["MCV"])
        ], dtype=torch.float32)

        # ----------------------------------------
        # Geo Feature
        # ----------------------------------------
        geo = torch.tensor(
            [float(row["Geo"])],
            dtype=torch.float32
        )

        # ----------------------------------------
        # Label
        # ----------------------------------------
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return image, clinical, geo, label


if __name__ == "__main__":

    from ml.config import IMAGE_DIR

    dataset = FusionDataset(
        csv_file="data/processed/fusion_data.csv",
        image_root=IMAGE_DIR
    )

    image, clinical, geo, label = dataset[0]

    print("=" * 60)
    print("Fusion Dataset Loaded Successfully")
    print("=" * 60)
    print("Image Shape    :", image.shape)
    print("Clinical Shape :", clinical.shape)
    print("Geo Shape      :", geo.shape)
    print("Label          :", label)