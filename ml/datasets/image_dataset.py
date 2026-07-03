from pathlib import Path

import cv2
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(self, csv_file, image_root, transform=None):

        self.data = pd.read_csv(csv_file)

        self.image_root = Path(image_root)

        if transform is None:

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = self.image_root / row["image_path"]

        # -------------------------------
        # Read image using OpenCV
        # -------------------------------
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        image = self.transform(image)

        label = int(row["label"])

        return image, label