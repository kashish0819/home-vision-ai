import pandas as pd
import torch
from torch.utils.data import Dataset


class GeoDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        # Geo Risk Feature
        risk = self.data.iloc[idx]["Risk"]

        # Convert to Tensor
        geo_feature = torch.tensor([risk], dtype=torch.float32)

        return geo_feature


if __name__ == "__main__":

    dataset = GeoDataset("../../data/geo/geo_risk.csv")

    print("=" * 60)
    print("Geo Dataset Loaded Successfully")
    print("=" * 60)

    print("Total Samples :", len(dataset))

    sample = dataset[0]

    print("Sample Shape  :", sample.shape)
    print("Sample Value  :", sample)