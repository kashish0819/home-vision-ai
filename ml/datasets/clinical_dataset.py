from pathlib import Path

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class ClinicalDataset(Dataset):

    def __init__(self, csv_file, train=True):

        df = pd.read_csv(csv_file)

        X = df[[
            "Gender",
            "Hemoglobin",
            "MCH",
            "MCHC",
            "MCV"
        ]]

        y = df["Result"]

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        if train:

            self.features = torch.tensor(
                X_train,
                dtype=torch.float32
            )

            self.labels = torch.tensor(
                y_train.values,
                dtype=torch.long
            )

        else:

            self.features = torch.tensor(
                X_test,
                dtype=torch.float32
            )

            self.labels = torch.tensor(
                y_test.values,
                dtype=torch.long
            )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return self.features[idx], self.labels[idx]