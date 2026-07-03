import torch
import torch.nn as nn


class GeoModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(1, 8),
            nn.ReLU(),

            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU()

        )

    def forward(self, x):
        return self.network(x)