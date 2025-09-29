import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, m=8, r=128, D=64):
        super().__init__()
        input_dim = m * r
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, D)
        )
        self.decoder = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Unflatten(1, (m, r))
        )

    def forward(self, x):
        e = self.encoder(x)
        out = self.decoder(e)
        return out, e
