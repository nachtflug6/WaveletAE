import numpy as np
import torch
from torch.utils.data import DataLoader

from waveletae.preprocessing.interpolation import linear_interpolate
from waveletae.preprocessing.timing import timing_signal
from waveletae.preprocessing.wavelet import apply_dwt
from waveletae.models.autoencoder import Autoencoder
from waveletae.training.dataset import ToyDataset
from waveletae.training.trainer import train
from waveletae.utils.plotting import plot_reconstruction
from waveletae.utils.config import load_config


def generate_toy_data(T=600, n=4, sparsity=0.5, r=128, timing_mode="gaussian"):
    signals = []
    masks = []
    for i in range(n):
        t = np.arange(T)
        freq = np.random.uniform(0.01, 0.05)
        phase = np.random.uniform(0, np.pi)
        sig = np.sin(2*np.pi*freq*t + phase) + 0.1*np.random.randn(T)

        mask = np.random.rand(T) > sparsity
        sig[~mask] = np.nan

        signals.append(sig)
        masks.append(mask)

    signals = np.array(signals)
    masks = np.array(masks)

    resampled_vals = []
    resampled_time = []
    for i in range(n):
        vals = linear_interpolate(signals[i], masks[i], r)
        times = timing_signal(masks[i], mode=timing_mode, r=r)
        resampled_vals.append(vals)
        resampled_time.append(times)

    X = np.stack(resampled_vals + resampled_time, axis=0)
    return apply_dwt(X)


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")

    X = generate_toy_data(**cfg["data"])
    dataset = ToyDataset(X[None, ...])  # batch=1 for now
    dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"])

    model = Autoencoder(**cfg["model"])
    train(model, dataloader, **cfg["train"])

    # Quick reconstruction
    with torch.no_grad():
        recon, _ = model(dataset[0].unsqueeze(0))
    plot_reconstruction(dataset[0].numpy(), recon.squeeze(0).numpy())
