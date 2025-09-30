from torch.utils.data import DataLoader, Subset
from waveletae.models.autoencoder import Autoencoder
from waveletae.training.trainer import train
from waveletae.training.dataset import WindowDataset
from waveletae.utils.plotting import plot_signals, plot_reconstruction
import torch
from waveletae.preprocessing.windowproc import forward_preprocess, backward_preprocess
from waveletae.utils.plotting import plot_reconstruction_split


def run_experiment_windows(
    win_signals, win_masks, win_times,
    r=128, D=32, epochs=20, batch_size=1,
    timing_mode="gaussian",
    train_frac=0.8
):
    print(f"Running AE on {win_signals.shape[0]} windows, r={r}, D={D}")

    dataset = WindowDataset(win_signals, win_masks, win_times)

    num_windows = len(dataset)
    train_size = int(train_frac * num_windows)

    train_set = Subset(dataset, range(0, train_size))
    test_set = Subset(dataset, range(train_size, num_windows))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    n = win_signals.shape[1]   # original number of signals
    m = n * 2                  # signals + timing
    model = Autoencoder(m=m, r=r, D=D)

    # Train with forward/backward preprocessing
    train(model, train_loader,
          epochs=epochs, lr=1e-3, device="cpu",
          r=r, timing_mode=timing_mode)

    # Plot first raw window for sanity check
    plot_signals(win_signals[0], win_times[0], masks=win_masks[0])

    # -------------------------
    # Test / visualization
    # -------------------------
    model.eval()
    with torch.no_grad():
        for signals, masks, times in test_loader:
            # preprocess forward
            X = forward_preprocess(
                signals, masks, r=r, timing_mode=timing_mode
            )  # (B, 2n, r)
            X = X.to(torch.float32)

            # forward through AE
            recon, emb = model(X)

            # inverse preprocess
            recon_back = backward_preprocess(recon)

            plot_reconstruction_split(
                signals[0].cpu().numpy(),
                times[0].cpu().numpy(),
                recon_back[0].cpu().numpy(),
                n=win_signals.shape[1],
                title_prefix="Test Window Reconstruction"
            )

    return model, dataset
