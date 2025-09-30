import matplotlib.pyplot as plt
import numpy as np


def plot_signals(signals, t, masks=None, max_plots=4, title="Generated Signals"):
    n = signals.shape[0]
    num_plots = min(n, max_plots)

    fig, axes = plt.subplots(
        num_plots, 1, figsize=(12, 2*num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        sig = signals[i]
        if masks is not None:
            sig = np.where(masks[i], sig, np.nan)
        axes[i].plot(t, sig, label=f"Signal {i}")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_reconstruction(x_true, x_recon, max_plots=4, title="Reconstruction"):
    """
    Plot reconstruction vs input (works in feature/DWT space).

    Args:
        x_true:  (m, r) input matrix
        x_recon: (m, r) reconstructed matrix
    """
    m = x_true.shape[0]
    num_plots = min(m, max_plots)

    fig, axes = plt.subplots(
        num_plots, 1, figsize=(12, 2*num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        axes[i].plot(x_true[i], label="Input", alpha=0.7)
        axes[i].plot(x_recon[i], label="Recon", alpha=0.7)
        axes[i].legend(loc="upper right")
        axes[i].grid(True)

    axes[-1].set_xlabel("Feature index")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_split(signal, timing, recon, n=1, title_prefix="Reconstruction"):
    """
    Plot signal and timing channels separately with stretched reconstructions.

    signal: (n, T) numpy array (interpolated original signals)
    timing: (n, T) numpy array (interpolated timing signals)
    recon: (2n, 1, r) numpy array (reconstructed [signal, timing] stacked)
    n: number of signals
    """

    T = signal.shape[1]   # original length (e.g. 1000)
    r = recon.shape[-1]   # recon length (e.g. 512)

    fig, axes = plt.subplots(2, n, figsize=(5*n, 6), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # --- Signal ---
        axes[0, i].plot(signal[i], label="Input", alpha=0.7)

        recon_sig = recon[2*i, 0]  # (r,)
        recon_sig_resampled = np.interp(
            np.linspace(0, r-1, T),  # target indices
            np.arange(r),            # source indices
            recon_sig
        )
        axes[0, i].plot(recon_sig_resampled, label="Recon", alpha=0.7)
        axes[0, i].set_title(f"Signal {i}")
        axes[0, i].legend()

        # --- Timing ---
        axes[1, i].plot(timing[i], label="Input", alpha=0.7)

        recon_tim = recon[2*i+1, 0]  # (r,)
        recon_tim_resampled = np.interp(
            np.linspace(0, r-1, T),
            np.arange(r),
            recon_tim
        )
        axes[1, i].plot(recon_tim_resampled, label="Recon", alpha=0.7)
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_title(f"Timing {i}")
        axes[1, i].legend()

    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()
