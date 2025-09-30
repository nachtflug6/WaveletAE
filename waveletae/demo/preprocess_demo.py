import numpy as np
import matplotlib.pyplot as plt
import pywt

from waveletae.preprocessing.interpolation import linear_interpolate
from waveletae.preprocessing.timing import timing_signal


def run_demo(T=100, r=64, freq=3, sampling_prob=0.6):
    # 1. Generate sine with random sampling
    t = np.linspace(0, 2*np.pi, T)
    signal = np.sin(freq * t)
    mask = np.random.rand(T) < sampling_prob
    observed = np.where(mask, signal, np.nan)

    # 2. Interpolate onto new uniform grid
    t_resamp = np.linspace(t.min(), t.max(), r)
    interp = linear_interpolate(signal, mask, r)

    # 3. Timing signal
    timing = timing_signal(mask, mode="gaussian", r=r)

    # 4. Forward + inverse DWT
    coeffs = pywt.wavedec(interp, "db4", level=None)
    recon = pywt.waverec(coeffs, "db4")
    recon = recon[:len(interp)]

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(t, signal, label="True sine")
    axes[0].plot(t, observed, "o", label="Observed")
    axes[0].set_title("Original with random sampling")
    axes[0].legend()

    axes[1].plot(t_resamp, interp, "r.-", label="Interpolated")
    axes[1].vlines(t_resamp, interp.min(), interp.max(),
                   color="gray", alpha=0.2, lw=0.5)
    axes[1].set_title("Interpolation on uniform grid")
    axes[1].legend()

    axes[2].plot(t_resamp, timing, "g.-", label="Timing (Gaussian)")
    axes[2].set_title("Timing signal")
    axes[2].legend()

    axes[3].plot(t_resamp, interp, "b.-", label="Interpolated")
    axes[3].plot(t_resamp, recon, "r--", label="DWT reconstruction")
    axes[3].set_title("Wavelet forward+inverse reconstruction")
    axes[3].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
