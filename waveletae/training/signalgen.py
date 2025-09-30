import numpy as np


def generate_multisine_signals(
    duration=60.0,
    n=4,
    freqs=None,
    max_components=3,
    dt=0.01,
    sparsity=0.0,
    seed=None,
    noise_std=0.05
):
    """
    Generate full-length multivariate multi-sine signals with optional missing values.

    Args:
        duration (float): total time span in seconds
        n (int): number of signals (variates)
        freqs (list or None): list of frequencies in Hz. If None, random selection.
        max_components (int): max sine components per variate if freqs=None
        dt (float): sampling interval in seconds
        sparsity (float): fraction of missing points
        seed (int or None): random seed
        noise_std (float): std of Gaussian noise

    Returns:
        signals: (n, T) array with NaNs at missing points
        masks:   (n, T) boolean mask
        t:       (T,) time vector in seconds
    """
    rng = np.random.default_rng(seed)
    T = int(duration / dt)
    t = np.arange(T) * dt

    signals = []
    masks = []

    for i in range(n):
        signal = np.zeros(T)

        if freqs is not None:
            # deterministic set of frequencies
            for f in freqs:
                amp = rng.uniform(0.5, 1.0)
                phase = rng.uniform(0, 2*np.pi)
                signal += amp * np.sin(2*np.pi*f*t + phase)
        else:
            # random multi-sine
            num_components = rng.integers(1, max_components + 1)
            for _ in range(num_components):
                f = rng.uniform(0.001, 0.2)  # Hz
                amp = rng.uniform(0.5, 1.0)
                phase = rng.uniform(0, 2*np.pi)
                signal += amp * np.sin(2*np.pi*f*t + phase)

        signal += noise_std * rng.standard_normal(T)

        # missing mask
        mask = rng.random(T) > sparsity
        signal[~mask] = np.nan

        signals.append(signal)
        masks.append(mask)

    return np.array(signals), np.array(masks), t
