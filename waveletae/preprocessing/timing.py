import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def timing_signal(mask, mode="gaussian", sigma=3, size=5, r=128):
    """
    Encode timing information as a smoothed signal.
    mask: (T,) boolean array for observed points
    """
    T = len(mask)
    timing = mask.astype(float)

    if mode == "gaussian":
        timing = gaussian_filter1d(timing, sigma=sigma)
    elif mode == "boxcar":
        timing = uniform_filter1d(timing, size=size)

    new_t = np.linspace(0, T-1, r)
    return np.interp(new_t, np.arange(T), timing)
