import numpy as np
from scipy.interpolate import interp1d


def linear_interpolate(signal, mask, r):
    """
    Linearly interpolate irregular signal onto a regular grid of length r.
    signal: (T,) with NaNs at missing points
    mask:   (T,) boolean mask for observed points
    """
    T = len(signal)
    t_obs = np.where(mask)[0]
    x_obs = signal[mask]

    if len(t_obs) > 1:
        f = interp1d(t_obs, x_obs, kind="linear", fill_value="extrapolate")
        new_t = np.linspace(0, T-1, r)
        return f(new_t)
    else:
        return np.zeros(r)
