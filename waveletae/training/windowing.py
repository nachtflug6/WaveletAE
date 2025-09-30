import numpy as np


def window_signals(signals, masks, t, window_size, step_size):
    """
    Partition signals into overlapping windows.

    Args:
        signals: (n, T) array
        masks:   (n, T) array
        t:       (T,) time vector
        window_size (float): window length in seconds
        step_size   (float): step between windows in seconds

    Returns:
        win_signals: (num_windows, n, W)
        win_masks:   (num_windows, n, W)
        win_times:   (num_windows, W) absolute times per window
    """
    dt = t[1] - t[0]
    W = int(window_size / dt)
    step = int(step_size / dt)
    T = signals.shape[1]

    windows = []
    window_masks = []
    window_times = []

    for start in range(0, T - W + 1, step):
        end = start + W
        windows.append(signals[:, start:end])
        window_masks.append(masks[:, start:end])
        window_times.append(t[start:end])

    return np.array(windows), np.array(window_masks), np.array(window_times)
