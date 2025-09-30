import torch
import torch.nn.functional as F


def timing_signal(mask, mode="gaussian", sigma=3, size=5, r=128):
    """
    Encode timing information as a smoothed signal in PyTorch.
    mask: (T,) boolean tensor for observed points
    Returns: (r,)
    """
    T = mask.shape[0]
    timing = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,T)

    if mode == "gaussian":
        # Gaussian kernel
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1,
                         dtype=torch.float32, device=mask.device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)
        timing = F.conv1d(timing, kernel, padding=radius)
    elif mode == "boxcar":
        kernel = torch.ones(1, 1, size, device=mask.device) / size
        timing = F.conv1d(timing, kernel, padding=size // 2)

    timing = timing.squeeze()  # (T,)

    # interpolate to length r
    old_t = torch.arange(T, device=mask.device, dtype=torch.float32)
    new_t = torch.linspace(0, T - 1, r, device=mask.device)

    # manual 1D linear interpolation
    idx = torch.bucketize(new_t, old_t)
    idx1 = torch.clamp(idx - 1, 0, T - 1)
    idx2 = torch.clamp(idx,     0, T - 1)

    t1, t2 = old_t[idx1], old_t[idx2]
    y1, y2 = timing[idx1], timing[idx2]

    denom = (t2 - t1)
    denom[denom == 0] = 1.0
    w = (new_t - t1) / denom
    timing_resampled = y1 + w * (y2 - y1)

    return timing_resampled
