import torch


def linear_interpolate(signal, mask, r, device=None):
    """
    Linearly interpolate irregular signal onto a regular grid of length r.
    signal: (T,)
    mask:   (T,) boolean
    Returns: (r,)
    """
    if device is None:
        device = signal.device

    T = signal.shape[0]
    mask = mask.to(torch.bool)

    t_obs = torch.arange(T, device=device)[mask]        # observed time indices
    x_obs = signal[mask]                                # observed values

    if t_obs.numel() > 1:
        new_t = torch.linspace(0, T - 1, r, device=device)

        # find bins
        idx = torch.bucketize(new_t, t_obs)

        idx1 = torch.clamp(idx - 1, 0, t_obs.numel() - 1)
        idx2 = torch.clamp(idx,     0, t_obs.numel() - 1)

        t1, t2 = t_obs[idx1], t_obs[idx2]
        x1, x2 = x_obs[idx1], x_obs[idx2]

        # linear interpolation
        denom = (t2 - t1)
        denom[denom == 0] = 1.0  # avoid div/0
        w = (new_t - t1) / denom
        x_new = x1 + w * (x2 - x1)

        return x_new
    else:
        return torch.zeros(r, device=device)
