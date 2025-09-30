import torch
from waveletae.preprocessing.wavelet import apply_dwt, apply_idwt, apply_idwt_concat
from waveletae.preprocessing.fourier import apply_dft, apply_idft
from waveletae.preprocessing.interpolation import linear_interpolate
from waveletae.preprocessing.timing import timing_signal


def forward_preprocess(signals, masks, r=128, timing_mode="gaussian", channel_config=None):
    """
    Batched forward preprocessing with interleaved [signal, timing] rows.
    signals: (B, n, T)
    masks:   (B, n, T)
    Returns: (B, 2n, r)
    """
    if channel_config is None:
        channel_config = {"signal": "dwt", "timing": "dwt"}

    B, n, T = signals.shape
    device = signals.device

    batch_out = []
    for b in range(B):
        rows = []
        for i in range(n):
            # interpolate individual signal + timing
            vals = linear_interpolate(signals[b, i], masks[b, i], r)  # (r,)
            tim = timing_signal(masks[b, i], r=r, mode=timing_mode)  # (r,)

            if channel_config["signal"] == "dwt":
                cA, cD = apply_dwt(vals.unsqueeze(0).unsqueeze(0))
                vals = torch.cat(
                    [cA, cD], dim=-1).squeeze(0).squeeze(0)  # (r,)
            elif channel_config["signal"] == "dft":
                vals = apply_dft(vals.unsqueeze(0)).squeeze(0)

            if channel_config["timing"] == "dwt":
                cA, cD = apply_dwt(tim.unsqueeze(0).unsqueeze(0))
                tim = torch.cat([cA, cD], dim=-1).squeeze(0).squeeze(0)  # (r,)
            elif channel_config["timing"] == "dft":
                tim = apply_dft(tim.unsqueeze(0)).squeeze(0)

            rows.append(vals)
            rows.append(tim)

        batch_out.append(torch.stack(rows, dim=0))  # (2n, r)

    return torch.stack(batch_out, dim=0)  # (B, 2n, r)


def backward_preprocess(X, channel_config=None):
    """
    Inverse preprocessing: undo wavelet/DFT.
    Accepts (B, 2n, r) or (2n, r).
    Returns same shape but reconstructed to time domain.
    """
    if channel_config is None:
        channel_config = {"signal": "dwt", "timing": "dwt"}

    # Ensure batch dimension
    if X.dim() == 2:  # (m, r)
        X = X.unsqueeze(0)  # -> (1, m, r)
    elif X.dim() != 3:
        raise ValueError(
            f"Unexpected shape {X.shape}, expected (B,2n,r) or (2n,r)")

    B, m, r = X.shape
    out_batches = []

    for b in range(B):
        rows = []
        for i in range(m):
            row = X[b, i]
            if (i % 2 == 0 and channel_config["signal"] == "dwt") or \
                    (i % 2 == 1 and channel_config["timing"] == "dwt"):
                row = apply_idwt_concat(row.unsqueeze(0)).squeeze(1)  # (B, r)

            elif (i % 2 == 0 and channel_config["signal"] == "dft") or \
                 (i % 2 == 1 and channel_config["timing"] == "dft"):
                row = apply_idft(row)
            rows.append(row)
        out_batches.append(torch.stack(rows, dim=0))
    return torch.stack(out_batches, dim=0)  # (B, 2n, r)
