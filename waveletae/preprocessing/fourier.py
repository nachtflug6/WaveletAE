import torch


def apply_dft(x):
    """
    Apply 1D DFT row-wise (real FFT) in torch.
    Input:  (m, r) tensor
    Output: (m, r//2 + 1) complex tensor
    """
    if x.ndim == 1:
        return torch.fft.rfft(x)
    else:
        return torch.stack([torch.fft.rfft(row) for row in x], dim=0)


def apply_idft(X, r=None):
    """
    Inverse of apply_dft_torch.
    """
    if X.ndim == 1:
        return torch.fft.irfft(X, n=r)
    else:
        return torch.stack([torch.fft.irfft(row, n=r) for row in X], dim=0)
