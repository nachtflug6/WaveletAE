import torch
import torch.nn.functional as F


def apply_dwt(x):
    """
    Single-level Haar DWT via conv1d.
    Input: (batch, 1, r) tensor
    Output: cA, cD (low-pass, high-pass)
    """
    # Haar filters
    h0 = torch.tensor([1/torch.sqrt(torch.tensor(2.0)), 1 /
                      torch.sqrt(torch.tensor(2.0))], dtype=x.dtype, device=x.device)
    h1 = torch.tensor([-1/torch.sqrt(torch.tensor(2.0)), 1 /
                      torch.sqrt(torch.tensor(2.0))], dtype=x.dtype, device=x.device)

    h0 = h0.view(1, 1, -1)
    h1 = h1.view(1, 1, -1)

    # Low-pass and high-pass conv
    cA = F.conv1d(x, h0, stride=2)
    cD = F.conv1d(x, h1, stride=2)
    return cA, cD


def apply_idwt(cA, cD):
    """
    Inverse Haar DWT via transposed conv1d.
    Input: cA, cD (batch, 1, r//2)
    Output: (batch, 1, r)
    """
    # Inverse filters
    g0 = torch.tensor([1/torch.sqrt(torch.tensor(2.0)), 1 /
                      torch.sqrt(torch.tensor(2.0))], dtype=cA.dtype, device=cA.device)
    g1 = torch.tensor([1/torch.sqrt(torch.tensor(2.0)), -1 /
                      torch.sqrt(torch.tensor(2.0))], dtype=cA.dtype, device=cA.device)

    g0 = g0.view(1, 1, -1)
    g1 = g1.view(1, 1, -1)

    # Transposed conv (upsampling)
    recA = F.conv_transpose1d(cA, g0, stride=2)
    recD = F.conv_transpose1d(cD, g1, stride=2)

    return recA + recD


def apply_idwt_concat(x_cat):
    """
    Inverse Haar DWT when cA and cD are concatenated along last dim.
    Input: (B, 1, r) where r = cA_len + cD_len
    Output: (B, 1, 2*len(cA))
    """
    half = x_cat.shape[-1] // 2
    cA, cD = x_cat[..., :half], x_cat[..., half:]
    return apply_idwt(cA.unsqueeze(1), cD.unsqueeze(1))
