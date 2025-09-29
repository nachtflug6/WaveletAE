import numpy as np
import pywt


def apply_dwt(matrix, wavelet="haar"):
    """
    Apply single-level DWT row-wise.
    matrix: (m, r)
    """
    coeffs = []
    for row in matrix:
        cA, cD = pywt.dwt(row, wavelet)
        coeffs.append(np.concatenate([cA, cD]))
    return np.stack(coeffs, axis=0)
