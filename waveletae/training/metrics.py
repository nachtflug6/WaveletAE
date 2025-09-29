import torch


def reconstruction_error(X, X_hat):
    return torch.mean((X - X_hat)**2).item()
