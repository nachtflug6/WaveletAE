import torch
from torch.utils.data import Dataset


import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Dataset wrapper for raw windowed signals.
    Expects tuples: (signals, masks, times)
    - signals: (n, T)
    - masks:   (n, T)
    - times:   (T,)
    """

    def __init__(self, win_signals, win_masks, win_times):
        self.win_signals = win_signals
        self.win_masks = win_masks
        self.win_times = win_times

    def __len__(self):
        return self.win_signals.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.win_signals[idx], dtype=torch.float32),
            torch.tensor(self.win_masks[idx], dtype=torch.float32),
            torch.tensor(self.win_times[idx], dtype=torch.float32),
        )
