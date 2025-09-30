import torch
import torch.nn as nn
from waveletae.preprocessing.windowproc import forward_preprocess, backward_preprocess


def train(
    model, dataloader, epochs=50, lr=1e-3, device="cpu",
    r=128, timing_mode="gaussian", channel_config=None
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for signals, masks, times in dataloader:
            # Ensure tensors are on correct device
            signals = signals.to(device)
            masks = masks.to(device)

            # Forward preprocess (torch only)
            X = forward_preprocess(
                signals, masks,
                r=r, timing_mode=timing_mode,
                channel_config=channel_config
            )  # (B, 2n, r)

            optimizer.zero_grad()
            recon, _ = model(X)

            # Backward preprocess (torch only)
            recon_back = backward_preprocess(
                recon, channel_config=channel_config
            )
            target_back = backward_preprocess(
                X, channel_config=channel_config
            )

            # Loss in interpolated (time-like) space
            loss = criterion(recon_back, target_back)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {total_loss/len(dataloader):.6f}")
