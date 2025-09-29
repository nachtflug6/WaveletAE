import torch
import torch.nn as nn


def train(model, dataloader, epochs=50, lr=1e-3, batch_size=5, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for X in dataloader:
            X = X.to(device)
            optimizer.zero_grad()
            recon, _ = model(X)
            loss = criterion(recon, X)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {total_loss/len(dataloader):.6f}")
