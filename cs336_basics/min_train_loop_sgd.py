from __future__ import annotations

import torch
from cs336_basics.optim import SGD


def run_toy_training(
    lr: float = 1.0,
    steps: int = 100,
    seed: int = 0,
    device: str = "cpu",
) -> list[float]:
    """Run a minimal training loop on loss = mean(weights^2)."""
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10), device=device))
    optimizer = SGD([weights], lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        optimizer.step()

    return losses


if __name__ == "__main__":
    for learning_rate in (1e1, 1e2, 1e3):
        losses = run_toy_training(lr=learning_rate, steps=10, seed=0)
        print(f"lr={learning_rate:.0e}: {losses}")
