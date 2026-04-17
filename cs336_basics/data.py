from __future__ import annotations

import os
from typing import IO, Any, BinaryIO

import torch


def get_batch(
    dataset: Any,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of contiguous token windows and next-token targets."""
    dataset_tensor = torch.as_tensor(dataset, dtype=torch.long)
    if dataset_tensor.ndim != 1:
        raise ValueError("dataset must be a 1D array of token ids")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_possible_starts = dataset_tensor.shape[0] - context_length
    if num_possible_starts <= 0:
        raise ValueError("dataset must be longer than context_length")

    starts = torch.randint(0, num_possible_starts, size=(batch_size,))
    offsets = torch.arange(context_length)
    positions = starts.unsqueeze(1) + offsets.unsqueeze(0)

    x = dataset_tensor[positions]
    y = dataset_tensor[positions + 1]
    return x.to(device), y.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model/optimizer state and iteration for resuming training."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model/optimizer state and return the saved iteration."""
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])
