from __future__ import annotations

import os
from typing import IO, Any, BinaryIO

import numpy as np
import torch


def get_batch(
    dataset: Any,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of contiguous token windows and next-token targets."""
    dataset_array = np.asarray(dataset)
    if dataset_array.ndim != 1:
        raise ValueError("dataset must be a 1D array of token ids")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_possible_starts = dataset_array.shape[0] - context_length
    if num_possible_starts <= 0:
        raise ValueError("dataset must be longer than context_length")

    starts = np.random.randint(0, num_possible_starts, size=(batch_size,))
    offsets = np.arange(context_length)
    positions = starts[:, None] + offsets[None, :]

    x = torch.tensor(dataset_array[positions], dtype=torch.long, device=device)
    y = torch.tensor(dataset_array[positions + 1], dtype=torch.long, device=device)
    return x, y


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
