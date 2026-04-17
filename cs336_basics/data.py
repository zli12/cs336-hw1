from __future__ import annotations

from typing import Any

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
