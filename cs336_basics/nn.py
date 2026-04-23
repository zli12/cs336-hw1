from __future__ import annotations

import torch
from cs336_basics.decoding import decode, sample_next_token
from cs336_basics.optim import AdamW, SGD, gradient_clipping, lr_cosine_schedule
from cs336_basics.transformer import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    scaled_dot_product_attention,
)

__all__ = [
    "softmax",
    "cross_entropy",
    "sample_next_token",
    "decode",
    "SGD",
    "AdamW",
    "gradient_clipping",
    "lr_cosine_schedule",
    "scaled_dot_product_attention",
    "Linear",
    "Embedding",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "CausalMultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerLM",
]


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply a numerically stable softmax over the specified dimension."""
    # Preserve caller-visible dtype (e.g., float16/bfloat16) after stable computation.
    in_dtype = in_features.dtype
    # Do exponentiation math in float32 to reduce overflow/underflow risk.
    x = in_features.to(torch.float32)
    # Shift each slice so its max is 0; softmax is invariant to constant shifts.
    x = x - torch.amax(x, dim=dim, keepdim=True)
    # Numerator of softmax.
    exp_x = torch.exp(x)
    # Normalize along the requested axis so probabilities sum to 1.
    probs = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return probs.to(in_dtype)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean cross-entropy over arbitrary leading batch dimensions."""
    # Compute logits math in float32 for better numerical stability.
    logits = inputs.to(torch.float32)
    # Shift by per-example max logit to avoid overflow in exp.
    shifted_logits = logits - torch.amax(logits, dim=-1, keepdim=True)
    # log(sum(exp(logits))) with stabilized shifted logits.
    log_normalizer = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    # Pick the target class logit for each example.
    target_logits = torch.gather(shifted_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # -log softmax(target) = log_normalizer - target_logit.
    return (log_normalizer - target_logits).mean()