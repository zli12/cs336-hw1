from __future__ import annotations

import torch
import torch.nn.functional as F
from cs336_basics.decoding import decode, sample_next_token
from cs336_basics.optim import AdamW, SGD, gradient_clipping, lr_cosine_schedule
from cs336_basics.transformer import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SiLUFeedForward,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    scaled_dot_product_attention,
    sdpa_attention,
)

__all__ = [
    "softmax",
    "cross_entropy",
    "cross_entropy_with_z_loss",
    "sample_next_token",
    "decode",
    "SGD",
    "AdamW",
    "gradient_clipping",
    "lr_cosine_schedule",
    "scaled_dot_product_attention",
    "sdpa_attention",
    "Linear",
    "Embedding",
    "RMSNorm",
    "SwiGLU",
    "SiLUFeedForward",
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
    """Compute mean cross-entropy over arbitrary leading batch dimensions.

    Delegates to ``F.cross_entropy``, which internally computes a stabilized
    ``log_softmax`` in fp32 *without* materializing a full fp32 copy of the
    ``(..., vocab)`` logits tensor. That fused path matters at vocab=32k +
    bs=96 + ctx=256 with bf16 autocast: a manual ``inputs.to(torch.float32)``
    upcast would allocate ~3 GiB of intermediate per loss call.
    """
    flat_inputs = inputs.reshape(-1, inputs.size(-1))
    flat_targets = targets.reshape(-1)
    return F.cross_entropy(flat_inputs, flat_targets)


def cross_entropy_with_z_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    z_weight: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy plus a small auxiliary penalty on log(Z).

    Returns ``(total, ce, z)`` where:
    - ``ce`` is the standard cross-entropy (computed via the fused
      ``F.cross_entropy`` path for memory efficiency under bf16 autocast).
    - ``z = mean(log_z**2)`` with ``log_z = logsumexp(logits, dim=-1)``.
    - ``total = ce + z_weight * z``.

    Why use it: the z-loss (PaLM, 2022) penalizes unbounded growth of the partition
    function ``Z = sum(exp(logits))``. With bf16 forward, large logits make
    ``exp(logits)`` saturate; pulling ``log Z`` toward zero keeps softmax in a
    well-conditioned regime without changing the argmax that would have been
    chosen by softmax. Standard weight is ``1e-4``.

    Memory note: previously this function explicitly cast ``inputs`` to fp32
    via ``inputs.to(torch.float32)``, which materialized an extra ~3 GiB
    intermediate at vocab=32k / bs=96 / ctx=256 / bf16. Combined with a
    ``--logit-soft-cap``-applied tanh of the same shape that backward also
    needed alive, that pushed peak memory above the A100-40GB headroom and
    OOM'd. Calling ``F.cross_entropy`` + ``torch.logsumexp`` directly on the
    bf16 input keeps every intermediate in bf16 and lets each kernel reduce
    along the vocab axis in fp32 internally, so peak loss-side memory stays
    in the ~1.5 GiB range instead of ~6 GiB.
    """
    flat_inputs = inputs.reshape(-1, inputs.size(-1))
    flat_targets = targets.reshape(-1)
    ce = F.cross_entropy(flat_inputs, flat_targets)
    # Compute logsumexp on the original-dtype input. PyTorch's logsumexp
    # reduces in fp32 internally without ever allocating a full fp32 copy
    # of the (N, V) logits tensor.
    log_z = torch.logsumexp(flat_inputs, dim=-1)
    z = log_z.float().square().mean()
    total = ce + z_weight * z
    return total, ce, z