from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    """A bias-free linear layer compatible with nn.Linear's core interface."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # Linear weights: N(0, 2/(din+dout)) truncated to [-3σ, 3σ].
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """A simple embedding lookup layer compatible with nn.Embedding's core interface."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        # Embedding: N(0, 1) truncated to [-3, 3].
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids are integer indices into the vocabulary axis.
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    This module normalizes across the last dimension and applies a learned gain.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable gain vector g in the RMSNorm equation.
        # Spec says RMSNorm gains are initialized to 1.
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep input dtype so we can return the same dtype after stable float32 math.
        _in_dtype = x.dtype

        # TODO: upcast to float32 before squaring to improve numerical stability.
        x_fp32 = x.to(torch.float32)        

        # TODO: compute RMS over the final dimension:
        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)

        # TODO: normalize then apply gain:
        out = x_fp32 / rms * self.weight

        return out.to(_in_dtype)

class SwiGLU(nn.Module):
    """Position-wise feed-forward block with SwiGLU gating.

    Computes: W2( SiLU(W1 x) ⊙ (W3 x) )
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            # Canonical LLM choice: inner width is about 8/3 of model width.
            # Round up to a multiple of 64 for hardware-friendly matmul sizes.
            approx = (8 * d_model) / 3
            d_ff = int(math.ceil(approx / 64) * 64)
        self.d_ff = d_ff

        # W1, W3: d_model -> d_ff ; W2: d_ff -> d_model
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First projection path that goes through SiLU nonlinearity.
        x_w1 = self.w1(x)
        # SiLU(z) = z * sigmoid(z)
        silu = x_w1 * torch.sigmoid(x_w1)
        # Second projection path acts as a gate (GLU-style elementwise product).
        gated = silu * self.w3(x)
        # Project back to model width.
        return self.w2(gated)


class RotaryPositionalEmbedding(nn.Module):
    """Applies rotary position embeddings (RoPE) to the last dimension.

    Input shape: (..., seq_len, d_k)
    Positions shape: (..., seq_len)
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # RoPE rotates pairs of features: (0,1), (2,3), ..., (d_k-2, d_k-1).
        # So we need one frequency per pair, i.e. d_k/2 frequencies total.
        pair_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        # inv_freq[k] = 1 / theta^(2k/d_k), where pair_idx already stores 2k.
        # Larger k -> smaller frequency -> slower rotation (captures longer-range structure).
        inv_freq = 1.0 / (theta ** (pair_idx / d_k))

        # Precompute all angles for all positions once:
        # angles[pos, k] = pos * inv_freq[k].
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)

        # Cache cos/sin lookup tables as buffers:
        # - not learnable (unlike nn.Parameter)
        # - moved with .to(device)
        # - persistent=False keeps checkpoints smaller
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions can be batched. Indexing with it gives
        # cos/sin shaped like (..., seq_len, d_k/2), matching x's batch/sequence dims.
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)

        # Split last dimension into even/odd coordinates of each 2D pair.
        # Example: [x0, x1, x2, x3] -> even=[x0, x2], odd=[x1, x3]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply standard 2D rotation:
        # [x_even']   [ cos -sin ] [x_even]
        # [x_odd' ] = [ sin  cos ] [x_odd ]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave rotated even/odd components back to original last-dim layout.
        return torch.stack((out_even, out_odd), dim=-1).reshape_as(x)