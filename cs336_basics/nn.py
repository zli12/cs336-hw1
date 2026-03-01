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
            approx = (8 * d_model) / 3
            d_ff = int(math.ceil(approx / 64) * 64)
        self.d_ff = d_ff

        # W1, W3: d_model -> d_ff ; W2: d_ff -> d_model
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w1 = self.w1(x)
        silu = x_w1 * torch.sigmoid(x_w1)
        gated = silu * self.w3(x)
        return self.w2(gated)