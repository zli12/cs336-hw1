from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import nn


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


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention with optional boolean masking."""
    # Q/K store key-query channels in their last dimension.
    d_k = Q.shape[-1]

    # Dot each query against each key using readable axis names:
    # - query_pos/key_pos are sequence positions
    # - d_k is the per-head channel dimension being summed out
    # Then scale by 1/sqrt(d_k) as in Vaswani et al.
    attention_logits = einsum(
        Q,
        K,
        "... query_pos d_k, ... key_pos d_k -> ... query_pos key_pos",
    ) / math.sqrt(d_k)
    if mask is not None:
        # True means "can attend"; False means "blocked".
        mask = mask.to(dtype=torch.bool, device=attention_logits.device)
        # Put -inf on blocked logits so softmax gives them zero probability.
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))

    # Convert logits to probabilities that sum to 1 across the key axis.
    attn_probs = softmax(attention_logits, dim=-1)

    # Weighted sum of value vectors for each query position.
    # value_dim is the output feature dimension for each value vector.
    return einsum(
        attn_probs.to(V.dtype),
        V,
        "... query_pos key_pos, ... key_pos value_dim -> ... query_pos value_dim",
    )


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
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


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

        # upcast to float32 before squaring to improve numerical stability.
        x_fp32 = x.to(torch.float32)        

        # compute RMS over the final dimension:
        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)

        # normalize then apply gain:
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
        # stack(..., dim=-1) makes pairs [even_i, odd_i], and reshape flattens to
        # [..., even_0, odd_0, even_1, odd_1, ...].
        return torch.stack((out_even, out_odd), dim=-1).reshape_as(x)


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One projection each for Q/K/V over all heads at once.
        self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

        # RoPE is optional for this class:
        # - if both theta and max_seq_len are provided, enable RoPE on Q/K
        # - otherwise run standard (non-RoPE) causal MHA
        if (max_seq_len is None) ^ (theta is None):
            raise ValueError("Provide both max_seq_len and theta together to enable RoPE.")
        self.rope: RotaryPositionalEmbedding | None = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Star-unpacking keeps any leading batch-like dims (e.g., batch, heads, shards),
        # then extracts sequence length and ignores the final model-width entry.
        *batch_dims, seq_len, _ = x.shape

        # Project once per tensor, then split model width into (num_heads, head_dim).
        # rearrange keeps the operation declarative and shape-focused.
        # Each linear has one matric multiplication, FLOPs = 2 * d_model * d_model * d_model.
        # Total FLOPs for Q, K, V: 2 * 3 * d_model * d_model * d_model = 6 * d_model^3.
        q = rearrange(self.q_proj(x), "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)

        # When RoPE is enabled, rotate Q/K in each head with identical position angles.
        # V is intentionally left unchanged.
        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Causal mask shared by all batch items/heads:
        # token i only sees tokens at positions <= i.
        # Example (seq_len=4), shape (Q_seq_len, K_seq_len), so Qi only attends to Kj for j <= i.
        # [[True, False, False, False],
        #  [True, True, False, False],
        #  [True, True, True, False],
        #  [True, True, True, True]]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn_out = scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal_mask)

        # Move heads back into the model-width axis before the output projection.
        attn_out = rearrange(attn_out, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        return self.output_proj(attn_out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE-enabled causal self-attention."""
    # Diagram flow, shape preserved as (... seq_len d_model):
    # x -> RMSNorm -> Causal MHA (with RoPE) -> Add residual
    #   -> RMSNorm -> Position-wise FFN (SwiGLU) -> Add residual.

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Pre-norm setup: normalize before each sublayer.
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # First sublayer: causal MHA with RoPE on Q/K.
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # Second sublayer: position-wise feed-forward network.
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path around attention sublayer.
        x = x + self.attn(self.ln1(x))
        # Residual path around feed-forward sublayer.
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model with tied architecture components."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        # Token IDs -> dense vectors.
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        # Decoder stack of identical pre-norm Transformer blocks.
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        # Final normalization and vocabulary projection to logits.
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        seq_len = in_indices.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds context_length ({self.context_length})."
            )

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)