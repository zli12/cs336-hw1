from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn


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
    attn_probs = torch.softmax(attention_logits, dim=-1)

    # Weighted sum of value vectors for each query position.
    # value_dim is the output feature dimension for each value vector.
    return einsum(
        attn_probs.to(V.dtype),
        V,
        "... query_pos key_pos, ... key_pos value_dim -> ... query_pos value_dim",
    )


def sdpa_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Attention via PyTorch's fused SDPA kernel (Flash-Attention 2 on capable GPUs).

    Inputs follow the (..., num_heads, seq_len, head_dim) layout already produced
    by CausalMultiHeadSelfAttention. With ``is_causal=True`` the kernel applies
    the lower-triangular mask implicitly without materializing it, which is the
    main reason this path is faster and uses less memory than the einsum path.
    """
    # F.scaled_dot_product_attention scales by 1/sqrt(head_dim) by default,
    # matching the einsum implementation above.
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


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


class SiLUFeedForward(nn.Module):
    """Ungated position-wise feed-forward block.

    Computes: W2( SiLU(W1 x) ).
    Matched parameter count to SwiGLU uses d_ff = 4 * d_model (per assignment).
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
            d_ff = 4 * d_model
        self.d_ff = d_ff

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w1 = self.w1(x)
        silu = x_w1 * torch.sigmoid(x_w1)
        return self.w2(silu)


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
    """Causal multi-head self-attention with optional RoPE and QK-Norm.

    When ``use_sdpa=True`` (the fast path), the attention math runs through
    PyTorch's fused ``F.scaled_dot_product_attention`` (Flash-Attention 2 on
    capable GPUs). Otherwise the einsum path materializes the causal mask and
    softmax explicitly; that path is kept for assignment-required tests and
    for CPU/MPS fallbacks.

    When ``qk_norm=True`` we apply a per-head RMSNorm to Q and K *after* RoPE
    rotation and *before* the attention dot product. This is a NanoGPT-speedrun
    style stabilizer that lets you push the optimizer LR significantly higher
    without losing the QK product to numerical drift.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_sdpa: bool = False,
        qk_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_sdpa = use_sdpa
        self.qk_norm = qk_norm

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

        # Per-head RMSNorm operates on the last dim (head_dim); the same weight
        # is applied to every head. We keep separate norms for Q and K because
        # they evolve with different gradients during training.
        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if qk_norm:
            self.q_norm = RMSNorm(d_model=self.head_dim, device=device, dtype=dtype)
            self.k_norm = RMSNorm(d_model=self.head_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Star-unpacking keeps any leading batch-like dims (e.g., batch, heads, shards),
        # then extracts sequence length and ignores the final model-width entry.
        *batch_dims, seq_len, _ = x.shape

        # Project once per tensor, then split model width into (num_heads, head_dim).
        # rearrange keeps the operation declarative and shape-focused.
        q = rearrange(
            self.q_proj(x),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        k = rearrange(
            self.k_proj(x),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        v = rearrange(
            self.v_proj(x),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )

        # When RoPE is enabled, rotate Q/K in each head with identical position angles.
        # V is intentionally left unchanged.
        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Apply QK-Norm post-RoPE so the gain weights live in the post-rotation basis.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_sdpa:
            # Causal mask is applied implicitly by the fused kernel.
            attn_out = sdpa_attention(Q=q, K=k, V=v, is_causal=True)
        else:
            # Causal mask shared by all batch items/heads:
            # token i only sees tokens at positions <= i.
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
            attn_out = scaled_dot_product_attention(Q=q, K=k, V=v, mask=causal_mask)

        # Move heads back into the model-width axis before the output projection.
        attn_out = rearrange(attn_out, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        return self.output_proj(attn_out)


class TransformerBlock(nn.Module):
    """Configurable Transformer block supporting pre/post-norm, no-norm, no-RoPE, and SwiGLU/SiLU FFN.

    Default behavior is pre-norm with RMSNorm, RoPE-enabled MHA, and SwiGLU FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        use_norm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        use_sdpa: bool = False,
        qk_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.post_norm = post_norm
        self.use_rope = use_rope
        self.ffn_type = ffn_type
        self.use_sdpa = use_sdpa
        self.qk_norm = qk_norm

        # Norms become identity passthroughs when disabled, preserving forward shape.
        if use_norm:
            self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        # CausalMultiHeadSelfAttention enables RoPE only when both kwargs are provided.
        attn_kwargs: dict[str, Any] = dict(
            d_model=d_model,
            num_heads=num_heads,
            use_sdpa=use_sdpa,
            qk_norm=qk_norm,
            device=device,
            dtype=dtype,
        )
        if use_rope:
            attn_kwargs["max_seq_len"] = max_seq_len
            attn_kwargs["theta"] = theta
        self.attn = CausalMultiHeadSelfAttention(**attn_kwargs)

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type!r} (expected 'swiglu' or 'silu')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model with optional ablation flags.

    Phase-1 leaderboard knobs (all default-off; turning them on should never
    regress validation loss but typically improves it):

    - ``qk_norm``: per-head RMSNorm on Q/K post-RoPE; stabilizes high-LR training.
    - ``tie_embeddings``: share parameters between the input embedding and the LM
      head (Vaswani 2017 §3.4). Needs a smaller ``embed_init_std`` to keep the
      initial logits unit-variance — default to ``1/sqrt(d_model)`` when tying.
    - ``embed_init_std``: override the default N(0,1) trunc init for the embedding
      table. Useful even without tying when the LM head is shared with embed init.
    - ``logit_soft_cap``: applies ``cap * tanh(logits / cap)`` to bound the output
      logits (Gemma-2 style); reduces outlier softmax explosions in bf16.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_norm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        use_sdpa: bool = False,
        qk_norm: bool = False,
        tie_embeddings: bool = False,
        embed_init_std: float | None = None,
        logit_soft_cap: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.use_sdpa = use_sdpa
        self.qk_norm = qk_norm
        self.tie_embeddings = tie_embeddings
        self.logit_soft_cap = logit_soft_cap

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        # When tying, the embedding doubles as the LM head, so its init must be
        # small enough that initial logits are roughly unit-variance. Default
        # to Llama-style 1/sqrt(d_model) when caller didn't override.
        effective_embed_std = embed_init_std
        if effective_embed_std is None and tie_embeddings:
            effective_embed_std = 1.0 / math.sqrt(d_model)
        if effective_embed_std is not None:
            with torch.no_grad():
                std = float(effective_embed_std)
                nn.init.trunc_normal_(
                    self.token_embeddings.weight,
                    mean=0.0,
                    std=std,
                    a=-3.0 * std,
                    b=3.0 * std,
                )
        self.embed_init_std = effective_embed_std

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    use_norm=use_norm,
                    post_norm=post_norm,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        # Final norm becomes identity when disabled to keep the forward path uniform.
        if use_norm:
            self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
        if tie_embeddings:
            # Embedding weight is shape (vocab, d_model); Linear weight is shape
            # (out=vocab, in=d_model). Same shape, so we can share the same
            # nn.Parameter directly. PyTorch's optimizer will only see the
            # parameter once because Module.parameters() de-duplicates.
            self.lm_head.weight = self.token_embeddings.weight

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
        logits = self.lm_head(x)
        if self.logit_soft_cap is not None:
            cap = float(self.logit_soft_cap)
            logits = torch.tanh(logits / cap) * cap
        return logits
