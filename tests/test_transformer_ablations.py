"""Construction + forward-shape tests for ablation flags on TransformerBlock/TransformerLM.

Validates the wiring added for problems in 7.3 (no-RMSNorm, post-norm, NoPE, SiLU FFN)
and the Phase-0 leaderboard infra flag (use_sdpa).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cs336_basics.transformer import (
    CausalMultiHeadSelfAttention,
    RMSNorm,
    SiLUFeedForward,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
)


def _build_block(**overrides) -> TransformerBlock:
    kwargs = dict(
        d_model=16,
        num_heads=4,
        d_ff=32,
        max_seq_len=8,
        theta=10_000.0,
    )
    kwargs.update(overrides)
    return TransformerBlock(**kwargs)


def _build_lm(**overrides) -> TransformerLM:
    kwargs = dict(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        rope_theta=10_000.0,
    )
    kwargs.update(overrides)
    return TransformerLM(**kwargs)


def test_default_block_uses_rmsnorm_swiglu_and_rope() -> None:
    block = _build_block()
    assert isinstance(block.ln1, RMSNorm)
    assert isinstance(block.ln2, RMSNorm)
    assert isinstance(block.ffn, SwiGLU)
    assert block.attn.rope is not None
    assert block.post_norm is False


def test_block_no_norm_replaces_layernorms_with_identity() -> None:
    block = _build_block(use_norm=False)
    assert isinstance(block.ln1, nn.Identity)
    assert isinstance(block.ln2, nn.Identity)


def test_block_no_rope_disables_rope_in_attn() -> None:
    block = _build_block(use_rope=False)
    assert block.attn.rope is None


def test_block_silu_ffn_uses_silu_feedforward() -> None:
    block = _build_block(ffn_type="silu")
    assert isinstance(block.ffn, SiLUFeedForward)


def test_block_invalid_ffn_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ffn_type"):
        _build_block(ffn_type="bogus")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # default pre-norm
        {"use_norm": False},
        {"post_norm": True},
        {"use_rope": False},
        {"ffn_type": "silu"},
    ],
)
def test_block_forward_preserves_shape(kwargs: dict) -> None:
    block = _build_block(**kwargs)
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape


def test_lm_no_norm_replaces_final_layernorm_with_identity() -> None:
    lm = _build_lm(use_norm=False)
    assert isinstance(lm.ln_final, nn.Identity)
    for layer in lm.layers:
        assert isinstance(layer.ln1, nn.Identity)
        assert isinstance(layer.ln2, nn.Identity)


def test_lm_post_norm_runs_blocks_in_post_norm_mode() -> None:
    lm = _build_lm(post_norm=True)
    for layer in lm.layers:
        assert layer.post_norm is True


def test_lm_no_rope_disables_rope_in_every_block() -> None:
    lm = _build_lm(use_rope=False)
    for layer in lm.layers:
        assert layer.attn.rope is None


def test_lm_silu_ffn_uses_silu_in_every_block() -> None:
    lm = _build_lm(ffn_type="silu")
    for layer in lm.layers:
        assert isinstance(layer.ffn, SiLUFeedForward)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"use_norm": False},
        {"post_norm": True},
        {"use_rope": False},
        {"ffn_type": "silu"},
    ],
)
def test_lm_forward_returns_logits_with_correct_shape(kwargs: dict) -> None:
    lm = _build_lm(**kwargs)
    in_indices = torch.randint(low=0, high=32, size=(2, 8))
    logits = lm(in_indices)
    assert logits.shape == (2, 8, 32)


def test_silu_feedforward_default_d_ff_matches_4_d_model() -> None:
    ff = SiLUFeedForward(d_model=64)
    assert ff.d_ff == 256


def test_block_use_sdpa_propagates_to_attention() -> None:
    block = _build_block(use_sdpa=True)
    assert block.use_sdpa is True
    assert block.attn.use_sdpa is True


def test_lm_use_sdpa_propagates_to_every_block() -> None:
    lm = _build_lm(use_sdpa=True)
    assert lm.use_sdpa is True
    for layer in lm.layers:
        assert layer.use_sdpa is True
        assert layer.attn.use_sdpa is True


def test_lm_forward_sdpa_path_returns_logits_with_correct_shape() -> None:
    lm = _build_lm(use_sdpa=True)
    in_indices = torch.randint(low=0, high=32, size=(2, 8))
    logits = lm(in_indices)
    assert logits.shape == (2, 8, 32)


def test_sdpa_path_matches_einsum_path_in_fp32() -> None:
    """The fused SDPA kernel and the einsum reference path should agree on causal output."""
    torch.manual_seed(123)
    common = dict(
        d_model=32,
        num_heads=4,
        max_seq_len=16,
        theta=10_000.0,
    )
    attn_ref = CausalMultiHeadSelfAttention(**common, use_sdpa=False)
    attn_sdpa = CausalMultiHeadSelfAttention(**common, use_sdpa=True)
    # Mirror weights so both paths see identical Q/K/V projections.
    attn_sdpa.load_state_dict(attn_ref.state_dict())

    x = torch.randn(2, 12, 32)
    y_ref = attn_ref(x)
    y_sdpa = attn_sdpa(x)
    torch.testing.assert_close(y_ref, y_sdpa, rtol=1e-4, atol=1e-5)


def test_attention_qk_norm_creates_per_head_rmsnorms() -> None:
    attn = CausalMultiHeadSelfAttention(d_model=32, num_heads=4, qk_norm=True)
    assert attn.qk_norm is True
    assert isinstance(attn.q_norm, RMSNorm)
    assert isinstance(attn.k_norm, RMSNorm)
    # Per-head RMSNorm normalizes head_dim=8.
    assert attn.q_norm.weight.shape == (32 // 4,)
    assert attn.k_norm.weight.shape == (32 // 4,)


def test_block_qk_norm_propagates_to_attention() -> None:
    block = _build_block(qk_norm=True)
    assert block.qk_norm is True
    assert block.attn.qk_norm is True


def test_lm_qk_norm_propagates_to_every_block() -> None:
    lm = _build_lm(qk_norm=True)
    for layer in lm.layers:
        assert layer.attn.qk_norm is True


def test_lm_forward_qk_norm_path_returns_logits_with_correct_shape() -> None:
    lm = _build_lm(qk_norm=True)
    in_indices = torch.randint(low=0, high=32, size=(2, 8))
    logits = lm(in_indices)
    assert logits.shape == (2, 8, 32)


def test_lm_tie_embeddings_shares_parameter_identity() -> None:
    lm = _build_lm(tie_embeddings=True)
    assert lm.tie_embeddings is True
    # Same Tensor object, so optimizer/save/load handle a single parameter.
    assert lm.lm_head.weight is lm.token_embeddings.weight


def test_lm_tie_embeddings_default_init_std_is_one_over_sqrt_d_model() -> None:
    import math
    lm = _build_lm(tie_embeddings=True)
    expected = 1.0 / math.sqrt(16)  # _build_lm uses d_model=16
    assert lm.embed_init_std is not None
    assert lm.embed_init_std == pytest.approx(expected, rel=1e-6)
    # Embedding weights should respect the smaller std (loose bound to allow trunc range).
    assert lm.token_embeddings.weight.std().item() < 1.0


def test_lm_embed_init_std_overrides_default() -> None:
    lm = _build_lm(embed_init_std=0.02)
    assert lm.embed_init_std == 0.02
    # Trunc-normal at std=0.02 keeps every weight within +/- 3 * 0.02 = 0.06.
    assert lm.token_embeddings.weight.abs().max().item() < 0.06 + 1e-6


def test_lm_logit_soft_cap_bounds_output_logits() -> None:
    lm = _build_lm(logit_soft_cap=1.0)
    in_indices = torch.randint(low=0, high=32, size=(4, 8))
    logits = lm(in_indices)
    # Bound: |cap*tanh(z/cap)| < cap. Strict <= since tanh saturates only at infinity.
    assert logits.abs().max().item() < 1.0


def test_lm_no_logit_soft_cap_has_unbounded_logits_in_principle() -> None:
    """Sanity check that the cap is opt-in: with cap=None the magnitude is not artificially clipped."""
    lm = _build_lm(logit_soft_cap=None)
    assert lm.logit_soft_cap is None


def test_lm_tie_embeddings_forward_returns_correct_shape() -> None:
    lm = _build_lm(tie_embeddings=True, logit_soft_cap=10.0, qk_norm=True)
    in_indices = torch.randint(low=0, high=32, size=(2, 8))
    logits = lm(in_indices)
    assert logits.shape == (2, 8, 32)
    assert logits.abs().max().item() < 10.0
