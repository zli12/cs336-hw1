"""Construction + forward-shape tests for ablation flags on TransformerBlock/TransformerLM.

Validates the wiring added for problems in 7.3 (no-RMSNorm, post-norm, NoPE, SiLU FFN).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cs336_basics.transformer import (
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
