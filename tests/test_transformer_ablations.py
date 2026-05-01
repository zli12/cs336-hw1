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


def test_block_default_attn_kernel_is_einsum() -> None:
    block = _build_block()
    assert block.attn.attn_kernel == "einsum"


def test_block_torch_attn_kernel_propagates() -> None:
    block = _build_block(attn_kernel="torch")
    assert block.attn.attn_kernel == "torch"


def test_block_invalid_attn_kernel_raises() -> None:
    with pytest.raises(ValueError, match="attn_kernel must be"):
        _build_block(attn_kernel="bogus")


def test_block_qk_norm_default_off() -> None:
    block = _build_block()
    assert block.qk_norm is False
    assert isinstance(block.attn.q_layernorm, nn.Identity)
    assert isinstance(block.attn.k_layernorm, nn.Identity)


def test_block_qk_norm_on_uses_rmsnorm_per_head() -> None:
    block = _build_block(qk_norm=True)
    assert block.qk_norm is True
    # head_dim = d_model / num_heads = 16/4 = 4 in the test fixture.
    assert isinstance(block.attn.q_layernorm, RMSNorm)
    assert isinstance(block.attn.k_layernorm, RMSNorm)
    assert block.attn.q_layernorm.d_model == 4
    assert block.attn.k_layernorm.d_model == 4


def test_block_torch_attn_kernel_matches_einsum_numerically() -> None:
    """Both attention kernels should produce numerically identical outputs (within fp tolerance)."""
    torch.manual_seed(0)
    block_einsum = _build_block(attn_kernel="einsum")
    block_torch = _build_block(attn_kernel="torch")
    # Copy weights so the only diff is the inner attention kernel.
    block_torch.load_state_dict(block_einsum.state_dict())
    x = torch.randn(2, 8, 16)
    out_einsum = block_einsum(x)
    out_torch = block_torch(x)
    torch.testing.assert_close(out_einsum, out_torch, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # default pre-norm
        {"use_norm": False},
        {"post_norm": True},
        {"use_rope": False},
        {"ffn_type": "silu"},
        {"attn_kernel": "torch"},
        {"qk_norm": True},
        {"qk_norm": True, "attn_kernel": "torch"},
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


def test_lm_torch_attn_kernel_propagates_to_every_block() -> None:
    lm = _build_lm(attn_kernel="torch")
    for layer in lm.layers:
        assert layer.attn.attn_kernel == "torch"


def test_lm_tie_embeddings_shares_weight_tensor() -> None:
    lm = _build_lm(tie_embeddings=True)
    # The two parameters must share the same Tensor storage so SGD updates
    # both embedding lookups and the LM head simultaneously.
    assert lm.lm_head.weight.data_ptr() == lm.token_embeddings.weight.data_ptr()


def test_lm_default_does_not_tie_embeddings() -> None:
    lm = _build_lm()
    assert lm.lm_head.weight.data_ptr() != lm.token_embeddings.weight.data_ptr()


def test_lm_tied_embedding_init_uses_smaller_std() -> None:
    """Tied embeddings should use 1/sqrt(d_model) init so logits start small."""
    torch.manual_seed(0)
    lm_untied = _build_lm()
    torch.manual_seed(0)
    lm_tied = _build_lm(tie_embeddings=True)
    # Tied embedding std should be ~1/sqrt(16) = 0.25, vs untied which is ~1.0.
    untied_std = lm_untied.token_embeddings.weight.detach().std().item()
    tied_std = lm_tied.token_embeddings.weight.detach().std().item()
    assert tied_std < untied_std / 2, f"Expected tied std << untied std, got tied={tied_std} untied={untied_std}"


def test_lm_qk_norm_propagates_to_every_block() -> None:
    lm = _build_lm(qk_norm=True)
    for layer in lm.layers:
        assert layer.qk_norm is True
        assert isinstance(layer.attn.q_layernorm, RMSNorm)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"use_norm": False},
        {"post_norm": True},
        {"use_rope": False},
        {"ffn_type": "silu"},
        {"attn_kernel": "torch"},
        {"qk_norm": True},
        {"tie_embeddings": True},
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
