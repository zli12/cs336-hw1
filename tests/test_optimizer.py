import numpy
import pytest
import torch

from cs336_basics.optim import (
    AdamW,
    Muon,
    _zeropower_via_newtonschulz5,
    build_mixed_param_groups,
    lr_wsd_schedule,
)

from .adapters import get_adamw_cls, run_get_lr_cosine_schedule


def _optimize(opt_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_adamw(numpy_snapshot):
    """
    Our reference implementation yields slightly different results than the
    PyTorch AdamW, since there are a couple different ways that you can apply
    weight decay that are equivalent in principle, but differ in practice due to
    floating point behavior. So, we test that the provided implementation matches
    _either_ our reference implementation's expected results or those from the PyTorch AdamW.
    """
    # expected_weights = torch.load(FIXTURES_PATH / "adamw_expected_params.pt")
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(get_adamw_cls())

    # Might need to exit early if the weights match pytorch, since that should also be valid
    matches_pytorch = torch.allclose(actual_weights, pytorch_weights, atol=1e-4)
    if matches_pytorch:
        return

    numpy_snapshot.assert_match(
        actual_weights,
        atol=1e-4,
    )


def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    actual_lrs = [
        run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]
    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))


# ---- Muon -----------------------------------------------------------------


def test_zeropower_via_newtonschulz5_preserves_shape_and_normalizes_singular_values():
    """NS5 should: (a) keep the input shape, (b) push the input toward an orthogonal-like matrix.

    For a true ``UV^T``, ``||UV^T||_F = sqrt(min(M, N))``. The NS5 coefficients are tuned to land
    singular values in (0.5, 1.5) rather than at exactly 1, so the Frobenius norm should be in the
    ballpark of ``sqrt(min(M, N))`` (within a factor of ~2).
    """
    torch.manual_seed(0)
    G = torch.randn(8, 16)
    X = _zeropower_via_newtonschulz5(G, ns_steps=5)
    assert X.shape == G.shape
    # Expected Frobenius norm for an orthogonal MxN matrix (M < N) is sqrt(M) = sqrt(8) ~= 2.83.
    # NS5 with the slope-at-zero coefficients lands singular values in [0.5, 1.5], so Frobenius
    # norm is in [sqrt(8)*0.5, sqrt(8)*1.5] = [1.41, 4.24].
    fro = X.norm().item()
    assert 1.0 < fro < 5.0, f"NS5 Frobenius norm {fro:.3f} outside expected range for 8x16"


def test_muon_step_reduces_quadratic_loss_on_2d_weight():
    """End-to-end: Muon should drive a 2D weight toward a target on a quadratic loss."""
    torch.manual_seed(0)
    target = torch.randn(4, 8)
    p = torch.nn.Parameter(torch.randn(4, 8) * 0.1)

    opt = Muon([p], lr=0.05, momentum=0.9, ns_steps=5)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss))

    # Expect monotone-ish decrease and a meaningful overall drop.
    assert losses[-1] < losses[0] * 0.5, f"Muon failed to reduce loss: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_muon_rejects_1d_params():
    """Muon should refuse 1D params (norm gains, biases) so callers wire splits correctly."""
    p_1d = torch.nn.Parameter(torch.randn(8))
    opt = Muon([p_1d], lr=0.01)
    p_1d.grad = torch.randn(8)
    with pytest.raises(ValueError, match="1D parameter"):
        opt.step()


def test_build_mixed_param_groups_splits_2d_weights_from_1d_norms_and_embeddings():
    """build_mixed_param_groups: 2D matmul weights -> Muon group, embeds + 1D + biases -> AdamW group."""

    class TinyTransformerLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embeddings = torch.nn.Embedding(100, 16)
            self.attn_q = torch.nn.Linear(16, 16, bias=False)
            self.attn_k = torch.nn.Linear(16, 16, bias=False)
            self.norm_gain = torch.nn.Parameter(torch.ones(16))  # 1D
            self.bias = torch.nn.Parameter(torch.zeros(16))      # 1D

    model = TinyTransformerLike()
    muon_groups, adamw_groups, manifest = build_mixed_param_groups(model)

    muon_param_ids = {id(p) for g in muon_groups for p in g["params"]}
    adamw_param_ids = {id(p) for g in adamw_groups for p in g["params"]}

    assert id(model.attn_q.weight) in muon_param_ids
    assert id(model.attn_k.weight) in muon_param_ids
    assert id(model.token_embeddings.weight) in adamw_param_ids
    assert id(model.norm_gain) in adamw_param_ids
    assert id(model.bias) in adamw_param_ids

    # No overlap between the two groups.
    assert muon_param_ids.isdisjoint(adamw_param_ids)
    # Manifest names are present and consistent.
    assert "attn_q.weight" in manifest["muon"]
    assert "token_embeddings.weight" in manifest["adamw"]
    assert "norm_gain" in manifest["adamw"]


def test_build_mixed_param_groups_dedups_tied_embeddings():
    """A tied LM head shares parameter identity with the embedding table; should be counted once."""

    class TiedLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embeddings = torch.nn.Embedding(100, 16)
            self.attn_proj = torch.nn.Linear(16, 16, bias=False)
            # Simulate weight tying by re-binding the same Tensor.
            self.lm_head = torch.nn.Linear(16, 100, bias=False)
            self.lm_head.weight = self.token_embeddings.weight

    model = TiedLM()
    muon_groups, adamw_groups, manifest = build_mixed_param_groups(model)

    # The shared (V, d) tensor should appear exactly once in the AdamW group, never in Muon.
    embed_id = id(model.token_embeddings.weight)
    adamw_ids = [id(p) for g in adamw_groups for p in g["params"]]
    muon_ids = [id(p) for g in muon_groups for p in g["params"]]
    assert adamw_ids.count(embed_id) == 1
    assert embed_id not in muon_ids
    # attn_proj.weight (the only "real" 2D matmul matrix here) should be in Muon.
    assert id(model.attn_proj.weight) in muon_ids


def test_lr_wsd_schedule_phases():
    """WSD: linear warmup -> stable -> linear decay -> pinned at min."""
    max_lr = 1.0
    min_lr = 0.1
    warmup = 10
    total = 100
    decay_frac = 0.2  # decay starts at step 80

    # Warmup midpoint.
    assert lr_wsd_schedule(5, max_lr, min_lr, warmup, total, decay_frac) == pytest.approx(0.5)
    # End of warmup -> max LR.
    assert lr_wsd_schedule(10, max_lr, min_lr, warmup, total, decay_frac) == pytest.approx(max_lr)
    # Mid stable phase.
    assert lr_wsd_schedule(50, max_lr, min_lr, warmup, total, decay_frac) == pytest.approx(max_lr)
    # Just before decay starts (step 79 < 80).
    assert lr_wsd_schedule(79, max_lr, min_lr, warmup, total, decay_frac) == pytest.approx(max_lr)
    # Mid decay (step 90, halfway through the [80, 100) window).
    assert lr_wsd_schedule(90, max_lr, min_lr, warmup, total, decay_frac) == pytest.approx(0.55, rel=1e-3)
    # End of training.
    assert lr_wsd_schedule(100, max_lr, min_lr, warmup, total, decay_frac) == pytest.approx(min_lr)
    # Past end stays pinned.
    assert lr_wsd_schedule(150, max_lr, min_lr, warmup, total, decay_frac) == pytest.approx(min_lr)
