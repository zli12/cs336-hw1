import numpy
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from cs336_basics.nn import cross_entropy_with_z_loss

from .adapters import run_cross_entropy, run_gradient_clipping, run_softmax


def test_softmax_matches_pytorch():
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(run_softmax(x, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-6)
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        run_softmax(x + 100, dim=-1).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-6,
    )


def test_cross_entropy():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1)).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )

    # Test that cross-entropy handles numerical overflow issues
    large_inputs = 1000.0 * inputs
    large_expected_cross_entropy = F.cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1)).detach().numpy(),
        large_expected_cross_entropy.detach().numpy(),
        atol=1e-4,
    )


def test_cross_entropy_with_z_loss_components():
    """z=0 collapses to plain CE; z>0 strictly adds a non-negative penalty."""
    torch.manual_seed(0)
    inputs = torch.randn(3, 7, 5)  # batch, seq, vocab
    targets = torch.randint(0, 5, (3, 7))

    flat_inputs = inputs.view(-1, inputs.size(-1))
    flat_targets = targets.view(-1)
    expected_ce = F.cross_entropy(flat_inputs, flat_targets)

    # When z_weight=0, total should equal plain CE within float tolerance.
    total_zero, ce_zero, z = cross_entropy_with_z_loss(flat_inputs, flat_targets, z_weight=0.0)
    numpy.testing.assert_allclose(ce_zero.detach().numpy(), expected_ce.detach().numpy(), atol=1e-5)
    numpy.testing.assert_allclose(total_zero.detach().numpy(), expected_ce.detach().numpy(), atol=1e-5)
    assert z.item() >= 0  # mean of squares is non-negative

    # Non-zero weight should add a strictly positive penalty (CE term unchanged).
    total, ce, z = cross_entropy_with_z_loss(flat_inputs, flat_targets, z_weight=1e-2)
    numpy.testing.assert_allclose(ce.detach().numpy(), expected_ce.detach().numpy(), atol=1e-5)
    assert total.item() > ce.item()
    numpy.testing.assert_allclose(
        total.detach().numpy(),
        (ce + 1e-2 * z).detach().numpy(),
        atol=1e-6,
    )


def test_cross_entropy_with_z_loss_handles_large_logits():
    """Numerical stability: massive logits shouldn't NaN/inf the loss."""
    torch.manual_seed(0)
    inputs = 1000.0 * torch.randn(2, 4, 5)
    targets = torch.randint(0, 5, (2, 4))
    flat_inputs = inputs.view(-1, inputs.size(-1))
    flat_targets = targets.view(-1)
    total, ce, z = cross_entropy_with_z_loss(flat_inputs, flat_targets, z_weight=1e-4)
    assert torch.isfinite(total)
    assert torch.isfinite(ce)
    assert torch.isfinite(z)


def test_gradient_clipping():
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2

    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    # Test freezing one parameter.
    t1[-1].requires_grad_(False)

    loss = torch.cat(t1).sum()
    loss.backward()
    clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]

    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]

    assert len(t1_grads) == len(t1_c_grads)

    for t1_grad, t1_c_grad in zip(t1_grads, t1_c_grads):
        numpy.testing.assert_allclose(
            t1_grad.detach().numpy(),
            t1_c_grad.detach().numpy(),
            atol=1e-6,
        )
