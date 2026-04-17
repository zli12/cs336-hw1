from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


class SGD(torch.optim.Optimizer):
    """SGD with step-dependent scaling: lr / sqrt(t + 1)."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None if closure is None else closure()
        # PyTorch optimizers may have multiple parameter groups with different hyperparameters.
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Keep per-parameter iteration count in optimizer state.
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                # Update rule: theta <- theta - (lr / sqrt(t + 1)) * grad.
                p.data -= (lr / math.sqrt(t + 1)) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer (Algorithm 2 of Loshchilov & Hutter, 2019)."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # On the very first step, initialize the moment buffers to zeros.
                # m = first moment  (tracks running mean of gradients)
                # v = second moment (tracks running mean of squared gradients)
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # t starts at 1 (first call = iteration 1).
                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # m <- beta1 * m + (1 - beta1) * grad
                # Exponential moving average of the gradient (momentum).
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v <- beta2 * v + (1 - beta2) * grad^2
                # Exponential moving average of squared gradient (adaptive scaling).
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Fold bias correction into the learning rate instead of
                # correcting m and v separately:
                #   alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_corrected_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # theta <- theta - alpha_t * m / (sqrt(v) + eps)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-bias_corrected_lr)

                # Decoupled weight decay: pull parameters toward zero.
                # theta <- theta - lr * lambda * theta
                # This is the key difference from original Adam, where weight
                # decay was entangled with the gradient update.
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip combined gradient l2-norm of *parameters* to at most *max_l2_norm* in-place."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total_norm = torch.sqrt(sum(p.grad.data.norm() ** 2 for p in params))
    clip_coeff = max_l2_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        for p in params:
            p.grad.data.mul_(clip_coeff)


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return learning rate at iteration *it* for cosine schedule with linear warmup."""
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate
