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


# ---- Muon ------------------------------------------------------------------


def _zeropower_via_newtonschulz5(G: torch.Tensor, ns_steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate ``U V^T`` (the orthogonal Procrustes solution) for ``G = U S V^T`` via a quintic
    Newton-Schulz iteration.

    Mathematical sketch: the iteration ``X <- a*X + (b*X X^T + c*(X X^T)^2) X`` with the coefficients
    ``(a, b, c) = (3.4445, -4.7750, 2.0315)`` is a polynomial in the singular values of ``X`` chosen
    by Keller Jordan to maximize the slope of that polynomial near zero. After enough steps the
    singular values land in roughly the (0.5, 1.5) interval, which is good enough as a stand-in for
    the SVD's ``U V^T = X / S`` whitening; the ratio doesn't matter for Muon's update direction
    because the sign-of-singular-values structure is preserved.

    Implementation notes:
    - We do the math in bf16 to match NanoGPT-speedrun and keep the iteration cheap; the input
      ``G`` may come in fp32 (Adam-style param-state convention).
    - We start by normalizing ``G`` so its top singular value is at most 1, which is required for
      the iteration to converge.
    - For wide matrices (more cols than rows) we work on ``G^T`` to keep the inner ``X X^T`` matrix
      as the smaller of the two (computational savings) and transpose back at the end.
    """
    assert G.ndim >= 2, f"Muon requires \u22652D params; got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Normalize spectral norm to <=1 so the iteration is in its convergence basin.
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.transpose(-1, -2)
        transposed = True
    for _ in range(ns_steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz (Keller Jordan, NanoGPT-speedrun).

    For each \u22652D weight matrix:
      buf <- momentum * buf + grad
      g   <- buf  (or, if nesterov=True, grad + momentum * buf)
      orth_g <- newton_schulz(g)
      p   <- p - lr * orth_g * sqrt(max(1, M / N))

    The ``sqrt(M / N)`` rescale comes from the original recipe and keeps the per-element update
    magnitude roughly comparable across rectangular matrices of different aspect ratios.

    Should be used together with AdamW for the parameters Muon doesn't apply to (embeddings,
    LM head, biases, RMSNorm gains, scalars). Use :func:`build_mixed_param_groups` to do that
    split correctly.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim < 2:
                    raise ValueError(
                        f"Muon got a {grad.ndim}D parameter (shape {tuple(p.shape)}); "
                        f"split 1D params (norms, biases) into a separate AdamW group "
                        f"via build_mixed_param_groups()."
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["t"] += 1
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update_dir = grad.add(buf, alpha=momentum) if nesterov else buf
                orth = _zeropower_via_newtonschulz5(update_dir, ns_steps=ns_steps)
                # Per Keller Jordan, scale by sqrt(max(1, rows/cols)) so wider matrices update
                # at a comparable per-element magnitude to taller ones.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.data.add_(orth, alpha=-lr * scale)
                # Decoupled weight decay (matches AdamW conventions). Off by default.
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
        return loss


def build_mixed_param_groups(
    model: torch.nn.Module,
    *,
    muon_kwargs: dict | None = None,
    adamw_kwargs: dict | None = None,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Split *model*'s parameters into a Muon group (\u22652D weight matrices in transformer blocks)
    and an AdamW group (everything else: embeddings, LM head, biases, 1D norm gains).

    Returns ``(muon_groups, adamw_groups, manifest)`` where:
    - ``muon_groups`` and ``adamw_groups`` are PyTorch-style ``[{"params": [...], **defaults}]``
      lists ready to feed into ``Muon(...)`` / ``AdamW(...)``.
    - ``manifest`` is ``{"muon": [...names...], "adamw": [...names...]}`` for debugging /
      reproducibility logging.

    Decision rules (matching NanoGPT-speedrun conventions):
    - 1D params (norm gains, biases, scalars) -> AdamW.
    - Tensors with the same identity as ``model.token_embeddings.weight`` (i.e. the embedding
      table, including the tied LM head) -> AdamW. This avoids running Muon on the (V, d_model)
      table where the orthogonalization rescale doesn't make physical sense and where AdamW's
      sparse-friendly behavior is preferred.
    - Anything else with ``ndim >= 2`` -> Muon (the Q/K/V/O projections, the FFN matrices, etc.).
    """
    muon_kwargs = dict(muon_kwargs or {})
    adamw_kwargs = dict(adamw_kwargs or {})

    embed_table_id: int | None = None
    embed = getattr(model, "token_embeddings", None)
    embed_w = getattr(embed, "weight", None) if embed is not None else None
    if isinstance(embed_w, torch.Tensor):
        embed_table_id = id(embed_w)

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            # A tied parameter shows up under multiple names (e.g. embed.weight + lm_head.weight
            # when --tie-embeddings is on). We only want to optimize each tensor once.
            continue
        seen.add(id(p))
        is_embed = embed_table_id is not None and id(p) == embed_table_id
        if p.ndim < 2 or is_embed:
            adamw_params.append(p)
            adamw_names.append(name)
        else:
            muon_params.append(p)
            muon_names.append(name)

    muon_groups = [{"params": muon_params, **muon_kwargs}] if muon_params else []
    adamw_groups = [{"params": adamw_params, **adamw_kwargs}] if adamw_params else []
    manifest = {"muon": muon_names, "adamw": adamw_names}
    return muon_groups, adamw_groups, manifest


def lr_wsd_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    total_iters: int,
    decay_frac: float = 0.2,
) -> float:
    """Warmup-Stable-Decay schedule.

    - ``[0, warmup_iters)``: linear warmup ``0 -> max_learning_rate``.
    - ``[warmup_iters, total_iters * (1 - decay_frac))``: stable at ``max_learning_rate``.
    - ``[total_iters * (1 - decay_frac), total_iters]``: linear decay ``max -> min``.
    - ``> total_iters``: pinned at ``min_learning_rate``.

    Compared to cosine, WSD is easier to extend (just keep the stable phase), and the late linear
    decay window is what consistently helps small-model speedruns squeeze out the last few percent
    of val loss.
    """
    if it < warmup_iters:
        # Avoid division-by-zero edge case for warmup_iters == 0.
        return max_learning_rate * (it / max(1, warmup_iters))
    decay_start = int(total_iters * (1.0 - decay_frac))
    if it < decay_start:
        return max_learning_rate
    if it >= total_iters:
        return min_learning_rate
    decay_progress = (it - decay_start) / max(1, total_iters - decay_start)
    return max_learning_rate + (min_learning_rate - max_learning_rate) * decay_progress
