from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

# Limit torch.compile parallel worker count BEFORE importing torch._inductor.
# Each worker grabs a separate CUDA context (~1.5 GiB on A10G) which can OOM
# the bs=96 forward graph. Single-threaded compilation is slower but bounded.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.nn import cross_entropy
from cs336_basics.optim import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.transformer import Embedding, RMSNorm, TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransformerLM on tokenized data.")

    # Data: tokenized arrays and how we sample fixed-length training windows from them.
    parser.add_argument("--train-data", type=Path, required=True, help="Path to training token file (.npy preferred).")
    parser.add_argument("--val-data", type=Path, default=None, help="Optional path to validation token file.")
    parser.add_argument(
        "--token-dtype",
        type=str,
        default="uint16",
        help="Token dtype for raw binary memmap files (ignored for .npy).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)

    # Model: architecture hyperparameters for TransformerLM.
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument(
        "--no-rms-norm",
        action="store_true",
        help="Disable RMSNorm in all blocks and final layer (layer-norm ablation).",
    )
    parser.add_argument(
        "--post-norm",
        action="store_true",
        help="Use post-norm Transformer blocks instead of the default pre-norm.",
    )
    parser.add_argument(
        "--no-rope",
        action="store_true",
        help="Disable RoPE position embeddings (NoPE ablation).",
    )
    parser.add_argument(
        "--ffn-type",
        type=str,
        default="swiglu",
        choices=["swiglu", "silu"],
        help="Feed-forward implementation: SwiGLU (default) or ungated SiLU.",
    )
    parser.add_argument(
        "--attn-kernel",
        type=str,
        default="einsum",
        choices=["einsum", "torch"],
        help="Attention inner kernel: 'einsum' (educational) or 'torch' (F.SDPA, dispatches to Flash on CUDA).",
    )
    parser.add_argument(
        "--tie-embeddings",
        action="store_true",
        help="Tie input embedding and LM head weights (saves vocab*d_model params).",
    )
    parser.add_argument(
        "--qk-norm",
        action="store_true",
        help="Apply RMSNorm to Q and K before RoPE (Llama-3 / Qwen-2.5 stabilizer).",
    )
    parser.add_argument(
        "--z-loss-coef",
        type=float,
        default=0.0,
        help="PaLM-style z-loss auxiliary coefficient on the squared log-normalizer (e.g. 1e-4).",
    )
    parser.add_argument(
        "--param-group-wd",
        action="store_true",
        help="Disable weight decay on RMSNorm gains and Embedding weights (matmul-only WD).",
    )

    # Optimization: update-rule hyperparameters.
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Learning-rate schedule (default fixed = backward compatible).
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="fixed",
        choices=["fixed", "cosine"],
        help="Learning-rate schedule kind. 'cosine' applies linear warmup + cosine decay to --min-learning-rate.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Minimum learning rate at end of cosine cycle (and after).",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=100,
        help="Number of linear warmup iterations (only used when --lr-schedule cosine).",
    )
    parser.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=None,
        help="Step at which cosine decay reaches --min-learning-rate; defaults to --max-steps.",
    )

    # Runtime: logging cadence, checkpoint behavior, and execution device.
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Compute dtype for fwd+bwd. 'bf16' enables torch.amp.autocast(bfloat16) on CUDA.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model in torch.compile after construction for fused-kernel speedups.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases if installed.")
    parser.add_argument("--wandb-project", type=str, default="cs336-basics")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV path for step/time-aligned train and validation metrics.",
    )

    return parser.parse_args()


def load_token_array(path: Path, token_dtype: str) -> np.ndarray:
    """Load token data in memory-mapped mode."""
    # .npy files can be memory-mapped directly with metadata preserved.
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
    else:
        # For raw binary dumps, caller must provide the on-disk dtype.
        arr = np.memmap(path, dtype=np.dtype(token_dtype), mode="r")
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D token array at {path}, got shape {arr.shape}")
    return arr


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
    autocast_ctx: Any = None,
) -> float:
    # Switch modules such as normalization/dropout (if added later) to eval behavior.
    model.eval()
    losses: list[float] = []
    for _ in range(num_batches):
        # Draw random subsequences and next-token labels from validation tokens.
        x, y = get_batch(
            dataset=data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        # Teacher forcing: predict y_t from tokens up to position t.
        # Mirror the train-step autocast so eval activations match train numerics.
        if autocast_ctx is None:
            logits = model(x)
            loss = cross_entropy(logits, y)
        else:
            with autocast_ctx():
                logits = model(x)
                loss = cross_entropy(logits, y)
        losses.append(float(loss.detach().cpu().item()))
    # Return to train mode so the caller can continue optimization immediately.
    model.train()
    return sum(losses) / len(losses)


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split model parameters so RMSNorm gains and Embedding weights skip weight decay.

    This follows the standard practice from GPT-2 / Llama / PaLM where decay is
    applied only to matmul (Linear) weights. With weight tying the shared
    embedding/LM-head tensor is encountered first via Embedding and therefore
    correctly placed in the no-decay group.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        for param in module.parameters(recurse=False):
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if isinstance(module, (RMSNorm, Embedding)):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb logging requested but wandb is not installed; continuing without wandb.")
        return None

    # Keep full CLI config in the run for reproducibility.
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)
    return wandb


def maybe_write_metrics_row(
    metrics_csv: Path | None,
    *,
    step: int,
    wallclock_sec: float,
    split: str,
    loss: float,
    tokens_per_sec: float | None,
    lr: float | None = None,
) -> None:
    if metrics_csv is None:
        return

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metrics_csv.exists()
    with metrics_csv.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "wallclock_sec", "split", "loss", "tokens_per_sec", "lr"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "step": step,
                "wallclock_sec": f"{wallclock_sec:.6f}",
                "split": split,
                "loss": f"{loss:.6f}",
                "tokens_per_sec": "" if tokens_per_sec is None else f"{tokens_per_sec:.6f}",
                "lr": "" if lr is None else f"{lr:.8e}",
            }
        )


def main() -> None:
    args = parse_args()
    # Seed both NumPy and PyTorch because data sampling and model init use both.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Lazily load token arrays; mmap keeps memory usage bounded even for huge corpora.
    train_data = load_token_array(args.train_data, args.token_dtype)
    val_data = load_token_array(args.val_data, args.token_dtype) if args.val_data is not None else None

    # Build model directly on the requested device.
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_norm=not args.no_rms_norm,
        post_norm=args.post_norm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
        attn_kernel=args.attn_kernel,
        qk_norm=args.qk_norm,
        tie_embeddings=args.tie_embeddings,
        device=torch.device(args.device),
    )
    # AdamW state (moments, etc.) is needed for faithful resume-from-checkpoint.
    if args.param_group_wd:
        # Two parameter groups: matmul weights decay, norms+embeddings do not.
        optimizer = AdamW(
            build_param_groups(model, args.weight_decay),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )

    step = 0
    if args.resume_from is not None:
        # Resume both model and optimizer so training dynamics continue smoothly.
        step = load_checkpoint(src=args.resume_from, model=model, optimizer=optimizer)
        print(f"Resumed from checkpoint at step={step} ({args.resume_from})")

    # Optionally wrap with torch.compile for fused kernels. We keep `model` as
    # the source of truth for state_dict / checkpointing and route fwd through
    # `forward_model` to avoid the compiled wrapper polluting state_dict keys.
    forward_model = torch.compile(model) if args.compile else model

    # bf16 autocast on CUDA. RMSNorm and cross_entropy already upcast to fp32
    # internally, so loss math stays well-conditioned. bf16 has fp32's exponent
    # range, so no GradScaler is needed.
    use_bf16 = args.dtype == "bf16"
    if use_bf16 and not (args.device.startswith("cuda")):
        print(f"Warning: --dtype bf16 requested but device is {args.device!r}; autocast will be a no-op.")
    autocast_ctx = lambda: torch.amp.autocast(  # noqa: E731 - tiny helper
        device_type="cuda" if args.device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=use_bf16,
    )

    wandb = maybe_init_wandb(args, vars(args))

    # Resolve cosine cycle length (default to full training).
    cosine_cycle = args.cosine_cycle_iters if args.cosine_cycle_iters is not None else args.max_steps

    def current_lr(it: int) -> float:
        if args.lr_schedule == "cosine":
            return lr_cosine_schedule(
                it=it,
                max_learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=cosine_cycle,
            )
        return args.learning_rate

    start_time = time.time()
    last_log_time = time.time()
    while step < args.max_steps:
        step += 1

        # Apply LR for this step (always, even for fixed schedule, so the value is logged).
        cur_lr = current_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        # Sample a fresh random minibatch every step.
        x, y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        # Standard training step: clear grads -> forward -> loss -> backward -> update.
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = forward_model(x)
            loss = cross_entropy(logits, y, z_loss_coef=args.z_loss_coef)
        loss.backward()
        if args.max_grad_norm > 0:
            # Clip global gradient norm to reduce instability from rare spikes.
            gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            dt = max(now - last_log_time, 1e-9)
            last_log_time = now
            # Throughput estimate over the logging window.
            tokens_per_sec = (args.batch_size * args.context_length * args.log_every) / dt
            train_loss = float(loss.detach().cpu().item())
            elapsed_sec = now - start_time
            print(
                f"step={step} train_loss={train_loss:.4f} lr={cur_lr:.3e} "
                f"tokens/s={tokens_per_sec:.1f} elapsed_s={elapsed_sec:.1f}"
            )
            maybe_write_metrics_row(
                args.metrics_csv,
                step=step,
                wallclock_sec=elapsed_sec,
                split="train",
                loss=train_loss,
                tokens_per_sec=tokens_per_sec,
                lr=cur_lr,
            )
            if wandb is not None:
                wandb.log(
                    {
                        "step": step,
                        "time/elapsed_sec": elapsed_sec,
                        "train/loss": train_loss,
                        "perf/tokens_per_sec": tokens_per_sec,
                        "train/lr": cur_lr,
                    },
                    step=step,
                )

        if val_data is not None and step % args.val_every == 0:
            # Free PyTorch's caching pool before validation so the fp32 logits
            # tensors materialized inside cross_entropy don't OOM against
            # leftover training-step fragments. Cheap on CUDA, no-op on CPU.
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            # Validation uses random batches and no gradients for speed.
            val_loss = evaluate(
                model=forward_model,
                data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.val_batches,
                autocast_ctx=autocast_ctx,
            )
            elapsed_sec = time.time() - start_time
            print(f"step={step} val_loss={val_loss:.4f} elapsed_s={elapsed_sec:.1f}")
            maybe_write_metrics_row(
                args.metrics_csv,
                step=step,
                wallclock_sec=elapsed_sec,
                split="val",
                loss=val_loss,
                tokens_per_sec=None,
                lr=cur_lr,
            )
            if wandb is not None:
                wandb.log({"step": step, "time/elapsed_sec": elapsed_sec, "val/loss": val_loss}, step=step)

        if args.checkpoint_path is not None and step % args.checkpoint_every == 0:
            # Ensure parent directory exists before writing checkpoint file.
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path} at step={step}")

    if args.checkpoint_path is not None:
        # Final snapshot even if max_steps is not aligned with checkpoint cadence.
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=args.checkpoint_path)
        print(f"Saved final checkpoint to {args.checkpoint_path} at step={step}")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    # Helpful default for local experimentation.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
