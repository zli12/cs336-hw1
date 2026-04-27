from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.nn import cross_entropy
from cs336_basics.optim import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.transformer import TransformerLM


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
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(float(loss.detach().cpu().item()))
    # Return to train mode so the caller can continue optimization immediately.
    model.train()
    return sum(losses) / len(losses)


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
        device=torch.device(args.device),
    )
    # AdamW state (moments, etc.) is needed for faithful resume-from-checkpoint.
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
        logits = model(x)
        loss = cross_entropy(logits, y)
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
            # Validation uses random batches and no gradients for speed.
            val_loss = evaluate(
                model=model,
                data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.val_batches,
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
