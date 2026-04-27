"""Plot learning curves from train_lm.py metrics CSVs.

Supports three modes via --kind:

- lr-sweep: overlay val loss vs step, one curve per LR. Optional fixed-LR baseline
  curve in a contrasting style.
- batch-sweep: 2-panel figure (val loss vs step, val loss vs wallclock seconds),
  one curve per batch size.
- edge-of-stability: overlay train loss vs step including divergent runs; uses log
  y-axis so divergent (large) losses do not crush the surviving curves.

Each CSV is expected to have columns produced by scripts/train_lm.py:
  step,wallclock_sec,split,loss,tokens_per_sec[,lr]
(The lr column is optional - older runs without it are still readable.)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Run:
    """One training run, decomposed into train and val rows."""

    label: str
    train_step: list[int]
    train_wall: list[float]
    train_loss: list[float]
    val_step: list[int]
    val_wall: list[float]
    val_loss: list[float]


def load_run(csv_path: Path, label: str) -> Run:
    """Read a metrics CSV and split rows by `split` column."""
    train_step: list[int] = []
    train_wall: list[float] = []
    train_loss: list[float] = []
    val_step: list[int] = []
    val_wall: list[float] = []
    val_loss: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            wall = float(row["wallclock_sec"])
            loss = float(row["loss"])
            split = row["split"]
            if split == "train":
                train_step.append(step)
                train_wall.append(wall)
                train_loss.append(loss)
            elif split == "val":
                val_step.append(step)
                val_wall.append(wall)
                val_loss.append(loss)
    return Run(
        label=label,
        train_step=train_step,
        train_wall=train_wall,
        train_loss=train_loss,
        val_step=val_step,
        val_wall=val_wall,
        val_loss=val_loss,
    )


def _baseline_label(p: Path) -> str:
    """Fallback label = file stem."""
    return p.stem


def parse_inputs(inputs: list[str], labels: list[str] | None) -> list[tuple[Path, str]]:
    """Pair each --inputs path with a label (--labels takes precedence)."""
    paths = [Path(p) for p in inputs]
    if labels is None:
        return [(p, _baseline_label(p)) for p in paths]
    if len(labels) != len(paths):
        raise SystemExit(f"--labels length ({len(labels)}) must match --inputs length ({len(paths)})")
    return list(zip(paths, labels, strict=True))


def plot_lr_sweep(runs: list[Run], out: Path, title: str | None) -> None:
    """One panel: val loss vs step, one curve per run."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for run in runs:
        if not run.val_step:
            continue
        ax.plot(run.val_step, run.val_loss, marker="o", markersize=3, linewidth=1.2, label=run.label)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_batch_sweep(runs: list[Run], out: Path, title: str | None) -> None:
    """Two panels: val loss vs step (left), val loss vs wallclock (right)."""
    fig, (ax_step, ax_wall) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for run in runs:
        if not run.val_step:
            continue
        ax_step.plot(run.val_step, run.val_loss, marker="o", markersize=3, linewidth=1.2, label=run.label)
        ax_wall.plot(run.val_wall, run.val_loss, marker="o", markersize=3, linewidth=1.2, label=run.label)
    ax_step.set_xlabel("step")
    ax_step.set_ylabel("validation loss")
    ax_step.grid(True, alpha=0.3)
    ax_step.legend(fontsize=8, loc="best")
    ax_wall.set_xlabel("wallclock seconds")
    ax_wall.set_ylabel("validation loss")
    ax_wall.grid(True, alpha=0.3)
    ax_wall.legend(fontsize=8, loc="best")
    if title:
        fig.suptitle(title)
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_edge_of_stability(runs: list[Run], out: Path, title: str | None) -> None:
    """One panel: train loss vs step, log y-axis so divergent runs are still visible."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for run in runs:
        if not run.train_step:
            continue
        # Replace any non-positive (shouldn't happen) and inf/nan with a large sentinel
        # so log-scale plotting still shows the divergence trend.
        clean = []
        for v in run.train_loss:
            if v != v or v == float("inf") or v <= 0:
                clean.append(1e3)
            else:
                clean.append(v)
        ax.plot(run.train_step, clean, linewidth=1.2, label=run.label)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (log scale)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out, format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=["lr-sweep", "batch-sweep", "edge-of-stability"])
    p.add_argument("--inputs", nargs="+", required=True, help="One or more metrics CSV paths.")
    p.add_argument("--labels", nargs="+", default=None, help="Labels matching --inputs (default: file stems).")
    p.add_argument("--out", type=Path, required=True, help="Output SVG path.")
    p.add_argument("--title", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs = parse_inputs(args.inputs, args.labels)
    runs = [load_run(p, lbl) for (p, lbl) in pairs]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "lr-sweep":
        plot_lr_sweep(runs, args.out, args.title)
    elif args.kind == "batch-sweep":
        plot_batch_sweep(runs, args.out, args.title)
    elif args.kind == "edge-of-stability":
        plot_edge_of_stability(runs, args.out, args.title)
    else:
        raise SystemExit(f"Unknown --kind: {args.kind}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
