#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import pstats
import time

from cs336_basics.bpe_trainer import BPETrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile serial BPE training with cProfile.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to training corpus.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Target vocabulary size.")
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("data/profile-bpe-serial.pstats"),
        help="Binary cProfile stats output path.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/profile-bpe-serial.txt"),
        help="Human-readable cProfile report output path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=80,
        help="Number of functions to print in the report.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["cumulative", "time", "calls", "pcalls", "name", "filename", "line", "nfl", "stdname"],
        default="cumulative",
        help="Sort key for profile report.",
    )
    args = parser.parse_args()

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    vocab, merges = BPETrainer.train(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    profiler.disable()
    elapsed = time.perf_counter() - start

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(args.stats_out))

    with args.report_out.open("w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats(args.sort_by)
        f.write(
            f"elapsed_seconds={elapsed:.3f}\n"
            f"input={args.input}\n"
            f"vocab_size={len(vocab)} merges={len(merges)}\n"
            f"sort_by={args.sort_by} top_n={args.top_n}\n\n"
        )
        stats.print_stats(args.top_n)

    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"vocab_size={len(vocab)} merges={len(merges)}")
    print(f"wrote_stats={args.stats_out}")
    print(f"wrote_report={args.report_out}")


if __name__ == "__main__":
    main()
