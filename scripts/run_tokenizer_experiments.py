#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def sample_documents(
    path: Path,
    n: int,
    delimiter: str,
    seed: int,
    read_chunk_chars: int = 4 * 1024 * 1024,
) -> list[str]:
    """Reservoir-sample n delimiter-separated documents from a large text file."""
    rng = random.Random(seed)
    samples: list[str] = []
    seen = 0
    buffer = ""

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(read_chunk_chars)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(delimiter)
            buffer = parts[-1]
            for doc in parts[:-1]:
                if not doc:
                    continue
                seen += 1
                if len(samples) < n:
                    samples.append(doc)
                else:
                    j = rng.randrange(seen)
                    if j < n:
                        samples[j] = doc

    # Last partial segment (if file doesn't end with delimiter) is also a document.
    if buffer:
        seen += 1
        if len(samples) < n:
            samples.append(buffer)
        else:
            j = rng.randrange(seen)
            if j < n:
                samples[j] = buffer

    return samples


def compression_ratio_bytes_per_token(tokenizer: Tokenizer, docs: list[str]) -> float:
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in docs)
    if total_tokens == 0:
        return float("inf")
    return total_bytes / total_tokens


def measure_throughput_bytes_per_second(
    tokenizer: Tokenizer,
    path: Path,
    max_bytes: int = 256 * 1024 * 1024,
    read_chunk_chars: int = 2 * 1024 * 1024,
) -> tuple[float, int, int]:
    """Measure approximate tokenizer throughput by encoding up to max_bytes from a file."""
    bytes_processed = 0
    tokens_processed = 0

    start = time.perf_counter()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while bytes_processed < max_bytes:
            chunk = f.read(read_chunk_chars)
            if not chunk:
                break
            chunk_bytes = len(chunk.encode("utf-8"))
            if bytes_processed + chunk_bytes > max_bytes:
                # Trim by chars for a rough cap.
                remaining = max_bytes - bytes_processed
                if remaining <= 0:
                    break
                chunk = chunk.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
                chunk_bytes = len(chunk.encode("utf-8"))
            token_ids = tokenizer.encode(chunk)
            bytes_processed += chunk_bytes
            tokens_processed += len(token_ids)
    elapsed = time.perf_counter() - start

    throughput = bytes_processed / elapsed if elapsed > 0 else 0.0
    return throughput, bytes_processed, tokens_processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokenizer experiments (parts a/b/c).")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of sampled documents per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--special-token",
        default="<|endoftext|>",
        help="Document delimiter / special token used in datasets.",
    )
    parser.add_argument("--tinystories-path", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--owt-path", type=Path, default=Path("data/owt_train.txt"))

    parser.add_argument("--tiny-vocab", type=Path, default=Path("experiments/tinystories-bpe-10k-vocab.json"))
    parser.add_argument("--tiny-merges", type=Path, default=Path("experiments/tinystories-bpe-10k-merges.txt"))
    parser.add_argument("--owt-vocab", type=Path, default=Path("experiments/owt-bpe-32k-vocab.json"))
    parser.add_argument("--owt-merges", type=Path, default=Path("experiments/owt-bpe-32k-merges.txt"))

    parser.add_argument(
        "--throughput-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Bytes to process when estimating throughput.",
    )
    parser.add_argument(
        "--pile-size-bytes",
        type=int,
        default=825 * (1024**3),
        help="Total bytes for Pile-size time estimate (default: 825 GiB).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("experiments/tokenizer-experiments-results.json"),
        help="Where to write structured experiment results.",
    )
    args = parser.parse_args()

    tiny_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.tiny_vocab),
        merges_filepath=str(args.tiny_merges),
        special_tokens=[args.special_token],
    )
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.owt_vocab),
        merges_filepath=str(args.owt_merges),
        special_tokens=[args.special_token],
    )

    tiny_docs = sample_documents(
        path=args.tinystories_path,
        n=args.sample_size,
        delimiter=args.special_token,
        seed=args.seed,
    )
    owt_docs = sample_documents(
        path=args.owt_path,
        n=args.sample_size,
        delimiter=args.special_token,
        seed=args.seed + 1,
    )

    # (a) Compression for each tokenizer on its own dataset sample.
    tiny_ratio = compression_ratio_bytes_per_token(tiny_tokenizer, tiny_docs)
    owt_ratio = compression_ratio_bytes_per_token(owt_tokenizer, owt_docs)

    # (b) Tokenize OWT sample with TinyStories tokenizer.
    owt_with_tiny_ratio = compression_ratio_bytes_per_token(tiny_tokenizer, owt_docs)

    # (c) Throughput estimate (use OWT tokenizer over OWT data).
    throughput_bps, measured_bytes, measured_tokens = measure_throughput_bytes_per_second(
        tokenizer=owt_tokenizer,
        path=args.owt_path,
        max_bytes=args.throughput_bytes,
    )
    est_pile_seconds = args.pile_size_bytes / throughput_bps if throughput_bps > 0 else float("inf")

    results = {
        "sample_size": args.sample_size,
        "special_token": args.special_token,
        "compression": {
            "tinystories_tokenizer_on_tinystories_sample_bytes_per_token": tiny_ratio,
            "owt_tokenizer_on_owt_sample_bytes_per_token": owt_ratio,
            "tinystories_tokenizer_on_owt_sample_bytes_per_token": owt_with_tiny_ratio,
        },
        "throughput": {
            "bytes_per_second": throughput_bps,
            "measured_bytes": measured_bytes,
            "measured_tokens": measured_tokens,
            "pile_estimated_seconds_for_825GiB": est_pile_seconds,
            "pile_estimated_hours_for_825GiB": est_pile_seconds / 3600,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"sample_size={args.sample_size}")
    print(f"tiny_ratio_bytes_per_token={tiny_ratio:.6f}")
    print(f"owt_ratio_bytes_per_token={owt_ratio:.6f}")
    print(f"owt_with_tiny_ratio_bytes_per_token={owt_with_tiny_ratio:.6f}")
    print(f"throughput_bytes_per_second={throughput_bps:.2f}")
    print(f"pile_estimated_hours={est_pile_seconds / 3600:.3f}")
    print(f"wrote_results={args.out_json}")


if __name__ == "__main__":
    main()
