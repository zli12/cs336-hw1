#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import psutil

from cs336_basics.bpe_train_multi import BPETrainerMulti


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(codepoint) for codepoint in cs]))


def _encode_token_bytes(token_bytes: bytes, encoder: dict[int, str]) -> str:
    return "".join(encoder[b] for b in token_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyStories BPE and serialize vocab/merges.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to TinyStories train file.",
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
        "--num-processes",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of processes for pretoken counting.",
    )
    parser.add_argument(
        "--out-vocab",
        type=Path,
        default=Path("data/tinystories-bpe-10k-vocab.json"),
        help="Output path for serialized vocab JSON.",
    )
    parser.add_argument(
        "--out-merges",
        type=Path,
        default=Path("data/tinystories-bpe-10k-merges.txt"),
        help="Output path for serialized merges text.",
    )
    args = parser.parse_args()

    proc = psutil.Process(os.getpid())
    peak_rss_sum = 0
    peak_uss_sum = 0
    running = True

    def sample_memory() -> None:
        nonlocal peak_rss_sum, peak_uss_sum, running
        while running:
            rss_sum = 0
            uss_sum = 0
            processes = [proc] + proc.children(recursive=True)
            for p in processes:
                try:
                    rss_sum += p.memory_info().rss
                    full_info = p.memory_full_info()
                    uss_sum += getattr(full_info, "uss", 0)
                except psutil.Error:
                    pass
            peak_rss_sum = max(peak_rss_sum, rss_sum)
            peak_uss_sum = max(peak_uss_sum, uss_sum)
            time.sleep(0.05)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    start = time.perf_counter()
    vocab, merges = BPETrainerMulti.train(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=max(1, args.num_processes),
    )
    elapsed = time.perf_counter() - start

    running = False
    sampler.join(timeout=1.0)

    encoder = _bytes_to_unicode()
    vocab_json = {_encode_token_bytes(token, encoder): idx for idx, token in vocab.items()}
    args.out_vocab.write_text(json.dumps(vocab_json, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_merges.write_text(
        "\n".join(
            f"{_encode_token_bytes(left, encoder)} {_encode_token_bytes(right, encoder)}"
            for left, right in merges
        )
        + "\n",
        encoding="utf-8",
    )

    longest_token = max(vocab.values(), key=len)
    print(f"num_processes={args.num_processes}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"elapsed_hours={elapsed / 3600:.6f}")
    print(f"peak_rss_sum_gib={peak_rss_sum / (1024**3):.3f}")
    print(f"peak_uss_sum_gib={peak_uss_sum / (1024**3):.3f}")
    print(f"vocab_size={len(vocab)} merges={len(merges)}")
    print(f"longest_token_len_bytes={len(longest_token)}")
    print(f"longest_token_preview={longest_token!r}")
    print(f"wrote_vocab={args.out_vocab}")
    print(f"wrote_merges={args.out_merges}")


if __name__ == "__main__":
    main()
