#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import tempfile

import regex as re

from cs336_basics.bpe_trainer import BPETrainer


def _format_token(token: bytes) -> str:
    decoded = token.decode("utf-8", errors="replace")
    return f"{decoded!r}"


def _format_sequence(sequence: list[bytes]) -> str:
    return "[" + " ".join(_format_token(token) for token in sequence) + "]"


def _merge_once(sequence: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    left, right = pair
    merged: list[bytes] = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and sequence[i] == left and sequence[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged


def _weighted_pair_counts(
    tokenized_pretokens: list[list[bytes]],
    frequencies: list[int],
) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for sequence, freq in zip(tokenized_pretokens, frequencies, strict=False):
        local = Counter(zip(sequence, sequence[1:], strict=False))
        for pair, pair_count in local.items():
            counts[pair] += pair_count * freq
    return counts


def _bar(value: int, max_value: int, width: int = 24) -> str:
    if max_value <= 0:
        return ""
    bar_len = max(1, int(round((value / max_value) * width)))
    return "#" * bar_len


def _pair_index_map(
    tokenized_pretokens: list[list[bytes]],
) -> dict[tuple[bytes, bytes], set[int]]:
    index_map: dict[tuple[bytes, bytes], set[int]] = {}
    for seq_idx, sequence in enumerate(tokenized_pretokens):
        if len(sequence) < 2:
            continue
        for pair in set(zip(sequence, sequence[1:], strict=False)):
            members = index_map.setdefault(pair, set())
            members.add(seq_idx)
    return index_map


def _format_pair(pair: tuple[bytes, bytes]) -> str:
    left, right = pair
    return f"({_format_token(left)} + {_format_token(right)})"


def _top_pairs_text(pair_counts: Counter[tuple[bytes, bytes]], limit: int) -> list[str]:
    top = pair_counts.most_common(max(1, limit))
    return [f"{rank:>2}. {_format_pair(pair)} -> {count}" for rank, (pair, count) in enumerate(top, start=1)]


def _collect_pretokens(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    segments = [text]
    if special_tokens:
        special_split_pattern = "|".join(re.escape(token) for token in special_tokens)
        segments = re.split(special_split_pattern, text)

    pretoken_frequencies: Counter[tuple[int, ...]] = Counter()
    for segment in segments:
        for match in BPETrainer.PATTERN.finditer(segment):
            pretoken = match.group(0)
            pretoken_frequencies[tuple(pretoken.encode("utf-8"))] += 1
    return pretoken_frequencies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize BPE training merges on a tiny corpus."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="the theater is in the theme park",
        help="Tiny corpus text used for the visualization.",
    )
    parser.add_argument(
        "--num-merges",
        type=int,
        default=12,
        help="How many merge steps to visualize.",
    )
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=[],
        help="Special token to preserve as hard boundaries.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of high-frequency pretokens to show each step.",
    )
    parser.add_argument(
        "--show-internals",
        action="store_true",
        help="Print pair_counts and pair->sequence lookup snapshots each merge step.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=8,
        help="Number of highest-count pairs to print in --show-internals mode.",
    )
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(args.text)
        tmp_path = Path(tmp.name)

    try:
        target_vocab_size = 256 + max(0, args.num_merges) + len(args.special_tokens)
        _, merges = BPETrainer.train(
            input_path=tmp_path,
            vocab_size=target_vocab_size,
            special_tokens=args.special_tokens,
        )
        merges = merges[: args.num_merges]
    finally:
        tmp_path.unlink(missing_ok=True)

    pretoken_frequencies = _collect_pretokens(args.text, args.special_tokens)
    sorted_pretokens = sorted(
        pretoken_frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    display = sorted_pretokens[: max(1, args.examples)]

    pretoken_labels = [bytes(token).decode("utf-8", errors="replace") for token, _ in display]
    display_tokenized = [[bytes([b]) for b in token] for token, _ in display]
    display_freqs = [freq for _, freq in display]

    all_tokenized = [[bytes([b]) for b in token] for token, _ in sorted_pretokens]
    all_freqs = [freq for _, freq in sorted_pretokens]
    all_labels = [bytes(token).decode("utf-8", errors="replace") for token, _ in sorted_pretokens]

    print("BPE merge visualization")
    print(f"corpus={args.text!r}")
    print(f"special_tokens={args.special_tokens}")
    print(f"shown_pretokens={list(zip(pretoken_labels, display_freqs, strict=False))}")
    print("-" * 88)

    if not merges:
        print("No merges were produced for this corpus/target size.")
        return

    for step, pair in enumerate(merges, start=1):
        pair_counts_before = _weighted_pair_counts(all_tokenized, all_freqs)
        pair_index_before = _pair_index_map(all_tokenized)
        pair_count = pair_counts_before.get(pair, 0)
        max_pair_count = max(pair_counts_before.values(), default=0)

        left, right = pair
        merged = left + right
        print(
            f"step {step:>2}: merge {_format_token(left)} + {_format_token(right)} -> "
            f"{_format_token(merged)}  count={pair_count:>4} {_bar(pair_count, max_pair_count)}"
        )
        if args.show_internals:
            print("   pair_counts (top before merge):")
            for line in _top_pairs_text(pair_counts_before, args.top_pairs):
                print(f"     {line}")
            affected = sorted(pair_index_before.get(pair, set()))
            print(f"   affected_sequence_indices={affected}")
            if affected:
                preview = ", ".join(
                    f"{idx}:{all_labels[idx]!r} x{all_freqs[idx]}"
                    for idx in affected[:8]
                )
                suffix = " ..." if len(affected) > 8 else ""
                print(f"   affected_sequences={preview}{suffix}")

        for i in range(len(all_tokenized)):
            all_tokenized[i] = _merge_once(all_tokenized[i], pair)

        any_displayed = False
        for i, (label, before) in enumerate(zip(pretoken_labels, display_tokenized, strict=False)):
            after = _merge_once(before, pair)
            if after != before:
                any_displayed = True
                print(
                    f"   {label!r:>12} x{display_freqs[i]:<3}: "
                    f"{_format_sequence(before)} -> {_format_sequence(after)}"
                )
            display_tokenized[i] = after
        if not any_displayed:
            print("   (no shown pretoken changed on this step)")
        if args.show_internals:
            pair_counts_after = _weighted_pair_counts(all_tokenized, all_freqs)
            print("   pair_count deltas (for current best pair):")
            print(
                f"     {_format_pair(pair)}: {pair_counts_before.get(pair, 0)} -> "
                f"{pair_counts_after.get(pair, 0)}"
            )
            print("   pair_counts (top after merge):")
            for line in _top_pairs_text(pair_counts_after, args.top_pairs):
                print(f"     {line}")
        print("-" * 88)


if __name__ == "__main__":
    main()
