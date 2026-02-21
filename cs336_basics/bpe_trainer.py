from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import regex as re

from cs336_basics.max_pair_heap import MaxPairHeap


class BPETrainer:
    """Train a byte-level BPE vocabulary and merge list."""

    PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    @classmethod
    def train(
        cls,
        input_path: str | Path,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train byte-level BPE merges and return (id->token vocab, merges)."""
        # 1) Read the full training corpus.
        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        # 2) Split around special tokens so we never merge across those boundaries.
        segments = [text]
        if special_tokens:
            # Escape each literal token so regex metacharacters (like "|") are treated literally.
            special_split_pattern = "|".join(re.escape(token) for token in special_tokens)
            segments = re.split(special_split_pattern, text)

        # 3) Pretokenize each segment with GPT-2 regex and count byte-sequence frequencies.
        #    Each key is a unique pretoken represented as tuple[int] of UTF-8 byte values.
        pretoken_frequencies: Counter[tuple[int, ...]] = Counter()
        for segment in segments:
            for match in cls.PATTERN.finditer(segment):
                pretoken = match.group(0)
                pretoken_frequencies[tuple(pretoken.encode("utf-8"))] += 1

        # 4) Initialize byte-level base vocabulary (all 256 possible byte tokens).
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        vocab_set = set(vocab)

        # Reserve space for special tokens that are not already present.
        # We train merges only until this reduced target size.
        encoded_special_tokens = [token.encode("utf-8") for token in special_tokens]
        special_tokens_to_add = [token for token in encoded_special_tokens if token not in vocab_set]
        target_vocab_size_without_specials = max(256, vocab_size - len(special_tokens_to_add))

        # Convert each unique pretoken to a mutable token-id list for in-place merge updates.
        token_sequences = [list(pretoken_tuple) for pretoken_tuple in pretoken_frequencies]
        # Keep aligned list of frequencies so token_sequences[i] has frequency sequence_frequencies[i].
        sequence_frequencies = [
            pretoken_frequencies[tuple(token_sequence)] for token_sequence in token_sequences
        ]

        # pair_counts[(a, b)] = weighted count across all token sequences.
        # pair_to_sequence_indices[(a, b)] = which token-sequence entries currently contain this pair.
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_sequence_indices: dict[tuple[int, int], set[int]] = defaultdict(set)

        # 5) Build initial adjacent-pair statistics.
        for sequence_index, token_sequence in enumerate(token_sequences):
            if len(token_sequence) < 2:
                continue
            # Count all pairs of consecutive tokens in this sequence.
            local_pair_counts = Counter(zip(token_sequence, token_sequence[1:], strict=False))
            sequence_frequency = sequence_frequencies[sequence_index]
            for pair, pair_occurrences_in_sequence in local_pair_counts.items():
                # Update global pair_counts with weighted count (occurrences * sequence frequency).
                pair_counts[pair] += pair_occurrences_in_sequence * sequence_frequency
                # Track which sequences contain this pair for efficient updates during merges.
                pair_to_sequence_indices[pair].add(sequence_index)

        merges: list[tuple[bytes, bytes]] = []
        max_pair_heap = MaxPairHeap(pair_counts=pair_counts, vocab=vocab)

        # 6) Repeatedly merge the highest-frequency pair until target vocab size.
        while len(vocab) < target_vocab_size_without_specials and pair_counts:
            best_pair_and_count = max_pair_heap.pop_max()
            if best_pair_and_count is None:
                break
            best_pair, best_count = best_pair_and_count
            if best_count <= 0:
                break

            left_id, right_id = best_pair
            merges.append((vocab[left_id], vocab[right_id]))
            merged_token = vocab[left_id] + vocab[right_id]
            merged_id = len(vocab)
            vocab.append(merged_token)
            vocab_set.add(merged_token)

            # Only sequences that currently contain best_pair can change after this merge.
            affected_sequence_indices = list(pair_to_sequence_indices.get(best_pair, set()))

            for sequence_index in affected_sequence_indices:
                old_token_sequence = token_sequences[sequence_index]
                if len(old_token_sequence) < 2:
                    continue

                old_sequence_pair_counts = Counter(
                    zip(old_token_sequence, old_token_sequence[1:], strict=False)
                )
                if best_pair not in old_sequence_pair_counts:
                    continue

                # Replace all non-overlapping occurrences of best_pair with merged_id.
                merged_token_sequence: list[int] = []
                token_index = 0
                while token_index < len(old_token_sequence):
                    if (
                        token_index + 1 < len(old_token_sequence)
                        and old_token_sequence[token_index] == left_id
                        and old_token_sequence[token_index + 1] == right_id
                    ):
                        merged_token_sequence.append(merged_id)
                        token_index += 2
                    else:
                        merged_token_sequence.append(old_token_sequence[token_index])
                        token_index += 1

                if merged_token_sequence == old_token_sequence:
                    continue

                new_sequence_pair_counts = Counter(
                    zip(merged_token_sequence, merged_token_sequence[1:], strict=False)
                )
                sequence_frequency = sequence_frequencies[sequence_index]

                # Incrementally update global pair stats instead of recounting all sequences.
                for pair, old_count in old_sequence_pair_counts.items():
                    new_count = new_sequence_pair_counts.get(pair, 0)
                    if new_count == old_count:
                        continue
                    delta = (new_count - old_count) * sequence_frequency
                    updated = pair_counts.get(pair, 0) + delta
                    if updated <= 0:
                        pair_counts.pop(pair, None)
                    else:
                        pair_counts[pair] = updated
                        max_pair_heap.update(pair=pair, count=updated)

                    if new_count == 0:
                        members = pair_to_sequence_indices.get(pair)
                        if members is not None:
                            members.discard(sequence_index)
                            if not members:
                                pair_to_sequence_indices.pop(pair, None)
                    else:
                        pair_to_sequence_indices[pair].add(sequence_index)

                # Handle pairs that are newly introduced by the merge in this sequence.
                for pair, new_count in new_sequence_pair_counts.items():
                    if pair in old_sequence_pair_counts:
                        continue
                    pair_counts[pair] += new_count * sequence_frequency
                    pair_to_sequence_indices[pair].add(sequence_index)
                    max_pair_heap.update(pair=pair, count=pair_counts[pair])

                token_sequences[sequence_index] = merged_token_sequence

            # best_pair is fully merged in all affected words and should not stay active.
            pair_counts.pop(best_pair, None)
            pair_to_sequence_indices.pop(best_pair, None)

        # 7) Append special tokens (if missing), preserving input order.
        for special_token in encoded_special_tokens:
            if len(vocab) >= vocab_size:
                break
            if special_token in vocab_set:
                continue
            vocab.append(special_token)
            vocab_set.add(special_token)

        # Return id->token mapping and merge list in creation order.
        return {idx: token for idx, token in enumerate(vocab)}, merges
