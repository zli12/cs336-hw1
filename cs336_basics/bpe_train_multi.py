from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import BinaryIO

import regex as re

from cs336_basics.max_pair_heap import MaxPairHeap


def _count_pretokens_in_chunk(
    chunk_text: str,
    special_tokens: tuple[str, ...],
    pattern_text: str,
) -> Counter[tuple[int, ...]]:
    """Count pretoken byte-sequence frequencies in one text chunk."""
    pattern = re.compile(pattern_text)
    pretoken_frequencies: Counter[tuple[int, ...]] = Counter()

    segments = [chunk_text]
    if special_tokens:
        # Drop special tokens from normal pretokenization by splitting around them.
        special_split_pattern = "|".join(re.escape(token) for token in special_tokens)
        segments = re.split(special_split_pattern, chunk_text)

    for segment in segments:
        for match in pattern.finditer(segment):
            pretoken = match.group(0)
            pretoken_frequencies[tuple(pretoken.encode("utf-8"))] += 1
    return pretoken_frequencies


class BPETrainerMulti:
    """Train a byte-level BPE tokenizer with optional multiprocessing pretokenization."""

    PATTERN_TEXT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """Find chunk boundaries by snapping to the next special-token start."""
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if desired_num_chunks <= 1:
            return [0, file_size]

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for boundary_index in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[boundary_index]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    chunk_boundaries[boundary_index] = file_size
                    break

                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    # Snap each boundary forward so chunks align on special-token starts.
                    chunk_boundaries[boundary_index] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Remove duplicates in case multiple boundaries snap to the same location.
        return sorted(set(chunk_boundaries))

    @classmethod
    def train(
        cls,
        input_path: str | Path,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 1,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train byte-level BPE merges and return (id->token vocab, merges)."""
        # Phase 1: pretokenize corpus text and count each unique byte sequence.
        pretoken_frequencies: Counter[tuple[int, ...]] = Counter()
        requested_processes = max(1, int(num_processes))
        # Multiprocessing is only safe here when we can align chunk splits to a sentinel token.
        should_parallelize = requested_processes > 1 and len(special_tokens) > 0

        if should_parallelize:
            # We treat special_tokens[0] as a split sentinel so chunk boundaries avoid slicing through it.
            split_token_bytes = special_tokens[0].encode("utf-8")
            with open(input_path, "rb") as binary_file:
                boundaries = cls._find_chunk_boundaries(
                    file=binary_file,
                    desired_num_chunks=requested_processes,
                    split_special_token=split_token_bytes,
                )
                # Materialize chunk payloads before forking workers so each process gets plain text input.
                chunk_texts: list[str] = []
                for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
                    binary_file.seek(start)
                    # Ignore decode errors at chunk edges; text boundaries are approximate.
                    chunk_texts.append(binary_file.read(end - start).decode("utf-8", errors="ignore"))

            with ProcessPoolExecutor(max_workers=requested_processes) as executor:
                # executor.map preserves input order, but ordering is irrelevant because we only sum counters.
                # We pass per-chunk text plus repeated immutable args (special tokens and regex pattern).
                future_results = executor.map(
                    _count_pretokens_in_chunk,
                    chunk_texts,
                    [tuple(special_tokens)] * len(chunk_texts),
                    [cls.PATTERN_TEXT] * len(chunk_texts),
                )
                for chunk_counter in future_results:
                    # Reduce step: merge worker-local counters into one global corpus counter.
                    pretoken_frequencies.update(chunk_counter)
        else:
            text = Path(input_path).read_text(encoding="utf-8")
            pretoken_frequencies = _count_pretokens_in_chunk(
                chunk_text=text,
                special_tokens=tuple(special_tokens),
                pattern_text=cls.PATTERN_TEXT,
            )

        # Phase 2: initialize byte vocabulary (0..255 are always present in byte-level BPE).
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        vocab_set = set(vocab)

        encoded_special_tokens = [token.encode("utf-8") for token in special_tokens]
        special_tokens_to_add = [token for token in encoded_special_tokens if token not in vocab_set]
        # Reserve room so learned merges fill the non-special portion of the vocabulary.
        target_vocab_size_without_specials = max(256, vocab_size - len(special_tokens_to_add))

        # Each unique pretoken becomes one token-id sequence; frequency is tracked separately.
        pretoken_items = list(pretoken_frequencies.items())
        token_sequences = [list(pretoken_tuple) for pretoken_tuple, _ in pretoken_items]
        sequence_frequencies = [frequency for _, frequency in pretoken_items]

        # Phase 3: build global adjacent-pair statistics across all sequences.
        pair_counts: Counter[tuple[int, int]] = Counter()
        # Inverted index: pair -> sequences currently containing that pair.
        pair_to_sequence_indices: dict[tuple[int, int], set[int]] = defaultdict(set)

        for sequence_index, token_sequence in enumerate(token_sequences):
            if len(token_sequence) < 2:
                continue
            local_pair_counts = Counter(zip(token_sequence, token_sequence[1:]))
            sequence_frequency = sequence_frequencies[sequence_index]
            for pair, pair_occurrences_in_sequence in local_pair_counts.items():
                # Weight local pair count by how often this whole sequence appears in the corpus.
                pair_counts[pair] += pair_occurrences_in_sequence * sequence_frequency
                pair_to_sequence_indices[pair].add(sequence_index)

        merges: list[tuple[bytes, bytes]] = []
        max_pair_heap = MaxPairHeap(pair_counts=pair_counts, vocab=vocab)

        # Phase 4: repeatedly merge the most frequent pair and update statistics incrementally.
        # Toy example (ids shown symbolically):
        #   old sequence: [A, B, A, B, C]
        #   best_pair: (A, B) -> M
        #   new sequence: [M, M, C]
        # Only pairs touching A/B positions can change, so we update counts/indexes for affected
        # sequences instead of recomputing pair stats for the whole corpus each iteration.
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

            # Only sequences containing best_pair can change; all others keep identical pair counts.
            affected_sequence_indices = list(pair_to_sequence_indices.get(best_pair, set()))

            for sequence_index in affected_sequence_indices:
                old_token_sequence = token_sequences[sequence_index]
                if len(old_token_sequence) < 2:
                    continue

                old_sequence_pair_counts = Counter(zip(old_token_sequence, old_token_sequence[1:]))
                if best_pair not in old_sequence_pair_counts:
                    continue

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

                # If no replacement happened, skip count/index updates for this sequence.
                if merged_token_sequence == old_token_sequence:
                    continue

                new_sequence_pair_counts = Counter(zip(merged_token_sequence, merged_token_sequence[1:]))
                sequence_frequency = sequence_frequencies[sequence_index]

                for pair, old_count in old_sequence_pair_counts.items():
                    new_count = new_sequence_pair_counts.get(pair, 0)
                    if new_count == old_count:
                        continue
                    # Update global counts by only this sequence's local change.
                    delta = (new_count - old_count) * sequence_frequency
                    updated = pair_counts.get(pair, 0) + delta
                    if updated <= 0:
                        pair_counts.pop(pair, None)
                    else:
                        pair_counts[pair] = updated
                        max_pair_heap.update(pair=pair, count=updated)

                    if new_count == 0:
                        # Pair disappeared from this sequence: remove membership from inverted index.
                        members = pair_to_sequence_indices.get(pair)
                        if members is not None:
                            members.discard(sequence_index)
                            if not members:
                                pair_to_sequence_indices.pop(pair, None)
                    else:
                        # Pair still exists (possibly with changed count): keep membership.
                        pair_to_sequence_indices[pair].add(sequence_index)

                for pair, new_count in new_sequence_pair_counts.items():
                    if pair in old_sequence_pair_counts:
                        continue
                    # Brand-new pair introduced by this merge in this sequence.
                    pair_counts[pair] += new_count * sequence_frequency
                    pair_to_sequence_indices[pair].add(sequence_index)
                    max_pair_heap.update(pair=pair, count=pair_counts[pair])

                token_sequences[sequence_index] = merged_token_sequence

            # best_pair no longer exists after merging; remove stale bookkeeping entries.
            pair_counts.pop(best_pair, None)
            pair_to_sequence_indices.pop(best_pair, None)

        # Phase 5: append requested special tokens at the tail of the vocabulary.
        for special_token in encoded_special_tokens:
            if len(vocab) >= vocab_size:
                break
            if special_token in vocab_set:
                continue
            vocab.append(special_token)
            vocab_set.add(special_token)

        return {idx: token for idx, token in enumerate(vocab)}, merges
