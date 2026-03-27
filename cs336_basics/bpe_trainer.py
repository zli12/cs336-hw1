from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import regex as re

from cs336_basics.max_pair_heap import MaxPairHeap

# Byte-level BPE (like GPT-2): training starts from 256 single-byte tokens. Each merge
# concatenates two existing vocabulary entries; new token IDs are assigned in order
# (256, 257, ...). Token sequences store integer IDs, not raw bytes, so we can update
# them in place as merges happen.


class BPETrainer:
    """Train a byte-level BPE vocabulary and merge list.

    Training flow (high level):
    1) Read corpus text.
    2) Split around special tokens so merges never cross those boundaries.
    3) Pretokenize each segment with GPT-2 regex and count unique byte sequences.
    4) Initialize base vocab with all 256 byte tokens.
    5) Repeatedly merge the highest-count adjacent token pair.
    6) Append special tokens (if missing) until ``vocab_size``.

    Quick example:
        >>> from pathlib import Path
        >>> from cs336_basics.bpe_trainer import BPETrainer
        >>> tmp = Path("tiny_bpe_demo.txt")
        >>> _ = tmp.write_text("low lower lowest low", encoding="utf-8")
        >>> vocab, merges = BPETrainer.train(
        ...     input_path=tmp,
        ...     vocab_size=270,
        ...     special_tokens=["<|endoftext|>"],
        ... )
        >>> len(vocab) >= 256
        True
        >>> len(merges) > 0
        True

    For a step-by-step visualization of how merges evolve token boundaries, run:
        ``python scripts/visualize_bpe_training.py``
    """

    # GPT-2-style pretokenization: splits text into chunks (words, numbers, punctuation,
    # whitespace runs) before BPE runs *within* each chunk. BPE never merges across
    # two different regex matches—only adjacent bytes inside one pretoken.
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
        """Train byte-level BPE merges and return (id->token vocab, merges).

        Pipeline: corpus → segments (no cross-special-token merges) → pretoken
        counts → per-pretoken byte ID sequences → global pair statistics →
        greedy merge loop → optional special-token rows in vocab.
        """
        # Entire training file as one string (Unicode).
        text = cls._read_corpus_text(input_path)
        # Chunks of text between special-token strings; statistics never span chunk boundaries.
        segments = cls._split_around_special_tokens(text, special_tokens)
        # Keys = one pretoken as tuple of UTF-8 byte values; values = corpus count.
        pretoken_counts = cls._get_pretoken_counts(segments)
        # vocab[token_id] -> bytes for that id; vocab_set enables fast "already in vocab?" checks.
        vocab, vocab_set = cls._initial_byte_vocab_and_set()
        # Same specials as strings, as raw bytes (for appending whole rows to vocab later).
        encoded_special_tokens = cls._encode_special_tokens(special_tokens)
        # Merge only until this size so we can still append unseen specials without exceeding vocab_size.
        target_vocab_size_without_specials = cls._target_vocab_size_excluding_new_specials(
            vocab_size=vocab_size,
            vocab_set=vocab_set,
            encoded_special_tokens=encoded_special_tokens,
        )
        # token_sequences[i] = mutable list of vocab IDs for one UNIQUE pretoken;
        # token_sequence_counts[i] = corpus count for that row (weights pair statistics).
        token_sequences, token_sequence_counts = cls._build_token_sequences_and_counts(
            pretoken_counts
        )
        # pair_counts: weighted adjacent-pair totals over the whole corpus.
        # pair_to_sequence_indices: reverse lookup — which rows in token_sequences still contain each pair.
        pair_counts, pair_to_sequence_indices = cls._build_initial_pair_statistics(
            token_sequences=token_sequences,
            token_sequence_counts=token_sequence_counts,
        )
        # Ordered list of (left_bytes, right_bytes) for each merge (defines the tokenizer).
        merges = cls._run_merge_loop_until_vocab_target(
            vocab=vocab,
            vocab_set=vocab_set,
            token_sequences=token_sequences,
            token_sequence_counts=token_sequence_counts,
            pair_counts=pair_counts,
            pair_to_sequence_indices=pair_to_sequence_indices,
            target_vocab_size_without_specials=target_vocab_size_without_specials,
        )
        cls._append_missing_special_tokens(
            vocab=vocab,
            vocab_set=vocab_set,
            encoded_special_tokens=encoded_special_tokens,
            vocab_size=vocab_size,
        )
        # Map vocab id -> token bytes; merges list is the BPE merge table (training artifact).
        return {idx: token for idx, token in enumerate(vocab)}, merges

    @staticmethod
    def _read_corpus_text(input_path: str | Path) -> str:
        """Load the entire training file as one Unicode string."""
        with open(input_path, encoding="utf-8") as f:
            return f.read()

    @classmethod
    def _split_around_special_tokens(cls, text: str, special_tokens: list[str]) -> list[str]:
        """Split the corpus on special strings so BPE statistics never span them.

        Example: if special is "<|endoftext|>", text before and after are separate
        segments; pretokens are counted independently in each segment.
        """
        if not special_tokens:
            return [text]
        # Alternation of literal specials; re.escape avoids "|", ".", etc. breaking the split.
        special_split_pattern = "|".join(re.escape(token) for token in special_tokens)
        return re.split(special_split_pattern, text)

    @classmethod
    def _get_pretoken_counts(cls, segments: list[str]) -> Counter[tuple[int, ...]]:
        """Count how often each pretoken appears, keyed by its UTF-8 bytes as ints.

        We store pretokens as tuple[int, ...] (byte values) so they are hashable and
        align with the initial vocab (one token ID per byte 0..255).
        """
        pretoken_counts: Counter[tuple[int, ...]] = Counter()
        for segment in segments:
            for match in cls.PATTERN.finditer(segment):
                pretoken = match.group(0)  # one GPT-2-regex chunk (word, space run, etc.)
                pretoken_counts[tuple(pretoken.encode("utf-8"))] += 1
        return pretoken_counts

    @staticmethod
    def _initial_byte_vocab_and_set() -> tuple[list[bytes], set[bytes]]:
        """Start with 256 vocabulary entries: one for each possible byte value."""
        # Index i holds the single-byte token bytes([i]) for i in 0..255.
        vocab = [bytes([i]) for i in range(256)]
        return vocab, set(vocab)

    @staticmethod
    def _encode_special_tokens(special_tokens: list[str]) -> list[bytes]:
        """UTF-8 bytes for each special token (used when appending to vocab later)."""
        return [token.encode("utf-8") for token in special_tokens]

    @staticmethod
    def _target_vocab_size_excluding_new_specials(
        vocab_size: int,
        vocab_set: set[bytes],
        encoded_special_tokens: list[bytes],
    ) -> int:
        """How large vocab may grow from merges before we reserve slots for specials.

        If a special token's byte sequence is not already a single vocab row, we will
        append it at the end without training a merge for it—so we stop merge training
        early enough that len(vocab) + len(new specials) <= vocab_size.
        """
        # Specials whose byte sequence is not already a single vocab entry need a reserved slot at the end.
        special_tokens_to_add = [t for t in encoded_special_tokens if t not in vocab_set]
        return max(256, vocab_size - len(special_tokens_to_add))

    @staticmethod
    def _build_token_sequences_and_counts(
        pretoken_counts: Counter[tuple[int, ...]],
    ) -> tuple[list[list[int]], list[int]]:
        """One mutable ID sequence per *unique* pretoken, plus its corpus count.

        Deduplication: if "the" appears 1M times, we keep one list [t,h,e] and store
        count 1M. All BPE updates reuse that single list—this is the main trick
        that makes training on large corpora feasible.
        """
        # Each list starts as raw byte IDs (0..255) for one distinct pretoken string.
        token_sequences = [list(pretoken_tuple) for pretoken_tuple in pretoken_counts]
        # Parallel to token_sequences: corpus count for that row (weights pair statistics).
        token_sequence_counts = [
            pretoken_counts[tuple(token_sequence)] for token_sequence in token_sequences
        ]
        return token_sequences, token_sequence_counts

    @staticmethod
    def _build_initial_pair_statistics(
        token_sequences: list[list[int]],
        token_sequence_counts: list[int],
    ) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[int]]]:
        """Global adjacent-pair counts and reverse index: pair → sequence indices.

        pair_counts[(i,j)] is *weighted*: sum over rows of
        (local count of (i,j) in that row) * (corpus count for that pretoken row).

        pair_to_sequence_indices lists which rows in token_sequences still contain each
        pair, so we do not scan the whole pretoken table after every merge.
        """
        # (left_id, right_id) -> total weighted count across all pretoken rows.
        pair_counts: Counter[tuple[int, int]] = Counter()
        # (left_id, right_id) -> row indices into token_sequences that currently contain that adjacent pair.
        pair_to_sequence_indices: dict[tuple[int, int], set[int]] = defaultdict(set)
        for sequence_index, token_sequence in enumerate(token_sequences):
            if len(token_sequence) < 2:
                continue
            # Adjacent pairs only (BPE never merges non-adjacent symbols).
            local_pair_counts = Counter(zip(token_sequence, token_sequence[1:], strict=False))
            row_count = token_sequence_counts[sequence_index]  # corpus count for this pretoken row
            for pair, pair_occurrences_in_sequence in local_pair_counts.items():
                pair_counts[pair] += pair_occurrences_in_sequence * row_count
                pair_to_sequence_indices[pair].add(sequence_index)
        return pair_counts, pair_to_sequence_indices

    @staticmethod
    def _replace_non_overlapping_pair(
        token_sequence: list[int],
        left_id: int,
        right_id: int,
        merged_id: int,
    ) -> list[int]:
        """Scan left-to-right; merge (left_id, right_id) without overlap.

        Example merging (a,a) on [a,a,a]: first pair becomes merged, third a stays
        alone → [merged_id, a], not [merged_id] from greedy triple overlap.
        """
        merged_token_sequence: list[int] = []  # output IDs after applying this one merge rule
        token_index = 0  # scan position in the input ID list
        while token_index < len(token_sequence):
            if (
                token_index + 1 < len(token_sequence)
                and token_sequence[token_index] == left_id
                and token_sequence[token_index + 1] == right_id
            ):
                merged_token_sequence.append(merged_id)
                token_index += 2  # skip both consumed IDs
            else:
                merged_token_sequence.append(token_sequence[token_index])
                token_index += 1
        return merged_token_sequence

    @staticmethod
    def _update_global_pair_stats_after_sequence_merge(
        *,
        old_sequence_pair_counts: Counter[tuple[int, int]],
        merged_token_sequence: list[int],
        row_corpus_count: int,
        sequence_index: int,
        pair_counts: Counter[tuple[int, int]],
        pair_to_sequence_indices: dict[tuple[int, int], set[int]],
        max_pair_heap: MaxPairHeap,
    ) -> None:
        """Adjust global pair_counts (and heap + reverse map) for one sequence change.

        Instead of recomputing all pairs from scratch over the whole corpus, we diff
        old vs new adjacent-pair multisets for this sequence only, scaled by the corpus
        count for this pretoken row (row_corpus_count).

        Args:
            old_sequence_pair_counts: adjacent (id, id) pairs in this row before the merge.
            merged_token_sequence: same row after non-overlapping replacement with merged_id.
            row_corpus_count: how often this pretoken appears in the corpus (weights deltas).
            sequence_index: row index into token_sequences / token_sequence_counts.
            pair_counts: global weighted adjacent-pair totals (mutated in place).
            pair_to_sequence_indices: reverse map from pair to row indices (mutated).
            max_pair_heap: priority structure for next merge (kept in sync with pair_counts).
        """
        # Before: old_sequence_pair_counts (argument). After: multiset of adjacent pairs in merged_token_sequence.
        new_sequence_pair_counts = Counter(
            zip(merged_token_sequence, merged_token_sequence[1:], strict=False)
        )

        # Pairs that existed before: apply count delta; drop from index if count hits 0.
        for pair, old_count in old_sequence_pair_counts.items():
            new_count = new_sequence_pair_counts.get(pair, 0)
            if new_count == old_count:
                continue
            # Scale local pair count change by how often this pretoken appears in the corpus.
            delta = (new_count - old_count) * row_corpus_count
            updated = pair_counts.get(pair, 0) + delta
            if updated <= 0:
                pair_counts.pop(pair, None)
            else:
                pair_counts[pair] = updated
                max_pair_heap.update(pair=pair, count=updated)

            if new_count == 0:
                # No row still has this pair once this sequence loses it — drop from reverse map if empty.
                members = pair_to_sequence_indices.get(pair)
                if members is not None:
                    members.discard(sequence_index)
                    if not members:
                        pair_to_sequence_indices.pop(pair, None)
            else:
                pair_to_sequence_indices[pair].add(sequence_index)

        # Brand-new pairs introduced by the merge (not in old_sequence_pair_counts).
        for pair, new_count in new_sequence_pair_counts.items():
            if pair in old_sequence_pair_counts:
                continue
            pair_counts[pair] += new_count * row_corpus_count
            pair_to_sequence_indices[pair].add(sequence_index)
            max_pair_heap.update(pair=pair, count=pair_counts[pair])

    @classmethod
    def _apply_one_merge_to_affected_sequences(
        cls,
        *,
        best_pair: tuple[int, int],
        left_id: int,
        right_id: int,
        merged_id: int,
        affected_sequence_indices: list[int],
        token_sequences: list[list[int]],
        token_sequence_counts: list[int],
        pair_counts: Counter[tuple[int, int]],
        pair_to_sequence_indices: dict[tuple[int, int], set[int]],
        max_pair_heap: MaxPairHeap,
    ) -> None:
        """Update only sequences listed in pair_to_sequence_indices[best_pair].

        That set can be slightly stale (heap popped an outdated max), so we skip any
        sequence that no longer actually contains best_pair.
        """
        for sequence_index in affected_sequence_indices:  # candidate rows; some may be stale
            old_token_sequence = token_sequences[sequence_index]  # vocab IDs for this pretoken before merge
            if len(old_token_sequence) < 2:
                continue

            old_sequence_pair_counts = Counter(
                zip(old_token_sequence, old_token_sequence[1:], strict=False)
            )
            if best_pair not in old_sequence_pair_counts:
                continue

            merged_token_sequence = cls._replace_non_overlapping_pair(
                old_token_sequence, left_id, right_id, merged_id
            )
            if merged_token_sequence == old_token_sequence:
                continue  # no structural change (should be rare if best_pair present)

            row_corpus_count = token_sequence_counts[sequence_index]
            cls._update_global_pair_stats_after_sequence_merge(
                old_sequence_pair_counts=old_sequence_pair_counts,
                merged_token_sequence=merged_token_sequence,
                row_corpus_count=row_corpus_count,
                sequence_index=sequence_index,
                pair_counts=pair_counts,
                pair_to_sequence_indices=pair_to_sequence_indices,
                max_pair_heap=max_pair_heap,
            )
            token_sequences[sequence_index] = merged_token_sequence

    @classmethod
    def _run_merge_loop_until_vocab_target(
        cls,
        *,
        vocab: list[bytes],
        vocab_set: set[bytes],
        token_sequences: list[list[int]],
        token_sequence_counts: list[int],
        pair_counts: Counter[tuple[int, int]],
        pair_to_sequence_indices: dict[tuple[int, int], set[int]],
        target_vocab_size_without_specials: int,
    ) -> list[tuple[bytes, bytes]]:
        """Greedy BPE: repeatedly merge the highest-weight adjacent pair.

        Each iteration: pop best pair from heap, append concat(left_bytes, right_bytes)
        to vocab with a fresh ID, rewrite affected token sequences, incrementally fix
        pair_counts, then remove the merged-away pair from global maps (count is 0).
        """
        merges: list[tuple[bytes, bytes]] = []  # human-readable merge rule per step
        # Tracks current best adjacent pairs; may lag true counts until update() calls.
        max_pair_heap = MaxPairHeap(pair_counts=pair_counts, vocab=vocab)

        while len(vocab) < target_vocab_size_without_specials and pair_counts:
            best_pair_and_count = max_pair_heap.pop_max()
            if best_pair_and_count is None:
                break
            best_pair, best_count = best_pair_and_count  # (left_id, right_id), weighted corpus count
            if best_count <= 0:
                break  # stale heap entry or exhausted valid merges

            left_id, right_id = best_pair  # vocabulary indices of the two subword bytes to merge
            merges.append((vocab[left_id], vocab[right_id]))
            merged_token = vocab[left_id] + vocab[right_id]  # new subword as concatenated bytes
            merged_id = len(vocab)  # fresh vocabulary index for merged_token
            vocab.append(merged_token)
            vocab_set.add(merged_token)

            # Rows in token_sequences that (may) still contain adjacent (left_id, right_id).
            affected_sequence_indices = list(pair_to_sequence_indices.get(best_pair, set()))
            cls._apply_one_merge_to_affected_sequences(
                best_pair=best_pair,
                left_id=left_id,
                right_id=right_id,
                merged_id=merged_id,
                affected_sequence_indices=affected_sequence_indices,
                token_sequences=token_sequences,
                token_sequence_counts=token_sequence_counts,
                pair_counts=pair_counts,
                pair_to_sequence_indices=pair_to_sequence_indices,
                max_pair_heap=max_pair_heap,
            )

            # This pair is fully replaced by merged_id everywhere it appeared; remove
            # from active structures so it cannot be popped again.
            pair_counts.pop(best_pair, None)
            pair_to_sequence_indices.pop(best_pair, None)

        return merges

    @staticmethod
    def _append_missing_special_tokens(
        *,
        vocab: list[bytes],
        vocab_set: set[bytes],
        encoded_special_tokens: list[bytes],
        vocab_size: int,
    ) -> None:
        """Add special token bytes as whole vocab rows if not already present.

        Order matches `special_tokens` argument; stops if vocab_size is reached.
        """
        for special_token in encoded_special_tokens:  # one whole vocab row per special (no BPE merge)
            if len(vocab) >= vocab_size:
                break
            if special_token in vocab_set:
                continue
            vocab.append(special_token)
            vocab_set.add(special_token)
