from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import heapify, heappop, heappush


@dataclass(frozen=True)
class _DescendingBytes:
    """Bytes wrapper that reverses lexical ordering for heap tie-breaks."""

    value: bytes

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _DescendingBytes):
            return NotImplemented
        # Reverse ordering so min-heap pops lexicographically larger bytes first.
        return self.value > other.value


class MaxPairHeap:
    """Max-oriented heap wrapper for pair-frequency selection."""

    def __init__(
        self,
        pair_counts: Counter[tuple[int, int]],
        vocab: list[bytes],
    ) -> None:
        self._pair_counts = pair_counts
        self._vocab = vocab
        self._heap: list[tuple[int, bytes, bytes, int, int]] = [
            self._entry(pair=pair, count=count)
            for pair, count in self._pair_counts.items()
            if count > 0
        ]
        heapify(self._heap)

    def _entry(
        self,
        pair: tuple[int, int],
        count: int,
    ) -> tuple[int, _DescendingBytes, _DescendingBytes, int, int]:
        left_id, right_id = pair
        # Python's heapq is a min-heap; negate count so the highest frequency comes first.
        # For equal counts, reverse byte lexical order to match max(..., key=(count,left,right)).
        return (
            -count,
            _DescendingBytes(self._vocab[left_id]),
            _DescendingBytes(self._vocab[right_id]),
            left_id,
            right_id,
        )

    def update(self, pair: tuple[int, int], count: int) -> None:
        """Insert refreshed priority for an active pair."""
        if count > 0:
            heappush(self._heap, self._entry(pair=pair, count=count))

    def pop_max(self) -> tuple[tuple[int, int], int] | None:
        """Return highest-priority live pair via lazy invalidation."""
        while self._heap:
            _, _, _, left_id, right_id = heappop(self._heap)
            candidate_pair = (left_id, right_id)
            current_count = self._pair_counts.get(candidate_pair, 0)
            if current_count <= 0:
                continue

            # Refresh priority so we can verify this candidate is not stale.
            self.update(pair=candidate_pair, count=current_count)
            entry = heappop(self._heap)
            refreshed_pair = (entry[3], entry[4])
            refreshed_count = self._pair_counts.get(refreshed_pair, 0)
            if refreshed_count <= 0:
                continue
            if -entry[0] != refreshed_count:
                self.update(pair=refreshed_pair, count=refreshed_count)
                continue

            return refreshed_pair, refreshed_count

        return None
