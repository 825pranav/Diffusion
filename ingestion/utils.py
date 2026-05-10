"""Shared utilities for Kafka producers."""

from __future__ import annotations

from collections import deque


class BoundedSeenSet:
    """
    In-memory seen-set with a soft size cap.

    When the set exceeds max_size, the oldest max_size // 2 entries are
    evicted to keep memory usage predictable.  Deduplication is best-effort:
    evicted entries may be re-processed, which is acceptable for idempotent
    Kafka producers.

    Supports `in` and `.add()` for drop-in use in polling loops.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._seen: set = set()
        self._order: deque = deque()
        self._max = max_size

    def __contains__(self, item: object) -> bool:
        return item in self._seen

    def add(self, item: object) -> None:
        if item in self._seen:
            return
        self._seen.add(item)
        self._order.append(item)
        if len(self._seen) > self._max:
            for _ in range(self._max // 2):
                self._seen.discard(self._order.popleft())
