from collections import deque


class DedupFilter:
    """
    In-memory deduplication using a bounded seen-set.

    Tracks event IDs across all platforms. Once the set exceeds
    max_size, the oldest max_size // 2 entries are evicted to keep
    memory usage predictable.
    """

    def __init__(self, max_size: int = 20_000) -> None:
        self._seen: set[str] = set()
        self._order: deque[str] = deque()
        self._max = max_size

    def is_new(self, record: dict) -> bool:
        if not record.get("id"):
            return False
        key = f"{record.get('platform', 'unknown')}:{record['id']}"
        if key in self._seen:
            return False
        self._seen.add(key)
        self._order.append(key)
        if len(self._seen) > self._max:
            evict_count = self._max // 2
            for _ in range(evict_count):
                self._seen.discard(self._order.popleft())
        return True
