"""Deduplication — the consumer filter and the producer-side seen-set."""

from __future__ import annotations

from ingestion.utils import BoundedSeenSet
from processing.dedup import DedupFilter


def test_first_sighting_is_new_and_repeats_are_not():
    dedup = DedupFilter()
    record = {"id": "abc", "platform": "hn"}
    assert dedup.is_new(record) is True
    assert dedup.is_new(record) is False


def test_same_id_on_different_platforms_is_not_a_duplicate():
    """Ids are only unique within a platform, so the key must include it."""
    dedup = DedupFilter()
    assert dedup.is_new({"id": "1", "platform": "hn"}) is True
    assert dedup.is_new({"id": "1", "platform": "bluesky"}) is True


def test_records_without_an_id_are_rejected():
    dedup = DedupFilter()
    assert dedup.is_new({"platform": "hn"}) is False
    assert dedup.is_new({"id": "", "platform": "hn"}) is False


def test_eviction_keeps_the_set_bounded():
    dedup = DedupFilter(max_size=10)
    for i in range(11):
        dedup.is_new({"id": str(i), "platform": "hn"})
    # Half the entries are dropped once the cap is passed.
    assert dedup.size <= 10


def test_eviction_drops_the_oldest_entries_first():
    dedup = DedupFilter(max_size=10)
    for i in range(11):
        dedup.is_new({"id": str(i), "platform": "hn"})
    # The oldest is forgotten and so looks new again; the newest is still known.
    assert dedup.is_new({"id": "0", "platform": "hn"}) is True
    assert dedup.is_new({"id": "10", "platform": "hn"}) is False


def test_bounded_seen_set_membership_and_eviction():
    seen = BoundedSeenSet(max_size=4)
    for i in range(5):
        seen.add(i)
    assert 4 in seen
    assert 0 not in seen


def test_bounded_seen_set_ignores_repeat_adds():
    seen = BoundedSeenSet(max_size=4)
    seen.add("x")
    seen.add("x")
    assert "x" in seen
