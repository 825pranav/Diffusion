"""Kafka pipeline — the processor's batch handling and the scorer's partitioning."""

from __future__ import annotations

import asyncio
import json
import random
import zlib
from types import SimpleNamespace

from graph.queries import insert_edges, upsert_nodes
from ingestion.utils import record_key
from processing.consumer import Item, dead_letter, mentions, prepare, strip_nul
from processing.dedup import DedupFilter
from processing.entity_extractor import EntitySet, Node
from processing.scorer import MentionScorer
from processing.velocity_scorer import VelocityScorer


def _msg(value, offset: int = 0, ts_ms: int = 1_700_000_000_000):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    return SimpleNamespace(topic="hn-raw", partition=0, offset=offset, timestamp=ts_ms, value=raw)


def _hn(item_id: int, title: str = "Rust 2.0 released by Mozilla") -> dict:
    return {"id": item_id, "platform": "hn", "type": "story", "title": title, "by": "pg"}


# ── processor ────────────────────────────────────────────────────────────────

def test_strip_nul_reaches_nested_strings_and_keys():
    record = {"text": "a\x00b", "meta": {"k\x00": ["x\x00", 3, None]}}
    assert strip_nul(record) == {"text": "ab", "meta": {"k": ["x", 3, None]}}


def test_a_record_with_nul_survives_preparation():
    """Postgres rejects U+0000; the first live run lost every record carrying one."""
    items, dead = prepare([_msg(_hn(1, "nul\x00here"))], DedupFilter())
    assert not dead
    assert all("\x00" not in n.label for n in items[0].entities.nodes)


def test_undecodable_records_are_dead_lettered_not_dropped():
    items, dead = prepare([_msg(b"{not json", offset=7)], DedupFilter())
    assert items == []
    assert dead[0]["offset"] == 7 and dead[0]["topic"] == "hn-raw"
    assert dead[0]["value"] == "{not json"
    assert dead[0]["error"].startswith("JSONDecodeError")


def test_a_redelivered_record_is_prepared_once():
    dedup = DedupFilter()
    items, _ = prepare([_msg(_hn(5)), _msg(_hn(5), offset=1)], dedup)
    assert len(items) == 1


def test_mentions_carry_event_time_and_only_scored_types():
    entities = EntitySet(nodes=[
        Node("hn:1", "content", "hn", "post"),
        Node("entity:mozilla", "named_entity", "hn", "Mozilla"),
        Node("hn:user:pg", "author", "hn", "pg"),
    ])
    item = Item(_msg(_hn(1), ts_ms=1_700_000_123_000), {"platform": "hn"}, entities)
    assert mentions(item) == [{"entity_id": "entity:mozilla", "platform": "hn", "ts": 1_700_000_123.0}]


def test_dead_letter_keeps_the_source_position():
    letter = dead_letter(_msg(b"x", offset=42), ValueError("boom"))
    assert (letter["topic"], letter["partition"], letter["offset"]) == ("hn-raw", 0, 42)
    assert letter["error"] == "ValueError: boom"


def test_record_key_is_stable_per_record():
    """Copies of one record must share a partition for per-process dedup to catch them."""
    a = {"platform": "bluesky", "id": "did:plc:x_3k", "ingested_at": "t1"}
    b = {**a, "ingested_at": "t2"}
    assert record_key(a) == record_key(b) == b"bluesky:did:plc:x_3k"


class _RecordingConn:
    def __init__(self):
        self.calls = []

    async def executemany(self, sql, rows):
        self.calls.append(list(rows))


def test_batch_writes_are_sorted_so_concurrent_batches_cannot_deadlock():
    conn = _RecordingConn()
    nodes = [("b", "content", "hn", "B", {}), ("a", "content", "hn", "A", {}), ("b", "content", "hn", "", {})]
    asyncio.run(upsert_nodes(conn, nodes))
    keys = [(r[0], r[3]) for r in conn.calls[0]]
    # Sorted by key, and a node's repeats keep their arrival order.
    assert keys == [("a", "A"), ("b", "B"), ("b", "")]

    conn = _RecordingConn()
    edges = [("y", "z", "reshare", "hn", None, 1.0), ("x", "z", "reshare", "hn", None, 1.0)]
    asyncio.run(insert_edges(conn, edges))
    assert [r[0] for r in conn.calls[0]] == ["x", "y"]


# ── scorer ───────────────────────────────────────────────────────────────────

def _burst_stream(seed: int = 0) -> list[dict]:
    """Steady background chatter on many entities, then one entity bursts."""
    rng = random.Random(seed)
    stream, t = [], 1_000_000.0
    for _ in range(4000):
        t += rng.expovariate(2.0)
        stream.append({"entity_id": f"e{rng.randrange(20)}", "platform": "hn", "ts": t})
    for _ in range(300):
        t += rng.expovariate(10.0)
        stream.append({"entity_id": "e3", "platform": "hn", "ts": t})
    return stream


def _partition(entity_id: str) -> int:
    return zlib.crc32(entity_id.encode()) % 3


def _detect(scorers: list[list[tuple[int, dict]]]) -> set[tuple[str, str]]:
    """Run one MentionScorer per list of (partition, mention) pairs; pool what they raise."""
    found = set()
    for owned in scorers:
        events = MentionScorer().score(owned)
        found |= {(e.entity_id, e.detected_at) for e in events}
    return found


def test_keyed_partitions_detect_exactly_what_one_scorer_does():
    """Splitting by entity loses nothing: each entity's whole history stays together."""
    stream = [(_partition(m["entity_id"]), m) for m in _burst_stream()]
    single = _detect([stream])
    split = [[pm for pm in stream if pm[0] == p] for p in range(3)]
    assert ("e3" in {e for e, _ in single}) and _detect(split) == single


def test_unkeyed_partitions_change_what_is_detected():
    """Round-robin splitting, as unkeyed raw records gave three processors, is not equivalent."""
    stream = [(0, m) for m in _burst_stream()]
    round_robin = [[(p, m) for _, m in stream[p::3]] for p in range(3)]
    assert _detect(round_robin) != _detect([stream])


def test_nothing_is_judged_until_a_full_window_has_been_seen():
    """
    A fresh window can only fill, so every count climbs and steady chatter looks
    like a spike in everything. Warm-up counts those mentions without judging.
    """
    stream = [(0, m) for m in _burst_stream()]
    t0 = stream[0][1]["ts"]
    scorer = MentionScorer(window_seconds=300)
    events = scorer.score(stream)
    assert events and all(e.detected_at >= _iso(t0 + 300) for e in events)


def test_a_reassigned_partition_warms_up_again():
    scorer = MentionScorer(window_seconds=60)
    assert not scorer.warm(1, 1000.0)
    assert scorer.warm(1, 1061.0)
    scorer.reset([1])
    assert not scorer.warm(1, 1062.0)


def _iso(ts: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts, UTC).isoformat()


def test_late_mentions_are_counted_and_still_evicted():
    scorer = VelocityScorer(window_seconds=60)
    scorer.record("e", ts=100.0)
    scorer.record("e", ts=150.0)
    late = scorer.record("e", ts=120.0)  # arrives after 150 from a lagging processor
    assert late.events_in_window == 3
    # At 185 the window starts at 125: 100 and the late 120 have aged out, 150
    # has not. Appended rather than inserted, 120 would sit behind 150 and
    # never be evicted.
    assert scorer.record("e", ts=185.0).events_in_window == 2
