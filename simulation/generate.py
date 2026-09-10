"""
Generate labeled synthetic cascades and load them into the propagation graph.

Usage:
    python -m simulation.generate --n 2000
    python -m simulation.generate --n 2000 --difficulty hard --seed 7 --truncate

Writes graph_nodes / graph_edges rows and a labels CSV holding the ground truth
for every cascade (root id, label, subtype, and the sampled parameters).  The
CSV is the only place the label exists — nothing in the graph itself reveals it.

Generation is deterministic given --seed, so the dataset can be reproduced
rather than committed.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import pathlib
import sys
from typing import Iterable

import asyncpg
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import DB_URL, configure_logging  # noqa: E402
from simulation.cascade import (  # noqa: E402
    DEFAULT_DIFFICULTY,
    DIFFICULTIES,
    Cascade,
    CascadeSimulator,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "simulation" / "labels.csv"

# Rows per executemany batch. Large enough to amortise round-trips, small enough
# that a failure does not roll back the entire load.
BATCH_SIZE = 5_000

LABEL_FIELDS = [
    "root_node_id",
    "label",
    "subtype",
    "topic_id",
    "t0",
    "n_content_nodes",
    "n_edges",
]
PARAM_FIELDS = [
    "root_attach",
    "target_size",
    "delay_median_s",
    "delay_sigma",
    "bot_frac",
    "hop_prob",
    "hop_lag_s",
]

log = logging.getLogger(__name__)


def _batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _load_cascades(conn: asyncpg.Connection, cascades: list[Cascade]) -> tuple[int, int]:
    """
    Bulk-load every node and edge.

    graph_edges carries an AFTER INSERT trigger that fires pg_notify per row for
    the live WebSocket feed.  At simulation volume that is hundreds of thousands
    of notifications nobody is listening for, so the trigger is disabled for the
    duration of the load and restored afterwards.
    """
    # Deduplicate across cascades: authors and topics recur by design.
    node_rows: dict[tuple[str, str], tuple] = {}
    edge_rows: list[tuple] = []

    for cascade in cascades:
        for node in cascade.nodes:
            node_rows[(node.id, node.platform)] = (
                node.id,
                node.type,
                node.platform,
                node.label,
                json.dumps(node.metadata),
            )
        for edge in cascade.edges:
            edge_rows.append(
                (
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.platform,
                    edge.ts,
                    edge.weight,
                )
            )

    await conn.execute("ALTER TABLE graph_edges DISABLE TRIGGER trg_graph_delta")
    try:
        for batch in _batched(list(node_rows.values()), BATCH_SIZE):
            await conn.executemany(
                """
                INSERT INTO graph_nodes (id, type, platform, label, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (id, platform) DO NOTHING
                """,
                batch,
            )
        for batch in _batched(edge_rows, BATCH_SIZE):
            await conn.executemany(
                """
                INSERT INTO graph_edges (source_id, target_id, edge_type, platform, ts, weight)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source_id, target_id, platform, ts) DO NOTHING
                """,
                batch,
            )
    finally:
        await conn.execute("ALTER TABLE graph_edges ENABLE TRIGGER trg_graph_delta")

    return len(node_rows), len(edge_rows)


async def _delete_matching(conn: asyncpg.Connection, pattern: str) -> None:
    """Remove simulation rows matching an id pattern, children first."""
    await conn.execute(
        "DELETE FROM graph_edges WHERE source_id LIKE $1 OR target_id LIKE $1", pattern
    )
    await conn.execute("DELETE FROM trend_embeddings WHERE node_id LIKE $1", pattern)
    await conn.execute("DELETE FROM graph_nodes WHERE id LIKE $1", pattern)


def _write_labels(path: pathlib.Path, cascades: list[Cascade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        param_headers = [f"param_{k}" for k in PARAM_FIELDS]
        writer = csv.DictWriter(fh, fieldnames=LABEL_FIELDS + param_headers)
        writer.writeheader()
        for c in cascades:
            row = {
                "root_node_id": c.root_id,
                "label": c.label,
                "subtype": c.subtype,
                "topic_id": c.topic_id,
                "t0": c.t0.isoformat(),
                "n_content_nodes": c.size,
                "n_edges": len(c.edges),
            }
            row.update({f"param_{k}": round(c.params[k], 4) for k in PARAM_FIELDS})
            writer.writerow(row)


def _summarise(cascades: list[Cascade]) -> str:
    """
    Class-conditional summary, printed so overlap can be eyeballed immediately.

    If the two classes look cleanly separated here, the downstream classifier
    will score near-perfectly and the evaluation is worthless — raise
    --difficulty rather than celebrating the F1.
    """
    lines = []
    for label in ("organic", "coordinated"):
        group = [c for c in cascades if c.label == label]
        if not group:
            continue
        sizes = np.array([c.size for c in group])
        delays = np.array([c.params["delay_median_s"] for c in group])
        bot = np.array([c.params["bot_frac"] for c in group])
        hard = sum(1 for c in group if c.subtype != "plain")
        lines.append(
            f"  {label:12s} n={len(group):5d}  hard={hard:4d}  "
            f"size p50={np.median(sizes):6.1f}  "
            f"delay_s p10/p50/p90={np.percentile(delays, 10):7.1f}/"
            f"{np.median(delays):7.1f}/{np.percentile(delays, 90):8.1f}  "
            f"bot_frac p50={np.median(bot):.2f}"
        )
    return "\n".join(lines)


async def run(
    n: int,
    difficulty: str,
    seed: int,
    truncate: bool,
    out_path: pathlib.Path,
) -> None:
    rng = np.random.default_rng(seed)
    run_id = f"{difficulty[0]}{seed}"
    sim = CascadeSimulator(rng=rng, run_id=run_id, difficulty=difficulty)

    # Balanced by construction, then shuffled so ordering carries no signal.
    labels = ["organic"] * (n // 2) + ["coordinated"] * (n - n // 2)
    rng.shuffle(labels)

    log.info("generating %d cascades (difficulty=%s, seed=%d, run_id=%s)", n, difficulty, seed, run_id)
    cascades = [sim.generate(label) for label in labels]

    log.info("class summary:\n%s", _summarise(cascades))

    conn = await asyncpg.connect(DB_URL)
    try:
        if truncate:
            await _delete_matching(conn, "sim:%")
            log.info("truncated all previous simulation rows")
        else:
            # Regeneration must be idempotent. Node ids are derived from run_id
            # and a counter, so re-running the same seed reuses them: nodes would
            # be skipped by ON CONFLICT while their edges inserted alongside the
            # old ones, silently grafting two datasets into one and inflating
            # every cascade past its target size.
            await _delete_matching(conn, f"sim:{run_id}:%")
        n_nodes, n_edges = await _load_cascades(conn, cascades)
        log.info("loaded %d unique nodes and %d edges", n_nodes, n_edges)
    finally:
        await conn.close()

    _write_labels(out_path, cascades)
    log.info("wrote %d labels to %s", len(cascades), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled synthetic cascades")
    parser.add_argument("--n", type=int, default=2000, help="number of cascades")
    parser.add_argument(
        "--difficulty",
        choices=sorted(DIFFICULTIES),
        default=DEFAULT_DIFFICULTY,
        help="how much the two classes overlap",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--truncate", action="store_true", help="delete previous sim:% rows first"
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(run(args.n, args.difficulty, args.seed, args.truncate, args.out))


if __name__ == "__main__":
    main()
