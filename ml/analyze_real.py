"""
Characterise the cascades observed in live traffic, and score them.

Usage:
    python -m ml.analyze_real
    python -m ml.analyze_real --min-size 3 --limit 2000

Answers three questions the simulated results cannot:

  1. What do real cascades look like — how large, how deep, how wide?
  2. Does the simulator resemble them, or was it calibrated to a fiction?
  3. What does the classifier say about real traffic it was never trained on?

The third is reported as a *distribution*, never as accuracy. Live cascades carry
no ground-truth label — that absence is the reason the simulator exists — so the
only honest claim is what the model asserts about real data, not whether it was
right.

A live firehose is a sample, not an archive, so most observed cascades are
shallow: a reply whose parent was never seen still yields a two-node tree. The
size distribution is reported in full rather than filtered to flatter it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib

import asyncpg
import numpy as np
import pandas as pd

from config import DB_URL, configure_logging
from ml.dataset import DEFAULT_LABELS
from ml.features import FEATURE_COLUMNS, build_feature_table
from ml.predict import load_model
from ml.report import markdown_table, upsert_section

SIM_PREFIX = "sim:"
DEFAULT_MIN_SIZE = 3
DEFAULT_LIMIT = 3000

log = logging.getLogger(__name__)

# Cascade roots in live data: a post that was reshared but is not itself a
# reshare of anything observed.
_ROOTS_SQL = """
SELECT DISTINCT e.source_id AS root_id
FROM graph_edges e
WHERE e.edge_type = 'reshare'
  AND e.source_id NOT LIKE $1
  AND NOT EXISTS (
      SELECT 1 FROM graph_edges p
      WHERE p.edge_type = 'reshare' AND p.target_id = e.source_id
  )
LIMIT $2
"""


async def ingestion_summary(conn: asyncpg.Connection) -> str:
    rows = await conn.fetch(
        """
        SELECT platform, edge_type, COUNT(*) AS n
        FROM graph_edges
        WHERE source_id NOT LIKE $1
        GROUP BY platform, edge_type
        ORDER BY platform, n DESC
        """,
        f"{SIM_PREFIX}%",
    )
    nodes = await conn.fetch(
        """
        SELECT platform, type, COUNT(*) AS n
        FROM graph_nodes
        WHERE id NOT LIKE $1
        GROUP BY platform, type
        ORDER BY platform, n DESC
        """,
        f"{SIM_PREFIX}%",
    )
    edge_rows = [[r["platform"], f"`{r['edge_type']}`", f"{r['n']:,}"] for r in rows]
    node_rows = [[r["platform"], f"`{r['type']}`", f"{r['n']:,}"] for r in nodes]
    return (
        markdown_table(["platform", "edge type", "count"], edge_rows)
        + "\n\n"
        + markdown_table(["platform", "node type", "count"], node_rows)
    )


def _dist(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "p50": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def shape_comparison(real: pd.DataFrame, sim: pd.DataFrame) -> str:
    """
    Real versus simulated cascade shape, on identical features.

    This is the check on the simulator. If the two populations are wildly
    different, every number trained on simulated cascades describes a world that
    does not exist, and the honest response is to say so rather than to quietly
    report the F1.
    """
    metrics = ["size", "max_depth", "root_fanout_share", "leaf_frac", "delay_median_s"]
    rows = []
    for metric in metrics:
        r, s = real[metric].to_numpy(float), sim[metric].to_numpy(float)
        rows.append(
            [
                f"`{metric}`",
                f"{np.median(r):.2f}",
                f"{np.percentile(r, 90):.2f}",
                f"{np.median(s):.2f}",
                f"{np.percentile(s, 90):.2f}",
            ]
        )
    return markdown_table(
        ["feature", "real p50", "real p90", "simulated p50", "simulated p90"], rows
    )


def score_distribution(model, real: pd.DataFrame) -> tuple[str, np.ndarray]:
    p = model.predict_proba(real[FEATURE_COLUMNS])[:, 1]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    counts = np.histogram(p, bins=bins)[0]
    rows = [
        [label, f"{c:,}", f"{100 * c / max(len(p), 1):.1f}%"]
        for label, c in zip(labels, counts, strict=True)
    ]
    return markdown_table(["P(coordinated)", "cascades", "share"], rows), p


async def trace_example(conn: asyncpg.Connection, root_id: str) -> str:
    """Render one real cascade as a readable propagation trace."""
    rows = await conn.fetch(
        """
        WITH RECURSIVE tree AS (
            SELECT e.source_id, e.target_id, e.ts, 1 AS depth
            FROM graph_edges e
            WHERE e.source_id = $1 AND e.edge_type = 'reshare'
            UNION ALL
            SELECT e.source_id, e.target_id, e.ts, t.depth + 1
            FROM tree t
            JOIN graph_edges e ON e.source_id = t.target_id AND e.edge_type = 'reshare'
            WHERE t.depth < 12
        )
        SELECT t.depth, t.ts, COALESCE(NULLIF(n.label, ''), '(not captured)') AS label
        FROM tree t
        LEFT JOIN graph_nodes n ON n.id = t.target_id
        ORDER BY t.ts
        LIMIT 12
        """,
        root_id,
    )
    if not rows:
        return "_no traceable cascade found_"

    root = await conn.fetchrow(
        "SELECT COALESCE(NULLIF(label, ''), '(not captured)') AS label FROM graph_nodes WHERE id = $1",
        root_id,
    )
    t0 = rows[0]["ts"]
    lines = [f"seed  ·  {(root['label'] if root else '(unknown)')[:90]}"]
    for r in rows:
        offset = (r["ts"] - t0).total_seconds()
        lines.append(
            f"  +{offset:7.0f}s  depth {r['depth']}  {r['label'][:80]}"
        )
    return "```\n" + "\n".join(lines) + "\n```"


async def main_async(min_size: int, limit: int, labels_path: pathlib.Path) -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        summary = await ingestion_summary(conn)

        roots = [r["root_id"] for r in await conn.fetch(_ROOTS_SQL, f"{SIM_PREFIX}%", limit)]
        if not roots:
            raise SystemExit(
                "no real cascades found — run: python -m scripts.ingest_live --minutes 20"
            )
        log.info("found %d candidate cascade roots in live data", len(roots))

        real = await build_feature_table(conn, roots)
        real_all = real.copy()
        real = real[real["size"] >= min_size]
        log.info(
            "%d cascades observed, %d with at least %d posts",
            len(real_all), len(real), min_size,
        )

        size_dist = _dist(real_all["size"].to_numpy(float))
        depth_dist = _dist(real_all["max_depth"].to_numpy(float))

        example = ""
        if not real.empty:
            biggest = real.sort_values("size", ascending=False).iloc[0]
            example = await trace_example(conn, biggest["root_node_id"])

        # Simulated cascades, featurised identically, for the shape comparison.
        sim_ids = pd.read_csv(labels_path)["root_node_id"].tolist()[:1500]
        sim = await build_feature_table(conn, sim_ids)
    finally:
        await conn.close()

    model = load_model()
    scored, p = (None, None)
    if model is not None and not real.empty:
        scored, p = score_distribution(model, real)

    body = f"""
Live Bluesky and Hacker News traffic ingested through the real producer
serialisation and `processing.consumer.handle()` — dedup, spaCy NER, graph
writes, velocity scoring and anomaly detection — with Kafka omitted as transport
(see `scripts/ingest_live.py`).

### What was ingested

{summary}

### Observed cascade shape

{markdown_table(
    ["metric", "cascades", "p50", "p90", "max"],
    [
        ["size (posts)", f"{size_dist['n']:,}", f"{size_dist['p50']:.0f}",
         f"{size_dist['p90']:.0f}", f"{size_dist['max']:.0f}"],
        ["depth", f"{depth_dist['n']:,}", f"{depth_dist['p50']:.0f}",
         f"{depth_dist['p90']:.0f}", f"{depth_dist['max']:.0f}"],
    ],
)}

A firehose is a sample, not an archive: a reply whose parent was never captured
still forms a two-node tree, so the distribution is dominated by small cascades.
{len(real):,} of {len(real_all):,} observed cascades reach {min_size}+ posts.

### Does the simulator resemble reality?

{shape_comparison(real, sim) if not real.empty else "_not enough real cascades to compare_"}

The comparison is the check on everything trained upstream. Simulated cascades
are grown to a target size and so are far larger than a sampled firehose shows;
the shape features — depth, root fan-out share, leaf fraction — are the ones
worth reading, because they describe structure rather than how much of it was
captured.

### Classifier applied to live traffic

{scored if scored is not None else "_no trained model available_"}

Reported as a distribution, not accuracy. Real cascades carry no ground-truth
label — that absence is precisely why the simulator exists — so what the model
asserts about live data is measurable, and whether it is correct is not.

### One observed propagation

{example}

Reproduce with `python -m scripts.ingest_live --minutes 20` then
`python -m ml.analyze_real`.
"""
    upsert_section("Live traffic — observed cascades", body)
    log.info("wrote results to docs/results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterise cascades in live traffic")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main_async(args.min_size, args.limit, args.labels))


if __name__ == "__main__":
    main()
