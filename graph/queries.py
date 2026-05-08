"""
Propagation graph queries.

All functions accept an open asyncpg connection and return plain dicts
so callers (agent tools, API routes) stay decoupled from the DB layer.

Queries:
  upsert_node            — insert or refresh last_seen on a graph node
  insert_edge            — idempotent edge write (ON CONFLICT DO NOTHING)
  get_propagation_path   — BFS from a root node up to max_depth hops
  get_node_degree        — in/out degree for a node
  get_cascade_size       — total nodes reachable from a root
  get_top_nodes_by_degree — highest-degree nodes in a time window
  mark_anomaly_investigated — flip investigated flag after agent run
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# ── writes ────────────────────────────────────────────────────────────────────

async def upsert_node(conn, node_id: str, node_type: str, platform: str, label: str, metadata: dict) -> None:
    await conn.execute(
        """
        INSERT INTO graph_nodes (id, type, platform, label, metadata)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (id, platform) DO UPDATE
            SET label     = EXCLUDED.label,
                metadata  = EXCLUDED.metadata,
                last_seen = now()
        """,
        node_id, node_type, platform, label, json.dumps(metadata),
    )


async def insert_edge(
    conn,
    source_id: str,
    target_id: str,
    edge_type: str,
    platform: str,
    ts: str | datetime | None,
    weight: float = 1.0,
) -> None:
    # asyncpg requires a datetime object for timestamptz columns
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)
    elif ts is None:
        ts = datetime.now(timezone.utc)

    await conn.execute(
        """
        INSERT INTO graph_edges (source_id, target_id, edge_type, platform, ts, weight)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (source_id, target_id, platform, ts) DO NOTHING
        """,
        source_id, target_id, edge_type, platform, ts, weight,
    )


# ── reads ─────────────────────────────────────────────────────────────────────

async def get_propagation_path(
    conn,
    root_id: str,
    max_depth: int = 6,
) -> list[dict[str, Any]]:
    """
    BFS over graph_edges starting at root_id.

    Returns a list of edge dicts: {source_id, target_id, edge_type, platform, ts, depth}.
    Depth is capped at max_depth to prevent runaway traversal on dense graphs.
    """
    rows = await conn.fetch(
        """
        WITH RECURSIVE propagation AS (
            SELECT
                source_id,
                target_id,
                edge_type,
                platform,
                ts,
                1 AS depth
            FROM graph_edges
            WHERE source_id = $1

            UNION ALL

            SELECT
                e.source_id,
                e.target_id,
                e.edge_type,
                e.platform,
                e.ts,
                p.depth + 1
            FROM graph_edges e
            JOIN propagation p ON e.source_id = p.target_id
            WHERE p.depth < $2
        )
        SELECT source_id, target_id, edge_type, platform, ts, depth
        FROM propagation
        ORDER BY depth, ts NULLS LAST
        """,
        root_id, max_depth,
    )
    return [dict(r) for r in rows]


async def get_node_degree(conn, node_id: str) -> dict[str, int]:
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE source_id = $1) AS out_degree,
            COUNT(*) FILTER (WHERE target_id = $1) AS in_degree
        FROM graph_edges
        WHERE source_id = $1 OR target_id = $1
        """,
        node_id,
    )
    return {"in_degree": row["in_degree"], "out_degree": row["out_degree"]}


async def get_cascade_size(conn, root_id: str, max_depth: int = 10) -> int:
    """Total distinct nodes reachable from root_id (excluding root itself)."""
    row = await conn.fetchrow(
        """
        WITH RECURSIVE cascade AS (
            SELECT target_id AS node_id, 1 AS depth
            FROM graph_edges
            WHERE source_id = $1

            UNION ALL

            SELECT e.target_id, c.depth + 1
            FROM graph_edges e
            JOIN cascade c ON e.source_id = c.node_id
            WHERE c.depth < $2
        )
        SELECT COUNT(DISTINCT node_id) AS size FROM cascade
        """,
        root_id, max_depth,
    )
    return row["size"]


async def get_top_nodes_by_degree(
    conn,
    platform: str | None = None,
    since_minutes: int = 60,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Highest in-degree nodes within the last `since_minutes` minutes.
    Useful for surfacing currently spreading content without a full BFS.
    """
    rows = await conn.fetch(
        """
        SELECT
            n.id,
            n.type,
            n.platform,
            n.label,
            COUNT(e.id) AS in_degree
        FROM graph_nodes n
        JOIN graph_edges e ON e.target_id = n.id
        WHERE e.ingested_at >= now() - $1 * interval '1 minute'
          AND ($2::text IS NULL OR n.platform = $2)
        GROUP BY n.id, n.type, n.platform, n.label
        ORDER BY in_degree DESC
        LIMIT $3
        """,
        since_minutes, platform, limit,
    )
    return [dict(r) for r in rows]


async def mark_anomaly_investigated(conn, anomaly_id: int) -> None:
    await conn.execute(
        "UPDATE anomaly_events SET investigated = TRUE WHERE id = $1",
        anomaly_id,
    )
