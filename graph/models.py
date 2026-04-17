"""
PostgreSQL schema for the propagation graph layer.

Tables:
  graph_nodes  — deduplicated entity nodes across platforms
  graph_edges  — directed propagation edges between nodes
"""

from __future__ import annotations

CREATE_EXTENSION_PGVECTOR = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id           TEXT        NOT NULL,
    type         TEXT        NOT NULL,
    platform     TEXT        NOT NULL,
    label        TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{}',
    first_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, platform)
);
"""

CREATE_EDGES_TABLE = """
CREATE TABLE IF NOT EXISTS graph_edges (
    id           BIGSERIAL   PRIMARY KEY,
    source_id    TEXT        NOT NULL,
    target_id    TEXT        NOT NULL,
    edge_type    TEXT        NOT NULL,
    platform     TEXT        NOT NULL,
    weight       FLOAT       NOT NULL DEFAULT 1.0,
    ts           TIMESTAMPTZ,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, target_id, platform, ts)
);
"""
