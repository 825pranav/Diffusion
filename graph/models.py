"""
PostgreSQL schema for the propagation graph layer.

Tables:
  graph_nodes     — deduplicated entity nodes across platforms
  graph_edges     — directed propagation edges between nodes
  trend_embeddings — pgvector embeddings for semantic similarity search
  anomaly_events  — records of detected velocity spikes that trigger the agent
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

# 1536 dims matches text-embedding-3-small; store model_name + model_version
# alongside every vector so similarity searches are scoped to matching checkpoints.
CREATE_TREND_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS trend_embeddings (
    id            BIGSERIAL    PRIMARY KEY,
    node_id       TEXT         NOT NULL,
    platform      TEXT         NOT NULL,
    embedding     vector(1536) NOT NULL,
    model_name    TEXT         NOT NULL,
    model_version TEXT         NOT NULL DEFAULT 'v1',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
"""

CREATE_ANOMALY_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS anomaly_events (
    id           BIGSERIAL   PRIMARY KEY,
    node_id      TEXT        NOT NULL,
    platform     TEXT        NOT NULL,
    z_score      FLOAT       NOT NULL,
    velocity     FLOAT       NOT NULL,
    detected_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    investigated BOOLEAN     NOT NULL DEFAULT FALSE
);
"""
