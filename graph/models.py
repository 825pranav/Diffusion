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

# 768 dims matches BAAI/bge-base-en-v1.5; store model_name + model_version
# alongside every vector so similarity searches are scoped to matching checkpoints.
CREATE_TREND_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS trend_embeddings (
    id            BIGSERIAL   PRIMARY KEY,
    node_id       TEXT        NOT NULL,
    platform      TEXT        NOT NULL,
    embedding     vector(768) NOT NULL,
    model_name    TEXT        NOT NULL,
    model_version TEXT        NOT NULL DEFAULT 'v1',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
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

CREATE_CASE_FILES_TABLE = """
CREATE TABLE IF NOT EXISTS case_files (
    id                   BIGSERIAL    PRIMARY KEY,
    anomaly_event_id     BIGINT       REFERENCES anomaly_events(id),
    trend                TEXT         NOT NULL,
    platform_origin      TEXT         NOT NULL,
    detected_at          TIMESTAMPTZ  NOT NULL,
    classification       TEXT         NOT NULL,
    confidence           FLOAT        NOT NULL,
    signals              JSONB        NOT NULL DEFAULT '[]',
    similar_past_cases   JSONB        NOT NULL DEFAULT '[]',
    ragas_scores         JSONB        NOT NULL DEFAULT '{}',
    agent_reasoning_steps INT         NOT NULL DEFAULT 0,
    reasoning_steps_detail JSONB      NOT NULL DEFAULT '[]',
    needs_review         BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
);
"""

# ── indexes ──────────────────────────────────────────────────────────────────

CREATE_INDEXES = [
    # fast node lookups by platform
    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_platform ON graph_nodes (platform);",
    # fast edge traversal in both directions
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges (source_id);",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges (target_id);",
    "CREATE INDEX IF NOT EXISTS idx_graph_edges_ts     ON graph_edges (ts DESC NULLS LAST);",
    # pgvector HNSW index — scoped queries filter on (model_name, model_version) first
    "CREATE INDEX IF NOT EXISTS idx_trend_embeddings_vec ON trend_embeddings USING hnsw (embedding vector_cosine_ops);",
    "CREATE INDEX IF NOT EXISTS idx_trend_embeddings_model ON trend_embeddings (model_name, model_version);",
    # agent queries uninvestigated anomalies
    "CREATE INDEX IF NOT EXISTS idx_anomaly_events_uninvestigated ON anomaly_events (detected_at DESC) WHERE investigated = FALSE;",
    # case file review queue
    "CREATE INDEX IF NOT EXISTS idx_case_files_review ON case_files (created_at DESC) WHERE needs_review = TRUE;",
]

# Postgres LISTEN/NOTIFY channel — anomaly_detector fires NOTIFY on this channel;
# the agent process wakes on receive.
ANOMALY_NOTIFY_CHANNEL = "anomaly_detected"

# Postgres LISTEN/NOTIFY channel — fires on every graph_edges INSERT;
# the WebSocket listener fans the payload out to connected clients.
GRAPH_DELTA_CHANNEL = "graph_delta"

CREATE_GRAPH_DELTA_TRIGGER = """
CREATE OR REPLACE FUNCTION notify_graph_delta() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('graph_delta', row_to_json(NEW)::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_graph_delta ON graph_edges;
CREATE TRIGGER trg_graph_delta
    AFTER INSERT ON graph_edges
    FOR EACH ROW EXECUTE FUNCTION notify_graph_delta();
"""

CREATE_ANOMALY_NOTIFY_TRIGGER = """
CREATE OR REPLACE FUNCTION notify_anomaly() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('anomaly_detected', row_to_json(NEW)::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_anomaly_notify ON anomaly_events;
CREATE TRIGGER trg_anomaly_notify
    AFTER INSERT ON anomaly_events
    FOR EACH ROW EXECUTE FUNCTION notify_anomaly();
"""

# ── schema bootstrap ──────────────────────────────────────────────────────────

_DDL_STATEMENTS = [
    CREATE_EXTENSION_PGVECTOR,
    CREATE_NODES_TABLE,
    CREATE_EDGES_TABLE,
    CREATE_TREND_EMBEDDINGS_TABLE,
    CREATE_ANOMALY_EVENTS_TABLE,
    CREATE_CASE_FILES_TABLE,
    *CREATE_INDEXES,
    CREATE_GRAPH_DELTA_TRIGGER,
    CREATE_ANOMALY_NOTIFY_TRIGGER,
]


async def init_schema(conn) -> None:
    """Run all DDL statements against an open asyncpg connection."""
    for stmt in _DDL_STATEMENTS:
        await conn.execute(stmt)
