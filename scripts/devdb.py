"""
Local development database — embedded PostgreSQL 16 + pgvector.

Docker is not required for local development.  The `pgserver` wheel ships a
self-contained PostgreSQL 16 binary with the pgvector extension bundled, so a
single `pip install` gives the same database the compose file provides.

The embedded server binds a *dynamic* port, so `start` writes the resolved URI
back into `.env` as DATABASE_URL.  Every other module keeps reading
`config.DB_URL` and needs no knowledge of how the server was started.

Usage:
    python -m scripts.devdb start    # start server, write .env, create schema
    python -m scripts.devdb url      # print the connection URI
    python -m scripts.devdb stop     # stop the server (data is preserved)
    python -m scripts.devdb reset    # drop every table, then recreate the schema

The compose Postgres remains the deployment target — set DATABASE_URL yourself
and these commands are unnecessary.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys

import asyncpg
import pgserver

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PGDATA_DIR = REPO_ROOT / ".pgdata"
ENV_FILE = REPO_ROOT / ".env"

# Tables owned by the schema, in dependency order (children first) so that
# `reset` can drop them without fighting foreign keys.
_SCHEMA_TABLES = [
    "case_files",
    "anomaly_events",
    "trend_embeddings",
    "graph_edges",
    "graph_nodes",
]

log = logging.getLogger(__name__)


def get_uri(create: bool = True) -> str:
    """
    Return the embedded server's connection URI, starting it if necessary.

    cleanup_mode=None leaves the server running after this process exits —
    without it the database would disappear between CLI invocations.
    """
    if not create and not PGDATA_DIR.exists():
        raise SystemExit("no dev database — run: python -m scripts.devdb start")
    PGDATA_DIR.mkdir(exist_ok=True)
    server = pgserver.get_server(str(PGDATA_DIR), cleanup_mode=None)
    return server.get_uri()


def write_env(uri: str) -> None:
    """Upsert DATABASE_URL in .env, preserving every other line."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    out, replaced = [], False
    for line in lines:
        if line.startswith("DATABASE_URL="):
            out.append(f"DATABASE_URL={uri}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"DATABASE_URL={uri}")
    ENV_FILE.write_text("\n".join(out) + "\n")


async def _init_schema(uri: str) -> None:
    from graph.models import init_schema

    conn = await asyncpg.connect(uri)
    try:
        await init_schema(conn)
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        log.info("schema ready — tables: %s", ", ".join(r["tablename"] for r in rows))
    finally:
        await conn.close()


async def _drop_schema(uri: str) -> None:
    conn = await asyncpg.connect(uri)
    try:
        for table in _SCHEMA_TABLES:
            await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        log.info("dropped %d tables", len(_SCHEMA_TABLES))
    finally:
        await conn.close()


def cmd_start() -> None:
    uri = get_uri()
    write_env(uri)
    asyncio.run(_init_schema(uri))
    log.info("dev database ready at %s", uri)
    log.info("DATABASE_URL written to %s", ENV_FILE)


def cmd_url() -> None:
    print(get_uri(create=False))


def cmd_stop() -> None:
    if not PGDATA_DIR.exists():
        log.info("no dev database to stop")
        return
    # cleanup_mode='stop' shuts the server down when this handle is released.
    pgserver.get_server(str(PGDATA_DIR), cleanup_mode="stop")
    log.info("dev database stopped (data preserved in %s)", PGDATA_DIR)


def cmd_reset() -> None:
    uri = get_uri()
    write_env(uri)
    asyncio.run(_drop_schema(uri))
    asyncio.run(_init_schema(uri))
    log.info("dev database reset")


_COMMANDS = {
    "start": cmd_start,
    "url": cmd_url,
    "stop": cmd_stop,
    "reset": cmd_reset,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("command", choices=sorted(_COMMANDS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _COMMANDS[args.command]()


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
