"""
End-to-end smoke test for the Diffusion pipeline.

Checks:
  1. API health endpoint responds
  2. DB is reachable and schema tables exist
  3. Inserts a synthetic anomaly event and waits for the agent to produce a case file
  4. Validates the case file schema (required fields, confidence in [0,1])

Usage:
    python scripts/validate_e2e.py [--timeout 120] [--api-url http://localhost:8000]

Exit code 0 = all checks passed. Non-zero = failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import string
import sys
import time
from datetime import datetime, timezone

import aiohttp
import asyncpg

DB_URL = os.getenv("DATABASE_URL", "postgresql://diffusion:diffusion@localhost:5432/diffusion")
DEFAULT_API_URL = "http://localhost:8000"

REQUIRED_TABLES = {"graph_nodes", "graph_edges", "anomaly_events", "case_files", "trend_embeddings"}
REQUIRED_CASE_FIELDS = {"id", "trend", "platform_origin", "classification", "confidence", "needs_review"}

log = logging.getLogger(__name__)


def _rand_node_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"e2e_test_{suffix}"


async def check_api_health(session: aiohttp.ClientSession, api_url: str) -> bool:
    try:
        async with session.get(f"{api_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                log.error("health check returned HTTP %d", resp.status)
                return False
            body = await resp.json()
            if body.get("status") != "ok":
                log.error("unexpected health body: %s", body)
                return False
            log.info("✓ API health — ok")
            return True
    except Exception as exc:
        log.error("✗ API unreachable: %s", exc)
        return False


async def check_db_schema(conn: asyncpg.Connection) -> bool:
    rows = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )
    present = {r["tablename"] for r in rows}
    missing = REQUIRED_TABLES - present
    if missing:
        log.error("✗ missing tables: %s", missing)
        return False
    log.info("✓ DB schema — all required tables present")
    return True


async def inject_anomaly(conn: asyncpg.Connection) -> tuple[int, str]:
    node_id = _rand_node_id()
    platform = random.choice(["reddit", "hn", "github"])

    # insert a node so propagation queries have something to find
    await conn.execute(
        """
        INSERT INTO graph_nodes (id, type, platform, label, metadata)
        VALUES ($1, 'post', $2, $3, '{"synthetic": true, "e2e": true}'::jsonb)
        ON CONFLICT (id, platform) DO NOTHING
        """,
        node_id, platform, f"E2E test node {node_id[-8:]}",
    )

    row = await conn.fetchrow(
        """
        INSERT INTO anomaly_events (node_id, platform, z_score, velocity)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        node_id, platform, round(random.uniform(2.6, 4.5), 2), round(random.uniform(15.0, 60.0), 1),
    )
    anomaly_id: int = row["id"]
    log.info("✓ anomaly injected — id=%d node=%s platform=%s", anomaly_id, node_id, platform)
    return anomaly_id, node_id


async def wait_for_case_file(
    conn: asyncpg.Connection, anomaly_id: int, timeout_s: float
) -> dict | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = await conn.fetchrow(
            "SELECT * FROM case_files WHERE anomaly_event_id = $1", anomaly_id
        )
        if row:
            return dict(row)
        remaining = deadline - time.monotonic()
        log.info("waiting for case file… (%.0fs remaining)", max(0, remaining))
        await asyncio.sleep(5)
    return None


def validate_case_file(case: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_CASE_FIELDS:
        if field not in case:
            errors.append(f"missing field: {field}")

    conf = case.get("confidence")
    if conf is not None and not (0.0 <= conf <= 1.0):
        errors.append(f"confidence out of range: {conf}")

    classification = case.get("classification", "")
    if classification not in {"organic", "coordinated_amplification", "uncertain"}:
        errors.append(f"unexpected classification: {classification!r}")

    signals = case.get("signals")
    if signals is not None:
        parsed = json.loads(signals) if isinstance(signals, str) else signals
        if not isinstance(parsed, list):
            errors.append("signals is not a list")

    return errors


async def run(api_url: str, timeout_s: float) -> bool:
    all_ok = True

    async with aiohttp.ClientSession() as session:
        if not await check_api_health(session, api_url):
            all_ok = False

    conn = await asyncpg.connect(DB_URL)
    try:
        if not await check_db_schema(conn):
            all_ok = False
            return all_ok

        anomaly_id, node_id = await inject_anomaly(conn)

        log.info("waiting up to %.0fs for agent to produce case file…", timeout_s)
        case = await wait_for_case_file(conn, anomaly_id, timeout_s)

        if case is None:
            log.error(
                "✗ no case file produced within %.0fs — "
                "is the agent running? (uvicorn api.main:app or python -m agent.agent)",
                timeout_s,
            )
            all_ok = False
        else:
            errors = validate_case_file(case)
            if errors:
                for e in errors:
                    log.error("✗ case file validation: %s", e)
                all_ok = False
            else:
                log.info(
                    "✓ case file valid — classification=%s confidence=%.2f needs_review=%s",
                    case["classification"],
                    case["confidence"],
                    case["needs_review"],
                )

        # clean up e2e test data
        await conn.execute("DELETE FROM anomaly_events WHERE node_id LIKE 'e2e_test_%'")
        await conn.execute("DELETE FROM graph_nodes WHERE id LIKE 'e2e_test_%'")
    finally:
        await conn.close()

    return all_ok


async def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion end-to-end smoke test")
    parser.add_argument("--timeout", type=float, default=120, help="seconds to wait for case file")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    args = parser.parse_args()

    ok = await run(args.api_url, args.timeout)
    if ok:
        log.info("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        log.error("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
