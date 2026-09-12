"""
Resolve the parents the firehose never showed us.

Jetstream is a live sample, not an archive. A reply arrives naming a parent
that may have floated past before we connected, or never appeared at all, so
the extractor stores a placeholder for it to keep the edge from dangling. The
consequence is measurable: observed cascades are far shallower than simulated
ones (`max_depth` p50 1 against 4), because a reply whose parent is unknown
becomes a two-node tree instead of joining the thread it belongs to. The
classifier is then asked to score live graphs that look nothing like the ones
it was trained on.

This sweep takes those placeholders, resolves them through the public Bluesky
API — `getPosts` needs no authentication — and walks up the reply chain until
it reaches thread roots or a depth bound. Each resolved post upgrades its
placeholder to a real node and, if it is itself a reply, contributes the edge
to *its* parent, so whole threads reassemble rather than isolated pairs.

    python -m ingestion.bluesky_backfill --limit 200

Runs standalone or on a timer beside the consumer.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

import asyncpg
import httpx

from graph.ids import BLUESKY_CONTENT_PREFIX, BLUESKY_USER_PREFIX
from graph.queries import insert_edge, upsert_node

log = logging.getLogger(__name__)

PUBLIC_API = os.getenv("BLUESKY_PUBLIC_API", "https://public.api.bsky.app/xrpc")
# getPosts accepts at most 25 URIs per call.
BATCH = 25
# How far up a thread to climb in one sweep. Threads are usually shallow, and
# an unbounded walk on a viral post would spend the whole budget in one chain.
MAX_DEPTH = int(os.getenv("BLUESKY_BACKFILL_DEPTH", "6"))
TIMEOUT = float(os.getenv("BLUESKY_BACKFILL_TIMEOUT", "30"))


def uri_to_id(uri: str | None) -> str | None:
    """AT-URI to the id the producer stores, so backfill and firehose agree."""
    if not uri or not uri.startswith("at://"):
        return None
    parts = uri.removeprefix("at://").split("/")
    if len(parts) != 3:
        return None
    return f"{BLUESKY_CONTENT_PREFIX}{parts[0]}_{parts[2]}"


def id_to_uri(node_id: str) -> str | None:
    """Inverse of uri_to_id, for ids we stored but never resolved."""
    raw = node_id.removeprefix(BLUESKY_CONTENT_PREFIX)
    did, _, rkey = raw.partition("_")
    if not did.startswith("did:") or not rkey:
        return None
    return f"at://{did}/app.bsky.feed.post/{rkey}"


async def find_placeholders(conn, limit: int) -> list[str]:
    """Bluesky content nodes that are still stubs, newest first."""
    rows = await conn.fetch(
        """
        SELECT id FROM graph_nodes
        WHERE platform = 'bluesky'
          AND type = 'content'
          AND metadata ? 'placeholder'
        ORDER BY last_seen DESC
        LIMIT $1
        """,
        limit,
    )
    return [r["id"] for r in rows]


async def fetch_posts(client: httpx.AsyncClient, uris: list[str]) -> list[dict]:
    """Resolve up to BATCH AT-URIs. Failures degrade to an empty list."""
    if not uris:
        return []
    try:
        resp = await client.get(
            f"{PUBLIC_API}/app.bsky.feed.getPosts",
            params=[("uris", u) for u in uris[:BATCH]],
        )
        if resp.status_code != 200:
            log.warning("getPosts returned %s; skipping this batch", resp.status_code)
            return []
        return resp.json().get("posts", []) or []
    except httpx.HTTPError as exc:
        log.warning("getPosts failed, skipping this batch: %s", exc)
        return []


async def store_post(conn, post: dict) -> str | None:
    """
    Write one resolved post and its authorship.

    Returns the parent's AT-URI when the post is itself a reply, so the caller
    can climb another level.
    """
    node_id = uri_to_id(post.get("uri"))
    if not node_id:
        return None

    record = post.get("record") or {}
    author = post.get("author") or {}
    text = (record.get("text") or "").strip()

    await upsert_node(
        conn, node_id, "content", "bluesky", text,
        {
            "backfilled": True,
            "cid": post.get("cid"),
            "created_at": record.get("createdAt"),
            "reply_count": post.get("replyCount"),
            "repost_count": post.get("repostCount"),
        },
    )

    handle = author.get("handle") or author.get("did")
    if handle:
        user_id = f"{BLUESKY_USER_PREFIX}{handle}"
        await upsert_node(conn, user_id, "user", "bluesky", handle, {})
        await insert_edge(conn, user_id, node_id, "authored", "bluesky",
                          record.get("createdAt"))

    parent_uri = ((record.get("reply") or {}).get("parent") or {}).get("uri")
    if parent_uri:
        parent_id = uri_to_id(parent_uri)
        if parent_id:
            # Placeholder for the parent, so the edge has both ends. A later
            # pass resolves it; upsert_node will not let this blank real text.
            await upsert_node(conn, parent_id, "content", "bluesky", "",
                              {"placeholder": True})
            await insert_edge(conn, parent_id, node_id, "reshare", "bluesky",
                              record.get("createdAt"))
    return parent_uri


async def backfill(conn, limit: int = 200) -> dict:
    """One sweep. Returns counts for logging and for the caller to assert on."""
    pending = [u for u in (id_to_uri(n) for n in await find_placeholders(conn, limit)) if u]
    if not pending:
        return {"requested": 0, "resolved": 0, "depth_reached": 0, "new_parents": 0}

    seen: set[str] = set()
    resolved = new_parents = 0
    depth = 0

    async with httpx.AsyncClient(timeout=TIMEOUT,
                                 headers={"User-Agent": "diffusion-backfill"}) as client:
        while pending and depth < MAX_DEPTH:
            depth += 1
            batch = [u for u in pending[:BATCH] if u not in seen]
            pending = pending[BATCH:]
            seen.update(batch)

            posts = await fetch_posts(client, batch)
            climbed: list[str] = []
            for post in posts:
                parent = await store_post(conn, post)
                resolved += 1
                if parent and parent not in seen:
                    climbed.append(parent)
                    new_parents += 1
            pending = climbed + pending

    return {"requested": len(seen), "resolved": resolved,
            "depth_reached": depth, "new_parents": new_parents}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=200,
                        help="placeholders to start from (default 200)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")

    conn = await asyncpg.connect(dsn)
    try:
        stats = await backfill(conn, args.limit)
    finally:
        await conn.close()

    print(f"requested {stats['requested']} URIs, resolved {stats['resolved']}, "
          f"climbed {stats['depth_reached']} levels, found {stats['new_parents']} "
          f"further parents")


if __name__ == "__main__":
    asyncio.run(main())
