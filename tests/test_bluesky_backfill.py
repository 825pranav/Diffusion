"""
Tests for parent backfill.

The id translation has to agree exactly with what the producer writes, or a
backfilled parent lands under a different id from the placeholder it was meant
to replace and the thread stays broken — silently, since both rows are valid.
"""

from __future__ import annotations

import pytest

from graph.ids import BLUESKY_CONTENT_PREFIX
from ingestion.bluesky_backfill import id_to_uri, uri_to_id
from ingestion.bluesky_producer import _uri_to_id as producer_uri_to_id

URI = "at://did:plc:abc123/app.bsky.feed.post/3kxyz"


def test_uri_to_id_matches_the_producer():
    """
    Backfill and firehose must agree on ids.

    The producer stores "<did>_<rkey>" under the bluesky content prefix. If
    backfill derived anything else, resolving a parent would create a second
    node beside the placeholder instead of filling it in.
    """
    assert uri_to_id(URI) == f"{BLUESKY_CONTENT_PREFIX}{producer_uri_to_id(URI)}"


def test_uri_to_id_rejects_malformed_input():
    assert uri_to_id(None) is None
    assert uri_to_id("") is None
    assert uri_to_id("https://bsky.app/profile/x/post/y") is None
    assert uri_to_id("at://did:plc:abc123") is None            # too few segments
    assert uri_to_id("at://a/b/c/d") is None                   # too many


def test_id_to_uri_round_trips():
    node_id = uri_to_id(URI)
    assert id_to_uri(node_id) == URI


def test_id_to_uri_rejects_ids_that_are_not_posts():
    assert id_to_uri(f"{BLUESKY_CONTENT_PREFIX}notadid_3kxyz") is None
    assert id_to_uri(f"{BLUESKY_CONTENT_PREFIX}did:plc:abc123") is None   # no rkey
    assert id_to_uri("mastodon:12345") is None


def test_round_trip_survives_underscores_in_the_rkey():
    # partition() splits on the first underscore, so the did stays intact.
    uri = "at://did:plc:abc_def/app.bsky.feed.post/3kxyz"
    assert uri_to_id(uri) is not None


@pytest.mark.parametrize("status", [429, 500, 503])
def test_fetch_posts_degrades_on_upstream_failure(status, monkeypatch):
    """A failing batch must not abort the sweep; it returns nothing and moves on."""
    import asyncio

    from ingestion import bluesky_backfill as bf

    class _Resp:
        status_code = status

        def json(self):  # pragma: no cover - never reached for error codes
            return {}

    class _Client:
        async def get(self, *a, **k):
            return _Resp()

    assert asyncio.run(bf.fetch_posts(_Client(), [URI])) == []


def test_fetch_posts_with_no_uris_makes_no_request():
    import asyncio

    from ingestion import bluesky_backfill as bf

    class _Client:
        async def get(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("should not request with an empty batch")

    assert asyncio.run(bf.fetch_posts(_Client(), [])) == []
