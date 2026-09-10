"""Entity extraction — one raw record per platform into nodes and edges."""

from __future__ import annotations

import pytest

from graph.ids import (
    BLUESKY_CONTENT_PREFIX,
    GITHUB_REPO_PREFIX,
    HN_CONTENT_PREFIX,
    MASTODON_TAG_PREFIX,
)
from processing.entity_extractor import extract_entity_set


def test_bluesky_post_yields_content_author_and_authorship():
    es = extract_entity_set(
        {
            "platform": "bluesky",
            "id": "abc",
            "author": "alice",
            "text": "a short post",
            "ingested_at": "2026-01-01T00:00:00Z",
        }
    )
    ids = {n.id for n in es.nodes}
    assert f"{BLUESKY_CONTENT_PREFIX}abc" in ids
    assert any(e.edge_type == "authored" for e in es.edges)


def test_mastodon_tags_become_community_nodes():
    es = extract_entity_set(
        {
            "platform": "mastodon",
            "id": "m1",
            "author": "bob",
            "text": "hello",
            "tags": ["Rust"],
            "ingested_at": "2026-01-01T00:00:00Z",
        }
    )
    assert f"{MASTODON_TAG_PREFIX}rust" in {n.id for n in es.nodes}
    assert any(e.edge_type == "posted_to" for e in es.edges)


def test_hn_story_uses_the_title_as_its_label():
    es = extract_entity_set(
        {"platform": "hn", "id": "42", "by": "carol", "title": "Show HN: a thing"}
    )
    content = next(n for n in es.nodes if n.id == f"{HN_CONTENT_PREFIX}42")
    assert content.label == "Show HN: a thing"


@pytest.mark.parametrize(
    "event_type,edge_type",
    [("WatchEvent", "watched"), ("ForkEvent", "forked"), ("PushEvent", "pushed"),
     ("WeirdEvent", "interacted")],
)
def test_github_event_types_map_to_edge_types(event_type, edge_type):
    es = extract_entity_set(
        {"platform": "github", "repo": "o/r", "actor": "dev", "type": event_type}
    )
    assert f"{GITHUB_REPO_PREFIX}o/r" in {n.id for n in es.nodes}
    assert {e.edge_type for e in es.edges} == {edge_type}


def test_unknown_platform_yields_nothing():
    es = extract_entity_set({"platform": "myspace", "id": "1"})
    assert es.nodes == [] and es.edges == []


def test_named_entities_produce_mention_edges():
    """The NER pass is what turns free text into trackable topics."""
    es = extract_entity_set(
        {
            "platform": "bluesky",
            "id": "x",
            "author": "dan",
            "text": "Microsoft and Google announced a partnership in Berlin",
            "ingested_at": "2026-01-01T00:00:00Z",
        }
    )
    assert any(n.type == "named_entity" for n in es.nodes)
    assert any(e.edge_type == "mentions" for e in es.edges)
