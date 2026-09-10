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


def test_bluesky_reply_creates_a_reshare_edge():
    """
    Replies are propagation. Without a reshare edge the graph records who posted
    and what was mentioned, but nothing about how anything spread.
    """
    es = extract_entity_set(
        {
            "platform": "bluesky", "id": "me_r1", "type": "post", "author": "me",
            "text": "replying", "parent_id": "you_p9",
            "ingested_at": "2026-01-01T00:00:00Z", "langs": [],
        }
    )
    reshares = [e for e in es.edges if e.edge_type == "reshare"]
    assert len(reshares) == 1
    assert reshares[0].source_id == f"{BLUESKY_CONTENT_PREFIX}you_p9"
    assert reshares[0].target_id == f"{BLUESKY_CONTENT_PREFIX}me_r1"


def test_bluesky_repost_creates_a_reshare_edge_without_text():
    es = extract_entity_set(
        {
            "platform": "bluesky", "id": "me_r2", "type": "repost", "author": "me",
            "text": "", "parent_id": "you_p9",
            "ingested_at": "2026-01-01T00:00:00Z", "langs": [],
        }
    )
    assert any(e.edge_type == "reshare" for e in es.edges)
    assert any(e.edge_type == "authored" for e in es.edges)


def test_unreferenced_parent_gets_a_placeholder_node():
    """
    The firehose is a sample, so a reply's parent may never be captured. The
    placeholder keeps the edge from dangling; upsert_node refuses to let its
    empty label overwrite the real post if it arrives later.
    """
    es = extract_entity_set(
        {
            "platform": "bluesky", "id": "me_r1", "type": "post", "author": "me",
            "text": "hi", "parent_id": "you_p9",
            "ingested_at": "2026-01-01T00:00:00Z", "langs": [],
        }
    )
    parent = next(n for n in es.nodes if n.id == f"{BLUESKY_CONTENT_PREFIX}you_p9")
    assert parent.label == ""
    assert parent.metadata.get("placeholder") is True


def test_standalone_post_has_no_reshare_edge():
    es = extract_entity_set(
        {
            "platform": "bluesky", "id": "me_r3", "type": "post", "author": "me",
            "text": "original thought", "parent_id": None,
            "ingested_at": "2026-01-01T00:00:00Z", "langs": [],
        }
    )
    assert not any(e.edge_type == "reshare" for e in es.edges)


def test_ner_is_skipped_for_unsupported_languages():
    """
    en_core_web_sm returns confident nonsense on non-English text.

    On a live Bluesky capture the loudest "trending topics" were an
    untranslated Hindi phrase and a lone Japanese bracket, both NER artefacts
    that went on to trip the anomaly detector.
    """
    es = extract_entity_set(
        {
            "platform": "bluesky", "id": "x1", "type": "post", "author": "a",
            "text": "पचासी लाख रुपये की कीमत", "langs": ["hi"],
            "ingested_at": "2026-01-01T00:00:00Z",
        }
    )
    assert not any(n.type == "named_entity" for n in es.nodes)
    # authorship is still recorded — only entity extraction is skipped
    assert any(e.edge_type == "authored" for e in es.edges)


def test_ner_runs_for_english_and_untagged_records():
    from processing.entity_extractor import is_ner_supported

    assert is_ner_supported(["en"]) is True
    assert is_ner_supported(["en-US"]) is True
    assert is_ner_supported([]) is True      # most untagged posts are English
    assert is_ner_supported(None) is True
    assert is_ner_supported(["ja"]) is False
    assert is_ner_supported(["hi", "ur"]) is False
