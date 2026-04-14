"""
Entity extractor for stream processing.

Parses raw events from Reddit, HN, and GitHub into typed graph nodes
and directed edges. Rule-based extraction handles all structured fields;
spaCy en_core_web_sm (CPU, ~12 MB) handles NER on free text.

The model is loaded once at first call and reused — no GPU required.
RTX 4050 VRAM is preserved for the embeddings layer in Stage 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import spacy

_nlp: spacy.language.Language | None = None
_NER_LABELS = {"ORG", "PRODUCT", "GPE", "PERSON"}


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


@dataclass
class Node:
    id: str
    type: str        # content | author | community | repo | named_entity
    platform: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: str   # authored | posted_to | watched | forked | pushed | created | mentions
    platform: str
    timestamp: str | None = None
    weight: float = 1.0


@dataclass
class EntitySet:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def _ner_nodes_and_edges(
    text: str,
    source_id: str,
    platform: str,
    timestamp: str | None,
) -> tuple[list[Node], list[Edge]]:
    """Extract named-entity nodes and mention edges from free text."""
    nodes: list[Node] = []
    edges: list[Edge] = []
    seen: set[str] = set()
    for ent in _get_nlp()(text).ents:
        if ent.label_ not in _NER_LABELS:
            continue
        ent_id = f"entity:{ent.label_.lower()}:{ent.text.lower().replace(' ', '_')}"
        if ent_id not in seen:
            nodes.append(Node(
                id=ent_id,
                type="named_entity",
                platform=platform,
                label=ent.text,
                metadata={"ner_type": ent.label_},
            ))
            seen.add(ent_id)
        edges.append(Edge(
            source_id=source_id,
            target_id=ent_id,
            edge_type="mentions",
            platform=platform,
            timestamp=timestamp,
        ))
    return nodes, edges


def _extract_reddit(record: dict) -> EntitySet:
    es = EntitySet()
    ts = record.get("ingested_at")

    content_id = f"reddit:{record['id']}"
    es.nodes.append(Node(
        id=content_id,
        type="content",
        platform="reddit",
        label=record.get("title", ""),
        metadata={
            "score": record.get("score", 0),
            "num_comments": record.get("num_comments", 0),
            "url": record.get("url"),
            "created_utc": record.get("created_utc"),
        },
    ))

    author = record.get("author")
    if author and author != "[deleted]":
        author_id = f"reddit:user:{author}"
        es.nodes.append(Node(id=author_id, type="author", platform="reddit", label=author))
        es.edges.append(Edge(
            source_id=author_id,
            target_id=content_id,
            edge_type="authored",
            platform="reddit",
            timestamp=ts,
        ))

    subreddit = record.get("subreddit")
    if subreddit:
        community_id = f"reddit:r:{subreddit}"
        es.nodes.append(Node(
            id=community_id,
            type="community",
            platform="reddit",
            label=f"r/{subreddit}",
        ))
        es.edges.append(Edge(
            source_id=content_id,
            target_id=community_id,
            edge_type="posted_to",
            platform="reddit",
            timestamp=ts,
        ))

    title = record.get("title", "")
    if title:
        ent_nodes, ent_edges = _ner_nodes_and_edges(title, content_id, "reddit", ts)
        es.nodes.extend(ent_nodes)
        es.edges.extend(ent_edges)

    return es


def _extract_hn(record: dict) -> EntitySet:
    es = EntitySet()
    ts = record.get("ingested_at")

    content_id = f"hn:{record['id']}"
    es.nodes.append(Node(
        id=content_id,
        type="content",
        platform="hn",
        label=record.get("title", ""),
        metadata={
            "score": record.get("score", 0),
            "descendants": record.get("descendants", 0),
            "url": record.get("url"),
            "created_utc": record.get("created_utc"),
        },
    ))

    author = record.get("by")
    if author:
        author_id = f"hn:user:{author}"
        es.nodes.append(Node(id=author_id, type="author", platform="hn", label=author))
        es.edges.append(Edge(
            source_id=author_id,
            target_id=content_id,
            edge_type="authored",
            platform="hn",
            timestamp=ts,
        ))

    title = record.get("title", "")
    if title:
        ent_nodes, ent_edges = _ner_nodes_and_edges(title, content_id, "hn", ts)
        es.nodes.extend(ent_nodes)
        es.edges.extend(ent_edges)

    return es


def _extract_github(record: dict) -> EntitySet:
    es = EntitySet()
    ts = record.get("created_at") or record.get("ingested_at")

    repo = record.get("repo", "")
    repo_id = f"github:repo:{repo}"
    es.nodes.append(Node(
        id=repo_id,
        type="repo",
        platform="github",
        label=repo,
        metadata={
            "event_type": record.get("type"),
            "created_at": record.get("created_at"),
        },
    ))

    actor = record.get("actor")
    if actor:
        actor_id = f"github:user:{actor}"
        es.nodes.append(Node(id=actor_id, type="author", platform="github", label=actor))
        edge_type = {
            "WatchEvent": "watched",
            "ForkEvent": "forked",
            "PushEvent": "pushed",
            "CreateEvent": "created",
        }.get(record.get("type", ""), "interacted")
        es.edges.append(Edge(
            source_id=actor_id,
            target_id=repo_id,
            edge_type=edge_type,
            platform="github",
            timestamp=ts,
        ))

    return es


_EXTRACTORS = {
    "reddit": _extract_reddit,
    "hn": _extract_hn,
    "github": _extract_github,
}


def extract_entities(record: dict) -> list[Node | Edge]:
    """
    Flat list of Node and Edge objects, as expected by the consumer.
    """
    platform = record.get("platform", "")
    extractor = _EXTRACTORS.get(platform)
    if extractor is None:
        return []
    es = extractor(record)
    return [*es.nodes, *es.edges]


def extract_entity_set(record: dict) -> EntitySet:
    """Structured EntitySet — use this when you need nodes and edges separately."""
    platform = record.get("platform", "")
    extractor = _EXTRACTORS.get(platform)
    if extractor is None:
        return EntitySet()
    return extractor(record)
