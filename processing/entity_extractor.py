"""
Entity extractor for the Diffusion stream processing layer.

Converts raw platform records from Kafka into structured nodes and edges
that represent the propagation graph. Each record yields:
  - Nodes: discrete entities (posts, users, subreddits, repos, URLs)
  - Edges: directed relationships between those entities
           (authored, posted_to, links_to, starred, forked, pushed_to, created)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator


@dataclass(frozen=True)
class ExtractedNode:
    """A discrete entity in the propagation graph."""

    id: str        # stable namespaced key, e.g. "reddit:subreddit:programming"
    type: str      # post | user | subreddit | repo | url
    platform: str
    label: str     # human-readable display name
    metadata: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class ExtractedEdge:
    """A directed propagation relationship between two nodes."""

    source_id: str
    target_id: str
    relation: str  # authored | posted_to | links_to | starred | forked | pushed_to | created
    platform: str
    timestamp: datetime
    weight: float = field(default=1.0, compare=False, hash=False)


@dataclass
class ExtractionResult:
    """Container returned by every platform extractor."""

    nodes: list[ExtractedNode] = field(default_factory=list)
    edges: list[ExtractedEdge] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nodes) + len(self.edges)

    def __iter__(self) -> Iterator[ExtractedNode | ExtractedEdge]:
        return iter([*self.nodes, *self.edges])

    def __bool__(self) -> bool:
        return bool(self.nodes or self.edges)
