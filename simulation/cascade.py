"""
Synthetic cascade generator.

Live social data carries no ground-truth label for "organic vs coordinated", so
the only way to get a *measurable* classifier is to simulate cascades whose label
is known by construction.

Two classes:
  organic      — branching process, heavy-tailed human response delays, mostly
                 distinct authors, slow cross-platform diffusion
  coordinated  — bot cluster, tightly synchronised timestamps, authors drawn
                 repeatedly from a small pool, fast cross-platform jump

Both are produced by *one* growth engine that differs only in its sampled
parameters.  Using a single engine matters: two hand-written structural
generators would leak the label through incidental shape differences, and the
classifier would learn the generator rather than the phenomenon.

The classes are deliberately not cleanly separable:

  * every parameter is drawn from a distribution that overlaps its counterpart
    (a slow coordinated campaign and a fast organic one occupy the same region)
  * a configurable fraction of each class is generated as a confusable subtype —
    "viral organic" cascades that burst like a bot campaign, and "stealth
    coordinated" cascades that stagger their timing and vary their authors

A simulator whose two populations were obviously different would yield F1 ≈ 1.0
and tell us nothing about the problem.

Graph shape produced per cascade:

    author  --authored-->  content  --reshare-->  content  --reshare--> ...
                              |
                              +--mentions-->  topic entity

Propagation structure lives entirely in the `reshare` edges; feature extraction
traverses those and ignores the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np

PLATFORMS = ("bluesky", "mastodon", "hn", "github")

Label = Literal["organic", "coordinated"]

# Node-id namespace.  Author ids carry no bot/human marker — reuse has to be
# recovered from the edge structure, otherwise the label leaks into the features.
CONTENT_FMT = "sim:{run}:c:{n}"
AUTHOR_FMT = "sim:{run}:a:{n}"
TOPIC_FMT = "sim:{run}:e:{n}"


@dataclass
class SimNode:
    id: str
    type: str          # content | author | named_entity
    platform: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimEdge:
    source_id: str
    target_id: str
    edge_type: str     # authored | reshare | mentions
    platform: str
    ts: datetime
    weight: float = 1.0


@dataclass
class Cascade:
    root_id: str
    label: Label
    subtype: str       # plain | viral_organic | stealth_coordinated
    topic_id: str
    t0: datetime
    params: dict[str, float]
    nodes: list[SimNode]
    edges: list[SimEdge]

    @property
    def size(self) -> int:
        """Number of content nodes in the cascade (its true size)."""
        return sum(1 for n in self.nodes if n.type == "content")


# ── difficulty presets ────────────────────────────────────────────────────────
#
# `hard_frac` is the share of each class generated as its confusable subtype.
# `spread` multiplies every lognormal sigma: larger spread means the two classes
# overlap more, because each one's tail reaches further into the other's centre.

@dataclass(frozen=True)
class Difficulty:
    name: str
    hard_frac: float
    spread: float


DIFFICULTIES: dict[str, Difficulty] = {
    "easy": Difficulty("easy", hard_frac=0.02, spread=0.6),
    "medium": Difficulty("medium", hard_frac=0.15, spread=1.0),
    "hard": Difficulty("hard", hard_frac=0.28, spread=1.35),
}

DEFAULT_DIFFICULTY = "medium"


class CascadeSimulator:
    """
    Stateful generator — author and topic pools persist across cascades.

    Persistence is the point for author reuse: a coordinated campaign draws from
    the same small bot pool every time, so "how many of this cascade's authors
    appeared in earlier cascades" only becomes a signal once several cascades
    share a simulator instance.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        run_id: str,
        difficulty: str = DEFAULT_DIFFICULTY,
        n_bots: int = 400,
        n_humans: int = 4000,
        n_topics: int = 200,
        window_days: int = 3,
    ) -> None:
        if difficulty not in DIFFICULTIES:
            raise ValueError(
                f"unknown difficulty {difficulty!r}; expected one of {sorted(DIFFICULTIES)}"
            )
        self.rng = rng
        self.run_id = run_id
        self.difficulty = DIFFICULTIES[difficulty]
        self.window_days = window_days

        # Author pool. Bots occupy the first n_bots slots, but nothing downstream
        # is told where the boundary is.
        self._n_bots = n_bots
        self._n_humans = n_humans
        self._n_topics = n_topics

        self._content_counter = 0
        self._emitted_authors: set[str] = set()

    # ── id helpers ────────────────────────────────────────────────────────────

    def _next_content_id(self) -> str:
        self._content_counter += 1
        return CONTENT_FMT.format(run=self.run_id, n=self._content_counter)

    def _bot_author(self) -> str:
        return AUTHOR_FMT.format(run=self.run_id, n=int(self.rng.integers(0, self._n_bots)))

    def _human_author(self) -> str:
        idx = self._n_bots + int(self.rng.integers(0, self._n_humans))
        return AUTHOR_FMT.format(run=self.run_id, n=idx)

    # ── parameter sampling ────────────────────────────────────────────────────

    def _lognormal(self, median: float, sigma: float) -> float:
        return float(self.rng.lognormal(mean=np.log(median), sigma=sigma * self.difficulty.spread))

    def _sample_params(self, label: Label, subtype: str) -> dict[str, float]:
        """
        Draw the growth parameters for one cascade.

        Ranges for the two classes overlap on purpose — see the module docstring.
        """
        rng = self.rng
        if label == "organic":
            params = {
                # Low root attachment: each new post reshares from somewhere in
                # the existing cascade, producing a deep random recursive tree.
                "root_attach": float(rng.uniform(0.05, 0.55)),
                "delay_median_s": self._lognormal(240.0, 0.9),
                "delay_sigma": float(rng.uniform(0.9, 1.4)),
                "bot_frac": float(rng.beta(2.0, 7.0)),
                "hop_prob": float(rng.uniform(0.05, 0.32)),
                "hop_lag_s": self._lognormal(1800.0, 0.95),
            }
            if subtype == "viral_organic":
                # Genuine virality: everyone piles onto the original post, fast.
                # Indistinguishable from a campaign to any rule that only looks
                # at speed or fan-out.
                params["root_attach"] = float(rng.uniform(0.25, 0.50))
                params["delay_median_s"] *= float(rng.uniform(0.08, 0.25))
                params["delay_sigma"] = float(rng.uniform(0.5, 0.9))
                params["hop_prob"] = float(rng.uniform(0.08, 0.20))
        else:
            params = {
                # High root attachment: a burst of accounts hitting the seed
                # directly, so the cascade is wide and shallow.
                "root_attach": float(rng.uniform(0.30, 0.90)),
                "delay_median_s": self._lognormal(45.0, 0.85),
                "delay_sigma": float(rng.uniform(0.25, 0.70)),
                "bot_frac": float(rng.beta(4.5, 3.0)),
                "hop_prob": float(rng.uniform(0.08, 0.40)),
                "hop_lag_s": self._lognormal(900.0, 0.9),
            }
            if subtype == "stealth_coordinated":
                # A campaign that paces itself, rotates accounts, and reshares
                # through intermediaries instead of swarming the seed.
                params["root_attach"] = float(rng.uniform(0.20, 0.45))
                params["delay_median_s"] *= float(rng.uniform(4.0, 12.0))
                params["delay_sigma"] = float(rng.uniform(0.9, 1.3))
                params["bot_frac"] = float(rng.beta(2.2, 3.0))

        # Target size is drawn from the *same* distribution for both classes and
        # growth continues until it is reached.  Letting the branching parameters
        # determine size instead would make raw cascade size a giveaway, and the
        # classifier would lean on it rather than on timing and structure.  The
        # floor of 8 guarantees every cascade has enough events for the
        # inter-arrival features to be defined.
        params["target_size"] = float(
            np.clip(int(self._lognormal(28.0, 0.75)), 8, 150)
        )
        return params

    def _pick_subtype(self, label: Label) -> str:
        if self.rng.random() < self.difficulty.hard_frac:
            return "viral_organic" if label == "organic" else "stealth_coordinated"
        return "plain"

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, label: Label) -> Cascade:
        """Grow one labeled cascade."""
        rng = self.rng
        subtype = self._pick_subtype(label)
        params = self._sample_params(label, subtype)

        t0 = datetime.now().astimezone() - timedelta(
            seconds=float(rng.uniform(0, self.window_days * 86400))
        )
        topic_idx = int(rng.integers(0, self._n_topics))
        topic_id = TOPIC_FMT.format(run=self.run_id, n=topic_idx)

        nodes: list[SimNode] = [
            SimNode(
                id=topic_id,
                type="named_entity",
                platform="bluesky",
                label=f"Simulated topic {topic_idx}",
                metadata={"synthetic": True, "ner_type": "ORG"},
            )
        ]
        edges: list[SimEdge] = []

        root_platform = str(rng.choice(PLATFORMS))
        root_id = self._next_content_id()
        self._add_content(nodes, edges, root_id, root_platform, t0, topic_id, params)

        # Grow one post at a time until the target size, choosing a parent for
        # each.  Attaching to the root produces a wide, shallow star; attaching
        # to an arbitrary earlier post produces a random recursive tree, whose
        # depth grows like log(n).  root_attach interpolates between the two and
        # is the only structural difference between the classes.
        target_size = int(params["target_size"])
        root_attach = params["root_attach"]
        # (content_id, platform, timestamp) for every post published so far.
        published: list[tuple[str, str, datetime]] = [(root_id, root_platform, t0)]

        while len(published) < target_size:
            if rng.random() < root_attach:
                parent_id, parent_platform, parent_ts = published[0]
            else:
                parent_id, parent_platform, parent_ts = published[
                    int(rng.integers(0, len(published)))
                ]

            delay = float(
                rng.lognormal(
                    mean=np.log(max(params["delay_median_s"], 1e-3)),
                    sigma=params["delay_sigma"],
                )
            )
            platform = parent_platform
            if rng.random() < params["hop_prob"]:
                platform = str(rng.choice([p for p in PLATFORMS if p != parent_platform]))
                delay += float(
                    rng.lognormal(mean=np.log(max(params["hop_lag_s"], 1e-3)), sigma=0.5)
                )

            child_ts = parent_ts + timedelta(seconds=delay)
            child_id = self._next_content_id()
            self._add_content(nodes, edges, child_id, platform, child_ts, topic_id, params)
            edges.append(
                SimEdge(
                    source_id=parent_id,
                    target_id=child_id,
                    edge_type="reshare",
                    platform=platform,
                    ts=child_ts,
                )
            )
            published.append((child_id, platform, child_ts))

        return Cascade(
            root_id=root_id,
            label=label,
            subtype=subtype,
            topic_id=topic_id,
            t0=t0,
            params=params,
            nodes=nodes,
            edges=edges,
        )

    def _add_content(
        self,
        nodes: list[SimNode],
        edges: list[SimEdge],
        content_id: str,
        platform: str,
        ts: datetime,
        topic_id: str,
        params: dict[str, float],
    ) -> None:
        """Append a content node plus its author and topic edges."""
        author_id = (
            self._bot_author() if self.rng.random() < params["bot_frac"] else self._human_author()
        )
        nodes.append(
            SimNode(
                id=content_id,
                type="content",
                platform=platform,
                label=f"Simulated post {content_id.rsplit(':', 1)[-1]}",
                metadata={"synthetic": True},
            )
        )
        if author_id not in self._emitted_authors:
            nodes.append(
                SimNode(
                    id=author_id,
                    type="author",
                    platform=platform,
                    label=author_id.rsplit(":", 1)[-1],
                    metadata={"synthetic": True},
                )
            )
            self._emitted_authors.add(author_id)

        edges.append(
            SimEdge(author_id, content_id, "authored", platform, ts)
        )
        edges.append(
            SimEdge(content_id, topic_id, "mentions", platform, ts)
        )
