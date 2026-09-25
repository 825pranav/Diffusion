"""
A graph neural network baseline for cascade classification.

The LightGBM classifier reads hand-built summaries of each cascade.  The obvious
objection is that the summaries might be leaving signal on the table, and that a
model reading the reshare tree directly would find it.  This module answers that
objection by building one: a Graph Isomorphism Network (Xu et al., 2019) over
each cascade's reshare tree, in plain PyTorch so it runs on a CPU with no
extension packages.

Node inputs are deliberately the *same information* the tabular model has —
timing, depth, platform hops, and causal author reuse — but per post rather than
aggregated, so a difference in score is a difference in how the information is
used, not in what information is available.

  is_root, depth, log parent->child delay, log time since the seed, platform hop,
  log out-degree, log prior-cascade count of the post's author (trailing window,
  causal), and whether the author already posted earlier in this cascade.

Messages flow along reshare edges in both directions (a parent learns about its
replies and a reply about its parent), then mean- and max-pooled over the tree at
every layer, jumping-knowledge style, so shallow and deep patterns both reach the
readout.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.features import AUTHOR_WINDOW_SECONDS, to_epoch_seconds

log = logging.getLogger(__name__)

NODE_FEATURES = [
    "is_root",
    "depth",
    "log_delay",
    "log_since_root",
    "hop",
    "log_out_degree",
    "log_prior_author",
    "author_repeat",
]


@dataclass
class CascadeGraph:
    root_id: str
    x: np.ndarray        # [n_nodes, len(NODE_FEATURES)]
    edges: np.ndarray    # [2, n_edges] parent -> child, local indices
    log_size: float


def build_graphs(tree: pd.DataFrame, authors: pd.DataFrame) -> dict[str, CascadeGraph]:
    """One `CascadeGraph` per root, from the frames `ml.features.fetch_frames` returns."""
    if tree.empty:
        return {}
    tree = tree.copy()
    tree["epoch"] = to_epoch_seconds(tree["ts"])
    authors = authors.copy()
    authors["epoch"] = to_epoch_seconds(authors["ts"])

    starts = tree.groupby("root_id")["epoch"].min().sort_values()
    author_of = dict(zip(authors["content_id"], authors["author_id"], strict=False))
    seeds = authors[authors["content_id"] == authors["root_id"]]
    seed_epoch = seeds.groupby("root_id")["epoch"].min().to_dict()
    seed_platform = seeds.groupby("root_id")["platform"].first().to_dict()
    authors_by_root = {r: set(g["author_id"]) for r, g in authors.groupby("root_id")}
    trees = {r: g for r, g in tree.groupby("root_id", sort=False)}

    # Causal, windowed author reuse — the same walk as ml.features._author_features.
    recent: Counter[str] = Counter()
    history: deque[tuple[float, set[str]]] = deque()
    graphs: dict[str, CascadeGraph] = {}

    for root_id, now in starts.items():
        while history and history[0][0] < now - AUTHOR_WINDOW_SECONDS:
            _, expired = history.popleft()
            for a in expired:
                recent[a] -= 1
                if recent[a] <= 0:
                    del recent[a]

        g = trees[root_id].sort_values("epoch")
        t0 = seed_epoch.get(root_id, float(g["epoch"].min()))
        nodes = [root_id, *g["target_id"].tolist()]
        index = {n: i for i, n in enumerate(nodes)}
        epoch = {root_id: t0, **dict(zip(g["target_id"], g["epoch"], strict=True))}
        platform = {
            root_id: seed_platform.get(root_id, g["platform"].iloc[0]),
            **dict(zip(g["target_id"], g["platform"], strict=True)),
        }
        depth = {root_id: 0, **dict(zip(g["target_id"], g["depth"], strict=True))}
        parent = dict(zip(g["target_id"], g["source_id"], strict=True))
        out_degree = g["source_id"].value_counts().to_dict()

        seen_in_cascade: set[str] = set()
        x = np.zeros((len(nodes), len(NODE_FEATURES)), dtype=np.float32)
        for i, node in enumerate(nodes):
            author = author_of.get(node)
            p = parent.get(node)
            delay = epoch[node] - epoch[p] if p is not None and p in epoch else 0.0
            x[i] = [
                1.0 if i == 0 else 0.0,
                depth[node] / 5.0,
                np.log1p(max(delay, 0.0)) / 5.0,
                np.log1p(max(epoch[node] - t0, 0.0)) / 5.0,
                1.0 if p is not None and platform.get(p) != platform[node] else 0.0,
                np.log1p(out_degree.get(node, 0)),
                np.log1p(recent.get(author, 0)) if author else 0.0,
                1.0 if author in seen_in_cascade else 0.0,
            ]
            if author:
                seen_in_cascade.add(author)

        edges = np.array(
            [[index[s] for s in g["source_id"]], [index[t] for t in g["target_id"]]],
            dtype=np.int64,
        )
        graphs[root_id] = CascadeGraph(root_id, x, edges, float(np.log(len(nodes))))

        cascade_authors = authors_by_root.get(root_id, set())
        for a in cascade_authors:
            recent[a] += 1
        history.append((now, cascade_authors))

    return graphs


# ── model ─────────────────────────────────────────────────────────────────────


def _torch():
    import torch

    return torch


def _collate(graphs: list[CascadeGraph], torch):
    xs, srcs, dsts, batch = [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g.x)
        srcs.append(g.edges[0] + offset)
        dsts.append(g.edges[1] + offset)
        batch.append(np.full(len(g.x), i, dtype=np.int64))
        offset += len(g.x)
    x = torch.from_numpy(np.concatenate(xs))
    src = torch.from_numpy(np.concatenate(srcs))
    dst = torch.from_numpy(np.concatenate(dsts))
    b = torch.from_numpy(np.concatenate(batch))
    log_size = torch.tensor([[g.log_size] for g in graphs], dtype=torch.float32)
    return x, src, dst, b, log_size


def _build_module(n_in: int, hidden: int, layers: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class GIN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(n_in, hidden)
            self.eps = nn.Parameter(torch.zeros(layers))
            # Separate transforms for messages from parents and from children:
            # direction matters in a reshare tree.
            self.up = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
            self.down = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
            self.mlps = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                )
                for _ in range(layers)
            )
            readout = 2 * hidden * (layers + 1) + 1
            self.head = nn.Sequential(
                nn.Linear(readout, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
            )

        def forward(self, x, src, dst, batch, log_size, n_graphs):
            h = torch.relu(self.inp(x))
            pooled = [self._pool(h, batch, n_graphs)]
            for k in range(len(self.mlps)):
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, self.down[k](h)[src])   # parent -> child
                agg.index_add_(0, src, self.up[k](h)[dst])     # child -> parent
                h = self.mlps[k]((1 + self.eps[k]) * h + agg)
                pooled.append(self._pool(h, batch, n_graphs))
            return self.head(torch.cat([*pooled, log_size], dim=1)).squeeze(1)

        @staticmethod
        def _pool(h, batch, n_graphs):
            counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(1)
            mean = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h) / counts
            mx = torch.full((n_graphs, h.shape[1]), -1e9).scatter_reduce(
                0, batch.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=True
            )
            return torch.cat([mean, mx], dim=1)

    return GIN()


class GNNClassifier:
    """Fit/predict wrapper so the GNN drops into the same CV loop as LightGBM."""

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.epochs, self.batch_size = epochs, batch_size
        self.lr, self.weight_decay, self.seed = lr, weight_decay, seed
        self.module = None
        self._mean = self._std = None

    def _standardise(self, graphs: list[CascadeGraph]) -> list[CascadeGraph]:
        return [
            CascadeGraph(g.root_id, (g.x - self._mean) / self._std, g.edges, g.log_size)
            for g in graphs
        ]

    def fit(self, graphs: list[CascadeGraph], y: np.ndarray) -> GNNClassifier:
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        stacked = np.concatenate([g.x for g in graphs])
        self._mean = stacked.mean(axis=0)
        self._std = stacked.std(axis=0) + 1e-6
        graphs = self._standardise(graphs)
        self.module = _build_module(len(NODE_FEATURES), self.hidden, self.layers, self.dropout)
        opt = torch.optim.AdamW(
            self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        y = np.asarray(y, dtype=np.float32)
        for _ in range(self.epochs):
            self.module.train()
            order = rng.permutation(len(graphs))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                if len(idx) < 2:
                    continue
                x, src, dst, b, ls = _collate([graphs[i] for i in idx], torch)
                logits = self.module(x, src, dst, b, ls, len(idx))
                loss = loss_fn(logits, torch.from_numpy(y[idx]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    def predict_proba(self, graphs: list[CascadeGraph]) -> np.ndarray:
        torch = _torch()
        graphs = self._standardise(graphs)
        self.module.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(graphs), 256):
                chunk = graphs[start : start + 256]
                x, src, dst, b, ls = _collate(chunk, torch)
                out.append(torch.sigmoid(self.module(x, src, dst, b, ls, len(chunk))).numpy())
        p = np.concatenate(out)
        return np.column_stack([1 - p, p])
