"""
In-memory cascade datasets — the simulator without the database round-trip.

`ml.dataset.load_dataset` reads cascades back out of Postgres, which is the
right path for anything the running system will see.  For experiments it is a
tax: every variant of the simulator (a harder difficulty, a more camouflaged
campaign) would have to be bulk-loaded, featurised through recursive SQL, and
cleaned up again.

This module builds the *same* two frames the SQL in `ml.features.fetch_frames`
returns — reshare edges tagged with root and depth, and authorship for every
post in each tree — straight from `Cascade` objects, so the feature code is
shared rather than reimplemented.  `tests/test_offline.py` pins the equivalence
on a small dataset, and `python -m ml.offline --check` compares it against the
database copy of the training data.

Generation is deterministic given a seed, and the default arguments reproduce
`python -m simulation.generate --n 2000` exactly: same rng stream, same root
ids, same labels.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.dataset import POSITIVE_CLASS
from ml.features import compute_features
from simulation.cascade import DEFAULT_DIFFICULTY, Cascade, CascadeSimulator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Shift:
    """
    A perturbation of the simulator, for testing generalisation.

    The classifier is trained on one simulator configuration; a shift asks how
    it holds up when the world differs from that configuration in a specific,
    adversarially motivated direction.  Every field defaults to "no change".

      hard_frac          share of each class generated as its confusable subtype
      coord_delay_mult   multiplies coordinated delay medians (slower campaigns)
      coord_sigma_add    widens coordinated delay dispersion (less synchrony)
      coord_bot_mult     scales coordinated bot fraction (more human accounts)
      coord_attach_mult  scales coordinated root attachment (less star-shaped)
      n_bots             size of the bot pool (bigger = less author reuse)
    """

    name: str = "in-distribution"
    hard_frac: float | None = None
    coord_delay_mult: float = 1.0
    coord_sigma_add: float = 0.0
    coord_bot_mult: float = 1.0
    coord_attach_mult: float = 1.0
    n_bots: int = 400
    notes: str = ""
    extra: dict = field(default_factory=dict)


class ShiftedSimulator(CascadeSimulator):
    """A `CascadeSimulator` whose coordinated parameters are perturbed by a `Shift`."""

    def __init__(self, *args, shift: Shift, **kwargs) -> None:
        super().__init__(*args, n_bots=shift.n_bots, **kwargs)
        self.shift = shift
        if shift.hard_frac is not None:
            # Difficulty is a frozen dataclass; swap in a copy with the new mix.
            from dataclasses import replace

            self.difficulty = replace(self.difficulty, hard_frac=shift.hard_frac)

    def _sample_params(self, label, subtype):  # noqa: D102 - documented on the base class
        params = super()._sample_params(label, subtype)
        if label == "coordinated":
            s = self.shift
            params["delay_median_s"] *= s.coord_delay_mult
            params["delay_sigma"] += s.coord_sigma_add
            params["bot_frac"] = float(np.clip(params["bot_frac"] * s.coord_bot_mult, 0.0, 1.0))
            params["root_attach"] = float(
                np.clip(params["root_attach"] * s.coord_attach_mult, 0.0, 1.0)
            )
        return params


def simulate(
    n: int = 2000,
    difficulty: str = DEFAULT_DIFFICULTY,
    seed: int = 42,
    shift: Shift | None = None,
) -> list[Cascade]:
    """
    Generate `n` labeled cascades, replaying `simulation.generate.run`'s rng use.

    With no shift this is the exact dataset the database holds for the same
    arguments (background chatter is drawn *after* the cascades, so it does not
    disturb the stream).
    """
    rng = np.random.default_rng(seed)
    run_id = f"{difficulty[0]}{seed}"
    if shift is None:
        sim = CascadeSimulator(rng=rng, run_id=run_id, difficulty=difficulty)
    else:
        sim = ShiftedSimulator(rng=rng, run_id=run_id, difficulty=difficulty, shift=shift)
    labels = ["organic"] * (n // 2) + ["coordinated"] * (n - n // 2)
    rng.shuffle(labels)
    return [sim.generate(label) for label in labels]


def cascade_frames(cascades: list[Cascade]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The (tree, authors) frames `ml.features.fetch_frames` would return.

    Depth is recomputed by walking parent links, as the recursive CTE does, so
    nothing here reads a stored attribute the database copy would not have.
    """
    tree_rows: list[tuple] = []
    author_rows: list[tuple] = []
    for c in cascades:
        depth = {c.root_id: 0}
        for e in c.edges:
            if e.edge_type != "reshare":
                continue
            # Parents always precede children in generation order.
            depth[e.target_id] = depth[e.source_id] + 1
            tree_rows.append((c.root_id, e.source_id, e.target_id, e.ts, e.platform,
                              depth[e.target_id]))
        for e in c.edges:
            if e.edge_type == "authored" and e.target_id in depth:
                author_rows.append((c.root_id, e.target_id, e.source_id, e.ts, e.platform))

    tree = pd.DataFrame(
        tree_rows, columns=["root_id", "source_id", "target_id", "ts", "platform", "depth"]
    )
    authors = pd.DataFrame(
        author_rows, columns=["root_id", "content_id", "author_id", "ts", "platform"]
    )
    for frame in (tree, authors):
        if not frame.empty:
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.as_unit("us")
    return tree, authors


def build_dataset(cascades: list[Cascade]) -> pd.DataFrame:
    """Labels joined to features, sorted by start time — `load_dataset`'s shape."""
    tree, authors = cascade_frames(cascades)
    features = compute_features(tree, authors)
    labels = pd.DataFrame(
        {
            "root_node_id": [c.root_id for c in cascades],
            "label": [c.label for c in cascades],
            "subtype": [c.subtype for c in cascades],
            "topic_id": [c.topic_id for c in cascades],
            "t0": pd.to_datetime([c.t0 for c in cascades], utc=True),
        }
    )
    merged = labels.merge(features, on="root_node_id", how="inner")
    merged["y"] = (merged["label"] == POSITIVE_CLASS).astype(int)
    return merged.sort_values("t0").reset_index(drop=True)


def load_offline_dataset(
    n: int = 2000,
    difficulty: str = DEFAULT_DIFFICULTY,
    seed: int = 42,
    shift: Shift | None = None,
) -> pd.DataFrame:
    """Simulate and featurise in one call."""
    return build_dataset(simulate(n, difficulty, seed, shift))


async def _check_against_database() -> None:
    """Compare the in-memory features with the database copy of the training data."""
    from ml.dataset import load_dataset
    from ml.features import FEATURE_COLUMNS

    db = await load_dataset()
    mem = load_offline_dataset()
    if set(db["root_node_id"]) != set(mem["root_node_id"]):
        raise SystemExit("root ids differ — the database holds a different dataset")
    db = db.set_index("root_node_id").sort_index()
    mem = mem.set_index("root_node_id").sort_index()
    worst = 0.0
    for col in FEATURE_COLUMNS:
        diff = np.abs(db[col].to_numpy(float) - mem[col].to_numpy(float))
        scale = np.maximum(np.abs(db[col].to_numpy(float)), 1.0)
        worst = max(worst, float((diff / scale).max()))
        if not np.allclose(db[col], mem[col], rtol=1e-6, atol=1e-6):
            log.error("feature %s differs: max abs diff %.6g", col, diff.max())
    labels_match = (db["label"] == mem["label"]).all()
    log.info(
        "checked %d cascades x %d features — worst relative diff %.2e, labels match: %s",
        len(db), len(FEATURE_COLUMNS), worst, labels_match,
    )


def main() -> None:
    from config import configure_logging

    parser = argparse.ArgumentParser(description="In-memory simulator datasets")
    parser.add_argument("--check", action="store_true", help="compare with the database copy")
    args = parser.parse_args()
    configure_logging()
    if args.check:
        asyncio.run(_check_against_database())
    else:
        data = load_offline_dataset()
        log.info("built %d rows, %.1f%% coordinated", len(data), 100 * data["y"].mean())


if __name__ == "__main__":
    main()
