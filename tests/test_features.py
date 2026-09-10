"""Cascade feature extraction, on hand-built graphs with known answers."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, compute_features, to_epoch_seconds

T0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _tree(rows: list[tuple[str, str, int, int]], root: str = "r") -> pd.DataFrame:
    """rows: (source, target, seconds_after_T0, depth)."""
    return pd.DataFrame(
        [
            {
                "root_id": root,
                "source_id": s,
                "target_id": t,
                "ts": T0 + pd.Timedelta(seconds=secs),
                "platform": "hn",
                "depth": depth,
            }
            for s, t, secs, depth in rows
        ]
    )


def _authors(pairs: list[tuple[str, str, int]], root: str = "r") -> pd.DataFrame:
    """pairs: (content_id, author_id, seconds_after_T0)."""
    return pd.DataFrame(
        [
            {
                "root_id": root,
                "content_id": c,
                "author_id": a,
                "ts": T0 + pd.Timedelta(seconds=secs),
                "platform": "hn",
            }
            for c, a, secs in pairs
        ]
    )


def test_star_cascade_shape():
    """Three posts all resharing the seed: wide, shallow, fully root-attached."""
    tree = _tree([("r", "c1", 60, 1), ("r", "c2", 120, 1), ("r", "c3", 180, 1)])
    authors = _authors([("r", "a0", 0), ("c1", "a1", 60), ("c2", "a2", 120), ("c3", "a3", 180)])

    row = compute_features(tree, authors).iloc[0]
    assert row["size"] == 4
    assert row["max_depth"] == 1
    assert row["root_fanout"] == 3
    assert row["root_fanout_share"] == 1.0
    assert row["leaf_frac"] == pytest.approx(0.75)


def test_chain_cascade_shape():
    """A chain of reshares: narrow and deep, barely root-attached."""
    tree = _tree([("r", "c1", 60, 1), ("c1", "c2", 120, 2), ("c2", "c3", 180, 3)])
    authors = _authors([("r", "a0", 0), ("c1", "a1", 60), ("c2", "a2", 120), ("c3", "a3", 180)])

    row = compute_features(tree, authors).iloc[0]
    assert row["size"] == 4
    assert row["max_depth"] == 3
    assert row["root_fanout"] == 1
    assert row["root_fanout_share"] == pytest.approx(1 / 3)


def test_delays_are_measured_from_the_true_root_time():
    """
    The seed's own timestamp comes from its authored edge.

    Falling back to the earliest child would force the first root-attached child
    to a delay of zero, and root attachment is exactly what separates the classes.
    """
    tree = _tree([("r", "c1", 60, 1), ("r", "c2", 120, 1)])
    authors = _authors([("r", "a0", 0), ("c1", "a1", 60), ("c2", "a2", 120)])

    row = compute_features(tree, authors).iloc[0]
    # Delays of 60s and 120s from the root, not 0s and 60s from the first child.
    assert row["delay_median_s"] == pytest.approx(90.0)
    assert row["duration_s"] == pytest.approx(120.0)


def test_author_reuse_is_causal():
    """
    A cascade may only see authors from cascades that started before it.

    Counting reuse across the whole dataset lets a cascade see campaigns that ran
    after it, which inflates the feature and the score with it.
    """
    early = _tree([("r1", "x1", 60, 1)], root="r1")
    late = _tree([("r2", "y1", 60, 1)], root="r2")
    tree = pd.concat([early, late], ignore_index=True)

    authors = pd.concat(
        [
            _authors([("r1", "shared", 0), ("x1", "shared", 60)], root="r1"),
            _authors([("r2", "shared", 3600), ("y1", "shared", 3660)], root="r2"),
        ],
        ignore_index=True,
    )

    features = compute_features(tree, authors, order=["r1", "r2"]).set_index("root_node_id")
    # The earlier cascade has no history behind it; the later one does.
    assert features.loc["r1", "prior_author_frac"] == 0.0
    assert features.loc["r2", "prior_author_frac"] == 1.0


def test_repeated_authors_lower_the_unique_ratio():
    tree = _tree([("r", "c1", 60, 1), ("r", "c2", 120, 1), ("r", "c3", 180, 1)])
    authors = _authors([("r", "bot", 0), ("c1", "bot", 60), ("c2", "bot", 120), ("c3", "a3", 180)])

    row = compute_features(tree, authors).iloc[0]
    assert row["unique_author_ratio"] == pytest.approx(0.5)
    assert row["max_author_share"] == pytest.approx(0.75)


def test_cross_platform_hop_and_sentinel():
    tree = _tree([("r", "c1", 60, 1), ("c1", "c2", 120, 2)])
    tree.loc[1, "platform"] = "bluesky"
    authors = _authors([("r", "a0", 0), ("c1", "a1", 60), ("c2", "a2", 120)])

    row = compute_features(tree, authors).iloc[0]
    assert row["n_platforms"] == 2
    assert row["cross_platform_edge_frac"] == pytest.approx(0.5)
    assert row["first_hop_lag_s"] == pytest.approx(120.0)


def test_no_hop_uses_a_negative_sentinel():
    """NaN would be imputed and read as a genuinely short lag."""
    tree = _tree([("r", "c1", 60, 1)])
    authors = _authors([("r", "a0", 0), ("c1", "a1", 60)])
    assert compute_features(tree, authors).iloc[0]["first_hop_lag_s"] == -1.0


def test_empty_input_yields_the_full_column_set():
    empty = compute_features(pd.DataFrame(), pd.DataFrame())
    assert list(empty.columns) == ["root_node_id", *FEATURE_COLUMNS]
    assert len(empty) == 0


def test_every_feature_column_is_populated():
    tree = _tree([("r", "c1", 60, 1), ("c1", "c2", 120, 2)])
    authors = _authors([("r", "a0", 0), ("c1", "a1", 60), ("c2", "a2", 120)])
    features = compute_features(tree, authors)
    assert list(features.columns) == ["root_node_id", *FEATURE_COLUMNS]
    assert features[FEATURE_COLUMNS].notna().all().all()


def test_epoch_conversion_is_resolution_independent():
    """
    asyncpg returns microsecond-resolution timestamps.

    Reading them as nanoseconds silently rescales every duration by 1000.
    """
    micros = pd.Series(pd.to_datetime([T0, T0 + pd.Timedelta(seconds=60)])).dt.as_unit("us")
    nanos = micros.dt.as_unit("ns")
    assert to_epoch_seconds(micros).tolist() == to_epoch_seconds(nanos).tolist()
    assert to_epoch_seconds(micros)[1] - to_epoch_seconds(micros)[0] == pytest.approx(60.0)
