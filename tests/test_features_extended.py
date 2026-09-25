"""Second-generation cascade features, on hand-built graphs with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    EXTENDED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    _root_attach_mle,
    _structural_virality,
    compute_features,
)

T0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _tree(rows, root="r", platform="hn"):
    """rows: (source, target, seconds_after_T0, depth)."""
    return pd.DataFrame(
        [
            {"root_id": root, "source_id": s, "target_id": t,
             "ts": T0 + pd.Timedelta(seconds=secs), "platform": platform, "depth": d}
            for s, t, secs, d in rows
        ]
    )


def _authors(pairs, root="r"):
    """pairs: (content_id, author_id, seconds_after_T0)."""
    return pd.DataFrame(
        [
            {"root_id": root, "content_id": c, "author_id": a,
             "ts": T0 + pd.Timedelta(seconds=secs), "platform": "hn"}
            for c, a, secs in pairs
        ]
    )


def test_extended_columns_are_part_of_the_model_input():
    assert set(EXTENDED_FEATURE_COLUMNS) <= set(FEATURE_COLUMNS)
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_structural_virality_of_a_star_and_a_chain():
    # Star, 3 leaves: 3 root-leaf pairs at distance 1, 3 leaf-leaf at 2 -> 9 / 6.
    star = _structural_virality(np.array(["r"] * 3), np.array(["a", "b", "c"]), "r")
    assert star == pytest.approx(1.5)
    # Chain of 4: distances 1,1,1,2,2,3 -> 10 / 6.
    chain = _structural_virality(np.array(["r", "a", "b"]), np.array(["a", "b", "c"]), "r")
    assert chain == pytest.approx(10 / 6)
    assert chain > star


def test_root_attach_mle_extremes():
    assert _root_attach_mle(np.array(["r"] * 6), "r") == pytest.approx(1.0)
    # Every post after the first attaches away from the seed.
    assert _root_attach_mle(np.array(["r", "a", "b", "c", "d"]), "r") == pytest.approx(0.0)
    assert _root_attach_mle(np.array([]), "r") == 0.0


def test_log_delay_spread_separates_synchronised_from_scattered():
    tight = _tree([("r", f"c{i}", 60 + i, 1) for i in range(6)], root="r")
    loose = _tree([("q", f"d{i}", 10 * 4**i, 1) for i in range(6)], root="q")
    authors = pd.concat(
        [
            _authors([("r", "a", 0)] + [(f"c{i}", f"u{i}", 60 + i) for i in range(6)], "r"),
            _authors([("q", "b", 0)] + [(f"d{i}", f"v{i}", 10 * 4**i) for i in range(6)], "q"),
        ]
    )
    f = compute_features(pd.concat([tight, loose]), authors).set_index("root_node_id")
    assert f.loc["r", "log_delay_std"] < 0.1
    assert f.loc["q", "log_delay_std"] > 1.0


def test_hop_lag_is_excluded_from_log_delay_spread():
    """A cross-platform hop adds its own lag; it must not read as dispersion."""
    tree = _tree([("r", "c1", 60, 1), ("r", "c2", 61, 1), ("r", "c3", 5000, 1)])
    tree.loc[2, "platform"] = "bluesky"
    authors = _authors([("r", "a0", 0), ("c1", "a1", 60), ("c2", "a2", 61), ("c3", "a3", 5000)])
    row = compute_features(tree, authors).iloc[0]
    assert row["log_delay_std"] < 0.05


def test_coauthor_features_see_a_group_reused_together():
    """Two accounts that co-appeared earlier are linked; a stranger is not."""
    first = _tree([("r1", "x1", 60, 1), ("r1", "x2", 70, 1)], root="r1")
    second = _tree([("r2", "y1", 3600, 1), ("r2", "y2", 3610, 1)], root="r2")
    authors = pd.concat(
        [
            _authors([("r1", "s0", 0), ("x1", "bot1", 60), ("x2", "bot2", 70)], "r1"),
            _authors([("r2", "s9", 3500), ("y1", "bot1", 3600), ("y2", "bot2", 3610)], "r2"),
        ]
    )
    f = compute_features(
        pd.concat([first, second]), authors, order=["r1", "r2"]
    ).set_index("root_node_id")
    assert f.loc["r1", "coauthor_linked_frac"] == 0.0
    # bot1 and bot2 are linked through r1; the new seed author s9 is not.
    assert f.loc["r2", "coauthor_linked_frac"] == pytest.approx(2 / 3)
    assert f.loc["r2", "coauthor_overlap_max"] == pytest.approx(2 / 3)


def test_coauthor_links_expire_with_the_window():
    first = _tree([("r1", "x1", 60, 1), ("r1", "x2", 70, 1)], root="r1")
    second = _tree([("r2", "y1", 86_400, 1), ("r2", "y2", 86_410, 1)], root="r2")
    authors = pd.concat(
        [
            _authors([("r1", "s0", 0), ("x1", "bot1", 60), ("x2", "bot2", 70)], "r1"),
            _authors([("r2", "s9", 86_300), ("y1", "bot1", 86_400), ("y2", "bot2", 86_410)], "r2"),
        ]
    )
    f = compute_features(
        pd.concat([first, second]), authors, order=["r1", "r2"]
    ).set_index("root_node_id")
    assert f.loc["r2", "coauthor_linked_frac"] == 0.0


def test_arrival_burstiness_is_bounded():
    tree = _tree([("r", f"c{i}", 1 + i**3, 1) for i in range(8)])
    authors = _authors([("r", "a", 0)] + [(f"c{i}", f"u{i}", 1 + i**3) for i in range(8)])
    row = compute_features(tree, authors).iloc[0]
    assert -1.0 <= row["arrival_burstiness"] <= 1.0
    assert -1.0 <= row["arrival_memory"] <= 1.0
