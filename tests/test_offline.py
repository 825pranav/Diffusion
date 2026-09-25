"""In-memory datasets and the graph builder for the GNN baseline."""

from __future__ import annotations

import numpy as np
import pytest

from ml.features import FEATURE_COLUMNS
from ml.gnn import NODE_FEATURES, GNNClassifier, build_graphs
from ml.offline import Shift, build_dataset, cascade_frames, simulate


@pytest.fixture(scope="module")
def cascades():
    return simulate(n=60, seed=3)


def test_simulation_is_deterministic(cascades):
    again = simulate(n=60, seed=3)
    assert [c.root_id for c in cascades] == [c.root_id for c in again]
    assert [c.label for c in cascades] == [c.label for c in again]


def test_frames_match_the_sql_shape(cascades):
    tree, authors = cascade_frames(cascades)
    assert list(tree.columns) == ["root_id", "source_id", "target_id", "ts", "platform", "depth"]
    assert list(authors.columns) == ["root_id", "content_id", "author_id", "ts", "platform"]
    # One reshare edge per non-root post, and one author per post including the seed.
    assert len(tree) == sum(c.size - 1 for c in cascades)
    assert len(authors) == sum(c.size for c in cascades)
    # Depth is recomputed from parent links: root children sit at depth 1.
    roots = {c.root_id for c in cascades}
    assert (tree.loc[tree["source_id"].isin(roots), "depth"] == 1).all()


def test_dataset_has_every_feature_and_a_label(cascades):
    data = build_dataset(cascades)
    assert len(data) == len(cascades)
    assert data[FEATURE_COLUMNS].notna().all().all()
    assert set(data["y"]) <= {0, 1}
    assert data["t0"].is_monotonic_increasing


def test_shift_changes_only_coordinated_parameters():
    slow = simulate(n=200, seed=5, shift=Shift("slow", coord_delay_mult=10.0))
    base = simulate(n=200, seed=5)
    for a, b in zip(base, slow, strict=True):
        if a.label == "organic":
            assert a.params == b.params
        else:
            assert b.params["delay_median_s"] == pytest.approx(10 * a.params["delay_median_s"])


def test_graphs_cover_every_post(cascades):
    tree, authors = cascade_frames(cascades)
    graphs = build_graphs(tree, authors)
    for c in cascades:
        g = graphs[c.root_id]
        assert g.x.shape == (c.size, len(NODE_FEATURES))
        assert g.edges.shape == (2, c.size - 1)
        assert g.x[0, 0] == 1.0 and g.x[1:, 0].sum() == 0.0  # only the seed is the root


def test_gnn_fits_and_predicts_probabilities(cascades):
    pytest.importorskip("torch")  # experiments-only dependency, not in CI's pin set
    tree, authors = cascade_frames(cascades)
    graphs = build_graphs(tree, authors)
    ordered = [graphs[c.root_id] for c in cascades]
    y = np.array([int(c.label == "coordinated") for c in cascades])
    model = GNNClassifier(epochs=2, hidden=8, layers=2).fit(ordered, y)
    p = model.predict_proba(ordered)
    assert p.shape == (len(cascades), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
