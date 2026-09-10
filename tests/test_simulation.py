"""
Cascade simulator invariants.

The point of these is not that the generator runs, but that it keeps the
properties the whole evaluation rests on: reproducibility, no label leakage into
the graph, and two classes that genuinely overlap.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.background import generate_background
from simulation.cascade import DIFFICULTIES, CascadeSimulator


def _sim(seed: int = 0, difficulty: str = "medium") -> CascadeSimulator:
    return CascadeSimulator(
        rng=np.random.default_rng(seed), run_id="t", difficulty=difficulty
    )


def test_same_seed_reproduces_the_same_cascade():
    a = _sim(1).generate("organic")
    b = _sim(1).generate("organic")
    assert a.root_id == b.root_id
    assert [e.target_id for e in a.edges] == [e.target_id for e in b.edges]
    assert a.params == b.params


def test_cascade_reaches_its_target_size():
    for label in ("organic", "coordinated"):
        cascade = _sim(2).generate(label)
        assert cascade.size == int(cascade.params["target_size"])


def test_target_size_never_falls_below_the_feature_floor():
    """Below a handful of posts the timing features are undefined."""
    sim = _sim(3)
    for _ in range(50):
        assert sim.generate("organic").size >= 8


def test_every_post_has_an_author_and_a_topic():
    cascade = _sim(4).generate("coordinated")
    content = {n.id for n in cascade.nodes if n.type == "content"}
    authored = {e.target_id for e in cascade.edges if e.edge_type == "authored"}
    mentioned = {e.source_id for e in cascade.edges if e.edge_type == "mentions"}
    assert content == authored
    assert content == mentioned


def test_reshare_edges_form_a_tree_rooted_at_the_seed():
    cascade = _sim(5).generate("organic")
    reshares = [e for e in cascade.edges if e.edge_type == "reshare"]
    # Every post except the root has exactly one parent.
    assert len(reshares) == cascade.size - 1
    assert len({e.target_id for e in reshares}) == len(reshares)
    assert cascade.root_id not in {e.target_id for e in reshares}


def test_children_never_precede_their_parent():
    cascade = _sim(6).generate("coordinated")
    ts = {cascade.root_id: cascade.t0}
    for edge in cascade.edges:
        if edge.edge_type == "reshare":
            ts[edge.target_id] = edge.ts
    for edge in cascade.edges:
        if edge.edge_type == "reshare" and edge.source_id in ts:
            assert edge.ts >= ts[edge.source_id]


def test_the_graph_never_reveals_the_label():
    """
    The label must exist only in the labels CSV.

    Anything in a node id, label, or metadata field would be read straight back
    out by feature extraction.
    """
    for label in ("organic", "coordinated"):
        cascade = _sim(7).generate(label)
        blob = " ".join(
            f"{n.id} {n.label} {n.metadata}" for n in cascade.nodes
        ) + " ".join(f"{e.source_id} {e.target_id} {e.edge_type}" for e in cascade.edges)
        assert "organic" not in blob
        assert "coordinated" not in blob


def test_author_ids_do_not_mark_bots():
    """Reuse has to be recovered from structure, not read off an id."""
    cascade = _sim(8).generate("coordinated")
    for node in cascade.nodes:
        if node.type == "author":
            assert "bot" not in node.id.lower()


def test_classes_overlap_on_every_parameter():
    """
    Cleanly separated parameters would produce a meaningless F1 near 1.0.

    Each class's range must reach into the other's.
    """
    sim = _sim(9)
    organic = [sim.generate("organic").params for _ in range(200)]
    coordinated = [sim.generate("coordinated").params for _ in range(200)]

    for key in ("root_attach", "delay_median_s", "bot_frac", "hop_prob"):
        o = np.array([p[key] for p in organic])
        c = np.array([p[key] for p in coordinated])
        assert o.max() > c.min(), f"{key} ranges do not overlap"


def test_harder_difficulty_produces_more_confusable_cases():
    counts = {}
    for name in ("easy", "medium", "hard"):
        sim = _sim(10, name)
        cascades = [sim.generate("organic") for _ in range(200)]
        counts[name] = sum(1 for c in cascades if c.subtype != "plain")
    assert counts["easy"] < counts["medium"] < counts["hard"]


def test_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError, match="unknown difficulty"):
        CascadeSimulator(rng=np.random.default_rng(0), run_id="t", difficulty="nope")


def test_difficulty_presets_are_ordered():
    assert DIFFICULTIES["easy"].hard_frac < DIFFICULTIES["hard"].hard_frac
    assert DIFFICULTIES["easy"].spread < DIFFICULTIES["hard"].spread


def test_background_events_belong_to_no_cascade():
    """Background traffic must raise velocity without joining a reshare tree."""
    import datetime as dt

    nodes, edges = generate_background(
        rng=np.random.default_rng(0),
        run_id="t",
        topic_ids=["sim:t:e:0"],
        window_days=1,
        end_time=dt.datetime.now(dt.UTC),
    )
    assert nodes and edges
    assert {e.edge_type for e in edges} == {"mentions"}
    assert all(n.metadata.get("background") for n in nodes)


def test_background_follows_a_daily_cycle():
    """The diurnal swing is what makes the baseline non-stationary."""
    import datetime as dt

    _, edges = generate_background(
        rng=np.random.default_rng(1),
        run_id="t",
        # One topic only: each gets its own random phase, so summing several
        # cancels the cycle out entirely.
        topic_ids=["sim:t:e:0"],
        window_days=6,
        end_time=dt.datetime.now(dt.UTC),
        rate_per_day=3000.0,
    )
    hours = np.array([e.ts.hour for e in edges])
    counts = np.bincount(hours, minlength=24)
    # A flat process would show no meaningful peak-to-trough spread.
    assert counts.max() > 1.3 * counts.min()
