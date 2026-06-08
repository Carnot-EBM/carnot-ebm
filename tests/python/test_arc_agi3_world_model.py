"""M0 gate tests for the ARC-AGI-3 GameGraph world-model substrate.

Plan: docs/research-notes/arc-agi3-agent-research-plan.md M0. Asserts the load-bearing invariants:
a transition is logged with the correct deterministic delta; a deadly action is recorded and never
re-offered as untested; the graph round-trips through JSON; perception primitives are deterministic.
Every test asserts (no skips) per CLAUDE.md "Tests Must Run and Assert".
"""

import numpy as np

from carnot.agentic.arc_agi3_world_model import (
    GameGraph, frame_hash, compute_grid_delta, objects, action_key)


def test_compute_grid_delta_counts_changed_cells():
    a = np.zeros((4, 4), dtype=np.int16)
    b = a.copy()
    b[1, 2] = 5
    d = compute_grid_delta(a, b)
    assert d["n_changed"] == 1
    assert d["cells"] == [(1, 2)]
    assert d["transitions"] == [(0, 5)]
    assert compute_grid_delta(a, a)["n_changed"] == 0  # no-effect action detectable


def test_frame_hash_stable_and_distinct():
    a = np.zeros((3, 3), dtype=np.int16)
    b = a.copy(); b[0, 0] = 1
    assert frame_hash(a) == frame_hash(a.copy())   # stable
    assert frame_hash(a) != frame_hash(b)          # distinct content -> distinct node


def test_objects_finds_components_and_action_key():
    g = np.zeros((5, 5), dtype=np.int16)
    g[0, 0] = 2          # one object top-left
    g[4, 4] = 3          # one object bottom-right
    objs = objects(g)
    assert len(objs) == 2
    assert action_key(6, {"x": 4, "y": 2}) == (6, 4, 2)
    assert action_key(3, None) == (3,)


def test_transition_logged_with_delta():
    g = GameGraph("test")
    a = np.zeros((4, 4), dtype=np.int16); b = a.copy(); b[2, 2] = 7
    g.record(frame_hash(a), (6, 2, 2), frame_hash(b), compute_grid_delta(a, b),
             level_delta=0, game_over=False)
    assert len(g.transition_store) == 1
    assert g.transition_store[0]["n_changed"] == 1
    assert g.tried(frame_hash(a), (6, 2, 2))


def test_deadly_action_never_untested():
    g = GameGraph("test")
    fh = frame_hash(np.zeros((4, 4), dtype=np.int16))
    nxt = frame_hash(np.ones((4, 4), dtype=np.int16))
    g.record(fh, (6, 1, 1), nxt, {"n_changed": 16, "cells": []}, level_delta=0, game_over=True)
    assert g.is_deadly(fh, (6, 1, 1))
    # a deadly action is excluded from untested candidates
    assert (6, 1, 1) not in g.untested(fh, [(6, 1, 1), (6, 2, 2)])
    assert (6, 2, 2) in g.untested(fh, [(6, 1, 1), (6, 2, 2)])


def test_graph_json_round_trip(tmp_path):
    g = GameGraph("vc33")
    a = np.zeros((4, 4), dtype=np.int16); b = a.copy(); b[0, 1] = 4
    g.record(frame_hash(a), (1,), frame_hash(b), compute_grid_delta(a, b), 0, False)
    g.deadly.add(g._ek(frame_hash(a), (2,)))
    p = tmp_path / "wm.json"
    g.persist(p)
    g2 = GameGraph.load(p)
    assert g2.game_id == "vc33"
    assert g2.is_deadly(frame_hash(a), (2,))
    assert g2.tried(frame_hash(a), (1,))


def test_shortest_path_action_navigates_to_frontier():
    g = GameGraph("nav")
    s0, s1, s2 = "aa", "bb", "cc"
    g.edges[g._ek(s0, (1,))] = {"from": s0, "akey": [1], "to": s1, "n_changed": 1,
                                "level_delta": 0, "game_over": False, "count": 1}
    g.edges[g._ek(s1, (2,))] = {"from": s1, "akey": [2], "to": s2, "n_changed": 1,
                                "level_delta": 0, "game_over": False, "count": 1}
    first = g.shortest_path_action(s0, {s2})
    assert first == (1,)   # the first hop toward the goal frontier
    assert g.shortest_path_action(s0, {"zz"}) is None  # unreachable
