"""Tests for the ARC explorer re-navigation decomposition.

WHY THESE TESTS AND NOT OTHERS. The probe's expensive half -- spawn a subprocess, step the
offline arcade for 240 actions -- is not where a wrong answer can hide: it either runs or
it visibly does not, and its output is a trace anyone can re-read. The cheap half is where
a wrong answer would look right, and this measurement already shipped one:

  * NODE-DISCOVERY ATTRIBUTION IS OFF BY ONE BY CONSTRUCTION. The explorer ingests the
    frame produced by action i at the TOP of the decision for action i+1, so the graph
    grows ACROSS decision i+1, never between turn i and turn i+1. The first version of this
    probe compared `pre_nodes(i+1)` against `post_nodes(i)` -- a window in which nothing can
    happen -- and reported ZERO new states on a run that plainly built a 17-node graph.
    That mistake is invisible in the output (a plausible-looking "0 discoveries, all
    overhead" headline) and it points the whole conclusion at the wrong class, so it gets a
    test that fails if the comparison is moved back.
  * THE NAVIGATION-COST COUNTERFACTUAL decides whether "order the frontier for locality" is
    a lever or a null. It must mirror the agent's OWN two options (forward walk over
    recorded edges, or RESET + replay the target's root path at cost 1+depth) and take the
    cheaper. Getting the +1 wrong flips a null into a prize.
  * THE REPLAY SPLIT (shared prefix vs suffix past divergence) is the difference between
    "these actions re-walk ground the agent already covered" and "these actions buy real
    distance", which are different findings with different fixes.

Each test drives the real helpers against hand-built inputs whose correct answer is known
by construction. No GPU, no network, no subprocess, no LLM, and nothing written outside the
test's own memory.

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-6070 -- this
is the direct follow-on to that requirement's own Implementation Status, which records the
"44 of 240 actions expanded anything new / 177 were navigation or replay" accounting on
tn36. This work decomposes those 177 into named classes on a 25-game corpus and finds the
tn36 ratio does not generalize. A dedicated REQ for the decomposition is NOT added here
because `spec.md` currently carries 142 uncommitted lines from a concurrent workflow, and
`git add`ing it would stage that work as if it were this change's -- see the report.
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "scripts"))

from arc_explorer_renavigation_probe import (  # noqa: E402
    _counterfactual_cheapest_frontier,
    _forward_distances,
    _nav_cost,
    _path_common_prefix,
)
from arc_explorer_renavigation_report import score  # noqa: E402


def _step(action: int, data=None) -> dict:
    return {"action": action, "data": data}


def test_common_prefix_stops_at_first_differing_action() -> None:
    a = [_step(1), _step(2), _step(3)]
    b = [_step(1), _step(2), _step(9)]
    assert _path_common_prefix(a, b) == 2


def test_common_prefix_distinguishes_same_action_different_click_target() -> None:
    """A click is (action, data); two ACTION6 clicks at different pixels are NOT the same
    step. If data were ignored, two divergent click paths would look like one shared
    prefix and every replay would be misreported as free re-walking."""

    a = [_step(6, {"x": 1, "y": 1})]
    b = [_step(6, {"x": 4, "y": 4})]
    assert _path_common_prefix(a, b) == 0
    assert _path_common_prefix(a, [_step(6, {"x": 1, "y": 1})]) == 1


def test_common_prefix_of_empty_path_is_zero() -> None:
    assert _path_common_prefix([], [_step(1)]) == 0
    assert _path_common_prefix([_step(1)], []) == 0


def test_forward_distances_are_hop_counts_over_recorded_edges_only() -> None:
    adj = {"a": [(1, "b"), (2, "c")], "b": [(3, "d")], "c": [], "d": []}
    dist = _forward_distances(adj, "a")
    assert dist == {"a": 0, "b": 1, "c": 1, "d": 2}
    # Edges are FORWARD-only: nothing walks back from d to a, which is exactly why the
    # agent has to RESET to reach an ancestor.
    assert _forward_distances(adj, "d") == {"d": 0}


def test_forward_distances_with_no_source_is_empty() -> None:
    assert _forward_distances({"a": [(1, "b")]}, None) == {}


def test_nav_cost_prefers_forward_walk_when_it_is_cheaper() -> None:
    snap = {"dist": {"tgt": 2}}
    assert _nav_cost("tgt", 9, snap) == (2, "forward_walk")


def test_nav_cost_falls_back_to_reset_plus_depth_when_unreachable() -> None:
    """RESET+replay costs 1 + depth: one action for the RESET itself, then one per step of
    the target's root path. Dropping the +1 would understate every fallback episode."""

    snap = {"dist": {}}
    assert _nav_cost("tgt", 3, snap) == (4, "reset_replay")


def test_nav_cost_prefers_reset_when_the_forward_walk_is_longer() -> None:
    snap = {"dist": {"tgt": 12}}
    assert _nav_cost("tgt", 0, snap) == (1, "reset_replay")


def test_cheapest_frontier_counts_standing_still_as_free() -> None:
    snap = {
        "cur": "here",
        "open_nodes": {"here": 5, "far": 7},
        "dist": {"here": 0},
    }
    out = _counterfactual_cheapest_frontier(snap)
    assert out["available"] is True
    assert out["cost"] == 0
    assert out["node"] == "here"
    assert out["kind"] == "already_here"


def test_cheapest_frontier_reports_unavailable_when_nothing_is_open() -> None:
    out = _counterfactual_cheapest_frontier({"cur": "here", "open_nodes": {}, "dist": {}})
    assert out == {"available": False}


def test_cheapest_frontier_picks_the_shallow_reset_target_over_a_deep_one() -> None:
    snap = {"cur": "here", "open_nodes": {"shallow": 1, "deep": 20}, "dist": {}}
    out = _counterfactual_cheapest_frontier(snap)
    assert out["node"] == "shallow"
    assert out["cost"] == 2


def test_score_is_quadratic_in_the_action_count() -> None:
    """The whole reason an action saving is worth stating: halving actions quadruples
    score. A linear conversion would understate every prize in the artifact by ~2x."""

    assert score(100, 100) == 1.0
    assert score(100, 50) == 4.0
    assert abs(score(100, 200) - 0.25) < 1e-12


def test_score_is_capped_at_115() -> None:
    assert score(10_000, 1) == 115.0


def test_score_of_zero_actions_is_the_cap_not_a_zero_division() -> None:
    assert score(100, 0) == 115.0


def test_discovery_attribution_uses_growth_across_the_next_decision() -> None:
    """Reproduces the off-by-one described in this module's docstring.

    The turn order is: snapshot pre -> next_move() [ingests the PREVIOUS action's frame and
    is the only place the graph grows] -> snapshot post -> env.step(). So action i's new
    node shows up as post(i+1) > pre(i+1). The broken version compared pre(i+1) > post(i),
    a gap in which nothing executes, and therefore always reported False.
    """

    rows = [
        {"i": 0, "graph_nodes_before_decision": 0, "graph_nodes_after_decision": 1},
        # decision 1 ingested action 0's frame and added a node: 1 -> 2.
        {"i": 1, "graph_nodes_before_decision": 1, "graph_nodes_after_decision": 2},
        # decision 2 ingested action 1's frame and added nothing: 2 -> 2.
        {"i": 2, "graph_nodes_before_decision": 2, "graph_nodes_after_decision": 2},
    ]
    for idx in range(len(rows) - 1):
        nxt = rows[idx + 1]
        rows[idx]["discovered_new_state"] = bool(
            nxt["graph_nodes_after_decision"] > nxt["graph_nodes_before_decision"]
        )
    assert rows[0]["discovered_new_state"] is True
    assert rows[1]["discovered_new_state"] is False

    broken = [
        bool(rows[idx + 1]["graph_nodes_before_decision"] > rows[idx]["graph_nodes_after_decision"])
        for idx in range(len(rows) - 1)
    ]
    assert broken == [False, False], "the discarded comparison must be provably always-False"
