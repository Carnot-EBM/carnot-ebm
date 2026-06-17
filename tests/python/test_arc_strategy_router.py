"""Unit tests for the STRATEGY-class router (python/carnot/agentic/arc_strategy_router.py).

The strategy router is the layer ABOVE the goal-distance heuristic router: it routes a detected
mechanic CLASS to a solving strategy, and crucially SHORT-CIRCUITS the goal-distance heuristic for
classes where it is a category error (program-editor games). These tests pin: detection precedence
(injected verdict > registry > default), the program-editor route + its short-circuit, the
graph-explore default, and honest handling of recognised-but-unwired classes.

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (ARC-AGI-3 frame-only mechanic-class strategy router).
"""

from carnot.agentic import arc_strategy_router as SR


_FAKE_REG = {
    "games": [
        {"game": "tn36", "mechanic_class": "program_editor"},
        {"game": "lp85"},  # no mechanic_class -> default graph_explore
    ]
}


def test_detect_precedence_injected_then_registry_then_default():
    # injected frame-only verdict wins over everything (the live unseen-game path)
    assert (
        SR.detect_mechanic("anything", mechanic="program_editor", reg=_FAKE_REG) == "program_editor"
    )
    # known game -> structured registry class
    assert SR.detect_mechanic("tn36", reg=_FAKE_REG) == "program_editor"
    # game with no recorded class -> default graph_explore
    assert SR.detect_mechanic("lp85", reg=_FAKE_REG) == "graph_explore"
    # unknown game entirely -> default
    assert SR.detect_mechanic("zz99", reg=_FAKE_REG) == "graph_explore"
    # an injected class outside the taxonomy falls back to default (route only to what we recognise)
    assert SR.detect_mechanic("tn36", mechanic="martian_mechanic", reg=_FAKE_REG) == "graph_explore"


def test_program_editor_routes_to_frame_only_model_and_skips_heuristic():
    r = SR.route_strategy("program_editor")
    assert r["name"] == "program_editor" and r["wired"] is True
    # the goal-distance heuristic is a category error here -> must be short-circuited
    assert r["uses_goal_distance_heuristic"] is False
    assert "frame_only_winner_search" in r["solver"]


def test_graph_explore_is_default_and_uses_heuristic():
    r = SR.route_strategy("graph_explore")
    assert r["name"] == "graph_explore" and r["uses_goal_distance_heuristic"] is True


def test_recognised_but_unwired_class_is_flagged_not_pretended():
    r = SR.route_strategy("timed_trap_aware")
    assert r["name"] == "timed_trap_aware" and r["wired"] is False
    assert "not yet wired" in r["reason"].lower()


def test_route_for_game_combines_detect_and_route():
    out = SR.route_for_game("tn36", reg=_FAKE_REG)
    assert out["game"] == "tn36" and out["name"] == "program_editor"
    assert out["routed_mechanic"] == "program_editor"
