"""Unit tests for the STRATEGY-class router (python/carnot/agentic/arc_strategy_router.py).

The strategy router is the layer ABOVE the goal-distance heuristic router: it routes a detected
mechanic CLASS to a solving strategy, and crucially SHORT-CIRCUITS the goal-distance heuristic for
classes where it is a category error (program-editor games). These tests pin: detection precedence
(injected verdict > registry > default), the program-editor route + its short-circuit, the
graph-explore default, and honest handling of recognised-but-unwired classes.

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (ARC-AGI-3 frame-only mechanic-class strategy router).
"""

from carnot.agentic import arc_strategy_router as router


_FAKE_REG = {
    "games": [
        {"game": "tn36", "mechanic_class": "program_editor"},
        {"game": "lp85"},  # no mechanic_class -> default graph_explore
    ]
}


def test_detect_precedence_injected_then_registry_then_default():
    # injected frame-only verdict wins over everything (the live unseen-game path)
    assert (
        router.detect_mechanic("anything", mechanic="program_editor", reg=_FAKE_REG)
        == "program_editor"
    )
    # known game -> structured registry class
    assert router.detect_mechanic("tn36", reg=_FAKE_REG) == "program_editor"
    # game with no recorded class -> default graph_explore
    assert router.detect_mechanic("lp85", reg=_FAKE_REG) == "graph_explore"
    # unknown game entirely -> default
    assert router.detect_mechanic("zz99", reg=_FAKE_REG) == "graph_explore"
    # an injected class outside the taxonomy falls back to default (route only to what we recognise)
    assert (
        router.detect_mechanic("tn36", mechanic="martian_mechanic", reg=_FAKE_REG)
        == "graph_explore"
    )


def test_program_editor_routes_to_frame_only_model_and_skips_heuristic():
    r = router.route_strategy("program_editor")
    assert r["name"] == "program_editor" and r["wired"] is True
    # the goal-distance heuristic is a category error here -> must be short-circuited
    assert r["uses_goal_distance_heuristic"] is False
    assert "frame_only_winner_search" in r["solver"]


def test_graph_explore_is_default_and_uses_heuristic():
    r = router.route_strategy("graph_explore")
    assert r["name"] == "graph_explore" and r["uses_goal_distance_heuristic"] is True


def test_checkpoint_and_timed_classes_route_to_the_maze_planner():
    # both maze classes are now WIRED to the reusable arc_maze_planner (checkpoint_multirun reproduces
    # tn36 L6; timed_trap_aware reproduces tn36 L7) and skip the goal-distance heuristic.
    for mech, fn in [
        ("checkpoint_multirun", "checkpoint_multirun_plan"),
        ("timed_trap_aware", "timed_trap_plan"),
    ]:
        r = router.route_strategy(mech)
        assert r["name"] == mech and r["wired"] is True
        assert r["uses_goal_distance_heuristic"] is False
        assert "arc_maze_planner" in r["solver"] and fn in r["solver"]


def test_route_flags_an_unwired_class_without_pretending():
    # the router must flag a recognised-but-unwired class honestly (no solver pretence). All shipped
    # classes are wired, so inject a temporary unwired class to pin the reason-formatting branch.
    fake = {
        "name": "future_class",
        "mechanic": "future_class",
        "wired": False,
        "uses_goal_distance_heuristic": False,
        "solver": "PENDING",
        "search_engine": "x",
        "needs": "a frame-only solver",
    }
    router._BY_MECHANIC["future_class"] = fake
    try:
        r = router.route_strategy("future_class")
        assert r["name"] == "future_class" and r["wired"] is False
        assert "not yet wired" in r["reason"].lower()
    finally:
        del router._BY_MECHANIC["future_class"]


def test_route_for_game_combines_detect_and_route():
    out = router.route_for_game("tn36", reg=_FAKE_REG)
    assert out["game"] == "tn36" and out["name"] == "program_editor"
    assert out["routed_mechanic"] == "program_editor"
