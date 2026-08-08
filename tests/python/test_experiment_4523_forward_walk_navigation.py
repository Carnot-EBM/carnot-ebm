"""Tests for Exp 4523 forward-walk navigation and frontier batching.

Spec refs: REQ-ARC-FCP-4523, SCENARIO-ARC-FCP-4523.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4523_forward_walk_navigation as exp4523
from carnot.agentic.arc_competition_agent import StepwiseExplorer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _baseline() -> dict[str, object]:
    return {
        "policy": "e3",
        "games": list(exp4523.GATE_GAMES),
        "action_metric": dict(exp4523.CANONICAL_ACTION_METRIC),
        "solved_count": 4,
        "solved_games": list(exp4523.CORE_GAMES),
        "actions_by_game": {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731},
        "median_actions_on_solved": 7760.0,
    }


def _measurement(
    *,
    median: float,
    reset_steps: int,
    solved_games: list[str] | None = None,
) -> dict[str, object]:
    solved = solved_games or list(exp4523.CORE_GAMES)
    return {
        "policy": "e3",
        "games": list(exp4523.GATE_GAMES),
        "action_metric": dict(exp4523.CANONICAL_ACTION_METRIC),
        "solved_count": len(solved),
        "solved_games": sorted(solved),
        "actions_by_game": {game: int(median) for game in solved},
        "median_actions_on_solved": float(median),
        "median_actions_on_core": float(median),
        "navigation_diagnostics": {
            "navigation_attempts": 4,
            "exact_shortest_path_hits": 1,
            "partial_forward_walk_hits": 1,
            "forward_walk_hits": 2,
            "reset_replay_fallbacks": 2,
            "forward_edges_recorded": 8,
            "forward_navigation_steps": 5,
            "reset_replay_steps": int(reset_steps),
            "forward_walk_hit_rate": 0.5,
        },
    }


def test_req_arc_fcp_4523_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-4523: OpenSpec anchors the sweep artifact and field principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4523" in spec
    assert "SCENARIO-ARC-FCP-4523" in spec
    assert exp4523.RESULT_RELATIVE_PATH in spec
    assert "k=1" in spec
    assert "navigation-cost frontier tie-break" in spec
    for field, principle in exp4523.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4523_frontier_tiebreak_keeps_depth_primary() -> None:
    """SCENARIO-ARC-FCP-4523: nav cost only breaks equal-depth frontier ties."""

    exp = StepwiseExplorer(online_discriminative=False, navigation_cost_tiebreak=True)
    exp.root = "R"
    exp.cur = "C"
    exp.start_level = 0
    exp.best_level = 0
    exp.graph = {
        "C": {"path": [{"action": 9, "data": None}], "untested": [], "value": 0.0},
        "shallow": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 8, "data": None}],
            "value": 0.0,
        },
        "replay_first": {
            "path": [{"action": 2, "data": None}, {"action": 3, "data": None}],
            "untested": [{"action": 4, "data": None}],
            "value": 0.0,
        },
        "forward_second": {
            "path": [{"action": 5, "data": None}, {"action": 6, "data": None}],
            "untested": [{"action": 7, "data": None}],
            "value": 0.0,
        },
    }
    exp.adj = {"C": [({"action": 10, "data": None}, "forward_second")]}

    assert exp._frontier() == "shallow"

    exp.graph["shallow"]["untested"] = []

    assert exp._frontier() == "forward_second"


def test_scenario_arc_fcp_4523_frontier_batch_control_and_k3_queue() -> None:
    """SCENARIO-ARC-FCP-4523: k=1 is control; k=3 queues more work after navigation."""

    def explorer(k: int) -> StepwiseExplorer:
        exp = StepwiseExplorer(online_discriminative=False, frontier_batch_size=k)
        exp.root = "R"
        exp.cur = "C"
        exp.start_level = 0
        exp.best_level = 0
        exp.graph = {
            "C": {"path": [{"action": 9, "data": None}], "untested": [], "value": 0.0},
            "T": {
                "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
                "untested": [
                    {"action": 10, "data": None},
                    {"action": 11, "data": None},
                    {"action": 12, "data": None},
                    {"action": 13, "data": None},
                ],
                "value": 0.0,
            },
        }
        return exp

    control = explorer(1)
    assert control.next_move([], None) == ("RESET", None)
    assert [item["kind"] for item in control.pending] == [1, 2, 10]
    assert [item["action"] for item in control.graph["T"]["untested"]] == [11, 12, 13]

    batched = explorer(3)
    assert batched.next_move([], None) == ("RESET", None)
    assert [item["kind"] for item in batched.pending] == [1, 2, 10, 11, 12]
    assert [item["action"] for item in batched.graph["T"]["untested"]] == [13]


def test_req_arc_fcp_4523_artifact_selects_only_strict_improved_core_winner(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4523: strict CORE improvement wires; null keeps submitted config unchanged."""

    control = {
        "k": 1,
        "navigation_cost_tiebreak": False,
        "measurement": _measurement(median=7760.0, reset_steps=9000),
    }
    improved = {
        "k": 3,
        "navigation_cost_tiebreak": True,
        "measurement": _measurement(median=7400.0, reset_steps=1000),
    }
    artifact = exp4523.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        baseline=_baseline(),
        config_sweep=[control, improved],
        positive_control={
            "passed": True,
            "reset_replay_steps_before": 9,
            "reset_replay_steps_after": 1,
        },
        random_seed=4523,
        duration_s=0.25,
    )

    assert (
        artifact["honest_verdict"] == "success: forward_walk_median_actions_on_core_7400_below_7760"
    )
    assert artifact["median_actions_on_core_control"] == 7760.0
    assert artifact["median_actions_on_core_best"] == 7400.0
    assert artifact["core_solves_preserved"] is True
    assert artifact["chosen_submitted_config"] == {
        "frontier_batch_size": 3,
        "navigation_cost_tiebreak": True,
    }
    assert artifact["nav_diagnostics_before_after"]["before"]["reset_replay_steps"] == 9000
    assert artifact["nav_diagnostics_before_after"]["after"]["reset_replay_steps"] == 1000
    assert exp4523.artifact_schema_errors(artifact) == []

    null_artifact = exp4523.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        baseline=_baseline(),
        config_sweep=[
            control,
            {**improved, "measurement": _measurement(median=7760.0, reset_steps=800)},
        ],
        positive_control={
            "passed": True,
            "reset_replay_steps_before": 9,
            "reset_replay_steps_after": 1,
        },
        random_seed=4523,
        duration_s=0.25,
    )

    assert null_artifact["honest_verdict"] == "complete: forward_walk_no_reduction_honest_null"
    assert null_artifact["chosen_submitted_config"] == "unchanged"
    assert exp4523.artifact_schema_errors(null_artifact) == []

    out = tmp_path / exp4523.RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True)
    exp4523.write_artifact(artifact, tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_shared_forward_bfs_cache_matches_per_node_bfs_and_is_reused() -> None:
    """2026-08-08 adversarial review, "Speed" finding 1: one shared single-source BFS from
    ``self.cur`` must return exactly what the old per-node ``_exact_shortest_path`` calls
    returned, and must not be recomputed unless ``self.cur`` moves or a new edge is learned.

    Graph: CUR -2-> A -4-> C, CUR -3-> B -5-> D -6-> E. ISLAND has no incoming edge at all.
    """

    exp = StepwiseExplorer(online_discriminative=False)
    exp.cur = "CUR"
    exp._record_forward_edge("CUR", {"action": 2, "data": None}, "A")
    exp._record_forward_edge("CUR", {"action": 3, "data": None}, "B")
    exp._record_forward_edge("A", {"action": 4, "data": None}, "C")
    exp._record_forward_edge("B", {"action": 5, "data": None}, "D")
    exp._record_forward_edge("D", {"action": 6, "data": None}, "E")

    # Correctness: the new shared-cache lookup (`_exact_forward_path`) must agree, node by
    # node, with the untouched original per-call BFS (`_exact_shortest_path`) it replaces.
    for node in ["CUR", "A", "B", "C", "D", "E", "ISLAND"]:
        assert exp._exact_forward_path("CUR", node) == exp._exact_shortest_path("CUR", node)

    # Reuse: two calls with no intervening `self.cur` move or new edge share one BFS map.
    first = exp._cur_forward_paths()
    second = exp._cur_forward_paths()
    assert first is second

    # Invalidation on a newly-learned edge: the map changes and is no longer the same object.
    exp._record_forward_edge("E", {"action": 7, "data": None}, "F")
    third = exp._cur_forward_paths()
    assert third is not first
    assert third["F"] == exp._exact_shortest_path("CUR", "F")

    # Invalidation on `self.cur` moving: recomputed relative to the new source.
    exp.cur = "A"
    fourth = exp._cur_forward_paths()
    assert fourth is not third
    assert fourth["C"] == exp._exact_shortest_path("A", "C")
    assert "CUR" not in fourth  # no edge back to CUR from A


def test_partial_forward_path_reuses_one_shared_bfs_and_matches_old_per_ancestor_scan() -> None:
    """2026-08-08 adversarial review, "Speed" finding 1: ``_partial_forward_path`` must stop
    running one BFS per candidate ancestor and reuse the single shared map instead, while
    still returning exactly what the old per-ancestor-BFS loop would have returned.

    ROOT/ANC1/ANC2 are candidate ancestors of DST by REPLAY path prefix; only ANC2 is
    forward-reachable from CUR over known-working edges (a shortcut edge CUR->ANC2), so the
    old code would run 3 separate from-scratch BFS calls (one per ancestor, including the
    two unreachable ones) to find that out.
    """

    exp = StepwiseExplorer(online_discriminative=False)
    exp.cur = "CUR"
    exp.graph = {
        "ROOT": {"path": []},
        "ANC1": {"path": [{"action": 1, "data": None}]},
        "ANC2": {"path": [{"action": 1, "data": None}, {"action": 2, "data": None}]},
        "DST": {
            "path": [
                {"action": 1, "data": None},
                {"action": 2, "data": None},
                {"action": 3, "data": None},
            ]
        },
    }
    exp._record_forward_edge("CUR", {"action": 10, "data": None}, "ANC2")

    calls: list[str | None] = []
    original_bfs = exp._forward_paths_from
    exp._forward_paths_from = lambda src: calls.append(src) or original_bfs(src)

    result = exp._partial_forward_path("CUR", "DST")

    assert calls == ["CUR"], "expected exactly one shared BFS run, not one per ancestor"

    # Correctness: replay the OLD per-ancestor-BFS algorithm verbatim (calling the untouched
    # `_exact_shortest_path` once per candidate ancestor) and confirm byte-identical output.
    old_best_depth = -1
    old_best_plan = None
    for ancestor, node in exp.graph.items():
        if ancestor == "DST":
            continue
        ancestor_path = node.get("path", [])
        if not exp._path_is_prefix(ancestor_path, exp.graph["DST"]["path"]):
            continue
        forward = exp._exact_shortest_path("CUR", ancestor)
        if forward is None:
            continue
        depth = len(ancestor_path)
        if depth <= old_best_depth:
            continue
        suffix = [
            {"action": int(s["action"]), "data": s.get("data")}
            for s in exp.graph["DST"]["path"][depth:]
        ]
        old_best_depth = depth
        old_best_plan = list(forward) + suffix

    assert result == old_best_plan
    assert result == [{"action": 10, "data": None}, {"action": 3, "data": None}]
