"""Tests for Exp5711's live relational goal-energy route.

Spec refs: REQ-ARC-WMTE-5711,
SCENARIO-ARC-WMTE-5711-LIVE-HOOK-REACHABILITY,
SCENARIO-ARC-WMTE-5711-SAFE-FALLBACK-AND-LEAKAGE.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
PREDICATE_CODE = 'def is_goal(state):\n    return state["unsatisfied_targets"] == 0\n'
pytestmark = pytest.mark.memory_watchdog_skip


def _mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> list[list[bool]]:
    out = np.zeros(shape, dtype=bool)
    for y, x in coords:
        out[y, x] = True
    return out.tolist()


def _state(grid: np.ndarray, receipt: dict[str, Any]) -> dict[str, Any]:
    return {"frame": np.asarray(grid, dtype=int), "relational_goal_receipt": receipt}


def test_req_arc_wmte_5711_spec_declares_live_route_contract() -> None:
    """REQ-ARC-WMTE-5711: OpenSpec declares reachability, fallback, and artifact gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5711") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        "SCENARIO-ARC-WMTE-5711-LIVE-HOOK-REACHABILITY",
        "SCENARIO-ARC-WMTE-5711-SAFE-FALLBACK-AND-LEAKAGE",
        "variance_floor=1e-12",
        "results/experiment_5711_arc_relational_goal_energy_live_qualification.json",
        "live_path_reachable_score",
    ):
        assert marker in section


def test_req_arc_wmte_5711_preserves_old_visible_fraction_energy() -> None:
    """REQ-ARC-WMTE-5711: legacy count/fraction state still delegates to Exp4020 energy."""

    from carnot.agentic.arc_goal_energy_live import (
        GoalSatisfactionEnergy,
        RelationalGoalEnergy,
    )

    legacy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    energy = RelationalGoalEnergy(fallback_goal_energy=legacy)

    state = {"total_targets": 4, "satisfied_targets": 1, "unsatisfied_targets": 3}
    assert energy(state) == pytest.approx(0.75)
    assert energy.diagnostics()["last_fallback_reason"] == "legacy_goal_satisfaction"


def test_req_arc_wmte_5711_scores_all_generic_relational_classes() -> None:
    """REQ-ARC-WMTE-5711: supported generic spatial classes separate win from near-win."""

    from carnot.agentic.arc_goal_energy_live import RelationalGoalEnergy

    energy = RelationalGoalEnergy()

    region_win = np.zeros((4, 6), dtype=int)
    region_win[1:3, 0:2] = np.array([[1, 2], [3, 4]])
    region_win[1:3, 4:6] = np.array([[1, 2], [3, 4]])
    region_near = region_win.copy()
    region_near[1, 4] = 9
    region_receipt = {
        "route_class": "region_pair_equality",
        "source_mask": _mask((4, 6), [(1, 4), (1, 5), (2, 4), (2, 5)]),
        "target_mask": _mask((4, 6), [(1, 0), (1, 1), (2, 0), (2, 1)]),
    }

    translated_win = np.zeros((5, 7), dtype=int)
    translated_win[1:3, 0:2] = 5
    translated_win[1:3, 4:6] = 5
    translated_near = translated_win.copy()
    translated_near[1, 4] = 0
    translated_receipt = {
        "route_class": "translated_within_frame_target_match",
        "offset": [0, 4],
        "source_mask": _mask((5, 7), [(1, 0), (1, 1), (2, 0), (2, 1)]),
    }

    run_win = np.zeros((3, 5), dtype=int)
    run_win[1, 1:4] = [1, 2, 3]
    run_near = run_win.copy()
    run_near[1, 2:4] = [3, 2]
    run_receipt = {
        "route_class": "ordered_run_relation",
        "run_mask": _mask((3, 5), [(1, 1), (1, 2), (1, 3)]),
        "order": "ascending",
    }

    centroid_win = np.zeros((5, 8), dtype=int)
    centroid_win[1, 1] = 7
    centroid_win[1, 5] = 8
    centroid_near = centroid_win.copy()
    centroid_near[1, 1] = 0
    centroid_near[3, 1] = 7
    centroid_receipt = {
        "route_class": "centroid_alignment",
        "source_mask": _mask((5, 8), [(1, 1), (2, 1), (3, 1)]),
        "target_mask": _mask((5, 8), [(1, 5), (2, 5), (3, 5)]),
    }

    fixtures = (
        (region_win, region_near, region_receipt),
        (translated_win, translated_near, translated_receipt),
        (run_win, run_near, run_receipt),
        (centroid_win, centroid_near, centroid_receipt),
    )
    for win, near, receipt in fixtures:
        assert energy(_state(win, receipt)) == 0.0
        assert energy(_state(near, receipt)) > 0.0


def test_scenario_arc_wmte_5711_fallback_preserves_frontier_order() -> None:
    """SCENARIO-ARC-WMTE-5711-SAFE-FALLBACK-AND-LEAKAGE: missing targets are no-ops."""

    from carnot.agentic.arc_competition_agent import StepwiseExplorer
    from carnot.agentic.arc_goal_energy_live import RelationalGoalEnergy

    no_bias = StepwiseExplorer(
        goal_bias=None,
        frame_change_scorer=None,
        candidate_router=None,
    )
    routed = StepwiseExplorer(
        goal_bias=RelationalGoalEnergy(),
        goal_bias_label="relational_goal_energy",
        frame_change_scorer=None,
        candidate_router=None,
    )
    graph = {
        "a": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 2, "data": None}],
            "value": 0.0,
            "frame": SimpleNamespace(frame=np.zeros((2, 2), dtype=int)),
        },
        "b": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [{"action": 3, "data": None}],
            "value": 0.0,
            "frame": SimpleNamespace(frame=np.ones((2, 2), dtype=int)),
        },
    }
    no_bias.graph = {key: dict(value) for key, value in graph.items()}
    routed.graph = {key: dict(value) for key, value in graph.items()}
    no_bias.cur = routed.cur = "a"

    assert routed._frontier() == no_bias._frontier()
    assert routed.goal_bias.diagnostics()["last_fallback_reason"] == "missing_relational_receipt"


def test_scenario_arc_wmte_5711_e3_policy_exercises_both_goal_hooks() -> None:
    """SCENARIO-ARC-WMTE-5711-LIVE-HOOK-REACHABILITY: E3 reaches both hook surfaces."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_goal_energy_live import RelationalGoalEnergy

    energy = RelationalGoalEnergy()
    policy = E3AgentPolicy(
        "synthetic-5711",
        proposer=None,
        target_levels=1,
        value_head=None,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
        action_prior=None,
        candidate_router=None,
        goal_bias=energy,
        goal_candidate_guidance=True,
        qd_generator=None,
        controllable_novelty=False,
        object_centric_proposal=False,
        program_synthesis_filter=False,
        inert_click_pruner=False,
        object_history_salience=False,
        amortized_first_contact_prior=False,
        go_explore_archive=False,
        similarity_retrieval=False,
    )

    grid = np.zeros((3, 6), dtype=int)
    grid[1, 0:2] = [1, 2]
    grid[1, 4:6] = [1, 2]
    near = grid.copy()
    near[1, 4] = 0
    receipt = {
        "route_class": "region_pair_equality",
        "source_mask": _mask((3, 6), [(1, 4), (1, 5)]),
        "target_mask": _mask((3, 6), [(1, 0), (1, 1)]),
    }

    policy.explorer._goal_bias_score({"frame": _state(near, receipt)})
    ranked = policy.explorer.goal_candidate_guidance.rank_candidates(
        object(),
        [
            {"action": 1, "data": None, "candidate_state": _state(near, receipt)},
            {"action": 2, "data": None, "candidate_state": _state(grid, receipt)},
        ],
    )

    assert [row["action"] for row in ranked] == [2, 1]
    assert policy.explorer.goal_bias_diagnostics()["nodes_scored"] == 1
    assert policy.explorer.goal_candidate_guidance_diagnostics()["arms_non_degenerate"] is True
    assert energy.diagnostics()["routed_call_count"] >= 3
