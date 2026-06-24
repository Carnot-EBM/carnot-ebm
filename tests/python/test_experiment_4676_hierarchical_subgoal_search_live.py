"""Tests for Exp 4676 hierarchical subgoal search.

Spec refs: REQ-ARC-WMTE-4676,
SCENARIO-ARC-WMTE-4676-DIAGNOSTIC,
SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN,
SCENARIO-ARC-WMTE-4676-ABLATIONS.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def test_req_arc_wmte_4676_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4676: OpenSpec anchors the 4676 artifact and fields."""

    from carnot import experiment_4676_hierarchical_subgoal_search_live as exp4676

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4676" in spec
    assert "SCENARIO-ARC-WMTE-4676-DIAGNOSTIC" in spec
    assert "SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN" in spec
    assert exp4676.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4676.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4676_hierarchical_plan_chains_reachable_legs() -> None:
    """SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN: each leg is planned and checked."""

    from carnot.agentic.arc_llm_reinduction import (
        SubgoalCandidate,
        plan_hierarchical_subgoals,
    )

    def engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[0, 0] += int(action)
        return out

    def planner(model: Any, goal: Any, start: np.ndarray) -> list[dict[str, Any]] | None:
        grid = np.asarray(start)
        path: list[dict[str, Any]] = []
        for _ in range(4):
            if bool(goal(grid)):
                return path
            path.append({"action": 1, "data": None})
            grid = model(grid, 1, None)
        return path if bool(goal(grid)) else None

    subgoals = [
        SubgoalCandidate(
            name="too_late",
            predicate=lambda grid: bool(np.asarray(grid)[0, 0] >= 3),
            source="failed_tree",
            score=0.1,
        ),
        SubgoalCandidate(
            name="bridge",
            predicate=lambda grid: bool(np.asarray(grid)[0, 0] >= 2),
            source="a1_goal_induction",
            score=0.9,
        ),
    ]

    result = plan_hierarchical_subgoals(
        engine=engine,
        final_goal=lambda grid: bool(np.asarray(grid)[0, 0] >= 4),
        start_grid=np.array([[0]], dtype=np.int16),
        subgoals=subgoals,
        plan_in_model=planner,
        value_head=lambda grid: float(np.asarray(grid)[0, 0]),
        max_subgoals=1,
    )

    assert result.planned is True
    assert [row["name"] for row in result.subgoal_decomposition] == ["bridge", "final_goal"]
    assert [row["reachable"] for row in result.per_subgoal_reachable] == [True, True]
    assert len(result.plan) == 4


def test_scenario_arc_wmte_4676_mines_subgoals_from_failed_tree_and_exemplar() -> None:
    """SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN: proposer has non-terminal candidates."""

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import propose_hierarchical_subgoals

    transitions = [
        Transition(
            grid=np.array([[0, 0], [0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 0], [0, 0]], dtype=np.int16),
            level_before=0,
            level_after=0,
        ),
        Transition(
            grid=np.array([[1, 0], [0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 1], [0, 0]], dtype=np.int16),
            level_before=0,
            level_after=0,
        ),
    ]
    exemplar = np.array([[1, 1], [1, 0]], dtype=np.int16)

    subgoals = propose_hierarchical_subgoals(
        game="fixture",
        transitions=transitions,
        proposer=SimpleNamespace(),
        previous_level_complete_grid=exemplar,
        max_subgoals=4,
    )

    assert any(row.source == "failed_search_tree" for row in subgoals)
    assert any(row.source == "previous_level_complete_exemplar" for row in subgoals)
    assert all(row.name != "final_goal" for row in subgoals)
    assert any(row.predicate(np.array([[1, 1], [0, 0]], dtype=np.int16)) for row in subgoals)


def test_scenario_arc_wmte_4676_artifact_requires_ablation_separation() -> None:
    """SCENARIO-ARC-WMTE-4676-ABLATIONS: success needs both controls below subgoal."""

    from carnot import experiment_4676_hierarchical_subgoal_search_live as exp4676

    artifact = exp4676.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        wall_diagnosis="l2_deepening",
        generic_first_win_by_config={
            "value_routed_budget_200": {
                "first_win_rate": 1.0,
                "multi_level_rate": 0.0,
            }
        },
        target_game="lp85",
        subgoal_result={
            "generic_agent_reached_level": 2,
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "subgoal_decomposition": [{"name": "bridge"}],
            "per_subgoal_reachable": [{"name": "bridge", "reachable": True}],
        },
        no_subgoal_result={"reached_level": 1},
        random_subgoal_result={"reached_level": 1},
        duration_s=60.0,
    )

    assert exp4676.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "success: hierarchical_subgoal_generic_agent_new_level_lp85_L2"
    assert artifact["chosen_submitted_config"]["hierarchical_subgoal_search_enabled"] is True

    collapsed = exp4676.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        wall_diagnosis="l2_deepening",
        generic_first_win_by_config={},
        target_game="lp85",
        subgoal_result={
            "generic_agent_reached_level": 2,
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "subgoal_decomposition": [{"name": "bridge"}],
            "per_subgoal_reachable": [{"name": "bridge", "reachable": True}],
        },
        no_subgoal_result={"reached_level": 2},
        random_subgoal_result={"reached_level": 1},
        duration_s=60.0,
    )

    assert collapsed["honest_verdict"].startswith("complete:")
    assert collapsed["chosen_submitted_config"] == "unchanged"
    assert collapsed["null_methodology_note"]


def test_scenario_arc_wmte_4676_e3_routes_subgoal_search_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN: live E3 passes the option through."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    captured: dict[str, Any] = {}

    def fake_reinduction(**kwargs: Any) -> LlmReinductionResult:
        captured.update(kwargs)
        return LlmReinductionResult(
            planned=True,
            plan=[{"action": 1, "data": None}],
            model_specs="Qwen3.5-9B-MTP",
            goal_predicate_satisfiable=True,
            subgoal_decomposition=[{"name": "bridge", "reachable": True}],
            per_subgoal_reachable=[{"name": "bridge", "reachable": True}],
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", fake_reinduction)

    policy = agent.E3AgentPolicy(
        "lp85",
        proposer=SimpleNamespace(model_specs="Qwen"),
        target_levels=2,
        subgoal_search=True,
        subgoal_budget=3,
    )
    policy._pending_induction_reason = "level_up_reinduction"
    policy._start_level = 0
    policy._current_goal_level = 2
    policy._previous_level_complete_grid = np.array([[8]], dtype=np.int16)
    policy.root_grid = np.array([[0]], dtype=np.int16)
    policy.transitions = [
        SimpleNamespace(
            grid=np.array([[0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1]], dtype=np.int16),
            level_before=1,
            level_after=1,
        )
    ]

    policy._induce_and_plan()

    assert captured["enable_subgoal_search"] is True
    assert captured["subgoal_budget"] == 3
    assert captured["value_head"] is policy.value_head
    assert policy.plan == [{"action": 1, "data": None}]
    assert policy.induction_attempts[-1]["subgoal_search_used"] is True
