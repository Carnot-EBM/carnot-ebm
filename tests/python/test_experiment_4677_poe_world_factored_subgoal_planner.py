"""Tests for Exp 4677 PoE-World factored executable subgoal planner.

Spec refs: REQ-ARC-WMTE-4677,
SCENARIO-ARC-WMTE-4677-TRUSTED-FACTORS,
SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING,
SCENARIO-ARC-WMTE-4677-COVERAGE-CONTROL.
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


def test_req_arc_wmte_4677_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4677: OpenSpec anchors the 4677 artifact and fields."""

    from carnot import experiment_4677_poe_world_factored_subgoal_planner as exp4677

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4677" in spec
    assert "SCENARIO-ARC-WMTE-4677-TRUSTED-FACTORS" in spec
    assert "SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING" in spec
    assert "SCENARIO-ARC-WMTE-4677-COVERAGE-CONTROL" in spec
    assert exp4677.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4677.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4677_trust_keeps_only_replay_stable_factors() -> None:
    """SCENARIO-ARC-WMTE-4677-TRUSTED-FACTORS: held-out trust filters experts."""

    from carnot.agentic.arc_executable_world_model import (
        Transition,
        induce_programmatic_object_experts,
    )

    transitions = [
        Transition(
            grid=np.array([[0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 1]], dtype=np.int16),
            level_before=0,
            level_after=0,
        ),
        Transition(
            grid=np.array([[0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 1]], dtype=np.int16),
            level_before=0,
            level_after=0,
        ),
        Transition(
            grid=np.array([[2, 2]], dtype=np.int16),
            action=2,
            data=None,
            next_grid=np.array([[2, 2]], dtype=np.int16),
            level_before=0,
            level_after=0,
        ),
    ]

    class FakeProposer:
        def induce_programmatic_experts(self, **_: Any) -> list[dict[str, Any]]:
            return [
                {
                    "name": "stable_zero_to_one",
                    "object_class": "color_0",
                    "kind": "color_rewrite",
                    "action": 1,
                    "from_color": 0,
                    "to_color": 1,
                },
                {
                    "name": "brittle_zero_to_two",
                    "object_class": "color_0",
                    "kind": "color_rewrite",
                    "action": 1,
                    "from_color": 0,
                    "to_color": 2,
                },
            ]

    result = induce_programmatic_object_experts(
        game="fixture",
        transitions=transitions,
        proposer=FakeProposer(),
        trust_threshold=0.75,
        heldout_fraction=0.34,
    )

    assert [expert.name for expert in result.experts] == ["stable_zero_to_one"]
    assert result.experts[0].trust == 1.0
    assert result.expert_trust_weights == [
        {
            "name": "stable_zero_to_one",
            "object_class": "color_0",
            "trust": 1.0,
            "heldout_correct": 1,
            "heldout_total": 1,
            "kept": True,
        },
        {
            "name": "brittle_zero_to_two",
            "object_class": "color_0",
            "trust": 0.0,
            "heldout_correct": 0,
            "heldout_total": 1,
            "kept": False,
        },
    ]


def test_scenario_arc_wmte_4677_product_planner_chains_subgoal_and_goal() -> None:
    """SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING: product model plans valid legs."""

    from carnot.agentic.arc_executable_world_model import (
        ProgrammaticExpert,
        plan_factored_subgoal_sequence,
    )

    def precondition(grid: np.ndarray, action: int, _data: Any) -> bool:
        return action == 1 and bool(np.any(np.asarray(grid) == 0))

    def effect(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[out == 0] = 1
        return out

    expert = ProgrammaticExpert(
        name="zero_to_one",
        object_class="color_0",
        precondition=precondition,
        effect=effect,
        action=1,
        trust=1.0,
        heldout_correct=2,
        heldout_total=2,
    )

    result = plan_factored_subgoal_sequence(
        start_grid=np.array([[0, 0]], dtype=np.int16),
        final_goal=lambda grid: bool(np.all(np.asarray(grid) == 1)),
        experts=[expert],
        subgoals=[
            {
                "name": "one_nonzero",
                "predicate": lambda grid: int(np.count_nonzero(np.asarray(grid))) >= 1,
                "score": 0.9,
            }
        ],
        value_head=lambda grid: float(np.count_nonzero(np.asarray(grid))),
        max_subgoals=1,
        max_depth=3,
        max_nodes=20,
    )

    assert result.planned is True
    assert result.plan == [{"action": 1, "data": None}]
    assert [row["name"] for row in result.subgoal_decomposition] == [
        "one_nonzero",
        "final_goal",
    ]
    assert [row["reachable"] for row in result.per_subgoal_reachable] == [True, True]
    assert result.expert_trust_weights[0]["trust"] == 1.0


def test_scenario_arc_wmte_4677_artifact_requires_coverage_and_live_lift() -> None:
    """SCENARIO-ARC-WMTE-4677-COVERAGE-CONTROL: success needs coverage and lift."""

    from carnot import experiment_4677_poe_world_factored_subgoal_planner as exp4677

    artifact = exp4677.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        target_games=["lp85"],
        candidate_generation_coverage_factored=1.0,
        candidate_generation_coverage_flat_baseline=0.0,
        live_first_win_rate_factored=1.0,
        live_solve_rate_factored=0.0,
        live_baseline_flat_search={"first_win_rate": 0.0, "solve_rate": 0.0},
        live_lift_ci={"metric": "first_win_rate_delta", "low": 0.2, "high": 1.0},
        expert_trust_weights=[{"name": "zero_to_one", "trust": 1.0, "kept": True}],
        bare_control_passed=True,
        offline_reproduced=True,
        duration_s=60.0,
    )

    assert exp4677.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == (
        "success: poe_world_factored_planner_coverage_up_live_firstwin_lift_lp85"
    )
    assert artifact["chosen_submitted_config"]["factored_planner_enabled"] is True

    null = exp4677.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        target_games=["lp85"],
        candidate_generation_coverage_factored=0.0,
        candidate_generation_coverage_flat_baseline=0.0,
        live_first_win_rate_factored=0.0,
        live_solve_rate_factored=0.0,
        live_baseline_flat_search={"first_win_rate": 0.0, "solve_rate": 0.0},
        live_lift_ci={"metric": "solve_rate_delta", "low": 0.0, "high": 0.0},
        expert_trust_weights=[],
        bare_control_passed=True,
        offline_reproduced=False,
        duration_s=60.0,
    )

    assert null["honest_verdict"] == (
        "complete: poe_world_factored_planner_no_coverage_gain_residual_logged"
    )
    assert null["chosen_submitted_config"] == "unchanged"
    assert null["null_methodology_note"]
    assert null["residual_bridge_gap"] in {
        "expert_factors_not_independent",
        "product_model_plans_live_invalid",
        "experts_overfit_prefix",
    }

    kept_null = exp4677.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        target_games=["lp85"],
        candidate_generation_coverage_factored=0.0,
        candidate_generation_coverage_flat_baseline=0.0,
        live_first_win_rate_factored=0.0,
        live_solve_rate_factored=0.0,
        live_baseline_flat_search={"first_win_rate": 0.0, "solve_rate": 0.0},
        live_lift_ci={"metric": "solve_rate_delta", "low": 0.0, "high": 0.0},
        expert_trust_weights=[{"name": "stable", "trust": 1.0, "kept": True}],
        bare_control_passed=True,
        offline_reproduced=False,
        duration_s=60.0,
    )
    assert kept_null["residual_bridge_gap"] == "expert_factors_not_independent"

    no_lift = exp4677.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        target_games=["lp85"],
        candidate_generation_coverage_factored=1.0,
        candidate_generation_coverage_flat_baseline=0.0,
        live_first_win_rate_factored=0.0,
        live_solve_rate_factored=0.0,
        live_baseline_flat_search={"first_win_rate": 0.0, "solve_rate": 0.0},
        live_lift_ci={"metric": "first_win_rate_delta", "low": "bad", "high": 0.0},
        expert_trust_weights=[{"name": "stable", "trust": 1.0, "kept": True}],
        bare_control_passed=True,
        offline_reproduced=True,
        duration_s=60.0,
    )
    assert no_lift["residual_bridge_gap"] == "product_model_plans_live_invalid"
    assert "did not support" in no_lift["null_methodology_note"]

    ci = exp4677._bootstrap_ci_delta([True, False], [False, False], seed=1, n_boot=10)
    assert ci["metric"] == "coverage_delta"
    assert ci["n_boot"] == 10
    assert exp4677._bootstrap_ci_delta([], [], seed=1)["n_boot"] == 0
    assert exp4677._ci_excludes_zero({"low": "bad"}) is False

    bad = dict(null)
    bad["honest_verdict"] = "oops"
    bad["verifier_is_oracle"] = True
    bad["solve_provenance"] = "development_proxy"
    bad["proposer_served_model"] = "gemma-4-12b"
    bad["null_methodology_note"] = ""
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = exp4677.artifact_schema_errors(bad)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "solve_provenance" in errors
    assert "proposer_served_model" in errors
    assert "null_methodology_note" in errors
    assert "reproducibility_checksum" in errors


def test_scenario_arc_wmte_4677_e3_routes_factored_planner_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING: live E3 passes the option through."""

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
            factored_planner_used=True,
            expert_trust_weights=[{"name": "zero_to_one", "trust": 1.0, "kept": True}],
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", fake_reinduction)

    policy = agent.E3AgentPolicy(
        "lp85",
        proposer=SimpleNamespace(model_specs="Qwen"),
        target_levels=2,
        factored_planner=True,
        factored_trust_threshold=0.75,
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

    assert captured["enable_factored_planner"] is True
    assert captured["factored_trust_threshold"] == 0.75
    assert captured["value_head"] is policy.value_head
    assert policy.plan == [{"action": 1, "data": None}]
    assert policy.induction_attempts[-1]["factored_planner_used"] is True
    assert policy.induction_attempts[-1]["expert_trust_weights"][0]["trust"] == 1.0
