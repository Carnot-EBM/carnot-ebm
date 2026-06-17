"""Tests for Exp 4353 learned ARC action-cost heuristic efficiency.

Spec refs: REQ-LEARN-4353, SCENARIO-LEARN-4353,
SCENARIO-LEARN-4353-BLOCKED.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4353_learned_action_cost_heuristic_efficiency as exp
from carnot.agentic.arc_solver_kit import OfflineSolver


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


class _Frame:
    def __init__(self, state: str, level: int = 0) -> None:
        self.state = state
        self.levels_completed = level
        self.frame = np.array([[level]], dtype=np.int16)


class _BranchEnv:
    def __init__(self) -> None:
        self._game = object()
        self.state = "start"

    def reset(self) -> _Frame:
        self.state = "start"
        return _Frame(self.state)


def _branch_actions(env: _BranchEnv, _frame: _Frame, _path: tuple[str, ...]) -> list[str]:
    if env.state == "start":
        return ["long1", "short1"]
    if env.state in {"long1", "long2", "long3"}:
        return [f"long{int(env.state[-1]) + 1}"]
    if env.state == "short1":
        return ["short2"]
    return []


def _branch_apply(env: _BranchEnv, label: str, _frame: Any) -> _Frame:
    if label in {"long4", "short2"}:
        env.state = "goal"
        return _Frame(env.state, level=1)
    env.state = label
    return _Frame(env.state)


def _branch_state_key(_game: Any, frame: _Frame) -> str:
    return frame.state


def _misleading_value(_game: Any, frame: _Frame) -> float:
    return 0.0 if frame.state.startswith("long") else 0.5


def test_req_learn_4353_spec_declares_action_cost_contract() -> None:
    """REQ-LEARN-4353: OpenSpec declares the action-cost artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-LEARN-4353",
        "SCENARIO-LEARN-4353",
        "SCENARIO-LEARN-4353-BLOCKED",
        "experiment_4353_learned_action_cost_heuristic_efficiency.json",
        "blocked_insufficient_solve_traces",
        "action_efficiency_improves",
        "held_out_actions_baseline",
        "held_out_actions_learned",
        "reproduction_gated",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4353_offline_solver_can_use_path_cost_astar() -> None:
    """REQ-LEARN-4353: path-cost A* avoids a greedy value-head's longer plan."""

    greedy = OfflineSolver(
        "branch",
        _branch_actions,
        _branch_apply,
        _branch_state_key,
        verifier=_misleading_value,
        path_cost_weight=0.0,
    )
    astar = OfflineSolver(
        "branch",
        _branch_actions,
        _branch_apply,
        _branch_state_key,
        verifier=_misleading_value,
        path_cost_weight=1.0,
    )

    greedy_path, _ = greedy.solve_level(_BranchEnv(), 0, [], depth_cap=5)
    astar_path, _ = astar.solve_level(_BranchEnv(), 0, [], depth_cap=5)

    assert greedy_path == ["long1", "long2", "long3", "long4"]
    assert astar_path == ["short1", "short2"]


def test_scenario_learn_4353_blocked_artifact_is_terminal_and_bare() -> None:
    """SCENARIO-LEARN-4353-BLOCKED: insufficient traces fail closed."""

    artifact = exp.build_blocked_artifact(
        usable_levels=["lp85:L1"],
        missing_sources=["results/missing.json"],
        preconditions_checked={"minimum_reproduced_levels": 5},
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_solve_traces"
    assert artifact["action_efficiency_improves"] is False
    assert artifact["held_out_actions_baseline"] == 0
    assert artifact["held_out_actions_learned"] == 0
    assert artifact["positive_control_passed"] is False
    assert artifact["reproduction_gated"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["preconditions_checked"]["minimum_reproduced_levels"] == 5
    assert exp.artifact_schema_errors(artifact) == []


def test_req_learn_4353_summary_gate_requires_action_reduction_and_controls() -> None:
    """REQ-LEARN-4353: improvement requires lower actions plus both gates."""

    rows = [
        {
            "held_out_level_id": "lp85:L3",
            "baseline_actions": 25,
            "learned_actions": 16,
            "baseline_reproduced": True,
            "learned_reproduced": True,
            "headroom_exists": True,
        },
        {
            "held_out_level_id": "r11l:L1",
            "baseline_actions": 3,
            "learned_actions": 3,
            "baseline_reproduced": True,
            "learned_reproduced": True,
            "headroom_exists": False,
        },
    ]

    summary = exp.summarize_action_efficiency(rows)

    assert summary["held_out_actions_baseline"] == 28
    assert summary["held_out_actions_learned"] == 19
    assert summary["positive_control_passed"] is True
    assert summary["reproduction_gated"] is True
    assert summary["action_efficiency_improves"] is True

    no_gate = exp.summarize_action_efficiency([{**rows[0], "learned_reproduced": False}])
    assert no_gate["reproduction_gated"] is False
    assert no_gate["action_efficiency_improves"] is False


def test_req_learn_4353_action_cost_regressor_orders_closer_states() -> None:
    """REQ-LEARN-4353: CPU heuristic learns lower actions-to-win for closer states."""

    assert exp.ActionCostRegressor().predict([1.0, 2.0]) == 0.0
    assert exp.ActionCostRegressor().rounded_weights() == []
    with pytest.raises(ValueError, match="no rows"):
        exp.ActionCostRegressor().fit([], [])
    with pytest.raises(ValueError, match="targets"):
        exp.ActionCostRegressor().fit([[1.0]], [1.0, 2.0])

    regressor = exp.ActionCostRegressor().fit(
        [[10.0, 1.0], [5.0, 1.0], [1.0, 0.0]],
        [10.0, 5.0, 1.0],
    )
    heuristic = exp.StateActionCostHeuristic(regressor, lambda game: game["features"])

    assert regressor.predict([1.0, 0.0]) < regressor.predict([10.0, 1.0])
    assert heuristic({"features": [1.0, 0.0]}) == pytest.approx(regressor.predict([1.0, 0.0]))
    assert regressor.n_samples == 3
    assert regressor.model_summary()["target"] == "minimal env-actions-to-win"


def test_req_learn_4353_complete_artifact_schema_and_gap_logging(tmp_path: Path) -> None:
    """REQ-LEARN-4353: complete artifacts preserve bare fields and log null gaps."""

    rows = [
        {
            "held_out_level_id": "r11l:L1",
            "baseline_actions": 3,
            "learned_actions": 3,
            "baseline_reproduced": True,
            "learned_reproduced": True,
            "headroom_exists": True,
        }
    ]
    artifact = exp.build_complete_artifact(
        held_out_rows=rows,
        split_spec={"train_level_ids": ["lp85:L1"], "held_out_level_ids": ["r11l:L1"]},
        model_specs={"value_head": {"n_samples": 2}},
        preconditions_checked={"usable_reproduced_level_count": 2},
        duration_s=0.5,
        adversarial_verify={"status": "clean", "returncode": 0, "flagged_count": 0},
    )

    assert artifact["honest_verdict"] == "complete: learned_action_cost_no_reduction_positive_control_passed"
    assert artifact["action_efficiency_improves"] is False
    assert artifact["held_out_actions_baseline"] == 3
    assert artifact["held_out_actions_learned"] == 3
    assert artifact["positive_control_passed"] is True
    assert artifact["reproduction_gated"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert exp.artifact_schema_errors(artifact) == []

    exp.ensure_gap_logged(tmp_path, artifact)
    gap_text = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp.GAP_ID in gap_text
    assert "r11l:L1" in gap_text
    exp.ensure_gap_logged(tmp_path, artifact)
    assert (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8") == gap_text

    success = exp.build_complete_artifact(
        held_out_rows=[
            {
                "held_out_level_id": "lp85:L3",
                "baseline_actions": 25,
                "learned_actions": 16,
                "baseline_reproduced": True,
                "learned_reproduced": True,
                "headroom_exists": True,
            }
        ],
        split_spec={"train_level_ids": ["lp85:L1"], "held_out_level_ids": ["lp85:L3"]},
        model_specs={"value_head": {"n_samples": 2}},
        preconditions_checked={},
        duration_s=0.1,
    )
    assert success["honest_verdict"] == "success: learned_action_cost_reduces_actions_25_to_16"
    assert success["missing_verifier_gaps"] == []
    exp.ensure_gap_logged(tmp_path, success)


def test_req_learn_4353_schema_rejects_non_bare_gate_fields() -> None:
    """REQ-LEARN-4353: schema rejects wrapped or non-bare gate fields."""

    artifact = exp.build_blocked_artifact(
        usable_levels=[],
        missing_sources=[],
        preconditions_checked={},
        duration_s=0.0,
    )
    bad = dict(artifact)
    bad["action_efficiency_improves"] = 1
    bad["held_out_actions_baseline"] = "0"
    bad["held_out_actions_learned"] = {"value": 0}
    bad["positive_control_passed"] = "false"
    bad["reproduction_gated"] = None
    bad["verifier_is_oracle"] = True
    bad["random_seed"] = "4353"

    errors = exp.artifact_schema_errors(bad)

    for field in (
        "action_efficiency_improves",
        "held_out_actions_baseline",
        "held_out_actions_learned",
        "positive_control_passed",
        "reproduction_gated",
        "verifier_is_oracle",
        "random_seed",
    ):
        assert any(field in error for error in errors)

    invalid_success = dict(artifact)
    invalid_success.update(
        {
            "honest_verdict": "not_terminal",
            "action_efficiency_improves": True,
            "held_out_actions_baseline": 3,
            "held_out_actions_learned": 3,
            "positive_control_passed": False,
            "reproduction_gated": False,
            "preconditions_checked": [],
            "model_specs": [],
            "field_principles": None,
        }
    )
    invalid_errors = exp.artifact_schema_errors(invalid_success)
    assert "honest_verdict must be terminal-prefixed" in invalid_errors
    assert "preconditions_checked must be an object" in invalid_errors
    assert "model_specs must be an object" in invalid_errors
    assert "field_principles must be an object" in invalid_errors
    assert "action_efficiency_improves requires positive_control_passed=true" in invalid_errors
    assert "action_efficiency_improves requires reproduction_gated=true" in invalid_errors
    assert "action_efficiency_improves requires learned actions < baseline actions" in invalid_errors

    missing = exp.artifact_schema_errors({"field_principles": exp.FIELD_PRINCIPLES})
    assert any(error.startswith("missing required field") for error in missing)
    assert "honest_verdict must be a string" in missing
