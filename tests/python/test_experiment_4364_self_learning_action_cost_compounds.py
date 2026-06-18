"""Tests for Exp 4364 deployed ARC action-cost compounding curve.

Spec refs: REQ-LEARN-4364, SCENARIO-LEARN-4364,
SCENARIO-LEARN-4364-BLOCKED.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_4364_self_learning_action_cost_compounds as exp
from carnot.agentic import arc_solver_kit as kit


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


def _held_out_row(*, learned_actions: int = 16, learned_reproduced: bool = True) -> dict[str, Any]:
    return {
        "held_out_level_id": "lp85:L3",
        "baseline_actions": 25,
        "learned_actions": learned_actions,
        "baseline_reproduced": True,
        "learned_reproduced": learned_reproduced,
        "headroom_exists": True,
        "required_train_level_ids": ["lp85:L1", "lp85:L2"],
    }


def test_req_learn_4364_spec_declares_deployment_and_curve_contract() -> None:
    """REQ-LEARN-4364: OpenSpec declares deployment and curve artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-LEARN-4364",
        "SCENARIO-LEARN-4364",
        "SCENARIO-LEARN-4364-BLOCKED",
        "experiment_4364_self_learning_action_cost_compounds.json",
        "action_efficiency_compounds",
        "compounding_curve",
        "deployed_into_solver_kit",
        "llm_heuristic_arm",
        "blocked_insufficient_solve_traces",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4364_offline_solver_defaults_to_standing_astar_cost() -> None:
    """REQ-LEARN-4364-2: default ARC planner uses g+h, with explicit baseline kept."""

    default_astar = kit.OfflineSolver(
        "branch",
        _branch_actions,
        _branch_apply,
        _branch_state_key,
        verifier=_misleading_value,
    )
    legacy_baseline = kit.OfflineSolver(
        "branch",
        _branch_actions,
        _branch_apply,
        _branch_state_key,
        verifier=_misleading_value,
        path_cost_weight=kit.ARC_BASELINE_PATH_COST_WEIGHT,
    )

    default_path, _ = default_astar.solve_level(_BranchEnv(), 0, [], depth_cap=5)
    baseline_path, _ = legacy_baseline.solve_level(_BranchEnv(), 0, [], depth_cap=5)

    assert default_astar.path_cost_weight == kit.ARC_STANDING_PATH_COST_WEIGHT
    assert legacy_baseline.path_cost_weight == kit.ARC_BASELINE_PATH_COST_WEIGHT
    assert default_path == ["short1", "short2"]
    assert baseline_path == ["long1", "long2", "long3", "long4"]


def test_req_learn_4364_compounding_curve_requires_reproduced_decrease() -> None:
    """REQ-LEARN-4364-3/4: compounding requires a reproduced held-out action drop."""

    raw_levels = [
        "cd82:L1",
        "cn04:L1",
        "lp85:L1",
        "lp85:L2",
        "lp85:L3",
        "lp85:L4",
        "cn04:L1",
        "ls20:L1",
        "m0r0:L1",
        "r11l:L1",
        "sk48:L1",
    ]
    levels = exp.order_corpus_for_per_game_curve(
        raw_levels,
        held_out_level_ids=["lp85:L3"],
        required_train_level_ids=["lp85:L1", "lp85:L2"],
    )
    curve = exp.build_compounding_curve([_held_out_row()], levels, prefix_sizes=[4, 8])
    summary = exp.summarize_compounding_curve(
        curve,
        deployed_into_solver_kit=True,
        positive_control_passed=True,
        reproduction_gated=True,
    )

    assert levels == ["cd82:L1", "cn04:L1", "ls20:L1", "m0r0:L1", "r11l:L1", "sk48:L1", "lp85:L1", "lp85:L2"]
    assert curve == [
        {"corpus_size_k": 4, "held_out_actions_to_solve": 25},
        {"corpus_size_k": 8, "held_out_actions_to_solve": 16},
    ]
    assert summary["action_efficiency_compounds"] is True

    plateau = exp.summarize_compounding_curve(
        [{"corpus_size_k": 4, "held_out_actions_to_solve": 16}],
        deployed_into_solver_kit=True,
        positive_control_passed=True,
        reproduction_gated=True,
    )
    assert plateau["action_efficiency_compounds"] is False

    ungated = exp.summarize_compounding_curve(
        curve,
        deployed_into_solver_kit=True,
        positive_control_passed=True,
        reproduction_gated=False,
    )
    assert ungated["action_efficiency_compounds"] is False


def test_scenario_learn_4364_blocked_artifact_is_terminal_and_bare() -> None:
    """SCENARIO-LEARN-4364-BLOCKED: insufficient traces fail closed."""

    artifact = exp.build_blocked_artifact(
        usable_levels=["lp85:L1"],
        missing_sources=["results/missing.json"],
        preconditions_checked={"minimum_reproduced_levels": 8},
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_solve_traces"
    assert artifact["action_efficiency_compounds"] is False
    assert artifact["compounding_curve"] == []
    assert artifact["deployed_into_solver_kit"] is False
    assert artifact["positive_control_passed"] is False
    assert artifact["reproduction_gated"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["preconditions_checked"]["minimum_reproduced_levels"] == 8
    assert exp.artifact_schema_errors(artifact) == []


def test_req_learn_4364_complete_artifact_schema_and_gap_logging(tmp_path: Path) -> None:
    """REQ-LEARN-4364: complete artifacts preserve bare gates and residual gaps."""

    curve = [
        {"corpus_size_k": 4, "held_out_actions_to_solve": 25},
        {"corpus_size_k": 8, "held_out_actions_to_solve": 16},
    ]
    artifact = exp.build_complete_artifact(
        held_out_rows=[_held_out_row()],
        compounding_curve=curve,
        split_spec={"held_out_level_ids": ["lp85:L3"]},
        model_specs={"heuristic": {"n_samples": 13}},
        preconditions_checked={"usable_reproduced_level_count": 8},
        deployment_check={"default_path_cost_weight": 1.0, "baseline_path_cost_weight": 0.0},
        duration_s=0.5,
        adversarial_verify={"status": "clean", "returncode": 0, "flagged_count": 0},
    )

    assert artifact["honest_verdict"] == "success: action_efficiency_compounds_25_to_16"
    assert artifact["action_efficiency_compounds"] is True
    assert artifact["deployed_into_solver_kit"] is True
    assert artifact["positive_control_passed"] is True
    assert artifact["reproduction_gated"] is True
    assert artifact["llm_heuristic_arm"] == {
        "ran": False,
        "beats_linear": False,
        "static_analysis_clean": True,
    }
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert exp.artifact_schema_errors(artifact) == []

    gap_artifact = exp.build_complete_artifact(
        held_out_rows=[_held_out_row(learned_actions=25)],
        compounding_curve=[
            {"corpus_size_k": 4, "held_out_actions_to_solve": 25},
            {"corpus_size_k": 8, "held_out_actions_to_solve": 25},
        ],
        split_spec={"held_out_level_ids": ["lp85:L3"]},
        model_specs={},
        preconditions_checked={},
        deployment_check={"default_path_cost_weight": 1.0, "baseline_path_cost_weight": 0.0},
        duration_s=0.1,
    )
    assert gap_artifact["action_efficiency_compounds"] is False
    assert gap_artifact["missing_verifier_gaps"]

    exp.ensure_gap_logged(tmp_path, gap_artifact)
    gap_text = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp.GAP_ID in gap_text
    assert "lp85:L3" in gap_text
    exp.ensure_gap_logged(tmp_path, gap_artifact)
    assert (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8") == gap_text


def test_req_learn_4364_schema_rejects_non_bare_gate_fields() -> None:
    """REQ-LEARN-4364: schema rejects wrapped or non-bare gate fields."""

    artifact = exp.build_blocked_artifact(
        usable_levels=[],
        missing_sources=[],
        preconditions_checked={},
        duration_s=0.0,
    )
    bad = dict(artifact)
    bad["action_efficiency_compounds"] = 1
    bad["compounding_curve"] = {"corpus_size_k": 4}
    bad["deployed_into_solver_kit"] = "false"
    bad["positive_control_passed"] = "false"
    bad["reproduction_gated"] = None
    bad["llm_heuristic_arm"] = {"ran": "false"}
    bad["verifier_is_oracle"] = True
    bad["random_seed"] = "4364"

    errors = exp.artifact_schema_errors(bad)

    for field in (
        "action_efficiency_compounds",
        "compounding_curve",
        "deployed_into_solver_kit",
        "positive_control_passed",
        "reproduction_gated",
        "llm_heuristic_arm",
        "verifier_is_oracle",
        "random_seed",
    ):
        assert any(field in error for error in errors)

    invalid_success = dict(artifact)
    invalid_success.update(
        {
            "honest_verdict": "not_terminal",
            "action_efficiency_compounds": True,
            "compounding_curve": [{"corpus_size_k": 4, "held_out_actions_to_solve": 4}],
            "deployed_into_solver_kit": False,
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
    assert "action_efficiency_compounds requires deployed_into_solver_kit=true" in invalid_errors
    assert "action_efficiency_compounds requires positive_control_passed=true" in invalid_errors
    assert "action_efficiency_compounds requires reproduction_gated=true" in invalid_errors
    assert "action_efficiency_compounds requires a decreasing compounding_curve" in invalid_errors


def test_req_learn_4364_defensive_schema_and_prefix_branches(tmp_path: Path) -> None:
    """REQ-LEARN-4364: defensive branches stay deterministic and schema-gated."""

    assert exp.corpus_prefix_sizes(0) == []
    assert exp.corpus_prefix_sizes(5) == [4, 5]
    assert exp.standing_solver_deployment_check()["default_is_additive_astar"] is True

    row_without_explicit_required = dict(_held_out_row())
    row_without_explicit_required.pop("required_train_level_ids")
    curve = exp.build_compounding_curve(
        [row_without_explicit_required],
        ["lp85:L1", "lp85:L2"],
        prefix_sizes=[1, 1, 2],
    )
    assert curve == [
        {"corpus_size_k": 1, "held_out_actions_to_solve": 25},
        {"corpus_size_k": 2, "held_out_actions_to_solve": 16},
    ]
    no_required_curve = exp.build_compounding_curve(
        [{"held_out_level_id": "other:L1", "baseline_actions": 7, "learned_actions": 3}],
        ["other:L0"],
        prefix_sizes=[1],
    )
    assert no_required_curve == [{"corpus_size_k": 1, "held_out_actions_to_solve": 7}]

    no_positive = exp.build_complete_artifact(
        held_out_rows=[{**_held_out_row(), "headroom_exists": False}],
        compounding_curve=[{"corpus_size_k": 4, "held_out_actions_to_solve": 25}],
        split_spec={},
        model_specs={},
        preconditions_checked={},
        deployment_check=exp.standing_solver_deployment_check(),
        duration_s=0.0,
    )
    assert no_positive["honest_verdict"] == "complete: action_efficiency_no_compounding_positive_control_failed"
    exp.ensure_gap_logged(tmp_path, no_positive)

    empty_rows = exp.build_complete_artifact(
        held_out_rows=[],
        compounding_curve=[],
        split_spec={},
        model_specs={},
        preconditions_checked={},
        deployment_check=exp.standing_solver_deployment_check(),
        duration_s=0.0,
    )
    assert empty_rows["reproduction_gated"] is False

    assert exp.build_complete_artifact(
        held_out_rows=[_held_out_row(learned_reproduced=False)],
        compounding_curve=curve,
        split_spec={},
        model_specs={},
        preconditions_checked={},
        deployment_check=exp.standing_solver_deployment_check(),
        duration_s=0.0,
    )["reproduction_gated"] is False

    assert exp.build_complete_artifact(
        held_out_rows=[{**_held_out_row(), "positive_control_reproduced": False}],
        compounding_curve=curve,
        split_spec={},
        model_specs={},
        preconditions_checked={},
        deployment_check=exp.standing_solver_deployment_check(),
        duration_s=0.0,
    )["reproduction_gated"] is False

    missing_errors = exp.artifact_schema_errors({})
    assert any(error.startswith("missing required field") for error in missing_errors)
    assert "honest_verdict must be a string" in missing_errors

    artifact = exp.build_blocked_artifact(
        usable_levels=[],
        missing_sources=[],
        preconditions_checked={},
        duration_s=0.0,
    )
    invalid_curve_errors = []
    for bad_curve in ([None], [{"held_out_actions_to_solve": 1}], [{"corpus_size_k": 1}]):
        bad = dict(artifact)
        bad["compounding_curve"] = bad_curve
        invalid_curve_errors.extend(exp.artifact_schema_errors(bad))
    assert invalid_curve_errors.count("compounding_curve must be a list of bare int points") == 3

    bad_llm = dict(artifact)
    bad_llm["llm_heuristic_arm"] = []
    assert "llm_heuristic_arm must contain bare bool fields" in exp.artifact_schema_errors(bad_llm)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = None
    assert "reproducibility_checksum must be a string" in exp.artifact_schema_errors(bad_checksum)

    bad_principle = dict(artifact)
    bad_principle["field_principles"] = {**exp.FIELD_PRINCIPLES, "honest_verdict": "wrong"}
    assert "field_principles mismatch for honest_verdict" in exp.artifact_schema_errors(bad_principle)
