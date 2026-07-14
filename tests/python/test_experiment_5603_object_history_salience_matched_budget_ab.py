"""Tests for Exp 5603 ObjectHistorySaliencePrior matched-budget A/B (the flip-decision
measurement task 10's own wiring follow-on named as still pending).

Spec refs: REQ-ARC-FCP-5591-3, SCENARIO-ARC-FCP-5591-3-DEFAULT-WEIGHT-NO-OP,
SCENARIO-ARC-FCP-5591-3-RESCALED-WEIGHT-STILL-NO-OP.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5603_object_history_salience_matched_budget_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5591_3_spec_declares_matched_budget_ab_contract() -> None:
    """REQ-ARC-FCP-5591-3: OpenSpec declares the matched-budget A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5591-3") :]

    for marker in (
        "REQ-ARC-FCP-5591-3",
        "SCENARIO-ARC-FCP-5591-3-DEFAULT-WEIGHT-NO-OP",
        "SCENARIO-ARC-FCP-5591-3-RESCALED-WEIGHT-STILL-NO-OP",
        "TRAJECTORY DIVERGENCE",
    ):
        assert marker in section


def test_scenario_5603_blocked_precondition_never_runs(monkeypatch) -> None:
    """A missing resource fails closed without attempting any run."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_makes_env": False,
            "e3_and_prior_import": True,
            "ok": False,
        },
    )

    def _fail_if_called(**_kwargs):
        raise AssertionError("_run_arm must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_run_arm", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["baseline"] == {}
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def _ok_preconds(root=mod.REPO_ROOT):
    return {"offline_arcade_makes_env": True, "e3_and_prior_import": True, "ok": True}


def test_scenario_5603_default_weight_divergence_detected(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5591-3-DEFAULT-WEIGHT-NO-OP's counterpart: when the default-weight
    treatment genuinely differs from baseline, the verdict reports real behavioral change."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    def _fake_run_arm(*, game, explore_budget, total_budget, object_history_salience):
        del game, explore_budget, total_budget
        if object_history_salience is False:
            return {
                "transitions_collected": 3,
                "actions": [{"action": 6, "data": {"x": 1, "y": 1}}] * 3,
            }
        if object_history_salience is True:
            return {
                "transitions_collected": 3,
                "actions": [{"action": 6, "data": {"x": 2, "y": 2}}] * 3,
            }
        return {
            "transitions_collected": 2,
            "actions": [{"action": 6, "data": {"x": 1, "y": 1}}] * 2,
        }

    monkeypatch.setattr(mod, "_run_arm", _fake_run_arm)

    artifact = mod.build_artifact()

    assert artifact["trajectories_diverge_at_default_weight"] is True
    assert artifact["honest_verdict"] == (
        "complete: object_history_salience_ab_default_weight_changes_behavior"
    )


def test_scenario_5603_default_no_op_rescaled_diverges(monkeypatch) -> None:
    """When the default weight is a no-op but the rescaled diagnostic diverges, the verdict
    flags a needs-retune finding rather than a blanket no-op."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    fixed_actions = [{"action": 6, "data": {"x": 1, "y": 1}}] * 3

    def _fake_run_arm(*, game, explore_budget, total_budget, object_history_salience):
        del game, explore_budget, total_budget
        if isinstance(object_history_salience, bool):
            return {"transitions_collected": 3, "actions": fixed_actions}
        return {
            "transitions_collected": 3,
            "actions": [{"action": 6, "data": {"x": 9, "y": 9}}] * 3,
        }

    monkeypatch.setattr(mod, "_run_arm", _fake_run_arm)

    artifact = mod.build_artifact()

    assert artifact["trajectories_diverge_at_default_weight"] is False
    assert artifact["trajectories_diverge_at_rescaled_weight"] is True
    assert artifact["honest_verdict"] == (
        "complete: object_history_salience_ab_default_weight_no_op_"
        "rescaled_weight_diverges_needs_retune"
    )


def test_scenario_5603_no_op_at_either_weight_is_honest_null(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5591-3-DEFAULT-WEIGHT-NO-OP + -RESCALED-WEIGHT-STILL-NO-OP: identical
    trajectories at both weights is reported as an honest, valid null, not a failure."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    fixed_actions = [{"action": 6, "data": {"x": 1, "y": 1}}] * 3

    def _fake_run_arm(*, game, explore_budget, total_budget, object_history_salience):
        del game, explore_budget, total_budget, object_history_salience
        return {"transitions_collected": 3, "actions": fixed_actions}

    monkeypatch.setattr(mod, "_run_arm", _fake_run_arm)

    artifact = mod.build_artifact()

    assert artifact["trajectories_diverge_at_default_weight"] is False
    assert artifact["trajectories_diverge_at_rescaled_weight"] is False
    assert (
        artifact["honest_verdict"] == "complete: object_history_salience_ab_no_op_at_either_weight"
    )


def test_req_arc_fcp_5591_3_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-FCP-5591-3: the checked-in real run measured ObjectHistorySaliencePrior
    against real E3AgentPolicy exploration on m0r0 at both the default and a rescaled
    change_bonus_weight -- an honest, real null (identical trajectories at both weights),
    correcting an initial informal hypothesis that lacked a baseline comparison."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["inference_substrate"] == (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert result["solve_provenance"] == "development_proxy"
    assert result["baseline"]["transitions_collected"] > 0
    assert result["default_weight_treatment"]["transitions_collected"] > 0
    assert result["rescaled_weight_diagnostic"]["transitions_collected"] > 0
    assert result["duration_s"] > 1.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
