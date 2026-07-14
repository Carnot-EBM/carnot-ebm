"""Tests for Exp 5703: mechanism-level trace of the sp80 candidate-scoring-stack
regression found by exp5701 (task 10 completion).
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5703_sp80_candidate_stack_mechanism_trace as mod


REPO = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_first_precondition_miss_reports_failing_key() -> None:
    assert mod._first_precondition_miss({"ok": False, "a": True, "b": False}) == "b"
    assert mod._first_precondition_miss({"ok": True}) is None


def test_build_artifact_blocked_when_precondition_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {"e3_policy_import": False, "ok": False},
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_play_traced must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_play_traced", _fail_if_called)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "complete: blocked_e3_policy_import"
    assert artifact["full_stack"] == {}
    assert artifact["bare_control"] == {}
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["prior_result"]["experiment_id"] == 5701


def _ok_preconds(root=mod.REPO_ROOT):
    return {"e3_policy_import": True, "ok": True}


def _fake_trace(
    *,
    levels_gained,
    router_changed=0,
    goal_bias_var=0.0,
    gcg_enabled=True,
    gcg_non_degenerate=False,
):
    return {
        "levels_gained": levels_gained,
        "total_actions": 100,
        "action_sequence": [],
        "candidate_router_present": True,
        "candidate_router_calls": 10,
        "candidate_router_changed_order_count": router_changed,
        "goal_bias_present": True,
        "goal_bias_n_scored": 50,
        "goal_bias_score_variance": goal_bias_var,
        "goal_bias_score_min": 1.0,
        "goal_bias_score_max": 1.0,
        "goal_candidate_guidance_diagnostics": {
            "enabled": gcg_enabled,
            "arms_non_degenerate": gcg_non_degenerate,
        },
    }


def test_build_artifact_all_inert_reproduces_regression(monkeypatch, tmp_path) -> None:
    """Mirrors the real sp80 finding: regression reproduces, all three learned
    mechanisms are confirmed structurally inert."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    def _fake_play(game, *, arm, budget):
        if arm == "full_stack":
            return _fake_trace(levels_gained=0, router_changed=0, goal_bias_var=0.0)
        return {
            **_fake_trace(levels_gained=1),
            "candidate_router_present": False,
            "goal_bias_present": False,
        }

    monkeypatch.setattr(mod, "_play_traced", _fake_play)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["inert_mechanisms"] == [
        "candidate_router",
        "goal_bias",
        "goal_candidate_guidance",
    ]
    assert (
        artifact["honest_verdict"]
        == "complete: regression_reproduced_but_all_three_learned_mechanisms_inert_cause_is_elsewhere_in_stack"
    )


def test_build_artifact_active_mechanism_is_implicated(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    def _fake_play(game, *, arm, budget):
        if arm == "full_stack":
            return _fake_trace(
                levels_gained=0, router_changed=5, goal_bias_var=0.3, gcg_non_degenerate=True
            )
        return {
            **_fake_trace(levels_gained=1),
            "candidate_router_present": False,
            "goal_bias_present": False,
        }

    monkeypatch.setattr(mod, "_play_traced", _fake_play)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["inert_mechanisms"] == []
    assert (
        artifact["honest_verdict"]
        == "complete: regression_reproduced_learned_mechanisms_active_and_implicated"
    )


def test_build_artifact_no_regression_this_run_is_reported_honestly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "_play_traced",
        lambda game, *, arm, budget: _fake_trace(levels_gained=1),
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "complete: sp80_regression_did_not_reproduce_this_run"


def test_req_arc_fcp_5701_repository_artifact_is_a_real_measured_trace() -> None:
    """The checked-in real trace reproduced exp5701's sp80 regression and confirmed
    all three learned candidate-scoring mechanisms were structurally inert on this
    game (goal_bias: 0 variance across every real invocation; candidate_router:
    present but never reordered; goal_candidate_guidance: self-detected its own
    degeneracy and correctly no-op'd). Adversarially clean."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["target_game"] == "sp80"
    assert result["full_stack"]["levels_gained"] < result["bare_control"]["levels_gained"]
    assert result["goal_bias_score_variance"] == 0.0
    assert result["goal_bias_n_scored"] > 0
    assert result["candidate_router_changed_order_count"] == 0
    assert set(result["inert_mechanisms"]) == {
        "candidate_router",
        "goal_bias",
        "goal_candidate_guidance",
    }
    assert result["prior_result"]["experiment_id"] == 5701
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert len(result["reproducibility_checksum"]) == 64
