"""Tests for Exp 5590 dict-candidate CNN fix matched-budget A/B.

Spec refs: REQ-ARC-FCP-5590, SCENARIO-ARC-FCP-5590-MATCHED-BUDGET-DELTA.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5590_frame_change_cnn_dict_candidate_fix_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5590_spec_declares_ab_contract() -> None:
    """REQ-ARC-FCP-5590: OpenSpec declares the matched-budget A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5590") :]

    for marker in (
        "REQ-ARC-FCP-5590",
        "SCENARIO-ARC-FCP-5590-MATCHED-BUDGET-DELTA",
        "levels_gained_headroom_present",
        "states_expanded_control_total",
        "CARNOT_ARC_DISABLE_INDUCTION",
    ):
        assert marker in section


def test_scenario_arc_fcp_5590_blocked_precondition_never_runs_arms(monkeypatch) -> None:
    """A missing resource fails closed without running any game."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "e3_policy_import": True,
            "as_action_like_import": True,
            "ok": False,
        },
    )

    def _fail_if_called(roster, *, budget):
        raise AssertionError("run_both_arms must not run when a precondition is missing")

    monkeypatch.setattr(mod, "run_both_arms", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["control_results"] == {}
    assert artifact["levels_gained_headroom_present"] is False
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_fcp_5590_synthetic_positive_delta(monkeypatch) -> None:
    """A synthetic treatment-wins result is classified as a helps verdict."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "as_action_like_import": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: {"levels": 0, "states_expanded": 10} for g in roster},
            {g: {"levels": (1 if g == roster[0] else 0), "states_expanded": 12} for g in roster},
            1.23,
        ),
    )

    artifact = mod.build_artifact(roster=("cd82", "sk48"))

    assert artifact["levels_gained_control_total"] == 0
    assert artifact["levels_gained_treatment_total"] == 1
    assert artifact["levels_gained_headroom_present"] is True
    assert artifact["honest_verdict"] == "complete: dict_candidate_fix_helps_0_to_1_levels"
    assert "combined_wall_clock_s" not in artifact


def test_scenario_arc_fcp_5590_synthetic_honest_null_with_headroom(monkeypatch) -> None:
    """A synthetic all-equal result with headroom present is an honest null, not fabricated."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "as_action_like_import": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: {"levels": (1 if g == roster[0] else 0), "states_expanded": 10} for g in roster},
            {g: {"levels": (1 if g == roster[0] else 0), "states_expanded": 10} for g in roster},
            4.56,
        ),
    )

    artifact = mod.build_artifact(roster=("cd82", "sk48"))

    assert artifact["levels_gained_headroom_present"] is True
    assert all(delta == 0 for delta in artifact["per_game_levels_delta"].values())
    assert artifact["honest_verdict"] == (
        "complete: dict_candidate_fix_honest_null_headroom_present_no_delta"
    )


def test_scenario_arc_fcp_5590_synthetic_regression_detected(monkeypatch) -> None:
    """A synthetic control-wins result is flagged as a regression, not silently dropped."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "as_action_like_import": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: {"levels": (1 if g == roster[0] else 0), "states_expanded": 10} for g in roster},
            {g: {"levels": 0, "states_expanded": 10} for g in roster},
            2.0,
        ),
    )

    artifact = mod.build_artifact(roster=("cd82", "sk48"))

    assert artifact["honest_verdict"] == "complete: dict_candidate_fix_regression_found"


def test_req_arc_fcp_5590_no_tautological_duplicate_timer_field() -> None:
    """REQ-ARC-FCP-5590: no redundant combined_wall_clock_s field (adversarial_verify.py
    TAUTOLOGY guard -- this script has no step between started_at and run_both_arms, so a
    second top-level timer would always match duration_s and trip the check on a structural
    coincidence, not a finding)."""

    assert "combined_wall_clock_s" not in mod.REQUIRED_ARTIFACT_FIELDS


def test_req_arc_fcp_5590_repository_artifact_confirms_real_headroomed_null() -> None:
    """REQ-ARC-FCP-5590: the checked-in real run is an honest, headroom-present null."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["levels_gained_headroom_present"] is True
    assert all(delta == 0 for delta in result["per_game_levels_delta"].values())
    assert result["states_expanded_control_total"] == result["states_expanded_treatment_total"]
    assert (
        result["inference_substrate"] == "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert "combined_wall_clock_s" not in result
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
