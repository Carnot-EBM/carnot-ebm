"""Tests for Exp 5602 InertClickSigPruner matched-budget A/B (the flip-decision
measurement task 9's own wiring follow-on named as still pending).

Spec refs: REQ-ARC-FCP-5595-2, SCENARIO-ARC-FCP-5595-2-MATCHED-BUDGET-AB.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5602_inert_click_pruner_matched_budget_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5595_2_spec_declares_matched_budget_ab_contract() -> None:
    """REQ-ARC-FCP-5595-2: OpenSpec declares the matched-budget A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5595-2") :]

    for marker in (
        "REQ-ARC-FCP-5595-2",
        "SCENARIO-ARC-FCP-5595-2-MATCHED-BUDGET-AB",
        "states_expanded",
        "reproduction gate",
    ):
        assert marker in section


def test_scenario_5602_blocked_precondition_never_runs(monkeypatch) -> None:
    """A missing resource fails closed without attempting any solve."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "adapter_registered": False,
            "inert_click_pruner_import": True,
            "solve_adaptered_import": True,
            "offline_arcade_makes_env": True,
            "ok": False,
        },
    )

    def _fail_if_called(**_kwargs):
        raise AssertionError("_run_arm must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_run_arm", _fail_if_called)
    monkeypatch.setattr(mod, "_live_wired_check", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["baseline"] == {}
    assert artifact["treatment"] == {}
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def _ok_preconds(root=mod.REPO_ROOT):
    return {
        "adapter_registered": True,
        "inert_click_pruner_import": True,
        "solve_adaptered_import": True,
        "offline_arcade_makes_env": True,
        "ok": True,
    }


def test_scenario_5602_reduction_confirmed_when_treatment_prunes(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5595-2-MATCHED-BUDGET-AB: a genuine states_expanded reduction with
    both arms reproduced and reaching the same level is classified as a real win."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    def _fake_run_arm(*, game, target_level, inert_click_prune):
        if inert_click_prune:
            return {
                "reached_level": 2,
                "states_expanded": 30,
                "offline_reproduced": True,
                "inert_click_prune": True,
                "pruner_stats": {"pruned": 8},
            }
        return {
            "reached_level": 2,
            "states_expanded": 38,
            "offline_reproduced": True,
            "inert_click_prune": False,
            "pruner_stats": None,
        }

    monkeypatch.setattr(mod, "_run_arm", _fake_run_arm)
    monkeypatch.setattr(
        mod,
        "_live_wired_check",
        lambda **_kwargs: {
            "transitions_collected": 37,
            "pruner_stats": {"pruned": 3},
        },
    )

    artifact = mod.build_artifact()

    assert artifact["states_expanded_reduction"] == 8
    assert artifact["honest_verdict"] == (
        "complete: inert_click_pruner_ab_reduces_states_expanded_8_live_pruned_3_on_m0r0"
    )


def test_scenario_5602_no_op_when_neither_offline_nor_live_prunes(monkeypatch) -> None:
    """A zero-reduction, zero-live-pruned outcome is an honest no-op verdict, not a failure."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "_run_arm",
        lambda *, game, target_level, inert_click_prune: {
            "reached_level": 2,
            "states_expanded": 38,
            "offline_reproduced": True,
            "inert_click_prune": inert_click_prune,
            "pruner_stats": {"pruned": 0} if inert_click_prune else None,
        },
    )
    monkeypatch.setattr(
        mod,
        "_live_wired_check",
        lambda **_kwargs: {
            "transitions_collected": 37,
            "pruner_stats": {"pruned": 0},
        },
    )

    artifact = mod.build_artifact()

    assert artifact["states_expanded_reduction"] == 0
    assert artifact["honest_verdict"] == (
        "complete: inert_click_pruner_ab_no_op_offline_and_live_at_this_budget_on_m0r0"
    )


def test_scenario_5602_regression_detected_when_reproduction_or_level_drops(monkeypatch) -> None:
    """A treatment run that fails the reproduction gate or reaches a lower level is a
    regression, regardless of any states_expanded reduction -- the correctness backstop
    overrides the efficiency number."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)

    def _fake_run_arm(*, game, target_level, inert_click_prune):
        if inert_click_prune:
            return {
                "reached_level": 1,
                "states_expanded": 10,
                "offline_reproduced": False,
                "inert_click_prune": True,
                "pruner_stats": {"pruned": 20},
            }
        return {
            "reached_level": 2,
            "states_expanded": 38,
            "offline_reproduced": True,
            "inert_click_prune": False,
            "pruner_stats": None,
        }

    monkeypatch.setattr(mod, "_run_arm", _fake_run_arm)
    monkeypatch.setattr(
        mod, "_live_wired_check", lambda **_kwargs: {"transitions_collected": 0, "pruner_stats": {}}
    )

    artifact = mod.build_artifact()

    assert (
        artifact["honest_verdict"]
        == "complete: inert_click_pruner_ab_regressed_reproduction_or_level"
    )


def test_scenario_5602_live_check_failure_is_non_fatal(monkeypatch) -> None:
    """A raising live-wired supplementary check must not break the primary A/B result."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "_run_arm",
        lambda *, game, target_level, inert_click_prune: {
            "reached_level": 2,
            "states_expanded": 38,
            "offline_reproduced": True,
            "inert_click_prune": inert_click_prune,
            "pruner_stats": None,
        },
    )

    def _raises(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_live_wired_check", _raises)

    artifact = mod.build_artifact()

    assert "error" in artifact["live_wired_supplementary_check"]
    assert artifact["honest_verdict"].startswith("complete: inert_click_pruner_ab_no_op")


def test_req_arc_fcp_5595_2_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-FCP-5595-2: the checked-in real run measured InertClickSigPruner against a
    real OfflineSolver A/B on m0r0 -- zero regression confirmed (both arms reproduced, same
    level reached), with an independent live-wired supplementary check corroborating the
    same honest null (no pruning at this budget, offline or live)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["inference_substrate"] == "offline_arc_sim_no_quota"
    assert result["solve_provenance"] == "development_proxy"
    assert result["baseline"]["offline_reproduced"] is True
    assert result["treatment"]["offline_reproduced"] is True
    assert result["baseline"]["reached_level"] == result["treatment"]["reached_level"]
    assert result["duration_s"] > 1.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
