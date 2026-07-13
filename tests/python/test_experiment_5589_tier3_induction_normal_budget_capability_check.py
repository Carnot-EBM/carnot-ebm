"""Tests for Exp 5589 tier-3 induction normal-budget capability check.

Spec refs: REQ-ARC-WMTE-5589, SCENARIO-ARC-WMTE-5589-NORMAL-BUDGET-OUTCOME,
SCENARIO-ARC-WMTE-5589-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5589_tier3_induction_normal_budget_capability_check as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5589_spec_declares_capability_check_contract() -> None:
    """REQ-ARC-WMTE-5589: OpenSpec declares the required Exp 5589 gate schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5589") :]

    for marker in (
        "REQ-ARC-WMTE-5589",
        "SCENARIO-ARC-WMTE-5589-NORMAL-BUDGET-OUTCOME",
        "SCENARIO-ARC-WMTE-5589-BLOCKED-PRECONDITION",
        "stall_attempt_transition_count",
        "levels_reached",
        "development_proxy",
    ):
        assert marker in section


def test_scenario_arc_wmte_5589_blocked_precondition_never_constructs_policy(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5589-BLOCKED-PRECONDITION: a missing resource fails closed."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": False,
            "port_8920_prewarmed": True,
            "ok": False,
        },
    )

    def _fail_if_called(**_kwargs):
        raise AssertionError("run_capability_check must not run when a precondition is missing")

    monkeypatch.setattr(mod, "run_capability_check", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert "llama_server_binary_present" in artifact["honest_verdict"]
    assert artifact["induction_attempts"] == []
    assert artifact["stall_attempt_reached"] is False
    assert artifact["levels_reached"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_wmte_5589_normal_budget_useful_plan(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5589-NORMAL-BUDGET-OUTCOME: a planned, crash-free stall is classified correctly."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    synthetic_attempts = [
        {
            "reason": "stall",
            "transition_count": 30,
            "skipped": "",
            "planned": True,
            "heldout_accuracy": 0.8,
            "binary_gate_pass": True,
        }
    ]
    monkeypatch.setattr(
        mod,
        "run_capability_check",
        lambda **_kwargs: (synthetic_attempts, {"game": "m0r0", "levels": 1}),
    )

    artifact = mod.build_artifact()

    assert artifact["stall_attempt_reached"] is True
    assert artifact["stall_attempt_crashed"] is False
    assert artifact["stall_attempt_planned"] is True
    assert artifact["stall_attempt_transition_count"] == 30
    assert artifact["levels_reached"] == 1
    assert (
        artifact["honest_verdict"]
        == "complete: tier3_induction_useful_at_normal_budget_plan_produced"
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_wmte_5589_normal_budget_still_no_plan(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5589-NORMAL-BUDGET-OUTCOME: crash-free but trust-gate-rejected is honest, not hidden."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_capability_check",
        lambda **_kwargs: (
            [
                {
                    "reason": "stall",
                    "transition_count": 25,
                    "skipped": "hidden_state_trust_below_threshold",
                    "planned": False,
                    "heldout_accuracy": 0.125,
                    "binary_gate_pass": False,
                }
            ],
            {"game": "m0r0", "levels": 0},
        ),
    )

    artifact = mod.build_artifact()

    assert artifact["stall_attempt_crashed"] is False
    assert artifact["stall_attempt_planned"] is False
    assert (
        artifact["honest_verdict"]
        == "complete: tier3_induction_crash_free_but_still_no_usable_plan_at_normal_budget"
    )


def test_scenario_arc_wmte_5589_crash_detected_if_it_recurs(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5589-NORMAL-BUDGET-OUTCOME: a regression is caught, not hidden."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_capability_check",
        lambda **_kwargs: (
            [{"reason": "stall", "transition_count": 25, "skipped": "exception", "planned": False}],
            {"game": "m0r0", "levels": 0},
        ),
    )

    artifact = mod.build_artifact()

    assert artifact["stall_attempt_crashed"] is True
    assert artifact["honest_verdict"] == "complete: tier3_induction_still_crashes_fix_incomplete"


def test_req_arc_wmte_5589_repository_artifact_confirms_realistic_stall() -> None:
    """REQ-ARC-WMTE-5589: the checked-in real run confirms crash-free behavior at a realistic budget."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["stall_attempt_reached"] is True
    assert result["stall_attempt_crashed"] is False
    assert result["explore_budget_forced"] is False
    assert result["target_game"] == "m0r0"
    # a realistic (non-forced) transition count strictly greater than exp5588's forced 7
    assert result["stall_attempt_transition_count"] > 7
    assert any(a.get("reason") == "stall" for a in result["induction_attempts"])
    assert all(a.get("skipped") != "exception" for a in result["induction_attempts"])
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert result["flagged_adversarial"] is True
