"""Tests for Exp 5153 GAP-4 scale-up protocol ledger.

Spec refs: REQ-VERIFY-5153, SCENARIO-VERIFY-5153,
SCENARIO-VERIFY-5153-SUCCESS-GATE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import gap4_scaleup_v472_5153 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _steps(passed: bool) -> list[mod.ProtocolStep]:
    return [
        mod.ProtocolStep(step_id=step_id, passed=passed, evidence=f"{step_id} fixture")
        for step_id in mod.CANONICAL_PROTOCOL_STEP_IDS
    ]


def test_req_5153_spec_declares_gap4_scaleup_contract() -> None:
    """REQ-VERIFY-5153: OpenSpec declares the GAP-4 protocol-ledger contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5153",
        "SCENARIO-VERIFY-5153",
        "SCENARIO-VERIFY-5153-SUCCESS-GATE",
        "python/carnot/reporting/gap4_scaleup_v472_5153.py",
        "results/experiment_5153_gap4_scaleup_v472.json",
        "protocol_steps_completed",
        "n_400_task_result",
        "gap4_status_recommendation",
        "solve_provenance",
        "development_proxy",
        "six discordant wins",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_5153_exact_test_min6_rule_is_pinned() -> None:
    """REQ-VERIFY-5153: zero-loss design needs >=6 discordant wins for p<0.05."""

    assert mod.exact_two_sided_sign_p_value(0, 0) == 1.0
    assert mod.exact_two_sided_sign_p_value(5, 0) == pytest.approx(0.0625)
    assert mod.exact_test_passes_min6_rule(5, 0) is False
    assert mod.exact_two_sided_sign_p_value(6, 0) == pytest.approx(0.03125)
    assert mod.exact_test_passes_min6_rule(6, 0) is True
    assert mod.exact_test_passes_min6_rule(6, 1) is False
    with pytest.raises(ValueError, match="non-negative"):
        mod.exact_two_sided_sign_p_value(-1, 0)


def test_scenario_5153_incomplete_scaleup_remains_still_open(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5153: partial protocol completion cannot round up to filled."""

    artifact = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["n_400_task_result"] is None
    assert artifact["gap4_status_recommendation"] == "still_open"
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["protocol_acceptance_passed"] is False
    step_map = {step["step_id"]: step for step in artifact["protocol_steps_completed"]}
    assert set(step_map) == set(mod.CANONICAL_PROTOCOL_STEP_IDS)
    assert step_map["sandboxed_400_task_reconfirmation"]["passed"] is False
    assert step_map["genuinely_heldout_tasks"]["passed"] is False
    assert step_map["local_open_weight_generator_arm"]["passed"] is False
    assert artifact["exact_test_discordant_wins"] == 4
    assert artifact["exact_test_discordant_losses"] == 0
    assert artifact["exact_test_passes_min6_rule"] is False
    assert (tmp_path / mod.OUTPUT_REL).exists()
    assert json.loads((tmp_path / mod.OUTPUT_REL).read_text(encoding="utf-8")) == artifact


def test_scenario_5153_success_gate_requires_every_protocol_step() -> None:
    """SCENARIO-VERIFY-5153-SUCCESS-GATE: filled requires all steps plus numeric scale result."""

    filled = mod.build_artifact(
        protocol_steps=_steps(True),
        n_400_task_result=0.615,
        exact_test_discordant_wins=6,
        exact_test_discordant_losses=0,
        cluster_bootstrap_delta_ci95=[0.012, 0.117],
        prior_positive_context={"arc1_headroom_recovered": 4, "arc1_vote_wins_lost": 0},
    )
    mod.validate_artifact(filled)
    assert filled["honest_verdict"].startswith("success_")
    assert filled["gap4_status_recommendation"] == "filled"
    assert filled["protocol_acceptance_passed"] is True

    missing_scale = mod.build_artifact(
        protocol_steps=_steps(True),
        n_400_task_result=None,
        exact_test_discordant_wins=6,
        exact_test_discordant_losses=0,
        cluster_bootstrap_delta_ci95=[0.012, 0.117],
        prior_positive_context={},
    )
    assert missing_scale["gap4_status_recommendation"] == "still_open"

    weak_stats = mod.build_artifact(
        protocol_steps=_steps(True),
        n_400_task_result=0.615,
        exact_test_discordant_wins=5,
        exact_test_discordant_losses=0,
        cluster_bootstrap_delta_ci95=[0.012, 0.117],
        prior_positive_context={},
    )
    assert weak_stats["gap4_status_recommendation"] == "still_open"


def test_req_5153_validation_rejects_non_actionable_artifacts() -> None:
    """REQ-VERIFY-5153: validation rejects malformed protocol-ledger claims."""

    artifact = mod.build_artifact(
        protocol_steps=_steps(False),
        n_400_task_result=None,
        exact_test_discordant_wins=4,
        exact_test_discordant_losses=0,
        cluster_bootstrap_delta_ci95=None,
        prior_positive_context={},
    )
    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "protocol_steps_completed"}, "missing"),
        ({**artifact, "honest_verdict": "pending"}, "terminal"),
        ({**artifact, "gap4_status_recommendation": "maybe"}, "status recommendation"),
        ({**artifact, "solve_provenance": "live_hidden_game"}, "solve_provenance"),
        ({**artifact, "n_400_task_result": True}, "n_400_task_result"),
        ({**artifact, "exact_test_discordant_wins": -1}, "discordant"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        (
            {
                **artifact,
                "protocol_steps_completed": [{"step_id": "sandboxed_400_task_reconfirmation"}],
            },
            "step",
        ),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    with pytest.raises(ValueError, match="float or null"):
        mod.build_artifact(
            protocol_steps=_steps(True),
            n_400_task_result=True,
            exact_test_discordant_wins=6,
            exact_test_discordant_losses=0,
            cluster_bootstrap_delta_ci95=[0.0, 0.1],
            prior_positive_context={},
        )
    with pytest.raises(ValueError, match="finite"):
        mod.build_artifact(
            protocol_steps=_steps(True),
            n_400_task_result=float("nan"),
            exact_test_discordant_wins=6,
            exact_test_discordant_losses=0,
            cluster_bootstrap_delta_ci95=[0.0, 0.1],
            prior_positive_context={},
        )


def test_req_5153_main_writes_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-5153: CLI entrypoint delegates to the tested run path."""

    assert mod.main(["--root", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert payload["experiment"] == mod.EXPERIMENT
