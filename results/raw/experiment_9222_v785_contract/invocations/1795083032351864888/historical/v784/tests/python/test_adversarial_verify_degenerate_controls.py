"""Tests for the DEGENERATE_CONTROLS adversarial-verify guard.

Spec refs: REQ-VERIFY-4308, SCENARIO-VERIFY-4308.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scripts.adversarial_verify as adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
EXP4293 = REPO / "results" / "experiment_4293_diffusiongemma_energy_guided_run_partial_state.json"


def _degenerate_control_flags(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [flag for flag in report["flags"] if flag["kind"] == "DEGENERATE_CONTROLS"]


def _check_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[Any] = []
    adversarial_verify.check_degenerate_controls(payload, flags)
    return [flag.to_dict() for flag in flags]


def test_req_4308_spec_declares_degenerate_controls_guard() -> None:
    """REQ-VERIFY-4308: OpenSpec declares the controls no-op guard."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-4308" in spec
    assert "SCENARIO-VERIFY-4308" in spec
    assert "DEGENERATE_CONTROLS" in spec


def test_scenario_4308_flags_exp4293_degenerate_controls() -> None:
    """SCENARIO-VERIFY-4308: Exp 4293's identical controls are flagged."""

    report = adversarial_verify.verify_artifact(EXP4293)
    flags = _degenerate_control_flags(report)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "condition_accuracy" in flags[0]["detail"]
    assert "entrgi=0.3" in flags[0]["detail"]
    assert "rfg=0.3" in flags[0]["detail"]
    assert "unguided=0.3" in flags[0]["detail"]


def test_req_4308_spares_differentiated_single_and_documented_placebo_controls() -> None:
    """REQ-VERIFY-4308: legitimate control shapes are not false positives."""

    differentiated = {
        "condition_accuracy": {
            "carnot": 0.83,
            "rfg": 0.45,
            "unguided": 0.31,
            "entrgi": 0.29,
        }
    }
    single_control = {"condition_accuracy": {"unguided": 0.31}}
    documented_placebo = {
        "identical_control_arms_expected": True,
        "condition_accuracy": {
            "placebo_a": 0.5,
            "placebo_b": 0.5,
            "unguided": 0.42,
        },
    }

    assert _check_payload(differentiated) == []
    assert _check_payload(single_control) == []
    assert _check_payload(documented_placebo) == []


def test_req_4308_covers_nested_arms_accuracy_maps() -> None:
    """REQ-VERIFY-4308: nested arms maps use each arm's accuracy metric."""

    flags = _check_payload(
        {
            "arms": {
                "carnot": {"accuracy": 0.8},
                "rfg": {"accuracy": 0.25},
                "unguided": {"accuracy": 0.25},
                "entrgi": {"accuracy": 0.25},
            }
        }
    )

    assert flags
    assert flags[0]["kind"] == "DEGENERATE_CONTROLS"
