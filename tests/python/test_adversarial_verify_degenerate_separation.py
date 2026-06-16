"""Tests for the DEGENERATE_SEPARATION adversarial-verify guard.

Spec refs: REQ-VERIFY-4297, SCENARIO-VERIFY-4297.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scripts.adversarial_verify as adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
EXP4282 = REPO / "results" / "experiment_4282_arcgen_cross_family_stress.json"
EXP4271 = REPO / "results" / "experiment_4271_arc_cross_family_transfer_existing_pool.json"


def _load_adversarial_verify() -> Any:
    return adversarial_verify


def _degenerate_flags(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [flag for flag in report["flags"] if flag["kind"] == "DEGENERATE_SEPARATION"]


def _check_payload(audit: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[Any] = []
    audit.check_degenerate_separation(payload, flags)
    return [flag.to_dict() for flag in flags]


def test_req_4297_spec_declares_degenerate_separation_guard() -> None:
    """REQ-VERIFY-4297: OpenSpec declares the mechanical safety check."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-4297" in spec
    assert "SCENARIO-VERIFY-4297" in spec
    assert "DEGENERATE_SEPARATION" in spec


def test_scenario_4297_flags_exp4282_but_spares_exp4271() -> None:
    """SCENARIO-VERIFY-4297: .396 degeneracy is flagged while .395 survives."""

    audit = _load_adversarial_verify()

    exp4282_report = audit.verify_artifact(EXP4282)
    exp4282_flags = _degenerate_flags(exp4282_report)

    assert exp4282_flags
    assert exp4282_flags[0]["severity"] == "critical"
    assert "delta=1.0" in exp4282_flags[0]["detail"]
    assert "vote_at_1=0.0" in exp4282_flags[0]["detail"]

    exp4271_report = audit.verify_artifact(EXP4271)

    assert _degenerate_flags(exp4271_report) == []


def test_req_4297_covers_matched_control_and_perfect_selector_signatures() -> None:
    """REQ-VERIFY-4297: both degenerate signatures are mechanically caught."""

    audit = _load_adversarial_verify()
    low_control_payload = {
        "honest_verdict": "complete: arcgen_cross_generator_generalizes",
        "cross_generator_delta": 0.96,
        "oracle_at_k": 0.82,
        "pass_rates": {
            "matched_control_at_1": 0.0,
            "set_encoder_at_1": 0.96,
        },
    }
    perfect_selector_payload = {
        "honest_verdict": "complete: cross_family_generalizes",
        "oracle_at_k": 1.0,
        "pass_rates": {
            "set_encoder_at_1": 1.0,
            "vote_at_1": 0.25,
        },
    }
    high_but_not_perfect_payload = {
        "honest_verdict": "complete: arcgen_cross_generator_generalizes",
        "cross_generator_delta": 0.94,
        "oracle_at_k": 0.99,
        "pass_rates": {
            "set_encoder_at_1": 0.98,
            "vote_at_1": 0.2,
            "matched_control_at_1": 0.18,
        },
    }
    unrelated_perfect_payload = {
        "honest_verdict": "complete: calibration_sanity_check",
        "oracle_at_k": 1.0,
        "pass_rates": {
            "set_encoder_at_1": 1.0,
            "vote_at_1": 0.0,
        },
    }

    assert _check_payload(audit, low_control_payload)[0]["kind"] == "DEGENERATE_SEPARATION"
    assert _check_payload(audit, perfect_selector_payload)[0]["kind"] == "DEGENERATE_SEPARATION"
    assert _check_payload(audit, high_but_not_perfect_payload) == []
    assert _check_payload(audit, unrelated_perfect_payload) == []
