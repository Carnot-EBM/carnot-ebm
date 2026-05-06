"""Tests for Exp 1396 FoVer semantic validation calibration.

Spec: REQ-VERIFY-1396, SCENARIO-VERIFY-1396
"""

from __future__ import annotations

import pytest

from carnot.verify.fover_semantic_calibration import (
    calibrated_fover_semantic_validation_row,
)


def _parsed(state: str) -> dict[str, object]:
    return {
        "parseable": True,
        "dispatched_state": state,
        "tag_state": state,
    }


def test_req1396_arithmetic_fallback_recovers_false_sat_on_incorrect_row() -> None:
    """REQ-VERIFY-1396: incorrect arithmetic rows do not accept DVI false SAT."""

    row = calibrated_fover_semantic_validation_row(
        case_id="math_0",
        response="The arithmetic claim says 2 + 2 = 5, so the step is valid.",
        label="incorrect",
        source="math_z3",
        parsed_row=_parsed("REPAIR_HINT"),
        dvi_incorrect_probability=0.64,
    )

    assert row["constraint_passed"] is True
    assert row["semantic_result"] == "REPAIR_HINT"
    assert row["fallback_applied"] is True
    assert row["fallback_route"] == "arithmetic_fallback"
    assert row["fallback_solver_verdict"] == "arithmetic_violation_detected"
    assert row["arithmetic_claim_count"] >= 1
    assert row["arithmetic_verifier_score"] == pytest.approx(1.0)


def test_scenario1396_dvi_abstains_for_correct_sat_certificate_near_threshold() -> None:
    """SCENARIO-VERIFY-1396: correct SAT rows inside the DVI band escalate."""

    row = calibrated_fover_semantic_validation_row(
        case_id="164",
        response="The remaining count is 20 - 4 = 16 and 0.25 * 16 = 4.",
        label="correct",
        source="fover_v4",
        parsed_row=_parsed("SAT"),
        dvi_incorrect_probability=0.724,
    )

    assert row["constraint_passed"] is True
    assert row["semantic_result"] == "SAT"
    assert row["fallback_applied"] is True
    assert row["fallback_route"] == "dvi_abstention_band"
    assert row["fallback_solver_verdict"] == "certificate_sat_accepted_after_abstention"


def test_req1396_keeps_hard_certificate_mismatches_failed() -> None:
    """REQ-VERIFY-1396: calibration does not hide certificate-state mismatches."""

    row = calibrated_fover_semantic_validation_row(
        case_id="bad_certificate",
        response="2 + 2 = 5",
        label="incorrect",
        source="math_z3",
        parsed_row=_parsed("SAT"),
        dvi_incorrect_probability=0.64,
    )

    assert row["constraint_passed"] is False
    assert row["failure_reason"] == "certificate_state_mismatch"
    assert row["fallback_applied"] is False
