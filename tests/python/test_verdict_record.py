"""Tests for structured verification verdict records.

Spec: REQ-VERIFY-1408, REQ-VERIFY-1409, REQ-VERIFY-1410,
SCENARIO-VERIFY-1408
"""

from __future__ import annotations

import pytest

from carnot.pipeline.sink_probe import SinkProbe
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline
from carnot.pipeline.verdict_record import (
    VerdictRecord,
    calibrated_confidence_from_energy,
    fit_verdict_calibration,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


class _EORMStub:
    """Tiny EORM stand-in that forces ThreeTierPipeline to reach Ising."""

    def __init__(self, energy: float) -> None:
        self._energy = energy

    def energy(self, _cot_input: object) -> float:
        return self._energy


def test_verdict_record_to_dict_is_json_compatible() -> None:
    """REQ-VERIFY-1408: VerdictRecord serializes required fields."""

    record = VerdictRecord(
        verdict="pass",
        energy=0.25,
        calibrated_confidence=1.5,
        producing_tier=3,
        tier_reached=3,
        rationale="constraints_satisfied",
        budget_ms_consumed=12.5,
        extras={"nested": {"ok": True}},
    )

    payload = record.to_dict()

    assert payload["verdict"] == "pass"
    assert payload["energy"] == pytest.approx(0.25)
    assert payload["calibrated_confidence"] == 1.0
    assert payload["producing_tier"] == 3
    assert payload["tier_reached"] == 3
    assert payload["rationale"] == "constraints_satisfied"
    assert payload["budget_ms_consumed"] == pytest.approx(12.5)
    assert payload["repairs_applied"] == []
    assert payload["extras"] == {"nested": {"ok": True}}


def test_verdict_record_rejects_invalid_verdict() -> None:
    """REQ-VERIFY-1408: VerdictRecord validates verdict enum."""

    with pytest.raises(ValueError, match="verdict"):
        VerdictRecord(
            verdict="maybe",  # type: ignore[arg-type]
            energy=0.0,
            calibrated_confidence=0.5,
            producing_tier=0,
            tier_reached=0,
            rationale="bad",
            budget_ms_consumed=0.0,
        )


def test_calibrated_confidence_is_monotonic_in_negative_energy() -> None:
    """REQ-VERIFY-1409: Lower energy maps to higher pass confidence."""

    low_energy = calibrated_confidence_from_energy(0.0)
    mid_energy = calibrated_confidence_from_energy(1.0)
    high_energy = calibrated_confidence_from_energy(5.0)

    assert 0.0 <= high_energy <= mid_energy <= low_energy <= 1.0
    assert calibrated_confidence_from_energy(float("nan")) == 0.0
    assert calibrated_confidence_from_energy(float("-inf")) == 1.0
    assert calibrated_confidence_from_energy(float("inf")) == 0.0
    with pytest.raises(ValueError, match="temperature"):
        calibrated_confidence_from_energy(1.0, temperature=0.0)


def test_fit_verdict_calibration_uses_heldout_pairs() -> None:
    """REQ-VERIFY-1409: Held-out calibration fits pass-confidence parameters."""

    calibration = fit_verdict_calibration(
        [
            (0.0, True),
            (0.2, True),
            (0.4, True),
            (2.0, False),
            (3.0, False),
            (4.0, False),
        ]
    )

    assert calibration.n_heldout == 6
    assert calibration.temperature > 0.0
    assert 0.0 <= calibration.brier_score <= 1.0
    assert calibration.confidence(0.1) > calibration.confidence(3.0)
    assert calibration.to_dict()["n_heldout"] == 6

    with pytest.raises(ValueError, match="heldout_pairs"):
        fit_verdict_calibration([])


def test_verify_repair_pipeline_verify_record_preserves_pass_fail() -> None:
    """SCENARIO-VERIFY-1408: VerifyRepairPipeline emits structured records."""

    pipeline = VerifyRepairPipeline()

    passed = pipeline.verify_record(
        question="What is 3 + 4?",
        response="3 + 4 = 7.",
        domain="arithmetic",
    )
    failed = pipeline.verify_record(
        question="What is 3 + 4?",
        response="3 + 4 = 8.",
        domain="arithmetic",
    )
    legacy = pipeline.verify_legacy(
        question="What is 3 + 4?",
        response="3 + 4 = 7.",
        domain="arithmetic",
    )

    assert passed.verdict == "pass"
    assert failed.verdict == "fail"
    assert passed.producing_tier == 3
    assert failed.tier_reached == 3
    assert passed.budget_ms_consumed >= 0.0
    assert failed.budget_ms_consumed >= 0.0
    assert passed.calibrated_confidence >= failed.calibrated_confidence
    assert failed.rationale.startswith("constraint_violation")
    assert failed.extras["n_violations"] == 1
    assert legacy.verified is True


def test_three_tier_pipeline_verify_record_preserves_legacy_tuple() -> None:
    """REQ-VERIFY-1410: ThreeTierPipeline structured API is non-breaking."""

    def ising_stub(_response: str, _question: str) -> tuple[bool, float]:
        return False, 2.5

    pipeline = ThreeTierPipeline(
        sink_probe=SinkProbe(threshold=0.3),
        eorm_model=_EORMStub(energy=10.0),  # type: ignore[arg-type]
        ising_pipeline=ising_stub,
        eorm_threshold=0.0,
    )

    legacy = pipeline.verify("bad answer", question="Q?")
    legacy_alias = pipeline.verify_legacy("bad answer", question="Q?")
    record = pipeline.verify_record("bad answer", question="Q?")

    assert legacy == (False, "ising", 2.5)
    assert legacy_alias == legacy
    assert record.verdict == "fail"
    assert record.energy == pytest.approx(2.5)
    assert record.producing_tier == 3
    assert record.tier_reached == 3
    assert record.extras["tier_used"] == "ising"
    assert record.budget_ms_consumed >= 0.0
