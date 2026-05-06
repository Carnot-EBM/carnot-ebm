"""Tests for probability-calibration verification.

Spec coverage: REQ-VERIFY-1414, REQ-VERIFY-1415, SCENARIO-VERIFY-1414
"""

from __future__ import annotations

import pytest

from carnot.pipeline.probability_calibration_verifier import ProbabilityCalibrationVerifier
from carnot.pipeline.verify_repair import VerifyRepairPipeline


def test_probability_claim_inside_reference_range_passes() -> None:
    """REQ-VERIFY-1414: in-range probability claims pass with zero energy."""
    verifier = ProbabilityCalibrationVerifier(tolerance=0.05)
    chain = "In comparable historical cases, 30 out of 100 had rain."

    record = verifier.score(chain, "P(rain)=0.30")

    assert record.verdict == "pass"
    assert record.energy == pytest.approx(0.0)
    assert record.extras["claimed_probability"] == pytest.approx(0.30)
    assert record.extras["implied_probability"] == pytest.approx(0.30)
    assert record.extras["evidence_count"] == 1


def test_probability_claim_outside_reference_range_fails() -> None:
    """REQ-VERIFY-1414: overconfident probability claims fail with positive energy."""
    verifier = ProbabilityCalibrationVerifier(tolerance=0.05)
    chain = "In comparable historical cases, 30 out of 100 had rain."

    record = verifier.score(chain, "P(rain)=0.80")

    assert record.verdict == "fail"
    assert record.energy == pytest.approx(0.45)
    assert record.extras["implied_range"] == pytest.approx([0.25, 0.35])


def test_probability_claim_without_evidence_abstains() -> None:
    """REQ-VERIFY-1414: underdetermined probability claims abstain."""
    verifier = ProbabilityCalibrationVerifier()

    record = verifier.score("The model is uncertain but gives no base rate.", "P(rain)=0.60")

    assert record.verdict == "abstain"
    assert record.energy == 0.0
    assert record.extras["evidence_count"] == 0


def test_score_text_extracts_percent_chance_claim() -> None:
    """REQ-VERIFY-1414: score_text finds natural-language probability claims."""
    verifier = ProbabilityCalibrationVerifier(tolerance=0.10)
    chain = (
        "The base rate is 0.40 for similar incidents. "
        "Therefore there is a 40% chance of recurrence."
    )

    records = verifier.score_text(chain)

    assert len(records) == 1
    assert records[0].verdict == "pass"
    assert records[0].extras["event"] == "recurrence"


def test_pipeline_default_behavior_unchanged_without_probability_verifier() -> None:
    """REQ-VERIFY-1415: default pipeline does not add probability constraints."""
    pipeline = VerifyRepairPipeline(model=None)
    response = "In comparable cases, 30 out of 100 had rain. Therefore P(rain)=0.80."

    result = pipeline.verify("Will it rain?", response, domain="nl")

    assert result.verified is True
    assert not any(c.constraint_type == "probability_calibration" for c in result.constraints)


def test_pipeline_probability_verifier_adds_violation_and_energy() -> None:
    """SCENARIO-VERIFY-1414: opt-in verifier catches reference-class probability gaps."""
    pipeline = VerifyRepairPipeline(
        model=None,
        probability_calibration_verifier=ProbabilityCalibrationVerifier(tolerance=0.05),
    )
    response = "In comparable cases, 30 out of 100 had rain. Therefore P(rain)=0.80."

    result = pipeline.verify("Will it rain?", response, domain="nl")

    assert result.verified is False
    assert result.energy == pytest.approx(0.45)
    assert [v.constraint_type for v in result.violations] == ["probability_calibration"]
    record = result.violations[0].metadata["verdict_record"]
    assert record["extras"]["claimed_probability"] == pytest.approx(0.80)
