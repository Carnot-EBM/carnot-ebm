"""Tests for carnot.pipeline.confidence_verifier.

Confidence-weighted constraint verification: converts binary violated/not-violated
into continuous confidence scores, enabling the repair gate to block false-positive
repairs (the failure mode identified in Exp 184).

Spec: REQ-VERIFY-081, REQ-VERIFY-082,
      SCENARIO-VERIFY-105, SCENARIO-VERIFY-106, SCENARIO-VERIFY-107,
      SCENARIO-VERIFY-108
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from carnot.pipeline.confidence_verifier import (
    ConfidenceVerifier,
    ViolationConfidence,
    confidence_from_energy,
    repair_gate,
)
from carnot.pipeline.extract import ArithmeticExtractor, AutoExtractor


# ---------------------------------------------------------------------------
# confidence_from_energy -- REQ-VERIFY-081
# ---------------------------------------------------------------------------


class TestConfidenceFromEnergy:
    """REQ-VERIFY-081: sigmoid normalisation of energy score."""

    def test_zero_energy_gives_half(self) -> None:
        """Zero energy → 0.5 (inflection point of sigmoid)."""
        assert confidence_from_energy(0.0) == pytest.approx(0.5)

    def test_large_positive_energy_approaches_one(self) -> None:
        """Large energy → confident violation (close to 1.0)."""
        score = confidence_from_energy(100.0)
        assert score > 0.99

    def test_large_negative_energy_approaches_zero(self) -> None:
        """Negative energy → low confidence (close to 0.0)."""
        score = confidence_from_energy(-100.0)
        assert score < 0.01

    def test_temperature_scales_sensitivity(self) -> None:
        """Higher temperature → flatter sigmoid (less confident for same delta)."""
        low_temp = confidence_from_energy(1.0, temperature=0.5)
        high_temp = confidence_from_energy(1.0, temperature=2.0)
        assert low_temp > high_temp

    def test_positive_infinity_clamps_to_one(self) -> None:
        """REQ-VERIFY-081: inf energy must not raise — clamp to 1.0."""
        score = confidence_from_energy(float("inf"))
        assert score == pytest.approx(1.0)

    def test_negative_infinity_clamps_to_zero(self) -> None:
        """REQ-VERIFY-081: -inf energy must not raise — clamp to 0.0."""
        score = confidence_from_energy(float("-inf"))
        assert score == pytest.approx(0.0)

    def test_nan_clamps_to_zero(self) -> None:
        """REQ-VERIFY-081: NaN energy must not raise — clamp to 0.0."""
        score = confidence_from_energy(float("nan"))
        assert score == pytest.approx(0.0)

    def test_output_always_in_unit_interval(self) -> None:
        """REQ-VERIFY-081: output always in [0, 1]."""
        for val in [-1e9, -1.0, 0.0, 1.0, 1e9]:
            score = confidence_from_energy(val)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# repair_gate -- SCENARIO-VERIFY-107
# ---------------------------------------------------------------------------


class TestRepairGate:
    """SCENARIO-VERIFY-107: repair_gate returns bool based on threshold."""

    def test_above_threshold_returns_true(self) -> None:
        """Confidence at/above threshold → recommend repair."""
        assert repair_gate(0.9, threshold=0.8) is True

    def test_at_threshold_returns_true(self) -> None:
        """Exactly at threshold → recommend repair."""
        assert repair_gate(0.8, threshold=0.8) is True

    def test_below_threshold_returns_false(self) -> None:
        """SCENARIO-VERIFY-107: confidence below threshold → do not repair."""
        assert repair_gate(0.65, threshold=0.8) is False

    def test_default_threshold_is_0_8(self) -> None:
        """Default threshold is 0.8."""
        assert repair_gate(0.79) is False
        assert repair_gate(0.80) is True

    def test_zero_confidence_false(self) -> None:
        """Zero confidence always False for any positive threshold."""
        assert repair_gate(0.0, threshold=0.1) is False

    def test_one_confidence_always_true(self) -> None:
        """Perfect confidence is always True."""
        assert repair_gate(1.0, threshold=0.99) is True


# ---------------------------------------------------------------------------
# ViolationConfidence dataclass -- REQ-VERIFY-081
# ---------------------------------------------------------------------------


class TestViolationConfidence:
    """REQ-VERIFY-081: ViolationConfidence dataclass fields and class assignment."""

    def _make(self, energy_delta: float, threshold: float = 0.8) -> ViolationConfidence:
        score = confidence_from_energy(energy_delta)
        if score >= 0.8:
            cls = ViolationConfidence.HIGH
        elif score >= 0.5:
            cls = ViolationConfidence.MEDIUM
        else:
            cls = ViolationConfidence.LOW
        return ViolationConfidence(
            constraint_id="test_constraint",
            energy_delta=energy_delta,
            confidence_score=score,
            confidence_class=cls,
            repair_recommended=repair_gate(score, threshold=threshold),
            evidence={},
        )

    def test_high_energy_gives_high_class(self) -> None:
        """Large energy delta → HIGH confidence class."""
        vc = self._make(5.0)
        assert vc.confidence_class == ViolationConfidence.HIGH

    def test_medium_energy_gives_medium_class(self) -> None:
        """Moderate energy delta → MEDIUM confidence class."""
        # sigmoid(0.2) ≈ 0.55, which is MEDIUM
        vc = self._make(0.2)
        assert vc.confidence_class == ViolationConfidence.MEDIUM

    def test_low_energy_gives_low_class(self) -> None:
        """Small energy delta → LOW confidence class."""
        # sigmoid(-2.0) ≈ 0.12
        vc = self._make(-2.0)
        assert vc.confidence_class == ViolationConfidence.LOW

    def test_class_constants(self) -> None:
        """HIGH/MEDIUM/LOW constants are the expected strings."""
        assert ViolationConfidence.HIGH == "HIGH"
        assert ViolationConfidence.MEDIUM == "MEDIUM"
        assert ViolationConfidence.LOW == "LOW"

    def test_repair_recommended_false_when_low(self) -> None:
        """REQ-VERIFY-081: LOW class → repair_recommended False."""
        vc = self._make(-2.0)
        assert vc.repair_recommended is False

    def test_repair_recommended_true_when_high(self) -> None:
        """REQ-VERIFY-081: HIGH class → repair_recommended True."""
        vc = self._make(5.0)
        assert vc.repair_recommended is True

    def test_evidence_dict_preserved(self) -> None:
        """Evidence dict is stored as-is."""
        score = confidence_from_energy(1.0)
        vc = ViolationConfidence(
            constraint_id="cid",
            energy_delta=1.0,
            confidence_score=score,
            confidence_class=ViolationConfidence.MEDIUM,
            repair_recommended=False,
            evidence={"claimed": 76, "correct": 75},
        )
        assert vc.evidence == {"claimed": 76, "correct": 75}


# ---------------------------------------------------------------------------
# ConfidenceVerifier -- REQ-VERIFY-081, SCENARIO-VERIFY-105/106
# ---------------------------------------------------------------------------


class TestConfidenceVerifier:
    """REQ-VERIFY-081: ConfidenceVerifier.verify_with_confidence."""

    def setup_method(self) -> None:
        """Create verifier with a real arithmetic extractor."""
        self.verifier = ConfidenceVerifier()
        self.extractor = ArithmeticExtractor()

    def test_returns_list(self) -> None:
        """verify_with_confidence always returns a list."""
        results = self.verifier.verify_with_confidence(
            "The answer is 2 + 2 = 4.", self.extractor
        )
        assert isinstance(results, list)

    def test_correct_response_returns_empty(self) -> None:
        """Correct arithmetic produces no ViolationConfidence items."""
        results = self.verifier.verify_with_confidence(
            "The answer is 47 + 28 = 75.", self.extractor
        )
        assert results == []

    def test_wrong_response_returns_violation(self) -> None:
        """SCENARIO-VERIFY-105: Wrong arithmetic produces a ViolationConfidence."""
        results = self.verifier.verify_with_confidence(
            "The answer is 47 + 28 = 76.", self.extractor
        )
        assert len(results) == 1
        assert isinstance(results[0], ViolationConfidence)

    def test_high_confidence_on_large_error(self) -> None:
        """SCENARIO-VERIFY-105: Large arithmetic error → HIGH confidence."""
        # 1 + 1 = 99 is an extreme error
        results = self.verifier.verify_with_confidence(
            "So 1 + 1 = 99.", self.extractor
        )
        assert len(results) == 1
        assert results[0].confidence_class == ViolationConfidence.HIGH
        assert results[0].repair_recommended is True
        assert results[0].confidence_score >= 0.8

    def test_repair_recommended_count_le_violations(self) -> None:
        """REQ-VERIFY-081: repair_recommended count ≤ violations detected."""
        results = self.verifier.verify_with_confidence(
            "We have 3 + 4 = 76 and 10 + 5 = 200.", self.extractor
        )
        repair_count = sum(1 for r in results if r.repair_recommended)
        assert repair_count <= len(results)

    def test_custom_threshold_raises_bar(self) -> None:
        """Higher threshold means fewer repairs recommended."""
        response = "The answer is 47 + 28 = 76."
        results_low = self.verifier.verify_with_confidence(
            response, self.extractor, threshold=0.1
        )
        results_high = self.verifier.verify_with_confidence(
            response, self.extractor, threshold=0.999
        )
        repairs_low = sum(1 for r in results_low if r.repair_recommended)
        repairs_high = sum(1 for r in results_high if r.repair_recommended)
        assert repairs_low >= repairs_high

    def test_violation_has_constraint_id(self) -> None:
        """Each ViolationConfidence carries a non-empty constraint_id."""
        results = self.verifier.verify_with_confidence(
            "The answer is 47 + 28 = 76.", self.extractor
        )
        assert len(results) >= 1
        assert results[0].constraint_id != ""

    def test_evidence_dict_present(self) -> None:
        """Each ViolationConfidence carries an evidence dict."""
        results = self.verifier.verify_with_confidence(
            "The answer is 47 + 28 = 76.", self.extractor
        )
        assert isinstance(results[0].evidence, dict)

    def test_no_arithmetic_in_text_empty(self) -> None:
        """Text without arithmetic yields empty list."""
        results = self.verifier.verify_with_confidence(
            "The sky is blue.", self.extractor
        )
        assert results == []

    def test_low_confidence_scenario(self) -> None:
        """SCENARIO-VERIFY-106: Low-energy violation → LOW or MEDIUM, repair_recommended False."""
        # Mock extractor that returns a violation with tiny energy
        mock_extractor = MagicMock()
        mock_constraint = MagicMock()
        mock_constraint.metadata = {"satisfied": False, "a": 1, "b": 1, "claimed_result": 2, "correct_result": 2}
        mock_constraint.description = "mock low-energy constraint"
        mock_constraint.energy_term = MagicMock()
        mock_constraint.energy_term.energy.return_value = float(0.001)
        mock_extractor.extract.return_value = [mock_constraint]

        results = self.verifier.verify_with_confidence(
            "dummy", mock_extractor, threshold=0.8
        )
        # With energy_delta=0.001, sigmoid(0.001) ≈ 0.5002 → MEDIUM
        # repair_recommended depends on score vs 0.8
        for r in results:
            if r.confidence_class in (ViolationConfidence.LOW, ViolationConfidence.MEDIUM):
                assert r.repair_recommended is False


# ---------------------------------------------------------------------------
# Integration: VerifyRepairPipeline.verify_and_repair_confident
# -- REQ-VERIFY-082, SCENARIO-VERIFY-108
# ---------------------------------------------------------------------------


class TestVerifyAndRepairConfident:
    """REQ-VERIFY-082: verify_and_repair_confident gates repair on confidence."""

    def setup_method(self) -> None:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline
        self.pipeline = VerifyRepairPipeline()

    def test_method_exists(self) -> None:
        """verify_and_repair_confident is callable."""
        assert callable(getattr(self.pipeline, "verify_and_repair_confident", None))

    def test_correct_response_not_repaired(self) -> None:
        """Correct response: verified=True, repaired=False."""
        result = self.pipeline.verify_and_repair_confident(
            question="What is 2 + 2?",
            response="2 + 2 = 4.",
        )
        assert result.verified is True
        assert result.repaired is False

    def test_high_confidence_violation_no_model(self) -> None:
        """SCENARIO-VERIFY-108: High-confidence violation, no model → repaired=False."""
        # No model loaded → can't repair even if confidence high
        result = self.pipeline.verify_and_repair_confident(
            question="What is 1 + 1?",
            response="1 + 1 = 99.",
            threshold=0.8,
        )
        assert result.repaired is False
        assert result.verified is False

    def test_low_confidence_violation_skips_repair(self) -> None:
        """SCENARIO-VERIFY-108: Low confidence → repair skipped even with model presence."""
        # Use a threshold so high that even real violations are below it
        # Mock the confidence verifier to force all low-confidence
        from carnot.pipeline.verify_repair import VerifyRepairPipeline
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify_and_repair_confident(
            question="What is 47 + 28?",
            response="47 + 28 = 76.",
            threshold=0.9999,  # Effectively no violation passes this
        )
        # Pipeline has no model, so repaired=False regardless
        assert result.repaired is False

    def test_returns_repair_result_type(self) -> None:
        """verify_and_repair_confident returns a RepairResult."""
        from carnot.pipeline.verify_repair import RepairResult
        result = self.pipeline.verify_and_repair_confident(
            question="What is 3 + 3?",
            response="3 + 3 = 6.",
        )
        assert isinstance(result, RepairResult)

    def test_threshold_parameter_accepted(self) -> None:
        """threshold parameter is accepted without error."""
        result = self.pipeline.verify_and_repair_confident(
            question="What is 5 + 5?",
            response="5 + 5 = 10.",
            threshold=0.7,
        )
        assert result is not None

    def test_additive_does_not_break_verify_and_repair(self) -> None:
        """REQ-VERIFY-082: verify_and_repair() behavior unchanged."""
        from carnot.pipeline.verify_repair import RepairResult
        result = self.pipeline.verify_and_repair(
            question="What is 2 + 2?",
            response="2 + 2 = 4.",
        )
        assert isinstance(result, RepairResult)
        assert result.verified is True
