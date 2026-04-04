"""Tests for semantic energy hallucination detection.

Spec coverage: REQ-INFER-008, REQ-INFER-009,
               SCENARIO-INFER-008, SCENARIO-INFER-009
"""

from carnot.inference.semantic_energy import (
    SemanticEnergyResult,
    classify_hallucination,
    compute_semantic_energy,
)


class TestComputeSemanticEnergy:
    """Tests for compute_semantic_energy — REQ-INFER-008."""

    def test_known_logprobs(self) -> None:
        """REQ-INFER-008: energy = mean(-logprob) for known values."""
        # logprobs: [-1.0, -2.0, -3.0] => energies: [1.0, 2.0, 3.0] => mean = 2.0
        energy = compute_semantic_energy([-1.0, -2.0, -3.0])
        assert abs(energy - 2.0) < 1e-9

    def test_empty_logprobs(self) -> None:
        """REQ-INFER-008: empty logprobs returns 0.0 gracefully."""
        energy = compute_semantic_energy([])
        assert energy == 0.0

    def test_single_token(self) -> None:
        """REQ-INFER-008: single token energy = -logprob."""
        energy = compute_semantic_energy([-3.5])
        assert abs(energy - 3.5) < 1e-9

    def test_high_confidence_low_energy(self) -> None:
        """SCENARIO-INFER-008: confident tokens produce low energy."""
        # Very high confidence: logprobs close to 0
        logprobs = [-0.1, -0.2, -0.1]
        energy = compute_semantic_energy(logprobs)
        # Expected: mean of [0.1, 0.2, 0.1] = 0.4/3 ~ 0.1333
        assert energy < 0.5
        assert abs(energy - 0.4 / 3) < 1e-9

    def test_low_confidence_high_energy(self) -> None:
        """SCENARIO-INFER-009: uncertain tokens produce high energy."""
        logprobs = [-5.0, -6.0, -4.5]
        energy = compute_semantic_energy(logprobs)
        # Expected: mean of [5.0, 6.0, 4.5] = 15.5/3 ~ 5.1667
        assert energy > 2.0
        expected = 15.5 / 3
        assert abs(energy - expected) < 1e-9

    def test_zero_logprob_zero_energy(self) -> None:
        """REQ-INFER-008: logprob of 0 (100% confidence) => zero energy."""
        energy = compute_semantic_energy([0.0, 0.0, 0.0])
        assert energy == 0.0


class TestClassifyHallucination:
    """Tests for classify_hallucination — REQ-INFER-009."""

    def test_below_threshold_not_hallucination(self) -> None:
        """REQ-INFER-009: energy below threshold => not a hallucination."""
        assert classify_hallucination(1.5, threshold=2.0) is False

    def test_above_threshold_is_hallucination(self) -> None:
        """REQ-INFER-009: energy above threshold => hallucination."""
        assert classify_hallucination(2.5, threshold=2.0) is True

    def test_exact_threshold_not_hallucination(self) -> None:
        """REQ-INFER-009: energy exactly at threshold => not hallucination (strict >)."""
        assert classify_hallucination(2.0, threshold=2.0) is False

    def test_custom_threshold(self) -> None:
        """REQ-INFER-009: custom threshold is respected."""
        # With threshold=1.0, energy of 1.5 IS a hallucination
        assert classify_hallucination(1.5, threshold=1.0) is True
        # With threshold=3.0, energy of 1.5 is NOT
        assert classify_hallucination(1.5, threshold=3.0) is False

    def test_default_threshold(self) -> None:
        """REQ-INFER-009: default threshold is 2.0."""
        assert classify_hallucination(1.9) is False
        assert classify_hallucination(2.1) is True


class TestSemanticEnergyResult:
    """Tests for SemanticEnergyResult dataclass — REQ-INFER-008, REQ-INFER-009."""

    def test_defaults(self) -> None:
        """REQ-INFER-008: default result has zero energy, no hallucination."""
        result = SemanticEnergyResult()
        assert result.energy == 0.0
        assert result.is_hallucination is False
        assert result.token_energies == []
        assert result.threshold == 2.0

    def test_from_logprobs_low_energy(self) -> None:
        """SCENARIO-INFER-008: from_logprobs with confident tokens."""
        result = SemanticEnergyResult.from_logprobs([-0.1, -0.2, -0.1])
        assert result.energy < 1.0
        assert result.is_hallucination is False
        assert len(result.token_energies) == 3
        assert all(te >= 0.0 for te in result.token_energies)
        # Token energies should be the negated logprobs
        assert abs(result.token_energies[0] - 0.1) < 1e-9
        assert abs(result.token_energies[1] - 0.2) < 1e-9
        assert abs(result.token_energies[2] - 0.1) < 1e-9

    def test_from_logprobs_high_energy(self) -> None:
        """SCENARIO-INFER-009: from_logprobs with uncertain tokens flags hallucination."""
        result = SemanticEnergyResult.from_logprobs([-5.0, -6.0, -4.5])
        expected_energy = 15.5 / 3
        assert abs(result.energy - expected_energy) < 1e-9
        assert result.is_hallucination is True
        assert len(result.token_energies) == 3

    def test_from_logprobs_custom_threshold(self) -> None:
        """REQ-INFER-009: from_logprobs respects custom threshold."""
        # Energy will be 1.0 — hallucination only if threshold < 1.0
        result_high = SemanticEnergyResult.from_logprobs([-1.0], threshold=0.5)
        assert result_high.is_hallucination is True
        assert result_high.threshold == 0.5

        result_low = SemanticEnergyResult.from_logprobs([-1.0], threshold=2.0)
        assert result_low.is_hallucination is False
        assert result_low.threshold == 2.0

    def test_from_logprobs_empty(self) -> None:
        """REQ-INFER-008: from_logprobs with empty list returns zero energy."""
        result = SemanticEnergyResult.from_logprobs([])
        assert result.energy == 0.0
        assert result.is_hallucination is False
        assert result.token_energies == []
