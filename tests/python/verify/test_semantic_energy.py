"""Tests for Exp 2338 Semantic Energy synthetic-logit detector.

Spec: REQ-TIER0-007, SCENARIO-TIER0-007
"""

from __future__ import annotations

import numpy as np

from carnot.verify.semantic_energy import SemanticEnergyDetector


def _synthetic_logits(seed: int, sigma: float, n_responses: int = 12) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, sigma, size=32) for _ in range(n_responses)]


def _responses(n_responses: int = 12) -> list[str]:
    return [f"answer variant {idx % 3}" for idx in range(n_responses)]


def test_compute_energy_returns_scalar_float_for_valid_logits() -> None:
    """REQ-TIER0-007-1: compute_energy returns a scalar Python float."""
    detector = SemanticEnergyDetector()

    energy = detector.compute_energy(np.array([0.0, 1.0, -1.0]))

    assert isinstance(energy, float)
    assert np.isfinite(energy)


def test_detect_flags_high_variance_logits_as_hallucination() -> None:
    """SCENARIO-TIER0-007: high-variance synthetic logits cross the threshold."""
    detector = SemanticEnergyDetector(threshold=0.05)

    result = detector.detect(_synthetic_logits(seed=42, sigma=2.0), _responses())

    assert result["semantic_energy_score"] > detector.threshold
    assert result["is_hallucination_predicted"] is True


def test_detect_keeps_low_variance_logits_as_confident_correct() -> None:
    """SCENARIO-TIER0-007: low-variance synthetic logits stay below threshold."""
    detector = SemanticEnergyDetector(threshold=0.05)

    result = detector.detect(_synthetic_logits(seed=42, sigma=0.5), _responses())

    assert result["semantic_energy_score"] < detector.threshold
    assert result["is_hallucination_predicted"] is False
