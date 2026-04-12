"""Tests for lookahead-energy constraint extraction.

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from carnot.pipeline.lookahead_energy import (
    DEFAULT_LOOKAHEAD_THRESHOLD,
    LookaheadEnergyConstraint,
    LookaheadEnergyExtractor,
)


class TestLookaheadEnergyConstraint:
    """Tests for LookaheadEnergyConstraint."""

    def test_negative_energy_raises(self) -> None:
        """REQ-VERIFY-001: Negative lookahead energies are rejected."""
        with pytest.raises(ValueError, match="lookahead_energy_value must be"):
            LookaheadEnergyConstraint(-0.1)

    def test_constant_energy_and_properties(self) -> None:
        """REQ-VERIFY-001: Constraint exposes properties and constant energy."""
        constraint = LookaheadEnergyConstraint(1.25, threshold=1.5)

        assert constraint.name == "lookahead_energy(1.2500)"
        assert constraint.satisfaction_threshold == 1.5
        assert float(constraint.energy(jnp.ones(3))) == pytest.approx(1.25)
        assert constraint.is_satisfied(jnp.zeros(1)) is True

        violated = LookaheadEnergyConstraint(
            DEFAULT_LOOKAHEAD_THRESHOLD + 0.5,
            threshold=DEFAULT_LOOKAHEAD_THRESHOLD,
        )
        assert violated.is_satisfied(jnp.zeros(1)) is False


class TestLookaheadEnergyExtractor:
    """Tests for LookaheadEnergyExtractor."""

    def setup_method(self) -> None:
        self.ext = LookaheadEnergyExtractor(threshold=0.5)

    def test_supported_domains(self) -> None:
        """REQ-VERIFY-001: Extractor advertises the factual domain."""
        assert self.ext.supported_domains == ["factual"]

    def test_domain_filter_and_missing_logits_return_empty(self) -> None:
        """SCENARIO-VERIFY-002: Incompatible domain or missing logits degrades cleanly."""
        assert self.ext.extract("text", domain="arithmetic") == []
        assert self.ext.extract("text", domain="factual") == []

    def test_extract_returns_constraint_result(self) -> None:
        """REQ-VERIFY-001: 1-D logits are converted into one constraint result."""
        logits = jnp.array([3.0, 0.0, -1.0], dtype=jnp.float32)
        results = self.ext.extract("candidate response", logits=logits)

        assert len(results) == 1
        result = results[0]
        assert result.constraint_type == "lookahead_energy"
        assert result.energy_term is not None
        assert result.metadata["text_snippet"] == "candidate response"
        assert result.metadata["threshold"] == 0.5
        assert isinstance(result.metadata["lookahead_energy"], float)

    def test_compute_lookahead_energy_for_vector_and_matrix(self) -> None:
        """REQ-VERIFY-001: Lookahead energy handles both 1-D and 2-D logits."""
        vector_logits = jnp.array([2.0, 0.0], dtype=jnp.float32)
        matrix_logits = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)

        vector_energy = self.ext._compute_lookahead_energy(vector_logits)
        matrix_energy = self.ext._compute_lookahead_energy(matrix_logits)

        expected = -math.log(math.exp(2.0) / (math.exp(2.0) + math.exp(0.0)))
        assert vector_energy == pytest.approx(expected, rel=1e-6)
        assert matrix_energy == pytest.approx(expected, rel=1e-6)
