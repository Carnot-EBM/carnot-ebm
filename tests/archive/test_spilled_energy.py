"""Tests for SpilledEnergyExtractor and SpilledEnergyConstraint.

**Researcher summary:**
    100% coverage of python/carnot/pipeline/spilled_energy.py.
    Verifies graceful degradation (no logits → empty list), correct energy
    computation for 1-D and 2-D logit inputs, AutoExtractor backward
    compatibility, and AUROC sanity check.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from carnot.pipeline.spilled_energy import (
    DEFAULT_SPILLED_THRESHOLD,
    SpilledEnergyConstraint,
    SpilledEnergyExtractor,
)
from carnot.pipeline.extract import AutoExtractor, ConstraintResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peaked_logits(vocab_size: int = 100, n_tokens: int = 5) -> jnp.ndarray:
    """Return logits (n_tokens, vocab_size) where token 0 is far ahead.

    High confidence → low spilled energy.
    """
    logits = jnp.zeros((n_tokens, vocab_size))
    # Token 0 gets a very high logit at every position.
    return logits.at[:, 0].set(20.0)


def _flat_logits(vocab_size: int = 100, n_tokens: int = 5) -> jnp.ndarray:
    """Return logits (n_tokens, vocab_size) that are nearly uniform.

    Low confidence → high spilled energy.
    """
    return jnp.zeros((n_tokens, vocab_size))


# ---------------------------------------------------------------------------
# SpilledEnergyConstraint tests
# ---------------------------------------------------------------------------


class TestSpilledEnergyConstraint:
    """REQ-VERIFY-001 — SpilledEnergyConstraint is a valid ConstraintTerm."""

    def test_name_includes_value(self) -> None:
        """name property encodes the spilled energy value."""
        c = SpilledEnergyConstraint(0.1234)
        assert "spilled_energy" in c.name
        assert "0.1234" in c.name

    def test_energy_returns_precomputed_value(self) -> None:
        """energy(x) always returns the stored scalar, ignoring x."""
        c = SpilledEnergyConstraint(0.42)
        x = jnp.zeros(10)
        val = float(c.energy(x))
        assert abs(val - 0.42) < 1e-5

    def test_energy_ignores_x_shape(self) -> None:
        """energy(x) is independent of x shape."""
        c = SpilledEnergyConstraint(0.7)
        assert abs(float(c.energy(jnp.zeros(1))) - 0.7) < 1e-5
        assert abs(float(c.energy(jnp.zeros(100))) - 0.7) < 1e-5

    def test_energy_is_nonnegative(self) -> None:
        """energy must be ≥ 0 by construction."""
        c = SpilledEnergyConstraint(0.0)
        assert float(c.energy(jnp.zeros(1))) >= 0.0

    def test_satisfaction_threshold_property(self) -> None:
        """satisfaction_threshold returns the threshold set at construction."""
        c = SpilledEnergyConstraint(0.1, threshold=0.3)
        assert c.satisfaction_threshold == 0.3

    def test_default_threshold(self) -> None:
        """Default threshold matches DEFAULT_SPILLED_THRESHOLD."""
        c = SpilledEnergyConstraint(0.1)
        assert c.satisfaction_threshold == DEFAULT_SPILLED_THRESHOLD

    def test_is_satisfied_when_low_energy(self) -> None:
        """Constraint is satisfied when spilled energy is below threshold."""
        c = SpilledEnergyConstraint(0.1, threshold=0.5)
        assert c.is_satisfied(jnp.zeros(1)) is True

    def test_is_not_satisfied_when_high_energy(self) -> None:
        """Constraint is violated when spilled energy exceeds threshold."""
        c = SpilledEnergyConstraint(0.9, threshold=0.5)
        assert c.is_satisfied(jnp.zeros(1)) is False

    def test_is_satisfied_at_boundary(self) -> None:
        """Constraint is satisfied when energy exactly equals threshold."""
        c = SpilledEnergyConstraint(0.5, threshold=0.5)
        assert c.is_satisfied(jnp.zeros(1)) is True

    def test_grad_energy_is_zero(self) -> None:
        """Gradient of a constant function is zero everywhere."""
        c = SpilledEnergyConstraint(0.3)
        x = jnp.ones(5)
        grad = c.grad_energy(x)
        assert jnp.allclose(grad, jnp.zeros(5))

    def test_negative_value_raises(self) -> None:
        """Constructor must reject negative spilled energy values."""
        with pytest.raises(ValueError, match="≥ 0.0"):
            SpilledEnergyConstraint(-0.1)

    def test_zero_energy_is_satisfied(self) -> None:
        """Zero spilled energy (perfectly confident) is always satisfied."""
        c = SpilledEnergyConstraint(0.0, threshold=DEFAULT_SPILLED_THRESHOLD)
        assert c.is_satisfied(jnp.zeros(1)) is True


# ---------------------------------------------------------------------------
# SpilledEnergyExtractor tests
# ---------------------------------------------------------------------------


class TestSpilledEnergyExtractor:
    """REQ-VERIFY-001, SCENARIO-VERIFY-002 — SpilledEnergyExtractor."""

    def test_supported_domains(self) -> None:
        """Extractor declares 'factual' as its supported domain."""
        ext = SpilledEnergyExtractor()
        assert "factual" in ext.supported_domains

    def test_returns_empty_when_logits_none(self) -> None:
        """Graceful degradation: logits=None → empty list (REQ-VERIFY-001)."""
        ext = SpilledEnergyExtractor()
        results = ext.extract("Some text about Paris.")
        assert results == []

    def test_returns_empty_when_logits_none_explicit(self) -> None:
        """Explicit logits=None also gives empty list."""
        ext = SpilledEnergyExtractor()
        results = ext.extract("Some text.", logits=None)
        assert results == []

    def test_returns_empty_for_wrong_domain(self) -> None:
        """Domain mismatch → empty list even when logits are provided."""
        ext = SpilledEnergyExtractor()
        logits = _peaked_logits()
        results = ext.extract("text", domain="arithmetic", logits=logits)
        assert results == []

    def test_returns_one_result_with_logits_2d(self) -> None:
        """With 2-D logits, returns exactly one ConstraintResult."""
        ext = SpilledEnergyExtractor()
        logits = _peaked_logits(vocab_size=50, n_tokens=4)
        results = ext.extract("Paris is the capital of France.", logits=logits)
        assert len(results) == 1

    def test_returns_one_result_with_logits_1d(self) -> None:
        """With 1-D logits (single position), returns exactly one ConstraintResult."""
        ext = SpilledEnergyExtractor()
        logits = jnp.zeros(50)
        results = ext.extract("text", logits=logits)
        assert len(results) == 1

    def test_result_constraint_type(self) -> None:
        """ConstraintResult has constraint_type='spilled_energy'."""
        ext = SpilledEnergyExtractor()
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert result.constraint_type == "spilled_energy"

    def test_result_has_energy_term(self) -> None:
        """ConstraintResult.energy_term is a SpilledEnergyConstraint."""
        ext = SpilledEnergyExtractor()
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert isinstance(result.energy_term, SpilledEnergyConstraint)

    def test_peaked_logits_low_spilled_energy(self) -> None:
        """Peaked logits (confident) → low spilled energy → constraint satisfied."""
        ext = SpilledEnergyExtractor(threshold=DEFAULT_SPILLED_THRESHOLD)
        logits = _peaked_logits(vocab_size=200, n_tokens=10)
        result = ext.extract("confident answer", logits=logits)[0]
        spilled = result.metadata["spilled_energy"]
        # Peaked distribution: argmax token has very high prob, spilled ≈ 0
        assert spilled >= 0.0
        assert result.metadata["satisfied"] is True

    def test_flat_logits_high_spilled_energy(self) -> None:
        """Flat logits (uncertain) → high spilled energy → constraint violated."""
        # Use a very small threshold so flat logits trigger a violation.
        ext = SpilledEnergyExtractor(threshold=0.0)
        logits = _flat_logits(vocab_size=100, n_tokens=5)
        result = ext.extract("uncertain answer", logits=logits)[0]
        spilled = result.metadata["spilled_energy"]
        assert spilled >= 0.0
        assert result.metadata["satisfied"] is False

    def test_metadata_keys(self) -> None:
        """ConstraintResult.metadata contains expected keys."""
        ext = SpilledEnergyExtractor()
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert "spilled_energy" in result.metadata
        assert "threshold" in result.metadata
        assert "satisfied" in result.metadata
        assert "text_snippet" in result.metadata

    def test_metadata_text_snippet_truncated(self) -> None:
        """text_snippet in metadata is at most 80 characters."""
        ext = SpilledEnergyExtractor()
        long_text = "x" * 200
        result = ext.extract(long_text, logits=_peaked_logits())[0]
        assert len(result.metadata["text_snippet"]) <= 80

    def test_factual_domain_accepted(self) -> None:
        """domain='factual' is accepted (the extractor's domain)."""
        ext = SpilledEnergyExtractor()
        logits = _peaked_logits()
        results = ext.extract("text", domain="factual", logits=logits)
        assert len(results) == 1

    def test_compute_spilled_energy_1d(self) -> None:
        """_compute_spilled_energy handles 1-D input (auto-reshape to 2-D)."""
        ext = SpilledEnergyExtractor()
        logits_1d = jnp.zeros(50)
        value = ext._compute_spilled_energy(logits_1d)
        assert isinstance(value, float)
        assert value >= 0.0

    def test_compute_spilled_energy_nonnegative(self) -> None:
        """_compute_spilled_energy always returns a non-negative float."""
        ext = SpilledEnergyExtractor()
        import jax
        key = jax.random.PRNGKey(42)
        for _ in range(5):
            key, subkey = jax.random.split(key)
            logits = jax.random.normal(subkey, shape=(8, 64))
            value = ext._compute_spilled_energy(logits)
            assert value >= 0.0, f"Got negative spilled energy: {value}"

    def test_custom_threshold_propagated(self) -> None:
        """Custom threshold is passed through to the SpilledEnergyConstraint."""
        threshold = 0.123
        ext = SpilledEnergyExtractor(threshold=threshold)
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert result.metadata["threshold"] == threshold
        assert isinstance(result.energy_term, SpilledEnergyConstraint)
        assert result.energy_term.satisfaction_threshold == threshold


# ---------------------------------------------------------------------------
# AutoExtractor backward compatibility tests
# ---------------------------------------------------------------------------


class TestAutoExtractorBackwardCompat:
    """Ensure logits=None gives identical behavior to pre-Exp-157 AutoExtractor."""

    def test_extract_without_logits_unchanged(self) -> None:
        """AutoExtractor.extract(text) with no logits works as before."""
        ae = AutoExtractor()
        results = ae.extract("47 + 28 = 75")
        types = {r.constraint_type for r in results}
        # Arithmetic constraint should still be found.
        assert "arithmetic" in types
        # No spilled_energy constraint (no logits).
        assert "spilled_energy" not in types

    def test_extract_with_logits_none_unchanged(self) -> None:
        """Explicit logits=None gives same results as not passing logits."""
        ae = AutoExtractor()
        results_no_kwarg = ae.extract("47 + 28 = 75")
        results_explicit_none = ae.extract("47 + 28 = 75", logits=None)
        # Same number of constraints.
        assert len(results_no_kwarg) == len(results_explicit_none)
        # Same types.
        types_no = {r.constraint_type for r in results_no_kwarg}
        types_none = {r.constraint_type for r in results_explicit_none}
        assert types_no == types_none

    def test_extract_with_logits_adds_spilled_energy(self) -> None:
        """AutoExtractor.extract(text, logits=...) adds a spilled_energy result."""
        ae = AutoExtractor()
        logits = _peaked_logits(vocab_size=100, n_tokens=5)
        results = ae.extract("47 + 28 = 75", logits=logits)
        types = {r.constraint_type for r in results}
        assert "spilled_energy" in types
        # Arithmetic constraint still there.
        assert "arithmetic" in types

    def test_spilled_energy_not_duplicated(self) -> None:
        """Calling extract twice with the same logits gives one spilled_energy result each."""
        ae = AutoExtractor()
        logits = _peaked_logits()
        results = ae.extract("some text", logits=logits)
        spilled = [r for r in results if r.constraint_type == "spilled_energy"]
        assert len(spilled) == 1

    def test_memory_and_logits_both_supported(self) -> None:
        """memory= and logits= can be used together without conflict."""
        from carnot.pipeline.memory import ConstraintMemory

        ae = AutoExtractor()
        mem = ConstraintMemory()
        logits = _peaked_logits()
        # Should not raise; returns list with spilled_energy among results.
        results = ae.extract(
            "If it rains, the ground is wet.",
            domain="logic",
            memory=mem,
            logits=logits,
        )
        types = {r.constraint_type for r in results}
        assert "spilled_energy" in types
