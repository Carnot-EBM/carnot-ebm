"""Tests for LookaheadEnergyExtractor and LookaheadEnergyConstraint.

**Researcher summary:**
    100% coverage of python/carnot/pipeline/lookahead_energy.py.
    Verifies graceful degradation (no logits → empty list), correct energy
    computation for 1-D and 2-D logit inputs, ConstraintTerm protocol
    compliance, AutoExtractor integration (both spilled_energy and
    lookahead_energy added when logits provided), and peak vs flat logit
    discrimination.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from carnot.pipeline.lookahead_energy import (
    DEFAULT_LOOKAHEAD_THRESHOLD,
    LookaheadEnergyConstraint,
    LookaheadEnergyExtractor,
)
from carnot.pipeline.extract import AutoExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peaked_logits(vocab_size: int = 100, n_tokens: int = 5) -> jnp.ndarray:
    """Return logits (n_tokens, vocab_size) where token 0 is far ahead.

    High confidence → low NLL → low lookahead energy.
    """
    logits = jnp.zeros((n_tokens, vocab_size))
    # Token 0 gets a very high logit at every position.
    return logits.at[:, 0].set(20.0)


def _flat_logits(vocab_size: int = 100, n_tokens: int = 5) -> jnp.ndarray:
    """Return logits (n_tokens, vocab_size) that are nearly uniform.

    Low confidence → high NLL → high lookahead energy.
    """
    return jnp.zeros((n_tokens, vocab_size))


# ---------------------------------------------------------------------------
# LookaheadEnergyConstraint tests
# ---------------------------------------------------------------------------


class TestLookaheadEnergyConstraint:
    """REQ-VERIFY-001 — LookaheadEnergyConstraint is a valid ConstraintTerm."""

    def test_name_includes_lookahead_energy(self) -> None:
        """name property includes 'lookahead_energy' and the value."""
        c = LookaheadEnergyConstraint(1.2345)
        assert "lookahead_energy" in c.name
        assert "1.2345" in c.name

    def test_energy_returns_precomputed_value(self) -> None:
        """energy(x) always returns the stored scalar, ignoring x."""
        c = LookaheadEnergyConstraint(1.5)
        x = jnp.zeros(10)
        val = float(c.energy(x))
        assert abs(val - 1.5) < 1e-5

    def test_energy_ignores_x_shape(self) -> None:
        """energy(x) is independent of x shape."""
        c = LookaheadEnergyConstraint(2.0)
        assert abs(float(c.energy(jnp.zeros(1))) - 2.0) < 1e-5
        assert abs(float(c.energy(jnp.zeros(100))) - 2.0) < 1e-5

    def test_energy_is_nonnegative(self) -> None:
        """energy must be ≥ 0 by construction."""
        c = LookaheadEnergyConstraint(0.0)
        assert float(c.energy(jnp.zeros(1))) >= 0.0

    def test_satisfaction_threshold_property(self) -> None:
        """satisfaction_threshold returns the threshold set at construction."""
        c = LookaheadEnergyConstraint(1.0, threshold=3.0)
        assert c.satisfaction_threshold == 3.0

    def test_default_threshold(self) -> None:
        """Default threshold matches DEFAULT_LOOKAHEAD_THRESHOLD."""
        c = LookaheadEnergyConstraint(1.0)
        assert c.satisfaction_threshold == DEFAULT_LOOKAHEAD_THRESHOLD

    def test_is_satisfied_when_low_energy(self) -> None:
        """Constraint is satisfied when lookahead energy is below threshold."""
        c = LookaheadEnergyConstraint(1.0, threshold=2.0)
        assert c.is_satisfied(jnp.zeros(1)) is True

    def test_is_not_satisfied_when_high_energy(self) -> None:
        """Constraint is violated when lookahead energy exceeds threshold."""
        c = LookaheadEnergyConstraint(3.0, threshold=2.0)
        assert c.is_satisfied(jnp.zeros(1)) is False

    def test_is_satisfied_at_boundary(self) -> None:
        """Constraint is satisfied when energy exactly equals threshold."""
        c = LookaheadEnergyConstraint(2.0, threshold=2.0)
        assert c.is_satisfied(jnp.zeros(1)) is True

    def test_grad_energy_is_zero(self) -> None:
        """Gradient of a constant energy function is zero everywhere."""
        c = LookaheadEnergyConstraint(1.5)
        x = jnp.ones(5)
        grad = c.grad_energy(x)
        assert jnp.allclose(grad, jnp.zeros(5))

    def test_negative_value_raises(self) -> None:
        """Constructor must reject negative lookahead energy values."""
        with pytest.raises(ValueError, match="≥ 0.0"):
            LookaheadEnergyConstraint(-0.1)

    def test_zero_energy_is_satisfied(self) -> None:
        """Zero lookahead energy is always satisfied (perfectly confident)."""
        c = LookaheadEnergyConstraint(0.0, threshold=DEFAULT_LOOKAHEAD_THRESHOLD)
        assert c.is_satisfied(jnp.zeros(1)) is True

    def test_energy_returns_jax_array(self) -> None:
        """energy(x) returns a JAX array (not a plain Python float)."""
        c = LookaheadEnergyConstraint(1.0)
        result = c.energy(jnp.zeros(1))
        # Should be a JAX array (has .shape attribute).
        assert hasattr(result, "shape")


# ---------------------------------------------------------------------------
# LookaheadEnergyExtractor tests
# ---------------------------------------------------------------------------


class TestLookaheadEnergyExtractor:
    """REQ-VERIFY-001, SCENARIO-VERIFY-002 — LookaheadEnergyExtractor."""

    def test_supported_domains(self) -> None:
        """Extractor declares 'factual' as its supported domain."""
        ext = LookaheadEnergyExtractor()
        assert "factual" in ext.supported_domains

    def test_returns_empty_when_logits_none(self) -> None:
        """Graceful degradation: logits=None → empty list (REQ-VERIFY-001)."""
        ext = LookaheadEnergyExtractor()
        results = ext.extract("Some text about Paris.")
        assert results == []

    def test_returns_empty_when_logits_none_explicit(self) -> None:
        """Explicit logits=None also gives empty list."""
        ext = LookaheadEnergyExtractor()
        results = ext.extract("Some text.", logits=None)
        assert results == []

    def test_returns_empty_for_wrong_domain(self) -> None:
        """Domain mismatch → empty list even when logits are provided."""
        ext = LookaheadEnergyExtractor()
        logits = _peaked_logits()
        results = ext.extract("text", domain="arithmetic", logits=logits)
        assert results == []

    def test_returns_one_result_with_logits_2d(self) -> None:
        """With 2-D logits, returns exactly one ConstraintResult."""
        ext = LookaheadEnergyExtractor()
        logits = _peaked_logits(vocab_size=50, n_tokens=4)
        results = ext.extract("Paris is the capital of France.", logits=logits)
        assert len(results) == 1

    def test_returns_one_result_with_logits_1d(self) -> None:
        """With 1-D logits (single position), returns exactly one ConstraintResult."""
        ext = LookaheadEnergyExtractor()
        logits = jnp.zeros(50)
        results = ext.extract("text", logits=logits)
        assert len(results) == 1

    def test_result_constraint_type(self) -> None:
        """ConstraintResult has constraint_type='lookahead_energy'."""
        ext = LookaheadEnergyExtractor()
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert result.constraint_type == "lookahead_energy"

    def test_result_has_energy_term(self) -> None:
        """ConstraintResult.energy_term is a LookaheadEnergyConstraint."""
        ext = LookaheadEnergyExtractor()
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert isinstance(result.energy_term, LookaheadEnergyConstraint)

    def test_peaked_logits_low_lookahead_energy(self) -> None:
        """Peaked logits (confident) → low NLL → constraint satisfied."""
        ext = LookaheadEnergyExtractor(threshold=DEFAULT_LOOKAHEAD_THRESHOLD)
        logits = _peaked_logits(vocab_size=200, n_tokens=10)
        result = ext.extract("confident answer", logits=logits)[0]
        energy = result.metadata["lookahead_energy"]
        # Peaked distribution: argmax token has very high prob, NLL ≈ 0.
        assert energy >= 0.0
        assert result.metadata["satisfied"] is True

    def test_flat_logits_high_lookahead_energy(self) -> None:
        """Flat logits (uncertain) → high NLL → constraint violated with tight threshold."""
        # Use a very small threshold so flat logits trigger a violation.
        ext = LookaheadEnergyExtractor(threshold=0.0)
        logits = _flat_logits(vocab_size=100, n_tokens=5)
        result = ext.extract("uncertain answer", logits=logits)[0]
        energy = result.metadata["lookahead_energy"]
        assert energy >= 0.0
        assert result.metadata["satisfied"] is False

    def test_peak_lower_energy_than_flat(self) -> None:
        """Peaked logits produce lower lookahead energy than flat logits."""
        ext = LookaheadEnergyExtractor()
        peaked = _peaked_logits(vocab_size=100, n_tokens=10)
        flat = _flat_logits(vocab_size=100, n_tokens=10)

        peak_result = ext.extract("peak", logits=peaked)[0]
        flat_result = ext.extract("flat", logits=flat)[0]

        peak_energy = peak_result.metadata["lookahead_energy"]
        flat_energy = flat_result.metadata["lookahead_energy"]

        assert peak_energy < flat_energy, (
            f"Expected peak energy {peak_energy:.4f} < flat energy {flat_energy:.4f}"
        )

    def test_metadata_keys(self) -> None:
        """ConstraintResult.metadata contains expected keys."""
        ext = LookaheadEnergyExtractor()
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert "lookahead_energy" in result.metadata
        assert "threshold" in result.metadata
        assert "satisfied" in result.metadata
        assert "text_snippet" in result.metadata

    def test_metadata_text_snippet_truncated(self) -> None:
        """text_snippet in metadata is at most 80 characters."""
        ext = LookaheadEnergyExtractor()
        long_text = "x" * 200
        result = ext.extract(long_text, logits=_peaked_logits())[0]
        assert len(result.metadata["text_snippet"]) <= 80

    def test_factual_domain_accepted(self) -> None:
        """domain='factual' is accepted (the extractor's domain)."""
        ext = LookaheadEnergyExtractor()
        logits = _peaked_logits()
        results = ext.extract("text", domain="factual", logits=logits)
        assert len(results) == 1

    def test_compute_lookahead_energy_1d(self) -> None:
        """_compute_lookahead_energy handles 1-D input (auto-reshape to 2-D)."""
        ext = LookaheadEnergyExtractor()
        logits_1d = jnp.zeros(50)
        value = ext._compute_lookahead_energy(logits_1d)
        assert isinstance(value, float)
        assert value >= 0.0

    def test_compute_lookahead_energy_nonnegative(self) -> None:
        """_compute_lookahead_energy always returns a non-negative float."""
        ext = LookaheadEnergyExtractor()
        key = jax.random.PRNGKey(42)
        for _ in range(5):
            key, subkey = jax.random.split(key)
            logits = jax.random.normal(subkey, shape=(8, 64))
            value = ext._compute_lookahead_energy(logits)
            assert value >= 0.0, f"Got negative lookahead energy: {value}"

    def test_custom_threshold_propagated(self) -> None:
        """Custom threshold is passed through to the LookaheadEnergyConstraint."""
        threshold = 1.234
        ext = LookaheadEnergyExtractor(threshold=threshold)
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert result.metadata["threshold"] == threshold
        assert isinstance(result.energy_term, LookaheadEnergyConstraint)
        assert result.energy_term.satisfaction_threshold == threshold

    def test_description_contains_satisfied_or_violated(self) -> None:
        """description string mentions 'satisfied' or 'violated'."""
        ext = LookaheadEnergyExtractor()
        logits = _peaked_logits()
        result = ext.extract("text", logits=logits)[0]
        assert "satisfied" in result.description or "violated" in result.description


# ---------------------------------------------------------------------------
# AutoExtractor integration tests
# ---------------------------------------------------------------------------


class TestAutoExtractorLookaheadIntegration:
    """AutoExtractor: logits= adds both spilled_energy and lookahead_energy."""

    def test_extract_without_logits_no_lookahead(self) -> None:
        """AutoExtractor.extract(text) with no logits has no lookahead_energy."""
        ae = AutoExtractor()
        results = ae.extract("47 + 28 = 75")
        types = {r.constraint_type for r in results}
        assert "lookahead_energy" not in types

    def test_extract_with_logits_none_no_lookahead(self) -> None:
        """Explicit logits=None gives no lookahead_energy constraint."""
        ae = AutoExtractor()
        results = ae.extract("47 + 28 = 75", logits=None)
        types = {r.constraint_type for r in results}
        assert "lookahead_energy" not in types

    def test_extract_with_logits_adds_lookahead_energy(self) -> None:
        """AutoExtractor.extract(text, logits=...) adds a lookahead_energy result."""
        ae = AutoExtractor()
        logits = _peaked_logits(vocab_size=100, n_tokens=5)
        results = ae.extract("47 + 28 = 75", logits=logits)
        types = {r.constraint_type for r in results}
        assert "lookahead_energy" in types

    def test_extract_with_logits_adds_both_signals(self) -> None:
        """AutoExtractor with logits adds both spilled_energy and lookahead_energy."""
        ae = AutoExtractor()
        logits = _peaked_logits(vocab_size=100, n_tokens=5)
        results = ae.extract("text", logits=logits)
        types = {r.constraint_type for r in results}
        assert "spilled_energy" in types
        assert "lookahead_energy" in types

    def test_lookahead_energy_not_duplicated(self) -> None:
        """Calling extract once gives exactly one lookahead_energy result."""
        ae = AutoExtractor()
        logits = _peaked_logits()
        results = ae.extract("some text", logits=logits)
        lookahead = [r for r in results if r.constraint_type == "lookahead_energy"]
        assert len(lookahead) == 1

    def test_arithmetic_constraint_still_present(self) -> None:
        """Arithmetic constraints are still extracted when logits are provided."""
        ae = AutoExtractor()
        logits = _peaked_logits()
        results = ae.extract("47 + 28 = 75", logits=logits)
        types = {r.constraint_type for r in results}
        assert "arithmetic" in types

    def test_lookahead_energy_result_has_correct_constraint_type(self) -> None:
        """The lookahead_energy result carries a LookaheadEnergyConstraint."""
        ae = AutoExtractor()
        logits = _peaked_logits()
        results = ae.extract("text", logits=logits)
        lookahead = next(r for r in results if r.constraint_type == "lookahead_energy")
        assert isinstance(lookahead.energy_term, LookaheadEnergyConstraint)
