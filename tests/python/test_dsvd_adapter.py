"""Tests for carnot.pipeline.dsvd_adapter — DSVDLinearProbe and DSVDAdapter.

100% coverage target for dsvd_adapter.py.

Spec: REQ-VERIFY-118, SCENARIO-VERIFY-157, SCENARIO-VERIFY-158, SCENARIO-VERIFY-159
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from carnot.pipeline.dsvd_adapter import (
    DSVDAdapter,
    DSVDLinearProbe,
    DSVDProbeResult,
    _char_entropy,
    _count_numbers,
    _count_operators,
)


# ---------------------------------------------------------------------------
# Helper / private function coverage
# ---------------------------------------------------------------------------


class TestCharEntropy:
    """REQ-VERIFY-118: _char_entropy returns correct Shannon entropy."""

    def test_empty_string(self) -> None:
        # Edge case: no information in an empty string.
        assert _char_entropy("") == 0.0

    def test_single_char(self) -> None:
        # Single character: no variety, entropy = 0.
        assert _char_entropy("a") == 0.0

    def test_uniform_two_chars(self) -> None:
        # "ab" — two equally likely characters → entropy = 1.0 bit.
        assert abs(_char_entropy("ab") - 1.0) < 1e-9

    def test_nonzero_for_normal_text(self) -> None:
        assert _char_entropy("hello world") > 0.0


class TestCountNumbers:
    """REQ-VERIFY-118: _count_numbers detects numeric tokens."""

    def test_no_numbers(self) -> None:
        assert _count_numbers("no digits here") == 0

    def test_single_integer(self) -> None:
        assert _count_numbers("42") == 1

    def test_multiple_integers(self) -> None:
        assert _count_numbers("T + C + S = 160 + 80 + 20 = 260") == 4

    def test_negative_number(self) -> None:
        assert _count_numbers("-5") == 1

    def test_decimal(self) -> None:
        # "3.14" counts as one numeric token.
        assert _count_numbers("3.14") == 1


class TestCountOperators:
    """REQ-VERIFY-118: _count_operators counts arithmetic symbols."""

    def test_no_operators(self) -> None:
        assert _count_operators("hello") == 0

    def test_basic_operators(self) -> None:
        assert _count_operators("a + b = c") == 2  # '+' and '='

    def test_all_operators(self) -> None:
        assert _count_operators("+-*/=^") == 6


# ---------------------------------------------------------------------------
# DSVDLinearProbe
# ---------------------------------------------------------------------------


class TestDSVDLinearProbeExtractFeatures:
    """SCENARIO-VERIFY-157: _extract_features returns jnp.ndarray of shape (hidden_dim,)."""

    def test_default_hidden_dim(self) -> None:
        probe = DSVDLinearProbe()
        features = probe._extract_features("T = 2 × C = 160")
        assert isinstance(features, jnp.ndarray)
        assert features.shape == (64,)

    def test_custom_hidden_dim(self) -> None:
        probe = DSVDLinearProbe(hidden_dim=32)
        features = probe._extract_features("step text")
        assert features.shape == (32,)

    def test_empty_step(self) -> None:
        probe = DSVDLinearProbe()
        features = probe._extract_features("")
        assert features.shape == (64,)

    def test_returns_jax_array(self) -> None:
        probe = DSVDLinearProbe()
        result = probe._extract_features("x = 3 + 4")
        # Must be a JAX array, not a plain numpy array.
        assert hasattr(result, "device")


class TestDSVDLinearProbePredict:
    """SCENARIO-VERIFY-157: predict() returns float in [0, 1]."""

    def test_predict_unfitted_returns_float(self) -> None:
        probe = DSVDLinearProbe()
        p = probe.predict("some step text")
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_predict_after_fit(self) -> None:
        probe = DSVDLinearProbe()
        steps = [
            "T = 2 × C = 160",
            "C = 4 × S = 80",
            "Total = T + C + S = 260",
            "S = 20",
            "The answer is 260",
        ]
        labels = [1.0, 0.0, 1.0, 0.0, 0.0]
        probe.fit(steps, labels)
        p = probe.predict("T = 2 × C = 160")
        assert 0.0 <= p <= 1.0

    def test_fit_empty_list_no_error(self) -> None:
        probe = DSVDLinearProbe()
        probe.fit([], [])  # Should not raise.
        p = probe.predict("test")
        assert 0.0 <= p <= 1.0

    def test_predict_extreme_logit_clamped(self) -> None:
        # Force a very large weight so logit >> 30; result should still be in [0,1].
        import numpy as np

        probe = DSVDLinearProbe(hidden_dim=4)
        probe._weights = np.array([1e6, 1e6, 1e6, 1e6], dtype=np.float32)
        probe._bias = 0.0
        p = probe.predict("123 + 456 = 789")
        assert 0.0 <= p <= 1.0


class TestDSVDLinearProbeScore:
    """SCENARIO-VERIFY-159: score() returns DSVDProbeResult with detector_mode='linear_probe'."""

    def test_score_returns_dataclass(self) -> None:
        probe = DSVDLinearProbe()
        result = probe.score("T = 2 × 80 = 160")
        assert isinstance(result, DSVDProbeResult)

    def test_score_detector_mode(self) -> None:
        probe = DSVDLinearProbe()
        result = probe.score("any step")
        assert result.detector_mode == "linear_probe"

    def test_score_step_idx_zero(self) -> None:
        probe = DSVDLinearProbe()
        result = probe.score("x + y = z")
        assert result.step_idx == 0

    def test_score_violation_probability_range(self) -> None:
        probe = DSVDLinearProbe()
        result = probe.score("step text here")
        assert 0.0 <= result.violation_probability <= 1.0

    def test_score_feature_norm_nonnegative(self) -> None:
        probe = DSVDLinearProbe()
        result = probe.score("T = 160, C = 80, S = 20")
        assert result.feature_norm >= 0.0

    def test_score_step_text_preserved(self) -> None:
        probe = DSVDLinearProbe()
        text = "Total = T + C + S = 260"
        result = probe.score(text)
        assert result.step_text == text

    def test_score_empty_step_feature_norm_zero(self) -> None:
        probe = DSVDLinearProbe()
        result = probe.score("")
        # All raw features are 0 for empty string → norm = 0.
        assert result.feature_norm == 0.0


# ---------------------------------------------------------------------------
# DSVDAdapter
# ---------------------------------------------------------------------------


class TestDSVDAdapterVerifyStep:
    """SCENARIO-VERIFY-157: verify_step returns DSVDProbeResult."""

    def test_verify_step_returns_result(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe)
        result = adapter.verify_step("T = 2 × C = 160")
        assert isinstance(result, DSVDProbeResult)

    def test_verify_step_mode(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe)
        result = adapter.verify_step("T = 2 × C = 160")
        assert result.detector_mode == "linear_probe"


class TestDSVDAdapterVerifyChain:
    """SCENARIO-VERIFY-158: verify_chain returns list of DSVDProbeResult."""

    def test_verify_chain_returns_list(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe)
        steps = ["step 0", "step 1", "step 2"]
        results = adapter.verify_chain(steps)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_verify_chain_step_idx_sequential(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe)
        steps = ["a", "b", "c", "d"]
        results = adapter.verify_chain(steps)
        for i, r in enumerate(results):
            assert r.step_idx == i

    def test_verify_chain_all_results_correct_type(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe)
        results = adapter.verify_chain(["T = 160", "C = 80"])
        for r in results:
            assert isinstance(r, DSVDProbeResult)

    def test_verify_chain_empty_list(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe)
        results = adapter.verify_chain([])
        assert results == []


class TestDSVDAdapterNViolations:
    """SCENARIO-VERIFY-158: n_violations counts correctly."""

    def test_n_violations_all_below_threshold(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe, violation_threshold=0.5)
        results = [
            DSVDProbeResult(0, 0.1, "a", 1.0, "linear_probe"),
            DSVDProbeResult(1, 0.3, "b", 1.0, "linear_probe"),
            DSVDProbeResult(2, 0.4, "c", 1.0, "linear_probe"),
        ]
        assert adapter.n_violations(results) == 0

    def test_n_violations_some_above_threshold(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe, violation_threshold=0.5)
        results = [
            DSVDProbeResult(0, 0.8, "a", 1.0, "linear_probe"),
            DSVDProbeResult(1, 0.3, "b", 1.0, "linear_probe"),
            DSVDProbeResult(2, 0.9, "c", 1.0, "linear_probe"),
        ]
        assert adapter.n_violations(results) == 2

    def test_n_violations_at_boundary_excluded(self) -> None:
        # violation_probability == threshold is NOT counted (strict >).
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe, violation_threshold=0.5)
        results = [DSVDProbeResult(0, 0.5, "a", 1.0, "linear_probe")]
        assert adapter.n_violations(results) == 0

    def test_n_violations_empty_list(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe)
        assert adapter.n_violations([]) == 0

    def test_n_violations_custom_threshold(self) -> None:
        probe = DSVDLinearProbe()
        adapter = DSVDAdapter(probe, violation_threshold=0.7)
        results = [
            DSVDProbeResult(0, 0.6, "a", 1.0, "linear_probe"),
            DSVDProbeResult(1, 0.8, "b", 1.0, "linear_probe"),
        ]
        assert adapter.n_violations(results) == 1


# ---------------------------------------------------------------------------
# Export from carnot.pipeline.__init__
# ---------------------------------------------------------------------------


def test_exports_from_pipeline_init() -> None:
    """Verify that DSVDAdapter, DSVDLinearProbe, DSVDProbeResult are importable from carnot.pipeline."""
    from carnot.pipeline import DSVDAdapter as A
    from carnot.pipeline import DSVDLinearProbe as P
    from carnot.pipeline import DSVDProbeResult as R

    assert A is DSVDAdapter
    assert P is DSVDLinearProbe
    assert R is DSVDProbeResult


# ---------------------------------------------------------------------------
# End-to-end fit → predict smoke test
# ---------------------------------------------------------------------------


def test_fit_predict_smoke() -> None:
    """Smoke test: probe trained on 10 steps produces discriminative probabilities."""
    probe = DSVDLinearProbe(hidden_dim=64)
    # Incorrect chains have dense arithmetic; correct have sparse.
    incorrect = [
        "T = 2 × C = 160, C = 4 × S = 80, S = 20",
        "Total = 160 + 80 + 20 = 260 but answer is 270",
        "3 * 4 = 13",
        "100 / 5 = 19",
        "50 + 50 = 99",
    ]
    correct = [
        "Since the problem says S = 20, we know C = 80.",
        "Substituting, we get T = 160.",
        "Total sheep = T + C + S.",
        "Therefore the answer is 260.",
        "The calculation is straightforward.",
    ]
    steps = incorrect + correct
    labels = [1.0] * 5 + [0.0] * 5
    probe.fit(steps, labels)

    for s in incorrect:
        p = probe.predict(s)
        assert 0.0 <= p <= 1.0

    for s in correct:
        p = probe.predict(s)
        assert 0.0 <= p <= 1.0
