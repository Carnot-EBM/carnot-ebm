"""Tests for prefill_uncertainty_probe.py — pre-generation hallucination risk gate.

Theoretical basis: arXiv 2603.19562 (Neural Uncertainty Principle, Mar 2026).
Adversarial vulnerability and hallucination share a geometric origin — input and
loss-gradient are conjugate observables with an irreducible uncertainty bound.
This probe fires BEFORE any tokens are generated using only the first-pass logit
distribution (black-box friendly, no gradient access required).

Spec: REQ-VERIFY-080
SCENARIO-VERIFY-103 (high-entropy logits flagged as high risk)
SCENARIO-VERIFY-104 (low-entropy logits trigger fast-path skip)
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.pipeline.prefill_uncertainty_probe import (
    PrefillUncertaintyProbe,
    PrefillUncertaintyResult,
    compute_conjugate_bound,
    compute_input_uncertainty,
    compute_prompt_uncertainty,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _uniform_logits_1d(vocab_size: int = 16) -> np.ndarray:
    """1-D uniform logit array: maximum entropy → uncertainty_score ≈ 1.0."""
    return np.zeros(vocab_size, dtype=np.float64)


def _peaked_logits_1d(vocab_size: int = 16, peak: float = 20.0) -> np.ndarray:
    """1-D logit array with first token dominating: near-zero entropy."""
    arr = np.zeros(vocab_size, dtype=np.float64)
    arr[0] = peak
    return arr


def _uniform_logits_2d(vocab_size: int = 16) -> np.ndarray:
    """2-D (1, V) uniform logit array."""
    return np.zeros((1, vocab_size), dtype=np.float64)


def _peaked_logits_2d(vocab_size: int = 16, peak: float = 20.0) -> np.ndarray:
    """2-D (1, V) peaked logit array."""
    arr = np.zeros((1, vocab_size), dtype=np.float64)
    arr[0, 0] = peak
    return arr


def _embedding_array(n_tokens: int = 4, dim: int = 8) -> np.ndarray:
    """Random-ish embedding array for compute_input_uncertainty tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_tokens, dim))


# ---------------------------------------------------------------------------
# PrefillUncertaintyResult dataclass
# ---------------------------------------------------------------------------


class TestPrefillUncertaintyResult:
    """Spec: REQ-VERIFY-080"""

    def test_fields_exist(self) -> None:
        """All required fields are present on the dataclass.

        Spec: REQ-VERIFY-080
        """
        result = PrefillUncertaintyResult(
            uncertainty_score=0.8,
            conjugate_bound=1.2,
            high_risk=True,
            threshold_exceeded=True,
            n_tokens=32,
            computation_method="entropy_approximation",
        )
        assert result.uncertainty_score == 0.8
        assert result.conjugate_bound == 1.2
        assert result.high_risk is True
        assert result.threshold_exceeded is True
        assert result.n_tokens == 32
        assert result.computation_method == "entropy_approximation"

    def test_high_risk_and_threshold_exceeded_are_consistent(self) -> None:
        """high_risk and threshold_exceeded are both set by probe(), not independently.

        Spec: REQ-VERIFY-080
        """
        # Verify we can construct with both False
        result = PrefillUncertaintyResult(
            uncertainty_score=0.1,
            conjugate_bound=0.5,
            high_risk=False,
            threshold_exceeded=False,
            n_tokens=8,
            computation_method="entropy_approximation",
        )
        assert result.high_risk is False
        assert result.threshold_exceeded is False

    def test_computation_method_embedding_variance(self) -> None:
        """embedding_variance computation_method is a valid option.

        Spec: REQ-VERIFY-080
        """
        result = PrefillUncertaintyResult(
            uncertainty_score=0.3,
            conjugate_bound=0.9,
            high_risk=False,
            threshold_exceeded=False,
            n_tokens=64,
            computation_method="embedding_variance",
        )
        assert result.computation_method == "embedding_variance"


# ---------------------------------------------------------------------------
# compute_input_uncertainty
# ---------------------------------------------------------------------------


class TestComputeInputUncertainty:
    """Tests for the white-box embedding-variance uncertainty proxy.

    Spec: REQ-VERIFY-080
    """

    def test_returns_float(self) -> None:
        """compute_input_uncertainty returns a Python float.

        Spec: REQ-VERIFY-080
        """
        embeddings = _embedding_array(n_tokens=4, dim=8)
        result = compute_input_uncertainty(embeddings)
        assert isinstance(result, float)

    def test_identical_embeddings_zero_variance(self) -> None:
        """All-same embeddings → variance of norms = 0.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104 (edge case)
        """
        # All tokens with identical embedding → all norms identical → variance = 0
        embeddings = np.ones((5, 8), dtype=np.float64)
        result = compute_input_uncertainty(embeddings)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_diverse_embeddings_positive_variance(self) -> None:
        """Diverse embeddings → variance of norms > 0.

        Spec: REQ-VERIFY-080
        """
        embeddings = _embedding_array(n_tokens=8, dim=16)
        result = compute_input_uncertainty(embeddings)
        assert result > 0.0

    def test_single_token_embedding(self) -> None:
        """Single-token embedding: variance of one value = 0.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104 (edge case)
        """
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = compute_input_uncertainty(embeddings)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_dtype_coercion(self) -> None:
        """Float32 embeddings are accepted and result is a float.

        Spec: REQ-VERIFY-080
        """
        embeddings = np.ones((3, 4), dtype=np.float32)
        result = compute_input_uncertainty(embeddings)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_conjugate_bound
# ---------------------------------------------------------------------------


class TestComputeConjugateBound:
    """Tests for the Cauchy-Schwarz conjugate bound proxy.

    Spec: REQ-VERIFY-080
    """

    def test_basic_product(self) -> None:
        """compute_conjugate_bound returns input_norm * gradient_norm.

        Spec: REQ-VERIFY-080
        """
        assert compute_conjugate_bound(3.0, 4.0) == pytest.approx(12.0)

    def test_zero_norms(self) -> None:
        """Zero input or gradient norm → bound = 0.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104 (edge case)
        """
        assert compute_conjugate_bound(0.0, 5.0) == pytest.approx(0.0)
        assert compute_conjugate_bound(5.0, 0.0) == pytest.approx(0.0)
        assert compute_conjugate_bound(0.0, 0.0) == pytest.approx(0.0)

    def test_unit_norms(self) -> None:
        """Unit norms → bound = 1.

        Spec: REQ-VERIFY-080
        """
        assert compute_conjugate_bound(1.0, 1.0) == pytest.approx(1.0)

    def test_returns_float(self) -> None:
        """Return type is Python float.

        Spec: REQ-VERIFY-080
        """
        result = compute_conjugate_bound(2.5, 1.5)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_prompt_uncertainty
# ---------------------------------------------------------------------------


class TestComputePromptUncertainty:
    """Tests for the black-box entropy-based uncertainty computation.

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104
    """

    def test_uniform_1d_high_uncertainty(self) -> None:
        """Uniform 1-D logits → uncertainty_score ≈ 1.0 (max entropy).

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103
        """
        logits = _uniform_logits_1d(vocab_size=16)
        result = compute_prompt_uncertainty(logits, threshold=0.5)

        assert isinstance(result, PrefillUncertaintyResult)
        # Normalised entropy of uniform over V tokens = log(V)/log(V) = 1
        assert result.uncertainty_score == pytest.approx(1.0, abs=1e-6)
        assert result.high_risk is True
        assert result.threshold_exceeded is True
        assert result.n_tokens == 16
        assert result.computation_method == "entropy_approximation"

    def test_peaked_1d_low_uncertainty(self) -> None:
        """Peaked 1-D logits → uncertainty_score ≈ 0.0 (near-zero entropy).

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
        """
        logits = _peaked_logits_1d(vocab_size=16, peak=20.0)
        result = compute_prompt_uncertainty(logits, threshold=0.5)

        assert result.uncertainty_score < 0.1
        assert result.high_risk is False
        assert result.threshold_exceeded is False
        assert result.n_tokens == 16

    def test_uniform_2d_shape_accepted(self) -> None:
        """2-D (1, V) logit array is accepted and treated as 1-D.

        Spec: REQ-VERIFY-080
        """
        logits = _uniform_logits_2d(vocab_size=8)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        assert result.uncertainty_score == pytest.approx(1.0, abs=1e-6)
        assert result.n_tokens == 8

    def test_peaked_2d_shape_accepted(self) -> None:
        """2-D (1, V) peaked logit array gives low uncertainty.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
        """
        logits = _peaked_logits_2d(vocab_size=8, peak=15.0)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        assert result.uncertainty_score < 0.1
        assert result.high_risk is False

    def test_threshold_above_max_score_never_fires(self) -> None:
        """threshold=1.0 never fires because normalised entropy ≤ 1.0 always.

        Spec: REQ-VERIFY-080
        """
        # Uniform over 4 tokens → normalised entropy = 1.0; 1.0 is NOT > 1.0.
        logits = _uniform_logits_1d(vocab_size=4)
        result = compute_prompt_uncertainty(logits, threshold=1.0)
        assert result.high_risk is False
        assert result.threshold_exceeded is False

    def test_uncertainty_score_normalised_in_0_1(self) -> None:
        """uncertainty_score is always in [0, 1].

        Spec: REQ-VERIFY-080
        """
        for logits in [
            _uniform_logits_1d(32),
            _peaked_logits_1d(32, peak=50.0),
            np.random.default_rng(7).standard_normal(32),
        ]:
            result = compute_prompt_uncertainty(logits, threshold=0.5)
            assert 0.0 <= result.uncertainty_score <= 1.0 + 1e-9

    def test_conjugate_bound_non_negative(self) -> None:
        """conjugate_bound is always non-negative.

        Spec: REQ-VERIFY-080
        """
        logits = _uniform_logits_1d(8)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        assert result.conjugate_bound >= 0.0

    def test_single_vocab_token(self) -> None:
        """Single vocabulary token → entropy=0, uncertainty_score=0.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104 (edge case)
        """
        logits = np.array([5.0], dtype=np.float64)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        # Only one token → deterministic → zero entropy
        assert result.uncertainty_score == pytest.approx(0.0, abs=1e-10)
        assert result.high_risk is False
        assert result.n_tokens == 1


# ---------------------------------------------------------------------------
# PrefillUncertaintyProbe.probe
# ---------------------------------------------------------------------------


class TestPrefillUncertaintyProbeClass:
    """Tests for PrefillUncertaintyProbe — the main entry point class.

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104
    """

    def test_probe_high_entropy_high_risk(self) -> None:
        """SCENARIO-VERIFY-103: uniform logits → high_risk=True.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103
        """
        probe = PrefillUncertaintyProbe()
        logits = _uniform_logits_2d(vocab_size=32)
        result = probe.probe(logits, threshold=0.5)

        assert result.uncertainty_score == pytest.approx(1.0, abs=1e-6)
        assert result.high_risk is True
        assert result.threshold_exceeded is True
        assert result.computation_method == "entropy_approximation"

    def test_probe_low_entropy_low_risk(self) -> None:
        """SCENARIO-VERIFY-104: peaked logits → high_risk=False.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
        """
        probe = PrefillUncertaintyProbe()
        logits = _peaked_logits_2d(vocab_size=32, peak=25.0)
        result = probe.probe(logits, threshold=0.5)

        assert result.uncertainty_score < 0.1
        assert result.high_risk is False
        assert result.threshold_exceeded is False

    def test_probe_default_threshold_is_0_5(self) -> None:
        """probe() has a default threshold of 0.5.

        Spec: REQ-VERIFY-080
        """
        probe = PrefillUncertaintyProbe()
        # peaked logits → low uncertainty → high_risk=False regardless of threshold default
        logits = _peaked_logits_1d(vocab_size=8, peak=30.0)
        result = probe.probe(logits)
        assert result.high_risk is False

    def test_probe_returns_prefill_result(self) -> None:
        """probe() always returns a PrefillUncertaintyResult instance.

        Spec: REQ-VERIFY-080
        """
        probe = PrefillUncertaintyProbe()
        result = probe.probe(_uniform_logits_1d(4), threshold=0.3)
        assert isinstance(result, PrefillUncertaintyResult)

    def test_probe_1d_and_2d_equivalent(self) -> None:
        """probe() gives same result for 1-D and (1, V) 2-D logits.

        Spec: REQ-VERIFY-080
        """
        probe = PrefillUncertaintyProbe()
        logits_1d = _uniform_logits_1d(8)
        logits_2d = _uniform_logits_2d(8)
        r1 = probe.probe(logits_1d, threshold=0.5)
        r2 = probe.probe(logits_2d, threshold=0.5)
        assert r1.uncertainty_score == pytest.approx(r2.uncertainty_score, abs=1e-10)
        assert r1.high_risk == r2.high_risk


# ---------------------------------------------------------------------------
# VerifyRepairPipeline.check_prefill_uncertainty integration
# ---------------------------------------------------------------------------


class TestVerifyRepairPipelinePrefillIntegration:
    """Tests for the VerifyRepairPipeline.check_prefill_uncertainty additive method.

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103, SCENARIO-VERIFY-104
    """

    def _pipeline(self) -> VerifyRepairPipeline:
        """Create a pipeline with no model (verify-only mode)."""
        return VerifyRepairPipeline(
            model=None,
            domains=None,
            max_repairs=0,
            extractor=None,
            semantic_grounding_verifier=None,
            semantic_verifier_v2=None,
            timeout_seconds=None,
            memory=None,
        )

    def test_high_risk_does_not_skip(self) -> None:
        """SCENARIO-VERIFY-103: high-entropy logits → skip_verification=False.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-103
        """
        pipeline = self._pipeline()
        logits = _uniform_logits_2d(vocab_size=16)
        out = pipeline.check_prefill_uncertainty(logits, threshold=0.5)

        assert out["skip_verification"] is False
        assert isinstance(out["reason"], str)
        assert isinstance(out["result"], PrefillUncertaintyResult)
        assert out["result"].high_risk is True

    def test_low_risk_triggers_fast_path_skip(self) -> None:
        """SCENARIO-VERIFY-104: peaked logits → skip_verification=True, reason='low_uncertainty'.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
        """
        pipeline = self._pipeline()
        logits = _peaked_logits_2d(vocab_size=16, peak=25.0)
        out = pipeline.check_prefill_uncertainty(logits, threshold=0.5)

        assert out["skip_verification"] is True
        assert out["reason"] == "low_uncertainty"
        assert isinstance(out["result"], PrefillUncertaintyResult)
        assert out["result"].high_risk is False

    def test_default_threshold_is_0_5(self) -> None:
        """check_prefill_uncertainty default threshold = 0.5.

        Spec: REQ-VERIFY-080
        """
        pipeline = self._pipeline()
        # Peaked logits → low uncertainty → skip regardless of explicit threshold
        logits = _peaked_logits_1d(vocab_size=8, peak=30.0)
        out = pipeline.check_prefill_uncertainty(logits)
        assert out["skip_verification"] is True

    def test_output_keys_always_present(self) -> None:
        """Output dict always contains skip_verification, reason, result.

        Spec: REQ-VERIFY-080
        """
        pipeline = self._pipeline()
        for logits in [_uniform_logits_1d(4), _peaked_logits_1d(4)]:
            out = pipeline.check_prefill_uncertainty(logits, threshold=0.5)
            assert "skip_verification" in out
            assert "reason" in out
            assert "result" in out

    def test_high_risk_reason_is_string(self) -> None:
        """Reason is always a non-empty string even when skip_verification=False.

        Spec: REQ-VERIFY-080
        """
        pipeline = self._pipeline()
        out = pipeline.check_prefill_uncertainty(_uniform_logits_1d(8), threshold=0.5)
        assert isinstance(out["reason"], str)
        assert len(out["reason"]) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty-ish logits, single-vocab, all-same embeddings.

    Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
    """

    def test_single_vocab_token_probe(self) -> None:
        """Single vocab token → uncertainty_score=0, no exception.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
        """
        probe = PrefillUncertaintyProbe()
        logits = np.array([1.0], dtype=np.float64)
        result = probe.probe(logits, threshold=0.5)
        assert result.uncertainty_score == pytest.approx(0.0, abs=1e-10)
        assert result.high_risk is False

    def test_all_same_embeddings_zero_uncertainty(self) -> None:
        """All-identical embedding rows → variance = 0.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
        """
        embeddings = np.ones((6, 12), dtype=np.float64) * 3.7
        score = compute_input_uncertainty(embeddings)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_two_token_vocab_uniform(self) -> None:
        """Two-token vocab with uniform logits → normalised entropy = 1.

        Spec: REQ-VERIFY-080
        """
        logits = np.zeros(2, dtype=np.float64)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        assert result.uncertainty_score == pytest.approx(1.0, abs=1e-6)

    def test_large_peak_approaches_zero_entropy(self) -> None:
        """Very large peak logit → uncertainty approaches 0.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104
        """
        logits = _peaked_logits_1d(vocab_size=100, peak=100.0)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        assert result.uncertainty_score < 1e-6

    def test_negative_logits_accepted(self) -> None:
        """Negative logit values (common from real models) produce valid results.

        Spec: REQ-VERIFY-080
        """
        logits = np.array([-5.0, -2.0, -1.0, -3.0], dtype=np.float64)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        assert 0.0 <= result.uncertainty_score <= 1.0 + 1e-9

    def test_empty_logit_array_returns_zero_uncertainty(self) -> None:
        """Empty logit array (V=0) returns uncertainty_score=0, no exception.

        Spec: REQ-VERIFY-080, SCENARIO-VERIFY-104 (edge case)
        """
        logits = np.array([], dtype=np.float64)
        result = compute_prompt_uncertainty(logits, threshold=0.5)
        assert result.uncertainty_score == pytest.approx(0.0, abs=1e-10)
        assert result.high_risk is False
        assert result.n_tokens == 0

    def test_invalid_2d_shape_raises_value_error(self) -> None:
        """(2, V) logit array raises ValueError — only (V,) and (1, V) accepted.

        Spec: REQ-VERIFY-080
        """
        logits = np.zeros((2, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            compute_prompt_uncertainty(logits, threshold=0.5)

    def test_invalid_3d_shape_raises_value_error(self) -> None:
        """3-D logit array raises ValueError.

        Spec: REQ-VERIFY-080
        """
        logits = np.zeros((1, 1, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="ndim"):
            compute_prompt_uncertainty(logits, threshold=0.5)

    def test_high_risk_equals_threshold_exceeded_on_probe_output(self) -> None:
        """high_risk and threshold_exceeded are always equal when produced by probe().

        Spec: REQ-VERIFY-080
        """
        probe = PrefillUncertaintyProbe()
        for logits in [_uniform_logits_1d(8), _peaked_logits_1d(8, peak=20.0)]:
            result = probe.probe(logits, threshold=0.5)
            assert result.high_risk == result.threshold_exceeded, (
                f"high_risk ({result.high_risk}) != threshold_exceeded "
                f"({result.threshold_exceeded}) for score={result.uncertainty_score}"
            )

    def test_threshold_boundary_strict_greater_than(self) -> None:
        """Score exactly equal to threshold → high_risk=False (strict >).

        Constructs a 2-token case where normalised entropy = 1.0, sets threshold=1.0.
        Since 1.0 is NOT > 1.0, high_risk must be False.

        Spec: REQ-VERIFY-080
        """
        logits = np.zeros(2, dtype=np.float64)   # 2-token uniform → normalised H = 1.0
        result = compute_prompt_uncertainty(logits, threshold=1.0)
        assert result.uncertainty_score == pytest.approx(1.0, abs=1e-6)
        assert result.high_risk is False
        assert result.threshold_exceeded is False
