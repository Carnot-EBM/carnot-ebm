"""Tests for JEPAFastPathPredictor and its VerifyRepairPipeline integration.

Covers:
  - Feature extraction produces expected keys and bounded values.
  - Short simple responses score low (fast-path eligible).
  - Longer/complex responses score higher (above threshold).
  - Pipeline with predictor returns JEPA_FAST_PATH result for simple responses.
  - Pipeline without predictor runs normal verification.
  - calls_fast_path counter is incremented correctly.

Spec: REQ-VERIFY-003, REQ-JEPA-002
"""

from __future__ import annotations

import pytest

from carnot.pipeline.jepa_fast_path import (
    JEPAFastPathPredictor,
    extract_response_features,
)
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractResponseFeatures:
    """REQ-JEPA-002: feature extraction returns bounded, correct-direction values."""

    def test_returns_expected_keys(self) -> None:
        """Feature dict must contain response_length_norm and logprob_variance_proxy."""
        features = extract_response_features("The answer is 75.")
        assert "response_length_norm" in features
        assert "logprob_variance_proxy" in features

    def test_empty_response_bounded(self) -> None:
        """Empty string produces features in [0, 1], no errors."""
        features = extract_response_features("")
        assert 0.0 <= features["response_length_norm"] <= 1.0
        assert 0.0 <= features["logprob_variance_proxy"] <= 1.0

    def test_short_simple_response_low_length(self) -> None:
        """Single-word numeric answer produces length_norm near zero."""
        features = extract_response_features("75")
        assert features["response_length_norm"] < 0.1

    def test_long_response_higher_length(self) -> None:
        """200-token response saturates length_norm at 1.0."""
        long_response = " ".join(["word"] * 250)
        features = extract_response_features(long_response)
        assert features["response_length_norm"] == pytest.approx(1.0)

    def test_features_bounded(self) -> None:
        """Both features are always in [0.0, 1.0] for arbitrary text."""
        for text in ["", "a", "42", "The quick brown fox.", "x" * 500]:
            f = extract_response_features(text)
            assert 0.0 <= f["response_length_norm"] <= 1.0, f"out of range for: {text!r}"
            assert 0.0 <= f["logprob_variance_proxy"] <= 1.0, f"out of range for: {text!r}"


# ---------------------------------------------------------------------------
# JEPAFastPathPredictor tests
# ---------------------------------------------------------------------------


class TestJEPAFastPathPredictor:
    """REQ-JEPA-002: predictor returns bounded probabilities; fast-path logic fires correctly."""

    def setup_method(self) -> None:
        self.predictor = JEPAFastPathPredictor()

    def test_short_answer_below_threshold(self) -> None:
        """Short numeric answers should have p_violation < 0.2 (fast-path eligible)."""
        p = self.predictor.predict_p_violation("75")
        assert p < 0.2

    def test_calls_total_incremented(self) -> None:
        """calls_total is incremented on each predict_p_violation call."""
        assert self.predictor.calls_total == 0
        self.predictor.predict_p_violation("hello")
        assert self.predictor.calls_total == 1
        self.predictor.predict_p_violation("world")
        assert self.predictor.calls_total == 2

    def test_probability_bounded(self) -> None:
        """predict_p_violation always returns a value in [0.0, 1.0]."""
        for text in ["", "42", "The quick brown fox jumps over the lazy dog." * 20]:
            p = self.predictor.predict_p_violation(text)
            assert 0.0 <= p <= 1.0, f"Out of bounds for: {text[:40]!r}"

    def test_fast_path_rate_nan_when_no_calls(self) -> None:
        """fast_path_rate is NaN before any calls (no ZeroDivisionError)."""
        import math
        assert math.isnan(self.predictor.fast_path_rate)

    def test_fast_path_rate_computed(self) -> None:
        """fast_path_rate equals calls_fast_path / calls_total."""
        self.predictor.predict_p_violation("75")  # total=1
        self.predictor.calls_fast_path = 1         # simulate pipeline incrementing
        assert self.predictor.fast_path_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VerifyRepairPipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineJEPAFastPath:
    """REQ-JEPA-002: pipeline-level fast-path fires correctly and is logged."""

    def test_fast_path_fires_for_simple_response(self) -> None:
        """Pipeline with predictor returns JEPA_FAST_PATH for a short numeric answer."""
        predictor = JEPAFastPathPredictor()
        pipeline = VerifyRepairPipeline(
            jepa_fast_path_predictor=predictor,
            jepa_fast_path_threshold=0.2,
        )
        result = pipeline.verify(
            question="What is 5 + 2?",
            response="7",
        )
        assert isinstance(result, VerificationResult)
        assert result.verified is True
        assert result.skipped is True
        assert result.mode == "JEPA_FAST_PATH"
        assert result.certificate.get("fast_path_used") is True

    def test_fast_path_increments_counter(self) -> None:
        """calls_fast_path is incremented when the fast-path fires."""
        predictor = JEPAFastPathPredictor()
        pipeline = VerifyRepairPipeline(
            jepa_fast_path_predictor=predictor,
            jepa_fast_path_threshold=0.2,
        )
        assert predictor.calls_fast_path == 0
        pipeline.verify(question="What is 1+1?", response="2")
        assert predictor.calls_fast_path == 1

    def test_no_predictor_runs_full_verification(self) -> None:
        """Pipeline without jepa_fast_path_predictor runs normal verification."""
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify(
            question="What is 2 + 2?",
            response="The answer is 4.",
            domain="arithmetic",
        )
        assert isinstance(result, VerificationResult)
        # Normal path: mode is not JEPA_FAST_PATH
        assert result.mode != "JEPA_FAST_PATH"

    def test_threshold_respected(self) -> None:
        """Setting threshold=0.0 means the fast-path never fires."""
        predictor = JEPAFastPathPredictor()
        pipeline = VerifyRepairPipeline(
            jepa_fast_path_predictor=predictor,
            jepa_fast_path_threshold=0.0,  # nothing can be < 0.0
        )
        result = pipeline.verify(question="What is 1+1?", response="2")
        # With threshold=0.0 the fast path cannot fire since p_violation >= 0.0
        assert result.mode != "JEPA_FAST_PATH"
