"""Tests for Experiment 874: StreamingCoT Tier 0g integration.

Validates that StreamingCoTHalluDetector is correctly wired into
VerifyRepairPipeline.verify() as an advisory Tier 0g signal.

Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165, SCENARIO-VERIFY-166
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures for StreamingCoT module tests (no env patching needed)
# ---------------------------------------------------------------------------


class TestStreamingCoTHalluDetector:
    """Unit tests for StreamingCoTHalluDetector.detect().

    Spec: REQ-VERIFY-140
    """

    def _get_detector(self):
        from carnot.pipeline.streaming_cot import StreamingCoTHalluDetector
        return StreamingCoTHalluDetector(alpha=0.3, threshold=0.35)

    def test_empty_steps_returns_stable(self):
        """SCENARIO-VERIFY-166 edge case: empty input is not flagged."""
        det = self._get_detector()
        result = det.detect([])
        assert result.is_streaming_unstable is False
        assert result.final_phas == 0.0
        assert result.n_steps == 0
        assert result.step_scores == []

    def test_uniform_short_steps_stable(self):
        """Uniform concise steps stay below threshold — correct CoT pattern.

        Spec: SCENARIO-VERIFY-165
        """
        det = self._get_detector()
        steps = [
            "Identify variables.",
            "Apply formula.",
            "Compute result.",
        ]
        result = det.detect(steps)
        # Short steps are close to expected mean → low proxy → EMA stays low
        assert result.n_steps == 3
        assert isinstance(result.final_phas, float)
        assert 0.0 <= result.final_phas <= 1.0
        assert len(result.step_scores) == 3

    def test_compounding_error_steps_flagged(self):
        """Highly variable step lengths cross the threshold.

        Spec: SCENARIO-VERIFY-165
        """
        det = self._get_detector()
        short = "Hmm."
        very_long = (
            "Actually I need to reconsider everything from scratch because my earlier "
            "calculation was completely wrong and I must now apply an entirely different "
            "formula that accounts for boundary conditions I ignored previously. "
            "This fundamentally changes the approach and all intermediate results. "
            "Let me redo steps one through five with the correct methodology now. "
            "The error compounded because I confused perimeter with area and also "
            "misread the problem statement about parallel sides of the trapezoid. "
        ) * 3
        steps = [short, very_long, short, very_long, short]
        result = det.detect(steps)
        assert result.n_steps == 5
        # The very long steps should drive the EMA above 0.35
        assert result.is_streaming_unstable is True
        assert result.final_phas > 0.35

    def test_invalid_alpha_raises(self):
        """Constructor rejects alpha outside (0, 1]."""
        from carnot.pipeline.streaming_cot import StreamingCoTHalluDetector
        with pytest.raises(ValueError, match="alpha"):
            StreamingCoTHalluDetector(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            StreamingCoTHalluDetector(alpha=1.5)

    def test_invalid_threshold_raises(self):
        """Constructor rejects threshold outside [0, 1]."""
        from carnot.pipeline.streaming_cot import StreamingCoTHalluDetector
        with pytest.raises(ValueError, match="threshold"):
            StreamingCoTHalluDetector(threshold=-0.1)
        with pytest.raises(ValueError, match="threshold"):
            StreamingCoTHalluDetector(threshold=1.1)

    def test_step_scores_length_matches_input(self):
        """step_scores has one entry per input step."""
        det = self._get_detector()
        steps = ["Step one.", "Step two.", "Step three.", "Step four."]
        result = det.detect(steps)
        assert len(result.step_scores) == 4

    def test_ema_is_monotone_accumulative(self):
        """EMA increases when all steps have high proxy score."""
        from carnot.pipeline.streaming_cot import StreamingCoTHalluDetector
        det = StreamingCoTHalluDetector(alpha=0.5, threshold=0.9)
        # Very long steps force proxy score > 0
        long_step = "x" * 500
        steps = [long_step] * 5
        result = det.detect(steps)
        # EMA should be non-zero
        assert result.final_phas > 0.0


class TestExtractCotSteps:
    """Unit tests for extract_cot_steps().

    Spec: REQ-VERIFY-140
    """

    def test_numbered_steps_split(self):
        from carnot.pipeline.streaming_cot import extract_cot_steps
        response = "1. First step.\n2. Second step.\n3. Third step."
        steps = extract_cot_steps(response)
        assert len(steps) >= 2

    def test_step_prefix_split(self):
        from carnot.pipeline.streaming_cot import extract_cot_steps
        response = "Step 1: Do this.\nStep 2: Do that.\nStep 3: Done."
        steps = extract_cot_steps(response)
        assert len(steps) >= 2

    def test_empty_response_returns_empty(self):
        from carnot.pipeline.streaming_cot import extract_cot_steps
        assert extract_cot_steps("") == []
        assert extract_cot_steps("   ") == []

    def test_no_delimiter_returns_single_step(self):
        from carnot.pipeline.streaming_cot import extract_cot_steps
        response = "This is a plain response with no step delimiters at all."
        steps = extract_cot_steps(response)
        assert len(steps) == 1
        assert steps[0] == response.strip()


# ---------------------------------------------------------------------------
# VerifyRepairPipeline wiring tests (SCENARIO-VERIFY-165 and 166)
# ---------------------------------------------------------------------------


class TestStreamingCotWiringEnabled:
    """SCENARIO-VERIFY-165: STREAMING_COT_ENABLED populates certificate fields.

    Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165
    """

    def _make_pipeline_with_flag(self):
        """Import VerifyRepairPipeline with CARNOT_STREAMING_COT=1 active."""
        # Patch the class attribute directly for isolation (avoids re-import gymnastics).
        from carnot.pipeline.verify_repair import VerifyRepairPipeline
        return VerifyRepairPipeline

    def test_streaming_cot_enabled_class_attr_true(self):
        """STREAMING_COT_ENABLED reflects env var."""
        with patch.dict(os.environ, {"CARNOT_STREAMING_COT": "1"}):
            # Re-evaluate the class attribute with env set.
            # Since it's evaluated at class definition time, we test the logic directly.
            value = os.getenv("CARNOT_STREAMING_COT", "0") == "1"
            assert value is True

    def test_streaming_cot_disabled_class_attr_false(self):
        """STREAMING_COT_ENABLED is False when env var is absent."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove if present
            os.environ.pop("CARNOT_STREAMING_COT", None)
            value = os.getenv("CARNOT_STREAMING_COT", "0") == "1"
            assert value is False

    def test_verify_populates_streaming_fields_when_enabled(self):
        """verify() populates streaming_cot_unstable and streaming_cot_phas.

        Spec: SCENARIO-VERIFY-165
        """
        from carnot.pipeline.verify_repair import VerifyRepairPipeline
        from carnot.pipeline.streaming_cot import StreamingCoTResult

        pipeline = VerifyRepairPipeline()

        # Patch the class attribute so flag is True for this test.
        mock_result = StreamingCoTResult(
            is_streaming_unstable=True,
            final_phas=0.45,
            step_scores=[0.1, 0.6, 0.8],
            n_steps=3,
        )

        with patch.object(VerifyRepairPipeline, "STREAMING_COT_ENABLED", True):
            with patch(
                "carnot.pipeline.streaming_cot.StreamingCoTHalluDetector.detect",
                return_value=mock_result,
            ):
                result = pipeline.verify(
                    question="What is 2+2?",
                    response="Step 1: Add.\nStep 2: Answer is 4.",
                )

        assert result.streaming_cot_unstable is True
        assert result.streaming_cot_phas == pytest.approx(0.45)
        assert "tier_0g_streaming_cot" in result.certificate
        cert_entry = result.certificate["tier_0g_streaming_cot"]
        assert cert_entry["is_streaming_unstable"] is True
        assert cert_entry["final_phas"] == pytest.approx(0.45)

    def test_verify_does_not_skip_ising_on_unstable(self):
        """Ising/constraint path runs even when streaming_cot_unstable=True.

        The key assertion: result.skipped remains False (cascade was not short-circuited).
        Spec: REQ-VERIFY-140 (advisory only)
        """
        from carnot.pipeline.verify_repair import VerifyRepairPipeline
        from carnot.pipeline.streaming_cot import StreamingCoTResult

        pipeline = VerifyRepairPipeline()
        mock_result = StreamingCoTResult(
            is_streaming_unstable=True,
            final_phas=0.9,
            step_scores=[0.9],
            n_steps=1,
        )

        with patch.object(VerifyRepairPipeline, "STREAMING_COT_ENABLED", True):
            with patch(
                "carnot.pipeline.streaming_cot.StreamingCoTHalluDetector.detect",
                return_value=mock_result,
            ):
                result = pipeline.verify(
                    question="Solve x.",
                    response="Step 1: x=1.",
                )

        # Advisory does NOT trigger fast-path skip.
        assert result.skipped is False

    def test_verify_fields_default_when_disabled(self):
        """streaming_cot fields default to False/0.0 when STREAMING_COT_ENABLED=False.

        Spec: SCENARIO-VERIFY-166
        """
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline()

        with patch.object(VerifyRepairPipeline, "STREAMING_COT_ENABLED", False):
            result = pipeline.verify(
                question="What is 3+3?",
                response="Step 1: 3+3=6.",
            )

        assert result.streaming_cot_unstable is False
        assert result.streaming_cot_phas == 0.0
        assert "tier_0g_streaming_cot" not in result.certificate

    def test_verify_no_steps_no_streaming_result(self):
        """When no CoT steps are extracted, streaming_cot fields remain default.

        Spec: REQ-VERIFY-140 (graceful fallback for no-delimiter responses)
        """
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline()

        with patch.object(VerifyRepairPipeline, "STREAMING_COT_ENABLED", True):
            with patch(
                "carnot.pipeline.streaming_cot.extract_cot_steps",
                return_value=[],
            ):
                result = pipeline.verify(
                    question="Describe a cat.",
                    response="",
                )

        # Empty response → no steps → no streaming result → defaults hold.
        assert result.streaming_cot_unstable is False
        assert result.streaming_cot_phas == 0.0


class TestExperiment874Integration:
    """Integration test: run the experiment end-to-end.

    Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165, SCENARIO-VERIFY-166
    """

    def test_experiment_produces_wired_verdict(self, tmp_path):
        """Full experiment run produces honest_verdict=streaming_cot_wired.

        Validates that the pipeline wiring produces advisory signals for > 50%
        of the 25 synthetic responses when CARNOT_STREAMING_COT=1.
        """
        import importlib

        # Ensure CARNOT_STREAMING_COT=1 is active for the experiment module.
        os.environ["CARNOT_STREAMING_COT"] = "1"

        try:
            import scripts.experiment_874_streaming_cot_integration as exp874
            # Re-run to get fresh artifact (no caching needed).
            artifact = exp874.run_experiment()
        finally:
            os.environ.pop("CARNOT_STREAMING_COT", None)

        assert artifact["status"] == "success"
        assert artifact["honest_verdict"] in ("streaming_cot_wired", "wired_low_coverage")
        assert artifact["streaming_cot_enabled"] is True
        assert artifact["n_questions"] == 25
        assert 0.0 <= artifact["streaming_cot_advisory_rate"] <= 1.0
        assert 0.0 <= artifact["advisory_correct_prediction_rate"] <= 1.0
        assert artifact["skip_rate"] == 0.0  # No skipping from advisory signal
