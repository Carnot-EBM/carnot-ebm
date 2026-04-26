"""Tests for Experiment 894: VJEPA Streaming Logit Filter.

Spec traces: REQ-VERIFY-177, SCENARIO-VERIFY-177

**Coverage targets (code added in Exp 894):**
    - VJEPAStreamingLogitsProcessor.__call__: penalty applied when threshold exceeded
    - VJEPAStreamingLogitsProcessor.__call__: scores unchanged when below threshold
    - VJEPAStreamingLogitsProcessor.applied_count: increments exactly on each trigger
    - VJEPAStreamingLogitsProcessor.violation_probability: returns float in [0,1]
    - assign_honest_verdict: all four outcome branches
    - _is_correct: extracts numeric answer correctly, handles missing answer
    - _extract_numeric_answer: GSM8K #### pattern, 'answer is N', fallback, None
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from python.carnot.pipeline.vjepa_streaming_processor import (
    VJEPAStreamingLogitsProcessor,
)
from experiment_894_vjepa_streaming_filter import (
    _extract_numeric_answer,
    _is_correct,
    assign_honest_verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(
    violation_prob: float, threshold: float = 0.75, penalty_scale: float = 2.0
) -> VJEPAStreamingLogitsProcessor:
    """Build a VJEPAStreamingLogitsProcessor whose VJEPA always returns violation_prob."""
    vjepa = MagicMock()
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "The answer is 42"

    processor = VJEPAStreamingLogitsProcessor(
        vjepa=vjepa,
        tokenizer=tokenizer,
        violation_threshold=threshold,
        penalty_scale=penalty_scale,
    )
    # Override violation_probability directly so we don't need JAX in unit tests.
    processor.violation_probability = MagicMock(return_value=violation_prob)
    return processor


def _make_tensors(logit_value: float = 1.0, batch: int = 1, vocab: int = 10):
    """Return mock (input_ids, scores) tensors using a float list stand-in."""
    import torch

    input_ids = torch.zeros((batch, 5), dtype=torch.long)
    scores = torch.full((batch, vocab), logit_value)
    return input_ids, scores


# ---------------------------------------------------------------------------
# REQ-VERIFY-177-1: scores unchanged when below threshold
# ---------------------------------------------------------------------------


class TestScoresUnchangedBelowThreshold:
    """REQ-VERIFY-177-1 — no penalty when violation_prob <= threshold."""

    def test_scores_unchanged_at_zero_violation(self):
        """violation_prob=0.0 should never trigger penalty."""
        import torch

        processor = _make_processor(violation_prob=0.0)
        input_ids, scores = _make_tensors(logit_value=4.0)
        result = processor(input_ids, scores)
        assert torch.allclose(result, scores)

    def test_scores_unchanged_exactly_at_threshold(self):
        """violation_prob == threshold is NOT strictly > threshold; no penalty."""
        import torch

        processor = _make_processor(violation_prob=0.75, threshold=0.75)
        input_ids, scores = _make_tensors(logit_value=3.0)
        result = processor(input_ids, scores)
        assert torch.allclose(result, scores)

    def test_applied_count_zero_when_no_penalty(self):
        """applied_count stays 0 when threshold never exceeded."""
        processor = _make_processor(violation_prob=0.5)
        input_ids, scores = _make_tensors()
        processor(input_ids, scores)
        processor(input_ids, scores)
        assert processor.applied_count == 0


# ---------------------------------------------------------------------------
# REQ-VERIFY-177-2: scores divided when above threshold
# ---------------------------------------------------------------------------


class TestScoresDividedAboveThreshold:
    """REQ-VERIFY-177-2 — logits divided by penalty_scale when threshold exceeded."""

    def test_scores_divided_by_penalty_scale(self):
        """violation_prob=0.9 > 0.75: scores / 2.0."""
        import torch

        processor = _make_processor(violation_prob=0.9, penalty_scale=2.0)
        input_ids, scores = _make_tensors(logit_value=4.0)
        result = processor(input_ids, scores)
        expected = scores / 2.0
        assert torch.allclose(result, expected)

    def test_custom_penalty_scale_applied(self):
        """penalty_scale=3.0 should divide by 3.0."""
        import torch

        processor = _make_processor(violation_prob=0.9, penalty_scale=3.0)
        input_ids, scores = _make_tensors(logit_value=9.0)
        result = processor(input_ids, scores)
        expected = scores / 3.0
        assert torch.allclose(result, expected)

    def test_just_above_threshold_triggers_penalty(self):
        """0.751 > 0.75 should trigger penalty."""
        import torch

        processor = _make_processor(violation_prob=0.751, threshold=0.75)
        input_ids, scores = _make_tensors(logit_value=2.0)
        result = processor(input_ids, scores)
        expected = scores / 2.0
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# REQ-VERIFY-177-3: applied_count increments exactly on each trigger
# ---------------------------------------------------------------------------


class TestAppliedCount:
    """REQ-VERIFY-177-3 — applied_count is accurate."""

    def test_applied_count_increments_once_per_trigger(self):
        """Each call above threshold increments applied_count by 1."""
        processor = _make_processor(violation_prob=0.9)
        input_ids, scores = _make_tensors()
        processor(input_ids, scores)
        assert processor.applied_count == 1
        processor(input_ids, scores)
        assert processor.applied_count == 2

    def test_applied_count_starts_at_zero(self):
        """Fresh processor has applied_count == 0."""
        processor = _make_processor(violation_prob=0.5)
        assert processor.applied_count == 0

    def test_applied_count_mixed_calls(self):
        """3 above-threshold calls and 2 below: applied_count == 3."""
        import torch

        high_proc = _make_processor(violation_prob=0.9)
        low_proc = _make_processor(violation_prob=0.1)
        input_ids, scores = _make_tensors()
        high_proc(input_ids, scores)
        high_proc(input_ids, scores)
        low_proc(input_ids, scores)
        high_proc(input_ids, scores)
        low_proc(input_ids, scores)
        assert high_proc.applied_count == 3
        assert low_proc.applied_count == 0


# ---------------------------------------------------------------------------
# REQ-VERIFY-177-4: default constructor arguments
# ---------------------------------------------------------------------------


class TestDefaultConstructorArguments:
    """REQ-VERIFY-177-4 — constructable with just vjepa and tokenizer."""

    def test_default_threshold_and_scale(self):
        """Default violation_threshold=0.75, penalty_scale=2.0."""
        vjepa = MagicMock()
        tokenizer = MagicMock()
        processor = VJEPAStreamingLogitsProcessor(vjepa=vjepa, tokenizer=tokenizer)
        assert processor.violation_threshold == 0.75
        assert processor.penalty_scale == 2.0


# ---------------------------------------------------------------------------
# assign_honest_verdict
# ---------------------------------------------------------------------------


class TestAssignHonestVerdict:
    """assign_honest_verdict maps outcomes correctly."""

    def test_no_gpu(self):
        assert assign_honest_verdict(5, gpu_available=False) == "streaming_blocked_no_gpu"

    def test_positive_improvement(self):
        assert assign_honest_verdict(3, gpu_available=True) == "streaming_positive"

    def test_neutral(self):
        assert assign_honest_verdict(0, gpu_available=True) == "streaming_neutral"

    def test_negative_improvement(self):
        assert assign_honest_verdict(-2, gpu_available=True) == "streaming_negative"


# ---------------------------------------------------------------------------
# _extract_numeric_answer and _is_correct
# ---------------------------------------------------------------------------


class TestExtractNumericAnswer:
    """_extract_numeric_answer handles multiple formats."""

    def test_gsm8k_hash_pattern(self):
        assert _extract_numeric_answer("Step 1: ...\n#### 72") == 72.0

    def test_answer_is_pattern(self):
        assert _extract_numeric_answer("The answer is 42.") == 42.0

    def test_fallback_last_number(self):
        assert _extract_numeric_answer("total = 100") == 100.0

    def test_returns_none_on_no_number(self):
        assert _extract_numeric_answer("no numbers here") is None

    def test_comma_separated_number(self):
        assert _extract_numeric_answer("#### 1,000") == 1000.0


class TestIsCorrect:
    """_is_correct matches within tolerance."""

    def test_correct_integer_match(self):
        assert _is_correct("#### 72", 72) is True

    def test_incorrect_answer(self):
        assert _is_correct("#### 71", 72) is False

    def test_none_prediction_is_incorrect(self):
        assert _is_correct("no answer", 42) is False

    def test_float_within_tolerance(self):
        assert _is_correct("#### 10", 10) is True
