"""Tests for MISECalibrator and MISETriple — 100% coverage of mise_calibrator.py.

Spec: REQ-LEARN-070, REQ-LEARN-071, SCENARIO-LEARN-110, SCENARIO-LEARN-111,
      SCENARIO-LEARN-112
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Module loader (deferred import to avoid top-level JAX/heavy imports)
# ---------------------------------------------------------------------------


def load_module():
    return importlib.import_module("carnot.pipeline.mise_calibrator")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_embed(text: str) -> list[float]:
    """Deterministic embed: character length as a 1-D vector of one element.

    WHY: simplest possible embed_fn so cosine similarity is predictable —
    longer strings have larger embeddings, allowing controlled unit tests.
    """
    return [float(len(text))]


def _zero_embed(text: str) -> list[float]:
    """Always returns the zero vector — triggers zero-norm branch."""
    return [0.0]


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-110: MISETriple stores all four fields correctly
# ---------------------------------------------------------------------------


class TestMISETriple:
    def test_fields_stored(self):
        """SCENARIO-LEARN-110: MISETriple stores question, original_response,
        repaired_response, and verdict_correct without mutation."""
        m = load_module()
        triple = m.MISETriple(
            question="What is 2+2?",
            original_response="4",
            repaired_response=None,
            verdict_correct=True,
        )
        assert triple.question == "What is 2+2?"
        assert triple.original_response == "4"
        assert triple.repaired_response is None
        assert triple.verdict_correct is True

    def test_repaired_response_set(self):
        """SCENARIO-LEARN-110: repaired_response can hold a string."""
        m = load_module()
        triple = m.MISETriple(
            question="q",
            original_response="wrong",
            repaired_response="correct",
            verdict_correct=True,
        )
        assert triple.repaired_response == "correct"

    def test_verdict_false(self):
        """SCENARIO-LEARN-110: verdict_correct=False is stored faithfully."""
        m = load_module()
        triple = m.MISETriple(
            question="q",
            original_response="bad",
            repaired_response=None,
            verdict_correct=False,
        )
        assert triple.verdict_correct is False


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-111: backward_inference_score returns valid cosine similarity
# ---------------------------------------------------------------------------


class TestBackwardInferenceScore:
    def test_identical_strings_return_one(self):
        """SCENARIO-LEARN-111: identical embeddings produce cosine similarity 1.0."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        # "abc" and "abc" have identical embeddings — cosine = 1.0.
        score = cal.backward_inference_score("abc", "abc")
        assert abs(score - 1.0) < 1e-9

    def test_different_strings_in_range(self):
        """SCENARIO-LEARN-111: cosine similarity is in [-1, 1] for non-zero inputs."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        score = cal.backward_inference_score("hello world", "hi")
        assert -1.0 <= score <= 1.0

    def test_zero_embed_returns_zero(self):
        """SCENARIO-LEARN-111: zero-norm embedding returns 0.0 without ZeroDivisionError."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_zero_embed)
        score = cal.backward_inference_score("any text", "any question")
        assert score == 0.0

    def test_returns_float(self):
        """SCENARIO-LEARN-111: backward_inference_score always returns a Python float."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        score = cal.backward_inference_score("response", "question")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-110 + 112: calibrate() returns correct statistics
# ---------------------------------------------------------------------------


class TestMISECalibratorCalibrate:
    def _make_triple(self, module, q: str, r: str, verdict: bool):
        return module.MISETriple(
            question=q,
            original_response=r,
            repaired_response=None,
            verdict_correct=verdict,
        )

    def test_empty_triples_returns_zeros(self):
        """SCENARIO-LEARN-112: empty triple list returns all-zero stats."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        result = cal.calibrate([])
        assert result["mean_alignment_correct"] == 0.0
        assert result["mean_alignment_incorrect"] == 0.0
        assert result["calibration_gap"] == 0.0

    def test_all_correct_incorrect_group_is_zero(self):
        """SCENARIO-LEARN-112: when all triples are correct, incorrect group is 0.0."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        triples = [self._make_triple(m, "q", "response", True)]
        result = cal.calibrate(triples)
        assert result["mean_alignment_incorrect"] == 0.0
        assert result["mean_alignment_correct"] > 0.0

    def test_all_incorrect_correct_group_is_zero(self):
        """SCENARIO-LEARN-112: when all triples are incorrect, correct group is 0.0."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        triples = [self._make_triple(m, "q", "bad_response", False)]
        result = cal.calibrate(triples)
        assert result["mean_alignment_correct"] == 0.0

    def test_calibration_gap_is_difference(self):
        """SCENARIO-LEARN-112: calibration_gap = mean_alignment_correct - mean_alignment_incorrect."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        triples = [
            self._make_triple(m, "question", "correct answer", True),
            self._make_triple(m, "question", "wrong", False),
        ]
        result = cal.calibrate(triples)
        expected_gap = result["mean_alignment_correct"] - result["mean_alignment_incorrect"]
        assert abs(result["calibration_gap"] - expected_gap) < 1e-9

    def test_repaired_response_used_when_available(self):
        """SCENARIO-LEARN-110: calibrate uses repaired_response when it is not None."""
        m = load_module()
        # Build an embed_fn that distinguishes repaired vs original by content.
        seen_texts = []

        def tracking_embed(text: str) -> list[float]:
            seen_texts.append(text)
            return [float(len(text))]

        cal = m.MISECalibrator(embed_fn=tracking_embed)
        triple = m.MISETriple(
            question="q",
            original_response="wrong response",
            repaired_response="repaired",
            verdict_correct=True,
        )
        cal.calibrate([triple])
        # The repaired_response "repaired" must appear in seen_texts, not original.
        assert "repaired" in seen_texts
        assert "wrong response" not in seen_texts

    def test_result_has_required_keys(self):
        """SCENARIO-LEARN-112: calibrate returns dict with all three required keys."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        result = cal.calibrate([])
        assert set(result.keys()) == {
            "mean_alignment_correct",
            "mean_alignment_incorrect",
            "calibration_gap",
        }

    def test_mixed_correct_incorrect(self):
        """SCENARIO-LEARN-112: mixed triples produce non-trivial statistics."""
        m = load_module()
        cal = m.MISECalibrator(embed_fn=_simple_embed)
        triples = [
            self._make_triple(m, "What is 2+2?", "The answer is 4", True),
            self._make_triple(m, "What is 2+2?", "I do not know", False),
            self._make_triple(m, "What is 2+2?", "4 is correct", True),
            self._make_triple(m, "What is 2+2?", "maybe 3", False),
        ]
        result = cal.calibrate(triples)
        # Each group has two entries — means must be non-negative.
        assert result["mean_alignment_correct"] >= 0.0
        assert result["mean_alignment_incorrect"] >= 0.0

    def test_original_response_used_when_repair_is_none(self):
        """SCENARIO-LEARN-110: calibrate uses original_response when repaired_response is None."""
        m = load_module()
        seen_texts = []

        def tracking_embed(text: str) -> list[float]:
            seen_texts.append(text)
            return [1.0]

        cal = m.MISECalibrator(embed_fn=tracking_embed)
        triple = m.MISETriple(
            question="q",
            original_response="the original",
            repaired_response=None,
            verdict_correct=True,
        )
        cal.calibrate([triple])
        assert "the original" in seen_texts


# ---------------------------------------------------------------------------
# Module-level __all__ export check
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_all_exports_present(self):
        """SCENARIO-LEARN-110: MISECalibrator and MISETriple are in __all__."""
        m = load_module()
        assert "MISECalibrator" in m.__all__
        assert "MISETriple" in m.__all__

    def test_pipeline_init_exports(self):
        """SCENARIO-LEARN-110: carnot.pipeline re-exports MISECalibrator and MISETriple."""
        import carnot.pipeline as p

        assert hasattr(p, "MISECalibrator")
        assert hasattr(p, "MISETriple")
