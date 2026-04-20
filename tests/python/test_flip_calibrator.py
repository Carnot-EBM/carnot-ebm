"""Tests for FLIPRewardCalibrator and FLIPRepairTriple — 100% coverage.

Spec: REQ-LEARN-076, SCENARIO-LEARN-118, SCENARIO-LEARN-119, SCENARIO-LEARN-120
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Module loader (deferred import to avoid top-level heavy imports)
# ---------------------------------------------------------------------------


def load_module():
    return importlib.import_module("carnot.pipeline.flip_calibrator")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_embed(text: str) -> list[float]:
    """Deterministic embed: one element per character, value = ord(c) % 10.

    WHY this embed: it produces stable, reproducible vectors for testing without
    requiring JAX or any ML library.  Strings that share characters will have
    non-zero cosine similarity, letting us test the 'improved' path.
    """
    chars = list(text)
    if not chars:
        return [0.0]
    return [float(ord(c) % 10) for c in chars]


def _zero_embed(text: str) -> list[float]:
    """Embed that always returns a zero vector — tests the divide-by-zero guard."""
    return [0.0] * max(1, len(text.split()))


# ---------------------------------------------------------------------------
# FLIPRepairTriple dataclass tests
# ---------------------------------------------------------------------------


class TestFLIPRepairTriple:
    """Tests for the FLIPRepairTriple dataclass. Spec: SCENARIO-LEARN-118"""

    def test_fields_set_correctly(self):
        """FLIPRepairTriple stores all fields without mutation."""
        mod = load_module()
        triple = mod.FLIPRepairTriple(
            question="What is 2+2?",
            original="2+2=5",
            repaired="2+2=4",
            verdict_correct=True,
        )
        assert triple.question == "What is 2+2?"
        assert triple.original == "2+2=5"
        assert triple.repaired == "2+2=4"
        assert triple.verdict_correct is True
        assert triple.flip_score is None  # default

    def test_repaired_none_allowed(self):
        """FLIPRepairTriple accepts repaired=None (no repair attempted)."""
        mod = load_module()
        triple = mod.FLIPRepairTriple(
            question="q", original="o", repaired=None, verdict_correct=False
        )
        assert triple.repaired is None

    def test_flip_score_can_be_set(self):
        """flip_score field can be mutated after construction."""
        mod = load_module()
        triple = mod.FLIPRepairTriple(
            question="q", original="o", repaired="r", verdict_correct=True
        )
        triple.flip_score = 0.75
        assert triple.flip_score == 0.75


# ---------------------------------------------------------------------------
# FLIPRewardCalibrator.backward_inference_score tests
# ---------------------------------------------------------------------------


class TestBackwardInferenceScore:
    """Tests for the cosine similarity core. Spec: SCENARIO-LEARN-119"""

    def test_identical_strings_score_one(self):
        """Identical response and question => cosine sim = 1.0."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        score = cal.backward_inference_score("hello world", "hello world")
        assert abs(score - 1.0) < 1e-6

    def test_zero_embed_returns_zero(self):
        """Zero embedding vectors => returns 0.0 (divide-by-zero guard)."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_zero_embed)
        score = cal.backward_inference_score("anything", "anything")
        assert score == 0.0

    def test_returns_float_in_range(self):
        """backward_inference_score returns a float in [-1, 1]. Spec: SCENARIO-LEARN-119"""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        score = cal.backward_inference_score("the answer is four", "what is two plus two")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_empty_response_with_zero_embed_guard(self):
        """Empty text mapped to zero vector => 0.0, no exception."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=lambda t: [0.0])
        score = cal.backward_inference_score("", "")
        assert score == 0.0


# ---------------------------------------------------------------------------
# FLIPRewardCalibrator.calibrate_repair tests
# ---------------------------------------------------------------------------


class TestCalibrateRepair:
    """Tests for per-triple repair calibration. Spec: SCENARIO-LEARN-118"""

    def test_repaired_none_sets_flip_score_none(self):
        """calibrate_repair sets flip_score=None when repaired is None."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        triple = mod.FLIPRepairTriple(
            question="q", original="o", repaired=None, verdict_correct=False
        )
        result = cal.calibrate_repair(triple)
        assert result.flip_score is None

    def test_repaired_sets_flip_score_float(self):
        """calibrate_repair sets flip_score to a float when repaired is present."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        triple = mod.FLIPRepairTriple(
            question="hello world",
            original="wrong answer",
            repaired="hello world",
            verdict_correct=True,
        )
        result = cal.calibrate_repair(triple)
        assert isinstance(result.flip_score, float)
        # repaired == question => score should be near 1.0
        assert result.flip_score > 0.9

    def test_returns_same_triple_object(self):
        """calibrate_repair mutates and returns the SAME triple object."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        triple = mod.FLIPRepairTriple(
            question="q", original="o", repaired="r", verdict_correct=True
        )
        result = cal.calibrate_repair(triple)
        assert result is triple


# ---------------------------------------------------------------------------
# FLIPRewardCalibrator.batch_calibrate tests
# ---------------------------------------------------------------------------


class TestBatchCalibrate:
    """Tests for batch statistics. Spec: SCENARIO-LEARN-120"""

    def test_empty_list_returns_neutral(self):
        """Empty triples list => mean=0.0, n_improved=0, quality='neutral'."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        result = cal.batch_calibrate([])
        assert result["mean_flip_score"] == 0.0
        assert result["n_improved"] == 0
        assert result["repair_quality"] == "neutral"

    def test_all_none_repaired_returns_neutral(self):
        """Triples where all repaired=None => same as empty (all skipped)."""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        triples = [
            mod.FLIPRepairTriple("q", "o", None, False),
            mod.FLIPRepairTriple("q", "o", None, True),
        ]
        result = cal.batch_calibrate(triples)
        assert result["mean_flip_score"] == 0.0
        assert result["n_improved"] == 0
        assert result["repair_quality"] == "neutral"

    def test_majority_improved_returns_good(self):
        """Majority of repairs improve alignment => repair_quality='good'. SCENARIO-LEARN-120"""
        mod = load_module()
        cal = mod.FLIPRewardCalibrator(embed_fn=_simple_embed)
        # question and repaired are identical => cosine=1.0, original is different => lower
        triples = [
            mod.FLIPRepairTriple("hello world", "zzzzz", "hello world", True),
            mod.FLIPRepairTriple("hello world", "zzzzz", "hello world", True),
            mod.FLIPRepairTriple("hello world", "zzzzz", "hello world", True),
        ]
        result = cal.batch_calibrate(triples)
        assert result["repair_quality"] == "good"
        assert result["n_improved"] == 3
        assert result["mean_flip_score"] > 0.0

    def test_no_improvements_returns_bad(self):
        """No repairs improve alignment => repair_quality='bad'."""
        mod = load_module()
        # Use embed that returns same vector for any input => score always == original
        # To force "not improved", make repaired == original so delta == 0 (not > 0)
        cal = mod.FLIPRewardCalibrator(embed_fn=lambda t: [1.0, 0.0])
        triples = [
            mod.FLIPRepairTriple("q", "o", "r", True),
        ]
        # All embed same => original_score == flip_score => not improved
        result = cal.batch_calibrate(triples)
        assert result["n_improved"] == 0
        assert result["repair_quality"] == "bad"

    def test_mixed_returns_neutral(self):
        """Half improved, half not => repair_quality='neutral'."""
        mod = load_module()

        improved_count = [0]

        def counting_embed(text: str) -> list[float]:
            # Makes "repaired_A" more aligned to "question" than "original_A",
            # and "repaired_B" equal to "original_B" (no improvement).
            if text == "question":
                return [1.0, 0.0]
            if text == "original_A":
                return [0.0, 1.0]  # low similarity to question
            if text == "repaired_A":
                return [1.0, 0.0]  # high similarity to question (improved)
            # B case: same vector for original and repaired => no improvement
            return [0.5, 0.5]

        cal = mod.FLIPRewardCalibrator(embed_fn=counting_embed)
        triples = [
            mod.FLIPRepairTriple("question", "original_A", "repaired_A", True),
            mod.FLIPRepairTriple("question", "B", "B_repaired", False),
        ]
        result = cal.batch_calibrate(triples)
        # 1 of 2 improved = exactly half = not > half => neutral
        assert result["repair_quality"] == "neutral"
        assert result["n_improved"] == 1

    def test_mean_flip_score_is_average(self):
        """mean_flip_score is the arithmetic mean of individual flip scores."""
        mod = load_module()

        def fixed_embed(text: str) -> list[float]:
            # Returns [1,0] for all input => cosine = 1.0 for any pair
            return [1.0, 0.0]

        cal = mod.FLIPRewardCalibrator(embed_fn=fixed_embed)
        triples = [
            mod.FLIPRepairTriple("q", "o1", "r1", True),
            mod.FLIPRepairTriple("q", "o2", "r2", True),
        ]
        result = cal.batch_calibrate(triples)
        assert abs(result["mean_flip_score"] - 1.0) < 1e-6
