"""Tests for NUP Probe v6 training and evaluation helpers.

Tests the functions in experiment_608_nup_probe_v6.py that are testable
without running the full experiment (no live GPU required).

Spec: REQ-VERIFY-140, REQ-VERIFY-141,
      SCENARIO-VERIFY-171, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_608_nup_probe_v6 import (  # noqa: E402
    _compute_auc,
    _encode,
    _score,
    _stratified_split,
)


class TestEncode:
    """Tests for the bigram feature encoder."""

    def test_short_text_returns_zero_vector(self):
        # Strings shorter than 2 characters produce a zero vector
        vec = _encode("")
        assert len(vec) == 32
        assert all(v == 0.0 for v in vec)

        vec = _encode("a")
        assert all(v == 0.0 for v in vec)

    def test_normal_text_returns_unit_length(self):
        vec = _encode("hello world")
        assert len(vec) == 32
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_custom_energy_dim(self):
        vec = _encode("hello", energy_dim=16)
        assert len(vec) == 16

    def test_different_texts_produce_different_vectors(self):
        v1 = _encode("correct step")
        v2 = _encode("wrong hallucination")
        assert v1 != v2


class TestScore:
    """Tests for the energy score computation."""

    def test_zero_weights_returns_bias(self):
        weights = [0.0] * 32
        bias = 3.5
        result = _score("any text", weights, bias)
        assert result == pytest.approx(3.5, abs=1e-6)

    def test_score_changes_with_weights(self):
        weights1 = [1.0] * 32
        weights2 = [-1.0] * 32
        s1 = _score("hello world", weights1, 0.0)
        s2 = _score("hello world", weights2, 0.0)
        # One should be positive and the other negative (or at least different)
        assert s1 != pytest.approx(s2)

    def test_short_text_score_equals_bias(self):
        # Single char encodes to zero vector, so score = bias
        result = _score("x", [1.0] * 32, 2.0)
        assert result == pytest.approx(2.0, abs=1e-6)


class TestComputeAuc:
    """Tests for AUC computation on scored lists."""

    def test_empty_correct_returns_half(self):
        result = _compute_auc([], ["hallucination"], [0.0] * 32, 0.0)
        assert result == pytest.approx(0.5)

    def test_empty_incorrect_returns_half(self):
        result = _compute_auc(["correct step"], [], [0.0] * 32, 0.0)
        assert result == pytest.approx(0.5)

    def test_perfect_separation_returns_one(self):
        # Single-char text encodes to zero vector -> score = bias only.
        # Multi-char text with all-positive weights gets score > bias.
        # correct="x" (zero vector, score=0), incorrect="hello world" (positive score).
        # AUC = 1.0 because every incorrect > every correct.
        weights = [1.0] * 32
        correct_steps = ["x"]        # zero vector, score = 0
        incorrect_steps = ["hello world"]  # positive score
        auc = _compute_auc(correct_steps, incorrect_steps, weights, 0.0)
        assert auc == pytest.approx(1.0, abs=1e-6)

    def test_reverse_separation_returns_near_zero(self):
        # Reversed: correct has high energy, incorrect has low energy -> AUC = 0.
        weights = [1.0] * 32
        correct_steps = ["hello world"]  # high energy
        incorrect_steps = ["x"]          # zero energy
        auc = _compute_auc(correct_steps, incorrect_steps, weights, 0.0)
        assert auc == pytest.approx(0.0, abs=1e-6)

    def test_auc_bounded_in_zero_one(self):
        weights = [0.1] * 32
        correct = ["step one", "step two"]
        incorrect = ["error one", "wrong two"]
        auc = _compute_auc(correct, incorrect, weights, 0.0)
        assert 0.0 <= auc <= 1.0


class TestStratifiedSplit:
    """Tests for corpus splitting logic."""

    def test_split_preserves_total_count(self):
        entries = [{"response": f"r{i}", "is_correct": i % 2 == 0} for i in range(20)]
        train, val = _stratified_split(entries, train_frac=0.8)
        assert len(train) + len(val) == len(entries)

    def test_split_contains_both_classes_in_each_split(self):
        entries = [{"response": f"r{i}", "is_correct": i % 2 == 0} for i in range(20)]
        train, val = _stratified_split(entries, train_frac=0.8)
        train_correct = [e for e in train if e["is_correct"]]
        train_incorrect = [e for e in train if not e["is_correct"]]
        val_correct = [e for e in val if e["is_correct"]]
        val_incorrect = [e for e in val if not e["is_correct"]]
        assert len(train_correct) > 0
        assert len(train_incorrect) > 0
        assert len(val_correct) > 0
        assert len(val_incorrect) > 0

    def test_split_respects_train_fraction(self):
        entries = [{"response": f"r{i}", "is_correct": i % 2 == 0} for i in range(100)]
        train, val = _stratified_split(entries, train_frac=0.8)
        # Within 5% of expected 80 train
        assert 70 <= len(train) <= 90

    def test_split_is_reproducible(self):
        entries = [{"response": f"r{i}", "is_correct": i % 2 == 0} for i in range(20)]
        train1, val1 = _stratified_split(entries, seed=42)
        train2, val2 = _stratified_split(entries, seed=42)
        assert [e["response"] for e in train1] == [e["response"] for e in train2]

    def test_single_correct_entry_still_splits(self):
        entries = [{"response": "only correct", "is_correct": True}]
        train, val = _stratified_split(entries, train_frac=0.8)
        # max(1, int(1*0.8)) = max(1, 0) = 1 -> train gets it, val is empty
        assert len(train) + len(val) == 1
