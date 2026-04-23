"""Tests for Experiment 782 — EDU-PRM Step Selection.

Spec: REQ-LEARN-050, REQ-LEARN-051, SCENARIO-LEARN-094, SCENARIO-LEARN-095

Coverage target: 100% of carnot/pipeline/edu_prm_selector.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make the package importable when running from repo root.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from carnot.pipeline.edu_prm_selector import (  # noqa: E402
    EDUPRMConfig,
    EDUPRMStepSelector,
    _LogisticRegression,
    _TFIDFVec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_steps(n: int = 20) -> tuple[list[str], list[int]]:
    """Generate synthetic step_texts and binary labels for unit testing."""
    texts = [
        f"Step {i}: {'correct calculation result' if i % 2 == 0 else 'incorrect wrong error'}"
        for i in range(n)
    ]
    labels = [i % 2 for i in range(n)]
    return texts, labels


# ---------------------------------------------------------------------------
# REQ-LEARN-050: bootstrap variance computation
# ---------------------------------------------------------------------------


class TestEDUPRMStepSelector:
    """Tests for EDUPRMStepSelector — REQ-LEARN-050."""

    def test_fit_bootstrap_runs_n_bootstrap_iterations(self):
        """fit_bootstrap MUST run exactly N_BOOTSTRAP=10 iterations and store
        N_BOOTSTRAP prediction lists (REQ-LEARN-050).
        """
        texts, labels = _make_steps(20)
        config = EDUPRMConfig(n_bootstrap=10, selection_pct=0.30)
        selector = EDUPRMStepSelector(config)
        selector.fit_bootstrap(texts, labels)
        # Must store exactly 10 bootstrap prediction lists.
        assert len(selector._bootstrap_preds) == 10, (
            f"Expected 10 bootstrap prediction lists, got {len(selector._bootstrap_preds)}"
        )

    def test_fit_bootstrap_stores_predictions_for_all_steps(self):
        """Each bootstrap prediction list MUST cover every step in the corpus
        (REQ-LEARN-050).
        """
        texts, labels = _make_steps(15)
        selector = EDUPRMStepSelector(EDUPRMConfig(n_bootstrap=10))
        selector.fit_bootstrap(texts, labels)
        for b, preds in enumerate(selector._bootstrap_preds):
            assert len(preds) == len(texts), (
                f"Bootstrap model {b}: expected {len(texts)} predictions, got {len(preds)}"
            )

    def test_select_returns_top_selection_pct_fraction(self):
        """select() MUST return indices covering exactly the top selection_pct
        fraction of steps by variance (REQ-LEARN-050).
        """
        import math

        texts, labels = _make_steps(20)
        config = EDUPRMConfig(n_bootstrap=10, selection_pct=0.30)
        selector = EDUPRMStepSelector(config)
        indices = selector.select(texts, labels)
        expected_k = math.ceil(20 * 0.30)  # ceil(6) = 6
        assert len(indices) == expected_k, (
            f"Expected {expected_k} selected indices, got {len(indices)}"
        )

    def test_select_indices_are_valid(self):
        """All returned indices MUST be valid positions in the input list
        (REQ-LEARN-050).
        """
        texts, labels = _make_steps(20)
        selector = EDUPRMStepSelector(EDUPRMConfig())
        indices = selector.select(texts, labels)
        for idx in indices:
            assert 0 <= idx < len(texts), f"Index {idx} out of range [0, {len(texts)})"

    def test_select_empty_corpus(self):
        """select() on an empty corpus MUST return an empty list without raising
        (REQ-LEARN-050).
        """
        selector = EDUPRMStepSelector(EDUPRMConfig())
        result = selector.select([], [])
        assert result == []

    def test_select_uses_variance_ordering(self):
        """The selected indices MUST be those with highest variance, not random
        (REQ-LEARN-050).

        We verify this by checking that _variances are computed and that each
        selected index has variance >= every non-selected index.
        """
        texts, labels = _make_steps(20)
        config = EDUPRMConfig(n_bootstrap=10, selection_pct=0.30)
        selector = EDUPRMStepSelector(config)
        indices = selector.select(texts, labels)
        selected_set = set(indices)
        all_indices = set(range(len(texts)))
        non_selected = all_indices - selected_set

        if non_selected:
            min_selected_var = min(selector._variances[i] for i in selected_set)
            max_nonsel_var = max(selector._variances[i] for i in non_selected)
            assert min_selected_var >= max_nonsel_var - 1e-12, (
                "Some non-selected step has higher variance than a selected step"
            )


# ---------------------------------------------------------------------------
# REQ-LEARN-051: diversity_score
# ---------------------------------------------------------------------------


class TestDiversityScore:
    """Tests for diversity_score — REQ-LEARN-051."""

    def test_diversity_score_is_positive_fraction(self):
        """diversity_score MUST equal positive_fraction of selected labels
        (REQ-LEARN-051).
        """
        selector = EDUPRMStepSelector()
        labels = [1, 1, 0, 0, 1]  # 3/5 = 0.6 positive
        score = selector.diversity_score(labels)
        assert abs(score - 0.6) < 1e-9, f"Expected 0.6, got {score}"

    def test_diversity_score_all_positive(self):
        """diversity_score of [1, 1, 1] MUST be 1.0 (REQ-LEARN-051)."""
        selector = EDUPRMStepSelector()
        assert selector.diversity_score([1, 1, 1]) == pytest.approx(1.0)

    def test_diversity_score_all_negative(self):
        """diversity_score of [0, 0, 0] MUST be 0.0 (REQ-LEARN-051)."""
        selector = EDUPRMStepSelector()
        assert selector.diversity_score([0, 0, 0]) == pytest.approx(0.0)

    def test_diversity_score_empty(self):
        """diversity_score of [] MUST return 0.0 without raising
        (REQ-LEARN-051).
        """
        selector = EDUPRMStepSelector()
        assert selector.diversity_score([]) == pytest.approx(0.0)

    def test_diversity_score_balanced(self):
        """diversity_score near 0.5 indicates balanced hard examples, the ideal
        outcome of EDU-PRM selection (REQ-LEARN-051).
        """
        selector = EDUPRMStepSelector()
        score = selector.diversity_score([1, 0, 1, 0])
        assert abs(score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Internal component tests (for 100% coverage)
# ---------------------------------------------------------------------------


class TestTFIDFVec:
    """Coverage for _TFIDFVec — used internally by EDUPRMStepSelector."""

    def test_fit_transform_returns_correct_length(self):
        vec = _TFIDFVec(max_features=10)
        corpus = ["hello world error", "correct result", "wrong calculation error"]
        vec.fit(corpus)
        result = vec.transform("hello error")
        assert len(result) == len(vec._vocab)

    def test_fit_respects_max_features(self):
        vec = _TFIDFVec(max_features=5)
        corpus = [f"word{i} extra text" for i in range(20)]
        vec.fit(corpus)
        assert len(vec._vocab) <= 5

    def test_transform_unknown_tokens_are_zero(self):
        vec = _TFIDFVec(max_features=10)
        vec.fit(["hello world"])
        result = vec.transform("zzzzunknowntoken")
        assert all(v == pytest.approx(0.0) for v in result)

    def test_empty_document_does_not_crash(self):
        vec = _TFIDFVec(max_features=10)
        vec.fit(["hello world"])
        result = vec.transform("")
        assert len(result) == len(vec._vocab)


class TestLogisticRegression:
    """Coverage for _LogisticRegression — used internally by EDUPRMStepSelector."""

    def test_fit_predict_binary(self):
        clf = _LogisticRegression(n_features=2, lr=1.0, n_epochs=100)
        X = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        y = [1, 1, 0, 0]
        clf.fit(X, y)
        preds = clf.predict_proba(X)
        # Class 1 examples should have higher probability than class 0.
        assert preds[0] > preds[2], "Class 1 should have higher probability than class 0"

    def test_sigmoid_positive_input(self):
        clf = _LogisticRegression(n_features=1)
        # Large positive z → sigmoid close to 1.
        assert clf._sigmoid(10.0) > 0.99

    def test_sigmoid_negative_input(self):
        clf = _LogisticRegression(n_features=1)
        # Large negative z → sigmoid close to 0.
        assert clf._sigmoid(-10.0) < 0.01

    def test_sigmoid_zero(self):
        clf = _LogisticRegression(n_features=1)
        assert abs(clf._sigmoid(0.0) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# EDUPRMConfig dataclass
# ---------------------------------------------------------------------------


class TestEDUPRMConfig:
    """Verify defaults match spec values (REQ-LEARN-050)."""

    def test_defaults(self):
        cfg = EDUPRMConfig()
        assert cfg.n_bootstrap == 10
        assert cfg.selection_pct == pytest.approx(0.30)
        assert cfg.max_features == 128
        assert cfg.random_seed == 42
