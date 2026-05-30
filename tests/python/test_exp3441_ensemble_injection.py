"""Tests for exp3441 verifier-ensemble vs adaptive injection corpus.

Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the scripts directory importable for testing utility functions.
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import the functions under test from the experiment module.
exp_mod = importlib.import_module(
    "experiment_3441_verifier_ensemble_vs_adaptive_injection_corpus_v3"
)
binary_auroc = exp_mod.binary_auroc
binary_auprc = exp_mod.binary_auprc
bootstrap_delta_ci = exp_mod.bootstrap_delta_ci
bootstrap_delta_ci_unpaired = exp_mod.bootstrap_delta_ci_unpaired


# ---------------------------------------------------------------------------
# AUROC tests
# ---------------------------------------------------------------------------


class TestBinaryAuroc:
    """REQ-VERIFY-1121: AUROC computation correctness."""

    def test_perfect_separation(self):
        """Positives ranked above all negatives -> AUROC = 1.0."""
        labels = [1, 1, 1, 0, 0, 0]
        scores = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]
        assert binary_auroc(labels, scores) == pytest.approx(1.0)

    def test_inverted_separation(self):
        """Negatives ranked above all positives -> AUROC = 0.0."""
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
        assert binary_auroc(labels, scores) == pytest.approx(0.0)

    def test_random_like(self):
        """Fixed interleaved scores give AUROC = 0.5."""
        labels = [1, 0, 1, 0, 1, 0]
        scores = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # With these scores: positive at 0.6, 0.4, 0.2; negative at 0.5, 0.3, 0.1
        # Wins = (0.6>0.5 + 0.6>0.3 + 0.6>0.1) + (0.4>0.3 + 0.4>0.1) + (0.2>0.1) = 3+2+1=6
        # Total = 3*3 = 9; AUROC = 6/9 = 0.667 != 0.5
        # Different arrangement for true 0.5:
        labels2 = [1, 0, 1, 0]
        scores2 = [0.7, 0.8, 0.3, 0.2]
        # pos at 0.7, 0.3; neg at 0.8, 0.2
        # pos0.7 vs neg0.8: 0 + pos0.7 vs neg0.2: 1 = 1
        # pos0.3 vs neg0.8: 0 + pos0.3 vs neg0.2: 1 = 1
        # AUROC = 2/4 = 0.5
        assert binary_auroc(labels2, scores2) == pytest.approx(0.5)

    def test_single_class_returns_half(self):
        """Single class present -> return 0.5 (no information)."""
        labels = [1, 1, 1]
        scores = [0.9, 0.8, 0.7]
        assert binary_auroc(labels, scores) == pytest.approx(0.5)

    def test_all_negative_class(self):
        """All negatives -> return 0.5."""
        labels = [0, 0, 0]
        scores = [0.9, 0.5, 0.1]
        assert binary_auroc(labels, scores) == pytest.approx(0.5)

    def test_with_ties(self):
        """Ties receive 0.5 credit per pair."""
        labels = [1, 0]
        scores = [0.5, 0.5]  # tie
        assert binary_auroc(labels, scores) == pytest.approx(0.5)

    def test_numpy_arrays(self):
        """Accepts numpy arrays as well as lists."""
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert binary_auroc(labels, scores) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AUPRC tests
# ---------------------------------------------------------------------------


class TestBinaryAuprc:
    """REQ-VERIFY-1121: AUPRC computation correctness."""

    def test_perfect_ranking(self):
        """All positives ranked before all negatives -> AUPRC = 1.0."""
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        assert binary_auprc(labels, scores) == pytest.approx(1.0)

    def test_all_positives_last(self):
        """All positives ranked last -> AUPRC below 1."""
        labels = [1, 1, 0, 0]
        scores = [0.1, 0.2, 0.8, 0.9]
        assert binary_auprc(labels, scores) < 0.6

    def test_no_positives(self):
        """No positive examples -> AUPRC = 0.0."""
        labels = [0, 0, 0]
        scores = [0.9, 0.5, 0.1]
        assert binary_auprc(labels, scores) == pytest.approx(0.0)

    def test_auprc_in_range(self):
        """AUPRC is always in [0, 1]."""
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 50)
        scores = rng.random(50)
        auprc = binary_auprc(labels, scores)
        assert 0.0 <= auprc <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------


class TestBootstrapDeltaCi:
    """REQ-VERIFY-1121: Bootstrap CI for AUROC delta."""

    def test_equal_scores_ci_includes_zero(self):
        """When ensemble and reference are identical, CI contains 0."""
        rng = np.random.default_rng(1)
        labels = np.array([1, 1, 1, 0, 0, 0] * 20)
        scores = rng.random(len(labels))
        delta, lo, hi, _ens_auroc = bootstrap_delta_ci(
            scores, scores, labels, n_bootstrap=200, seed=1
        )
        assert lo <= 0.0 <= hi

    def test_ci_direction_when_ensemble_is_better(self):
        """Ensemble clearly better than reference -> delta > 0 and lo may be > 0."""
        labels = np.array([1, 1, 1, 0, 0, 0] * 20)
        ens_scores = np.where(labels == 1, 0.9, 0.1).astype(float)
        ref_scores = np.full(len(labels), 0.5)
        delta, lo, hi, ens_auroc = bootstrap_delta_ci(
            ens_scores, ref_scores, labels, n_bootstrap=500, seed=2
        )
        assert delta > 0
        assert ens_auroc > 0.9

    def test_returns_four_values(self):
        """bootstrap_delta_ci returns (delta, lo, hi, ens_auroc)."""
        labels = np.array([1, 0, 1, 0, 1, 0])
        scores = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1])
        result = bootstrap_delta_ci(scores, scores, labels, n_bootstrap=50, seed=3)
        assert len(result) == 4


class TestBootstrapDeltaCiUnpaired:
    """REQ-VERIFY-1121: Unpaired bootstrap CI vs fixed reference AUROC."""

    def test_delta_equals_ens_minus_ref(self):
        """Delta should be ensemble_auroc - reference_auroc."""
        labels = np.array([1, 1, 0, 0] * 10)
        scores = np.where(labels == 1, 0.8, 0.2).astype(float)
        ens_auroc = binary_auroc(labels, scores)
        ref_auroc = 0.5
        delta, lo, hi = bootstrap_delta_ci_unpaired(
            scores, labels, ref_auroc, n_bootstrap=100, seed=4
        )
        assert delta == pytest.approx(ens_auroc - ref_auroc, abs=1e-6)
        assert lo <= delta <= hi

    def test_noninferiority_margin(self):
        """CI lower bound > -0.02 means non-inferiority passes."""
        labels = np.array([1, 1, 0, 0] * 20)
        scores = np.where(labels == 1, 0.75, 0.25).astype(float)
        _delta, lo, _hi = bootstrap_delta_ci_unpaired(
            scores, labels, 0.5, n_bootstrap=500, seed=5
        )
        # Ensemble significantly above reference -> lo should be > -0.02
        assert lo > -0.02

    def test_returns_three_values(self):
        """Returns (delta, ci_lower, ci_upper)."""
        labels = np.array([1, 0] * 5)
        scores = np.full(10, 0.5)
        result = bootstrap_delta_ci_unpaired(scores, labels, 0.5, n_bootstrap=50, seed=6)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Label mapping tests
# ---------------------------------------------------------------------------


class TestLabelMapping:
    """SCENARIO-PHASE1D-001: Label mapping correctness."""

    def test_injection_maps_to_1(self):
        """source_label 'injection' -> 1."""
        rows = [{"source_label": "injection"}, {"source_label": "benign"}]
        labels = [1 if r["source_label"] == "injection" else 0 for r in rows]
        assert labels == [1, 0]

    def test_teacher_label_mapping(self):
        """teacher_label maps same as source_label."""
        rows = [
            {"teacher_label": "injection"},
            {"teacher_label": "benign"},
            {"teacher_label": "benign"},
        ]
        teacher = [1 if r["teacher_label"] == "injection" else 0 for r in rows]
        assert teacher == [1, 0, 0]


# ---------------------------------------------------------------------------
# Verifier loading smoke test
# ---------------------------------------------------------------------------


class TestVerifierLoading:
    """SCENARIO-PHASE1D-001: Verifier ensemble loads without error."""

    def test_diversity_registry_non_empty(self):
        """VERIFIER_REGISTRY has at least one entry."""
        from carnot.verify.verifier_ensemble_diversity import VERIFIER_REGISTRY

        assert len(VERIFIER_REGISTRY) >= 1

    def test_build_verifier_set_returns_list(self):
        """build_verifier_set() returns a non-empty list of (name, klass, fn)."""
        from carnot.verify.verifier_ensemble_diversity import build_verifier_set

        verifiers = build_verifier_set()
        assert isinstance(verifiers, list)
        assert len(verifiers) >= 1

    def test_verifier_scores_are_finite(self):
        """Each verifier in registry produces a finite float for simple text."""
        from carnot.verify.verifier_ensemble_diversity import build_verifier_set

        verifiers = build_verifier_set()
        record = {"step_text": "This is a test input for prompt injection detection."}
        for name, _klass, fn in verifiers:
            score = fn(record)
            assert isinstance(score, (int, float)), f"{name} did not return a number"
            assert np.isfinite(score), f"{name} returned non-finite score: {score}"


# ---------------------------------------------------------------------------
# Per-category AUROC edge case
# ---------------------------------------------------------------------------


class TestPerCategoryAuroc:
    """REQ-VERIFY-1121: Per-category AUROC handles single-class categories."""

    def test_single_class_category_returns_half(self):
        """A category with only one label class returns AUROC 0.5."""
        labels = np.array([1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7])
        assert binary_auroc(labels, scores) == pytest.approx(0.5)

    def test_mixed_category(self):
        """A mixed category computes AUROC correctly."""
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        assert binary_auroc(labels, scores) == pytest.approx(1.0)
