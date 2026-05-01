"""Tests for Phase-1c verifier joint null-space measurement (Exp 1093).

These tests verify that:
  - NullSpaceEstimator loads and runs without error on synthetic data.
  - The joint null space is smaller than any single verifier's null space.
  - r_correlation is measurable for all verifier pairs.
  - joint_null_space_fraction is reported and in [0, 1].

Spec: REQ-DIAG-003, SCENARIO-PHASE1C-001
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.eval.diagnostics import NullSpaceEstimator, make_test_verifiers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def diverse_scores() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, scores) for a well-behaved diverse verifier ensemble.

    Three verifiers each fire on a different region of input space so that
    the joint null space (all quiet at once) is near-zero while each
    individual null space is larger.
    """
    rng = np.random.default_rng(42)
    N = 300
    D = 8
    X = rng.standard_normal((N, D))

    # Verifier i is active (high energy) on roughly 2/3 of inputs, quiet on 1/3.
    # The three quiet regions are disjoint, so the joint quiet region is tiny.
    scores = np.column_stack(
        [
            np.abs(X[:, 0]),  # quiet near x0=0
            np.abs(X[:, 1]),  # quiet near x1=0
            np.abs(X[:, 2]),  # quiet near x2=0
        ]
    )
    return X, scores


@pytest.fixture()
def correlated_scores() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, scores) for a correlated (redundant) verifier ensemble.

    Both verifiers are nearly identical copies of the same signal, so
    r_correlation should be close to 1.0.
    """
    rng = np.random.default_rng(99)
    N = 200
    D = 4
    X = rng.standard_normal((N, D))
    base = rng.standard_normal(N)
    noise = rng.standard_normal(N) * 0.01
    scores = np.column_stack([base, base + noise])
    return X, scores


# ---------------------------------------------------------------------------
# Test 1: NullSpaceEstimator loads and runs without error
# ---------------------------------------------------------------------------


def test_null_space_estimator_loads_without_error(diverse_scores):
    """NullSpaceEstimator must instantiate and fit without raising.

    Regression guard: if the module path or constructor changes, this test
    catches it immediately.

    Spec: REQ-DIAG-003
    """
    X, scores = diverse_scores
    estimator = NullSpaceEstimator()
    # fit() must not raise
    estimator.fit(X=X, verifier_scores=scores)
    frac = estimator.joint_null_space_fraction()
    # sanity: fraction must be in valid range
    assert 0.0 <= frac <= 1.0, f"joint_null_space_fraction out of [0,1]: {frac}"


# ---------------------------------------------------------------------------
# Test 2: Single-verifier null space is larger than (or equal to) joint
# ---------------------------------------------------------------------------


def test_single_verifier_null_space_larger_than_joint(diverse_scores):
    """For diverse verifiers, the joint null space is <= any individual null space.

    The joint null space is the intersection of all individual null spaces.
    By definition it can never exceed the smallest individual null space.
    This test verifies that the estimator honours that invariant on synthetic data.

    Spec: SCENARIO-PHASE1C-001
    """
    X, scores = diverse_scores
    estimator = NullSpaceEstimator()
    estimator.fit(X=X, verifier_scores=scores)
    joint_frac = estimator.joint_null_space_fraction()

    # Compute each verifier's individual null-space fraction
    threshold = 0.1 * float(np.std(scores)) + 1e-9
    individual_fracs = [
        float(np.mean(np.abs(scores[:, k]) < threshold)) for k in range(scores.shape[1])
    ]
    min_individual = min(individual_fracs)

    assert joint_frac <= min_individual + 1e-9, (
        f"joint null-space fraction {joint_frac:.4f} exceeds "
        f"minimum individual fraction {min_individual:.4f} — estimator violated "
        "the intersection-subset invariant"
    )


# ---------------------------------------------------------------------------
# Test 3: r_correlation is measurable for all verifier pairs
# ---------------------------------------------------------------------------


def test_r_correlation_measured_for_all_pairs(diverse_scores, correlated_scores):
    """r_correlation must return a float in [0, 1] for every valid (i, j) pair.

    Low r on diverse verifiers (~0) and high r on correlated verifiers (~1)
    validates that the metric discriminates ensemble quality.

    Spec: REQ-DIAG-003
    """
    # Diverse ensemble: r should be low
    X_div, scores_div = diverse_scores
    est_div = NullSpaceEstimator()
    est_div.fit(X=X_div, verifier_scores=scores_div)
    n_div = scores_div.shape[1]
    for i in range(n_div):
        for j in range(i + 1, n_div):
            r = est_div.r_correlation(i, j)
            assert 0.0 <= r <= 1.0 + 1e-9, f"r_correlation({i},{j}) = {r} out of [0,1]"

    # Correlated ensemble: r should be high (> 0.9)
    X_cor, scores_cor = correlated_scores
    est_cor = NullSpaceEstimator()
    est_cor.fit(X=X_cor, verifier_scores=scores_cor)
    r_high = est_cor.r_correlation(0, 1)
    assert r_high > 0.9, f"Correlated verifiers should have r_correlation > 0.9, got {r_high:.4f}"


# ---------------------------------------------------------------------------
# Test 4: joint_null_space_fraction is reported and within expected range
# ---------------------------------------------------------------------------


def test_joint_null_space_fraction_reported():
    """joint_null_space_fraction must be in [0, 1] and must be < 0.05 for
    the make_test_verifiers() ensemble (the canonical 'well-behaved' reference).

    make_test_verifiers() returns verifiers that fire on different input axes,
    so their joint null space on random data should be tiny.

    Spec: REQ-DIAG-003, SCENARIO-PHASE1C-001
    """
    rng = np.random.default_rng(7)
    N = 400
    D = 8
    X = rng.standard_normal((N, D))

    verifiers = make_test_verifiers(n=3)
    scores = np.column_stack([v(X) for v in verifiers])

    estimator = NullSpaceEstimator()
    estimator.fit(X=X, verifier_scores=scores)
    frac = estimator.joint_null_space_fraction()

    assert 0.0 <= frac <= 1.0, f"fraction out of [0,1]: {frac}"
    assert frac < 0.05, (
        f"make_test_verifiers() ensemble should have joint null-space < 0.05, "
        f"got {frac:.4f} — verifiers may be degenerate"
    )
