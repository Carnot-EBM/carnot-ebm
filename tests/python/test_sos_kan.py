"""Tests for SOSKANEnergy — verifies type-level monotonicity and non-negativity invariants.

Spec: REQ-MODEL-SOS-001 (type-level monotonicity invariant),
      REQ-SAMPLE-015 (energy model interface compatibility)
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.models.sos_kan import SOSKANEnergy, _hat_basis, _precompute_phi_grid


# ---------------------------------------------------------------------------
# Test 1: Instantiation without error
# ---------------------------------------------------------------------------


def test_instantiation():
    """SOSKANEnergy instantiates with default and custom parameters without error.

    REQ-MODEL-SOS-001: model must be constructable with n_sos_basis >= 2.
    """
    m1 = SOSKANEnergy()
    assert m1.n_splines == 8
    assert m1.n_sos_basis == 2
    assert m1.n_features == 16

    m2 = SOSKANEnergy(n_splines=4, n_sos_basis=3, n_features=8, seed=7)
    assert m2.n_splines == 4
    assert m2.n_sos_basis == 3
    assert m2.n_features == 8

    # V shape must be (n_features, n_splines, n_sos_basis)
    assert m2.V.shape == (8, 4, 3)

    # n_sos_basis < 2 must raise
    with pytest.raises(ValueError, match="Burer-Monteiro"):
        SOSKANEnergy(n_sos_basis=1)


# ---------------------------------------------------------------------------
# Test 2: forward() produces non-negative outputs for arbitrary parameters
# ---------------------------------------------------------------------------


def test_forward_nonnegative():
    """forward() returns E >= 0 for any V (including large random values).

    REQ-MODEL-SOS-001: non-negativity is a type-level invariant — not a
    constraint that must be checked after training, but a property guaranteed
    by the c² + ∫ ||V^T B||² construction for all V.
    """
    rng = np.random.default_rng(42)
    model = SOSKANEnergy(n_splines=8, n_sos_basis=2, n_features=8, seed=42)

    # Overwrite V with large random values (stress test)
    model.V = rng.normal(0.0, 10.0, model.V.shape)
    model.c = rng.normal(0.0, 5.0, model.c.shape)

    xs = rng.uniform(-1.0, 1.0, (50, model.n_features))
    for i in range(len(xs)):
        e = model.energy(xs[i])
        assert e >= -1e-9, f"Negative energy {e:.6f} at sample {i}"


# ---------------------------------------------------------------------------
# Test 3: Zero monotonicity violations on random test grid
# ---------------------------------------------------------------------------


def test_zero_monotonicity_violations():
    """verify_invariants() reports 0 monotonicity violations for any V.

    REQ-MODEL-SOS-001: ψ'(x) = ||V^T B(x)||² >= 0 by SOS construction.
    This must hold for all V (untrained, trained, or adversarially set).
    """
    rng = np.random.default_rng(99)
    model = SOSKANEnergy(n_splines=8, n_sos_basis=2, n_features=6, seed=0)

    # Use adversarially large/negative V to stress the invariant
    model.V = rng.normal(0.0, 5.0, model.V.shape)
    model.c = rng.normal(-3.0, 1.0, model.c.shape)

    result = model.verify_invariants(n_samples=500, eps_monotone=1e-6, rng_seed=7)
    assert result["n_monotone_violations"] == 0, (
        f"Expected 0 monotonicity violations, got {result['n_monotone_violations']}"
    )
    assert result["invariants_hold"] is True


# ---------------------------------------------------------------------------
# Test 4: AUROC >= 0.5 on toy binary classification
# ---------------------------------------------------------------------------


def test_auroc_above_chance():
    """After training on separable toy data, AUROC >= 0.5.

    REQ-MODEL-SOS-001, REQ-EVAL-001: the trained model must achieve at least
    chance-level discrimination on a test set. We use a linearly separable
    toy dataset where one feature clearly discriminates the two classes.
    """
    rng = np.random.default_rng(42)
    n_train, n_test = 40, 20
    n_features = 4

    # Separable construction: class 1 (correct) has lower values on feature 0
    X_pos = rng.uniform(-1.0, 0.0, (n_train // 2, n_features))
    X_neg = rng.uniform(0.0, 1.0, (n_train // 2, n_features))
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.array([1] * (n_train // 2) + [0] * (n_train // 2), dtype=np.float64)

    X_pos_t = rng.uniform(-1.0, 0.0, (n_test // 2, n_features))
    X_neg_t = rng.uniform(0.0, 1.0, (n_test // 2, n_features))
    X_test = np.vstack([X_pos_t, X_neg_t])
    y_test = np.array([1] * (n_test // 2) + [0] * (n_test // 2), dtype=np.float64)

    model = SOSKANEnergy(n_splines=4, n_sos_basis=2, n_features=n_features, seed=42)
    model.fit(X_train, y_train, n_epochs=50, lr=0.01)
    auc = model.auroc(X_test, y_test)

    assert auc >= 0.5, f"AUROC {auc:.4f} below chance (0.5) after training on separable data"


# ---------------------------------------------------------------------------
# Test 5: fit() converges (loss decreases over 10 epochs)
# ---------------------------------------------------------------------------


def test_fit_converges():
    """fit() reduces BCE loss over 20 epochs on structured binary classification data.

    REQ-MODEL-SOS-001: training must make measurable progress; the Adam
    optimizer must update V to reduce the classification loss.

    We use a separable dataset (positive class has low feature 0, negative has
    high feature 0) to ensure there IS a learnable signal. Pure random data has
    no signal and Adam may not improve loss in just 10 epochs.
    """
    rng = np.random.default_rng(42)
    n_features = 4
    n_each = 20

    # Positive class (y=1): low feature 0, other features random
    X_pos = rng.uniform(-1.0, -0.3, (n_each, n_features))
    # Negative class (y=0): high feature 0, other features random
    X_neg = rng.uniform(0.3, 1.0, (n_each, n_features))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * n_each + [0.0] * n_each)

    model = SOSKANEnergy(n_splines=4, n_sos_basis=2, n_features=n_features, seed=42)
    losses = model.fit(X, y, n_epochs=20, lr=0.01)

    assert len(losses) == 20, f"Expected 20 loss values, got {len(losses)}"
    # Compare average of last 5 epochs vs first 5 epochs (smoothed comparison)
    avg_first = float(np.mean(losses[:5]))
    avg_last = float(np.mean(losses[15:]))
    assert avg_last < avg_first, (
        f"Loss did not decrease: avg first 5 epochs {avg_first:.6f}, "
        f"avg last 5 epochs {avg_last:.6f}"
    )


# ---------------------------------------------------------------------------
# Additional: hat basis and phi precomputation correctness
# ---------------------------------------------------------------------------


def test_hat_basis_partition_of_unity():
    """Hat basis functions sum to 1 at every interior point (partition of unity).

    This is a fundamental property of B-spline hat functions on a uniform knot
    grid. It ensures the SOS construction is well-calibrated: the integral
    Φ_{ij}(x) at x=1 equals the total area under B_i * B_j, which is bounded.
    """
    n = 8
    knots = np.linspace(-1.0, 1.0, n)
    xs = np.linspace(-0.9, 0.9, 50)  # interior points only
    B = _hat_basis(xs, knots)  # (50, 8)
    row_sums = B.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, 1.0, atol=1e-10, err_msg="Hat basis violates partition of unity"
    )


def test_phi_grid_nonnegative():
    """Precomputed Φ_{ij}(x) values are all non-negative (they are integrals of B_i*B_j >= 0)."""
    x_grid, phi_grid = _precompute_phi_grid(n_splines=6, grid_size=100)
    assert np.all(phi_grid >= -1e-12), f"phi_grid has negative values (min={phi_grid.min():.2e})"
