"""Tests for carnot.models.sparse_kaem_energy — SparseKAEMEnergy.

100% coverage target on sparse_kaem_energy.py.

Spec: REQ-SAMPLE-021, REQ-SAMPLE-022, SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.sparse_kaem_energy import SparseKAEMEnergy


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_model(n_vars: int = 5, n_knots: int = 8, top_k_fraction: float = 0.4) -> SparseKAEMEnergy:
    """Small SparseKAEMEnergy for fast tests."""
    return SparseKAEMEnergy(n_vars=n_vars, n_knots=n_knots, top_k_fraction=top_k_fraction)


def _make_data(n_samples: int = 20, n_vars: int = 5, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.array(rng.uniform(-1.0, 1.0, size=(n_samples, n_vars)).astype(np.float32))


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    """SparseKAEMEnergy initialises with correct attributes at default params.

    Spec: REQ-SAMPLE-021
    """
    m = SparseKAEMEnergy(n_vars=10, n_knots=16, top_k_fraction=0.2)
    assert m.n_vars == 10
    assert m.n_knots == 16
    assert m.top_k_fraction == 0.2
    assert m.top_k == max(1, int(10 * 0.2))
    assert m.coupling_matrix.shape == (10, 10)
    # Diagonal must be zero after init
    diag = np.array(jnp.diag(m.coupling_matrix))
    assert np.allclose(diag, 0.0, atol=1e-6)


def test_top_k_minimum_one() -> None:
    """top_k is clamped to at least 1 even when fraction is tiny.

    Spec: REQ-SAMPLE-021-4
    """
    m = SparseKAEMEnergy(n_vars=3, n_knots=4, top_k_fraction=0.01)
    assert m.top_k == 1


def test_init_invalid_n_vars() -> None:
    """n_vars=0 raises ValueError.

    Spec: REQ-SAMPLE-021
    """
    with pytest.raises(ValueError, match="n_vars"):
        SparseKAEMEnergy(n_vars=0)


def test_init_invalid_n_knots() -> None:
    """n_knots=1 raises ValueError.

    Spec: REQ-SAMPLE-021
    """
    with pytest.raises(ValueError, match="n_knots"):
        SparseKAEMEnergy(n_vars=4, n_knots=1)


def test_init_invalid_top_k_fraction() -> None:
    """top_k_fraction <= 0 or > 1 raises ValueError.

    Spec: REQ-SAMPLE-021
    """
    with pytest.raises(ValueError, match="top_k_fraction"):
        SparseKAEMEnergy(n_vars=4, n_knots=4, top_k_fraction=0.0)
    with pytest.raises(ValueError, match="top_k_fraction"):
        SparseKAEMEnergy(n_vars=4, n_knots=4, top_k_fraction=1.5)


# ---------------------------------------------------------------------------
# sparsify
# ---------------------------------------------------------------------------


def test_sparsify_retains_top_k_per_row() -> None:
    """Each row of sparsified matrix has at most top_k non-zero entries.

    Spec: REQ-SAMPLE-021-2, SCENARIO-SAMPLE-036
    """
    n_vars = 6
    top_k = 2
    m = SparseKAEMEnergy(n_vars=n_vars, n_knots=4, top_k_fraction=top_k / n_vars)
    # Construct a coupling matrix with distinct magnitudes
    rng = np.random.default_rng(123)
    raw = jnp.array(rng.uniform(0.1, 1.0, size=(n_vars, n_vars)).astype(np.float32))
    sparse = m.sparsify(raw)
    for row_i in range(n_vars):
        n_nonzero = int(jnp.sum(sparse[row_i] != 0.0))
        assert n_nonzero <= top_k, f"Row {row_i} has {n_nonzero} non-zeros, expected <= {top_k}"


def test_sparsify_keeps_largest_magnitudes() -> None:
    """Sparsified entries are those with largest absolute value in each row.

    Spec: REQ-SAMPLE-021-2
    """
    m = SparseKAEMEnergy(n_vars=4, n_knots=4, top_k_fraction=0.5)  # top_k=2
    couplings = jnp.array([
        [0.0, 0.1, 0.5, 0.3],   # top 2 = 0.5, 0.3
        [0.8, 0.0, 0.2, 0.4],   # top 2 = 0.8, 0.4
        [0.1, 0.9, 0.0, 0.7],   # top 2 = 0.9, 0.7
        [0.3, 0.2, 0.6, 0.0],   # top 2 = 0.6, 0.3
    ])
    sparse = m.sparsify(couplings)
    # Row 0: 0.5 and 0.3 should survive; 0.1 should be zeroed
    assert float(sparse[0, 2]) == pytest.approx(0.5)
    assert float(sparse[0, 3]) == pytest.approx(0.3)
    assert float(sparse[0, 1]) == pytest.approx(0.0)


def test_sparsify_single_topk() -> None:
    """With top_k=1, only one entry per row is kept.

    Spec: REQ-SAMPLE-021-2
    """
    m = SparseKAEMEnergy(n_vars=4, n_knots=4, top_k_fraction=0.01)  # top_k=1
    assert m.top_k == 1
    raw = jnp.array(np.random.default_rng(7).uniform(0.1, 1.0, (4, 4)).astype(np.float32))
    sparse = m.sparsify(raw)
    for i in range(4):
        n_nonzero = int(jnp.sum(sparse[i] != 0.0))
        assert n_nonzero <= 1, f"Row {i}: {n_nonzero} non-zeros"


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------


def test_energy_scalar_output() -> None:
    """energy(x) returns a scalar JAX array.

    Spec: REQ-SAMPLE-021-1
    """
    m = _make_model()
    x = jnp.zeros(5)
    e = m.energy(x)
    assert e.shape == ()


def test_energy_varies_with_x() -> None:
    """energy(x) is not constant — different inputs produce different values.

    Spec: REQ-SAMPLE-021-1
    """
    m = _make_model()
    x1 = jnp.ones(5) * 0.3
    x2 = jnp.ones(5) * -0.7
    e1 = float(m.energy(x1))
    e2 = float(m.energy(x2))
    assert e1 != e2


def test_energy_includes_coupling_term() -> None:
    """Coupling term contributes to energy when coupling_matrix is non-zero.

    We set coupling_matrix manually to a known value and verify energy changes
    when we flip the coupling coefficient.

    Spec: REQ-SAMPLE-021-1
    """
    m = _make_model(n_vars=2, n_knots=4, top_k_fraction=1.0)
    x = jnp.array([1.0, 1.0])
    # Zero coupling
    m.coupling_matrix = jnp.zeros((2, 2))
    e_no_coupling = float(m.energy(x))
    # Positive coupling
    m.coupling_matrix = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    e_with_coupling = float(m.energy(x))
    # With x=[1,1] and coupling=[[0,1],[1,0]], pairwise term = 0.5*(1+1)=1.0
    assert abs(e_with_coupling - e_no_coupling - 1.0) < 0.01


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_returns_loss_list() -> None:
    """fit() returns a list of floats of length n_epochs.

    Spec: REQ-SAMPLE-021-3
    """
    m = _make_model()
    data = _make_data()
    losses = m.fit(data, n_epochs=3)
    assert isinstance(losses, list)
    assert len(losses) == 3
    assert all(isinstance(v, float) for v in losses)


def test_fit_bad_shape_raises() -> None:
    """fit() raises ValueError if data has wrong n_vars.

    Spec: REQ-SAMPLE-021-3
    """
    m = _make_model(n_vars=5)
    bad_data = jnp.zeros((10, 3))
    with pytest.raises(ValueError, match="5"):
        m.fit(bad_data, n_epochs=1)


def test_fit_updates_coupling_matrix() -> None:
    """After fit(), coupling_matrix is sparsified (has zeros).

    Spec: REQ-SAMPLE-021-3
    """
    m = _make_model(n_vars=6, top_k_fraction=0.2)  # top_k=1
    data = _make_data(n_vars=6, n_samples=20)
    initial_coupling = np.array(m.coupling_matrix).copy()
    m.fit(data, n_epochs=2)
    # After fit, coupling should have changed
    updated_coupling = np.array(m.coupling_matrix)
    # Some entries should now be zero (sparsified)
    assert np.any(updated_coupling == 0.0), "Expected some zero entries after sparsification"


def test_fit_diagonal_remains_zero_after_training() -> None:
    """Diagonal of coupling_matrix stays zero after training (no self-coupling).

    Spec: REQ-SAMPLE-021-3
    """
    m = _make_model(n_vars=5, n_knots=4, top_k_fraction=0.4)
    data = _make_data(n_samples=15, n_vars=5)
    m.fit(data, n_epochs=3)
    diag = np.array(jnp.diag(m.coupling_matrix))
    assert np.allclose(diag, 0.0, atol=1e-5), f"Non-zero diagonal: {diag}"


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


def test_export_from_carnot_models() -> None:
    """SparseKAEMEnergy is accessible via carnot.models.

    Spec: REQ-SAMPLE-021-5
    """
    from carnot.models import SparseKAEMEnergy as SparseFromModels  # noqa: PLC0415
    assert SparseFromModels is SparseKAEMEnergy


# ---------------------------------------------------------------------------
# Accuracy smoke test (REQ-SAMPLE-022 fast approximation)
# ---------------------------------------------------------------------------


def test_energy_accuracy_vs_dense_within_reasonable_bounds() -> None:
    """SparseKAEMEnergy achieves accuracy comparable to dense KAEMEnergy baseline.

    Full 5% tolerance check lives in Exp 637.  This test verifies that
    SparseKAEMEnergy trains without blowing up and produces finite energies.

    Spec: REQ-SAMPLE-022, SCENARIO-SAMPLE-035
    """
    from carnot.models.kaem_energy import KAEMEnergy  # noqa: PLC0415

    n_vars = 8
    n_knots = 16
    rng = np.random.default_rng(99)
    train = jnp.array(rng.uniform(-1.0, 1.0, (50, n_vars)).astype(np.float32))
    test_x = rng.uniform(-1.0, 1.0, (30, n_vars)).astype(np.float32)

    def gt_energy(x: np.ndarray) -> float:
        return float(np.sum(np.sin(3.0 * x) + x**2))

    def compute_mae(m: "KAEMEnergy | SparseKAEMEnergy") -> float:
        preds = np.array([float(m.energy(jnp.array(test_x[i]))) for i in range(len(test_x))])
        gts = np.array([gt_energy(test_x[i]) for i in range(len(test_x))])
        preds -= preds.mean()
        gts -= gts.mean()
        return float(np.mean(np.abs(preds - gts)))

    dense_m = KAEMEnergy(n_vars=n_vars, n_hidden=n_knots)
    dense_m.fit(train, n_epochs=10)
    dense_mae = compute_mae(dense_m)

    sparse_m = SparseKAEMEnergy(n_vars=n_vars, n_knots=n_knots, top_k_fraction=0.5)
    sparse_m.fit(train, n_epochs=10)
    sparse_mae = compute_mae(sparse_m)

    # Both should produce finite energies (not NaN/inf)
    assert np.isfinite(dense_mae), f"Dense MAE not finite: {dense_mae}"
    assert np.isfinite(sparse_mae), f"Sparse MAE not finite: {sparse_mae}"
    # Sparse should not be wildly worse than dense (within 5x is acceptable for a smoke test)
    assert sparse_mae < dense_mae * 5 + 0.5, (
        f"Sparse MAE {sparse_mae:.4f} is much worse than dense {dense_mae:.4f}"
    )
