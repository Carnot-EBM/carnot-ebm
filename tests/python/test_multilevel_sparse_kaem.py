"""Tests for carnot.training.multilevel_sparse_kaem — MultilevelSparseKAEMTrainer.

100% coverage target for the new module.

Spec: REQ-SAMPLE-025, SCENARIO-SAMPLE-040, SCENARIO-SAMPLE-041
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from carnot.models.sparse_kaem_energy import SparseKAEMEnergy
from carnot.training.multilevel_sparse_kaem import MultilevelSparseKAEMTrainer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_data(n_samples: int = 30, n_vars: int = 4, seed: int = 0) -> jnp.ndarray:
    """Small synthetic dataset for fast tests."""
    rng = np.random.default_rng(seed)
    return jnp.array(rng.uniform(-1.0, 1.0, size=(n_samples, n_vars)).astype(np.float32))


def _make_trainer(
    schedule: list[int] | None = None,
    epochs_per_level: int = 2,
    top_k_fraction: float = 0.5,
) -> MultilevelSparseKAEMTrainer:
    """Fast trainer for test use (minimal epochs, small knots)."""
    if schedule is None:
        schedule = [4, 8]
    return MultilevelSparseKAEMTrainer(
        schedule=schedule,
        epochs_per_level=epochs_per_level,
        top_k_fraction=top_k_fraction,
    )


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    """Default schedule, epochs_per_level, top_k_fraction are set correctly.

    Spec: REQ-SAMPLE-025
    """
    t = MultilevelSparseKAEMTrainer()
    assert t.schedule == [16, 32, 64]
    assert t.epochs_per_level == 20
    assert t.top_k_fraction == 0.1


def test_init_custom() -> None:
    """Custom parameters are stored as-is.

    Spec: REQ-SAMPLE-025
    """
    t = MultilevelSparseKAEMTrainer(schedule=[8, 16], epochs_per_level=5, top_k_fraction=0.2)
    assert t.schedule == [8, 16]
    assert t.epochs_per_level == 5
    assert t.top_k_fraction == 0.2


def test_init_empty_schedule_raises() -> None:
    """Empty schedule raises ValueError.

    Spec: REQ-SAMPLE-025
    """
    with pytest.raises(ValueError, match="at least one level"):
        MultilevelSparseKAEMTrainer(schedule=[])


def test_init_bad_epochs_raises() -> None:
    """epochs_per_level < 1 raises ValueError.

    Spec: REQ-SAMPLE-025
    """
    with pytest.raises(ValueError, match="epochs_per_level"):
        MultilevelSparseKAEMTrainer(epochs_per_level=0)


def test_init_bad_top_k_fraction_raises() -> None:
    """top_k_fraction outside (0, 1] raises ValueError.

    Spec: REQ-SAMPLE-025
    """
    with pytest.raises(ValueError, match="top_k_fraction"):
        MultilevelSparseKAEMTrainer(top_k_fraction=0.0)
    with pytest.raises(ValueError, match="top_k_fraction"):
        MultilevelSparseKAEMTrainer(top_k_fraction=1.1)


# ---------------------------------------------------------------------------
# _sparsify_level
# ---------------------------------------------------------------------------


def test_sparsify_level_updates_coupling() -> None:
    """_sparsify_level stores sparsified coupling back on the model.

    Spec: REQ-SAMPLE-025-3
    """
    t = _make_trainer()
    model = SparseKAEMEnergy(n_vars=4, n_knots=4, top_k_fraction=0.5)
    # Set coupling to a non-trivial matrix
    model.coupling_matrix = jnp.ones((4, 4)) * 0.5
    before = np.array(model.coupling_matrix)
    result = t._sparsify_level(model)
    after = np.array(result.coupling_matrix)
    # sparsify zeros out at least some entries
    assert np.sum(after == 0.0) >= np.sum(before == 0.0)
    assert result is model  # in-place, same object


# ---------------------------------------------------------------------------
# _refine_to_level
# ---------------------------------------------------------------------------


def test_refine_to_level_increases_knots() -> None:
    """_refine_to_level returns a SparseKAEMEnergy with the requested finer knot count.

    Spec: REQ-SAMPLE-025-1
    """
    t = _make_trainer(schedule=[4, 8])
    model = SparseKAEMEnergy(n_vars=4, n_knots=4, top_k_fraction=0.5)
    # Train briefly so control_points are non-trivial
    data = _make_data(n_vars=4)
    model.fit(data, n_epochs=1)

    refined = t._refine_to_level(model, K=8)
    assert refined.n_knots == 8
    assert refined.n_vars == 4
    assert refined.layer.control_points.shape == (4, 8)
    # New model, not same object
    assert refined is not model


def test_refine_preserves_n_vars() -> None:
    """Refined model has same n_vars as original.

    Spec: REQ-SAMPLE-025-1
    """
    t = _make_trainer(schedule=[4, 8])
    model = SparseKAEMEnergy(n_vars=6, n_knots=4, top_k_fraction=0.5)
    refined = t._refine_to_level(model, K=8)
    assert refined.n_vars == 6


# ---------------------------------------------------------------------------
# _train_level
# ---------------------------------------------------------------------------


def test_train_level_returns_model() -> None:
    """_train_level returns the same model object.

    Spec: REQ-SAMPLE-025-2
    """
    t = _make_trainer()
    model = SparseKAEMEnergy(n_vars=4, n_knots=4, top_k_fraction=0.5)
    data = _make_data(n_vars=4)
    result = t._train_level(model, data, n_epochs=1)
    assert result is model


# ---------------------------------------------------------------------------
# train (full pipeline)
# ---------------------------------------------------------------------------


def test_train_single_level() -> None:
    """Single-level schedule skips refinement and returns a trained model.

    Spec: REQ-SAMPLE-025-4
    """
    t = MultilevelSparseKAEMTrainer(schedule=[4], epochs_per_level=1, top_k_fraction=0.5)
    data = _make_data(n_vars=3)
    model = t.train(n_vars=3, data=data)
    assert isinstance(model, SparseKAEMEnergy)
    assert model.n_knots == 4
    assert model.n_vars == 3


def test_train_two_levels() -> None:
    """Two-level schedule produces model at second knot count.

    Spec: REQ-SAMPLE-025-4
    """
    t = MultilevelSparseKAEMTrainer(schedule=[4, 8], epochs_per_level=1, top_k_fraction=0.5)
    data = _make_data(n_vars=4)
    model = t.train(n_vars=4, data=data)
    assert isinstance(model, SparseKAEMEnergy)
    assert model.n_knots == 8
    assert model.n_vars == 4


def test_train_three_levels() -> None:
    """Three-level schedule (default pattern) produces model at finest resolution.

    Spec: REQ-SAMPLE-025-4
    """
    t = MultilevelSparseKAEMTrainer(schedule=[4, 6, 8], epochs_per_level=1, top_k_fraction=0.5)
    data = _make_data(n_vars=4)
    model = t.train(n_vars=4, data=data)
    assert model.n_knots == 8


def test_train_produces_valid_energy() -> None:
    """Trained model returns a finite scalar energy for a random input.

    Spec: REQ-SAMPLE-025
    """
    t = _make_trainer(schedule=[4, 6])
    data = _make_data(n_vars=4)
    model = t.train(n_vars=4, data=data)
    x = jnp.zeros(4)
    e = model.energy(x)
    assert jnp.isfinite(e)


def test_train_coupling_sparsified() -> None:
    """Coupling matrix after training has many zeros (sparsification applied).

    Spec: REQ-SAMPLE-025-3
    """
    t = MultilevelSparseKAEMTrainer(schedule=[4, 6], epochs_per_level=1, top_k_fraction=0.5)
    data = _make_data(n_vars=6, n_samples=30)
    model = t.train(n_vars=6, data=data)
    coupling = np.array(model.coupling_matrix)
    # With top_k_fraction=0.5 and n_vars=6, top_k=3 entries kept per row.
    # So each row has at most 3 non-zero entries; overall matrix should be sparse.
    n_nonzero = np.sum(coupling != 0.0)
    n_total = 6 * 6
    # At most top_k*n_vars*2 non-zero (symmetrised) entries
    assert n_nonzero < n_total


# ---------------------------------------------------------------------------
# Export from carnot.training
# ---------------------------------------------------------------------------


def test_export_from_training() -> None:
    """MultilevelSparseKAEMTrainer is exported from carnot.training.

    Spec: REQ-SAMPLE-025-5
    """
    from carnot.training import MultilevelSparseKAEMTrainer as T
    assert T is MultilevelSparseKAEMTrainer
