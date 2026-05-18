"""Tests for KAN-CL Continual Learning algorithm.

Spec: REQ-KAN-1826, SCENARIO-KAN-1826
"""

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.learning.kan_cl import (
    KanClLearner,
    build_split_task_benchmark_payload,
    make_split_task_constraint_tasks,
)
from carnot.models.kan_cl import KANCLRegularizer, ImportanceTracker
from carnot.models.kan import KAN


def test_kan_cl_regularizer():
    """Test the KAN-CL penalty computation."""
    reg = KANCLRegularizer(importance_weight=0.5)

    anchored = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    current = jnp.array([[1.5, 2.0], [3.0, 3.0]])

    importance = jnp.array([[2.0, 1.0], [1.0, 0.5]])

    penalty = reg.compute_penalty(current, anchored, importance)
    expected = 0.5 * 1.0

    assert jnp.allclose(penalty, expected)

    # Test callable alias
    penalty_call = reg(current, anchored, importance)
    assert jnp.allclose(penalty_call, expected)


def test_kan_cl_regularizer_shape_mismatch():
    """Test that shape mismatches raise ValueError."""
    reg = KANCLRegularizer()
    anchored = jnp.ones((2, 2))
    current = jnp.ones((2, 3))
    importance = jnp.ones((2, 2))

    with pytest.raises(ValueError, match="Shape mismatch between current and anchored"):
        reg(current, anchored, importance)

    current_2 = jnp.ones((2, 2))
    importance_2 = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="Shape mismatch between control points and importance"):
        reg(current_2, anchored, importance_2)


def test_importance_tracker():
    """Test the knot importance tracker update mechanism."""
    tracker = ImportanceTracker(shape=(2,))
    assert jnp.allclose(tracker.get_importance(), jnp.zeros(2))

    grads1 = jnp.array([1.0, 2.0])
    tracker.update(grads1)

    grads2 = jnp.array([3.0, 1.0])
    tracker.update(grads2)

    importance = tracker.get_importance()
    assert jnp.allclose(importance, jnp.array([5.0, 2.5]))


def test_importance_tracker_shape_mismatch():
    """Test that tracker catches shape mismatches."""
    tracker = ImportanceTracker(shape=(2,))
    with pytest.raises(ValueError, match="Shape mismatch between gradients and importance matrix"):
        tracker.update(jnp.array([1.0, 2.0, 3.0]))


def test_minimal_kan_import_exposes_n256():
    """REQ-KAN-1826: `from carnot.models.kan import KAN` exposes an n=256 KAN."""
    model = KAN(n_params=256, seed=42)

    assert model.n_params == 256
    assert model.coefficients.shape == (256,)


def test_kancl_importance_uses_activation_frequency():
    """SCENARIO-KAN-1826-N256: Importance is per-knot activation frequency."""
    learner = KanClLearner(n_params=4, epochs=1, seed=42)
    x = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )

    importance = learner.compute_importance(x)

    np.testing.assert_allclose(importance, np.array([2 / 3, 1 / 3, 2 / 3, 1 / 3]))


def test_kancl_fit_predict_and_task_importance():
    """REQ-KAN-1826: KanClLearner fits, predicts, and records task importances."""
    tasks = make_split_task_constraint_tasks(n_params=256, examples_per_task=50, seed=42)
    learner = KanClLearner(n_params=256, epochs=80, learning_rate=0.08, seed=42)

    learner.fit(tasks[0].X, tasks[0].y, task_id=tasks[0].task_id)
    preds = learner.predict(tasks[0].X)

    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})
    assert tasks[0].task_id in learner.task_importances
    assert learner.task_importances[tasks[0].task_id].shape == (256,)
    assert float(np.max(learner.task_importances[tasks[0].task_id])) <= 1.0


def test_split_task_benchmark_validates_kancl_n256():
    """SCENARIO-KAN-1826-N256: Benchmark gates KAN-CL on >=50% forgetting reduction."""
    payload = build_split_task_benchmark_payload(seed=42)

    assert payload["n_tasks"] == 3
    assert payload["n_params"] == 256
    assert payload["random_seed"] == 42
    assert payload["kancl_n256_validated"] is True
    assert payload["forgetting_reduction_pct"] >= 50.0
