"""Tests for KAN-CL Continual Learning algorithm.

Spec: REQ-KAN-1826, SCENARIO-KAN-1826
"""

import jax.numpy as jnp
import pytest

from carnot.models.kan_cl import KANCLRegularizer, ImportanceTracker


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
