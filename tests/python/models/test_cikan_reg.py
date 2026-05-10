"""Tests for CIKAN Regularizer.

Spec: REQ-KAN-1688, SCENARIO-KAN-1688
"""

import jax.numpy as jnp
import pytest

from carnot.models.cikan_reg import CIKANRegularizer


def test_cikan_regularizer_increasing():
    # REQ-KAN-1688
    # SCENARIO-KAN-1688
    reg = CIKANRegularizer(weight=1.0, increasing=True)

    # Monotonic increasing
    coeffs_ok = jnp.array([1.0, 2.0, 3.0, 4.0])
    penalty_ok = reg(coeffs_ok)
    assert float(penalty_ok) == 0.0

    # Non-monotonic
    coeffs_bad = jnp.array([1.0, 3.0, 2.0, 4.0])
    # diffs: 2.0, -1.0, 2.0
    # violations: 0.0, 1.0, 0.0 -> sum = 1.0
    penalty_bad = reg(coeffs_bad)
    assert float(penalty_bad) == 1.0


def test_cikan_regularizer_decreasing():
    # REQ-KAN-1688
    reg = CIKANRegularizer(weight=1.0, increasing=False)

    # Monotonic decreasing
    coeffs_ok = jnp.array([4.0, 3.0, 2.0, 1.0])
    penalty_ok = reg(coeffs_ok)
    assert float(penalty_ok) == 0.0

    # Non-monotonic
    coeffs_bad = jnp.array([4.0, 2.0, 3.0, 1.0])
    # diffs: -2.0, 1.0, -2.0
    # violations: 0.0, 1.0, 0.0 -> sum = 1.0
    penalty_bad = reg(coeffs_bad)
    assert float(penalty_bad) == 1.0


def test_cikan_regularizer_2d():
    # REQ-KAN-1688
    reg = CIKANRegularizer(weight=0.5, increasing=True)
    
    # 2 splines
    coeffs = jnp.array([
        [1.0, 2.0, 3.0], # diffs: 1, 1 -> 0
        [3.0, 1.0, 4.0]  # diffs: -2, 3 -> 2
    ])
    penalty = reg(coeffs)
    assert float(penalty) == 1.0  # 0.5 * 2.0


def test_cikan_regularizer_invalid_ndim():
    reg = CIKANRegularizer()
    with pytest.raises(ValueError):
        reg(jnp.array([[[1.0]]]))
