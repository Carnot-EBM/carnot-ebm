"""CIKAN Regularizer — Constraint-Informed KAN monotonic regularizer.

This module provides a regularizer that penalizes non-monotonic coefficient
sequences in B-splines. By applying this penalty during training, we can
encourage splines to be monotonically increasing without requiring post-hoc
projection.

Spec: REQ-KAN-1688, SCENARIO-KAN-1688
"""

import jax
import jax.numpy as jnp


class CIKANRegularizer:
    """Regularizer for enforcing monotonic behavior in B-spline coefficients.

    For a spline to be monotonically increasing, its control points (coefficients)
    must be monotonically increasing: c_{i+1} >= c_i.
    This regularizer penalizes violations by computing sum(relu(c_i - c_{i+1})).

    Parameters
    ----------
    weight : float
        The weight of the regularization penalty.
    increasing : bool
        If True, penalizes non-increasing sequences. If False, penalizes
        non-decreasing sequences.
    """

    def __init__(self, weight: float = 1.0, increasing: bool = True):
        self.weight = weight
        self.increasing = increasing

    def compute_penalty(self, control_points: jax.Array) -> jax.Array:
        """Compute the monotonicity penalty for a set of control points.

        Parameters
        ----------
        control_points : jax.Array
            A 1D or 2D array of control points. If 2D, it is assumed to be
            shape (n_splines, n_knots), and the penalty is summed over all
            splines.

        Returns
        -------
        jax.Array
            A scalar penalty value.
        """
        if control_points.ndim == 1:
            diffs = jnp.diff(control_points)
        elif control_points.ndim == 2:
            diffs = jnp.diff(control_points, axis=-1)
        else:
            raise ValueError("control_points must be 1D or 2D")

        if self.increasing:
            # Penalize when c_{i+1} < c_i, i.e., diffs < 0
            violations = jax.nn.relu(-diffs)
        else:
            # Penalize when c_{i+1} > c_i, i.e., diffs > 0
            violations = jax.nn.relu(diffs)

        return self.weight * jnp.sum(violations)

    def __call__(self, control_points: jax.Array) -> jax.Array:
        """Alias for compute_penalty."""
        return self.compute_penalty(control_points)
