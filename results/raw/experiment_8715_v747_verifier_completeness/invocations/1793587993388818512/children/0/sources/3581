"""KAN-CL Continual Learning — Per-Knot Importance Regularization.

**Researcher summary:**
    Continuous learning in KANs often suffers from catastrophic forgetting. KAN-CL
    (arXiv:2605.11181) addresses this by applying a penalty proportional to the
    importance of each B-spline control point (knot). Knots that were crucial in
    previous learning phases are heavily penalized for deviating, while unimportant
    knots can adapt to new tasks.

Spec: REQ-KAN-1826, SCENARIO-KAN-1826
"""

import jax
import jax.numpy as jnp


class KANCLRegularizer:
    """Computes the KAN-CL per-knot importance regularization penalty.

    This regularizer anchors B-spline control points to their values from previous
    learning phases, weighted by an importance matrix (e.g., Fisher Information
    Matrix diagonal or path integral importance).

    Parameters
    ----------
    importance_weight : float
        The overall scaling factor (lambda) for the KAN-CL penalty.
    """

    def __init__(self, importance_weight: float = 1.0):
        self.importance_weight = importance_weight

    def compute_penalty(
        self,
        current_control_points: jax.Array,
        anchored_control_points: jax.Array,
        importance_matrix: jax.Array,
    ) -> jax.Array:
        """Compute the KAN-CL importance-weighted penalty.

        Formula:
            L_reg = lambda * sum( importance * (current - anchored)^2 )

        Parameters
        ----------
        current_control_points : jax.Array
            The current values of the control points being trained.
        anchored_control_points : jax.Array
            The frozen values of the control points from the previous task.
        importance_matrix : jax.Array
            The per-knot importance weights. Must match the shape of the
            control points.

        Returns
        -------
        jax.Array
            A scalar penalty value to be added to the loss.
        """
        if current_control_points.shape != anchored_control_points.shape:
            raise ValueError("Shape mismatch between current and anchored control points.")
        if current_control_points.shape != importance_matrix.shape:
            raise ValueError("Shape mismatch between control points and importance matrix.")

        diff_sq = jnp.square(current_control_points - anchored_control_points)
        weighted_diff = importance_matrix * diff_sq
        return self.importance_weight * jnp.sum(weighted_diff)

    def __call__(
        self,
        current_control_points: jax.Array,
        anchored_control_points: jax.Array,
        importance_matrix: jax.Array,
    ) -> jax.Array:
        """Alias for compute_penalty."""
        return self.compute_penalty(
            current_control_points,
            anchored_control_points,
            importance_matrix,
        )


class ImportanceTracker:
    """Tracks importance weights for B-spline knots during training.

    This can be used to accumulate squared gradients (as a Fisher Information proxy)
    or other importance signals over a training phase.
    """

    def __init__(self, shape: tuple):
        self.importance = jnp.zeros(shape)
        self.num_samples = 0

    def update(self, gradients: jax.Array):
        """Update the importance matrix using the squared gradients of a batch.

        Parameters
        ----------
        gradients : jax.Array
            The gradients of the loss with respect to the control points.
        """
        if gradients.shape != self.importance.shape:
            raise ValueError("Shape mismatch between gradients and importance matrix.")
        self.importance = self.importance + jnp.square(gradients)
        self.num_samples += 1

    def get_importance(self) -> jax.Array:
        """Get the normalized importance matrix.

        Returns
        -------
        jax.Array
            The importance matrix.
        """
        if self.num_samples == 0:
            return self.importance
        return self.importance / self.num_samples
