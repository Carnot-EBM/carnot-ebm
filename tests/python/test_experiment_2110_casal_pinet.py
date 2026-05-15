"""Tests for Exp 2110: CASAL + PiNet Douglas-Rachford projection integration.

Spec coverage: REQ-SAMPLE-2110, REQ-SAMPLE-2110-1, REQ-SAMPLE-2110-2,
               REQ-SAMPLE-2110-3, REQ-SAMPLE-2110-5, SCENARIO-SAMPLE-2110
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from carnot.models.pinet_layer import DouglasRachfordPiNetLayer, LinearConstraintSet
from carnot.samplers.casal import casal_sample


def _halfspace_layer() -> DouglasRachfordPiNetLayer:
    """DR layer encoding sum(state) >= 1  (i.e., -x1 - x2 <= -1)."""
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        inequality_matrix=[[-1.0, -1.0]],
        inequality_bound=[-1.0],
        name="halfspace_sum_ge_1",
    )
    return DouglasRachfordPiNetLayer(constraints, max_steps=64, tolerance=1e-5)


def _simplex_layer() -> DouglasRachfordPiNetLayer:
    """DR layer encoding the 2-simplex: x1 + x2 = 1, x1 >= 0, x2 >= 0."""
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        equality_matrix=[[1.0, 1.0]],
        equality_target=[1.0],
        inequality_matrix=[[-1.0, 0.0], [0.0, -1.0]],
        inequality_bound=[0.0, 0.0],
        name="simplex_2d",
    )
    return DouglasRachfordPiNetLayer(constraints, max_steps=64, tolerance=1e-5)


class TestCasalPiNetSignature:
    """REQ-SAMPLE-2110-1: casal_sample accepts pinet_layer parameter."""

    def test_pinet_layer_none_is_default(self) -> None:
        """REQ-SAMPLE-2110-5: pinet_layer=None preserves original behavior."""
        key = jax.random.PRNGKey(0)

        def energy_fn(state: jax.Array) -> jax.Array:
            return jnp.sum(state ** 2)

        def constraint_fn(state: jax.Array) -> jax.Array:
            return jax.nn.relu(1.0 - jnp.sum(state))

        init = jnp.array([1.0, 1.0])
        # Default call (no pinet_layer) must still work
        result = casal_sample(
            energy_fn=energy_fn,
            constraint_fn=constraint_fn,
            init_state=init,
            steps=20,
            key=key,
        )
        assert jnp.all(jnp.isfinite(result))

    def test_pinet_layer_accepted(self) -> None:
        """REQ-SAMPLE-2110-1: casal_sample accepts a DouglasRachfordPiNetLayer."""
        key = jax.random.PRNGKey(1)
        layer = _halfspace_layer()

        def energy_fn(state: jax.Array) -> jax.Array:
            return jnp.sum(state ** 2)

        def constraint_fn(state: jax.Array) -> jax.Array:
            return jax.nn.relu(1.0 - jnp.sum(state))

        init = jnp.array([1.0, 1.0])
        result = casal_sample(
            energy_fn=energy_fn,
            constraint_fn=constraint_fn,
            init_state=init,
            steps=20,
            key=key,
            pinet_layer=layer,
        )
        assert jnp.all(jnp.isfinite(result))


class TestCasalPiNetProjection:
    """REQ-SAMPLE-2110-2, REQ-SAMPLE-2110-3: DR projection replaces gradient descent."""

    def test_zero_violation_halfspace(self) -> None:
        """SCENARIO-SAMPLE-2110: No violation with halfspace PiNet layer."""
        key = jax.random.PRNGKey(42)
        layer = _halfspace_layer()

        def energy_fn(state: jax.Array) -> jax.Array:
            return jnp.sum(state ** 2)

        def constraint_fn(state: jax.Array) -> jax.Array:
            # sum(state) >= 1 <=> relu(1 - sum(state)) == 0
            return jax.nn.relu(1.0 - jnp.sum(state))

        init = jnp.array([1.0, 1.0])
        final = casal_sample(
            energy_fn=energy_fn,
            constraint_fn=constraint_fn,
            init_state=init,
            steps=200,
            key=key,
            step_size=0.05,
            pinet_layer=layer,
        )
        assert float(constraint_fn(final)) <= 1e-5, (
            f"Constraint violated: violation={float(constraint_fn(final))}"
        )

    def test_output_finite_simplex(self) -> None:
        """REQ-SAMPLE-2110-2: DR projection on simplex keeps state finite."""
        key = jax.random.PRNGKey(7)
        layer = _simplex_layer()

        def energy_fn(state: jax.Array) -> jax.Array:
            return jnp.sum(state ** 2)

        def constraint_fn(state: jax.Array) -> jax.Array:
            eq_err = jnp.abs(jnp.sum(state) - 1.0)
            ineq_err = jnp.max(jnp.maximum(-state, 0.0))
            return jnp.maximum(eq_err, ineq_err)

        init = jnp.array([0.5, 0.5])
        final = casal_sample(
            energy_fn=energy_fn,
            constraint_fn=constraint_fn,
            init_state=init,
            steps=100,
            key=key,
            step_size=0.01,
            pinet_layer=layer,
        )
        assert jnp.all(jnp.isfinite(final))

    def test_pinet_projects_before_gate(self) -> None:
        """REQ-SAMPLE-2110-3: Acceptance gate fires after PiNet projection.

        A large step_size causes proposed states far outside the constraint.
        PiNet projection brings them back; the gate accepts the projected state.
        """
        key = jax.random.PRNGKey(99)
        layer = _halfspace_layer()

        def energy_fn(state: jax.Array) -> jax.Array:
            return jnp.sum(state ** 2)

        def constraint_fn(state: jax.Array) -> jax.Array:
            return jax.nn.relu(1.0 - jnp.sum(state))

        init = jnp.array([5.0, 5.0])
        final = casal_sample(
            energy_fn=energy_fn,
            constraint_fn=constraint_fn,
            init_state=init,
            steps=50,
            key=key,
            step_size=1.0,   # large — will often violate without projection
            pinet_layer=layer,
        )
        assert float(constraint_fn(final)) <= 1e-5
