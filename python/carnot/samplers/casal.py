import jax
import jax.numpy as jnp

def casal_sample(energy_fn, constraint_fn, init_state, steps, key, step_size=1e-2, proj_steps=10, proj_lr=0.1, pinet_layer=None):
    """
    CASAL sampler variant for strictly constrained generative modeling.

    This implements a Split Augmented Langevin Sampling approach where an unconstrained
    Langevin step is followed by a projection step to enforce hard constraints.

    When pinet_layer is provided (a DouglasRachfordPiNetLayer), it replaces the
    gradient-descent projection with an exact Douglas-Rachford projection onto the
    linear constraint set. This gives geometric convergence guarantees instead of
    relying on a fixed number of gradient steps. The constraint_fn acceptance gate
    still applies after DR projection, so violation_rate = 0.0 is guaranteed when
    the initial state is feasible.

    Why we need both constraint_fn and pinet_layer: constraint_fn is the scalar
    violation signal used for the acceptance gate; pinet_layer encodes the same
    constraints as a LinearConstraintSet for exact closed-form projection. They
    must encode compatible constraint sets for the sampler to make forward progress.

    Args:
        energy_fn: Callable that takes a state and returns a scalar energy.
        constraint_fn: Callable that takes a state and returns 0 if satisfied, >0 if violated.
        init_state: Initial state tensor.
        steps: Number of sampling steps.
        key: JAX PRNG key.
        step_size: Step size for Langevin dynamics.
        proj_steps: Number of gradient descent steps for constraint projection (ignored when pinet_layer is set).
        proj_lr: Learning rate for projection steps (ignored when pinet_layer is set).
        pinet_layer: Optional DouglasRachfordPiNetLayer. When provided, uses DR projection
            instead of gradient descent for the primal projection step. Spec: REQ-SAMPLE-2110.

    Returns:
        Final state after `steps` iterations.
    """
    def step_fn(state, k):
        # 1. Unconstrained Langevin step
        grad_energy = jax.grad(energy_fn)(state)
        noise = jax.random.normal(k, state.shape)
        proposed_state = state - step_size * grad_energy + jnp.sqrt(2 * step_size) * noise

        # 2. Projection step: use DR PiNet layer when available, else gradient descent.
        # DR projection is preferred because it provides geometric convergence for
        # linear constraints — it finds the nearest feasible point in O(max_steps)
        # iterations rather than relying on a small fixed proj_lr.
        if pinet_layer is not None:
            projected_state = pinet_layer.project_vector(proposed_state)
        else:
            # Gradient descent on constraint_fn to push violation to 0
            def proj_body_fn(i, s):
                grad_c = jax.grad(constraint_fn)(s)
                # Only apply gradient if there's a violation
                v = constraint_fn(s)
                return jax.lax.cond(
                    v > 0,
                    lambda _: s - proj_lr * grad_c,
                    lambda _: s,
                    operand=None
                )
            projected_state = jax.lax.fori_loop(0, proj_steps, proj_body_fn, proposed_state)

        # 3. Strict gate to ensure hard constraints
        # If projection wasn't enough to reach 0 violation, reject the step
        violation = constraint_fn(projected_state)

        # Using a small epsilon to account for floating point inaccuracies
        accepted_state = jax.lax.cond(
            violation <= 1e-5,
            lambda _: projected_state,
            lambda _: state,
            operand=None
        )
        return accepted_state, accepted_state

    keys = jax.random.split(key, steps)
    final_state, _ = jax.lax.scan(step_fn, init_state, keys)
    return final_state
