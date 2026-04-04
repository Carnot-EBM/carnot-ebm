"""Train EBMs by backpropagating through gradient descent repair steps.

**Researcher summary:**
    Implements "training through optimization" from arxiv 2507.02092 (EBT paper,
    Algorithm 1). Instead of the standard NCE objective ("correct has lower energy
    than incorrect"), this trains the energy landscape so that gradient descent from
    random initializations reliably finds the correct answer. The energy landscape
    is optimized to be EASY TO REPAIR.

**Detailed explanation for engineers:**
    Standard EBM training (NCE, score matching) shapes the energy so that correct
    configurations have low energy and incorrect ones have high energy. But this
    does not guarantee that gradient descent will actually find those low-energy
    configurations from arbitrary starting points. The energy landscape might have
    spurious local minima, flat regions, or maze-like valleys.

    **Optimization training** directly addresses this: we train the energy function
    so that running gradient descent on it actually works. The training loop is:

    1. Start from random initial guesses: y_hat_0 ~ N(0, I)
    2. Run N gradient descent steps on the energy function:
       y_hat_{i+1} = y_hat_i - alpha * grad_E(y_hat_i)
    3. Measure how close the final y_hat_N is to the true target y_true
    4. Backpropagate through ALL N gradient descent steps to update the
       energy function's parameters

    **The key insight:** JAX's automatic differentiation can differentiate through
    differentiation itself. When we call jax.grad on a function that internally
    calls jax.grad (the inner gradient descent steps), JAX automatically computes
    Hessian-vector products. This gives us exact second-order gradients without
    ever forming the full Hessian matrix.

    **Why this is strictly stronger than NCE:**
    NCE only ensures E(correct) < E(incorrect). Optimization training ensures
    that gradient descent starting from random points converges to the correct
    answer. This implies the energy landscape has:
    - No spurious local minima near the correct answer
    - Smooth gradients pointing toward the correct answer
    - No flat regions that would stall gradient descent

    **Computational cost:**
    The second-order derivatives make each training step ~2-3x more expensive
    than first-order methods. However, the resulting energy landscapes are much
    better behaved, often requiring fewer sampling steps at inference time.

    **Reference:** arxiv 2507.02092, "Energy-Based Transformers," Algorithm 1.

Spec: REQ-TRAIN-005
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

# Type alias for energy functions: a callable that maps a 1-D array to a scalar.
# This is intentionally simpler than the full EnergyFunction protocol because
# optimization training only needs the energy computation, and using a plain
# callable makes the code JIT-friendly without closure complications.
EnergyCallable = Callable[[jax.Array], jax.Array]


def _unrolled_optimization_single(
    energy_fn: EnergyCallable,
    x_init: jax.Array,
    n_steps: int,
    step_size: float,
) -> jax.Array:
    """Run gradient descent on the energy function for n_steps, return final state.

    **Researcher summary:**
        Unrolled gradient descent: x_{i+1} = x_i - alpha * grad_E(x_i).
        Uses jax.lax.fori_loop for JIT compatibility.

    **Detailed explanation for engineers:**
        This function simulates running an optimizer on the energy landscape.
        Starting from x_init, it takes n_steps gradient descent steps, each
        moving in the direction of steepest energy decrease.

        **Why jax.lax.fori_loop instead of a Python for loop?**
        JAX's JIT compiler cannot trace through Python for loops with dynamic
        iteration counts. jax.lax.fori_loop is a JAX primitive that the
        compiler understands, allowing the entire unrolled optimization to
        be compiled into a single efficient kernel. It also enables JAX to
        differentiate through the loop (needed for the outer training loss).

        **What happens inside each step:**
        1. Compute the gradient of the energy at the current position:
           grad = dE/dx at x_current
        2. Move against the gradient (toward lower energy):
           x_next = x_current - step_size * grad

    Args:
        energy_fn: Callable that takes a 1-D array and returns a scalar energy.
        x_init: Starting point for gradient descent, shape (dim,).
        n_steps: Number of gradient descent steps to run.
        step_size: Learning rate (alpha) for gradient descent. Larger values
            mean bigger steps but risk overshooting minima.

    Returns:
        The final position after n_steps of gradient descent, shape (dim,).

    Spec: REQ-TRAIN-005
    """

    def body_fn(
        _i: int,
        x: jax.Array,
    ) -> jax.Array:
        """One step of gradient descent on the energy function.

        The underscore prefix on _i indicates we do not use the iteration
        index — every step is identical (constant step size, no scheduling).
        """
        grad = jax.grad(energy_fn)(x)
        return x - step_size * grad

    # jax.lax.fori_loop(lower, upper, body_fn, init_val) runs:
    #   val = init_val
    #   for i in range(lower, upper):
    #       val = body_fn(i, val)
    #   return val
    # This is fully differentiable and JIT-compatible.
    return jax.lax.fori_loop(0, n_steps, body_fn, x_init)


def optimization_training_loss(
    energy_fn: EnergyCallable,
    data_batch: jax.Array,
    n_optimization_steps: int = 10,
    step_size: float = 0.1,
    key: jax.Array | None = None,
) -> jax.Array:
    """Loss that backpropagates through gradient descent on energy.

    **Researcher summary:**
        L = mean(||y_hat_N - y_true||^2) where y_hat_N is obtained by running
        N gradient descent steps on E starting from random y_hat_0 ~ N(0, I).
        JAX backpropagates through all N gradient steps automatically via
        Hessian-vector products.

    **Detailed explanation for engineers:**
        This function implements Algorithm 1 from the EBT paper. The idea is:

        1. **Sample random starting points:** For each target data point y_true
           in the batch, sample a random initial guess y_hat_0 from a standard
           normal distribution N(0, I). These are deliberately bad guesses.

        2. **Run gradient descent on the energy:** For each random start,
           execute N steps of gradient descent on the energy function E:
               y_hat_{i+1} = y_hat_i - alpha * grad_E(y_hat_i)
           After N steps, we get y_hat_N — the energy function's "best guess"
           for what the correct configuration is.

        3. **Compute reconstruction loss:** Measure how close each y_hat_N is
           to the corresponding y_true using mean squared error:
               L = mean(||y_hat_N - y_true||^2)

        4. **Backpropagate through everything:** JAX's autodiff traces through
           the entire computation — including the N inner gradient descent steps.
           This means the outer gradient (used to update the energy function's
           parameters) involves second-order derivatives (Hessian-vector products).
           JAX handles this automatically via its nested differentiation support.

        **Why this works:**
        By minimizing this loss, we are directly training the energy function
        to have a landscape where gradient descent converges to the correct
        answer. If the energy has spurious minima or flat regions that prevent
        convergence, the loss will be high, and the training signal will reshape
        the landscape to eliminate those obstacles.

        **Practical considerations:**
        - n_optimization_steps: More steps = better convergence but more
          expensive (both forward and backward pass). 10-20 is typical.
        - step_size: Should be small enough to avoid divergence. 0.01-0.1
          is typical for most energy functions.
        - The random initialization means each training step explores different
          parts of the energy landscape, providing diverse training signal.

    Args:
        energy_fn: Callable that takes a 1-D array and returns scalar energy.
            This is the energy function being trained — its parameters will
            be updated by the outer optimizer.
        data_batch: Target configurations that gradient descent should find.
            Shape (batch_size, dim). These are the "correct answers."
        n_optimization_steps: Number of inner gradient descent steps. More
            steps give the optimizer more chance to converge but make the
            backward pass more expensive. Default: 10.
        step_size: Step size (learning rate) for the inner gradient descent.
            Default: 0.1.
        key: JAX PRNG key for sampling random initial guesses. If None,
            uses a default key (seed=0). Pass different keys each training
            iteration for stochastic training.

    Returns:
        Scalar loss: mean squared error between gradient-descent outputs
        and target configurations.

    For example::

        import jax
        import jax.numpy as jnp

        # Simple quadratic energy: gradient descent should find the origin
        def energy(x):
            return 0.5 * jnp.sum(x ** 2)

        data = jnp.zeros((8, 4))  # targets are at the origin
        loss = optimization_training_loss(energy, data, n_optimization_steps=20)
        # loss should be small because gradient descent on a quadratic converges

    Spec: REQ-TRAIN-005
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    batch_size, dim = data_batch.shape

    # Step 1: Sample random initial guesses from N(0, I).
    # Each row is an independent starting point for gradient descent.
    x_inits = jax.random.normal(key, shape=(batch_size, dim))

    # Step 2: Run unrolled gradient descent for each initial guess.
    # jax.vmap vectorizes the single-sample function over the batch dimension,
    # so all samples are optimized in parallel (no Python loop).
    optimized = jax.vmap(
        lambda x_init: _unrolled_optimization_single(
            energy_fn, x_init, n_optimization_steps, step_size
        )
    )(x_inits)

    # Step 3: Compute mean squared error between optimized points and targets.
    # ||y_hat_N - y_true||^2 summed over dimensions, averaged over the batch.
    diff = optimized - data_batch
    per_sample_mse = jnp.sum(diff**2, axis=-1)
    return jnp.mean(per_sample_mse)
