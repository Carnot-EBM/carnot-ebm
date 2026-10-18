import jax
import jax.numpy as jnp
from typing import Callable, Tuple

def mpemba_init(
    key: jax.Array,
    energy_fn: Callable[[jax.Array], jax.Array],
    shape: Tuple[int, ...],
    hot_beta: float,
    target_beta: float,
    num_optim_steps: int = 10,
    step_size: float = 0.05
) -> jax.Array:
    """
    Generates a non-equilibrium initial state inspired by the Mpemba effect
    to accelerate convergence to the target Boltzmann distribution.

    Args:
        key: JAX PRNG key.
        energy_fn: Function that computes the energy of a state.
        shape: Shape of the initial state.
        hot_beta: Inverse temperature of the hot state (lower beta = hotter).
        target_beta: Target inverse temperature.
        num_optim_steps: Number of digital optimization steps.
        step_size: Step size for the digital optimization.

    Returns:
        The optimized initial state.
    """
    # Initialize from a hot thermal state
    x = jax.random.normal(key, shape) / jnp.sqrt(hot_beta)

    # Digitally optimize the state to create a non-equilibrium state
    # that cools faster (Mpemba effect)
    grad_fn = jax.grad(energy_fn)
    
    def step_fn(carry, _):
        x = carry
        grad = grad_fn(x)
        # Gradient descent step to precondition the state
        x_new = x - step_size * grad
        return x_new, None

    x_opt, _ = jax.lax.scan(step_fn, x, jnp.arange(num_optim_steps))
    return x_opt
