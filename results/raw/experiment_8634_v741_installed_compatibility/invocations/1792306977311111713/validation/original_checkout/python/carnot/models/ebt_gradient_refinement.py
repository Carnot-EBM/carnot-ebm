import jax
import jax.numpy as jnp
from typing import Callable

def gradient_refinement_loop(
    initial_state: jax.Array,
    energy_fn: Callable[[jax.Array], jax.Array],
    n_iters: int = 100,
    lr: float = 0.01
) -> tuple[jax.Array, list[float]]:
    """
    Run a minimal EBT-style gradient refinement loop (energy gradient descent).
    
    Args:
        initial_state: The starting continuous latent state.
        energy_fn: A function that computes scalar energy from the state.
        n_iters: Number of gradient descent steps.
        lr: Learning rate.
        
    Returns:
        final_state: The state after n_iters steps.
        energy_history: List of energies at each step (including initial).
    """
    grad_fn = jax.value_and_grad(energy_fn)
    state = initial_state
    
    energy_history = []
    
    for _ in range(n_iters):
        val, grad = grad_fn(state)
        energy_history.append(float(val))
        state = state - lr * grad
        
    # Append final energy
    final_val = float(energy_fn(state))
    energy_history.append(final_val)
    
    return state, energy_history
