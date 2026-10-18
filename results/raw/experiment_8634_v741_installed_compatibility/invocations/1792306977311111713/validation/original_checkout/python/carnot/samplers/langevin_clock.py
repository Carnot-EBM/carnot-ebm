import jax
import jax.numpy as jnp

def langevin_clock_step(
    state: jax.Array,
    grad: jax.Array,
    step_size: float,
    force_scale: float,
    noise_scale: float,
    key: jax.Array
) -> jax.Array:
    """
    Computes a modified Langevin step by scaling the deterministic forces and 
    adding a specific noise structure. (arXiv:2605.12782).
    
    Args:
        state: The current state array.
        grad: The gradient of the energy function at the current state.
        step_size: The base step size for the update.
        force_scale: Factor by which to scale the deterministic force.
        noise_scale: Factor by which to scale the random noise.
        key: JAX PRNG key for generating noise.
        
    Returns:
        The updated state array.
    """
    noise = jax.random.normal(key, shape=state.shape)
    
    deterministic_update = - (step_size * force_scale) * grad
    stochastic_update = noise_scale * jnp.sqrt(2.0 * step_size) * noise
    
    return state + deterministic_update + stochastic_update
