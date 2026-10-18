import jax.numpy as jnp

def extract_eqm_gradient(hidden_states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extracts an EqM energy landscape gradient from FAR continuous hidden states.
    For the toy constraints, we map the state to a simple quadratic bowl energy.
    
    Args:
        hidden_states: Continuous hidden states, shape (batch_size, dim).
        
    Returns:
        A tuple of (energies, gradients).
    """
    # Toy constraint: encourage hidden states to be near zero (quadratic bowl)
    energies = jnp.sum(hidden_states ** 2, axis=-1)
    gradients = 2.0 * hidden_states
    
    return energies, gradients
