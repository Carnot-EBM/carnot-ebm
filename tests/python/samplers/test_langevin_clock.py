import jax
import jax.numpy as jnp
from carnot.samplers.langevin_clock import langevin_clock_step

def test_langevin_clock_step_execution():
    """
    Test that langevin_clock_step executes and returns the correct shape.
    References:
    - REQ-SAMPLE-2605-12782-1
    - REQ-SAMPLE-2605-12782-2
    - REQ-SAMPLE-2605-12782-3
    - SCENARIO-SAMPLE-2605-12782
    """
    key = jax.random.PRNGKey(0)
    state = jnp.array([1.0, 2.0, 3.0])
    grad = jnp.array([0.1, 0.2, 0.3])
    step_size = 0.01
    force_scale = 1.5
    noise_scale = 0.5
    
    next_state = langevin_clock_step(state, grad, step_size, force_scale, noise_scale, key)
    
    assert next_state.shape == state.shape
    assert not jnp.allclose(next_state, state)
