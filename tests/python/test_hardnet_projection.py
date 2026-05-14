import jax.numpy as jnp
from carnot.solvers.hardnet_projection import damped_local_linearization

def test_hardnet_projection_basic():
    # REQ-HARDNET-001
    # SCENARIO-HARDNET-001
    
    def g_fn(val):
        return jnp.sum(jnp.square(val)) - 1.0
        
    # Case 1: Outside unit circle
    x_init = jnp.array([2.0, 2.0])
    x_proj = damped_local_linearization(x_init, g_fn, damping=0.5, max_iter=100)
    assert g_fn(x_proj) <= 1e-3
    
    # Case 2: Inside unit circle (no projection needed)
    x_init_in = jnp.array([0.5, 0.5])
    x_proj_in = damped_local_linearization(x_init_in, g_fn, damping=0.5, max_iter=100)
    # Should not move much
    assert jnp.allclose(x_init_in, x_proj_in, atol=1e-3)
