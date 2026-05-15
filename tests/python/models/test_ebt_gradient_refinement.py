import jax
import jax.numpy as jnp
from carnot.models.ebt_gradient_refinement import gradient_refinement_loop

def test_gradient_refinement_loop():
    """Test the EBT-style gradient refinement loop."""
    # REQ-EBT-1742-1
    # Simple quadratic bowl energy function: E(x) = sum(x^2)
    def energy_fn(x):
        return jnp.sum(x**2)
    
    initial_state = jnp.array([1.0, -2.0, 3.0])
    
    # Run the loop
    final_state, energy_history = gradient_refinement_loop(
        initial_state, energy_fn, n_iters=10, lr=0.1
    )
    
    assert len(energy_history) == 11
    
    # Energy should strictly decrease
    for i in range(len(energy_history) - 1):
        assert energy_history[i+1] < energy_history[i]
        
    # The minimum should be closer to 0
    assert energy_history[-1] < energy_history[0]
    
    # Verify final state is updated
    assert not jnp.allclose(final_state, initial_state)
