import jax.numpy as jnp
from carnot.models.compositional_energy import CompositionalEnergyMinimizer

def test_compositional_energy_minimizer():
    # REQ-VERIFY-2055
    # SCENARIO-VERIFY-2055
    
    # 10-step pathfinding problem
    def boundary_start(x):
        return (x[0] - 0.0)**2
        
    def boundary_end(x):
        return (x[9] - 9.0)**2
        
    def path_step(i):
        def _step(x):
            return (x[i] - x[i-1] - 1.0)**2
        return _step
        
    sub_energies = [boundary_start, boundary_end] + [path_step(i) for i in range(1, 10)]
    
    minimizer = CompositionalEnergyMinimizer(sub_energies, learning_rate=0.1)
    init_state = jnp.zeros(10)
    
    final_state, energy_history = minimizer.minimize(init_state, steps=500)
    
    assert len(energy_history) == 500
    # Final energy should be close to 0
    assert energy_history[-1] < 1.0
    
    # Final state should be close to [0, 1, 2, ..., 9]
    expected_state = jnp.arange(10, dtype=jnp.float32)
    assert jnp.allclose(final_state, expected_state, atol=1.0)
