import jax
import jax.numpy as jnp
from carnot.phase3.pem_optimizer import PEMOptimizer
from carnot.phase3.compositional_energy import CompositionalEnergy

def test_pem_optimizer_convergence():
    """
    Test that the PEM optimizer converges on simple composed quadratic wells.
    REQ-PEM-001: Run gradient-based Langevin dynamics over composed landscapes.
    """
    # Two quadratic wells composed together
    # Well 1 minimum at x=2.0
    def well1(x):
        return jnp.sum((x - 2.0)**2)
        
    # Well 2 minimum at x=-2.0
    def well2(x):
        return jnp.sum((x + 2.0)**2)
        
    energy_fn = CompositionalEnergy([well1, well2])
    
    # Global minimum is at x=0.0
    
    optimizer = PEMOptimizer(
        energy_fn=energy_fn,
        learning_rate=0.05,
        noise_scale=0.0  # Zero noise for exact convergence check
    )
    
    x_init = jnp.array([5.0, 5.0])
    key = jax.random.PRNGKey(0)
    
    x_opt, final_key = optimizer.optimize(x_init, key, steps=200)
    
    # Should converge near 0
    assert jnp.allclose(x_opt, jnp.zeros_like(x_opt), atol=1e-2)

def test_pem_optimizer_step():
    """Test a single step of the PEM optimizer."""
    def well1(x):
        return jnp.sum((x - 2.0)**2)
        
    energy_fn = CompositionalEnergy([well1])
    optimizer = PEMOptimizer(
        energy_fn=energy_fn,
        learning_rate=0.01,
        noise_scale=0.1
    )
    
    x_init = jnp.array([0.0])
    key = jax.random.PRNGKey(0)
    
    x_new, energy, new_key = optimizer.step(x_init, key)
    
    # Check that x_new shape is correct
    assert x_new.shape == x_init.shape
    assert energy.shape == ()
    assert not jnp.array_equal(key, new_key)
