import jax.numpy as jnp
from carnot.phase3.compositional_energy import CompositionalEnergy

def test_compositional_energy():
    """
    REQ-COMP-001: The composition system MUST take multiple callable energy potentials.
    REQ-COMP-002: The system MUST evaluate them in parallel over the latent state and sum them.
    SCENARIO-COMP-01: Two independent quadratic potentials are correctly summed.
    """
    def p1(x):
        return jnp.sum(x**2)
        
    def p2(x):
        return jnp.sum((x - 1.0)**2)
        
    potentials = [p1, p2]
    comp_energy = CompositionalEnergy(potentials)
    
    x = jnp.array([0.5, 0.5])
    
    # p1(x) = 0.5^2 + 0.5^2 = 0.25 + 0.25 = 0.5
    # p2(x) = (-0.5)^2 + (-0.5)^2 = 0.25 + 0.25 = 0.5
    # sum = 1.0
    
    total = comp_energy(x)
    assert jnp.allclose(total, 1.0)
    
    arr = comp_energy.evaluate_array(x)
    assert arr.shape == (2,)
    assert jnp.allclose(arr[0], 0.5)
    assert jnp.allclose(arr[1], 0.5)

def test_compositional_energy_empty():
    """
    SCENARIO-COMP-02: Empty potentials list returns 0.0 scalar or empty array.
    """
    comp_energy = CompositionalEnergy([])
    x = jnp.array([0.5, 0.5])
    total = comp_energy(x)
    assert jnp.allclose(total, 0.0)
    
    arr = comp_energy.evaluate_array(x)
    assert arr.shape == (0,)
