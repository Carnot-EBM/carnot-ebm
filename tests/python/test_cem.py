"""
Tests for Compositional Energy Minimizer.
Spec: SCENARIO-CEM-002
"""

import jax
import jax.numpy as jnp
import pytest
from carnot.models.ising import IsingModel, IsingConfig
from carnot.models.cem import CompositionalEnergyMinimizer

def test_cem_3sat_n16():
    """
    Test CEM on a 3-SAT toy problem (n=16) by mapping each clause to a sub-EBM.
    Spec: SCENARIO-CEM-002, REQ-CEM-004
    """
    n = 16
    key = jax.random.PRNGKey(42)
    
    # We create a planted solution
    planted_solution = jnp.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1])
    
    # Create 10 random 3-variable clauses that are satisfied by the planted solution
    clauses = []
    for _ in range(10):
        key, subkey = jax.random.split(key)
        # Pick 3 distinct variables
        vars_idx = jax.random.choice(subkey, n, shape=(3,), replace=False)
        
        config = IsingConfig(input_dim=n, coupling_init="zeros")
        model = IsingModel(config, key=key)
        
        J = jnp.zeros((n, n))
        b = jnp.zeros(n)
        for i in vars_idx:
            # Set bias to favor planted solution
            b = b.at[i].set(planted_solution[i] * 0.5)
            for j in vars_idx:
                if i != j:
                    # Set pairwise interaction
                    J = J.at[i, j].set(planted_solution[i] * planted_solution[j] * 0.5)
                    
        model.coupling = J
        model.bias = b
        clauses.append(model)
        
    cem = CompositionalEnergyMinimizer(sub_models=clauses, learning_rate=0.1)
    
    init_state = jnp.zeros(n)
    final_state, energy_history = cem.minimize(init_state, steps=100)
    
    # Energy should strictly decrease
    assert energy_history[-1] < energy_history[0]
    
    # Verify the global minimum matches the planted solution
    final_discrete = jnp.sign(final_state)
    final_discrete = jnp.where(final_discrete == 0, 1.0, final_discrete)
    assert jnp.all(final_discrete == planted_solution)
