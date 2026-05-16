"""Tests for CEM and ClauseEBM on a 3-SAT toy problem.

Spec: REQ-CEM-005, SCENARIO-CEM-003
"""

import jax.numpy as jnp
from carnot.models.cem import ClauseEBM, CompositionalEnergyMinimizer

def test_clause_ebm_satisfied():
    # REQ-CEM-005
    clause = ClauseEBM([0, 1, 2], [1, -1, 1])
    # x0=1, x1=-1, x2=-1
    # Satisfied by x0=1 and x1=-1
    state = jnp.array([1.0, -1.0, -1.0])
    energy = clause.energy(state)
    assert jnp.allclose(energy, 0.0)

def test_clause_ebm_unsatisfied():
    # REQ-CEM-005
    clause = ClauseEBM([0, 1, 2], [1, 1, 1])
    # All wrong signs
    state = jnp.array([-1.0, -1.0, -1.0])
    energy = clause.energy(state)
    # (1 - (1*-1)) * (1 - (1*-1)) * (1 - (1*-1)) = 2 * 2 * 2 = 8
    assert jnp.allclose(energy, 8.0)

def test_cem_3sat_composition():
    # SCENARIO-CEM-003
    c1 = ClauseEBM([0, 1, 2], [1, 1, 1])
    c2 = ClauseEBM([0, 1, 3], [1, -1, -1])
    
    cem = CompositionalEnergyMinimizer([c1, c2], learning_rate=0.1)
    
    # State satisfying both: x0=1
    state_sat = jnp.array([1.0, -1.0, 1.0, -1.0])
    e_sat = cem.compute_total_energy(state_sat)
    assert jnp.allclose(e_sat, 0.0)
    
    # Minimize from bad state
    state_bad = jnp.array([-0.5, -0.5, -0.5, 0.5])
    e_bad = cem.compute_total_energy(state_bad)
    assert e_bad > 0.0
    
    final_state, history = cem.minimize(state_bad, steps=50)
    
    # Since it's a simple problem, it should decrease energy
    assert history[-1] < history[0]
