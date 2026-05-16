"""Tests for IGD sampler.
Spec: REQ-IGD-001, REQ-IGD-002, REQ-IGD-003, REQ-IGD-1961, SCENARIO-IGD-001, SCENARIO-IGD-1961
"""
import jax
import jax.numpy as jnp
from carnot.models.gibbs.igd_sampler import IGDSampler

def test_igd_sampler_smoke():
    # Simple MAX-3-SAT energy: 3 variables, 1 clause (x0 or x1 or x2)
    # Violated if all are 0. So energy = 1 if sum == 0 else 0
    def energy_fn(state):
        return jnp.where(jnp.sum(state) == 0, 1.0, 0.0)

    sampler = IGDSampler(energy_fn, num_vars=3, q=2)
    key = jax.random.PRNGKey(0)
    state = jnp.array([0, 0, 0], dtype=jnp.int32)
    logits = jnp.zeros(3)

    key, new_state, new_logits = sampler.sweep(key, state, logits, step_size=0.1)
    
    assert new_state.shape == (3,)
    assert new_logits.shape == (3,)
    # Output should still be valid binary
    assert jnp.all((new_state == 0) | (new_state == 1))

def test_igd_sampler_3sat_structure_generation():
    # Deterministic synthetic MAX-3-SAT instance with 3 literals per clause
    # x0 OR x1 OR ~x2
    # ~x0 OR x2 OR x3
    clauses = [
        [(0, 1), (1, 1), (2, 0)],
        [(0, 0), (2, 1), (3, 1)]
    ]
    
    def energy_fn(state):
        energy = 0.0
        for clause in clauses:
            violated = 1.0
            for var_idx, sign in clause:
                val = state[var_idx]
                # Using continuous-compatible logic:
                # if sign=1, literal is true if val=1. if sign=0, true if val=0.
                is_true = jnp.where(sign == 1, val, 1 - val)
                violated = violated * (1.0 - is_true)
            energy += violated
        return energy

    sampler = IGDSampler(energy_fn, num_vars=4, q=2)
    key = jax.random.PRNGKey(42)
    state = jnp.array([0, 0, 1, 0], dtype=jnp.int32) # violates first clause
    logits = jnp.zeros(4)

    for _ in range(5):
        key, state, logits = sampler.sweep(key, state, logits, step_size=0.5)

    assert state.shape == (4,)
    assert jnp.all((state >= 0) & (state <= 1))
