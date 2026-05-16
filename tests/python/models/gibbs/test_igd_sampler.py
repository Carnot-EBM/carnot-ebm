"""Tests for IGDSampler.

Spec: REQ-IGD-001, REQ-IGD-002, REQ-IGD-003, REQ-IGD-1961, SCENARIO-IGD-001
"""

import jax
import jax.numpy as jnp
from carnot.models.gibbs.igd_sampler import IGDSampler

def test_igd_sampler_sweep():
    # Mock 3-SAT style energy: favor all variables being 1
    def mock_3sat_energy(state):
        return jnp.sum(1 - state) * 1.0

    num_vars = 5
    q = 2
    sampler = IGDSampler(energy_fn=mock_3sat_energy, num_vars=num_vars, q=q)
    
    key = jax.random.PRNGKey(0)
    state = jnp.zeros(num_vars, dtype=jnp.int32)
    logits = jnp.zeros((num_vars, q), dtype=jnp.float32)
    
    next_state, next_logits = sampler.sweep(key, state, logits, step_size=0.1)
    
    assert next_state.shape == (num_vars,)
    assert next_logits.shape == (num_vars, q)
    assert not jnp.array_equal(logits, next_logits)
    assert next_state.dtype == jnp.int32
