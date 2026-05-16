"""Tests for DTM Simulator.

Spec: REQ-SAMPLE-2067
"""

import jax
import jax.numpy as jnp
import pytest

from carnot.samplers.dtm_sampler import DTMSampler


class DummyQuadraticEnergy:
    """A dummy energy function E(x) = 0.5 * sum(x^2)"""
    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(x**2)
        
    def grad_energy(self, x: jax.Array) -> jax.Array:
        return x


def test_dtm_sampler_convergence():
    """Test that DTMSampler reduces energy over time (SCENARIO-SAMPLE-2067)."""
    sampler = DTMSampler(step_size=0.05, clip_norm=10.0)
    energy_fn = DummyQuadraticEnergy()
    
    init_state = jnp.ones((10,)) * 5.0  # high energy state
    key = jax.random.PRNGKey(42)
    
    chain = sampler.sample_chain(energy_fn, init_state, n_steps=200, key=key)
    
    assert chain.shape == (200, 10)
    
    init_energy = energy_fn.energy(init_state)
    final_energy = energy_fn.energy(chain[-1])
    
    assert final_energy < init_energy, "Energy should decrease"
    assert not jnp.isnan(final_energy), "Final energy should not be NaN"

def test_dtm_sampler_sample():
    """Test sample() method of DTMSampler."""
    sampler = DTMSampler(step_size=0.01)
    energy_fn = DummyQuadraticEnergy()
    init_state = jnp.zeros((5,))
    key = jax.random.PRNGKey(0)
    
    final_state = sampler.sample(energy_fn, init_state, n_steps=10, key=key)
    assert final_state.shape == (5,)

def test_dtm_sampler_custom_beta_schedule():
    """Test DTMSampler with custom beta schedule."""
    sampler = DTMSampler(step_size=0.01)
    energy_fn = DummyQuadraticEnergy()
    init_state = jnp.zeros((5,))
    key = jax.random.PRNGKey(0)
    
    beta_schedule = jnp.ones(10) * 0.5
    final_state = sampler.sample(energy_fn, init_state, n_steps=10, key=key, beta_schedule=beta_schedule)
    assert final_state.shape == (5,)

def test_dtm_sampler_none_key():
    """Test DTMSampler with key=None."""
    sampler = DTMSampler(step_size=0.01)
    energy_fn = DummyQuadraticEnergy()
    init_state = jnp.zeros((5,))
    
    final_state = sampler.sample(energy_fn, init_state, n_steps=10, key=None)
    assert final_state.shape == (5,)
