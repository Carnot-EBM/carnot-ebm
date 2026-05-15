"""Tests for h_schedule in parallel Ising sampler."""

import jax
import jax.numpy as jnp
import numpy as np

from carnot.samplers.parallel_ising import ParallelIsingSampler
from carnot.samplers.backend import CpuBackend

# REQ-SAMPLE-003

def test_h_schedule_in_sampler():
    sampler = ParallelIsingSampler(n_warmup=100, n_samples=10, steps_per_sample=1)
    n_spins = 10
    biases = jnp.zeros((n_spins,))
    coupling_matrix = jnp.zeros((n_spins, n_spins))
    key = jax.random.PRNGKey(42)
    
    # Run with h_schedule=1
    samples1 = sampler.sample(key, biases, coupling_matrix, beta=1.0, h_schedule=1)
    assert samples1.shape == (10, n_spins)
    
    # Run with h_schedule=2
    samples2 = sampler.sample(key, biases, coupling_matrix, beta=1.0, h_schedule=2)
    assert samples2.shape == (10, n_spins)

def test_h_schedule_in_backend():
    backend = CpuBackend(seed=42)
    n_spins = 10
    biases = np.zeros((n_spins,))
    couplings = np.zeros((n_spins, n_spins))
    
    config = {
        "beta": 1.0,
        "n_warmup": 100,
        "steps_per_sample": 1,
        "h_schedule": 1
    }
    
    samples = backend.sample(biases, couplings, 10, config)
    assert samples.shape == (10, n_spins)
