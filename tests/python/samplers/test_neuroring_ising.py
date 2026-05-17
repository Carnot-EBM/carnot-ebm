import json
from pathlib import Path
from carnot.samplers.parallel_ising import NeuroRingIsingSampler

def test_neuroring_ising_hardware_accounting():
    sampler = NeuroRingIsingSampler(ring_size=4)
    # Sudoku is 9x9x9 = 729 spins
    n_spins = 729
    total_sweeps = 1000
    
    metrics = sampler.hardware_accounting(n_spins=n_spins, total_sweeps=total_sweeps)
    
    assert metrics["n_spins"] == 729
    assert metrics["ring_size"] == 4
    assert metrics["total_sweeps"] == 1000
    assert metrics["latency_cycles_per_sweep"] == 2
    assert "total_bops" in metrics

def test_neuroring_ising_sample():
    import jax
    import jax.numpy as jnp
    sampler = NeuroRingIsingSampler(ring_size=4, n_warmup=10, n_samples=2, steps_per_sample=2)
    key = jax.random.PRNGKey(0)
    biases = jnp.zeros((10,), dtype=jnp.float32)
    coupling_matrix = jnp.zeros((10, 10), dtype=jnp.float32)
    
    samples = sampler.sample(key, biases, coupling_matrix, beta=1.0)
    assert samples.shape == (2, 10)
    assert samples.dtype == jnp.bool_
