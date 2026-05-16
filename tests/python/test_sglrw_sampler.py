"""Tests for the SGLRW sampler.

Spec: REQ-SAMPLE-2080
"""
import jax
import jax.numpy as jnp
from carnot.samplers.sglrw_sampler import SGLRWSampler

def test_sglrw_sampler_shapes():
    """Test that SGLRWSampler produces output of the correct shape and type."""
    sampler = SGLRWSampler(n_warmup=10, n_samples=5, steps_per_sample=2)
    key = jax.random.PRNGKey(0)
    biases = jnp.array([0.1, -0.2, 0.3])
    J = jnp.array([
        [0.0, 1.0, -1.0],
        [1.0, 0.0, 0.5],
        [-1.0, 0.5, 0.0]
    ])
    
    samples = sampler.sample(key, biases, J, beta=1.0)
    
    assert samples.shape == (5, 3)
    assert samples.dtype == jnp.bool_

def test_sglrw_sampler_with_init_spins():
    """Test that SGLRWSampler works with provided initial spins."""
    sampler = SGLRWSampler(n_warmup=1, n_samples=1, steps_per_sample=1)
    key = jax.random.PRNGKey(0)
    biases = jnp.array([0.0, 0.0])
    J = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    init_spins = jnp.array([True, False])
    
    samples = sampler.sample(key, biases, J, beta=1.0, init_spins=init_spins)
    assert samples.shape == (1, 2)

def test_sglrw_sampler_convergence():
    """Test that SGLRWSampler converges to expected states for a simple system."""
    sampler = SGLRWSampler(n_warmup=100, n_samples=100, steps_per_sample=5)
    key = jax.random.PRNGKey(42)
    
    # Simple ferromagnetic pair
    biases = jnp.array([0.0, 0.0])
    J = jnp.array([[0.0, 5.0], [5.0, 0.0]])
    
    samples = sampler.sample(key, biases, J, beta=5.0)
    
    # They should mostly agree
    agreements = (samples[:, 0] == samples[:, 1])
    assert jnp.mean(agreements) > 0.8
