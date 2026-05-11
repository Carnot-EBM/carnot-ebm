"""Tests for continuous latent constraint optimizer.

Spec: REQ-OPT-1771, SCENARIO-OPT-1771
"""

import jax
import jax.numpy as jnp
from carnot.models.latent_optimizer import LatentOptimizer
from carnot.models.hrm_verifier import HRMVerifier

def test_latent_optimizer_initialization():
    """REQ-OPT-1771-1: Initialize with step size, noise scale, and max steps."""
    optimizer = LatentOptimizer(step_size=0.05, noise_scale=0.02, max_steps=50)
    assert optimizer.step_size == 0.05
    assert optimizer.noise_scale == 0.02
    assert optimizer.max_steps == 50

def test_latent_optimizer_energy_descent():
    """REQ-OPT-1771-3: Verify energy descent after optimization.
    
    SCENARIO-OPT-1771: Langevin Dynamics on HRM.
    """
    optimizer = LatentOptimizer(step_size=0.1, noise_scale=0.0, max_steps=100)
    
    # Define a simple quadratic energy function to simulate a continuous relaxation of constraints.
    # The minimum is at z = 2.0
    def energy_fn(z):
        return jnp.sum((z - 2.0) ** 2)

    key = jax.random.PRNGKey(0)
    z_init = jnp.array([0.0, 0.0])
    
    z_opt, energies = optimizer.optimize(z_init, energy_fn, key)
    
    # Energy should descend
    assert energies[-1] < energies[0]
    
    # Since noise is 0, we should converge close to the minimum [2.0, 2.0]
    assert jnp.allclose(z_opt, jnp.array([2.0, 2.0]), atol=0.1)

def test_latent_optimizer_with_noise():
    """REQ-OPT-1771-2: Optimize continuous latent constraints by performing Langevin dynamics."""
    optimizer = LatentOptimizer(step_size=0.05, noise_scale=0.1, max_steps=200)
    
    def energy_fn(z):
        # A simple potential well
        return jnp.sum(z ** 2)

    key = jax.random.PRNGKey(42)
    z_init = jnp.array([5.0, -5.0])
    
    z_opt, energies = optimizer.optimize(z_init, energy_fn, key)
    
    # Energy should descend overall, despite the noise
    assert energies[-1] < energies[0]
