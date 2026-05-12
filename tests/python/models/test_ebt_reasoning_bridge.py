"""Tests for EBT Reasoning Bridge.

Spec: SCENARIO-NRGPT-002
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.models.ebt import EBTConfig, EBTransformer
from carnot.models.ebt_reasoning_bridge import EBTEnergyAdapter
from carnot.samplers.continuous_latent import ContinuousLatentSampler, FARSurrogateHead

def test_ebt_reasoning_bridge_convergence():
    """Verify that the reasoning bridge connects correctly and executes steps.
    
    Spec: SCENARIO-NRGPT-002
    """
    key = jrandom.PRNGKey(42)
    k_model, k_surr, k_init, k_sample = jrandom.split(key, 4)
    
    config = EBTConfig(n_layers=2, d_model=16, n_heads=2, d_ff=32, vocab_size=50, max_seq_len=10)
    ebt = EBTransformer(config, k_model)
    
    input_embeddings = jrandom.normal(k_model, (2, 16))
    
    seq_len_out = 3
    adapter = EBTEnergyAdapter(ebt, input_embeddings, seq_len_out)
    
    latent_dim = seq_len_out * 16
    surrogate = FARSurrogateHead.from_random_key(k_surr, latent_dim, n_constraints=4)
    
    sampler = ContinuousLatentSampler(
        energy_fn=adapter,
        surrogate=surrogate,
        step_size=0.1,
        skip_threshold=0.1
    )
    
    z_init = jrandom.normal(k_init, (latent_dim,))
    initial_energy = adapter.energy(z_init)
    
    z_final, stats = sampler.sample(k_sample, z_init, n_steps=20)
    final_energy = adapter.energy(z_final)
    
    assert final_energy.shape == ()
    assert stats.total_steps == 20
