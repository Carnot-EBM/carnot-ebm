"""Tests for the Process-Reward Energy Model (PREM)."""

import jax
import jax.numpy as jnp
import pytest

from carnot.models.prem import PREMConfig, PREMModel

def test_prem_config_validation():
    """Test PREM configuration inheritance and validation."""
    config = PREMConfig(input_dim=10, hidden_dims=[8, 4])
    config.validate()
    assert config.input_dim == 10
    assert config.hidden_dims == [8, 4]
    
    with pytest.raises(ValueError):
        PREMConfig(input_dim=-1).validate()

def test_prem_step_energy():
    """Test evaluating a single step's energy.
    
    Spec: REQ-PREM-002, SCENARIO-PREM-001
    """
    key = jax.random.PRNGKey(0)
    config = PREMConfig(input_dim=5, hidden_dims=[4])
    model = PREMModel(config, key)
    
    x_step = jnp.ones(5)
    energy = model.step_energy(x_step)
    
    assert energy.shape == ()
    assert not jnp.isnan(energy)

def test_prem_sequence_energy():
    """Test evaluating a sequence of steps.
    
    Spec: REQ-PREM-003, SCENARIO-PREM-002
    """
    key = jax.random.PRNGKey(0)
    config = PREMConfig(input_dim=5, hidden_dims=[4])
    model = PREMModel(config, key)
    
    seq_len = 3
    x_seq = jnp.ones((seq_len, 5))
    energies = model.sequence_energy(x_seq)
    
    assert energies.shape == (seq_len,)
    assert not jnp.any(jnp.isnan(energies))

def test_prem_process_reward():
    """Test computing the total process reward for a sequence."""
    key = jax.random.PRNGKey(0)
    config = PREMConfig(input_dim=5, hidden_dims=[4])
    model = PREMModel(config, key)
    
    seq_len = 3
    x_seq = jnp.ones((seq_len, 5))
    reward = model.process_reward(x_seq)
    
    assert reward.shape == ()
    assert not jnp.isnan(reward)
