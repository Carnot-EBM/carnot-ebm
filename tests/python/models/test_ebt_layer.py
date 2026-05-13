"""Tests for EBTLayer MoE Bridge.

Spec: SCENARIO-NRGPT-005
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from carnot.models.ebt_layer import EBTLayer

def test_ebt_layer_compute_energy():
    """Verify that EBTLayer computes energy from MoE outputs.
    
    Spec: SCENARIO-NRGPT-005
    """
    hidden_dim = 16
    layer = EBTLayer(hidden_dim)
    
    k_input = jrandom.PRNGKey(42)
    # Shape: (batch_size, sequence_length, hidden_dim)
    hidden_states = jrandom.normal(k_input, (2, 5, hidden_dim))
    
    energy_values = layer.compute_energy(hidden_states)
    
    assert energy_values.shape == (2, 5, 1)
