"""
EBTLayer: Energy-Based Transformer computation layer.
Bridges MoE outputs to PyO3 safetensors.
"""
import jax
import jax.numpy as jnp

class EBTLayer:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        # Scaffold weights for assigning energy values directly to input-prediction pairs
        self.weights = jnp.zeros((hidden_dim, 1))

    def compute_energy(self, hidden_states: jax.Array) -> jax.Array:
        """
        Assigns energy values directly to input-prediction pairs based on hidden states.
        
        Args:
            hidden_states: Array of shape (batch, seq_len, hidden_dim) from MoE outputs.
        Returns:
            energy_values: Array of shape (batch, seq_len, 1).
        """
        return jnp.dot(hidden_states, self.weights)
