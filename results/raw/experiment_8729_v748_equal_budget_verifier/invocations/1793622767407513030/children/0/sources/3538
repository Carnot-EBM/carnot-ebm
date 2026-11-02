"""EBT Reasoning Bridge — connects continuous latent sampler with EBT energy.

Spec: REQ-NRGPT-002
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from carnot.core.energy import AutoGradMixin

if TYPE_CHECKING:
    from carnot.models.ebt import EBTransformer

class EBTEnergyAdapter(AutoGradMixin):
    """Adapts EBTransformer continuous energy to the EnergyFunction protocol.
    
    The ContinuousLatentSampler expects a 1-D vector z. This adapter takes z,
    reshapes it to (seq_len_out, d_model), concatenates it with fixed input
    embeddings, and evaluates the EBT energy.
    
    Spec: REQ-NRGPT-002
    """
    
    def __init__(self, ebt: "EBTransformer", input_embeddings: jax.Array, seq_len_out: int):
        self.ebt = ebt
        self.input_embeddings = input_embeddings
        self.seq_len_out = seq_len_out
        self.d_model = ebt.config.d_model
        
    def energy(self, z: jax.Array) -> jax.Array:
        """Compute energy for continuous latent state z.
        
        Args:
            z: A 1-D JAX array of shape (seq_len_out * d_model,).
            
        Returns:
            Scalar energy.
        """
        cand_embeddings = z.reshape((self.seq_len_out, self.d_model))
        h = jnp.concatenate([self.input_embeddings, cand_embeddings], axis=0)
        return self.ebt.energy_from_embeddings(h)
