"""
EBCN (Energy-Based Coherence Network) Prototype.

Implements a dual-head attention state-space model to score the coherence of
reasoning traces and detect logical contradictions via a scalar energy score.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

class EBCNCoherenceModel(nn.Module):
    """
    Dual-head attention state-space model for coherence scoring.
    """
    hidden_dim: int = 128
    num_heads: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, padding_mask: jnp.ndarray = None) -> jnp.ndarray:
        """
        Args:
            x: Input sequence embeddings of shape (batch, seq_len, embed_dim)
            padding_mask: Boolean mask of shape (batch, seq_len), True for valid tokens

        Returns:
            Scalar energy score per sequence: (batch,)
        """
        # Linear projection to hidden dim
        x = nn.Dense(self.hidden_dim)(x)
        
        # Dual-head attention for relational reasoning
        attn_mask = None
        if padding_mask is not None:
            attn_mask = nn.make_attention_mask(padding_mask, padding_mask)

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.hidden_dim
        )(x, x, mask=attn_mask)
        
        # Residual + LayerNorm
        x = nn.LayerNorm()(x + attn_out)

        # State-space like pooling (simplified global average pooling over valid tokens)
        if padding_mask is not None:
            mask_expanded = jnp.expand_dims(padding_mask, -1)
            sum_x = jnp.sum(x * mask_expanded, axis=1)
            count = jnp.maximum(jnp.sum(padding_mask, axis=1, keepdims=True), 1.0)
            pooled = sum_x / count
        else:
            pooled = jnp.mean(x, axis=1)

        # Energy projection head
        # We output a scalar energy (higher energy = less coherent/contradiction)
        energy = nn.Dense(1)(pooled)
        
        return jnp.squeeze(energy, axis=-1)

def evaluate_contradictions(model: EBCNCoherenceModel, params: dict, coherent_trace: jnp.ndarray, contradictory_trace: jnp.ndarray) -> dict:
    """
    Evaluates the model on coherent vs contradictory traces.
    """
    energy_coherent = model.apply({'params': params}, coherent_trace)
    energy_contradictory = model.apply({'params': params}, contradictory_trace)
    
    return {
        "coherent_energy": float(jnp.mean(energy_coherent)),
        "contradictory_energy": float(jnp.mean(energy_contradictory)),
        "detects_contradiction": float(jnp.mean(energy_contradictory)) > float(jnp.mean(energy_coherent))
    }
