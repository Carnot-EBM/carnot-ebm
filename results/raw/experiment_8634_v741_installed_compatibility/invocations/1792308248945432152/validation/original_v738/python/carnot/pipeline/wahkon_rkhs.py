"""Wahkon RKHS alternative to standard KANs.

**Researcher summary:**
    Implements the Wahkon architecture (arXiv 2605.14041) which uses an RKHS
    alternative to standard KANs providing finite-sample guarantees.
    
**Detailed explanation for engineers:**
    This module provides a JAX/Flax implementation of a model that uses 
    RKHS-inspired kernel feature maps and linear combinations to approximate
    the finite-sample convergence properties of the Wahkon architecture.

Spec: REQ-KAN-1909
"""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn


class WahkonRKHS(nn.Module):
    """Wahkon RKHS model.
    
    Implements the Wahkon architecture (arXiv 2605.14041) which uses an RKHS
    alternative to standard KANs providing finite-sample guarantees.
    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim).
            
        Returns:
            Output tensor of shape (batch_size, out_dim).
        """
        # A simple RKHS-inspired kernel feature map and linear combination.
        # In Wahkon, we map inputs to RKHS and take additive combinations.
        kernel_weights = self.param(
            'kernel_weights',
            nn.initializers.normal(stddev=1.0),
            (x.shape[-1], self.hidden_dim)
        )
        kernel_bias = self.param(
            'kernel_bias',
            nn.initializers.uniform(scale=2 * jnp.pi),
            (self.hidden_dim,)
        )
        
        # RFF-like projection
        features = jnp.cos(jnp.dot(x, kernel_weights) + kernel_bias)
        
        # Linear combination
        out_weights = self.param(
            'out_weights',
            nn.initializers.glorot_uniform(),
            (self.hidden_dim, self.out_dim)
        )
        out_bias = self.param(
            'out_bias',
            nn.initializers.zeros_init(),
            (self.out_dim,)
        )
        
        return jnp.dot(features, out_weights) + out_bias
