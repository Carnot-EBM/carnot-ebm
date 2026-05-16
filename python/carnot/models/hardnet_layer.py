"""HardNet-style differentiable enforcement layer.

Spec references: REQ-HARDNET-2086, SCENARIO-HARDNET-2086.
"""

from typing import Union
import jax
import jax.numpy as jnp

class HardNetLayer:
    """HardNet-style differentiable enforcement layer for logical inequality constraints.
    
    This layer guarantees hard constraint satisfaction (bounds) in the forward pass
    while preserving gradients using a Straight-Through Estimator (STE). This allows
    the layer to be used effectively with gradient-based optimizers like those in Optax.
    """
    
    def __init__(
        self, 
        lower_bound: Union[float, jnp.ndarray] = -jnp.inf, 
        upper_bound: Union[float, jnp.ndarray] = jnp.inf
    ):
        """Initializes the HardNet layer with logical inequality constraints.
        
        Args:
            lower_bound: The minimum allowed value (feasible lower bound).
            upper_bound: The maximum allowed value (feasible upper bound).
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies the differentiable projection to the input tensor.
        
        Args:
            x: Continuous latent outputs to be clamped.
            
        Returns:
            The projected values clamped to the feasible logical bounds, 
            with gradients flowing through via STE.
        """
        # Forward pass: closed-form hard constraint satisfaction
        clipped = jnp.clip(x, self.lower_bound, self.upper_bound)
        
        # Straight-through estimator: f(x) = clipped in forward, but grad is passed as if f(x) = x
        return x + jax.lax.stop_gradient(clipped - x)
