import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class DeepSaDeLayer(nn.Module):
    """
    DeepSaDe-style guaranteed constraint layer.
    
    Implements a hybrid logic where SGD optimizes the general direction, 
    and a MaxSMT-like projection step guarantees that the final output
    strictly satisfies the domain constraints.
    """
    features: int
    lower_bound: float = -1.0
    upper_bound: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Applies the DeepSaDe constraint logic.
        """
        # REQ-DEEPSADE-001: Constraint Layer
        # REQ-DEEPSADE-002: Hybrid MaxSMT+SGD
        
        # 1. SGD optimized part
        h = nn.Dense(features=self.features)(x)
        
        # 2. MaxSMT simulated projection part.
        # In a real MaxSMT solver, this projects to the nearest SAT region.
        # Here we guarantee the domain constraints via clipping.
        h_projected = jnp.clip(h, self.lower_bound, self.upper_bound)
        
        # Calculate violation penalty of the unprojected output (for SGD guidance)
        violation_penalty = jnp.sum(jnp.maximum(0.0, h - self.upper_bound) + jnp.maximum(0.0, self.lower_bound - h), axis=-1)
        
        return h_projected, violation_penalty

def evaluate_satisfaction_rate(outputs: jnp.ndarray, lower: float, upper: float) -> float:
    """
    Evaluates the constraint satisfaction rate guarantees.
    REQ-DEEPSADE-003: Satisfaction Guarantees
    
    Returns the percentage of outputs that satisfy the bounds.
    """
    satisfies_lower = outputs >= (lower - 1e-5)
    satisfies_upper = outputs <= (upper + 1e-5)
    satisfies_both = jnp.logical_and(satisfies_lower, satisfies_upper)
    # The jnp.all acts on the last dimension if features > 1, but we evaluate on each element individually here or all features.
    # Let's say a sample satisfies if all its features satisfy.
    satisfies_sample = jnp.all(satisfies_both, axis=-1) if outputs.ndim > 1 else satisfies_both
    return float(jnp.mean(satisfies_sample))
