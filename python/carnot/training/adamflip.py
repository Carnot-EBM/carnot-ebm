"""AdamFLIP optimizer for feedback linearization."""
import jax.numpy as jnp
from typing import Optional

class AdamFLIP:
    """AdamFLIP applies adaptive momentum feedback linearization to solve hard constraints robustly."""
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[jnp.ndarray] = None
        self.v: Optional[jnp.ndarray] = None
        self.t: int = 0
        
    def update(self, residuals: jnp.ndarray) -> jnp.ndarray:
        """Update using constraint residuals as feedback input."""
        if self.m is None:
            self.m = jnp.zeros_like(residuals)
            self.v = jnp.zeros_like(residuals)
            
        self.t += 1
        
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * residuals
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (residuals ** 2)
        
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        
        return self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
