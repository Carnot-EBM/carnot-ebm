import jax
import jax.numpy as jnp
from typing import Optional

def newton_schulz_matrix_sign(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    """
    Computes the matrix sign function of a 2D matrix G using Newton-Schulz iteration.
    This approximates the orthogonal projection.
    """
    if G.ndim != 2:
        raise ValueError("G must be a 2D matrix")
        
    X = G / (jnp.linalg.norm(G, ord='fro') + 1e-8)
    
    # Standard Newton-Schulz iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k * X_k^T * X_k
    for _ in range(steps):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
        
    return X

class MuonOGD:
    """Muon-OGD Spectral Orthogonal Gradient Projection."""
    
    def __init__(self, learning_rate: float = 0.01, ns_steps: int = 5):
        self.learning_rate = learning_rate
        self.ns_steps = ns_steps

    def update(self, params: jnp.ndarray, grads: jnp.ndarray, prior_memory: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Updates parameters using Muon-OGD. If prior_memory is provided,
        projects the gradient orthogonally to previously learned knowledge.
        """
        orig_shape = grads.shape
        if grads.ndim != 2:
            grads_2d = grads.reshape(grads.shape[0], -1)
        else:
            grads_2d = grads
            
        orthogonalized_grad = newton_schulz_matrix_sign(grads_2d, self.ns_steps)
        
        if prior_memory is not None:
            # OGD: project orthogonally to prior memory
            if prior_memory.ndim != 2:
                prior_memory_2d = prior_memory.reshape(prior_memory.shape[0], -1)
            else:
                prior_memory_2d = prior_memory
                
            proj = orthogonalized_grad @ prior_memory_2d.T @ prior_memory_2d
            orthogonalized_grad = orthogonalized_grad - proj

        updated_params = params.reshape(grads_2d.shape) - self.learning_rate * orthogonalized_grad
        return updated_params.reshape(orig_shape)
