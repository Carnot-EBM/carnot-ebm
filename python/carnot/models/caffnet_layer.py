"""
CAffNet Layer: Differentiable affine constraint projection layer.
"""
import jax
import jax.numpy as jnp

def project_affine(logits: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Project unconstrained logits onto the affine subspace Ax = b.
    
    Args:
        logits: Unconstrained input vector (1D).
        A: Constraint matrix (M x N).
        b: Constraint vector (M).
        
    Returns:
        Projected vector satisfying Ax = b.
    """
    # x_proj = x - A^T (A A^T)^-1 (A x - b)
    # To make it robust and efficient, we use jnp.linalg.solve or pseudo-inverse
    # For A A^T being invertible:
    A_A_T = A @ A.T
    
    # We solve (A A^T) y = A x - b  for y
    # Then x_proj = x - A^T y
    A_x_minus_b = A @ logits - b
    
    # Using lstsq for numerical stability in case A A^T is poorly conditioned
    # or jnp.linalg.solve if we assume full row rank.
    # We'll use solve for simplicity and assuming well-posed constraints.
    try:
        y = jnp.linalg.solve(A_A_T, A_x_minus_b)
    except Exception:
        # Fallback to pseudo-inverse
        y = jnp.linalg.lstsq(A_A_T, A_x_minus_b, rcond=None)[0]
        
    x_proj = logits - A.T @ y
    return x_proj

class CAffNetLayer:
    """
    Differentiable constraint layer that guarantees affine constraints.
    """
    def __init__(self, A: jnp.ndarray, b: jnp.ndarray):
        """
        Initialize the CAffNetLayer.
        
        Args:
            A: Constraint matrix of shape (M, N)
            b: Constraint vector of shape (M,)
        """
        self.A = jnp.array(A)
        self.b = jnp.array(b)
        
    def apply(self, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the projection to the input logits.
        """
        x_proj = project_affine(logits, self.A, self.b)
        return x_proj
