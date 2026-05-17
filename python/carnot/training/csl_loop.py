"""Continuous Self-Learning (CSL) loop."""
from typing import Any, Dict, Optional
import jax.numpy as jnp
from carnot.training.muon_ogd import MuonOGD

class CSLLoop:
    """Continuous Self-Learning Loop with Muon-OGD."""
    
    def __init__(self, optimizer: MuonOGD):
        self.optimizer = optimizer
        self.memory: Optional[jnp.ndarray] = None

    def step(self, params: jnp.ndarray, grads: jnp.ndarray) -> jnp.ndarray:
        """Perform a single CSL loop step with Muon-OGD."""
        updated_params = self.optimizer.update(params, grads, prior_memory=self.memory)
        
        if self.memory is None:
            norm_val = jnp.linalg.norm(grads)
            if norm_val > 0:
                self.memory = grads / norm_val
            else:
                self.memory = grads
                
        return updated_params

def run_csl_loop(params: jnp.ndarray, grads: jnp.ndarray) -> Dict[str, Any]:
    """Run a CSL loop and return the result."""
    optimizer = MuonOGD(learning_rate=0.01)
    csl = CSLLoop(optimizer)
    
    # First step (populates memory)
    updated = csl.step(params, grads)
    
    # Second step (uses memory for OGD projection)
    updated_again = csl.step(updated, grads)
    
    return {
        "status": "success",
        "muon_ogd_applied": True,
        "updated_norm": float(jnp.linalg.norm(updated_again))
    }
