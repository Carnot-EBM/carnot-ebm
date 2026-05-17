"""Continuous Self-Learning (CSL) loop."""
from typing import Any, Dict, Optional
import jax.numpy as jnp
from carnot.training.muon_ogd import MuonOGD
from carnot.training.adamflip import AdamFLIP

class CSLLoop:
    """Continuous Self-Learning Loop with Muon-OGD and AdamFLIP."""
    
    def __init__(self, optimizer: MuonOGD, grid_translation: Optional[jnp.ndarray] = None):
        self.optimizer = optimizer
        self.memory: Optional[jnp.ndarray] = None
        self.adamflip = AdamFLIP(learning_rate=0.01)
        self.grid_translation = grid_translation

    def step(self, params: jnp.ndarray, grads: jnp.ndarray, residuals: Optional[jnp.ndarray] = None, substrate_shift: Optional[jnp.ndarray] = None, prem_intrinsic_reward: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Perform a single CSL loop step with Muon-OGD and AdamFLIP."""
        updated_params = self.optimizer.update(params, grads, prior_memory=self.memory)
        
        if residuals is not None:
            feedback = self.adamflip.update(residuals)
            updated_params = updated_params - feedback
            
        shift = substrate_shift if substrate_shift is not None else self.grid_translation
        if shift is not None:
            updated_params = updated_params + shift

        if prem_intrinsic_reward is not None:
            updated_params = updated_params + prem_intrinsic_reward
            
        if self.memory is None:
            norm_val = jnp.linalg.norm(grads)
            if norm_val > 0:
                self.memory = grads / norm_val
            else:
                self.memory = grads
                
        return updated_params

def run_csl_loop(params: jnp.ndarray, grads: jnp.ndarray, residuals: Optional[jnp.ndarray] = None, substrate_shift: Optional[jnp.ndarray] = None, prem_intrinsic_reward: Optional[jnp.ndarray] = None) -> Dict[str, Any]:
    """Run a CSL loop and return the result."""
    optimizer = MuonOGD(learning_rate=0.01)
    csl = CSLLoop(optimizer, grid_translation=substrate_shift)
    
    # First step (populates memory)
    updated = csl.step(params, grads, residuals, substrate_shift=substrate_shift, prem_intrinsic_reward=prem_intrinsic_reward)
    
    # Second step (uses memory for OGD projection)
    updated_again = csl.step(updated, grads, residuals, substrate_shift=substrate_shift, prem_intrinsic_reward=prem_intrinsic_reward)
    
    return {
        "status": "success",
        "muon_ogd_applied": True,
        "adamflip_applied": residuals is not None,
        "substrate_shifting_applied": substrate_shift is not None,
        "prem_intrinsic_applied": prem_intrinsic_reward is not None,
        "updated_norm": float(jnp.linalg.norm(updated_again))
    }
