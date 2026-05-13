"""Dummy proxy for extracting continuous latent scoring metadata.

Spec: REQ-VERIFY-001
"""

from typing import Any, Dict
import jax
import jax.numpy as jnp
from carnot.verify.constraint import ConstraintTerm

class DummyEnergyExtractionProxy:
    """A proxy to yield metadata for Glauber/Diffusion loops from validators."""
    
    def __init__(self, validator: ConstraintTerm):
        self.validator = validator
        
    def extract_metadata(self, x: jax.Array) -> Dict[str, Any]:
        """Extract metadata compatible with Glauber and Diffusion loops."""
        energy_val = float(self.validator.energy(x))
        grad = self.validator.grad_energy(x)
        grad_norm = float(jnp.linalg.norm(grad))
        
        return {
            "glauber_compatible": True,
            "diffusion_compatible": True,
            "validator_name": self.validator.name,
            "energy_val": energy_val,
            "grad_norm": grad_norm,
            "continuous_latent_scoring_ready": True,
            "automata_metadata": {},
            "validator_metadata": {},
            "generator_integration_claim": False,
        }
