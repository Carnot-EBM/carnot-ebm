"""Evaluates multi-step reasoning traces using EBT compatibility checking.

Spec: REQ-NRGPT-004, SCENARIO-NRGPT-004
"""

import jax
import jax.numpy as jnp
from carnot.models.ebt_compatibility import EBTCompatibilityModel


class EBTReasoningEvaluator:
    """Evaluates multi-step reasoning traces using EBT compatibility checking.
    
    Spec: REQ-NRGPT-004, SCENARIO-NRGPT-004
    """
    
    def __init__(self, ebt_model: EBTCompatibilityModel):
        self.ebt_model = ebt_model
        
    def evaluate_trace(self, input_seq: jax.Array, predicted_steps: list[jax.Array], truth_seq: jax.Array) -> dict:
        """Evaluates a multi-step reasoning trace.
        
        Args:
            input_seq: The input query/prompt sequence.
            predicted_steps: A list of partial generated sequences (reasoning trace).
            truth_seq: The human-verified ground truth sequence.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        partial_energies = []
        for step_seq in predicted_steps:
            energy = self.ebt_model.energy(input_seq, step_seq)
            partial_energies.append(float(energy))
            
        if not predicted_steps:
            final_energy = float('inf')
        else:
            final_energy = partial_energies[-1]
            
        truth_energy = float(self.ebt_model.energy(input_seq, truth_seq))
        
        return {
            "partial_energies": partial_energies,
            "final_energy": final_energy,
            "truth_energy": truth_energy,
            "compatibility_gap": final_energy - truth_energy,
        }

    def compute_distribution(self, input_seq: jax.Array, predicted_steps: list[jax.Array]) -> dict:
        """Computes statistical distribution over the partial trace energies.
        
        Args:
            input_seq: The input query/prompt sequence.
            predicted_steps: A list of partial generated sequences (reasoning trace).
            
        Returns:
            Dictionary with mean, min, max, and var of energies.
        """
        energies = [float(self.ebt_model.energy(input_seq, step)) for step in predicted_steps]
        
        if not energies:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "var": 0.0}
            
        arr = jnp.array(energies)
        return {
            "mean": float(jnp.mean(arr)),
            "min": float(jnp.min(arr)),
            "max": float(jnp.max(arr)),
            "var": float(jnp.var(arr)),
        }
