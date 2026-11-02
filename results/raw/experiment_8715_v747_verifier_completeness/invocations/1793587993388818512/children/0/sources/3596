"""Process-Reward Energy Model (PREM).

This module implements the PREM architecture, designed to evaluate the energy
of intermediate steps in a generation or reasoning process. It subclasses
standard Energy-Based Models (like GibbsModel) and provides methods for
step-wise energy computation.

Spec: REQ-PREM-001, REQ-PREM-002, REQ-PREM-003
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from carnot.models.gibbs import GibbsModel, GibbsConfig

@dataclass
class PREMConfig(GibbsConfig):
    """Configuration for the Process-Reward Energy Model.
    
    Inherits from GibbsConfig, representing the underlying energy network
    architecture used to score each step in the process.
    """
    pass

class PREMModel(GibbsModel):
    """Process-Reward Energy Model based on a Gibbs network.
    
    This model evaluates the energy of individual steps within a trajectory
    or sequence. Lower energy indicates higher process reward (more desirable
    intermediate state).
    
    Spec: REQ-PREM-002
    """
    
    def __init__(self, config: PREMConfig, key: jax.Array | None = None) -> None:
        """Create a PREM model.
        
        Args:
            config: Model architecture configuration.
            key: JAX PRNG key for weight initialization.
        """
        super().__init__(config, key)

    def step_energy(self, x_step: jax.Array) -> jax.Array:
        """Compute the scalar energy for a single intermediate step.
        
        Args:
            x_step: A 1-D JAX array of shape (input_dim,) representing the state
                    at a specific process step.
                    
        Returns:
            A scalar JAX array representing the step's energy.
            
        Spec: REQ-PREM-002, SCENARIO-PREM-001
        """
        return self.energy(x_step)
        
    def sequence_energy(self, x_seq: jax.Array) -> jax.Array:
        """Compute step-wise energies for an entire sequence of steps.
        
        Args:
            x_seq: A 2-D JAX array of shape (seq_len, input_dim).
            
        Returns:
            A 1-D JAX array of shape (seq_len,) containing the energy of each step.
            
        Spec: REQ-PREM-003, SCENARIO-PREM-002
        """
        # Vectorize step_energy over the sequence dimension
        return jax.vmap(self.step_energy)(x_seq)

    def process_reward(self, x_seq: jax.Array) -> jax.Array:
        """Compute the total process reward (negative total energy) of a sequence.
        
        Args:
            x_seq: A 2-D JAX array of shape (seq_len, input_dim).
            
        Returns:
            A scalar JAX array representing the cumulative reward.
        """
        energies = self.sequence_energy(x_seq)
        return -jnp.sum(energies)
