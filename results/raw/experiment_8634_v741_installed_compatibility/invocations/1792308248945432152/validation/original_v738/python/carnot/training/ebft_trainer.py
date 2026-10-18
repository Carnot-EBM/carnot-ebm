"""Energy-Based Fine-Tuning (EBFT) Trainer.

Spec: REQ-TRAIN-007
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from carnot.training.ebft_objective import ebft_loss
from carnot.models.latent_optimizer import LatentOptimizer
from carnot.pipeline.differentiable_memory import DifferentiableMemoryBank


class EBFTTrainer:
    """Trainer for Energy-Based Fine-Tuning with latent alignment and memory integration.
    
    Provides dense semantic feedback to the continuous self-learning loop by leveraging
    sequence-level feature matching (Jiang et al., 2026).
    
    Spec: REQ-TRAIN-007
    """

    def __init__(
        self,
        model: Any,
        optimizer: optax.GradientTransformation,
        latent_optimizer: Optional[LatentOptimizer] = None,
        memory_bank: Optional[DifferentiableMemoryBank] = None,
    ) -> None:
        """Initialize the EBFT Trainer.
        
        Args:
            model: The energy-based model (Flax module).
            optimizer: The Optax gradient transformation.
            latent_optimizer: Optional LatentOptimizer for aligning rollouts to expert states.
            memory_bank: Optional DifferentiableMemoryBank for long-term memory updates.
        """
        self.model = model
        self.optimizer = optimizer
        self.latent_optimizer = latent_optimizer
        self.memory_bank = memory_bank

    def create_train_state(self, key: jax.Array, input_shape: tuple[int, ...]) -> train_state.TrainState:
        """Initializes the train state.
        
        Args:
            key: JAX PRNG key.
            input_shape: Shape of the input sequences (without batch dim or with it, 
                         depending on model init).
                         
        Returns:
            A flax.training.train_state.TrainState instance.
        """
        dummy_input = jnp.ones(input_shape)
        params = self.model.init(key, dummy_input)['params']
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
        )

    def align_latents(
        self,
        state: train_state.TrainState,
        rollout_sequences: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """Aligns rollout sequences to lower-energy states using the LatentOptimizer.
        
        Args:
            state: Current training state.
            rollout_sequences: The sequences to align.
            key: JAX PRNG key for Langevin dynamics.
            
        Returns:
            Optimized sequences acting as pseudo-experts.
        """
        if self.latent_optimizer is None:
            raise ValueError("LatentOptimizer is required for latent alignment.")

        def energy_fn(seqs):
            # Sum energies across batch for scalar output if required by optimizer,
            # or rely on LatentOptimizer handling batching correctly (it uses vmap or similar,
            # but currently LatentOptimizer in context uses value_and_grad, so it needs a scalar).
            energies = state.apply_fn({'params': state.params}, seqs)
            return jnp.sum(energies)

        optimized_seqs, _ = self.latent_optimizer.optimize(rollout_sequences, energy_fn, key)
        return optimized_seqs

    def update_memory(self, rollout_sequences: jnp.ndarray, expert_sequences: jnp.ndarray) -> None:
        """Updates the differentiable memory bank with dense semantic feedback.
        
        Args:
            rollout_sequences: The generated rollouts.
            expert_sequences: The target expert sequences.
        """
        if self.memory_bank is None:
            return
            
        import numpy as np
        # Convert to numpy as DifferentiableMemoryBank uses numpy
        expert_np = np.array(expert_sequences)
        rollout_np = np.array(rollout_sequences)
        
        # Flatten batch dimension if necessary or update sequentially
        # Here we do a simple mean across batch to get a representative vector
        # Assuming shape is (batch, vector_dim)
        if rollout_np.ndim > 1:
            mean_rollout = np.mean(rollout_np, axis=0)
            mean_expert = np.mean(expert_np, axis=0)
            self.memory_bank.update(mean_rollout, mean_expert)
        else:
            self.memory_bank.update(rollout_np, expert_np)

    def train_step(
        self,
        state: train_state.TrainState,
        rollout_sequences: jnp.ndarray,
        expert_sequences: Optional[jnp.ndarray] = None,
        key: Optional[jax.Array] = None,
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        """Performs a single EBFT training step.
        
        Args:
            state: Current training state.
            rollout_sequences: Generated sequences to push energy up.
            expert_sequences: Ground truth sequences. If None, generated via align_latents.
            key: JAX PRNG key, required if expert_sequences is None.
            
        Returns:
            A tuple of (updated_state, loss_value).
        """
        if expert_sequences is None:
            if key is None:
                raise ValueError("PRNG key must be provided if expert_sequences is None.")
            expert_sequences = self.align_latents(state, rollout_sequences, key)

        def loss_fn(params):
            def model_energy_fn(sequences):
                return state.apply_fn({'params': params}, sequences)
            
            return ebft_loss(
                energy_fn=model_energy_fn,
                expert_sequences=expert_sequences,
                rollout_sequences=rollout_sequences,
            )
            
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        
        # Update memory bank to provide continuous feedback
        self.update_memory(rollout_sequences, expert_sequences)
        
        return state, loss
