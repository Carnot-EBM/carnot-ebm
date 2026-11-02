"""Energy-Based Reward Model (EBRM) Trainer.

Spec: REQ-LEARN-2066
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

class EBRMMLP(nn.Module):
    """A small JAX MLP to act as the Energy Reward Model."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class EBRMTrainer:
    """Trainer for Energy-Based Reward Model using Contrastive Divergence.
    
    Spec: REQ-LEARN-2066
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
    ) -> None:
        """Initialize the EBRM Trainer.
        
        Args:
            model: The energy reward model (Flax module).
            optimizer: The Optax gradient transformation.
        """
        self.model = model
        self.optimizer = optimizer

    def create_train_state(self, key: jax.Array, input_shape: tuple[int, ...]) -> train_state.TrainState:
        """Initializes the train state.
        
        Args:
            key: JAX PRNG key.
            input_shape: Shape of the input traces.
                         
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

    def train_step(
        self,
        state: train_state.TrainState,
        positive_traces: jnp.ndarray,
        negative_traces: jnp.ndarray,
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        """Performs a single Contrastive Divergence training step.
        
        Args:
            state: Current training state.
            positive_traces: Ground truth or high-quality logical traces.
            negative_traces: Synthetic or generated logical traces (model samples).
            
        Returns:
            A tuple of (updated_state, loss_value).
        """
        def loss_fn(params):
            pos_energy = state.apply_fn({'params': params}, positive_traces)
            neg_energy = state.apply_fn({'params': params}, negative_traces)
            
            # Contrastive divergence loss: E(pos) - E(neg)
            # We want to minimize pos_energy and maximize neg_energy.
            # Assuming energy is lower for better traces.
            loss = jnp.mean(pos_energy) - jnp.mean(neg_energy)
            return loss
            
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss

    def train(
        self,
        state: train_state.TrainState,
        positive_traces: jnp.ndarray,
        negative_traces: jnp.ndarray,
        epochs: int = 1,
    ) -> tuple[train_state.TrainState, list[float]]:
        """Trains the energy model using Contrastive Divergence over multiple epochs.
        
        Args:
            state: Current training state.
            positive_traces: Ground truth traces.
            negative_traces: Synthetic traces.
            epochs: Number of epochs to train.
            
        Returns:
            A tuple of (updated_state, list of loss values per epoch).
        """
        losses = []
        for _ in range(epochs):
            state, loss = self.train_step(state, positive_traces, negative_traces)
            losses.append(float(loss))
        return state, losses
