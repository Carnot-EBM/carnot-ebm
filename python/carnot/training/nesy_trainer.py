"""Trainer for Neuro-Symbolic Energy-Based Models using Alternating MAP Inference.

Spec: REQ-SYMKAN-2075
"""

from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state


class NeSyTrainer:
    """Trainer for Neuro-Symbolic Energy-Based Models using Alternating MAP Inference.
    
    Spec: REQ-SYMKAN-2075
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optax.GradientTransformation,
        energy_fns: List[Callable[[jnp.ndarray], jnp.ndarray]],
        map_steps: int = 10,
        map_step_size: float = 0.01,
    ) -> None:
        """Initialize the NeSyTrainer.
        
        Args:
            model: Flax module predicting latents.
            optimizer: Optax optimizer.
            energy_fns: List of energy functions (e.g. from NeSyEncoder).
            map_steps: Number of MAP inference gradient steps.
            map_step_size: Step size for MAP inference.
        """
        self.model = model
        self.optimizer = optimizer
        self.energy_fns = energy_fns
        self.map_steps = map_steps
        self.map_step_size = map_step_size

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

    def total_energy(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Computes the total symbolic energy for the given latents.
        
        Args:
            latents: The predicted latents.
            
        Returns:
            Scalar total energy.
        """
        total = 0.0
        for e_fn in self.energy_fns:
            total += jnp.sum(e_fn(latents))
        return total

    def map_inference(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Refines latents by taking gradient steps to minimize total energy.
        
        Args:
            latents: Initial latents from the neural model.
            
        Returns:
            Refined latents after MAP inference.
        """
        def loss_fn(l: jnp.ndarray) -> jnp.ndarray:
            return self.total_energy(l)
            
        grad_fn = jax.grad(loss_fn)
        
        def body_fn(i: int, val: jnp.ndarray) -> jnp.ndarray:
            return val - self.map_step_size * grad_fn(val)
            
        refined = jax.lax.fori_loop(0, self.map_steps, body_fn, latents)
        return refined

    def train_step(
        self,
        state: train_state.TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, jnp.ndarray]:
        """Performs one step of alternating MAP inference training.
        
        Args:
            state: Current training state.
            inputs: Input features.
            targets: Target outputs.
            
        Returns:
            Updated training state and loss.
        """
        # Forward pass to get latents
        latents = state.apply_fn({'params': state.params}, inputs)
        
        # Project onto symbolic constraints using MAP inference
        refined_latents = self.map_inference(latents)
        
        # Stop gradients to use refined latents as targets
        refined_latents = jax.lax.stop_gradient(refined_latents)
        
        def loss_fn(params: dict) -> jnp.ndarray:
            preds = state.apply_fn({'params': params}, inputs)
            task_loss = jnp.mean((preds - targets) ** 2)
            prior_loss = jnp.mean((preds - refined_latents) ** 2)
            return task_loss + prior_loss
            
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss

    def train(
        self,
        state: train_state.TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        epochs: int = 1,
    ) -> Tuple[train_state.TrainState, List[float]]:
        """Trains the model over multiple epochs.
        
        Args:
            state: Current training state.
            inputs: Input features.
            targets: Target outputs.
            epochs: Number of epochs to train.
            
        Returns:
            Tuple of updated state and list of losses.
        """
        losses = []
        for _ in range(epochs):
            state, loss = self.train_step(state, inputs, targets)
            losses.append(float(loss))
        return state, losses
