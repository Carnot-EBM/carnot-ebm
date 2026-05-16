"""Tests for NeSy MAP Inference Alternating Training.

References:
    - REQ-SYMKAN-2075
    - SCENARIO-SYMKAN-2075
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from carnot.training.nesy_trainer import NeSyTrainer
from carnot.embeddings.nesy_encoder import NeSyEncoder


class DummyModel(nn.Module):
    """Dummy model outputting 2 features."""
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(2)(x)


def test_nesy_trainer_map_inference():
    """
    Test MAP inference and alternating training step.
    
    Scenario: SCENARIO-SYMKAN-2075: Alternating MAP Inference Training
    """
    key = jax.random.PRNGKey(0)
    
    model = DummyModel()
    optimizer = optax.adam(learning_rate=0.01)
    
    encoder = NeSyEncoder()
    # The encoder returns an energy function that takes a 1D tensor
    # We vmap it to work with a batch of predictions
    energy_fn = encoder.compile_predicate("VAR_0 == VAR_1")
    batch_energy_fn = jax.vmap(energy_fn)
    
    trainer = NeSyTrainer(
        model=model,
        optimizer=optimizer,
        energy_fns=[batch_energy_fn],
        map_steps=5,
        map_step_size=0.1,
    )
    
    input_shape = (4, 3)
    state = trainer.create_train_state(key, input_shape)
    
    # Input batch of 4 examples, 3 features each
    inputs = jnp.ones(input_shape)
    # Targets for the 2 outputs
    targets = jnp.zeros((4, 2))
    
    # Test total_energy
    latents = jnp.array([[1.0, 2.0], [3.0, 3.0], [4.0, 5.0], [0.0, 0.0]])
    energy = trainer.total_energy(latents)
    assert energy > 0.0
    
    # Test map_inference
    refined_latents = trainer.map_inference(latents)
    refined_energy = trainer.total_energy(refined_latents)
    assert refined_energy < energy
    
    # Test train_step
    new_state, loss = trainer.train_step(state, inputs, targets)
    assert loss > 0.0
    assert new_state.step > state.step
    
    # Test train (multiple epochs)
    final_state, losses = trainer.train(state, inputs, targets, epochs=2)
    assert len(losses) == 2
    assert final_state.step == state.step + 2
