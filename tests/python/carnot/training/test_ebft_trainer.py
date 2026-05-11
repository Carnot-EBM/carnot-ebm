"""Tests for the Energy-Based Fine-Tuning (EBFT) Trainer.

Spec: REQ-TRAIN-007, SCENARIO-TRAIN-007
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn

from carnot.training.ebft_trainer import EBFTTrainer
from carnot.models.latent_optimizer import LatentOptimizer
from carnot.pipeline.differentiable_memory import DifferentiableMemoryBank


class DummyEnergyModel(nn.Module):
    """A simple linear energy model for testing."""
    @nn.compact
    def __call__(self, x):
        # Flatten x and project to scalar energy
        x_flat = x.reshape((x.shape[0], -1))
        energy = nn.Dense(1)(x_flat)
        return jnp.squeeze(energy, axis=-1)


def test_ebft_trainer_init():
    """Test trainer initialization."""
    model = DummyEnergyModel()
    optimizer = optax.adam(1e-3)
    trainer = EBFTTrainer(model=model, optimizer=optimizer)
    assert trainer.model is model
    assert trainer.optimizer is optimizer
    assert trainer.latent_optimizer is None
    assert trainer.memory_bank is None


def test_ebft_trainer_create_train_state():
    """Test creating train state."""
    model = DummyEnergyModel()
    optimizer = optax.adam(1e-3)
    trainer = EBFTTrainer(model=model, optimizer=optimizer)
    
    key = jax.random.PRNGKey(0)
    input_shape = (2, 4)
    state = trainer.create_train_state(key, input_shape)
    
    assert state.step == 0
    assert len(state.params) > 0


def test_ebft_trainer_train_step_with_experts():
    """Test a single training step with provided expert sequences."""
    model = DummyEnergyModel()
    optimizer = optax.sgd(0.1)
    trainer = EBFTTrainer(model=model, optimizer=optimizer)
    
    key = jax.random.PRNGKey(0)
    state = trainer.create_train_state(key, (2, 4))
    
    expert_sequences = jnp.ones((2, 4))
    rollout_sequences = jnp.zeros((2, 4))
    
    new_state, loss = trainer.train_step(state, rollout_sequences, expert_sequences=expert_sequences)
    
    assert loss.shape == ()
    assert new_state.step == 1
    # Check that parameters changed
    old_kernel = state.params['Dense_0']['kernel']
    new_kernel = new_state.params['Dense_0']['kernel']
    assert not jnp.allclose(old_kernel, new_kernel)


def test_ebft_trainer_align_latents_missing_optimizer():
    """Test error when aligning latents without an optimizer."""
    model = DummyEnergyModel()
    optimizer = optax.sgd(0.1)
    trainer = EBFTTrainer(model=model, optimizer=optimizer)
    
    key = jax.random.PRNGKey(0)
    state = trainer.create_train_state(key, (2, 4))
    
    with pytest.raises(ValueError, match="LatentOptimizer is required"):
        trainer.align_latents(state, jnp.zeros((2, 4)), key)


def test_ebft_trainer_train_step_without_experts_missing_key():
    """Test error when training without experts and missing PRNG key."""
    model = DummyEnergyModel()
    optimizer = optax.sgd(0.1)
    latent_opt = LatentOptimizer(step_size=0.1, noise_scale=0.0, max_steps=1)
    trainer = EBFTTrainer(model=model, optimizer=optimizer, latent_optimizer=latent_opt)
    
    key = jax.random.PRNGKey(0)
    state = trainer.create_train_state(key, (2, 4))
    
    with pytest.raises(ValueError, match="PRNG key must be provided"):
        trainer.train_step(state, jnp.zeros((2, 4)))


def test_ebft_trainer_train_step_with_latent_alignment():
    """Test training step using latent alignment to generate experts."""
    model = DummyEnergyModel()
    optimizer = optax.sgd(0.1)
    latent_opt = LatentOptimizer(step_size=0.1, noise_scale=0.0, max_steps=2)
    trainer = EBFTTrainer(model=model, optimizer=optimizer, latent_optimizer=latent_opt)
    
    key = jax.random.PRNGKey(0)
    state = trainer.create_train_state(key, (2, 4))
    
    rollout_sequences = jax.random.normal(key, (2, 4))
    key, subkey = jax.random.split(key)
    
    new_state, loss = trainer.train_step(state, rollout_sequences, key=subkey)
    assert new_state.step == 1
    assert loss.shape == ()


def test_ebft_trainer_update_memory_multi_dim():
    """Test memory bank update with 2D sequences."""
    model = DummyEnergyModel()
    optimizer = optax.sgd(0.1)
    memory_bank = DifferentiableMemoryBank(memory_size=10, vector_dim=4)
    trainer = EBFTTrainer(model=model, optimizer=optimizer, memory_bank=memory_bank)
    
    key = jax.random.PRNGKey(0)
    state = trainer.create_train_state(key, (2, 4))
    
    expert_sequences = jnp.ones((2, 4))
    rollout_sequences = jnp.zeros((2, 4))
    
    initial_values = memory_bank.values.copy()
    trainer.train_step(state, rollout_sequences, expert_sequences=expert_sequences)
    
    # Check that memory values changed
    assert not (memory_bank.values == initial_values).all()


def test_ebft_trainer_update_memory_1d():
    """Test memory bank update with 1D sequences."""
    model = DummyEnergyModel()
    optimizer = optax.sgd(0.1)
    memory_bank = DifferentiableMemoryBank(memory_size=10, vector_dim=4)
    trainer = EBFTTrainer(model=model, optimizer=optimizer, memory_bank=memory_bank)
    
    expert_sequence = jnp.ones((4,))
    rollout_sequence = jnp.zeros((4,))
    
    initial_values = memory_bank.values.copy()
    trainer.update_memory(rollout_sequence, expert_sequence)
    
    # Check that memory values changed
    assert not (memory_bank.values == initial_values).all()
