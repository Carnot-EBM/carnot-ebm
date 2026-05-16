"""Tests for EBRM Trainer.

Spec: REQ-LEARN-2066
"""

import jax
import jax.numpy as jnp
import optax

from carnot.training.ebrm_trainer import EBRMMLP, EBRMTrainer


def test_ebrm_trainer_cd_step():
    """Test the CD training step for EBRMTrainer (REQ-LEARN-2066-3)."""
    model = EBRMMLP(hidden_dim=16)
    optimizer = optax.adam(1e-3)
    trainer = EBRMTrainer(model, optimizer)
    
    key = jax.random.PRNGKey(0)
    input_shape = (10, 4)
    state = trainer.create_train_state(key, input_shape)
    
    key1, key2 = jax.random.split(key)
    positive_traces = jax.random.normal(key1, input_shape)
    negative_traces = jax.random.normal(key2, input_shape)
    
    new_state, loss = trainer.train_step(state, positive_traces, negative_traces)
    
    assert loss is not None
    assert not jnp.isnan(loss)
    assert new_state.step == 1


def test_ebrm_trainer_train_loop():
    """Test the train method for EBRMTrainer (REQ-LEARN-2066-3, SCENARIO-LEARN-2066)."""
    model = EBRMMLP(hidden_dim=16)
    optimizer = optax.adam(1e-3)
    trainer = EBRMTrainer(model, optimizer)
    
    key = jax.random.PRNGKey(0)
    input_shape = (10, 4)
    state = trainer.create_train_state(key, input_shape)
    
    key1, key2 = jax.random.split(key)
    positive_traces = jax.random.normal(key1, input_shape)
    negative_traces = jax.random.normal(key2, input_shape)
    
    new_state, losses = trainer.train(state, positive_traces, negative_traces, epochs=5)
    
    assert len(losses) == 5
    assert new_state.step == 5
