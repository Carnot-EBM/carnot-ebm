import pytest
import jax.numpy as jnp
import jax.random as jrandom

from carnot.training.semantic_pruning import EmbeddingSemanticPruner
from carnot.training.replay_buffer import ReplayBuffer

def test_embedding_semantic_pruning_filter_batch():
    pruner = EmbeddingSemanticPruner(similarity_threshold=0.9)
    
    # Existing buffer with 2 vectors
    existing_buffer = [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0])
    ]
    
    # New batch: 
    # 1. highly similar to [1,0,0] (should be pruned)
    # 2. orthogonal (should be kept)
    # 3. highly similar to the second (should be pruned)
    new_batch = jnp.array([
        [0.95, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.95, 1.0] # Sim to [0,1,0] is 0.95/sqrt(0.95^2 + 1) = 0.95 / 1.38 < 0.9, so wait, let's make it more similar
    ])
    new_batch = new_batch.at[2].set(jnp.array([0.0, 0.99, 0.1])) # Sim is 0.99 / sqrt(0.99^2 + 0.1^2) = 0.995 -> pruned
    
    filtered = pruner.filter_batch(new_batch, existing_buffer)
    
    assert len(filtered) == 1
    assert jnp.allclose(filtered[0], jnp.array([0.0, 0.0, 1.0]))

def test_embedding_semantic_pruning_empty_existing():
    pruner = EmbeddingSemanticPruner(similarity_threshold=0.9)
    existing_buffer = []
    
    # Batch has self-redundancy
    new_batch = jnp.array([
        [1.0, 0.0, 0.0],
        [0.95, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    filtered = pruner.filter_batch(new_batch, existing_buffer)
    assert len(filtered) == 2
    assert jnp.allclose(filtered[0], jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(filtered[1], jnp.array([0.0, 1.0, 0.0]))

def test_replay_buffer_pruning():
    buffer = ReplayBuffer(max_size=10, similarity_threshold=0.9)
    
    batch = jnp.array([
        [1.0, 0.0],
        [0.99, 0.0],
        [0.0, 1.0]
    ])
    
    buffer.add(batch)
    assert len(buffer) == 2
    
    buffer.add(jnp.array([0.98, 0.0]))
    assert len(buffer) == 2
