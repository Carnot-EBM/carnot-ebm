import jax
import jax.numpy as jnp
import pytest

from carnot.training.fr11_continuous import StructuralDiversityReplayBuffer

def test_fr11_continuous_replay_buffer_add_and_sample():
    """Test that StructuralDiversityReplayBuffer categorizes and samples correctly.
    
    Spec traces: REQ-FR11-040, SCENARIO-FR11-040
    """
    buffer = StructuralDiversityReplayBuffer(max_size_per_type=2)
    
    # Add constraint type A (flooded)
    buffer.add(jnp.array([1.0, 1.0]), "type_a")
    buffer.add(jnp.array([1.0, 2.0]), "type_a")
    buffer.add(jnp.array([1.0, 3.0]), "type_a")  # Should evict [1.0, 1.0]
    
    # Add constraint type B (rare)
    buffer.add(jnp.array([2.0, 1.0]), "type_b")
    
    assert len(buffer) == 3
    assert len(buffer._buffer["type_a"]) == 2
    assert len(buffer._buffer["type_b"]) == 1
    
    # Sample 4 items
    key = jax.random.PRNGKey(0)
    samples = buffer.sample(4, key)
    
    assert samples.shape == (4, 2)
    
    # Due to diversity sampling, we expect roughly equal representation
    # even though type_a is more frequent in raw additions.
    # 4 items / 2 types = 2 items per type.
    # type_b only has [2.0, 1.0], so it should be sampled twice.
    type_b_count = jnp.sum(samples[:, 0] == 2.0)
    assert type_b_count == 2
    
def test_fr11_continuous_replay_buffer_empty():
    buffer = StructuralDiversityReplayBuffer()
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        buffer.sample(1, key)

def test_fr11_continuous_replay_buffer_batch_add():
    buffer = StructuralDiversityReplayBuffer(max_size_per_type=10)
    batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    buffer.add(batch, "type_c")
    assert len(buffer) == 2

def test_fr11_continuous_replay_buffer_remainder_and_zero():
    buffer = StructuralDiversityReplayBuffer(max_size_per_type=10)
    buffer.add(jnp.array([1.0]), "type_a")
    buffer.add(jnp.array([2.0]), "type_b")
    buffer.add(jnp.array([3.0]), "type_c")
    
    key = jax.random.PRNGKey(42)
    # Sample 4 items from 3 types -> remainder = 1
    samples = buffer.sample(4, key)
    assert samples.shape == (4, 1)
    
    # Sample 1 item from 3 types -> remainder = 1, but two types will have num_to_sample == 0
    samples_1 = buffer.sample(1, key)
    assert samples_1.shape == (1, 1)

