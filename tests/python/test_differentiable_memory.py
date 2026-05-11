import numpy as np
from carnot.pipeline.differentiable_memory import DifferentiableMemoryBank

def test_differentiable_memory_initialization():
    mem = DifferentiableMemoryBank(memory_size=10, vector_dim=8)
    assert mem.keys.shape == (10, 8)
    assert mem.values.shape == (10, 8)

def test_differentiable_memory_read_write():
    # Set seed for reproducibility
    np.random.seed(42)
    mem = DifferentiableMemoryBank(memory_size=5, vector_dim=4)
    
    key = np.array([1.0, 0.0, 0.0, 0.0])
    val = np.array([0.5, 0.5, 0.5, 0.5])
    
    mem.write(key, val)
    
    # Read with same key
    retrieved = mem.read(key)
    # The read should be somewhat close to val due to dot product attention
    # We wrote it to least used. The key we wrote will have dot product 1.0 with query, 
    # others will be random near 0.
    assert retrieved.shape == (4,)
    
def test_differentiable_memory_update():
    np.random.seed(42)
    mem = DifferentiableMemoryBank(memory_size=3, vector_dim=3)
    
    query = np.array([1.0, 1.0, 1.0])
    target_val = np.array([2.0, 2.0, 2.0])
    
    # Initial read
    init_val = mem.read(query)
    
    # Update
    mem.update(query, target_val, learning_rate=0.5)
    
    # Read again
    post_val = mem.read(query)
    
    # The new read should be closer to target_val than init_val
    dist_init = np.linalg.norm(init_val - target_val)
    dist_post = np.linalg.norm(post_val - target_val)
    
    assert dist_post < dist_init

def test_batch_read():
    np.random.seed(42)
    mem = DifferentiableMemoryBank(memory_size=5, vector_dim=4)
    queries = np.random.randn(3, 4)
    retrieved = mem.read(queries)
    assert retrieved.shape == (3, 4)
