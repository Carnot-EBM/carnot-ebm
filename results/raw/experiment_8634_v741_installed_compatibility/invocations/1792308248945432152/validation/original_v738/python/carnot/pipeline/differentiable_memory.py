import numpy as np

class DifferentiableMemoryBank:
    """
    A differentiable constraint memory bank for multi-session continual learning.
    Implements differentiable read, write, and update operations using attention.
    """
    def __init__(self, memory_size: int, vector_dim: int):
        self.memory_size = memory_size
        self.vector_dim = vector_dim
        # Initialize memory keys and values
        self.keys = np.random.randn(memory_size, vector_dim) * 0.1
        self.values = np.zeros((memory_size, vector_dim))
        # Keep track of usage for simple LRU or allocation if needed
        self.usage = np.zeros(memory_size)
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def read(self, query: np.ndarray) -> np.ndarray:
        """
        Reads from memory using softmax attention.
        Args:
            query: np.ndarray of shape (vector_dim,) or (batch_size, vector_dim)
        Returns:
            Retrieved value of same shape as query.
        """
        # Ensure query is 2D
        is_1d = query.ndim == 1
        if is_1d:
            query = query.reshape(1, -1)
            
        # Compute attention scores: (batch_size, vector_dim) @ (vector_dim, memory_size) -> (batch_size, memory_size)
        scores = np.dot(query, self.keys.T)
        
        # Softmax attention weights
        attention = self._softmax(scores)
        
        # Update usage
        self.usage += np.sum(attention, axis=0)
        
        # Weighted sum of values: (batch_size, memory_size) @ (memory_size, vector_dim) -> (batch_size, vector_dim)
        retrieved = np.dot(attention, self.values)
        
        if is_1d:
            return retrieved.flatten()
        return retrieved

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """
        Writes a new key-value pair into the memory slot with the least usage.
        (A simplified differentiable-inspired hard write for initialization/storage).
        """
        # Find least used slot
        idx = np.argmin(self.usage)
        self.keys[idx] = key
        self.values[idx] = value
        self.usage[idx] += 1.0

    def update(self, query: np.ndarray, value: np.ndarray, learning_rate: float = 0.1) -> None:
        """
        Updates memory values differentially based on attention weights.
        Args:
            query: (vector_dim,)
            value: (vector_dim,) target value to move towards
        """
        query_2d = query.reshape(1, -1)
        scores = np.dot(query_2d, self.keys.T)
        attention = self._softmax(scores).flatten()
        
        # Update values: move towards the target value weighted by attention
        for i in range(self.memory_size):
            self.values[i] += learning_rate * attention[i] * (value - self.values[i])
