from __future__ import annotations

from dataclasses import dataclass, field
import jax
import jax.numpy as jnp


@dataclass
class StructuralDiversityReplayBuffer:
    """Continual learning replay buffer prioritizing structural constraint diversity.
    
    Categorizes samples by their structural constraint type to prevent catastrophic
    forgetting of rare constraints during the FR-11 self-discovery loop.
    
    Spec: REQ-FR11-040
    """
    max_size_per_type: int = 1000
    _buffer: dict[str, list[jax.Array]] = field(default_factory=dict)

    def add(self, state: jax.Array, constraint_type: str) -> None:
        """Add a state to the buffer under the given constraint type.
        
        Spec: REQ-FR11-040-1, REQ-FR11-040-2
        """
        if constraint_type not in self._buffer:
            self._buffer[constraint_type] = []
            
        if state.ndim == 1:
            self._buffer[constraint_type].append(state)
        else:
            for i in range(state.shape[0]):
                self._buffer[constraint_type].append(state[i])
                
        # Evict oldest entries if we exceeded max_size_per_type (FIFO).
        while len(self._buffer[constraint_type]) > self.max_size_per_type:
            self._buffer[constraint_type].pop(0)

    def sample(self, n: int, key: jax.Array) -> jax.Array:
        """Sample n states uniformly across available constraint types.
        
        This preserves structural diversity by ensuring rare constraint types
        are sampled proportionally more often than their raw frequency.
        
        Spec: REQ-FR11-040-3
        """
        if not self._buffer:
            raise ValueError("Cannot sample from an empty replay buffer.")
            
        available_types = list(self._buffer.keys())
        n_types = len(available_types)
        
        # Determine how many samples to draw per type
        samples_per_type = [n // n_types] * n_types
        remainder = n % n_types
        
        # Distribute remainder randomly
        key, subkey = jax.random.split(key)
        if remainder > 0:
            extra_indices = jax.random.choice(subkey, n_types, shape=(remainder,), replace=False)
            for idx in extra_indices:
                samples_per_type[int(idx)] += 1
                
        selected = []
        for type_idx, c_type in enumerate(available_types):
            num_to_sample = samples_per_type[type_idx]
            if num_to_sample == 0:
                continue
                
            type_buffer = self._buffer[c_type]
            type_len = len(type_buffer)
            
            key, subkey = jax.random.split(key)
            indices = jax.random.randint(subkey, shape=(num_to_sample,), minval=0, maxval=type_len)
            
            for idx in indices:
                selected.append(type_buffer[int(idx)])
                
        return jnp.stack(selected)

    def __len__(self) -> int:
        """Return the total number of states in the buffer across all types."""
        return sum(len(v) for v in self._buffer.values())
