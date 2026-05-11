import numpy as np
from typing import List, Dict, Any

class SemanticPruner:
    """Prunes redundant states from the continual memory buffer based on semantic similarity."""
    
    def __init__(self, threshold: float = 0.9):
        """
        Initializes the pruner with a cosine similarity threshold.
        """
        self.threshold = threshold
        
    def prune(self, memory_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters memory_states by keeping only those that are not semantically redundant.
        A state is redundant if its cosine similarity with a previously kept state is >= threshold.
        """
        kept_states = []
        kept_vectors = []
        
        for state in memory_states:
            vector = state.get("vector")
            if vector is None:
                continue
                
            norm = np.linalg.norm(vector)
            if norm == 0:
                norm_vector = vector
            else:
                norm_vector = vector / norm
                
            is_redundant = False
            for kv in kept_vectors:
                similarity = np.dot(norm_vector, kv)
                if similarity >= self.threshold:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                kept_states.append(state)
                kept_vectors.append(norm_vector)
                
        return kept_states
