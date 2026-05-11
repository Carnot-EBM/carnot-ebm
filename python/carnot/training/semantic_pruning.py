from typing import List, Set
import re


class SemanticPruner:
    """Semantic pruning mechanism for structural constraint rules.

    Identifies and removes redundant structural constraint rules based on
    semantic similarity (Jaccard similarity of normalized words) to reduce
    semantic redundancy in the FR-11 replay buffer.

    Spec: REQ-FR11-041
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def _normalize(self, text: str) -> Set[str]:
        """Normalize text and return a set of words."""
        # Convert to lowercase and extract alphanumeric words
        words = re.findall(r"\b\w+\b", text.lower())
        return set(words)

    def _calculate_similarity(self, rule1: str, rule2: str) -> float:
        """Calculate Jaccard similarity between two constraint rules.

        Spec: REQ-FR11-041-1
        """
        set1 = self._normalize(rule1)
        set2 = self._normalize(rule2)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def prune_redundant_rules(self, rules: List[str]) -> List[str]:
        """Prune redundant rules from a list of constraint rules.

        Spec: REQ-FR11-041-2
        """
        if not rules:
            return []

        pruned_rules: List[str] = []
        for new_rule in rules:
            is_redundant = False
            for existing_rule in pruned_rules:
                similarity = self._calculate_similarity(new_rule, existing_rule)
                if similarity >= self.similarity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                pruned_rules.append(new_rule)

        return pruned_rules

import jax
import jax.numpy as jnp
from typing import List, Optional

class EmbeddingSemanticPruner:
    """Semantic pruning mechanism for structural constraint rules (AST/logic embeddings).
    
    Identifies and removes redundant states based on cosine similarity of their 
    embeddings to reduce interference in the replay buffer.
    
    Spec: REQ-FR11-042
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold

    def filter_batch(self, states: jax.Array, existing_buffer: List[jax.Array]) -> List[jax.Array]:
        """Filter a batch of states against an existing buffer of states."""
        if states.ndim == 1:
            states = jnp.expand_dims(states, 0)
            
        # Normalize states
        norms = jnp.linalg.norm(states, axis=1, keepdims=True)
        norm_states = states / jnp.maximum(norms, 1e-8)
        
        # Check against existing
        if len(existing_buffer) > 0:
            existing_stacked = jnp.stack(existing_buffer)
            existing_norms = jnp.linalg.norm(existing_stacked, axis=1, keepdims=True)
            norm_existing = existing_stacked / jnp.maximum(existing_norms, 1e-8)
            
            # shape: (batch_size, existing_size)
            sim_with_existing = jnp.dot(norm_states, norm_existing.T)
            max_sim_existing = jnp.max(sim_with_existing, axis=1)
            redundant_with_existing = max_sim_existing >= self.similarity_threshold
        else:
            redundant_with_existing = jnp.zeros(states.shape[0], dtype=bool)
            
        # Pairwise similarities within the batch
        # shape: (batch_size, batch_size)
        sim_within_batch = jnp.dot(norm_states, norm_states.T)
        
        states_to_add = []
        kept_indices = []
        
        # Convert to numpy for fast python loop
        import numpy as np
        redundant_with_existing = np.array(redundant_with_existing)
        sim_within_batch = np.array(sim_within_batch)
        
        for i in range(states.shape[0]):
            if redundant_with_existing[i]:
                continue
                
            # Check against previously kept items in THIS batch
            is_redundant = False
            for j in kept_indices:
                if sim_within_batch[i, j] >= self.similarity_threshold:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                kept_indices.append(i)
                states_to_add.append(states[i])
                
        return states_to_add
