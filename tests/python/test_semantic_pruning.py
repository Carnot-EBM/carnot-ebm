import numpy as np
from carnot.pipeline.semantic_pruning import SemanticPruner

# REQ-LEARN-1761, SCENARIO-LEARN-1761
def test_semantic_pruning():
    pruner = SemanticPruner(threshold=0.9)
    state1 = {"vector": np.array([1.0, 0.0, 0.0]), "metadata": {"id": 1}}
    state2 = {"vector": np.array([0.95, 0.0, 0.0]), "metadata": {"id": 2}} # Highly similar to state1
    state3 = {"vector": np.array([0.0, 1.0, 0.0]), "metadata": {"id": 3}} # Orthogonal
    state4 = {"vector": np.array([0.0, 0.0, 0.0]), "metadata": {"id": 4}} # Zero vector
    
    states = [state1, state2, state3, state4]
    
    pruned_states = pruner.prune(states)
    
    # State1 is kept. State2 is redundant (similarity > 0.9). State3 is kept. State4 has 0 norm, similarity 0, kept.
    assert len(pruned_states) == 3
    assert pruned_states[0]["metadata"]["id"] == 1
    assert pruned_states[1]["metadata"]["id"] == 3
    assert pruned_states[2]["metadata"]["id"] == 4

def test_semantic_pruning_missing_vector():
    pruner = SemanticPruner(threshold=0.9)
    state1 = {"metadata": {"id": 1}}
    states = [state1]
    
    pruned_states = pruner.prune(states)
    assert len(pruned_states) == 0
