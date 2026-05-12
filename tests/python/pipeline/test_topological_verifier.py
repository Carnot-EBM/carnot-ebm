"""Tests for Ontology NN Topological Verification.

References:
    - REQ-VERIFY-1946
    - SCENARIO-VERIFY-1946
"""

import numpy as np
from carnot.pipeline.topological_verifier import TopologicalVerifier

def test_topological_verifier_mutually_exclusive_constraints():
    """
    Test that the topological verifier identifies invalid combinations 
    via negative Forman-Ricci curvature, and projects them back using Deep Delta Learning.
    
    Spec: SCENARIO-VERIFY-1946
    """
    verifier = TopologicalVerifier(learning_rate=0.01)
    
    # Create a star graph where center node conflicts with 5 leaves
    # degrees[0] = 5, degrees[leaf] = 1
    # Edge(0, leaf): 4 - 5 - 1 = -2
    adjacency = np.zeros((6, 6))
    adjacency[0, 1:] = 1
    adjacency[1:, 0] = 1
    
    states = np.ones(6) # All constraints active (invalid mutually exclusive state)
    
    result = verifier.verify(states, adjacency)
    
    assert result["is_valid"] is False
    assert result["curvature"][0] == -2.0
    
    # Verify Deep Delta Learning projection pushed the center state down
    assert result["projected_states"][0] < 1.0
    
    # It should decrement by 0.01 * 2.0 each step for 10 steps = 0.2
    # So 1.0 - 0.2 = 0.8
    assert np.isclose(result["projected_states"][0], 0.8)

def test_topological_verifier_valid_constraints():
    """
    Test a valid constraint configuration.
    """
    verifier = TopologicalVerifier()
    
    # Two nodes connected, no conflicts
    adjacency = np.array([
        [0, 1],
        [1, 0]
    ], dtype=float)
    
    states = np.array([1.0, 1.0])
    
    result = verifier.verify(states, adjacency)
    
    assert result["is_valid"] is True
    # degrees: 1, 1
    # F(e): 4 - 1 - 1 = 2
    assert result["curvature"][0] == 2.0
    assert result["projected_states"][0] == 1.0
