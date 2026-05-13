"""Tests for the EBT Reasoning Evaluator.

Spec: REQ-NRGPT-004, SCENARIO-NRGPT-004
"""

import jax
import jax.numpy as jnp
from carnot.models.ebt_compatibility import EBTCompatibilityModel
from carnot.models.ebt_reasoning_eval import EBTReasoningEvaluator


def test_ebt_reasoning_evaluation():
    """Test full reasoning trace evaluation using EBT.
    
    Spec: REQ-NRGPT-004, SCENARIO-NRGPT-004
    """
    key = jax.random.PRNGKey(0)
    input_dim = 4
    hidden_dim = 8
    
    model = EBTCompatibilityModel(input_dim=input_dim, hidden_dim=hidden_dim, key=key)
    evaluator = EBTReasoningEvaluator(model)
    
    input_seq = jnp.array([1.0, 0.5, -0.2, 0.1])
    truth_seq = jnp.array([0.9, 0.4, -0.1, 0.0])
    
    predicted_steps = [
        jnp.array([0.5, 0.5, 0.0, 0.0]),
        jnp.array([0.7, 0.4, -0.1, 0.0]),
        jnp.array([0.8, 0.4, -0.1, 0.0]),
    ]
    
    results = evaluator.evaluate_trace(input_seq, predicted_steps, truth_seq)
    
    assert "partial_energies" in results
    assert len(results["partial_energies"]) == 3
    assert "final_energy" in results
    assert "truth_energy" in results
    assert "compatibility_gap" in results
    
    assert results["final_energy"] == results["partial_energies"][-1]
    
    dist = evaluator.compute_distribution(input_seq, predicted_steps)
    assert "mean" in dist
    assert "min" in dist
    assert "max" in dist
    assert "var" in dist
    
    empty_dist = evaluator.compute_distribution(input_seq, [])
    assert empty_dist["mean"] == 0.0
    
    empty_results = evaluator.evaluate_trace(input_seq, [], truth_seq)
    assert empty_results["final_energy"] == float('inf')
