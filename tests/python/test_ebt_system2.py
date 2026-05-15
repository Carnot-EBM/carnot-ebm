"""Tests for EBT System 2 Loop.

References: REQ-EBT-1805, SCENARIO-EBT-1805.
"""
from carnot.ebt_system2 import EBTSystem2Loop

def test_ebt_system2_loop():
    """Verify REQ-EBT-1805 and SCENARIO-EBT-1805."""
    loop = EBTSystem2Loop()
    result = loop.optimize_candidates(["initial candidate"], max_steps=3)
    
    assert result["model_used"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert "final_candidate" in result
    assert result["improved_satisfaction"] is True
    assert len(result["optimization_history"]) == 3
    
    # Check that constraint satisfaction increases
    history = result["optimization_history"]
    assert history[-1]["constraint_satisfaction"] > history[0]["constraint_satisfaction"]

def test_score_candidate_bounds():
    """Ensure energy stays within bounds."""
    loop = EBTSystem2Loop()
    # A very long string with "target" could theoretically go negative,
    # but the method uses max(0.0, energy)
    long_target = "target" * 50
    energy = loop.score_candidate(long_target)
    assert energy == 0.0
