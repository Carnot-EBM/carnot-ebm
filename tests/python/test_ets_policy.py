"""Tests for ETS Policy Evaluator."""
import math
from carnot.memory.ets_policy import EtsPolicyEvaluator

def test_scale_test_time_compute():
    """Test that compute scales with uncertainty (SCENARIO-FR11-1681)."""
    evaluator = EtsPolicyEvaluator(base_compute=10.0, scaling_factor=2.0)
    
    # Zero uncertainty = base compute
    assert evaluator.scale_test_time_compute(10.0, 0.0) == 10.0
    
    # High uncertainty = scaled compute
    assert evaluator.scale_test_time_compute(10.0, 0.5) == 20.0
    assert evaluator.scale_test_time_compute(10.0, 1.0) == 30.0

def test_promote_policy():
    """Test policy promotion incorporating ETS (REQ-FR11-1681)."""
    evaluator = EtsPolicyEvaluator(base_compute=1.0)
    
    candidate = {"id": "policy_v2"}
    probs = [0.8, 0.9, 0.7] # High transition probabilities
    uncertainty = 0.2
    
    decision = evaluator.promote_policy(candidate, probs, uncertainty)
    
    assert decision["policy_id"] == "policy_v2"
    assert decision["method"] == "ETS"
    assert decision["scaled_compute"] > 1.0
    assert math.isclose(decision["expected_energy"], 0.8)
    assert decision["is_promoted"] is True
    
    # Test rejection with low probabilities
    probs_low = [0.2, 0.3, 0.1]
    decision_reject = evaluator.promote_policy(candidate, probs_low, uncertainty)
    assert decision_reject["is_promoted"] is False
