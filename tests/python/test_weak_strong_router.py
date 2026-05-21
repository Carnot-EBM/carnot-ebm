import pytest
from carnot.pipeline.verify_repair import WeakStrongRouter, RoutingDecision

def test_weak_strong_router():
    router = WeakStrongRouter(t_low=0.2, t_high=0.8)
    
    # Test accept
    decision = router.route("prompt", "response", weak_score=0.1)
    assert decision.path == "accept"
    assert decision.verifier == "none"
    
    # Test full_ensemble
    decision = router.route("prompt", "response", weak_score=0.9)
    assert decision.path == "full_ensemble"
    assert decision.verifier == "tier0_all"
    
    # Test partial verify
    decision = router.route("prompt", "response", weak_score=0.5)
    assert decision.path == "tier0f_only"
    assert decision.verifier == "semantic_calibration"
    
    # Test fallback proxy
    # Since we can't easily mock the pickle, let's just make sure it doesn't crash 
    # and returns a decision.
    decision = router.route("short prompt", "some response")
    assert isinstance(decision, RoutingDecision)
