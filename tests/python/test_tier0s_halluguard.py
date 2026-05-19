import pytest
from carnot.verify.tier0s_halluguard import Tier0sVerifier

def test_tier0s_verifier():
    verifier = Tier0sVerifier(threshold=0.5)
    score = verifier.halluguard_ntk_score("test response")
    assert 0.0 <= score <= 1.0
    
    result = verifier.detect("test response")
    assert "tier0s_score" in result
    assert "is_hallucination_predicted" in result
    assert isinstance(result["is_hallucination_predicted"], bool)
