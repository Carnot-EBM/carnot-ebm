import pytest
from carnot.verify.tier0s_halluguard import Tier0sVerifier

def test_tier0s_halluguard_valid_score():
    verifier = Tier0sVerifier(threshold=0.5)
    # Valid mathematical reasoning string
    response = "The sum of 2+3 is 5. Therefore the answer is 5."
    score = verifier.halluguard_ntk_score(response)
    # Expected: 2+3=5, actual=5 -> variance=0, jump=5-5=0 -> score=0.0
    assert score < 0.1

def test_tier0s_halluguard_hallucinated_score():
    verifier = Tier0sVerifier(threshold=0.5)
    # Hallucinated mathematical reasoning string
    response = "The sum of 2+3 is 6. Therefore the answer is 7."
    score = verifier.halluguard_ntk_score(response)
    # Expected: 2+3=5, actual=6 -> variance=1, jump=7-6=1 -> score=1.0
    assert score > 0.9

def test_tier0s_halluguard_detect():
    verifier = Tier0sVerifier(threshold=0.5)
    valid_response = "The sum of 2+3 is 5. Therefore the answer is 5."
    hallucinated_response = "The sum of 2+3 is 6. Therefore the answer is 7."
    
    assert not verifier.detect(valid_response)
    assert verifier.detect(hallucinated_response)

def test_tier0s_halluguard_insufficient_numbers():
    verifier = Tier0sVerifier(threshold=0.5)
    # Less than 3 numbers -> logprob_variance = 0.0, semantic_jump = 0.0
    response = "The sum is large."
    score = verifier.halluguard_ntk_score(response)
    assert score == 0.0
