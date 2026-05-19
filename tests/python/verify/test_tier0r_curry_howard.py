import pytest
from carnot.verify.tier0r_curry_howard import Tier0rVerifier

def test_tier0r_importable():
    # If this module runs, it was successfully imported
    verifier = Tier0rVerifier()
    assert verifier is not None

def test_tier0r_score_returns_float_in_range():
    verifier = Tier0rVerifier()
    
    # Valid response
    valid_resp = "We have 5 apples. We add 3 apples. So we have 8 apples."
    score1 = verifier.score(valid_resp)
    assert isinstance(score1, float)
    assert 0.0 <= score1 <= 1.0
    
    # Invalid response
    invalid_resp = "10\nNoah buys 10 units. claim to command initial state constraint: Therefore, 10 apples."
    score2 = verifier.score(invalid_resp)
    assert isinstance(score2, float)
    assert 0.0 <= score2 <= 1.0
    
    # Ensure invalid is penalized
    assert score2 > score1
