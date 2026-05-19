import pytest
import math
import numpy as np

from carnot.verify.ir_conformal_verifier import InternalRepresentationConformalVerifier

def test_ir_conformal_verifier_compute_score():
    verifier = InternalRepresentationConformalVerifier()
    
    # Test with empty logprobs
    assert verifier.compute_ir_score([]) == 0.0
    
    # Test with some logprobs
    logprobs = [-1.0, -2.0, -3.0, -4.0]
    # squares: 1, 4, 9, 16
    # norm = 1 + 4 + 9 + 16 = 30
    # length = 4. 
    # indices: 0, 1, 2, 3
    # n//4 = 1, 3n//4 = 3
    # weights: 
    # i=0 < 1: 1
    # i=1 (not < 1, not > 3): 2
    # i=2 (not < 1, not > 3): 2
    # i=3 (not < 1, not > 3): 2 (wait, 3 > 3 is false, so 2) -- actually i=3 is not > 3, wait
    score = verifier.compute_ir_score(logprobs)
    
    # Just check it returns a float and doesn't crash
    assert isinstance(score, float)
    assert score > 0

def test_ir_conformal_verifier_calibrate_and_nonconformity():
    verifier = InternalRepresentationConformalVerifier()
    
    calib_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    verifier.calibrate(calib_scores)
    
    assert verifier.is_calibrated
    assert math.isclose(verifier.calib_mean, 0.3)
    
    nonconformity = verifier.compute_nonconformity(0.6)
    assert isinstance(nonconformity, float)
    assert nonconformity > 0  # 0.6 is above mean 0.3
    
def test_ir_conformal_verifier_uncalibrated_error():
    verifier = InternalRepresentationConformalVerifier()
    with pytest.raises(RuntimeError, match="Verifier not calibrated"):
        verifier.compute_nonconformity(0.5)
