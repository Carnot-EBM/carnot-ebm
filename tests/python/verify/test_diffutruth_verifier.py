import math
import numpy as np
from carnot.verify.diffutruth_verifier import DiffuTruthVerifier

def test_diffutruth_verifier_valid():
    verifier = DiffuTruthVerifier()
    logprobs = [-0.1, -0.5, -2.0]
    entry = {"token_logprobs": logprobs}
    
    result = verifier.verify(entry)
    
    assert "diffutruth_score" in result
    assert result["energy_proxy_method"] == "token_logprobs_std_times_mean_abs"
    
    logprobs_arr = np.array(logprobs)
    expected_score = np.std(logprobs_arr) * np.mean(np.abs(logprobs_arr))
    assert math.isclose(result["diffutruth_score"], expected_score)

def test_diffutruth_verifier_empty():
    verifier = DiffuTruthVerifier()
    entry = {"token_logprobs": []}
    result = verifier.verify(entry)
    assert result["diffutruth_score"] == 0.0

def test_diffutruth_verifier_none():
    verifier = DiffuTruthVerifier()
    entry = {}
    result = verifier.verify(entry)
    assert result["diffutruth_score"] == 0.0

def test_diffutruth_verifier_invalid_floats():
    verifier = DiffuTruthVerifier()
    entry = {"token_logprobs": [float('inf'), float('nan')]}
    result = verifier.verify(entry)
    assert result["diffutruth_score"] == 0.0
