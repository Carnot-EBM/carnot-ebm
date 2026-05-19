import json
import os
import sys
from unittest import mock

# Need to make sure the script is importable or just test the logic directly
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))
import experiment_2508

def test_extract_steps():
    texts = ["a", "b", "\n\n", "c", "\n\n", "d"]
    logprobs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    steps = experiment_2508.extract_steps(texts, logprobs)
    
    assert len(steps) == 3
    assert steps[0][0] == ["a", "b"]
    assert steps[1][0] == ["\n\n", "c"]
    assert steps[2][0] == ["\n\n", "d"]

def test_top_logprobs_to_logit_vector():
    top_logprobs = [
        {"a": 0.1, "b": 0.2},
        {"c": 0.3}
    ]
    vector = experiment_2508.top_logprobs_to_logit_vector(top_logprobs)
    assert len(vector) == 3
    assert vector[0] == 0.2  # sorted
    assert vector[1] == 0.1
    assert vector[2] == 0.3

def test_semantic_energy_compute():
    detector = experiment_2508.SemanticEnergy(temperature=1.0)
    import numpy as np
    logits = np.array([0.1, 0.2, 0.3])
    energy = detector.compute_energy(logits)
    assert isinstance(energy, float)
    assert energy < 0  # Since logits are small positive, sum(exp) > 1 -> log > 0 -> energy < 0

if __name__ == "__main__":
    test_extract_steps()
    test_top_logprobs_to_logit_vector()
    test_semantic_energy_compute()
    print("All tests passed!")
