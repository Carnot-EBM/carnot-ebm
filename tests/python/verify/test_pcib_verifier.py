import pytest
import numpy as np
from carnot.verify.pcib_verifier import extract_pc_features, PCIBVerifier

def test_extract_pc_features():
    entry = {"token_logprobs": [-0.1, -0.2, -0.3]}
    features = extract_pc_features(entry)
    assert len(features) == 6
    assert np.isclose(features[0], -0.2)  # mean

def test_pcib_verifier():
    verifier = PCIBVerifier()
    X = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    y = np.array([0, 1])
    verifier.fit(X, y)
    scores = verifier.decision_function(X)
    assert len(scores) == 2
