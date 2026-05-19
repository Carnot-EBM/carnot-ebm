import pytest
import numpy as np
from carnot.learn.jepa_predictor import JEPAViolationPredictor

def test_jepa_predictor_extract_features():
    predictor = JEPAViolationPredictor()
    logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    
    features = predictor.extract_features(logprobs)
    
    # First half should be len=3: [-0.1, -0.2, -0.3]
    assert len(features) == 5
    assert features[4] == 3.0  # length
    assert features[2] == -0.3  # min
    np.testing.assert_almost_equal(features[0], -0.2)  # mean

def test_jepa_predictor_extract_features_empty():
    predictor = JEPAViolationPredictor()
    logprobs = []
    
    features = predictor.extract_features(logprobs)
    assert features == [0.0, 0.0, 0.0, 0.0, 0]

def test_jepa_predictor_extract_features_single():
    predictor = JEPAViolationPredictor()
    logprobs = [-0.5]
    
    features = predictor.extract_features(logprobs)
    # len=1 // 2 = 0, but falls back to 1
    assert features[4] == 1.0
    assert features[0] == -0.5

def test_jepa_predictor_fit_predict():
    predictor = JEPAViolationPredictor(max_iter=10)
    X = np.array([
        [-0.1, 0.01, -0.2, 0.0, 5],
        [-2.0, 1.5, -5.0, -1.0, 5],
        [-0.05, 0.001, -0.1, 0.1, 5],
        [-3.0, 2.0, -6.0, -2.0, 5]
    ])
    y = np.array([0, 1, 0, 1])
    
    predictor.fit(X, y)
    probs = predictor.predict_proba(X)
    
    assert probs.shape == (4, 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)
