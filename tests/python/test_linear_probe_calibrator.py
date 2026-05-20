import pytest
import numpy as np
from carnot.verify.linear_probe_calibrator import LinearProbeCalibrator

def test_linear_probe_calibrator_fit_calibrate():
    # Simple synthetic data
    np.random.seed(42)
    # 100 samples, 5 features
    features = np.random.randn(100, 5)
    # Labels correlated with first feature
    labels = (features[:, 0] > 0).astype(int)
    
    calibrator = LinearProbeCalibrator()
    calibrator.fit(features, labels)
    
    probs = calibrator.calibrate(features)
    assert probs.shape == (100,)
    assert np.all((probs >= 0) & (probs <= 1))

def test_linear_probe_calibrator_ece():
    calibrator = LinearProbeCalibrator()
    
    # Perfect calibration
    predictions = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])  # but wait, 0.1 means 10% correct. If 2 samples have 0.1, 0 correct is 0%, error = 0.1.
    
    # To get perfect calibration:
    # 10 samples with 0.1 probability, 1 is correct.
    predictions = np.array([0.1]*10 + [0.9]*10)
    labels = np.array([1] + [0]*9 + [1]*9 + [0])
    
    ece = calibrator.ece(predictions, labels, n_bins=10)
    # Expected ECE:
    # Bin 0.0-0.1 (contains 0.1): 10 items, mean prob 0.1, mean label 0.1. Error = 0.
    # Bin 0.8-0.9 (contains 0.9): 10 items, mean prob 0.9, mean label 0.9. Error = 0.
    # Total ECE ~ 0
    assert ece < 0.01

def test_linear_probe_calibrator_not_fitted():
    calibrator = LinearProbeCalibrator()
    with pytest.raises(ValueError, match="Calibrator is not fitted yet."):
        calibrator.calibrate(np.random.randn(10, 5))
