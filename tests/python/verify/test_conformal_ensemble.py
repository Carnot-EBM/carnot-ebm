import numpy as np
import pytest
from carnot.verify.conformal_ensemble import ConformalEnsemble

def test_conformal_ensemble_fit_predict():
    # 3 verifiers, 4 calibration examples
    cal_scores = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.15, 0.15, 0.35],
        [0.05, 0.25, 0.25]
    ])
    names = ["v1", "v2", "v3"]
    
    ensemble = ConformalEnsemble()
    ensemble.fit(cal_scores, names)
    
    assert ensemble.n_cal == 4
    assert len(ensemble.calibration_scores) == 3
    # Check sorting
    assert np.allclose(ensemble.calibration_scores["v1"], [0.05, 0.1, 0.15, 0.2])
    
    # 2 test examples
    test_scores = np.array([
        [0.12, 0.2, 0.3],   # In-distribution (should have low score)
        [0.9, 0.9, 0.9]     # Out-of-distribution (hallucination, high score)
    ])
    
    p_values = ensemble.predict_p_values(test_scores)
    assert p_values.shape == (2, 3)
    
    # For test 1, v1=0.12, alpha_1=[0.05, 0.1, 0.15, 0.2]. Count >= 0.12 is 2 (0.15, 0.2). p = 2 / 5 = 0.4
    assert np.isclose(p_values[0, 0], 0.4)
    
    # For test 2, v1=0.9. Count >= 0.9 is 0. p = 0.0
    assert np.isclose(p_values[1, 0], 0.0)
    
    final_scores = ensemble.predict(test_scores)
    assert final_scores.shape == (2,)
    # Second should be higher anomaly score
    assert final_scores[1] > final_scores[0]
    
    # With p=0, log(clip(0, 1e-15)) = -34.5. chi2 = large. sf = 0.0. score = 1.0
    assert np.isclose(final_scores[1], 1.0)
