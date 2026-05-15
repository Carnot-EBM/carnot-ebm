"""Tests for QAOD Verifier."""
import numpy as np
from carnot.verify.qaod_verifier import QAODProbe

def test_qaod_probe_residual():
    # REQ-VERIFY-1740: QAOD orthogonal decomposition
    probe = QAODProbe(threshold=0.5)
    
    q = np.array([[1.0, 0.0]])
    a1 = np.array([[1.0, 0.0]])  # Aligned, residual 0
    a2 = np.array([[0.0, 1.0]])  # Orthogonal, residual 1
    
    res1 = probe.compute_residual_magnitude(a1, q)
    res2 = probe.compute_residual_magnitude(a2, q)
    
    assert np.isclose(res1[0], 0.0)
    assert np.isclose(res2[0], 1.0)

def test_qaod_probe_predict():
    # SCENARIO-VERIFY-1740: Prediction behavior
    probe = QAODProbe(threshold=0.5)
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    a = np.array([[1.0, 0.0], [1.0, 0.0]]) # 1st is aligned, 2nd is orthogonal
    
    preds = probe.predict(a, q)
    assert preds[0] == 0
    assert preds[1] == 1

def test_qaod_probe_predict_proba():
    probe = QAODProbe(threshold=0.5)
    q = np.array([[1.0, 0.0]])
    a = np.array([[0.0, 1.0]])
    
    proba = probe.predict_proba(a, q)
    assert proba.shape == (1, 2)
    assert proba[0, 1] > 0.5  # Logistic(1.0 - 0.5) > 0.5

def test_qaod_probe_fit():
    probe = QAODProbe()
    probe.fit(np.array([[1.0]]), np.array([1]))

def test_qaod_probe_zero_q():
    probe = QAODProbe()
    q = np.array([[0.0, 0.0]])
    a = np.array([[1.0, 1.0]])
    res = probe.compute_residual_magnitude(a, q)
    assert np.isclose(res[0], np.linalg.norm(a))
