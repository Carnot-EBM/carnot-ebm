"""Tests for python/carnot/verify/ensemble.py

Spec: REQ-VERIFY-1732
"""

import numpy as np
import pytest

from carnot.verify.ensemble import DeentangledReweighter


def test_deentangled_reweighter_fit():
    """Test fitting the reweighter on a failure matrix."""
    rng = np.random.RandomState(42)
    failure_matrix = rng.binomial(1, 0.3, size=(100, 16)).astype(float)
    
    reweighter = DeentangledReweighter(ridge=1e-4)
    reweighter.fit(failure_matrix)
    
    assert reweighter.weights_ is not None
    assert len(reweighter.weights_) == 16
    assert np.isclose(reweighter.weights_.sum(), 1.0)
    assert np.all(reweighter.weights_ >= 0)


def test_deentangled_reweighter_predict():
    """Test predict_weighted_score."""
    rng = np.random.RandomState(42)
    failure_matrix = rng.binomial(1, 0.3, size=(100, 16)).astype(float)
    
    reweighter = DeentangledReweighter(ridge=1e-4)
    reweighter.fit(failure_matrix)
    
    scores = reweighter.predict_weighted_score(failure_matrix)
    assert scores.shape == (100,)
    assert np.all(scores >= 0)
    # the maximum score could be 1.0 if all failed, but here we just check type
    assert np.all(scores <= 1.0)


def test_deentangled_reweighter_not_fitted():
    """Test predict raises error if not fitted."""
    reweighter = DeentangledReweighter()
    failure_matrix = np.zeros((10, 16))
    with pytest.raises(ValueError, match="Reweighter is not fitted."):
        reweighter.predict_weighted_score(failure_matrix)


def test_deentangled_reweighter_zero_weights_fallback():
    """Test fallback when weights sum to zero."""
    reweighter = DeentangledReweighter(ridge=1e-4)
    failure_matrix = np.zeros((10, 16))
    reweighter.fit(failure_matrix)
    
    assert reweighter.weights_ is not None
    assert np.isclose(reweighter.weights_.sum(), 1.0)
    assert np.all(np.isclose(reweighter.weights_, 1.0 / 16))

def test_deentangled_reweighter_fallback_branch():
    """Force the fallback branch by injecting a negative weight sum."""
    reweighter = DeentangledReweighter(ridge=1e-4)
    failure_matrix = np.zeros((10, 16))
    # We patch np.clip to return all zeros so w_sum is 0
    original_clip = np.clip
    try:
        np.clip = lambda a, a_min, a_max, out=None, **kwargs: np.zeros_like(a)
        reweighter.fit(failure_matrix)
    finally:
        np.clip = original_clip

    assert reweighter.weights_ is not None
    assert np.isclose(reweighter.weights_.sum(), 1.0)
    assert np.all(np.isclose(reweighter.weights_, 1.0 / 16))
