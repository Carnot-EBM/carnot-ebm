import numpy as np
from carnot.pipeline.fep_aggregator import FEPAggregator

def test_fep_aggregator_fit():
    X = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.8, 0.2]
    ])
    y = np.array([0, 0, 1, 1])
    
    agg = FEPAggregator()
    agg.fit(X, y)
    
    assert agg.coefficients is not None
    
    score_0 = agg.aggregate([0.15, 0.85])
    score_1 = agg.aggregate([0.85, 0.15])
    
    assert score_0 < 0.5
    assert score_1 > 0.5

def test_fep_aggregator_init_with_coef():
    agg = FEPAggregator(coefficients=[1.0, -1.0], intercept=0.0)
    score = agg.aggregate([0.5, 0.5])
    assert np.isclose(score, 0.5)
    
    score_high = agg.aggregate([1.0, 0.0])
    assert score_high > 0.5
