"""Tests for Tier-4 Adaptive Structure.

Spec: REQ-LEARN-061, SCENARIO-LEARN-105
"""

import numpy as np
from carnot.pipeline.tier4_adaptive_structure import (
    compute_marginal_contributions,
    prune_verifiers,
    flag_residual_regions,
)

def mock_auroc_fn(labels, scores):
    # Simple mock: return mean score for positive labels
    pos_scores = scores[labels == 1]
    if len(pos_scores) == 0:
        return 0.0
    return float(np.mean(pos_scores))

def test_compute_marginal_contributions():
    labels = np.array([1, 0, 1, 0])
    scores_by_verifier = {
        'v1': np.array([0.9, 0.1, 0.8, 0.2]),
        'v2': np.array([0.4, 0.6, 0.4, 0.6])
    }
    weights = {'v1': 1.0, 'v2': 0.1}
    
    contributions = compute_marginal_contributions(
        labels, scores_by_verifier, weights, mock_auroc_fn
    )
    
    assert 'v1' in contributions
    assert 'v2' in contributions
    # full score pos mean = mean([0.94, 0.84]) = 0.89
    # dropped v1 score pos mean = mean([0.04, 0.04]) = 0.04
    # dropped v2 score pos mean = mean([0.9, 0.8]) = 0.85
    # marginal v1 = 0.89 - 0.04 = 0.85
    # marginal v2 = 0.89 - 0.85 = 0.04
    assert np.isclose(contributions['v1'], 0.85)
    assert np.isclose(contributions['v2'], 0.04)

def test_prune_verifiers():
    marginals = {'v1': 0.85, 'v2': 0.001, 'v3': -0.005}
    pruned, retained = prune_verifiers(marginals, threshold=0.002)
    assert pruned == ['v2', 'v3']
    assert retained == ['v1']

def test_flag_residual_regions():
    labels = np.array([1, 1, 1, 0])
    scores_by_verifier = {
        'v1': np.array([0.9, 0.1, 0.1, 0.9]),
        'v2': np.array([0.1, 0.6, 0.1, 0.9])
    }
    retained = ['v1', 'v2']
    # label[0]=1, v1 catches (0.9 >= 0.5)
    # label[1]=1, v2 catches (0.6 >= 0.5)
    # label[2]=1, neither catches (0.1 < 0.5)
    # label[3]=0, not gold-incorrect
    residual = flag_residual_regions(labels, scores_by_verifier, retained, threshold=0.5)
    assert residual == [2]
