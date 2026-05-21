import numpy as np
from carnot.verify.diversity_selection import diversity_select

def test_diversity_select_basic():
    # 4 samples, 3 verifiers
    # labels: [1, 0, 1, 0]
    labels = np.array([1, 0, 1, 0])
    
    # Verifier 0: Perfect on [0, 1] but fails [2, 3]
    # Verifier 1: Perfect on [2, 3] but fails [0, 1]
    # Verifier 2: Random garbage
    scores = np.array([
        [0.9, 0.1, 0.2, 0.8], # V0: auc=0.5 (preds: 1, 0, 0, 1) -> correct on 0, 1; missed 2, 3
        [0.2, 0.8, 0.9, 0.1], # V1: auc=0.5 (preds: 0, 1, 1, 0) -> correct on 2, 3; missed 0, 1
        [0.1, 0.1, 0.1, 0.1], # V2: auc=0.5 (preds: 0, 0, 0, 0) -> correct on 1, 3; missed 0, 2
    ]).T
    
    # Actually, let's make V0 clearly the best AUROC to start with
    # V0: [0.9, 0.1, 0.9, 0.9] -> labels [1, 0, 1, 0]
    # auc: 1s vs 0s: 1s have scores [0.9, 0.9], 0s have scores [0.1, 0.9]
    scores = np.array([
        [0.9, 0.1, 0.9, 0.9], # V0 (highest AUROC, preds: 1, 0, 1, 1). Misses sample 3 (true=0, pred=1).
        [0.1, 0.9, 0.9, 0.1], # V1 (perfect on sample 3, pred=0. true=0)
        [0.9, 0.9, 0.9, 0.9], # V2 (all 1s)
    ]).T
    
    selected = diversity_select(scores, labels, k_target=2)
    assert len(selected) == 2
    assert selected[0] == 0
    # On missed sample 3 (label=0), V1 predicts 0 (correct), V2 predicts 1 (wrong)
    # V1 maximizes F1/accuracy on the missed subset
    assert selected[1] == 1

def test_diversity_select_no_missed():
    labels = np.array([1, 0, 1, 0])
    # V0 is perfect
    scores = np.array([
        [0.9, 0.1, 0.9, 0.1], # V0
        [0.1, 0.9, 0.1, 0.9], # V1
    ]).T
    selected = diversity_select(scores, labels, k_target=2)
    assert len(selected) == 2
    assert selected[0] == 0
    assert selected[1] == 1 # Fallback since no misses
