import json
from pathlib import Path
import numpy as np
REPO_ROOT = Path('.')
import sys
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.fr11 import continuous_self_learning_v10 as v10

def evaluate(seed, drop_verifiers=None):
    labels, scores = v10.score_fover_corpus(REPO_ROOT, n_examples=1000, random_seed=seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    
    # Base weights
    weights = {
        'tier0r_curry_howard': 0.9,
        'tier0u_logical_consistency': 0.1,
        'fr11_session_memory': 1.0,
        'tier0s_arithmetic_gap': 0.0
    }
    
    if drop_verifiers:
        for v in drop_verifiers:
            weights[v] = 0.0
            
    ensemble_scores = sum(weights[v] * np.asarray(scores[v]) for v in weights)
    return v10.exp3644.tie_aware_auroc(labels_arr, ensemble_scores)

seeds = [42, 137, 271, 314, 1729]
pruned = ['tier0s_arithmetic_gap', 'tier0u_logical_consistency']
pruned_auroc = np.mean([evaluate(s, drop_verifiers=pruned) for s in seeds])
print(f"Pruned AUROC: {pruned_auroc}")
