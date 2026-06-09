import json
from pathlib import Path
import numpy as np
REPO_ROOT = Path('.')
import sys
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.fr11 import continuous_self_learning_v10 as v10

def evaluate(seed, drop_verifier=None):
    labels, scores = v10.score_fover_corpus(REPO_ROOT, n_examples=1000, random_seed=seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    
    # Base weights
    weights = {
        'tier0r_curry_howard': 0.9,
        'tier0u_logical_consistency': 0.1,
        'fr11_session_memory': 1.0,
        'tier0s_arithmetic_gap': 0.0
    }
    
    if drop_verifier:
        weights[drop_verifier] = 0.0
        
    ensemble_scores = weights['tier0r_curry_howard'] * np.asarray(scores['tier0r_curry_howard']) + \
                      weights['tier0u_logical_consistency'] * np.asarray(scores['tier0u_logical_consistency']) + \
                      weights['fr11_session_memory'] * np.asarray(scores['fr11_session_memory']) + \
                      weights['tier0s_arithmetic_gap'] * np.asarray(scores['tier0s_arithmetic_gap'])
    return v10.exp3644.tie_aware_auroc(labels_arr, ensemble_scores)

seeds = [42, 137, 271, 314, 1729]
full_auroc = np.mean([evaluate(s) for s in seeds])
print(f"Full AUROC: {full_auroc}")

verifiers = ['fr11_session_memory', 'tier0r_curry_howard', 'tier0s_arithmetic_gap', 'tier0u_logical_consistency']

for v in verifiers:
    dropped_auroc = np.mean([evaluate(s, drop_verifier=v) for s in seeds])
    print(f"Drop {v}: {dropped_auroc} (marginal: {full_auroc - dropped_auroc})")

