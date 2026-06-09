import json
from pathlib import Path
import numpy as np
REPO_ROOT = Path('.')
import sys
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.fr11 import continuous_self_learning_v10 as v10

def evaluate(seed):
    labels, scores = v10.score_fover_corpus(REPO_ROOT, n_examples=1000, random_seed=seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    ensemble_scores = 0.9 * np.asarray(scores['tier0r_curry_howard']) + \
                      0.1 * np.asarray(scores['tier0u_logical_consistency']) + \
                      np.asarray(scores['fr11_session_memory'])
    return v10.exp3644.tie_aware_auroc(labels_arr, ensemble_scores)

aurocs = [evaluate(s) for s in [42, 137, 271, 314, 1729]]
print(f"AUROCs: {aurocs}")
print(f"Mean AUROC: {np.mean(aurocs)}")
