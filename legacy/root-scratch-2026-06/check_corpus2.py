import json
from pathlib import Path
import numpy as np
REPO_ROOT = Path('.')
import sys
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.fr11 import continuous_self_learning_v10 as v10

labels, scores = v10.score_fover_corpus(REPO_ROOT, n_examples=1000, random_seed=42)

labels_arr = np.asarray(labels, dtype=np.int64)

def calc_auroc(ensemble_scores):
    return v10.exp3644.tie_aware_auroc(labels_arr, ensemble_scores)

ensemble_scores = 0.9 * np.asarray(scores['tier0r_curry_howard']) + \
                  0.1 * np.asarray(scores['tier0u_logical_consistency']) + \
                  np.asarray(scores['fr11_session_memory'])

full_auroc = calc_auroc(ensemble_scores)
print(f"Full AUROC: {full_auroc}")

for v in scores.keys():
    score_v = np.asarray(scores[v])
    print(f"{v}: {calc_auroc(score_v)}")
