import json
from pathlib import Path
import numpy as np
REPO_ROOT = Path('.')
import sys
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.fr11 import continuous_self_learning_v10 as v10

labels, scores = v10.score_fover_corpus(REPO_ROOT, n_examples=1000, random_seed=42)
labels_arr = np.asarray(labels, dtype=np.int64)

retained = ['tier0r_curry_howard', 'fr11_session_memory']
residual_regions = []

for i, label in enumerate(labels_arr):
    if label == 1:
        catches = [scores[v][i] >= 0.5 for v in retained]
        if not any(catches):
            residual_regions.append(i)

print(f"Residual regions: {len(residual_regions)}")
print(residual_regions[:10])
