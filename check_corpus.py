import json
from pathlib import Path
REPO_ROOT = Path('.')
import sys
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.fr11 import continuous_self_learning_v10 as v10

labels, scores = v10.score_fover_corpus(REPO_ROOT, n_examples=1000, random_seed=42)
print(scores.keys())
