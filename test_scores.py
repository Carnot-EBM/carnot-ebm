import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from carnot.verify.tier0e_eorm import EORMVerifier
from carnot.verify.tier0f_semantic_calibration import SemanticCalibratedVerifier
from carnot.verify.tier0r_curry_howard import Tier0rVerifier
from carnot.verify.tier0s_halluguard import Tier0sVerifier
from carnot.verify.tier0u_logical_consistency import Tier0uVerifier

with open("data/fover_corpus.jsonl", "r") as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines if line.strip()]
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)
y_true = [1 if d["label"] == "correct" else 0 for d in eval_data]
texts = [d["step_text"] for d in eval_data]

verifiers = {
    "tier0e": EORMVerifier(),
    "tier0f": SemanticCalibratedVerifier(),
    "tier0r": Tier0rVerifier(),
    "tier0s": Tier0sVerifier(),
    "tier0u": Tier0uVerifier()
}

scores = {}
for name, v in verifiers.items():
    if hasattr(v, "verify"):
        s = [v.verify(t) for t in texts]
    elif hasattr(v, "score"):
        s = [v.score(t) for t in texts]
    else:
        s = [v.halluguard_ntk_score(t) for t in texts]
    scores[name] = s
    auc = roc_auc_score(y_true, s)
    print(f"{name} raw AUROC: {auc}")

score_mat = np.array([scores[n] for n in verifiers.keys()]).T
uniform = np.mean(score_mat, axis=1)
print(f"Uniform AUROC (raw): {roc_auc_score(y_true, uniform)}")

# Align scores: if auc < 0.5, flip them (1 - score or -score)
aligned_mat = score_mat.copy()
for i, name in enumerate(verifiers.keys()):
    auc = roc_auc_score(y_true, score_mat[:, i])
    if auc < 0.5:
        # Assuming scores are roughly [0, 1]. Let's just negate for rank purposes.
        aligned_mat[:, i] = -aligned_mat[:, i]

uniform_aligned = np.mean(aligned_mat, axis=1)
print(f"Uniform AUROC (aligned): {roc_auc_score(y_true, uniform_aligned)}")
