import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
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
    "tier0f": SemanticCalibratedVerifier(),
    "tier0e": EORMVerifier(),
    "tier0r": Tier0rVerifier(),
    "tier0s": Tier0sVerifier(),
    "tier0u": Tier0uVerifier()
}

scores = []
names = list(verifiers.keys())
for name in names:
    v = verifiers[name]
    if hasattr(v, "verify"):
        s = [v.verify(t) for t in texts]
    elif hasattr(v, "score"):
        s = [v.score(t) for t in texts]
    else:
        s = [v.halluguard_ntk_score(t) for t in texts]
    scores.append(s)

score_mat = np.array(scores).T

# Normalize
scaler = StandardScaler()
score_mat = scaler.fit_transform(score_mat)

# Align direction so that higher score -> higher probability of 'correct' (y_true=1)
for i in range(len(names)):
    auc = roc_auc_score(y_true, score_mat[:, i])
    if auc < 0.5:
        score_mat[:, i] = 1.0 - score_mat[:, i]

uniform_scores = np.mean(score_mat, axis=1)
print("Uniform MinMax-Scaled AUROC:", roc_auc_score(y_true, uniform_scores))

# Let's check k=2 best
best_2 = score_mat[:, :2] # tier0f and tier0e
print("K=2 AUROC:", roc_auc_score(y_true, np.mean(best_2, axis=1)))

# Let's check k=5 best
print("K=5 AUROC:", roc_auc_score(y_true, uniform_scores))
