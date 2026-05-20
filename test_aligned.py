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
# y_true = 1 if hallucinated (incorrect)
y_true = [1 if d["label"] == "incorrect" else 0 for d in eval_data]
texts = [d["step_text"] for d in eval_data]

verifiers = {
    "tier0e": EORMVerifier(),
    "tier0f": SemanticCalibratedVerifier(),
    "tier0r": Tier0rVerifier(),
    "tier0s": Tier0sVerifier(),
    "tier0u": Tier0uVerifier()
}

names = ["tier0f", "tier0e", "tier0r", "tier0u", "tier0s"]
score_list = []
for name in names:
    v = verifiers[name]
    if hasattr(v, "verify"):
        s = [v.verify(t) for t in texts]
    elif hasattr(v, "score"):
        s = [v.score(t) for t in texts]
    else:
        s = [v.halluguard_ntk_score(t) for t in texts]
    score_list.append(s)

score_mat = np.array(score_list).T

for i in range(5):
    auc = roc_auc_score(y_true, score_mat[:, i])
    print(f"{names[i]}: {auc}")

aligned_mat = score_mat.copy()
for i in range(5):
    auc = roc_auc_score(y_true, score_mat[:, i])
    if auc < 0.5:
        # Scale to match direction
        aligned_mat[:, i] = 1.0 - score_mat[:, i]

# Wait, if they did k=5 uniform ensemble, did they average the aligned scores?
uniform_5 = np.mean(aligned_mat, axis=1)
print(f"Uniform 5 aligned AUROC: {roc_auc_score(y_true, uniform_5)}")
