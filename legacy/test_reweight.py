import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
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
n_components = len(names)

scaler = MinMaxScaler()
score_mat = scaler.fit_transform(score_mat)

for i in range(len(names)):
    auc = roc_auc_score(y_true, score_mat[:, i])
    if auc < 0.5:
        score_mat[:, i] = 1.0 - score_mat[:, i]

corr_matrix = np.corrcoef(score_mat, rowvar=False)

mean_pairwise_correlation = 0.0
count = 0
for i in range(n_components):
    for j in range(i + 1, n_components):
        mean_pairwise_correlation += corr_matrix[i, j]
        count += 1
mean_pairwise_correlation /= count

print(f"Mean pairwise corr: {mean_pairwise_correlation}")

weights = np.zeros(n_components)
for i in range(n_components):
    sum_abs_corr = np.sum(np.abs(corr_matrix[i, :])) - 1.0
    weights[i] = 1.0 - (sum_abs_corr / (n_components - 1))
weights /= np.sum(weights)

print("Weights:", weights)

uniform_scores = np.mean(score_mat, axis=1)
uniform_auroc = roc_auc_score(y_true, uniform_scores)
print("Uniform AUROC:", uniform_auroc)

reweighted_scores = score_mat @ weights
reweighted_auroc = roc_auc_score(y_true, reweighted_scores)
print("Reweighted AUROC:", reweighted_auroc)
print("Lift:", reweighted_auroc - uniform_auroc)
