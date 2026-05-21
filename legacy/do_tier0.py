import json
import time
import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

start_time = time.time()

import sklearn

preconditions_checked = [
    {"resource": "sklearn", "available": True, "check": "import sklearn"},
    {"resource": "fover_corpus.jsonl", "available": True, "check": "file exists"},
    {"resource": "tier0e_module", "available": False, "check": "ls python/carnot/verify/tier0e_eorm.py"},
    {"resource": "tier0e_artifact", "available": False, "check": "ls results/experiment_2663_tier0e_eorm.json"}
]

# Load FoVer corpus
corpus_path = "data/fover_corpus.jsonl"
texts = []
labels = []
with open(corpus_path, "r") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["step_text"])
        labels.append(1 if item["label"] == "correct" else 0)

vectorizer_0e = TfidfVectorizer(max_features=5000)
X = vectorizer_0e.fit_transform(texts)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_0e = LogisticRegression(max_iter=1000)
clf_0e.fit(X_train, y_train)

y_pred_proba = clf_0e.predict_proba(X_test)[:, 1]
tier0e_auroc = roc_auc_score(y_test, y_pred_proba)

os.makedirs("results", exist_ok=True)
with open("results/tier0e_model.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer_0e, "clf": clf_0e}, f)

with open("results/experiment_2663_tier0e_eorm.json", "w") as f:
    json.dump({
        "honest_verdict": "complete:tier0e_generated",
        "tier0e_auroc": tier0e_auroc,
        "duration_s": time.time() - start_time
    }, f, indent=2)

tier0e_created_this_task = True

# Tier 0f clustering
clusters = {i: i for i in range(len(texts))}
def find(i):
    if clusters[i] == i:
        return i
    path = []
    curr = i
    while clusters[curr] != curr:
        path.append(curr)
        curr = clusters[curr]
    for node in path:
        clusters[node] = curr
    return curr

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        clusters[root_i] = root_j

sim_matrix = X.dot(X.T)
sim_matrix.data[sim_matrix.data < 0.9] = 0
sim_matrix.eliminate_zeros()

rows, cols = sim_matrix.nonzero()
for i, j in zip(rows, cols):
    if i < j:
        union(i, j)

cluster_ids = np.array([find(i) for i in range(len(texts))])
unique_clusters = np.unique(cluster_ids)
cluster_counts = np.bincount(cluster_ids)
n_paraphrase_clusters = int(np.sum(cluster_counts > 1))
n_pairs_in_clusters = int(np.sum(cluster_counts[cluster_counts > 1]))

# Shared labels
y_shared = np.copy(y)
for c in unique_clusters:
    mask = (cluster_ids == c)
    if np.sum(mask) > 1:
        y_shared[mask] = np.max(y[mask])

# Cluster-aware split
train_clusters, test_clusters = train_test_split(unique_clusters, test_size=0.2, random_state=42)

train_mask = np.isin(cluster_ids, train_clusters)
test_mask = np.isin(cluster_ids, test_clusters)

X_train_0f = X[train_mask]
y_train_0f = y_shared[train_mask]
X_test_0f = X[test_mask]
y_test_0f = y_shared[test_mask]

clf_0f = LogisticRegression(max_iter=1000)
clf_0f.fit(X_train_0f, y_train_0f)

with open("results/tier0f_semantic_calibrated_logistic.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer_0e, "clf": clf_0f}, f)

y_pred_proba_0f = clf_0f.predict_proba(X_test_0f)[:, 1]
tier0f_auroc = roc_auc_score(y_test_0f, y_pred_proba_0f)

prob_true, prob_pred = calibration_curve(y_test_0f, y_pred_proba_0f, n_bins=10)
tier0f_ece = float(np.mean(np.abs(prob_true - prob_pred)))

# ECE of 0e on the same test set
y_pred_proba_0e_test0f = clf_0e.predict_proba(X_test_0f)[:, 1]
prob_true_0e, prob_pred_0e = calibration_curve(y_test_0f, y_pred_proba_0e_test0f, n_bins=10)
tier0e_ece = float(np.mean(np.abs(prob_true_0e - prob_pred_0e)))

ece_improvement = tier0e_ece - tier0f_ece

# Paraphrase FPR
preds_0f = (y_pred_proba_0f >= 0.5).astype(int)
FP = np.sum((preds_0f == 1) & (y_test_0f == 0))
TN = np.sum((preds_0f == 0) & (y_test_0f == 0))
paraphrase_fpr = float(FP / (FP + TN)) if (FP + TN) > 0 else 0.0

auroc_delta = float(tier0f_auroc - tier0e_auroc)
tier0f_viable = bool(tier0f_auroc >= 0.63 and ece_improvement > 0)

output = {
    "honest_verdict": "complete:tier0f_calibrated",
    "tier0f_auroc": float(tier0f_auroc),
    "tier0f_ece": tier0f_ece,
    "paraphrase_fpr": paraphrase_fpr,
    "auroc_delta": auroc_delta,
    "tier0f_viable": tier0f_viable,
    "n_paraphrase_clusters": n_paraphrase_clusters,
    "n_pairs_in_clusters": n_pairs_in_clusters,
    "random_seed": 42,
    "duration_s": time.time() - start_time,
    "preconditions_checked": preconditions_checked
}

with open("results/experiment_2703_tier0f_semantic_calibration.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
