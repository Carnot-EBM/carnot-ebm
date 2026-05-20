import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import connected_components
import time
import os

start_time = time.time()

# 1. Load FoVer corpus
with open("data/fover_corpus.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

texts = [d["step_text"] for d in data]
labels = [1 if d["label"] == "correct" else 0 for d in data]

# 2. Vectorize for clustering
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(texts)

# Compute cosine similarity
similarity = X.dot(X.T)
similarity.setdiag(0)

# We want edges where similarity >= 0.9
mask = similarity.data >= 0.9
filtered_similarity = similarity.copy()
filtered_similarity.data[~mask] = 0
filtered_similarity.eliminate_zeros()

n_components, labels_comp = connected_components(csgraph=filtered_similarity, directed=False, return_labels=True)

from collections import defaultdict
clusters = defaultdict(list)
for i, comp_id in enumerate(labels_comp):
    clusters[comp_id].append(i)

# Count paraphrase clusters (size > 1)
paraphrase_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
n_paraphrase_clusters = len(paraphrase_clusters)
n_pairs_in_clusters = sum(len(v) for v in paraphrase_clusters.values())

# Assign shared labels (majority vote)
shared_labels = np.array(labels)
for comp_id, indices in clusters.items():
    if len(indices) > 1:
        cluster_labels = [labels[i] for i in indices]
        majority_label = 1 if sum(cluster_labels) >= len(cluster_labels) / 2.0 else 0
        for i in indices:
            shared_labels[i] = majority_label

# 3. Cluster-aware train/eval split
comp_ids = list(clusters.keys())
np.random.seed(42)
np.random.shuffle(comp_ids)

split_idx = int(len(comp_ids) * 0.8)
train_comp_ids = set(comp_ids[:split_idx])

train_indices = []
eval_indices = []
for comp_id, indices in clusters.items():
    if comp_id in train_comp_ids:
        train_indices.extend(indices)
    else:
        eval_indices.extend(indices)

X_train = X[train_indices]
y_train = shared_labels[train_indices]
X_eval = X[eval_indices]
y_eval = shared_labels[eval_indices]

# 4. Train logistic regression
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

y_pred_proba = clf.predict_proba(X_eval)[:, 1]

tier0f_auroc = roc_auc_score(y_eval, y_pred_proba)

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if i == n_bins - 1:
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / len(y_prob)) * np.abs(bin_acc - bin_conf)
    return float(ece)

tier0f_ece = compute_ece(y_eval, y_pred_proba)

eval_paraphrase_indices = [i for i in range(len(eval_indices)) if len(clusters[labels_comp[eval_indices[i]]]) > 1]
if len(eval_paraphrase_indices) > 0:
    y_eval_para = y_eval[eval_paraphrase_indices]
    y_pred_para = y_pred_proba[eval_paraphrase_indices] >= 0.5
    neg_mask = (y_eval_para == 0)
    if np.sum(neg_mask) > 0:
        paraphrase_fpr = float(np.sum(y_pred_para[neg_mask]) / np.sum(neg_mask))
    else:
        paraphrase_fpr = 0.0
else:
    paraphrase_fpr = 0.0

with open("results/experiment_2663_tier0e_eorm.json", "r") as f:
    tier0e_data = json.load(f)
    tier0e_auroc = tier0e_data.get("tier0e_auroc", 0.0)

with open("results/tier0e_model.pkl", "rb") as f:
    tier0e_model_data = pickle.load(f)
    tier0e_vectorizer = tier0e_model_data["vectorizer"]
    tier0e_clf = tier0e_model_data["clf"]

X_eval_0e = tier0e_vectorizer.transform([texts[i] for i in eval_indices])
y_pred_proba_0e = tier0e_clf.predict_proba(X_eval_0e)[:, 1]
tier0e_ece = compute_ece(y_eval, y_pred_proba_0e)

auroc_delta = tier0f_auroc - tier0e_auroc
ece_improvement = tier0e_ece - tier0f_ece
tier0f_viable = bool(tier0f_auroc >= 0.63 and auroc_delta >= -0.05 and ece_improvement > 0)

duration_s = time.time() - start_time

result = {
    "honest_verdict": "complete:tier0f_semantic_calibration",
    "tier0f_auroc": float(tier0f_auroc),
    "tier0f_ece": float(tier0f_ece),
    "paraphrase_fpr": float(paraphrase_fpr),
    "auroc_delta": float(auroc_delta),
    "tier0f_viable": tier0f_viable,
    "n_paraphrase_clusters": int(n_paraphrase_clusters),
    "random_seed": 42,
    "duration_s": float(duration_s),
    "preconditions_checked": [
        {"resource": "sklearn", "available": True, "check": "import sklearn"},
        {"resource": "fover_corpus", "available": True, "check": "file exists"},
        {"resource": "tier0e", "available": True, "check": "file exists"}
    ]
}

with open("results/experiment_2716_tier0f_semantic_calibration_v2.json", "w") as f:
    json.dump(result, f, indent=2)

with open("results/tier0f_semantic_calibrated_logistic.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)

print(json.dumps(result, indent=2))
