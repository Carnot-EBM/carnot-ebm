import json
import numpy as np
from sklearn.metrics import roc_auc_score
from carnot.verify.laab_verifier import compute_laab_score

scores = []
labels = []
with open("/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/live_sota_balanced_telemetry_manifest_1480.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        score = compute_laab_score(entry)
        label = 1 if entry.get("correctness_label") != "correct" else 0
        scores.append(score)
        labels.append(label)

auroc = roc_auc_score(labels, scores)
print("Original AUROC:", auroc)
