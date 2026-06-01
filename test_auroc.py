import json
from sklearn.metrics import roc_auc_score

data = []
with open("data/realistic_factual_corpus_v2.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

y_true = [1 - d["is_hallucination"] for d in data]
y_score = [d["model_confidence"] for d in data]

print(f"AUROC: {roc_auc_score(y_true, y_score)}")
