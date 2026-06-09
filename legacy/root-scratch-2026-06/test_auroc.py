import json
from carnot.verify.and_composition_verifier import build_default_verifier_ensemble
from sklearn.metrics import roc_auc_score

with open("data/fover_corpus.jsonl") as f:
    rows = [json.loads(line) for line in f]

v = build_default_verifier_ensemble()

scores = []
labels = []
for row in rows:
    res = v.verify("", row.get("step_text", ""))
    scores.append(max(res.per_verifier_scores.values()))
    labels.append(1 if row["label"] in {"incorrect", 1, "1"} else 0)

print("Full ensemble AUROC:", roc_auc_score(labels, scores))
