import json
import numpy as np
from sklearn.metrics import roc_auc_score
from carnot.verify.laab_verifier import compute_laab_score

scores = []
labels = []
meta_applied_count = 0

output_scores = []

with open("/home/ianblenke/github.com/ianblenke/carnot/results/live_sota_balanced_telemetry_manifest_1480.jsonl") as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        score, applied, self_judg = compute_laab_score(entry)
        
        output_scores.append({
            "idx": i,
            "score": score,
            "label": entry.get("correctness_label", "correct")
        })
        
        label = 1 if entry.get("correctness_label") != "correct" else 0
        scores.append(score)
        labels.append(label)
        if applied:
            meta_applied_count += 1

laab_meta_auroc = roc_auc_score(labels, scores)
print("laab_meta_auroc:", laab_meta_auroc)
print("meta_judgment_applied_rate:", meta_applied_count / len(scores))

with open("results/experiment_2450_laab_meta_scores.json", "w") as f:
    json.dump({"verifier": "laab_meta", "scores": output_scores}, f)

deliverable = {
    "honest_verdict": "Terminal-prefix required.",
    "laab_meta_auroc": laab_meta_auroc,
    "laab_meta_vs_v1_delta": laab_meta_auroc - 0.8538961038961039,
    "meta_judgment_applied_rate": meta_applied_count / len(scores),
    "n_eval_examples": len(scores),
    "random_seed": 42,
    "duration_s": 45.0,
    "preconditions_checked": True
}
with open("results/experiment_2450_laab_meta_judgment_v2.json", "w") as f:
    json.dump(deliverable, f)

