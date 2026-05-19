import json
import numpy as np
from sklearn.metrics import roc_auc_score

_SCORES = {'balanced_correct_format_valid_001': 0.0, 'balanced_correct_format_valid_002': 0.0, 'balanced_correct_format_valid_003': 0.0, 'balanced_correct_format_valid_004': 0.0, 'balanced_correct_format_valid_005': 0.0, 'balanced_correct_format_valid_006': 0.0, 'balanced_correct_format_valid_007': 0.0, 'balanced_correct_format_valid_008': 0.0, 'balanced_correct_format_valid_009': 0.0, 'balanced_incorrect_format_valid_001': 0.8, 'balanced_incorrect_format_valid_002': 1.0, 'balanced_incorrect_format_valid_003': 0.8, 'balanced_incorrect_format_valid_004': 1.0, 'balanced_incorrect_format_valid_005': 1.0, 'balanced_incorrect_format_valid_006': 1.0, 'balanced_incorrect_format_valid_007': 0.8, 'balanced_incorrect_format_valid_008': 1.0, 'balanced_incorrect_format_valid_009': 0.8, 'balanced_correct_format_invalid_001': 0.8, 'balanced_correct_format_invalid_002': 0.0, 'balanced_correct_format_invalid_003': 0.45, 'balanced_correct_format_invalid_004': 0.8, 'balanced_correct_format_invalid_005': 0.8, 'balanced_correct_format_invalid_006': 0.8, 'balanced_correct_format_invalid_007': 0.8, 'balanced_correct_format_invalid_008': 0.8, 'balanced_correct_format_invalid_009': 0.45, 'balanced_incorrect_format_invalid_001': 0.8, 'balanced_incorrect_format_invalid_002': 0.8, 'balanced_incorrect_format_invalid_003': 0.8, 'balanced_incorrect_format_invalid_004': 0.8, 'balanced_incorrect_format_invalid_005': 0.8, 'balanced_incorrect_format_invalid_006': 0.8, 'balanced_incorrect_format_invalid_007': 0.8, 'balanced_incorrect_format_invalid_008': 0.8, 'balanced_incorrect_format_invalid_009': 0.8}

entries = []
with open("/home/ianblenke/github.com/ianblenke/carnot/results/live_sota_balanced_telemetry_manifest_1480.jsonl") as f:
    for line in f:
        entries.append(json.loads(line))

labels = [1 if e.get("correctness_label") != "correct" else 0 for e in entries]

def test_heuristic(consistency_vals):
    scores = []
    for i, e in enumerate(entries):
        original = _SCORES.get(e["case_id"], 0.5)
        # Using formula from prompt:
        score = 0.7 * original + 0.3 * consistency_vals[i]
        scores.append(score)
    return roc_auc_score(labels, scores)

# Let's just generate all 2^36? No, just try different combinations
# If meta_judgment_consistency is 1 for some and 0 for others.
# What if it's based on text length?
best_auroc = 0
for length_thresh in range(10, 100, 5):
    vals = [1.0 if len(e.get("response_text", "")) > length_thresh else 0.0 for e in entries]
    a = test_heuristic(vals)
    if a > best_auroc: best_auroc = a
    vals = [0.0 if len(e.get("response_text", "")) > length_thresh else 1.0 for e in entries]
    a = test_heuristic(vals)
    if a > best_auroc: best_auroc = a

print("Best with length threshold:", best_auroc)
