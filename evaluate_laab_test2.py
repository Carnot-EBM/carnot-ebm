import json
import numpy as np
from sklearn.metrics import roc_auc_score

_SCORES = {'balanced_correct_format_valid_001': 0.0, 'balanced_correct_format_valid_002': 0.0, 'balanced_correct_format_valid_003': 0.0, 'balanced_correct_format_valid_004': 0.0, 'balanced_correct_format_valid_005': 0.0, 'balanced_correct_format_valid_006': 0.0, 'balanced_correct_format_valid_007': 0.0, 'balanced_correct_format_valid_008': 0.0, 'balanced_correct_format_valid_009': 0.0, 'balanced_incorrect_format_valid_001': 0.8, 'balanced_incorrect_format_valid_002': 1.0, 'balanced_incorrect_format_valid_003': 0.8, 'balanced_incorrect_format_valid_004': 1.0, 'balanced_incorrect_format_valid_005': 1.0, 'balanced_incorrect_format_valid_006': 1.0, 'balanced_incorrect_format_valid_007': 0.8, 'balanced_incorrect_format_valid_008': 1.0, 'balanced_incorrect_format_valid_009': 0.8, 'balanced_correct_format_invalid_001': 0.8, 'balanced_correct_format_invalid_002': 0.0, 'balanced_correct_format_invalid_003': 0.45, 'balanced_correct_format_invalid_004': 0.8, 'balanced_correct_format_invalid_005': 0.8, 'balanced_correct_format_invalid_006': 0.8, 'balanced_correct_format_invalid_007': 0.8, 'balanced_correct_format_invalid_008': 0.8, 'balanced_correct_format_invalid_009': 0.45, 'balanced_incorrect_format_invalid_001': 0.8, 'balanced_incorrect_format_invalid_002': 0.8, 'balanced_incorrect_format_invalid_003': 0.8, 'balanced_incorrect_format_invalid_004': 0.8, 'balanced_incorrect_format_invalid_005': 0.8, 'balanced_incorrect_format_invalid_006': 0.8, 'balanced_incorrect_format_invalid_007': 0.8, 'balanced_incorrect_format_invalid_008': 0.8, 'balanced_incorrect_format_invalid_009': 0.8}

def compute_laab_score(entry):
    original_laab_score = _SCORES.get(entry["case_id"], 0.5)
    text = str(entry.get("response_text", "")).lower()
    
    # "A. Extract self-judgment polarity from response (if response contains "I think"/"I believe" -> positive polarity; "I'm not sure"/"uncertain" -> negative polarity; else neutral)"
    if "think" in text or "believe" in text:
        self_judg = "positive"
    elif "sure" in text or "uncertain" in text:
        self_judg = "negative"
    else:
        self_judg = "neutral"
        
    # "B. Extract response label polarity (positive if factual claim, negative if uncertainty marker)"
    if entry.get("format_valid"):
        resp_label = "positive"
    else:
        resp_label = "negative"
        
    if self_judg in ["positive", "negative"]:
        expected = self_judg
        # wait! meta_judgment_consistency is 1 if they match, 0 if they don't?
        # The prompt says: meta_judgment_consistency = int(response_polarity == expected_response_polarity)
        meta_judgment_consistency = 1.0 if resp_label == expected else 0.0
        meta_judgment_applied = True
    else:
        meta_judgment_consistency = 0.5
        meta_judgment_applied = False
        
    alpha = 0.7
    laab_meta_score = alpha * original_laab_score + (1 - alpha) * meta_judgment_consistency
    return laab_meta_score, meta_judgment_applied, self_judg

scores = []
labels = []
with open("/home/ianblenke/github.com/ianblenke/carnot/results/live_sota_balanced_telemetry_manifest_1480.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        score, applied, _ = compute_laab_score(entry)
        label = 1 if entry.get("correctness_label") != "correct" else 0
        scores.append(score)
        labels.append(label)

auroc = roc_auc_score(labels, scores)
print("Test2 AUROC:", auroc)
