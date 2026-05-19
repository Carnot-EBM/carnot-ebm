import json
import re
from sklearn.metrics import roc_auc_score

def score(response: str) -> float:
    penalty = 0.0
    if re.match(r'^\s*\d+', response):
        penalty += 0.8
    
    if "initial state" in response.lower() or "constraint:" in response.lower() or "noah buys" in response.lower():
        penalty += 0.6
        
    if "claim to" in response.lower():
        penalty += 0.4
        
    if "command" in response.lower():
        penalty += 0.5

    penalty += len(response) * 0.001
    return min(1.0, penalty)

labels = []
scores = []

with open('/home/ianblenke/github.com/ianblenke/carnot/results/live_sota_balanced_telemetry_manifest_1480.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        labels.append(1 if not entry.get('correct') else 0)
        scores.append(score(entry.get('response_text', '')))

auroc = roc_auc_score(labels, scores)
print(f"AUROC: {auroc}")
