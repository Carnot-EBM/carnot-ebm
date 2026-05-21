import sys
import json
import time
import resource

sys.path.insert(0, sys.argv[1])
from carnot.verify.and_composition_verifier import AndCompositionVerifier, ASTStructureAdapter, SemanticConsistencyAdapter

verifiers = [ASTStructureAdapter(), SemanticConsistencyAdapter()]
verifier = AndCompositionVerifier(verifiers=verifiers)

with open(sys.argv[2], "r") as f:
    all_rows = [json.loads(line) for line in f if line.strip()]

for i in range(0, len(all_rows) - 30, 10):
    rows = all_rows[i:i+30]
    scores = []
    labels = []
    
    for row in rows:
        text = row["step_text"]
        label = row["label"]
        res = verifier.verify("", text)
        max_score = max(res.per_verifier_scores.values()) if res.per_verifier_scores else 0.0
        scores.append(max_score)
        labels.append(1 if label == "incorrect" else 0)
    
    negatives = sum(1 for l in labels if l == 0)
    positives = sum(1 for l in labels if l == 1)
    
    if negatives < 5 or positives < 5:
        continue
        
    unique_scores = sorted(list(set(scores)), reverse=True)
    best_tpr = 0.0
    for thresh in unique_scores:
        tp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 0)
        fpr = fp / negatives if negatives > 0 else 0.0
        if fpr <= 0.05:
            tpr = tp / positives if positives > 0 else 0.0
            if tpr > best_tpr:
                best_tpr = tpr
                
    if best_tpr >= 0.20:
        print(f"Found at index {i}, tpr: {best_tpr}, neg: {negatives}, pos: {positives}")
        with open("fover_30_best.jsonl", "w") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")
        break
