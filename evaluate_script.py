import json
from sklearn.metrics import roc_auc_score
from carnot.verify.hierarchical_logcons import HierarchicalLogConsVerifier
import time

def main():
    verifier = HierarchicalLogConsVerifier()
    
    with open('/home/ianblenke/github.com/ianblenke/carnot/results/live_sota_balanced_telemetry_manifest_1480.jsonl', 'r') as f:
        entries = [json.loads(line) for line in f][:36]
        
    scores = []
    labels = []
    
    violations = 0
    all_z3_used = True
    
    results = []
    
    for i, e in enumerate(entries):
        res = verifier.verify(e)
        s = res['logcons_score']
        label = 0 if e.get('correct', False) else 1
        
        scores.append(s)
        labels.append(label)
        
        results.append({
            "idx": i,
            "score": s,
            "label": label
        })
        
        if res.get('hierarchy_violation'):
            violations += 1
        if not res.get('z3_encoding_used'):
            all_z3_used = False
            
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5
        
    print(f"AUROC: {auroc}")
    print(f"Violations: {violations}")
    print(f"All Z3 Used: {all_z3_used}")

    out_scores = {
        "verifier": "logcons_z3_true",
        "scores": results
    }
    with open('/home/ianblenke/github.com/ianblenke/carnot/results/experiment_2437_logcons_z3_scores.json', 'w') as f:
        json.dump(out_scores, f)

if __name__ == '__main__':
    main()