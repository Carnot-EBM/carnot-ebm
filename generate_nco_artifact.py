import json
import os
import time
from sklearn.metrics import roc_auc_score
from carnot.extraction.nsvif_extractor import NsvifExtractor

def get_patterns():
    patterns_file = "results/constraint_patterns_v4.json"
    patterns = []
    if os.path.exists(patterns_file):
        with open(patterns_file, "r") as f:
            data = json.load(f)
            patterns = data.get("patterns", [])
    
    if not patterns:
        print("Extracting patterns from telemetry...")
        extractor = NsvifExtractor()
        with open("results/live_sota_balanced_telemetry_manifest_1480.jsonl", "r") as f:
            unsat_count = 0
            for line in f:
                if unsat_count >= 5:
                    break
                entry = json.loads(line)
                result = extractor.verify(entry["response_text"])
                if not result.get("satisfiable", True):
                    # UNSAT!
                    violations = result.get("violations", [])
                    patterns.extend(violations)
                    unsat_count += 1
        
        # Save them just in case? Or just use them
        patterns = list(set(patterns))
    return patterns

patterns = get_patterns()
print("Found patterns:", patterns)
