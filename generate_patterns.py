import json
import os
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
        files = [
            "results/live_sota_balanced_telemetry_manifest_1480.jsonl",
            "results/live_sota_telemetry_manifest_1468.jsonl"
        ]
        unsat_examples = 0
        for fname in files:
            if unsat_examples >= 5:
                break
            if not os.path.exists(fname):
                continue
            with open(fname, "r") as f:
                for line in f:
                    if unsat_examples >= 5:
                        break
                    entry = json.loads(line)
                    result = extractor.verify(entry.get("response_text", ""))
                    if not result.get("satisfiable", True):
                        violations = result.get("violations", [])
                        if violations:
                            patterns.extend(violations)
                            unsat_examples += 1

        patterns = list(set(patterns))
        
        if not patterns:
            # Fallback for degenerate tests if no UNSAT found anywhere
            print("No UNSAT found, using fallback patterns")
            patterns = ["12 + 7 = 20", "20 / 3 = 7", "4 times 6 equals 25", "100 divided by 4 equals 26"]
    
    return patterns

if __name__ == "__main__":
    p = get_patterns()
    print("Found patterns:", p)
