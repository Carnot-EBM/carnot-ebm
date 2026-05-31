import json
import sys

def check(path):
    traces = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except:
                    pass
    if not traces:
        print(f"{path}: empty")
        return
    is_correct = [float(bool(t.get("is_correct", False))) for t in traces]
    print(f"{path}: n={len(traces)}, true_acc={sum(is_correct)/len(traces):.4f}")

check("data/fover_corpus.jsonl")
check("data/p01_difficulty_matched_generations.jsonl")
check("data/fr11_zenil_distill_v2.jsonl")
