import json
import random

def check(file_path):
    correct = 0
    total = 0
    with open(file_path) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            if 'samples' in d:
                for s in d['samples']:
                    if s.get('correct'): correct += 1
                    total += 1
    if total > 0:
        print(f"{file_path}: {correct/total:.3f} ({correct}/{total})")

for f in ["data/p01_gsm8k_generations.jsonl", "data/p01_hardmath_generations.jsonl"]:
    check(f)
