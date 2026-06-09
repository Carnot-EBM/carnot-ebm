import json
from collections import Counter

def check():
    n = 0
    sc_correct = 0
    oracle_correct = 0
    with open("data/p01_difficulty_matched_generations.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            samples = rec.get("samples", [])
            if len(samples) < 4: continue
            gold = rec.get("gold_answer_norm") or rec.get("gold_answer")
            if not gold: continue
            answers = [s.get("extracted_answer_norm") or s.get("extracted_answer") for s in samples]
            valid = [a for a in answers if a is not None]
            if not valid: continue
            n += 1
            if any(a == gold for a in valid):
                oracle_correct += 1
            scores = {}
            for i, ans in enumerate(valid):
                scores[ans] = scores.get(ans, 0.0) + 1.0 / (i + 1)
            voted = max(scores.items(), key=lambda x: x[1])[0]
            if voted == gold:
                sc_correct += 1
    print(f"n={n}, oracle={oracle_correct/n}, sc={sc_correct/n}")

check()
