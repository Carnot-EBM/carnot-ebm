import json

def check(file_path):
    correct = 0
    total = 0
    with open(file_path) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            # handle both formats
            c = d.get('correct', d.get('is_correct'))
            if c is None:
                label = d.get('label')
                if label:
                    c = (label == 'correct')
                else:
                    continue
            if c: correct += 1
            total += 1
    if total > 0:
        print(f"{file_path}: {correct/total:.3f} ({correct}/{total})")

for f in ["data/fover_corpus.jsonl", "data/p01_difficulty_matched_generations_flattened_v2.jsonl", "data/p01_gsm8k_generations.jsonl", "data/p01_hardmath_generations.jsonl"]:
    check(f)
