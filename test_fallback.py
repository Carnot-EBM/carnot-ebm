import json
from test_headroom import compute_headroom_stats

records = []
with open("data/p01_difficulty_matched_generations.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        samples = rec.get("samples") or []
        if len(samples) >= 4:
            records.append(rec)

print(compute_headroom_stats(records))
