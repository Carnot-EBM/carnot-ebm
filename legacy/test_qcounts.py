import json
from collections import Counter
q_counts = Counter()
with open("data/fover_corpus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        qid = d.get("question_id")
        q_counts[qid] += 1
print("Unique questions:", len(q_counts))
print("Most common counts:", q_counts.most_common(5))
