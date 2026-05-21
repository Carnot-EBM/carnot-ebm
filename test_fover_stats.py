import json
from collections import Counter
correct = 0
incorrect = 0
with open("data/fover_corpus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("label") == "correct":
            correct += 1
        elif d.get("label") == "incorrect":
            incorrect += 1
print("Correct:", correct, "Incorrect:", incorrect)
