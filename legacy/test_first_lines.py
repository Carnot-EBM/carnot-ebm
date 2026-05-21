import json

with open("data/fover_corpus.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5: break
        print(line.strip())