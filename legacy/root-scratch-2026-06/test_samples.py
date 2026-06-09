import json
with open("data/p01_greedy_wrong_headroom_corpus.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        samples = rec.get("samples", [])
        extracted = [s.get("extracted_answer") for s in samples]
        print(f"extracted: {extracted}")
