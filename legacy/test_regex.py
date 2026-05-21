import json
import re

n_real = 0
patterns = {}
with open("data/fover_corpus.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        entry = json.loads(line)
        if entry.get("label") != "incorrect":
            continue
            
        incorrect = entry.get("incorrect", entry.get("step_text", ""))
        words = re.findall(r'[a-zA-Z]+', incorrect)
        pattern = " ".join(words[:5]).lower()
        patterns[pattern] = patterns.get(pattern, 0) + 1
        
        n_real += 1
        if n_real >= 100:
            break
            
for p, c in patterns.items():
    if c >= 3:
        print(f"{c}: {p}")
