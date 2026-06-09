import json

in_path = "data/p01_difficulty_matched_generations.jsonl"
out_path = "data/p01_difficulty_matched_generations_flattened_v2.jsonl"

out_traces = []
with open(in_path) as f:
    for line in f:
        item = json.loads(line)
        for s in item.get("samples", []):
            trace = dict(s)
            trace["is_correct"] = s.get("correct", False)
            out_traces.append(trace)

with open(out_path, "w") as f:
    for t in out_traces:
        f.write(json.dumps(t) + "\n")

print(f"Wrote {len(out_traces)} traces to {out_path}")
