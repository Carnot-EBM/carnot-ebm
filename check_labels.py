import json
with open("results/live_sota_balanced_telemetry_manifest_1480.jsonl", "r") as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        if entry.get("correctness_label") == "incorrect" or entry.get("correct") == False:
            print("Found incorrect at", i)
            break
