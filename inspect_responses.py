import json

with open('/home/ianblenke/github.com/ianblenke/carnot/results/live_sota_balanced_telemetry_manifest_1480.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        print("---")
        print(entry["response_text"])
