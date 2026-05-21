import json
from pathlib import Path
from carnot.verify.halt_probe import HALTTier0jProbe
rows = []
with Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl").open() as f:
    for i, line in enumerate(f):
        if i >= 36: break
        rows.append(json.loads(line))

probe = HALTTier0jProbe()
scores = probe.score(rows)
print("HALT scores generated:", len(scores))
