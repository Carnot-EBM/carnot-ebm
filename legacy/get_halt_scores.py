from carnot.verify.halt_probe import HaltProbeDetector, _read_jsonl, label_from_entry
import numpy as np

rows = _read_jsonl("results/live_sota_balanced_telemetry_manifest_1480.jsonl", limit=36)
labels = [label_from_entry(r) for r in rows]
detector = HaltProbeDetector()
detector.fit(rows, labels)
scores = [detector.verify(r)["halt_risk_score"] for r in rows]
print("HALT SCORES:", len(scores))
print(scores[:5])
