import sys
import json
import time
import os

def get_ram_mib():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def median(lst):
    s = sorted(lst)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n//2]
    return (s[n//2 - 1] + s[n//2]) / 2.0

sys.path.insert(0, sys.argv[1])
from carnot.verify.and_composition_verifier import AndCompositionVerifier, ASTStructureAdapter, SemanticConsistencyAdapter

verifiers = [ASTStructureAdapter(), SemanticConsistencyAdapter()]
verifier = AndCompositionVerifier(verifiers=verifiers)

with open(sys.argv[2], "r") as f:
    rows = [json.loads(line) for line in f if line.strip()]

latencies = []
scores = []
labels = []

verifier.verify("", rows[0]["step_text"])

for row in rows:
    text = row["step_text"]
    label = row["label"]
    t0 = time.perf_counter()
    res = verifier.verify("", text)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000.0)
    
    max_score = max(res.per_verifier_scores.values()) if res.per_verifier_scores else 0.0
    scores.append(max_score)
    labels.append(1 if label == "incorrect" else 0)

peak_ram = get_ram_mib()

negatives = sum(1 for l in labels if l == 0)
positives = sum(1 for l in labels if l == 1)

unique_scores = sorted(list(set(scores)), reverse=True)
best_tpr = 0.0
for thresh in unique_scores:
    tp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l == 0)
    fpr = fp / negatives if negatives > 0 else 0.0
    if fpr <= 0.05:
        tpr = tp / positives if positives > 0 else 0.0
        if tpr > best_tpr:
            best_tpr = tpr

print(json.dumps({
    "n_examples": len(rows),
    "per_example_latency_ms_p50": median(latencies),
    "tpr_at_fpr5": best_tpr,
    "ram_peak_mib": peak_ram
}))
