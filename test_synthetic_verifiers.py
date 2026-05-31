import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "python"))
import numpy as np
from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import compute_decision_covariance, compute_sigma_metrics, _compute_at_risk_scores

traces = []
with open("data/p01_difficulty_matched_generations_flattened_v2.jsonl") as f:
    for line in f:
        traces.append(json.loads(line))
        if len(traces) >= 200: break

aw = 0.045
# Single verifier
sig_single = compute_decision_covariance(traces, aw, n_channels=1, seed=42)
print("Single metrics:", compute_sigma_metrics(sig_single))

# Diverse verifiers (e.g. Weaver-style combination)
# Weaver-style combination: average the scores of k verifiers.
def diverse_scores(traces, aw, k, seed):
    n = len(traces)
    scores = np.zeros(n)
    for j in range(k):
        rng_active = np.random.RandomState(seed)
        rng_null = np.random.RandomState(seed + 1000 + j)
        is_correct = np.array([bool(t.get("is_correct", t.get("correct", False))) for t in traces], dtype=float)
        active_signal = 0.9 * is_correct + 0.1 * rng_active.random(n)
        null_signal = rng_null.random(n)
        score_j = aw * active_signal + (1.0 - aw) * null_signal
        scores += score_j
    return scores / k

sig_diverse = compute_decision_covariance(traces, aw, n_channels=5, seed=42)
print("Diverse metrics:", compute_sigma_metrics(sig_diverse))
