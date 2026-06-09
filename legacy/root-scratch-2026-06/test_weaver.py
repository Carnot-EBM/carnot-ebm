import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "python"))
import numpy as np
from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _compute_at_risk_scores
from carnot.fr11.conservative_default_deploy_nondegenerate_corpus_v3 import run_arm_closed_loop

traces = []
with open("data/p01_difficulty_matched_generations_flattened_v2.jsonl") as f:
    for line in f:
        traces.append(json.loads(line))

aw = 0.045
k_diverse = 5
seed = 42

single_scores = _compute_at_risk_scores(traces, aw, seed)

diverse_scores = np.zeros(len(traces))
for j in range(k_diverse):
    diverse_scores += _compute_at_risk_scores(traces, aw, seed + j)
diverse_scores /= k_diverse

res_single = run_arm_closed_loop(traces, single_scores, 200, 0.5, "test", "SINGLE")
res_diverse = run_arm_closed_loop(traces, diverse_scores, 200, 0.5, "test", "DIVERSE")
res_control = run_arm_closed_loop(traces, single_scores, 200, 0.0, "test", "CONTROL")

print("Single Collapsed:", res_single["collapse_detected"])
print("Diverse Collapsed:", res_diverse["collapse_detected"])
print("Control Collapsed:", res_control["collapse_detected"])
print("Diverse helps (final true acc):", res_diverse["final_true_accuracy"] > res_single["final_true_accuracy"])
