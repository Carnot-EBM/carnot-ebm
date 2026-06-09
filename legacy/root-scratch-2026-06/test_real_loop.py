import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "python"))
import numpy as np
from carnot.verify.verifier_ensemble_diversity import build_verifier_set, run_diversity_audit
from carnot.fr11.conservative_default_deploy_nondegenerate_corpus_v3 import run_arm_closed_loop

# Load 200 samples
records = []
with open("data/p01_difficulty_matched_generations_flattened_v2.jsonl") as f:
    for line in f:
        d = json.loads(line)
        records.append({
            "step_text": d.get("text", d.get("step_text", "")),
            "label": "correct" if d.get("is_correct", d.get("correct")) else "incorrect",
            "is_correct": d.get("is_correct", d.get("correct"))
        })
        if len(records) >= 200: break

verifiers = build_verifier_set(["z3_math", "ast_structure", "pcib_semantic", "rprm_heuristic", "semantic_consistency"])
audit = run_diversity_audit(records, verifiers)
scores_matrix = np.array(audit["scores_matrix"])
# Single verifier (just the first one, e.g., z3_math)
single_scores = scores_matrix[:, 0]
# Diverse Weaver-style (average of all 5)
diverse_scores = scores_matrix.mean(axis=1)

res_single = run_arm_closed_loop(records, single_scores, 50, 0.5, "test", "SINGLE")
res_diverse = run_arm_closed_loop(records, diverse_scores, 50, 0.5, "test", "DIVERSE")
res_control = run_arm_closed_loop(records, single_scores, 50, 0.0, "test", "CONTROL")

print("Single:", res_single["final_true_accuracy"], "Collapsed:", res_single["collapse_detected"])
print("Diverse:", res_diverse["final_true_accuracy"], "Collapsed:", res_diverse["collapse_detected"])
print("Control:", res_control["final_true_accuracy"], "Collapsed:", res_control["collapse_detected"])
