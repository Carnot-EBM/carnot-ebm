import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "python"))
import numpy as np
from carnot.verify.verifier_ensemble_diversity import build_verifier_set, run_diversity_audit
from carnot.fr11.conservative_default_deploy_nondegenerate_corpus_v3 import run_arm_closed_loop

records = []
with open("data/p01_difficulty_matched_generations_flattened_v2.jsonl") as f:
    for line in f:
        d = json.loads(line)
        records.append({
            "step_text": d.get("text", d.get("step_text", "")),
            "label": "correct" if d.get("is_correct", d.get("correct")) else "incorrect",
            "is_correct": d.get("is_correct", d.get("correct"))
        })

verifiers = build_verifier_set(["z3_math", "ast_structure", "pcib_semantic", "rprm_heuristic", "semantic_consistency"])
audit = run_diversity_audit(records, verifiers)
scores_matrix = np.array(audit["scores_matrix"])
single_scores = scores_matrix[:, 0]

res_control = run_arm_closed_loop(records, single_scores, 200, 0.0, "test", "CONTROL")
print("Control Collapsed:", res_control["collapse_detected"], "Final Mode Mass:", res_control["final_mode_mass"], "Final Ent:", res_control["final_entropy"])
noisy_single_scores = single_scores + 0.01 * np.random.RandomState(42).random(len(single_scores))
res_control_noisy = run_arm_closed_loop(records, noisy_single_scores, 200, 0.0, "test", "CONTROL_NOISY")
print("Control Noisy Collapsed:", res_control_noisy["collapse_detected"], res_control_noisy["final_mode_mass"], res_control_noisy["final_entropy"])
