import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "python"))
from carnot.verify.verifier_ensemble_diversity import build_verifier_set, run_diversity_audit

records = []
with open("data/p01_difficulty_matched_generations_flattened_v2.jsonl") as f:
    for line in f:
        d = json.loads(line)
        # convert to fover format
        records.append({
            "step_text": d.get("text", ""),
            "label": "correct" if d.get("is_correct", d.get("correct")) else "incorrect"
        })

verifiers = build_verifier_set(["z3_math", "ast_structure", "pcib_semantic", "rprm_heuristic", "semantic_consistency"])
result = run_diversity_audit(records[:100], verifiers)
print("Lambda min:", result["lambda_min_sigma"])
