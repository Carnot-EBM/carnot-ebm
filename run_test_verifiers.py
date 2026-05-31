import sys
import os
import time

REPO_ROOT = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from carnot.verify.verifier_ensemble_diversity import build_verifier_set, run_diversity_audit, load_fover_corpus

records = load_fover_corpus(os.path.join(REPO_ROOT, "data/fover_corpus.jsonl"), max_examples=100)
verifiers = build_verifier_set(["z3_math", "ast_structure", "pcib_semantic", "rprm_heuristic", "semantic_consistency"])
result = run_diversity_audit(records, verifiers)
print("SUCCESS", result["lambda_min_sigma"])
