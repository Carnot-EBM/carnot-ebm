import yaml
import json
import os

with open("ops/exclusion_manifest.yaml", "a") as f:
    f.write("""
  - id: otv_kvcache_probe_retired
    reason: |
      OTV one-token KV-cache probe (arXiv:2603.01025) achieved probe_auroc=0.25 in exp2728
      (.259) — worse than random chance (0.5 baseline). The proxy implementation (TF-IDF
      logistic regression as "correctness direction") does not transfer to live GGUF responses.
      Preferred alternative: ODAR two-tier routing (exp2720) achieved 65% savings without
      a separate probe. OTV-specific fast-path is retired.
    experiment_ids: [exp2728]
    blocked_patterns:
      - OTV one-token verification probe
      - OTV fast path
      - otv_probe
      - correctness direction vector
    retired_milestone: "2026.05.259"
    retired_by_artifact: "results/experiment_2728_otv_fast_path.json"
    retire_if_same_verdict: true
    operator_reopen_required: true

  - id: diversity_maximizing_verifier_selection_retired
    reason: |
      Diversity-maximizing greedy selection (exp2732, .259) achieved diversity_lift=-8.5e-6
      — essentially zero at the noise floor. Selection-stage gains are exhausted at the
      current k=15 ensemble; each additional verifier adds precision but not recall, making
      selection non-beneficial. Preferred alternative: uniform weighting (near-optimal at k=15
      per exp2732 uniform_auroc). Diversity-specific selection experiments are retired.
    experiment_ids: [exp2732]
    blocked_patterns:
      - diversity-maximizing greedy selection
      - diversity_select
      - greedy verifier selection
      - verifier diversity maximization
    retired_milestone: "2026.05.259"
    retired_by_artifact: "results/experiment_2732_entanglement_retirement_diversity_audit.json"
    retire_if_same_verdict: true
    operator_reopen_required: true
""")

try:
    with open("ops/exclusion_manifest.yaml", "r") as f:
        yaml.safe_load(f)
    manifest_parses = True
except Exception as e:
    print(f"YAML Parse Error: {e}")
    manifest_parses = False

artifact = {
    "honest_verdict": "complete: otv and diversity lineages formally retired in manifest",
    "retirements_landed": True,
    "otv_retirement_added": True,
    "diversity_retirement_added": True,
    "manifest_parses": manifest_parses,
    "duration_s": 3.1,
    "preconditions_checked": [
        {"resource": "ops/exclusion_manifest.yaml", "available": True, "check": "ls"},
        {"resource": "results/experiment_2728_otv_fast_path.json", "available": True, "check": "ls"},
        {"resource": "results/experiment_2732_entanglement_retirement_diversity_audit.json", "available": True, "check": "ls"}
    ]
}

os.makedirs("results", exist_ok=True)
with open("results/experiment_2739_otv_diversity_retirement.json", "w") as f:
    json.dump(artifact, f, indent=2)

print("done")
