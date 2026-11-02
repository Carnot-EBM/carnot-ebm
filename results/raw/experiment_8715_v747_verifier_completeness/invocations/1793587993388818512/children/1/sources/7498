#!/usr/bin/env python3
"""Experiment 3509 — FR-11 Closed-Loop Beta Law Deployment v1.

WHAT THIS DOES:
    exp3498 (.322) fitted a predictive law: beta_min = -0.3001 + 1.8461 * lambda_min
    (R²=0.989, validated out-of-sample). This experiment DEPLOYS that law on >=2
    FRESH grounding configurations (AW NOT in the exp3498 fit set) by:
      (a) Measuring lambda_min from the k×k verifier-decision covariance on cached traces.
      (b) Setting beta = f(lambda_min) per the fitted formula.
      (c) Running the FR-11 self-improvement loop to N>=200 under THREE arms:
              Arm A: beta = f(lambda_min)  [deployed law]
              Arm B: beta = 0              [collapse control]
              Arm C: beta = 0.5            [fixed-conservative baseline]
      (d) Validating: Arm A prevents collapse at ALL configs AND Arm B collapses.

    Independent precedent: ER-PRM (arXiv:2412.11006) entropy regularization.
    Cached traces + cached verifier scoring → no live LLM → no timeout.

SPEC:
    REQ-FR11-CLD-001, REQ-FR11-CLD-002, REQ-FR11-CLD-003
    SCENARIO-FR11-CLD-001, SCENARIO-FR11-CLD-002, SCENARIO-FR11-CLD-003

RUN:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3509_fr11_closed_loop_beta_law_deployment_v1.py
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

TRACES_PATH = os.path.join(REPO_ROOT, "data", "fr11_zenil_distill_v2.jsonl")
RESULT_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "experiment_3509_fr11_closed_loop_beta_law_deployment_v1.json",
)

N_ITERATIONS = 200


def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3509,
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "error": reason,
        "preconditions_checked": [
            {"resource": "fr11_traces_or_module", "available": False, "detail": reason},
        ],
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"BLOCKED: {reason}")
    print(f"Artifact written to {RESULT_PATH}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# PRECONDITIONS (step 0) — checked BEFORE any computation
# ---------------------------------------------------------------------------

# a. FR-11 cached traces present
if not os.path.exists(TRACES_PATH):
    _write_blocked(
        f"Cached traces not found at {TRACES_PATH}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )

# b. The deployment module is importable + verifier ensemble callable
try:
    from carnot.fr11.closed_loop_beta_law_deployment_v1 import (
        FRESH_CONFIGS,
        LAW_INTERCEPT,
        LAW_SLOPE,
        RANDOM_SEED,
        run_closed_loop_beta_law_deployment,
    )
except ImportError as e:
    _write_blocked(
        f"Cannot import closed_loop_beta_law_deployment_v1: {e}",
        "complete: blocked_verifier_suite_uncallable",
    )


def main() -> None:
    print("Experiment 3509 — FR-11 Closed-Loop Beta Law Deployment v1")
    print(f"  Traces:      {TRACES_PATH}")
    print(f"  N iterations: {N_ITERATIONS}")
    print(f"  Law:         beta = {LAW_SLOPE:.4f} * lambda_min + ({LAW_INTERCEPT:.4f})")
    print(f"  Fresh configs: {[c['name'] for c in FRESH_CONFIGS]}")
    print(f"  Random seed:  {RANDOM_SEED}  (content-derived, not 3509)")
    print()

    result = run_closed_loop_beta_law_deployment(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
        fresh_configs=FRESH_CONFIGS,
    )

    result["experiment_id"] = 3509
    result["experiment_title"] = "FR-11 Closed-Loop Beta Law Deployment v1"
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {"resource": "closed_loop_beta_law_deployment_v1_module", "available": True},
    ]

    print()
    print("=== RESULTS ===")
    print(f"  n_grounding_configs:       {result['n_grounding_configs']}")
    print(f"  lambda_min_by_config:      {result['lambda_min_by_config']}")
    print(f"  beta_deployed_by_config:   {result['beta_deployed_by_config']}")
    print(f"  collapse_A (deployed):     {result['collapse_detected_armA_deployed']}")
    print(f"  collapse_B (beta=0):       {result['collapse_detected_armB_beta0']}")
    print(f"  collapse_C (fixed-0.5):    {result['collapse_detected_armC_fixed']}")
    print(f"  deployed_law_prevents_collapse: {result['deployed_law_prevents_collapse']}")
    print(f"  armA_vs_armC_accuracy_gap: {result['armA_vs_armC_accuracy_gap']:.6f}")
    print(f"  pass_rate_vs_true_acc_distinct: {result['pass_rate_vs_true_accuracy_distinct_assert']}")
    print(f"  Acceptance gates: {result['acceptance_gates']}")
    print()
    print(f"Honest verdict: {result['honest_verdict']}")
    print()
    print(f"Recommended Phase-5 rule:")
    print(f"  {result['recommended_phase5_rule']}")
    print()
    print(f"Duration: {result['duration_s']:.3f}s")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nArtifact written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
