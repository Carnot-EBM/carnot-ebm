#!/usr/bin/env python3
"""Experiment 3521 — FR-11 Adaptive Online Beta Robust Default v1.

WHAT THIS DOES:
    exp3509 (.323) found the static offline law beta = f(lambda_min(Sigma_initial))
    did not generalize to fresh configs and fell back to a conservative default.
    This module evaluates the true Phase-5 deployable rule: ADAPTIVE ONLINE beta.
    Instead of measuring lambda_min once at t=0, we measure the probability-weighted
    lambda_min(Sigma_t) online at each step. As the distribution collapses, diversity
    drops, lambda_min drops, and the deployed beta naturally increases to halt collapse.

    Arms:
      Arm A: Adaptive online beta = clamp(f(lambda_min_t), beta_floor)
      Arm B: beta=0.0 (control — expects collapse)
      Arm C: beta=0.5 (fixed-conservative baseline)
      Arm D: Static offline law (exp3498 formula at t=0 lambda_min)

SPEC:
    REQ-FR11-CLD-004: Evaluate adaptive online beta vs conservative default.

RUN:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3521_fr11_adaptive_online_beta_robust_default_v1.py
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
    "experiment_3521_fr11_adaptive_online_beta_robust_default_v1.json",
)

N_ITERATIONS = 400

def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3521,
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


if not os.path.exists(TRACES_PATH):
    _write_blocked(
        f"Cached traces not found at {TRACES_PATH}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )

try:
    from carnot.fr11.adaptive_online_beta_robust_default_v1 import (
        FRESH_CONFIGS,
        LAW_INTERCEPT,
        LAW_SLOPE,
        RANDOM_SEED,
        run_adaptive_online_beta_robust_default,
    )
except ImportError as e:
    _write_blocked(
        f"Cannot import adaptive_online_beta_robust_default_v1: {e}",
        "complete: blocked_verifier_suite_uncallable",
    )


def main() -> None:
    print("Experiment 3521 — FR-11 Adaptive Online Beta Robust Default v1")
    print(f"  Traces:      {TRACES_PATH}")
    print(f"  N iterations: {N_ITERATIONS}")
    print(f"  Law:         beta = {LAW_SLOPE:.4f} * lambda_min + ({LAW_INTERCEPT:.4f})")
    print(f"  Fresh configs: {[c['name'] for c in FRESH_CONFIGS]}")
    print(f"  Random seed:  {RANDOM_SEED}  (content-derived, not 3521)")
    print()

    result = run_adaptive_online_beta_robust_default(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
        fresh_configs=FRESH_CONFIGS,
    )

    result["experiment_id"] = 3521
    result["experiment_title"] = "FR-11 Adaptive Online Beta Robust Default v1"
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {"resource": "adaptive_online_beta_robust_default_v1", "available": True},
    ]

    print()
    print("=== RESULTS ===")
    print(f"  n_grounding_configs:       {result['n_grounding_configs']}")
    print(f"  collapse_A (adaptive):     {result['collapse_detected_armA_adaptive_online']}")
    print(f"  collapse_B (beta=0):       {result['collapse_detected_armB_beta0']}")
    print(f"  collapse_C (fixed-0.5):    {result['collapse_detected_armC_conservative']}")
    print(f"  collapse_D (static-law):   {result['collapse_detected_armD_static_offline_law']}")
    print(f"  adaptive_prevents_collapse: {result['adaptive_online_prevents_collapse']}")
    print(f"  conservative_prevents:     {result['conservative_default_prevents_collapse']}")
    print(f"  winning_arm_vs_least_reg_gap: {result['winning_arm_vs_least_regularized_accuracy_gap']:.6f}")
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
