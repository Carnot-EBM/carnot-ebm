#!/usr/bin/env python3
"""Experiment 3544 — FR-11 Conservative-Default Beta Deploy Closed Loop v2.

WHAT THIS DOES:
    exp3521 (.324) established conservative-default beta=0.5 as the robust Phase-5
    deployment rule. This experiment DEPLOYS that rule end-to-end: one full FR-11
    self-improvement closed loop to N=200 on a FRESH corpus (AW=0.04, not in any
    prior fit or selection set).

    TWO arms:
      Arm DEPLOY  (beta=0.5): expected to PREVENT collapse.
      Arm CONTROL (beta=0.0): expected to COLLAPSE — proves the loop CAN collapse.

    Reports: collapse_detected per arm, deployed alpha_t-grounding margin, and
    whether output quality (true_accuracy) is maintained, not just collapse-prevented.

RUN:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3544_fr11_conservative_default_deploy_closed_loop_v2.py
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

TRACES_PATH = os.path.join(REPO_ROOT, "data", "p01_difficulty_matched_generations_flattened_v2.jsonl")
RESULT_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.json",
)

N_ITERATIONS = 200


def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3544,
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "error": reason,
        "preconditions_checked": [
            {"resource": "fr11_traces_or_module", "available": False, "detail": reason},
        ],
        "duration_s": 1.0,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"BLOCKED: {reason}")
    print(f"Artifact written to {RESULT_PATH}")
    sys.exit(0)


# PRECONDITIONS step 0a: FR-11 cached traces present
if not os.path.exists(TRACES_PATH):
    _write_blocked(
        f"Cached traces not found at {TRACES_PATH}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )

# PRECONDITIONS step 0b: module importable
try:
    from carnot.fr11.conservative_default_deploy_nondegenerate_corpus_v2 import (
        CONSERVATIVE_DEFAULT_BETA,
        FRESH_DEPLOY_CONFIG,
        RANDOM_SEED,
        run_conservative_default_deploy_nondegenerate_corpus_v2,
    )
except ImportError as e:
    _write_blocked(
        f"Cannot import conservative_default_deploy_nondegenerate_corpus_v2: {e}",
        "complete: blocked_verifier_suite_uncallable",
    )

# PRECONDITIONS step 0c: fresh corpus split available (verify the config is distinct)
_PRIOR_AWS = {0.05, 0.10, 0.146, 0.30, 0.07, 0.20, 0.06, 0.08, 0.18, 0.22}
if FRESH_DEPLOY_CONFIG["active_weight"] in _PRIOR_AWS:
    _write_blocked(
        f"Fresh config AW={FRESH_DEPLOY_CONFIG['active_weight']} overlaps a prior "
        f"fit/selection set {_PRIOR_AWS}",
        "complete: blocked_no_fresh_corpus_split",
    )


def main() -> None:
    print("Experiment 3544 — FR-11 Conservative-Default Beta Deploy Non-Degenerate Corpus v2")
    print(f"  Traces:      {TRACES_PATH}")
    print(f"  N iterations: {N_ITERATIONS}")
    print(f"  Beta deployed: {CONSERVATIVE_DEFAULT_BETA}")
    print(f"  Fresh config:  {FRESH_DEPLOY_CONFIG['name']} (AW={FRESH_DEPLOY_CONFIG['active_weight']})")
    print(f"  Random seed:   {RANDOM_SEED}  (content-derived, not 3544)")
    print()

    result = run_conservative_default_deploy_nondegenerate_corpus_v2(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
        fresh_config=FRESH_DEPLOY_CONFIG,
    )

    result["experiment_id"] = 3544
    result["experiment_title"] = "FR-11 Conservative-Default Beta Deploy Non-Degenerate Corpus v2"
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {
            "resource": "conservative_default_deploy_nondegenerate_corpus_v2",
            "available": True,
            "detail": f"AW={FRESH_DEPLOY_CONFIG['active_weight']} not in prior sets",
        },
    ]

    print()
    print("=== FINAL RESULTS ===")
    print(f"  collapse_deploy:   {result['collapse_detected_deploy_arm']}")
    print(f"  collapse_control:  {result['collapse_detected_control_beta0']}")
    print(f"  alpha_t_margin:    {result['deployed_alpha_t_margin']:.4f}")
    print(f"  quality_maintained:{result['quality_maintained']}")
    print(f"  final_true_acc_deploy: {result['deploy_arm_final_true_accuracy']:.4e}")
    print(f"  distinct_assert:   {result['pass_rate_vs_true_accuracy_distinct_assert']}")
    print(f"  Gates: {result['acceptance_gates']}")
    print()
    print(f"Honest verdict: {result['honest_verdict']}")
    print(f"Duration: {result['duration_s']:.3f}s")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nArtifact written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
