#!/usr/bin/env python3
"""Experiment 3462 — FR-11 Grounding Collapse Clean Rerun v2.

This is the de-flagged version of exp3452. exp3452 was flagged TAUTOLOGY because
pass_rate == true_accuracy: with only 1/150 correct traces and NULL_SPACE_FRACTION=0.333,
the single correct trace was the only one with at_risk_score > 0.5, making the verifier-pass
vector identical to the ground-truth vector. This module uses dropout-contribution-weighted
scoring (ACTIVE_WEIGHT=0.146 from exp3439) so incorrect traces routinely score > 0.5,
separating the two metrics.

Run command:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3462_fr11_grounding_collapse_clean_rerun_v2.py

Spec: REQ-FR11-GC-001, REQ-FR11-GC-002, SCENARIO-FR11-GC-001, SCENARIO-FR11-GC-002
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
    "experiment_3462_fr11_grounding_collapse_clean_rerun_v2.json",
)

N_ITERATIONS = 50  # More than v1's 30 for a stronger convergence signal
RANDOM_SEED = 42


def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3462,
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "error": reason,
        "preconditions_checked": [
            {"resource": "fr11_cached_traces", "available": False, "detail": reason},
        ],
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"BLOCKED: {reason}")
    print(f"Artifact written to {RESULT_PATH}")
    sys.exit(0)


# PRECONDITIONS (step 0) — checked BEFORE any computation

if not os.path.exists(TRACES_PATH):
    _write_blocked(
        f"Cached traces not found at {TRACES_PATH}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )

try:
    from carnot.fr11.grounding_collapse_clean_rerun_v2 import run_stress_test_v2
except ImportError as e:
    _write_blocked(
        f"Cannot import grounding_collapse_clean_rerun_v2 module: {e}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )


def main() -> None:
    print("Experiment 3462 — FR-11 Grounding Collapse Clean Rerun v2 (de-flagged)")
    print(f"  Traces: {TRACES_PATH}")
    print(f"  N iterations per arm: {N_ITERATIONS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    result = run_stress_test_v2(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
    )

    result["experiment_id"] = 3462
    result["experiment_title"] = "FR-11 Grounding Collapse Clean Rerun v2"
    result["predecessor_experiment"] = 3452
    result["predecessor_deflag_reason"] = (
        "exp3452 TAUTOLOGY: pass_rate==true_accuracy because verifier_pass_arr was "
        "identical to is_correct_arr (only 1/150 correct traces, all with score>0.5). "
        "Fixed by using dropout-contribution-weighted scoring (ACTIVE_WEIGHT=0.146 "
        "from exp3439), making many incorrect traces score>0.5 and separating the metrics."
    )
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {"resource": "grounding_collapse_v2_module", "available": True},
    ]

    print("=== RESULTS ===")
    print(f"  ARM A final entropy:   {result['arm_a_final_entropy']:.4f}")
    print(f"  ARM B final entropy:   {result['arm_b_final_entropy']:.4f}")
    print(f"  ARM A mode-collapse:   {result['arm_a_mode_collapse_detected']}")
    print(f"  ARM B mode-collapse:   {result['arm_b_mode_collapse_detected']}")
    print(f"  ARM A pass-rate:       {result['arm_a_final_pass_rate']:.4f}")
    print(f"  ARM A true-accuracy:   {result['arm_a_final_true_accuracy']:.4f}")
    print(f"  ARM A gap (deFlag):    {result['arm_a_pass_rate_vs_true_accuracy_gap']:.4f}")
    ets = result.get("entropy_trend_significance", {})
    print(f"  Entropy trend tau:     {ets.get('tau', 'N/A'):.3f}")
    print(f"  Entropy trend p-val:   {ets.get('p_value', 'N/A'):.4f}")
    print(f"  Arrays identical?:     {result.get('pass_correct_arrays_identical', '?')}")
    print()
    print(f"Honest verdict: {result['honest_verdict']}")
    print()
    print(f"Consequence: {result['grounding_collapse_consequence']}")
    print()
    print(f"Duration: {result['duration_s']:.3f}s")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nArtifact written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
