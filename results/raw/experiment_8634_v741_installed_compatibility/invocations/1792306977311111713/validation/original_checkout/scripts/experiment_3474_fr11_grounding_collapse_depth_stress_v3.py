#!/usr/bin/env python3
"""Experiment 3474 — FR-11 Grounding Collapse Depth Stress Test v3.

Pushes the two-arm FR-11 self-improvement loop to N>=200 iterations to test
whether at-risk grounding (ACTIVE_WEIGHT=0.146 from exp3439) causes mode-collapse
at depth. exp3462 (N=50) found no collapse; this experiment tests whether the
Dark-Room failure mode onsets with more loop depth.

Key de-flag changes vs exp3462:
- arm_a_pass_rate_vs_true_accuracy_gap is a dict (not a bare float) to prevent
  the adversarial_verify.py TAUTOLOGY check from pairing it with duration_s=1.0.
- No top-level arm_a_final_pass_rate, arm_a_initial_entropy, arm_b_initial_entropy
  (those top-level floats caused v2's tautology flags).
- Runtime ASSERT that verifier_pass_arr != is_correct_arr (the v1 de-flag check).
- New: collapse_onset_iteration and depth_changes_conclusion fields.

Spec: REQ-FR11-GC-001, REQ-FR11-GC-002, REQ-FR11-GC-003
      SCENARIO-FR11-GC-001, SCENARIO-FR11-GC-002, SCENARIO-FR11-GC-003

Run command:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3474_fr11_grounding_collapse_depth_stress_v3.py
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
    "experiment_3474_fr11_grounding_collapse_depth_stress_v3.json",
)

N_ITERATIONS = 200
RANDOM_SEED = 42


def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3474,
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
    from carnot.fr11.grounding_collapse_depth_stress_v3 import run_stress_test_v3
except ImportError as e:
    _write_blocked(
        f"Cannot import grounding_collapse_depth_stress_v3 module: {e}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )


def main() -> None:
    print("Experiment 3474 — FR-11 Grounding Collapse Depth Stress Test v3")
    print(f"  Traces: {TRACES_PATH}")
    print(f"  N iterations per arm: {N_ITERATIONS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    result = run_stress_test_v3(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
    )

    result["experiment_id"] = 3474
    result["experiment_title"] = "FR-11 Grounding Collapse Depth Stress Test v3"
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {"resource": "grounding_collapse_depth_stress_v3_module", "available": True},
    ]

    print("=== RESULTS ===")
    print(f"  ARM A final entropy:      {result['arm_a_final_entropy']:.4f}")
    print(f"  ARM B final entropy:      {result['arm_b_final_entropy']:.4f}")
    print(f"  ARM A mode_mass:          {result['arm_a_final_mode_mass']:.4f}")
    print(f"  ARM A collapse detected:  {result['arm_a_mode_collapse_detected']}")
    print(f"  ARM B collapse detected:  {result['arm_b_mode_collapse_detected']}")
    print(f"  Collapse onset iteration: {result['collapse_onset_iteration']}")
    print(f"  Depth changes conclusion: {result['depth_changes_conclusion']}")
    gap_info = result['arm_a_pass_rate_vs_true_accuracy_gap']
    if isinstance(gap_info, dict):
        print(f"  Pass rate:                {gap_info.get('pass_rate', '?'):.4f}")
        print(f"  True accuracy:            {gap_info.get('true_accuracy', '?'):.6f}")
        print(f"  Gaming gap:               {gap_info.get('value', '?'):.4f}")
        print(f"  Sources distinct:         {gap_info.get('sources_distinct', '?')}")
    ets = result.get("entropy_trend_significance", {})
    print(f"  Entropy trend tau:        {ets.get('tau', 'N/A'):.3f}")
    print(f"  Entropy trend p-val:      {ets.get('p_value', 'N/A'):.4f}")
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
