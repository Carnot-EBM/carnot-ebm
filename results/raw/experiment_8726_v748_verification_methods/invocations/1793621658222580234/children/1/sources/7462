#!/usr/bin/env python3
"""Experiment 3452 — FR-11 Grounding Collapse Stress Test v1.

**Research question:**
    exp3439 confirmed the production verifier ensemble is AT-RISK:
    lambda_min(Sigma) ≈ 0, effective-k ≈ 3.54, with 2 of 6 verifiers
    (pcib_semantic, length_antivacuity) contributing ≈ 0 discrimination.
    Does this at-risk grounding actually cause mode-collapse in the FR-11
    self-improvement loop (Hypothesis-B / Q12 Dark-Room)? Or does the
    residual eff-k 3.54 diversity hold the line?

**Methodology:**
    Two-arm simulation over cached traces (data/fr11_zenil_distill_v2.jsonl).
    No LLM loaded — this is a verifier_ensemble_against_cached_candidates run.

    ARM A (control): FR-11 self-improvement loop with NO entropy regularization.
    ARM B (treatment): Same loop WITH entropy regularization (Q12 antidote).

    Both arms run N_ITERATIONS=30 steps. We track per-arm: distribution entropy,
    mode mass (concentration), and verifier-pass rate. Collapse = entropy falls
    to near-0 while mode_mass → 1 and pass_rate rises (null-space gaming).

**Acceptance gates:**
    G1 COLLAPSE-CONFIRMED-AND-CURABLE:
        arm_a_mode_collapse_detected AND NOT arm_b_mode_collapse_detected
        → at-risk grounding causes loop collapse; entropy reg is the antidote.
    G1' GROUNDING-HOLDS:
        NOT arm_a_mode_collapse_detected
        → residual eff-k 3.54 diversity is sufficient; honest negative.

**Run command:**
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3452_fr11_grounding_collapse_stress_test_v1.py

Spec: REQ-FR11-GC-001, SCENARIO-FR11-GC-001, SCENARIO-FR11-GC-002
"""

from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# PRECONDITIONS (step 0) — checked BEFORE any computation
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

TRACES_PATH = os.path.join(REPO_ROOT, "data", "fr11_zenil_distill_v2.jsonl")
RESULT_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "experiment_3452_fr11_grounding_collapse_stress_test_v1.json",
)

N_ITERATIONS = 30
RANDOM_SEED = 42


def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3452,
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


# --- Precondition a: cached traces present ---
if not os.path.exists(TRACES_PATH):
    _write_blocked(
        f"Cached traces not found at {TRACES_PATH}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )

# --- Precondition b: FR-11 grounding-collapse module importable ---
try:
    from carnot.fr11.grounding_collapse_stress_test import run_stress_test
except ImportError as e:
    _write_blocked(
        f"Cannot import grounding_collapse_stress_test module: {e}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )

# ---------------------------------------------------------------------------
# Main experiment execution
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Experiment 3452 — FR-11 Grounding Collapse Stress Test v1")
    print(f"  Traces: {TRACES_PATH}")
    print(f"  N iterations per arm: {N_ITERATIONS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    # Run the stress test
    result = run_stress_test(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
    )

    # Annotate with experiment metadata
    result["experiment_id"] = 3452
    result["experiment_title"] = "FR-11 Grounding Collapse Stress Test v1"
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {"resource": "grounding_collapse_module", "available": True},
    ]

    # Print summary
    print("=== RESULTS ===")
    print(f"  ARM A final entropy:   {result['arm_a_final_entropy']:.4f}")
    print(f"  ARM B final entropy:   {result['arm_b_final_entropy']:.4f}")
    print(f"  ARM A mode-collapse:   {result['arm_a_mode_collapse_detected']}")
    print(f"  ARM B mode-collapse:   {result['arm_b_mode_collapse_detected']}")
    print(f"  ARM A pass-rate:       {result['arm_a_final_pass_rate']:.3f}")
    print(f"  ARM B pass-rate:       {result['arm_b_final_pass_rate']:.3f}")
    print(f"  ARM A entropy drop:    {result['arm_a_entropy_drop_ratio']:.3f}")
    print(f"  ARM B entropy drop:    {result['arm_b_entropy_drop_ratio']:.3f}")
    print()
    print(f"Honest verdict: {result['honest_verdict']}")
    print()
    print(f"Consequence: {result['grounding_collapse_consequence']}")
    print()
    print(f"Duration: {result['duration_s']:.3f}s")

    # Write artifact
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nArtifact written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
