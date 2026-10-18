#!/usr/bin/env python3
"""Experiment 3486 — FR-11 Minimal Beta + Grounding-Dependence Sweep v4.

Sweeps entropy beta {0, 0.1, 0.25, 0.5} on the N>=200 FR-11 self-improvement
loop to find the MINIMAL beta that prevents mode-collapse, and varies the
grounding strength (ACTIVE_WEIGHT at {0.146 (at-risk from exp3439), 0.30
(healthier)}) to test whether the required beta / collapse onset depends on
grounding diversity.

Context:
    exp3474 (.320) showed ARM A (beta=0) COLLAPSES at N=200 (onset ≈138) and
    ARM B (beta=0.50) PREVENTS it, for ACTIVE_WEIGHT=0.146. This experiment
    narrows "0.50 works" to the minimal sufficient beta and characterizes whether
    grounding diversity changes the picture.

    Independent precedent: ER-PRM (arXiv:2412.11006) shows entropy regularization
    stabilizes PRM training (+2-3% MATH BoN). On cached traces (no live model,
    no timeout).

Spec:
    REQ-FR11-MB-001, REQ-FR11-MB-002, REQ-FR11-MB-003
    SCENARIO-FR11-MB-001, SCENARIO-FR11-MB-002, SCENARIO-FR11-MB-003

Run command:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3486_fr11_minimal_beta_grounding_dependence_v4.py
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
    "experiment_3486_fr11_minimal_beta_grounding_dependence_v4.json",
)

N_ITERATIONS = 200
RANDOM_SEED = 42


def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3486,
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
    from carnot.fr11.minimal_beta_grounding_dependence_v4 import (
        BETA_GRID,
        GROUNDING_STRENGTHS,
        run_minimal_beta_sweep,
    )
except ImportError as e:
    _write_blocked(
        f"Cannot import minimal_beta_grounding_dependence_v4 module: {e}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )


def main() -> None:
    print("Experiment 3486 — FR-11 Minimal Beta + Grounding-Dependence Sweep v4")
    print(f"  Traces: {TRACES_PATH}")
    print(f"  N iterations per arm: {N_ITERATIONS}")
    print(f"  Beta grid: {BETA_GRID}")
    print(f"  Grounding strengths: {GROUNDING_STRENGTHS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    result = run_minimal_beta_sweep(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
        beta_grid=BETA_GRID,
        grounding_strengths=GROUNDING_STRENGTHS,
    )

    result["experiment_id"] = 3486
    result["experiment_title"] = "FR-11 Minimal Beta + Grounding-Dependence Sweep v4"
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {"resource": "minimal_beta_grounding_dependence_v4_module", "available": True},
    ]

    print("=== RESULTS ===")
    print(f"  Minimal sufficient beta:       {result['minimal_sufficient_beta']}")
    print(f"  Grounding depends on beta:     {result['minimal_beta_depends_on_grounding']}")
    print(f"  Collapse onset by beta:        {result['collapse_onset_by_beta']}")
    print(f"  Per-grounding minimal betas:   {result['minimal_betas_per_grounding']}")
    print(f"  Acceptance gates:              {result['acceptance_gates']}")
    ets = result.get("entropy_trend_significance_beta0", {})
    print(f"  Entropy trend tau (beta=0):    {ets.get('tau', 'N/A'):.3f}")
    print(f"  Entropy trend p-val (beta=0):  {ets.get('p_value', 'N/A'):.4f}")
    print()
    print(f"Honest verdict: {result['honest_verdict']}")
    print()
    print(f"Recommended Phase-5 default:")
    print(f"  {result['recommended_phase5_default']}")
    print()
    print(f"Duration: {result['duration_s']:.3f}s")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nArtifact written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
