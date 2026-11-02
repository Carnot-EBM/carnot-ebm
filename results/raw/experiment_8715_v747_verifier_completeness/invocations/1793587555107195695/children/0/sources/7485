#!/usr/bin/env python3
"""Experiment 3498 — FR-11 Beta-Min / Lambda-Min Predictive Law v1.

At >=3 grounding configurations (different ACTIVE_WEIGHTs yielding different
lambda_min), this experiment:
  (a) Measures lambda_min(Sigma) + participation-ratio effective-k from the
      k×k decision covariance on the cached corpus.
  (b) Finds the minimal-sufficient entropy beta that prevents depth-N>=200 collapse.
Then fits and held-out-tests a predictive law beta_min ~= f(lambda_min).

Context:
    exp3486 (.321): minimal beta found (0.10 for at-risk, 0.0 for healthy).
    exp3439 (.309): lambda_min=0 at ACTIVE_WEIGHT=0.146 (P0.2 at-risk finding).
    Open question: is the required beta PREDICTABLE from the measured lambda_min?
    If so, Phase-5 deployment has a formula: measure lambda_min, set beta accordingly.

    ER-PRM (arXiv:2412.11006): entropy regularization stabilizes PRM training.
    Cached traces + verifier scoring → no live model → no timeout.

Spec:
    REQ-FR11-BML-001, REQ-FR11-BML-002, REQ-FR11-BML-003
    SCENARIO-FR11-BML-001, SCENARIO-FR11-BML-002, SCENARIO-FR11-BML-003

Run command:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3498_fr11_beta_min_lambda_min_predictive_law_v1.py
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
    "experiment_3498_fr11_beta_min_lambda_min_predictive_law_v1.json",
)

N_ITERATIONS = 200
RANDOM_SEED = 42


def _write_blocked(reason: str, honest_verdict: str) -> None:
    artifact = {
        "experiment_id": 3498,
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "error": reason,
        "preconditions_checked": [
            {"resource": "fr11_cached_traces_or_module", "available": False, "detail": reason},
        ],
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"BLOCKED: {reason}")
    print(f"Artifact written to {RESULT_PATH}")
    sys.exit(0)


# PRECONDITIONS (step 0) — checked BEFORE any computation.
# a. FR-11 self-learning module + cached traces present.
if not os.path.exists(TRACES_PATH):
    _write_blocked(
        f"Cached traces not found at {TRACES_PATH}",
        "complete: blocked_fr11_module_or_traces_unavailable",
    )

# b. The verifier ensemble module is importable.
try:
    from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import (
        BETA_GRID,
        GROUNDING_CONFIGS,
        N_CHANNELS,
        run_beta_min_lambda_min_sweep,
    )
except ImportError as e:
    _write_blocked(
        f"Cannot import beta_min_lambda_min_predictive_law_v1 module: {e}",
        "complete: blocked_verifier_suite_uncallable",
    )


def main() -> None:
    print("Experiment 3498 — FR-11 Beta-Min / Lambda-Min Predictive Law v1")
    print(f"  Traces: {TRACES_PATH}")
    print(f"  N iterations per arm: {N_ITERATIONS}")
    print(f"  Beta grid: {BETA_GRID}")
    print(f"  Grounding configs: {[c['name'] for c in GROUNDING_CONFIGS]}")
    print(f"  N channels (Sigma): {N_CHANNELS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print()

    result = run_beta_min_lambda_min_sweep(
        traces_path=TRACES_PATH,
        n_iterations=N_ITERATIONS,
        seed=RANDOM_SEED,
        beta_grid=BETA_GRID,
        grounding_configs=GROUNDING_CONFIGS,
    )

    result["experiment_id"] = 3498
    result["experiment_title"] = "FR-11 Beta-Min / Lambda-Min Predictive Law v1"
    result["preconditions_checked"] = [
        {"resource": "fr11_cached_traces", "available": True, "path": TRACES_PATH},
        {"resource": "beta_min_lambda_min_predictive_law_v1_module", "available": True},
    ]

    print()
    print("=== RESULTS ===")
    print(f"  N grounding configs: {result['n_grounding_configs']}")
    print(f"  lambda_min by config:  {result['lambda_min_by_config']}")
    print(f"  effective_k by config: {result['effective_k_by_config']}")
    print(f"  minimal_beta by config:{result['minimal_beta_by_config']}")
    law = result.get("beta_min_lambda_min_fit", {})
    print(f"  Law fit: slope={law.get('slope'):.4f}, intercept={law.get('intercept'):.4f}, R²={law.get('r_squared'):.4f}")
    loo = result.get("leave_one_out_validation", {})
    print(f"  Hold-out: predicted={loo.get('predicted_beta_min')}, actual={loo.get('actual_beta_min')}, error={loo.get('prediction_error'):.4f}")
    print(f"  law_holds_out_of_sample: {result['law_holds_out_of_sample']}")
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
