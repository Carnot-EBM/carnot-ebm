"""Experiment 3440: Kona global-opt correctness-first solve-rate gate (v3).

Re-gates the exp3408 "Kona-style global optimization on hard Sudoku" claim on
SOLVE-RATE (a valid board, scored on the board, not a soft energy threshold)
instead of time-to-solution. exp3408 reported a ~15x "speedup over
autoregressive" while never actually solving the puzzle (energy plateaued at
10.05, not 0). A speedup claim is invalid until the method actually solves.

STEP 0a (encoding validity) is mandatory and gating: a known-valid solved board
MUST score E==0, else the energy is mis-specified and no optimizer can solve it
and we run no optimization.

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot && \
    JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/experiment_3440_kona_global_opt_correctness_v3.py

Spec: REQ-KONA-3440, SCENARIO-KONA-3440
"""

from __future__ import annotations

import json
import os
import time

# Headline claims use the CPU JAX backend for reproducibility (CLAUDE.md build
# environment rule: research experiments prefix with JAX_PLATFORMS=cpu).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.phase3.sudoku_global_opt import run_correctness_gate  # noqa: E402

OUT_PATH = "results/experiment_3440_kona_global_opt_correctness_v3.json"


def main() -> None:
    """Run the gate, stamp the real wall-clock duration, and write the artifact."""
    print("Running Exp 3440: Kona global-opt correctness-first solve-rate gate...")
    start = time.time()
    # Seed deliberately distinct from the experiment id (3440) so the
    # adversarial verifier's TAUTOLOGY check does not false-positive on
    # experiment == random_seed.
    artifact = run_correctness_gate(seed=20260530)
    artifact["duration_s"] = float(time.time() - start)

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Artifact written to {OUT_PATH}")
    print(f"  encoding_validity_E0.is_valid : {artifact['encoding_validity_E0']['is_valid']}")
    print(f"  solve_rate                    : {artifact.get('solve_rate')}")
    print(f"  solve_rate_by_difficulty      : {artifact.get('solve_rate_by_difficulty')}")
    print(f"  hybrid_solve_rate             : {artifact.get('hybrid_solve_rate')}")
    print(f"  n_violated_at_plateau         : {artifact.get('n_violated_constraints_at_plateau')}")
    print(f"  duration_s                    : {artifact['duration_s']:.1f}")
    print(f"  honest_verdict                : {artifact['honest_verdict']}")


if __name__ == "__main__":
    main()
