"""Experiment 3494: P0.1 Sudoku correctness-first solve-rate gate (v1).

Validates the Sudoku-Ising energy encoding against the published QUBO
(arXiv:2403.04816), confirms easy-tier solvability, then runs a full optimizer
ladder on >=20 puzzles across difficulty tiers, reporting SOLVE-RATE (final
board valid / E==0 verified ON THE BOARD) by difficulty and optimizer variant.

QUBO cross-validation: Carnot's energy implements the same four constraint
families as arXiv:2403.04816 (row/col/box uniqueness + cell one-hot pinning).
A valid completed board is the unique global minimum (E=0), asserted in Step 0a
before any optimization runs.

CPU-only (no GGUF, no CUDA, no live LLM generation). Immune to the
thinking-400 / tokenizer / CUDA failures that blocked exp3408-3475.

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot && \\
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3494_p01_sudoku_correctness_first_solve_rate_gate_v1.py

Spec: REQ-KONA-3494, SCENARIO-KONA-3494
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.phase3.sudoku_p01_gate import run_p01_gate  # noqa: E402

OUT_PATH = "results/experiment_3494_p01_sudoku_correctness_first_solve_rate_gate_v1.json"

SEED = 20260531


def main() -> None:
    """Run the gate, stamp the real wall-clock duration, and write the artifact."""
    print("Running Exp 3494: P0.1 Sudoku correctness-first solve-rate gate...")
    start = time.time()
    artifact = run_p01_gate(seed=SEED)
    artifact["duration_s"] = float(time.time() - start)

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written to {OUT_PATH}")
    print(f"  encoding_validity_E0.is_valid : {artifact['encoding_validity_E0']['is_valid']}")
    print(f"  easy_tier_solve_rate          : {artifact.get('easy_tier_solve_rate')}")
    print(f"  solve_rate                    : {artifact.get('solve_rate')}")
    print(f"  solve_rate_by_difficulty      : {artifact.get('solve_rate_by_difficulty')}")
    print(f"  hybrid_solve_rate             : {artifact.get('hybrid_solve_rate')}")
    print(f"  ar_baseline_solve_rate        : {artifact.get('ar_baseline_solve_rate')}")
    print(f"  n_violated_at_plateau_mean    : {artifact.get('n_violated_constraints_at_plateau')}")
    print(f"  duration_s                    : {artifact['duration_s']:.1f}")
    print(f"  honest_verdict                : {artifact['honest_verdict']}")


if __name__ == "__main__":
    main()
