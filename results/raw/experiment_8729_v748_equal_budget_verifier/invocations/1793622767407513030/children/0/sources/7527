#!/usr/bin/env python3
"""Run Capstone v323 — Depth-Over-Breadth IX milestone.

Aggregates .323 upstream artifacts (exp3505-exp3513), synthesizes G1-G4 gate
status from unflagged primary experiments, reports P0.1 Route 1 POSITIVE
(Sudoku optimizer-ladder solve_rate=1.0 vs AR baseline 0.0), Route 2 FLAGGED
(exp3507 TAUTOLOGY), step-to-final gap FLAGGED (exp3508), FR-11 beta-law
deployment NOT validated (exp3509 CLEAN but deployed_law_prevents_collapse=False),
and G2 package regression clean (exp3510 CLEAN, external run pending).

Key .323 findings:
  - P0.1 Route 1 (Sudoku, exp3505): POSITIVE — real combinatorial optimizers
    (SA 20 restarts, parallel tempering, exact CP) achieve solve_rate=1.0
    across easy/medium/hard (21/21). AR greedy baseline=0.0. First clean
    positive P0.1 datapoint.
  - P0.1 Route 2 (in-band, exp3507): FLAGGED adversarial (TAUTOLOGY: all
    energy metrics collapsed to SC baseline 0.653061; flip_count=0).
  - Step-to-final gap (exp3508): FLAGGED adversarial.
  - FR-11 beta-law deployment (exp3509): CLEAN but NOT validated —
    deployed_law_prevents_collapse=False; use conservative default beta.
  - G2 regression (exp3510): CLEAN — package regression clean; external
    run pending (G2 operator-gated per Operator-Only External Publication).
  - Gate synthesis (exp3513): G1/G3/G4 met; G2 pending; P0.1 has clean
    verdict; depth_forcing_function_can_relax=True.
  - Depth-Forcing-Function: CAN RELAX — Route 1 positive, G2 in-motion.

random_seed is fixed at 20260531 (NOT the experiment number) — per the
exp3503 tautology fix (adversarial_verify flags random_seed == experiment_id).
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v323_3514 import run_capstone


def main() -> None:
    start = time.perf_counter()
    result = run_capstone()
    result["duration_s"] = round(time.perf_counter() - start, 6)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_3514_capstone_v323.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"Written: {out_path}")
    print(f"honest_verdict: {result['honest_verdict']}")
    print(f"G1={result['g1']} G2={result['g2']} G3={result['g3']} G4={result['g4']}")
    print(f"unmet_gates: {result['unmet_gates']}")
    print(f"paper_ready: {result['paper_ready']}")
    print(f"P0.1 has_clean_verdict: {result['p0_1_has_clean_verdict']}")
    print(f"P0.1 Route 1: {result['p0_1_route1_verdict'][:80]}...")
    print(f"P0.1 Route 1 solve_rate: {result['p0_1_route1_solve_rate']}")
    print(f"P0.1 Route 1 AR baseline: {result['p0_1_route1_ar_baseline_solve_rate']}")
    print(f"FR-11 beta-law deployment validated: {result['fr11_beta_law_deployment_validated']}")
    print(f"G2 package status: {result['g2_package_status'][:80]}...")
    print(f"depth_can_relax: {result['depth_forcing_function_can_relax']}")
    print(f"random_seed: {result['random_seed']}")
    print(f"capstone_v323_ready: {result['capstone_v323_ready']}")


if __name__ == "__main__":
    main()
