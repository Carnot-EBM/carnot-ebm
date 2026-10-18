#!/usr/bin/env python3
"""Run Capstone v322 — Depth-Over-Breadth VIII milestone.

Aggregates .322 upstream artifacts (exp3494-exp3501, plus exp3502 which is
flagged adversarial and handled separately), synthesizes G1-G4 gate status
from unflagged primary experiments, reports both P0.1 routes (both BLOCKED),
MATH-aware calibration recovery (exp3497 CLEAN), FR-11 beta_min=f(lambda_min)
law (exp3498 CLEAN), and G2 package regression status (exp3499 CLEAN).

Key .322 findings:
  - P0.1 Route 1 (Sudoku, exp3494): BLOCKED — encoding valid (E=0 for correct
    board), but gradient optimizer cannot escape local minima; representational,
    not substrate failure.
  - P0.1 Route 2 (in-band, exp3495): BLOCKED — contested subset n=21 < 40.
  - Calibration v5 (exp3497): CLEAN — MATH-aware recalibration recovers
    correctness signal (0.601→0.625 AUROC); domain-shift confound identified.
  - FR-11 beta_min law (exp3498): CLEAN — Phase-5 deployment rule: beta_min
    = -0.3001 + 1.8461 * lambda_min (R²=0.989), out-of-sample validated.
  - G2 regression (exp3499): CLEAN — package regression clean; external run
    pending (G2 operator-gated per Operator-Only External Publication rule).
  - KV260 (exp3500): SSH unreachable.
  - PolarFire (exp3501): CLEAN — SSH reachable, continuity confirmed.
  - Gate synthesis (exp3502): FLAGGED adversarial (TAUTOLOGY false-positive:
    experiment==random_seed==3502 by construction; not a measurement issue).
    Gate status derived from primary unflagged experiments in this capstone.
  - Depth-Forcing-Function: REMAINS ACTIVE — no clean P0.1 verdict on either
    route; G2 external run still pending.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v322_3503 import run_capstone


def main() -> None:
    start = time.perf_counter()
    result = run_capstone()
    result["duration_s"] = round(time.perf_counter() - start, 6)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_3503_capstone_v322.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"Written: {out_path}")
    print(f"honest_verdict: {result['honest_verdict']}")
    print(f"G1={result['g1']} G2={result['g2']} G3={result['g3']} G4={result['g4']}")
    print(f"unmet_gates: {result['unmet_gates']}")
    print(f"paper_ready: {result['paper_ready']}")
    print(f"P0.1 has_clean_verdict: {result['p0_1_has_clean_verdict']}")
    print(f"P0.1 Route 1: {result['p0_1_route1_verdict']}")
    print(f"P0.1 Route 2: {result['p0_1_route2_verdict']}")
    print(f"calibration_diagnosis: {result['calibration_diagnosis'][:80]}...")
    print(f"FR-11 law R²: {result['fr11_r2']}")
    print(f"G2 package status: {result['g2_package_status']}")
    print(f"depth_can_relax: {result['depth_forcing_function_can_relax']}")
    print(f"capstone_v322_ready: {result['capstone_v322_ready']}")


if __name__ == "__main__":
    main()
