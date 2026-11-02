#!/usr/bin/env python3
"""Run Capstone v315 — first Depth-Over-Breadth milestone.

Aggregates .315 upstream artifacts (exp3416, exp3312, exp3313, exp3417,
exp3419-3423), synthesizes G1-G4 gate status, reports the P0.1 verdict,
and emits paper_v6_safe_claims / paper_v6_forbidden_claims honoring the
Paper-v6 Narrowing Discipline.

Skips any artifact carrying flagged_adversarial=true (exp3397, exp3405).
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v315_3424 import run_capstone


def main() -> None:
    start = time.perf_counter()
    result = run_capstone()
    result["duration_s"] = round(time.perf_counter() - start, 6)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_3424_capstone_v315.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"Written: {out_path}")
    print(f"honest_verdict: {result['honest_verdict']}")
    print(f"G1={result['g1']} G2={result['g2']} G3={result['g3']} G4={result['g4']}")
    print(f"unmet_gates: {result['unmet_gates']}")
    print(f"paper_ready: {result['paper_ready']}")
    print(f"P0.1: {result['p0_1_verdict']}")
    print(f"depth_can_relax: {result['depth_forcing_function_can_relax']}")


if __name__ == "__main__":
    main()
