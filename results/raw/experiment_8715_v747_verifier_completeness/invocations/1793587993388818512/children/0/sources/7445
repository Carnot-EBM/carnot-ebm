#!/usr/bin/env python3
"""Run Capstone v318 — Depth-Over-Breadth IV milestone.

Aggregates .318 upstream artifacts (exp3447-exp3456), synthesizes G1-G4 gate
status, reports the P0.1 v4 verdict (flagged_adversarial TAUTOLOGY — no clean
verdict), and emits paper_v6_safe_claims / paper_v6_forbidden_claims honoring
the Paper-v6 Narrowing Discipline.

Key .318 findings:
  - P0.1 v4 (exp3449): FLAGGED adversarial (TAUTOLOGY).  Substrate bug, not
    a result.  exp3450 (clean) explains it: energy_correctness_auroc=0.5160 —
    the IsingVerifier does not discriminate correct from incorrect answers.
  - G2: CI/Docker mechanism shipped (exp3451); external run still pending.
  - FR-11: grounding-collapse directional finding (exp3452, flagged; advisory).
  - Depth-Forcing-Function remains active.

Skips any artifact carrying flagged_adversarial=True per the fabrication gate
(CLAUDE.md Adversarial Artifact Verification Discipline).
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v318_3457 import run_capstone


def main() -> None:
    start = time.perf_counter()
    result = run_capstone()
    result["duration_s"] = round(time.perf_counter() - start, 6)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_3457_capstone_v318.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"Written: {out_path}")
    print(f"honest_verdict: {result['honest_verdict']}")
    print(f"G1={result['g1']} G2={result['g2']} G3={result['g3']} G4={result['g4']}")
    print(f"unmet_gates: {result['unmet_gates']}")
    print(f"paper_ready: {result['paper_ready']}")
    print(f"P0.1 v4 clean: {result['p0_1_v4_is_clean']}")
    print(f"energy_correctness_auroc: {result['energy_correctness_auroc']}")
    print(f"g2_ci_status: {result['g2_ci_status']}")
    print(f"depth_can_relax: {result['depth_forcing_function_can_relax']}")
    print(f"capstone_v318_ready: {result['capstone_v318_ready']}")


if __name__ == "__main__":
    main()
