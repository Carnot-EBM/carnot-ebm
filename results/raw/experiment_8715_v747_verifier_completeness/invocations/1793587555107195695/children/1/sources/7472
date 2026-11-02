#!/usr/bin/env python3
"""Run Capstone v320 — Depth-Over-Breadth VI milestone.

Aggregates .320 upstream artifacts (exp3471-exp3479), synthesizes G1-G4 gate
status, reports the P0.1 v6 verdict (BLOCKED — corpus outside headroom band;
no energy-vs-SC comparison was run), the FR-11 depth-collapse confirmation
(CLEAN, collapse at N=200, ARM B prevents), and the G2 self-contained package
status (exp3476, internally verified, external run pending).  Emits
paper_v6_safe_claims / paper_v6_forbidden_claims honoring the Paper-v6
Narrowing Discipline.

Key .320 findings:
  - P0.1 v6 (exp3472): BLOCKED — MATH Level 5 yields SC ~0.265, outside
    headroom band [0.40, 0.70].  No energy comparison was run.
  - Calibration v3 (exp3473): FLAGGED adversarial (TAUTOLOGY).  Advisory:
    process energy AUROC=0.441 (below chance) on MATH — domain specificity.
  - FR-11 depth collapse (exp3474): CLEAN key finding — collapse at N=200
    onset=138; ARM B (entropy_beta=0.50) prevents.  Entropy reg mandatory.
  - Kona harder instances (exp3475): BLOCKED — instances saturated.
  - G2 package (exp3476): CLEAN — self-contained tar.gz + SHA256 + IPFS CID.
    External run still pending.  G2 = false.
  - KV260 (exp3477): SSH unreachable.
  - GateMate (exp3478): toolchain incomplete.
  - PolarFire (exp3479): CLEAN — reachable.
  - Depth-Forcing-Function: CANNOT relax (P0.1 blocked + G2 unmet).

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

from carnot.reporting.capstone_v320_3481 import run_capstone


def main() -> None:
    start = time.perf_counter()
    result = run_capstone()
    result["duration_s"] = round(time.perf_counter() - start, 6)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_3481_capstone_v320.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"Written: {out_path}")
    print(f"honest_verdict: {result['honest_verdict']}")
    print(f"G1={result['g1']} G2={result['g2']} G3={result['g3']} G4={result['g4']}")
    print(f"unmet_gates: {result['unmet_gates']}")
    print(f"paper_ready: {result['paper_ready']}")
    print(f"P0.1 v6 blocked: {result['p0_1_v6_blocked']}")
    print(f"FR-11 collapse@N=200: {result['fr11_collapse_confirmed_at_n200']}")
    print(f"G2 package status: {result['g2_package_status']}")
    print(f"depth_can_relax: {result['depth_forcing_function_can_relax']}")
    print(f"capstone_v320_ready: {result['capstone_v320_ready']}")


if __name__ == "__main__":
    main()
