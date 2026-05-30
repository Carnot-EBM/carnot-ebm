#!/usr/bin/env python3
"""Run Capstone v319 — Depth-Over-Breadth V milestone.

Aggregates .319 upstream artifacts (exp3458-exp3468), synthesizes G1-G4 gate
status, reports the P0.1 v5 verdict (trained energy reranker: MATCHED but did
NOT beat self-consistency; exp3460 flagged_adversarial TAUTOLOGY), reports the
calibration v2 result (trained_energy_correctness_auroc=0.629, CLEAN, > 0.55
threshold — key advance over .318), and emits paper_v6_safe_claims /
paper_v6_forbidden_claims honoring the Paper-v6 Narrowing Discipline.

Key .319 findings:
  - P0.1 v5 (exp3460): FLAGGED adversarial (TAUTOLOGY).  Real exact tie per
    methodology note, but adversarial_verify cannot distinguish from stub.
    Numbers excluded.  Directional: trained energy MATCHES but does NOT BEAT SC.
  - Calibration v2 (exp3461): CLEAN.  trained_energy_correctness_auroc=0.629 >
    0.55 threshold.  Training lifts energy from 0.516 (chance) to 0.629.
  - G2: CI dry-run green (exp3463, CLEAN).  Handoff package ready.  External
    run still pending.  G2 unmet.
  - FR-11 collapse (exp3462): FLAGGED.  Directional: no collapse at N=50.
  - Kona hybrid (exp3464): CLEAN.  delta=0.0 — trained energy no lift.
  - Depth-Forcing-Function: cannot relax (P0.1 not clean + G2 unmet).

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

from carnot.reporting.capstone_v319_3469 import run_capstone


def main() -> None:
    start = time.perf_counter()
    result = run_capstone()
    result["duration_s"] = round(time.perf_counter() - start, 6)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_3469_capstone_v319.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"Written: {out_path}")
    print(f"honest_verdict: {result['honest_verdict']}")
    print(f"G1={result['g1']} G2={result['g2']} G3={result['g3']} G4={result['g4']}")
    print(f"unmet_gates: {result['unmet_gates']}")
    print(f"paper_ready: {result['paper_ready']}")
    print(f"P0.1 v5 clean: {result['p0_1_v5_is_clean']}")
    print(f"trained_energy_correctness_auroc: {result['trained_energy_correctness_auroc']}")
    print(f"trained_energy_auroc_lift: {result['trained_energy_auroc_lift_over_untrained']}")
    print(f"g2_ci_status: {result['g2_ci_status']}")
    print(f"kona_trained_hybrid_delta: {result['kona_trained_hybrid_delta']}")
    print(f"depth_can_relax: {result['depth_forcing_function_can_relax']}")
    print(f"capstone_v319_ready: {result['capstone_v319_ready']}")


if __name__ == "__main__":
    main()
