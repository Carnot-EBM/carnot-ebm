"""
Experiment 929 — Milestone 2026.04.72 Pre-flight v21

PURPOSE: Document the .71 gate-check discipline failure lesson, close
RETRO-LAGRANGE-ENTROPY-DEGENERATE, triage the six open RETROs entering .72,
record .72 gates, and emit a stable preflight artifact that the conductor
can use as the starting checkpoint for the .72 cycle.

WHY THIS MATTERS: Milestone .71 met only 2/12 criteria because the planner
that generated research-roadmap-v71.yaml omitted prior_failures fields on 7
of 12 tasks.  The conductor's rerun-discipline gate did exactly what it should
— it blocked those tasks immediately.  The lesson is planner-layer, not
code-layer: consult research-complete.yaml before writing any task YAML.

This script codifies that lesson in a machine-readable artifact so the .72
planner can read it as a required-reading checkpoint.
"""

import json
import sys
from datetime import datetime, timezone, UTC


def build_preflight_artifact() -> dict:
    """
    Assemble the Exp 929 preflight artifact.

    All values are derived from Exp 928 (milestone retro) and Exp 917 (preflight
    v20 gate-check analysis).  No external dependencies — runs in < 1 s on any
    machine with Python 3.11+.
    """
    return {
        "experiment": 929,
        "title": "Milestone 2026.04.72 Pre-flight v21",
        "milestone": "2026.04.72",
        "preflight_version": 21,
        "run_date": "20260426",
        "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # --- .71 summary ---
        "predecessor_milestone": "2026.04.71",
        "predecessor_criteria_met": 2,
        "predecessor_criteria_total": 12,
        "predecessor_root_cause": "gate_check_discipline_failure",
        "predecessor_root_cause_detail": (
            "7 of 12 YAML tasks lacked prior_failures fields. "
            "Conductor gate correctly blocked all 7 (Exps 917, 919, 920, 921, 922, 925, 926, 927). "
            "Planner must consult research-complete.yaml before writing any task YAML."
        ),
        # --- lessons documented ---
        "gate_check_lesson_documented": True,
        "gate_check_lesson_location": "ops/known-issues.md § GATE-CHECK DISCIPLINE",
        # --- RETRO triage ---
        "retro_lagrange_entropy_closed": True,
        "open_retros": [
            "RETRO-MANIFEST-FULL-SCOPE",
            "RETRO-XILINX-TOOLS-UNAVAILABLE",
            "RETRO-RERUN-DISCIPLINE-GATE-CASCADE",
            "RETRO-HEURISTIC-RPRM-FLAT-SIGNAL",
            "RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS",
            "RETRO-HF-SOPS-CREDENTIAL-INJECTION",
        ],
        "retro_statuses": {
            "RETRO-MANIFEST-FULL-SCOPE": "HUMAN_REQUIRED",
            "RETRO-XILINX-TOOLS-UNAVAILABLE": "HUMAN_REQUIRED",
            "RETRO-RERUN-DISCIPLINE-GATE-CASCADE": "HUMAN_REQUIRED",
            "RETRO-HEURISTIC-RPRM-FLAT-SIGNAL": "TARGETED",
            "RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS": "TARGETED",
            "RETRO-HF-SOPS-CREDENTIAL-INJECTION": "HUMAN_REQUIRED",
            "RETRO-LAGRANGE-ENTROPY-DEGENERATE": "CLOSED_BY_EXP918",
        },
        # --- .72 gates ---
        "milestone_72_gates": {
            "exp_931_combined_pipeline": "GATED on Exp 930 signed_improvement > 0",
            "exp_934_ipfs_mirror": "runs after Exp 933 regardless of HF publish verdict",
        },
        # --- deliverable ---
        "honest_verdict": "preflight_complete",
        "status": "success",
    }


def main() -> int:
    artifact = build_preflight_artifact()
    artifact["finished_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path = "results/experiment_929_preflight_v21.json"
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")

    print(f"Wrote {output_path}")
    print(
        f"  predecessor_criteria_met: {artifact['predecessor_criteria_met']}/{artifact['predecessor_criteria_total']}"
    )
    print(f"  root_cause: {artifact['predecessor_root_cause']}")
    print(f"  retro_lagrange_entropy_closed: {artifact['retro_lagrange_entropy_closed']}")
    print(f"  open_retros: {len(artifact['open_retros'])}")
    print(f"  honest_verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
