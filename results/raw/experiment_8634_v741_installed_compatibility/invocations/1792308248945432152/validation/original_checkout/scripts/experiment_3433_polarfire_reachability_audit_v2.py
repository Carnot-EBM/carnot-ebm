#!/usr/bin/env python3
"""Exp 3433 PolarFire reachability audit v2 — experiment runner.

Spec refs: REQ-HW-070, SCENARIO-HW-070.

Why this script exists:
    Hardware-Task Continuity Discipline requires at least one task per
    attached board per milestone. PolarFire is opportunistic-only (north-star
    §3), so this is the minimal light audit: SSH reachability check + uptime
    capture, no new workload. Emits a structured JSON artifact that the
    adversarial verifier can validate (inference_substrate=hardware_smoke,
    plausible duration_s for an SSH ping).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "results" / "experiment_3433_polarfire_reachability_audit_v2.json"

sys.path.insert(0, str(REPO_ROOT))

from carnot.hardware.polarfire_reachability_audit_3433 import run_audit


def main() -> None:
    artifact = run_audit()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"Artifact written to {OUTPUT_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"duration_s: {artifact['duration_s']:.3f}")


if __name__ == "__main__":
    main()
