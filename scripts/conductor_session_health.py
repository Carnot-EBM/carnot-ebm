#!/usr/bin/env python3
"""Standalone conductor session health check — run BEFORE the first experiment.

This script is meant to be called by the conductor at session startup, before
any experiment subprocess is spawned.  It does NOT use ExperimentTemplate
because ExperimentTemplate assumes an experiment is already running; this check
runs in the gap before the first experiment.

Exit codes:
    0 — session is healthy (or was remediated successfully)
    1 — thermal gate triggered: conductor must pause until a human reviews GPU temps

Spec: REQ-INFRA-036, REQ-INFRA-037, REQ-INFRA-038
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.session_health_check import ConductorSessionHealthCheck


def main() -> None:
    # Step 1: apply env fix before anything else (belt-and-suspenders, RETRO-022)
    env_result = apply_env_autofix()

    # Step 2: run full health check
    result = ConductorSessionHealthCheck(auto_remediate=True).run()

    # Step 3: print summary
    print("=== Conductor Session Health Check ===")
    print(f"  env_ok:          {result.env_ok}")
    print(f"  gpu_ok:          {result.gpu_ok}")
    print(f"  zombies_killed:  {result.zombies_killed}")
    print(f"  thermal_ok:      {result.thermal_ok}")
    print(f"  honest_verdict:  {result.honest_verdict}")
    print(f"  env_autofix:     gpu_detected={env_result.gpu_detected} "
          f"auto_fix_applied={env_result.auto_fix_applied}")

    # Step 4: thermal gate (REQ-INFRA-038)
    if not result.thermal_ok:
        print()
        print("CONDUCTOR PAUSED — GPU temp >= 80C")
        print("At least one GPU is overheating.  The conductor will NOT start until")
        print("temperatures drop below 80°C.  Check `nvidia-smi` and wait for cooling.")
        print("This check closes RETRO-034 thermal gate (RTX 3090 at 82°C shortens life).")
        sys.exit(1)

    # Step 5: env warning (non-fatal)
    if not result.env_ok:
        print()
        print("WARNING — CARNOT_FORCE_LIVE not propagating")
        print("GPU was not detected OR the env fix did not take effect.")
        print("Remediation: export CARNOT_FORCE_LIVE=1 in the conductor's shell before")
        print("launching.  See RETRO-022 for root cause documentation.")


if __name__ == "__main__":
    main()
