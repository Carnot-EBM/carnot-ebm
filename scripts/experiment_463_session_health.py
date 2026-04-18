#!/usr/bin/env python3
"""Experiment 463: Conductor Session Health Check — infrastructure validation.

**What this experiment validates:**
    ConductorSessionHealthCheck inspects GPU health, zombie processes, thermal
    state, and environment propagation at conductor session startup — the
    RETRO-034 root cause (three zombie processes held 23,795 MB on GPU 0 for
    11.5 hours during milestone .34, causing 97% VRAM saturation at 0% util).

    This experiment runs in non-destructive mode (auto_remediate=False) so it
    is safe in CI, does not kill any processes, and can be reproduced at any time.

**Root causes addressed:**
    - RETRO-034 (milestone .34): zombie processes blocked GPU 0 entire milestone.
    - RETRO-022: CARNOT_FORCE_LIVE not propagating into conductor subprocesses.
    - Thermal event: GPU at 82°C during runaway experiments.

Spec: REQ-INFRA-036, REQ-INFRA-037, REQ-INFRA-038,
      SCENARIO-INFRA-044, SCENARIO-INFRA-045, SCENARIO-INFRA-046
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix FIRST (belt-and-suspenders, RETRO-022)
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.session_health_check import (
    ConductorSessionHealthCheck,
    GPUHealth,
)
from scripts.experiment_template import ExperimentTemplate

_DELIVERABLE = "results/experiment_463_session_health.json"


def main() -> None:
    """Run Exp 463: session health check validation and write deliverable JSON."""

    with ExperimentTimeoutWatchdog(463, timeout_minutes=20, result_path=_DELIVERABLE):
        tmpl = ExperimentTemplate(
            463,
            "Conductor Session Health Check",
            _DELIVERABLE,
        )
        tmpl.setup()

        # Run non-destructive health check (auto_remediate=False for CI safety)
        chk = ConductorSessionHealthCheck(auto_remediate=False)
        result = chk.run()

        # Gather per-GPU health for the artifact
        gpu_healths = chk._check_gpu_health()

        gpu0_health = gpu_healths[0].to_dict() if len(gpu_healths) > 0 else None
        gpu1_health = gpu_healths[1].to_dict() if len(gpu_healths) > 1 else None

        # Count zombies found (but not killed — auto_remediate=False)
        zombie_gpu_indices = [
            g.gpu_index for g in gpu_healths if g.is_zombie_saturated
        ]
        zombies_found_list = chk._find_zombie_processes(zombie_gpu_indices)
        zombies_found = len(zombies_found_list)

        artifact = tmpl.build_result(
            {
                "schema": "carnot.session_health.v1",
                "env_ok": result.env_ok,
                "gpu0_health": gpu0_health,
                "gpu1_health": gpu1_health,
                "zombies_found": zombies_found,
                "zombies_killed": 0,  # always 0: auto_remediate=False in CI
                "thermal_ok": result.thermal_ok,
                "honest_verdict": result.honest_verdict,
                "env_autofix_gpu_detected": _env_result.gpu_detected,
                "env_autofix_auto_fix_applied": _env_result.auto_fix_applied,
            },
            status="success",
        )

        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))

    # FINAL line: assert deliverable was written (REQ-INFRA-033, RETRO-032/033/036)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
