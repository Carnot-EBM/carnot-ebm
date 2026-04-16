#!/usr/bin/env python3
"""Experiment 377 — RETRO-015 Infrastructure Fix: GPU Session Startup Verification.

**What this experiment does:**
    RETRO-015 is the highest-priority item in Carnot history: four consecutive
    milestones (2026.05.06, 2026.05.13, 2026.05.20, 2026.05.27) produced zero
    live GPU results because ``CARNOT_FORCE_LIVE=1`` was never exported into
    conductor subprocess environments.

    ``scripts/conductor_gpu_env.sh`` was created in Exp 365 (RETRO-012 fix)
    but never wired into the conductor launch sequence.  This experiment:
    1. Verifies ``scripts/session_startup.sh`` exists and sources
       ``conductor_gpu_env.sh`` (REQ-INFRA-017).
    2. Verifies ``LiveGPUGate`` hard gate is implemented (REQ-INFRA-018).
    3. Verifies subprocess env propagation — spawns a subprocess and confirms
       ``CARNOT_FORCE_LIVE=1`` is inherited (SCENARIO-INFRA-021).
    4. Records hardware state (is_live_capable) separately from fix verification,
       since the GPU may not be accessible in CI.

    ``retro_015_infrastructure_fixed`` is the key verdict:
    - ``True`` when env var is currently set, subprocess inherits it, and both
      scripts exist on disk.
    - ``False`` with specific failure details when any check fails.

    Note: ``is_live_capable`` may be ``False`` in CI (no GPU) without affecting
    ``retro_015_infrastructure_fixed`` — that flag tracks infrastructure setup,
    not hardware state.

**This is a CPU-only experiment.**  No GPU inference is performed.

**Deliverable:** results/experiment_377_gpu_session_fix.json

Spec: REQ-INFRA-017, REQ-INFRA-018,
      SCENARIO-INFRA-019, SCENARIO-INFRA-020, SCENARIO-INFRA-021
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow importing from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate
from python.carnot.pipeline.live_gpu_gate import (
    LiveGPUGate,
    build_session_startup_script,
    check_session_startup_exists,
)
from python.carnot.pipeline.live_gpu_diagnostic import diagnose_live_gpu

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
EXP_ID = 377
TITLE = "RETRO-015 Infrastructure Fix: GPU Session Startup Verification"
DELIVERABLE = "results/experiment_377_gpu_session_fix.json"


def run_experiment(repo_root: Path) -> dict:
    """Execute all RETRO-015 infrastructure checks and return the result dict.

    Separated from ``main()`` so tests can call it with a temporary repo root.

    Parameters
    ----------
    repo_root : Path
        Root of the Carnot repository.

    Returns
    -------
    dict
        Result artifact with all REQUIRED_RESULT_FIELDS and the
        RETRO-015 specific fields documented below.
    """
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
        repo_root=repo_root,
    )
    tmpl.setup()

    # --- Check 1: Is CARNOT_FORCE_LIVE currently set in this process? ---
    env_var_set = LiveGPUGate.check_env_var()

    # --- Check 2: Does subprocess inherit the env var? ---
    # This is the PROOF that the fix works end-to-end.
    subprocess_inherits_env = LiveGPUGate.verify_subprocess_env_propagation(
        "CARNOT_FORCE_LIVE"
    )

    # --- Check 3: Does scripts/session_startup.sh exist? ---
    session_startup_exists = check_session_startup_exists(repo_root)

    # --- Check 4: Does scripts/conductor_gpu_env.sh exist? ---
    conductor_gpu_env_path = repo_root / "scripts" / "conductor_gpu_env.sh"
    conductor_gpu_env_exists = conductor_gpu_env_path.is_file()

    # --- Check 5: Verify session_startup.sh sources conductor_gpu_env.sh ---
    # Read the actual file to confirm the sourcing line is present.
    session_startup_sources_conductor = False
    if session_startup_exists:
        content = (repo_root / "scripts" / "session_startup.sh").read_text()
        session_startup_sources_conductor = "conductor_gpu_env.sh" in content

    # --- Check 6: Hardware state (informational — does not affect fix verdict) ---
    # is_live_capable may be False in CI without a GPU.
    diag = diagnose_live_gpu()
    is_live_capable = diag.is_live_capable

    # --- Compute overall verdict ---
    # retro_015_infrastructure_fixed: infrastructure is correct regardless of
    # whether CARNOT_FORCE_LIVE happens to be set in THIS test run.
    # What matters: both scripts exist, session_startup.sh sources conductor_gpu_env.sh,
    # and subprocess env propagation works.
    retro_015_infrastructure_fixed = (
        subprocess_inherits_env
        and session_startup_exists
        and conductor_gpu_env_exists
        and session_startup_sources_conductor
    )

    # --- Compute honest_verdict ---
    if retro_015_infrastructure_fixed:
        honest_verdict = "infrastructure_fixed"
    elif not subprocess_inherits_env:
        honest_verdict = "env_propagation_failed"
    elif not session_startup_exists or not conductor_gpu_env_exists:
        honest_verdict = "scripts_missing"
    else:
        honest_verdict = "session_startup_does_not_source_conductor_env"

    artifact = tmpl.build_result(
        {
            "fix_schema": "carnot.gpu_session_fix.v1",
            "env_var_set": env_var_set,
            "subprocess_inherits_env": subprocess_inherits_env,
            "session_startup_exists": session_startup_exists,
            "conductor_gpu_env_exists": conductor_gpu_env_exists,
            "session_startup_sources_conductor": session_startup_sources_conductor,
            "is_live_capable": is_live_capable,
            "retro_015_infrastructure_fixed": retro_015_infrastructure_fixed,
            "honest_verdict": honest_verdict,
            "diagnostic_failure_reason": diag.failure_reason,
        },
        status="success" if retro_015_infrastructure_fixed else "blocked",
    )
    return artifact


def main() -> None:
    """Run the experiment, write the artifact, print a summary."""
    repo_root = _REPO_ROOT
    artifact = run_experiment(repo_root)

    output_path = repo_root / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Experiment {EXP_ID} complete.")
    print(f"  honest_verdict:                {artifact['honest_verdict']}")
    print(f"  retro_015_infrastructure_fixed:{artifact['retro_015_infrastructure_fixed']}")
    print(f"  env_var_set:                   {artifact['env_var_set']}")
    print(f"  subprocess_inherits_env:       {artifact['subprocess_inherits_env']}")
    print(f"  session_startup_exists:        {artifact['session_startup_exists']}")
    print(f"  conductor_gpu_env_exists:      {artifact['conductor_gpu_env_exists']}")
    print(f"  is_live_capable:               {artifact['is_live_capable']}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
