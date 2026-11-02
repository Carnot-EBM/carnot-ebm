#!/usr/bin/env python3
"""Experiment 390 — GPU Node Preflight Check (RETRO-019 first action).

**Purpose:**
    This is the FIRST experiment of the 2026.04.29 milestone.  Its only job is to
    confirm that the GPU node is physically powered on and reachable before any
    GPU-dependent experiments (394-400) are attempted.

    For FIVE consecutive milestones (2026.04.24 through 2026.04.28), the GPU node
    was offline during conductor sessions, producing zero live results.  This script
    makes the failure explicit and immediate: if the GPU is not confirmed live, it
    writes a blocked artifact and exits with code 1 so the conductor halts all
    GPU-dependent work.

**How it works:**
    Calls ``run_gpu_preflight()`` which checks six layers in order:
    1. CARNOT_FORCE_LIVE=1 in current process env
    2. CARNOT_FORCE_LIVE=1 inherited by subprocesses
    3. scripts/session_startup.sh exists
    4. scripts/conductor_gpu_env.sh exists
    5. diagnose_live_gpu() — nvidia-smi, torch.cuda, model tokenizers
    6. run_smoke_test() for each model — actual GPU inference

    If honest_verdict != "gpu_confirmed_live", prints an ACTION REQUIRED message
    with exact fix steps and exits with code 1.

**Action if blocked:**
    1. Physically power on the GPU node and confirm network connectivity.
    2. Verify ``nvidia-smi`` returns exit code 0.
    3. Source ``scripts/session_startup.sh`` in the conductor shell:
           source scripts/session_startup.sh
    4. Confirm ``echo $CARNOT_FORCE_LIVE`` prints ``1``.
    5. Re-run this script.

Spec: REQ-INFRA-017, REQ-INFRA-018,
      SCENARIO-INFRA-019, SCENARIO-INFRA-020, SCENARIO-INFRA-021
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring — allow running from repo root without pip install
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.gpu_preflight import build_preflight_artifact, run_gpu_preflight
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_PATH = "results/experiment_390_gpu_preflight.json"
_MODEL_IDS = ["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"]

_ACTION_REQUIRED_MESSAGES = {
    "scripts_missing": (
        "ACTION REQUIRED: scripts/session_startup.sh or scripts/conductor_gpu_env.sh is missing.\n"
        "  Fix:\n"
        "    1. Verify both files exist:\n"
        "         ls scripts/session_startup.sh scripts/conductor_gpu_env.sh\n"
        "    2. If missing, re-run Exp 377 to regenerate them:\n"
        "         python scripts/experiment_377_gpu_session_fix.py\n"
        "    3. Re-run this preflight: python scripts/experiment_390_gpu_preflight.py"
    ),
    "env_not_propagating": (
        "ACTION REQUIRED: CARNOT_FORCE_LIVE=1 is not being inherited by subprocesses.\n"
        "  Fix:\n"
        "    1. Source the startup script in your conductor shell:\n"
        "         source scripts/session_startup.sh\n"
        "    2. Confirm the variable is exported:\n"
        "         echo $CARNOT_FORCE_LIVE   # should print 1\n"
        "    3. Re-run this preflight: python scripts/experiment_390_gpu_preflight.py"
    ),
    "gpu_hardware_not_live": (
        "ACTION REQUIRED: GPU node is offline or not accessible.\n"
        "  This is RETRO-019 — the physical GPU node must be powered on BEFORE the\n"
        "  conductor session starts.\n"
        "  Fix:\n"
        "    1. Physically power on the GPU node and confirm network connectivity.\n"
        "    2. Verify GPU driver: nvidia-smi  (should list RTX 3090 x2)\n"
        "    3. Source session startup: source scripts/session_startup.sh\n"
        "    4. Confirm: echo $CARNOT_FORCE_LIVE  (should print 1)\n"
        "    5. Re-run this preflight: python scripts/experiment_390_gpu_preflight.py\n"
        "  DO NOT run GPU-dependent experiments (394-400) until this is resolved."
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run GPU preflight and return exit code (0 = confirmed live, 1 = blocked)."""
    tmpl = ExperimentTemplate(
        390,
        "GPU node preflight",
        _OUTPUT_PATH,
    )
    tmpl.setup()

    print("[Exp 390] Running GPU node preflight check (RETRO-019)...")
    result = run_gpu_preflight(_REPO_ROOT, model_ids=_MODEL_IDS)

    # Build the full experiment artifact using ExperimentTemplate for standard fields.
    preflight_data = build_preflight_artifact(result)
    status = "success" if result.retro_019_resolved else "blocked"
    artifact = tmpl.build_result(preflight_data, status=status)

    # Write artifact to disk.
    output_path = _REPO_ROOT / _OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[Exp 390] Artifact written to {_OUTPUT_PATH}")

    # Report outcome.
    if result.honest_verdict == "gpu_confirmed_live":
        print(
            "[Exp 390] GPU CONFIRMED LIVE — RETRO-019 RESOLVED.\n"
            f"  Models loadable: {result.model_ids_loadable}\n"
            "  Proceed with GPU-dependent experiments (394-400)."
        )
        return 0
    else:
        action_msg = _ACTION_REQUIRED_MESSAGES.get(
            result.honest_verdict,
            f"ACTION REQUIRED: honest_verdict={result.honest_verdict!r}. "
            "Check the artifact for details.",
        )
        print(f"\n[Exp 390] BLOCKED — honest_verdict={result.honest_verdict!r}")
        print(action_msg)
        print(
            "\n  Preflight details:\n"
            f"    env_var_set              = {result.env_var_set}\n"
            f"    subprocess_inherits_env  = {result.subprocess_inherits_env}\n"
            f"    session_startup_exists   = {result.session_startup_exists}\n"
            f"    conductor_gpu_env_exists = {result.conductor_gpu_env_exists}\n"
            f"    is_live_capable          = {result.is_live_capable}\n"
            f"    smoke_test_passed        = {result.smoke_test_passed}\n"
            f"    model_ids_loadable       = {result.model_ids_loadable}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
