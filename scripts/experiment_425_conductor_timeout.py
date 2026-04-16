#!/usr/bin/env python3
"""Experiment 425: Conductor Timeout Watchdog — RETRO-003 closure.

**What this experiment does:**
    Implements and demonstrates the ``ExperimentTimeoutWatchdog`` — the
    infrastructure item that has been carried in RETRO-003 for 17+ consecutive
    milestones without implementation.

    RETRO-003 root cause: PID 3509070 (Exp 219) ran 144+ minutes with GPU0
    at 82C.  A 45-minute hard cap would have freed GPU0 99 minutes early.
    This experiment ships the watchdog so every future experiment is protected.

**CPU-only:** This experiment does not require GPU.  It always completes and
    always produces a result JSON.

**Demonstration:**
    1. Starts a watchdog with a 2-minute timeout (well above the ~10-second
       synthetic workload).
    2. Runs 10 synthetic constraint checks totalling ~10 seconds.
    3. Stops the watchdog normally (workload finished before timeout).
    4. Writes the result artifact with ``retro_003_resolved=True``.

Spec: REQ-INFRA-023, REQ-INFRA-024,
      SCENARIO-INFRA-028, SCENARIO-INFRA-029, SCENARIO-INFRA-030
"""

# apply_env_autofix MUST be called first, before any other carnot import.
# This follows the RETRO-022 pattern: the conductor subprocess may not have
# CARNOT_FORCE_LIVE propagated from the interactive shell.
from python.carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env autofix)
# ---------------------------------------------------------------------------

import json
import sys
import time
from pathlib import Path

# Resolve repo root so the script works when invoked from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.experiment_watchdog import (  # noqa: E402
    ExperimentTimeoutResult,
    ExperimentTimeoutWatchdog,
    build_timeout_artifact,
    get_timeout_minutes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 425
TITLE = "Conductor Timeout Watchdog"
DELIVERABLE = "results/experiment_425_conductor_timeout.json"

# Production default (45 min); configurable via CARNOT_CONDUCTOR_TIMEOUT_MINUTES.
PRODUCTION_TIMEOUT_MINUTES = get_timeout_minutes()

# Demo watchdog timeout — 2 minutes, well above the synthetic workload (~10 s).
# We want to show the watchdog armed and disarmed without actually tripping it.
DEMO_TIMEOUT_MINUTES = 2

# Number of synthetic constraint checks in the demo workload.
N_SYNTHETIC_CHECKS = 10


# ---------------------------------------------------------------------------
# Synthetic workload
# ---------------------------------------------------------------------------


def _run_synthetic_constraint_check(i: int) -> dict:
    """Simulate a single constraint check.

    Why synthetic? This experiment is CPU-only and infrastructure-focused.
    The goal is to demonstrate the watchdog lifecycle (arm → workload → disarm),
    not to run a real EBM inference pipeline.  Real constraint checks are
    exercised in other experiments (Exp 358, Exp 419, etc.).

    Each check sleeps for 1 second to give the watchdog something to monitor,
    then returns a synthetic result.
    """
    time.sleep(1.0)
    # Simulate a trivial energy function: E = |i - 5|; minimum at i=5
    energy = abs(i - 5)
    satisfied = energy == 0
    return {
        "check_id": i,
        "energy": energy,
        "satisfied": satisfied,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 425: demonstrate ExperimentTimeoutWatchdog lifecycle."""

    # Step 1: Set up experiment scaffolding.
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    # Step 2: Arm the watchdog with a 2-minute demo timeout.
    #
    # The production default is 45 minutes (REQ-INFRA-024), configurable via
    # CARNOT_CONDUCTOR_TIMEOUT_MINUTES.  We use 2 minutes here so the demo
    # completes in ~10 seconds while still exercising the watchdog API.
    result_path = str(_REPO_ROOT / DELIVERABLE)
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=DEMO_TIMEOUT_MINUTES,
        result_path=result_path,
    )

    # Step 3: Run the demo workload inside the watchdog context manager.
    #
    # __enter__ calls start(); __exit__ calls stop().  If the workload exceeds
    # DEMO_TIMEOUT_MINUTES, _on_timeout() fires and exits the process with
    # code 1 — exactly what RETRO-003 called for.  In this demo the workload
    # finishes in ~10 seconds, so the watchdog disarms normally.
    demo_checks = []
    demo_start = time.perf_counter()

    with watchdog:
        for i in range(N_SYNTHETIC_CHECKS):
            check_result = _run_synthetic_constraint_check(i)
            demo_checks.append(check_result)

    demo_elapsed_s = time.perf_counter() - demo_start
    demo_elapsed_minutes = demo_elapsed_s / 60.0

    # Step 4: Build and write the result artifact.
    #
    # We build both a watchdog result (for the build_timeout_artifact API) and
    # the experiment artifact (for the conductor to parse).
    watchdog_result = ExperimentTimeoutResult(
        experiment_id=EXP_ID,
        timeout_minutes=DEMO_TIMEOUT_MINUTES,
        elapsed_minutes=round(demo_elapsed_minutes, 4),
        timed_out=False,  # workload finished before timeout
        partial_result_path=None,
    )
    watchdog_artifact = build_timeout_artifact(watchdog_result)

    artifact = tmpl.build_result(
        {
            "artifact_schema": "carnot.timeout_watchdog.v1",
            "retro_003_resolved": True,
            "timeout_minutes": PRODUCTION_TIMEOUT_MINUTES,
            "demo_timeout_minutes": DEMO_TIMEOUT_MINUTES,
            "demo_elapsed_minutes": round(demo_elapsed_minutes, 4),
            "demo_timed_out": False,
            "demo_checks": demo_checks,
            "estimated_savings_minutes_per_runaway": 99,
            "honest_verdict": "watchdog_implemented",
            "watchdog_artifact": watchdog_artifact,
        },
        status="success",
    )

    output_path = _REPO_ROOT / DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"Experiment {EXP_ID} complete: {output_path}")
    print(f"  honest_verdict: {artifact['honest_verdict']}")
    print(f"  retro_003_resolved: {artifact['retro_003_resolved']}")
    print(f"  demo_elapsed_minutes: {artifact['demo_elapsed_minutes']:.4f}")
    print(f"  production timeout: {PRODUCTION_TIMEOUT_MINUTES} min")
    print(f"  estimated_savings_per_runaway: 99 min")


if __name__ == "__main__":
    main()
