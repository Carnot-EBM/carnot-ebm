#!/usr/bin/env python3
"""Experiment 426: DualGPU Fix + Temp Guard — close RETRO-025.

**Researcher summary (RETRO-025):**
    PID 3509070 held 1786 MB on GPU1 at 0% utilization while GPU0 ran at 88%
    for 144+ minutes.  GPU0 also reached 82C.  This experiment verifies that
    the two fixes (GPU1 zombie detection and temperature guard) are correctly
    implemented and reports an honest verdict about the current GPU state.

**What this experiment does:**
    1. Calls ``apply_env_autofix()`` to ensure CARNOT_FORCE_LIVE is set if GPU
       is present.
    2. Uses ``ExperimentTimeoutWatchdog`` (Exp 425 pattern) as a context
       manager to cap wall time at 45 minutes.
    3. Calls ``check_dual_gpu_health()`` to snapshot current GPU state.
    4. Reads ``results/operational_retro_2026_04_31.json`` for RETRO-025
       context (GPU1 VRAM and utilization values from the incident).
    5. Builds an artifact with ``honest_verdict``, ``retro_025_status``, and
       the full dual-GPU health snapshot.
    6. Writes the artifact to ``results/experiment_426_dual_gpu_fix.json``.

**Why CPU-only / light GPU?**
    This experiment is about metadata-level infrastructure checks, not
    inference.  No models are loaded; no GPU compute is required.  The
    experiment runs correctly on CI machines with no GPU hardware.

**Verdict semantics:**
    - ``'zombie_detected'`` — GPU1 currently has >500 MB VRAM but <1% util
      (RETRO-025 pattern is still active)
    - ``'gpu1_healthy'`` — GPU1 either has no VRAM allocation or is computing
      (RETRO-025 zombie pattern is cleared)

Spec: REQ-INFRA-025, REQ-INFRA-026,
      SCENARIO-INFRA-031, SCENARIO-INFRA-032, SCENARIO-INFRA-033 (Exp 426)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path so all imports resolve.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() FIRST — before any other imports that touch GPU.
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Now import the rest of the pipeline.
# ---------------------------------------------------------------------------

from carnot.pipeline.dual_gpu_health import (  # noqa: E402
    build_gpu_fix_artifact,
    check_dual_gpu_health,
)
from carnot.pipeline.experiment_watchdog import (  # noqa: E402
    ExperimentTimeoutWatchdog,
    get_timeout_minutes,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 426
EXP_TITLE = "DualGPU Fix + Temp Guard"
DELIVERABLE = "results/experiment_426_dual_gpu_fix.json"
RETRO_PATH = "results/operational_retro_2026_04_31.json"


def run_experiment() -> dict:
    """Core experiment logic.

    Reads RETRO-025 context, checks current GPU health, and builds the
    artifact.  Separated from ``main()`` so tests can call it directly.
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    # --- Read RETRO-025 context ---
    retro_data: dict = {}
    retro_path = _REPO_ROOT / RETRO_PATH
    if retro_path.exists():
        try:
            retro_data = json.loads(retro_path.read_text())
            _log.info("Loaded RETRO-025 context from %s", retro_path)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to read retro file: %s", exc)
    else:
        _log.warning("RETRO-025 file not found at %s — proceeding without context", retro_path)

    # --- Step 4: check_dual_gpu_health() ---
    _log.info("Running check_dual_gpu_health()...")
    health = check_dual_gpu_health(timeout_seconds=60)

    _log.info(
        "GPU health snapshot: gpu0_util=%.0f%% gpu1_util=%.0f%% "
        "gpu0_temp=%.0fC gpu1_temp=%.0fC gpu1_vram=%.0fMB "
        "zombie=%s temp_warn=%s factor=%.2f",
        health.gpu0_util_pct,
        health.gpu1_util_pct,
        health.gpu0_temp_c,
        health.gpu1_temp_c,
        health.gpu1_vram_mb,
        health.gpu1_is_zombie,
        health.temperature_warning,
        health.recommended_batch_size_factor,
    )

    # --- Step 5: build artifact ---
    gpu_fix = build_gpu_fix_artifact(health, prior_retro_path=RETRO_PATH)

    # Extract RETRO-025 incident values for comparison (from the retro JSON)
    retro_gpu1_vram = None
    retro_gpu1_util = None
    if retro_data:
        gpu_util = retro_data.get("gpu_utilization", {})
        retro_gpu1_vram = gpu_util.get("gpu_1_vram_used_mb")
        retro_gpu1_util = gpu_util.get("gpu_1_utilization_pct")

    artifact = tmpl.build_result(
        {
            **gpu_fix,
            "env_autofix": {
                "gpu_detected": _autofix_result.gpu_detected,
                "auto_fix_applied": _autofix_result.auto_fix_applied,
                "final_env_value": _autofix_result.final_env_value,
            },
            "retro_025_incident_gpu1_vram_mb": retro_gpu1_vram,
            "retro_025_incident_gpu1_util_pct": retro_gpu1_util,
        },
        status="success",
    )

    # --- Write output ---
    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)

    return artifact


def main() -> None:
    """Entry point: run experiment inside the timeout watchdog."""
    timeout_minutes = get_timeout_minutes()
    result_path = str(_REPO_ROOT / DELIVERABLE)

    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=timeout_minutes,
        result_path=result_path,
    ):
        artifact = run_experiment()

    # Log headline results
    _log.info(
        "Exp %d complete: honest_verdict=%s retro_025_status=%s "
        "temperature_warning=%s recommended_batch_factor=%.2f",
        EXP_ID,
        artifact.get("honest_verdict"),
        artifact.get("retro_025_status"),
        artifact.get("temperature_warning"),
        artifact.get("recommended_batch_size_factor", 1.0),
    )


if __name__ == "__main__":
    main()
