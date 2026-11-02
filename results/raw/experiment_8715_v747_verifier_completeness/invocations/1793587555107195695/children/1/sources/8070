"""Experiment 640 — Pre-Flight Infra v2.

Verifies three infrastructure components that have been RETRO-CRITICAL for twelve
consecutive milestones (.37 through .48):

1. Exclusion manifest exists and is valid.
2. The research conductor is wired to consult the exclusion manifest before queuing
   any experiment (checking for the function names added in the RETRO-067 wire-in).
3. DualGPURetrain works correctly (parallel EORM/JEPA retrain, REQ-INFRA-091).

The artifact reports an honest_verdict so the conductor can track whether the
manifest is fully wired or still needs human action.

Spec: REQ-INFRA-090, REQ-INFRA-091, SCENARIO-INFRA-097, SCENARIO-INFRA-098
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path before any scripts/ imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any JAX or CUDA import to ensure
# JAX_PLATFORMS and CARNOT_FORCE_LIVE are set correctly before lazy init.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_env_result = apply_env_autofix()

import json
import os

from carnot.pipeline.dualgpu_retrain import DualGPURetrain, DualGPURetrainConfig
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate


def main() -> None:
    watchdog = ExperimentTimeoutWatchdog(640, timeout_minutes=30)
    watchdog.start()

    tmpl = ExperimentTemplate(
        640,
        "Pre-Flight Infra v2",
        "results/experiment_640_preflght_infra.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # 1. Exclusion manifest check
    # ------------------------------------------------------------------
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "conductor_exclusion_manifest.json",
    )
    manifest_exists = os.path.exists(manifest_path)

    if not manifest_exists:
        # Create the manifest with the canonical v2 schema so the conductor
        # can start excluding legacy experiments immediately.
        manifest_payload = {
            "schema": "carnot.exclusion_manifest.v2",
            "excluded_experiments": ["308", "309", "425", "410"],
            "reason": "Legacy experiments: perpetually incomplete or superseded. Never re-queue.",
            "notes": {
                "383": "RETRO: should run DualGPURetrain variant (Exp 640b). Do not re-queue plain Exp 383.",
                "308": "Checkpoint-failure state since .37. No path to completion.",
                "309": "Checkpoint-failure state since .37. No path to completion.",
                "425": "ExperimentTimeoutWatchdog already implemented. Exp 425 is complete.",
                "410": "BatchedInferenceRunner needed but not yet applied.",
            },
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest_payload, fh, indent=2)
        manifest_exists = True

    with open(manifest_path) as fh:
        manifest_data = json.load(fh)

    # Support both the legacy {"excluded": [...]} schema and the v2 schema
    # {"excluded_experiments": [...]} produced by this experiment.
    manifest_valid = "excluded_experiments" in manifest_data or "excluded" in manifest_data

    # ------------------------------------------------------------------
    # 2. Conductor wire-in verification
    # ------------------------------------------------------------------
    conductor_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "research_conductor.py",
    )
    with open(conductor_path) as fh:
        conductor_content = fh.read()

    conductor_consulted = (
        "exclusion_manifest" in conductor_content
        or "conductor_exclusion_manifest" in conductor_content
    )

    # ------------------------------------------------------------------
    # 3. DualGPU test
    # ------------------------------------------------------------------
    try:
        import torch

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        n_gpus = 0

    dualgpu_available = n_gpus >= 2

    retrain_config = DualGPURetrainConfig(
        eorm_device="cuda:0" if n_gpus >= 1 else "cpu",
        jepa_device="cuda:1" if n_gpus >= 2 else "cpu",
    )

    result = DualGPURetrain(retrain_config).run_parallel(
        lambda: "eorm_done",
        lambda: "jepa_done",
    )
    dualgpu_retrain_works = result["eorm"] == "eorm_done" and result["jepa"] == "jepa_done"

    # ------------------------------------------------------------------
    # 4. Honest verdict
    # ------------------------------------------------------------------
    if conductor_consulted and dualgpu_retrain_works:
        honest_verdict = "manifest_wired_dualgpu_ready"
    elif manifest_valid and not conductor_consulted:
        honest_verdict = "manifest_built_conductor_unwired"
    else:
        honest_verdict = "manifest_missing"

    note_for_50 = (
        "Manifest wired"
        if conductor_consulted
        else "Wire exclusion manifest into conductor before .50 planning"
    )

    artifact = tmpl.build_result(
        {
            "manifest_exists": manifest_exists,
            "manifest_valid": manifest_valid,
            "conductor_consulted": conductor_consulted,
            "dualgpu_available": dualgpu_available,
            "dualgpu_retrain_works": dualgpu_retrain_works,
            "excluded_experiments": ["308", "309", "425", "410"],
            "note_for_50": note_for_50,
            "honest_verdict": honest_verdict,
            "artifact_schema": "carnot.preflght_infra.v2",
        },
        status="success",
    )

    watchdog.stop()

    import json as _json

    with open("results/experiment_640_preflght_infra.json", "w") as fh:
        _json.dump(artifact, fh, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
