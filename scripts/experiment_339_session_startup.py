#!/usr/bin/env python3
"""Exp 339: Pre-session startup health check (RETRO-007 + RETRO-008).

**Research question:**
    Does running a standardised pre-session GPU health check before the research
    conductor launches eliminate the zombie-VRAM waste and missing pre-flight
    issues identified in the 2026.05.06 retrospective?

**What this experiment does:**
    1. Calls ``run_session_startup(dry_run=True)`` — invokes scripts/session_startup.sh
       with --dry-run, so zombie PIDs are logged but never killed.
    2. Parses the output into a structured health report.
    3. Emits a ``carnot.session_startup.v1`` artifact with:
       - ``n_gpus_detected``: number of CUDA GPUs visible to nvidia-smi
       - ``n_zombies_found``: zombie processes at session start (0% util, >100 MiB VRAM)
       - ``n_zombies_killed``: always 0 in dry-run mode
       - ``all_healthy``: True iff n_gpus_detected >= 2 AND n_zombies_found == 0
       - ``retro_items_implemented``: ["RETRO-007", "RETRO-008"]

Spec: REQ-INFRA-008,
      SCENARIO-INFRA-012, SCENARIO-INFRA-013
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Allow script to be run from repo root without installing
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 339
TITLE = "Pre-session startup health check (RETRO-007 + RETRO-008)"
DELIVERABLE = "results/experiment_339_session_startup.json"
SCHEMA = "carnot.session_startup.v1"

RETRO_ITEMS_IMPLEMENTED = ["RETRO-007", "RETRO-008"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Run the pre-session startup health check (dry-run: no kills)
    # ------------------------------------------------------------------
    startup_result: dict[str, Any] = {}
    try:
        from carnot.pipeline.session_startup import run_session_startup

        _log.info("Running session_startup.sh --dry-run ...")
        startup_result = run_session_startup(dry_run=True)
        _log.info(
            "Session startup result: n_gpus=%d zombies=%d killed=%d all_healthy=%s",
            startup_result.get("n_gpus_detected", 0),
            startup_result.get("n_zombies_found", 0),
            startup_result.get("n_zombies_killed", 0),
            startup_result.get("all_healthy", False),
        )
    except Exception as exc:
        _log.warning("session_startup unavailable: %s — reporting unhealthy", exc)
        startup_result = {
            "n_gpus_detected": 0,
            "n_zombies_found": 0,
            "n_zombies_killed": 0,
            "all_healthy": False,
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            # "schema" is overwritten by build_result() with a sorted key list.
            # Use "artifact_schema" to persist the version string (Exp 338 pattern).
            "artifact_schema": SCHEMA,
            "n_gpus_detected": startup_result.get("n_gpus_detected", 0),
            "n_zombies_found": startup_result.get("n_zombies_found", 0),
            "n_zombies_killed": startup_result.get("n_zombies_killed", 0),
            "all_healthy": startup_result.get("all_healthy", False),
            "retro_items_implemented": RETRO_ITEMS_IMPLEMENTED,
        },
        status="success",
    )

    import json

    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)


if __name__ == "__main__":
    main()
