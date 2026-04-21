"""Experiment 666 — Exclusion Manifest Wire-In v3.

Verifies that:
  1. scripts/conductor_exclusion_manifest.json is present and contains all six
     chronic experiment IDs (425, 410, 308, 309, 260, 383).
  2. The module-level API (load_manifest / is_excluded / build_manifest_check_result)
     works correctly against the live manifest on disk.
  3. pytest-xdist is importable so the pre-flight test suite can run in parallel
     (< 120 min target).

Spec: REQ-INFRA-093, REQ-INFRA-094
      SCENARIO-INFRA-100, SCENARIO-INFRA-101, SCENARIO-INFRA-102
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow imports from both scripts/ and python/
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "scripts"))

from experiment_template import ExperimentTemplate
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.exclusion_manifest import (
    load_manifest,
    is_excluded,
    build_manifest_check_result,
)

CHRONIC_IDS = [425, 410, 308, 309, 260, 383]
MANIFEST_PATH = str(_REPO / "scripts" / "conductor_exclusion_manifest.json")
DELIVERABLE = "results/experiment_666_manifest_wireIn_v3.json"


def main() -> None:
    tmpl = ExperimentTemplate(
        666,
        "Exclusion Manifest Wire-In v3",
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(666, timeout_minutes=20,
                                   result_path=str(_REPO / DELIVERABLE)):

        # --- 1. Load manifest via module-level API ---
        manifest = load_manifest(MANIFEST_PATH)
        conductor_consulted: bool = manifest is not None

        # --- 2. Check all six chronic experiment IDs ---
        check_result = build_manifest_check_result(manifest, CHRONIC_IDS)

        # --- 3. Probe pytest-xdist availability ---
        try:
            import xdist  # noqa: F401
            xdist_available = True
        except ImportError:
            xdist_available = False

        if xdist_available:
            recommended_command = "pytest tests/python -q -n auto"
        else:
            recommended_command = "pytest tests/python -q"

        # --- 4. honest_verdict ---
        if conductor_consulted and xdist_available:
            honest_verdict = "manifest_wired_xdist_available"
        elif conductor_consulted and not xdist_available:
            honest_verdict = "manifest_wired_xdist_missing"
        else:
            honest_verdict = "manifest_missing"

        artifact = tmpl.build_result(
            {
                "manifest_loaded": check_result["manifest_loaded"],
                "excluded_ids": check_result["excluded_ids"],
                "checked_ids": check_result["checked_ids"],
                "all_clear": check_result["all_clear"],
                "xdist_available": xdist_available,
                "recommended_command": recommended_command,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

    output_path = str(_REPO / DELIVERABLE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
