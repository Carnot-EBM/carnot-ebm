#!/usr/bin/env python3
"""Exp 338: Host prerequisites registry + DualGPU auto-assignment as default.

**Research question:**
    Can a centralised host-prereqs registry (RETRO-006) and automatic DualGPU
    GPU-index assignment (RETRO-004) prevent the repeated discovery waste that
    cost ~4 experiment slots in milestone 2026.04.24?

**What this experiment does:**
    1. Instantiates ``HostPrereqRegistry`` and loads ``ops/host-prereqs.md``.
    2. Runs ``check_prereqs()`` for three experiment classes: npu, fpga, live_gpu.
    3. Verifies that ``ExperimentTemplate.setup_gpu()`` now returns
       ``dual_gpu_auto_assigned=True`` when called with 2 mock model specs
       and a mocked 2-GPU environment.
    4. Emits a ``carnot.host_prereqs.v1`` artifact with:
       - ``n_packages_registered``: how many entries are in ops/host-prereqs.md
       - ``n_classes_checked``: number of experiment classes checked
       - ``dual_gpu_auto_assign_enabled``: whether REQ-INFRA-007 is active
       - ``retro_items_implemented``: list of RETRO item IDs addressed

Spec: REQ-INFRA-006, REQ-INFRA-007,
      SCENARIO-INFRA-009, SCENARIO-INFRA-010, SCENARIO-INFRA-011
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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

EXP_ID = 338
TITLE = "Host prerequisites registry + DualGPU auto-assignment default (RETRO-004/006)"
DELIVERABLE = "results/experiment_338_host_prereqs.json"
SCHEMA = "carnot.host_prereqs.v1"

# Experiment classes to check prerequisites for
CLASSES_TO_CHECK = ["npu", "fpga", "live_gpu"]

# RETRO items addressed by this experiment
RETRO_ITEMS_IMPLEMENTED = ["RETRO-004", "RETRO-006"]


# ---------------------------------------------------------------------------
# Helper: mock prewarm for DualGPU auto-assignment test
# ---------------------------------------------------------------------------


def _make_mock_prewarm():
    """Return a deterministic mock prewarm_fn for testing GPU auto-assignment.

    The mock always reports health_ok=True so the test is not blocked by
    real GPU availability.
    """
    mock_result = MagicMock()
    mock_result.health_ok = True
    mock_result.load_time_s = 0.01
    mock_result.stall_root_cause = None

    def prewarm_fn(name, hf_id, gpu):
        return mock_result

    return prewarm_fn


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
    # Step 1: Load HostPrereqRegistry
    # ------------------------------------------------------------------
    prereq_data: dict[str, Any] = {}
    try:
        from carnot.pipeline.host_prereq_registry import HostPrereqRegistry

        registry = HostPrereqRegistry()
        n_packages = len(registry.entries)
        _log.info("HostPrereqRegistry: %d packages loaded", n_packages)

        # Check prereqs for each experiment class
        class_results: dict[str, list[str]] = {}
        for cls in CLASSES_TO_CHECK:
            missing = registry.check_prereqs(experiment_class=cls)
            class_results[cls] = missing
            if missing:
                _log.info("Class %r — missing: %s", cls, missing)
            else:
                _log.info("Class %r — all prerequisites satisfied", cls)

        prereq_data = {
            "n_packages_registered": n_packages,
            "n_classes_checked": len(CLASSES_TO_CHECK),
            "classes_checked": CLASSES_TO_CHECK,
            "class_results": class_results,
            "registry_loaded": True,
        }
    except Exception as exc:
        _log.warning("HostPrereqRegistry unavailable: %s", exc)
        prereq_data = {
            "n_packages_registered": 0,
            "n_classes_checked": 0,
            "classes_checked": [],
            "class_results": {},
            "registry_loaded": False,
            "registry_error": str(exc),
        }

    # ------------------------------------------------------------------
    # Step 2: Verify DualGPU auto-assignment (REQ-INFRA-007)
    # ------------------------------------------------------------------
    dual_gpu_data: dict[str, Any] = {}
    dual_gpu_auto_assign_enabled = False
    try:
        model_specs = [
            {"name": "ModelA", "hf_id": "org/modelA", "gpu": 0},
            {"name": "ModelB", "hf_id": "org/modelB", "gpu": 0},
        ]

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch(
                "carnot.pipeline.dual_gpu_monitor.DualGPUMonitor._get_gpu_count",
                return_value=2,
            ),
        ):
            gpu_status = tmpl.setup_gpu(model_specs, prewarm_fn=_make_mock_prewarm())

        dual_gpu_auto_assign_enabled = gpu_status.get("dual_gpu_auto_assigned", False)
        dual_gpu_data = {
            "dual_gpu_auto_assign_enabled": dual_gpu_auto_assign_enabled,
            "assigned_gpu_model_a": model_specs[0]["gpu"],
            "assigned_gpu_model_b": model_specs[1]["gpu"],
            "setup_gpu_keys": sorted(gpu_status.keys()),
        }
        _log.info(
            "DualGPU auto-assignment: enabled=%s, ModelA→GPU%d, ModelB→GPU%d",
            dual_gpu_auto_assign_enabled,
            model_specs[0]["gpu"],
            model_specs[1]["gpu"],
        )
    except Exception as exc:
        _log.warning("DualGPU auto-assignment check failed: %s", exc)
        dual_gpu_data = {
            "dual_gpu_auto_assign_enabled": False,
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # Step 3: Build artifact
    # ------------------------------------------------------------------
    artifact_data: dict[str, Any] = {
        **prereq_data,
        **dual_gpu_data,
        "retro_items_implemented": RETRO_ITEMS_IMPLEMENTED,
        # artifact_schema persists as a named key; "schema" key from build_result()
        # is always the sorted list of artifact keys (not the schema version string).
        "artifact_schema": SCHEMA,
    }

    status = "success" if prereq_data.get("registry_loaded", False) else "blocked"

    artifact = tmpl.build_result(artifact_data, status=status)

    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))

    _log.info(
        "Exp 338 complete: n_packages=%d, classes=%d, dual_gpu=%s, status=%s — %s",
        artifact_data.get("n_packages_registered", 0),
        artifact_data.get("n_classes_checked", 0),
        dual_gpu_auto_assign_enabled,
        status,
        output_path,
    )


if __name__ == "__main__":
    main()
