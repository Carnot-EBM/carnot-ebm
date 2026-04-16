#!/usr/bin/env python3
"""Experiment 326: Dual GPU configuration health check.

**Purpose:**
    Implement RETRO-002 (GPU zombie detection) and RETRO-003 (dual-GPU utilisation
    enforcement) from the 2026.04.23 operational retrospective.

    Exp 219/221 wasted ~105 minutes because two models ran sequentially on GPU 0
    while GPU 1 sat idle.  Zombie GPU processes (PIDs 2592400/2595103) held ~1050 MB
    VRAM at 0% utilisation and were invisible to the experiment scaffolding.

    This experiment:
    1. Instantiates ``DualGPUMonitor`` to inspect the current GPU state.
    2. Calls ``ExperimentTemplate.setup_gpu()`` with a stub pre-warm function to
       exercise the full integration path (including the new ``gpu_monitor_results``
       key in the returned dict).
    3. Records the health snapshot in the result artifact so it becomes part of the
       permanent research record.

**Usage:**
    JAX_PLATFORMS=cpu python scripts/experiment_326_dual_gpu_config.py

Spec: REQ-INFRA-003, REQ-INFRA-004,
      SCENARIO-INFRA-004, SCENARIO-INFRA-005, SCENARIO-INFRA-006
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so ``scripts.*`` imports resolve correctly.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def main() -> None:
    """Run the Exp 326 dual-GPU configuration health check."""
    tmpl = ExperimentTemplate(
        exp_id=326,
        title="Dual GPU Configuration Health Check (RETRO-002 + RETRO-003)",
        deliverable="results/experiment_326_dual_gpu_config.json",
        requires_gpu=True,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 1: Run DualGPUMonitor directly to collect process-level data.
    # ------------------------------------------------------------------
    _log.info("Running DualGPUMonitor health check …")
    monitor = DualGPUMonitor()
    monitor_dict = monitor.to_dict()
    health = monitor_dict["health"]
    processes = monitor_dict["processes"]

    _log.info(
        "GPU health: n_gpus=%d, n_zombies=%d, idle_gpus=%s, all_healthy=%s",
        health["n_gpus_detected"],
        health["n_zombies"],
        health["idle_gpus"],
        health["all_healthy"],
    )

    if processes:
        _log.info("Detected %d GPU compute process(es):", len(processes))
        for p in processes:
            label = " [ZOMBIE]" if p["is_zombie"] else ""
            _log.info(
                "  PID %d  GPU %d  %d MiB  %d%% util%s",
                p["pid"], p["gpu_index"], p["vram_mb"], p["utilization_pct"], label,
            )
    else:
        _log.info("No GPU compute processes detected (CI mode or no GPUs present).")

    # ------------------------------------------------------------------
    # Step 2: Exercise ExperimentTemplate.setup_gpu() integration path.
    # A stub pre-warm function is used so the experiment can run without
    # live HuggingFace models being present.
    # ------------------------------------------------------------------
    _log.info("Exercising ExperimentTemplate.setup_gpu() integration …")

    def _stub_prewarm(name: str, hf_id: str, gpu: int) -> SimpleNamespace:
        """Stub that mimics a healthy model pre-warm result without loading any model."""
        return SimpleNamespace(health_ok=True, load_time_s=0.001, stall_root_cause=None)

    gpu_status = tmpl.setup_gpu(
        model_specs=[
            {"name": "StubModel-A", "hf_id": "org/stub-a", "gpu": 0},
            {"name": "StubModel-B", "hf_id": "org/stub-b", "gpu": 1},
        ],
        prewarm_fn=_stub_prewarm,
    )

    gpu_monitor_results = gpu_status.get("gpu_monitor_results", {})
    _log.info("setup_gpu() gpu_monitor_results: %s", gpu_monitor_results)

    # ------------------------------------------------------------------
    # Step 3: Build and write artifact.
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "gpu_monitor_direct": monitor_dict,
            "gpu_monitor_results": gpu_monitor_results,
            "prewarm_time_s": gpu_status["prewarm_time_s"],
            "retro_items_implemented": ["RETRO-002", "RETRO-003"],
        },
        status="success",
        schema="carnot.dual_gpu_config.v1",
    )

    output_path = _REPO_ROOT / "results" / "experiment_326_dual_gpu_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)


if __name__ == "__main__":
    main()
