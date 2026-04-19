#!/usr/bin/env python3
"""Experiment 501 — Conductor CPU Routing: VRAMBudgetLedger milestone feasibility forecast.

**Researcher summary:**
    After Exp 500 (Gemma4 INT4 quantization), the conductor still holds ~9 GiB GPU 0
    VRAM from its JAX-compiled computation graph.  This leaves only 15 GiB free for
    experiment models.  Many milestone .38 experiments require 18 GiB (Gemma4-INT4 +
    Qwen simultaneously), which exceeds the 15 GiB available.

    The fix: start the conductor with JAX_PLATFORMS=cpu.  The JAX graph runs on CPU,
    the conductor holds 0 GiB GPU VRAM, and all 24 GiB become available for inference.

    This experiment uses VRAMBudgetLedger to enumerate the .38 milestone experiments,
    compute feasibility under both GPU-routed and CPU-routed conductor modes, and emit
    a structured artifact that the conductor can use as a pre-milestone planning signal.

**Artifact schema:** carnot.vram_budget_ledger.v1

Spec: REQ-INFRA-054, REQ-INFRA-055, REQ-INFRA-056,
      SCENARIO-INFRA-062, SCENARIO-INFRA-063, SCENARIO-INFRA-064
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Apply env autofix FIRST — must precede any GPU-touching import (RETRO-022 fix)
from carnot.pipeline.env_autofix import apply_env_autofix

_env_fix = apply_env_autofix()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo path setup — experiment_template.py lives in scripts/, add to path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.vram_budget_ledger import VRAMBudgetLedger  # noqa: E402

_DELIVERABLE = "results/experiment_501_conductor_cpu_routing.json"

# ---------------------------------------------------------------------------
# Milestone .38 experiment VRAM requirements (from model spec + Exp 500 data)
# ---------------------------------------------------------------------------
# exp502/503/504: Gemma4-INT4 (~9 GiB) + Qwen3-35B-A3B (~9 GiB) = 18 GiB peak
# exp507/510: single-model runs requiring ~9 GiB each
# exp511: NPU offload path — only ~2 GiB GPU VRAM (bulk on NPU SRAM)
_MILESTONE_38_EXPERIMENTS = {
    "exp502": 18.0,
    "exp503": 18.0,
    "exp504": 18.0,
    "exp507": 9.0,
    "exp510": 9.0,
    "exp511": 2.0,
}

_CONDUCTOR_VRAM_GPU_GB = 9.0   # JAX-GPU mode: conductor holds ~9 GiB GPU 0
_CONDUCTOR_VRAM_CPU_GB = 0.0   # JAX_PLATFORMS=cpu: conductor holds 0 GiB GPU
_GPU_TOTAL_GB = 24.0            # RTX 3090 / observed in Exp 480+


def main() -> None:
    """Run Experiment 501 — VRAMBudgetLedger feasibility forecast for milestone .38."""

    guard = DeliverableGuard(_DELIVERABLE)

    tmpl = ExperimentTemplate(
        exp_id=501,
        title="Conductor CPU Routing",
        deliverable=_DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(501, timeout_minutes=15):
        _run(tmpl, guard)

    tmpl.assert_deliverable_written()


def _run(tmpl: ExperimentTemplate, guard: DeliverableGuard) -> None:
    """Inner experiment body — VRAMBudgetLedger feasibility check."""

    # --- Build GPU-routed ledger (conductor on GPU, default mode) ---
    ledger = VRAMBudgetLedger(
        conductor_vram_gb=_CONDUCTOR_VRAM_GPU_GB,
        gpu_total_gb=_GPU_TOTAL_GB,
    )
    for exp_id, req_gb in _MILESTONE_38_EXPERIMENTS.items():
        ledger.add_experiment(exp_id, req_gb)
        _log.info("Registered %s: %.1f GiB required", exp_id, req_gb)

    _log.info(
        "GPU-routed conductor: conductor=%.1f GiB, total=%.1f GiB, available=%.1f GiB",
        _CONDUCTOR_VRAM_GPU_GB, _GPU_TOTAL_GB, ledger.available_gb,
    )

    forecasts = ledger.check_all()

    # Summarize results
    all_feasible = all(f.is_feasible for f in forecasts)
    some_feasible = any(f.is_feasible for f in forecasts)
    blocking_experiments = [f.exp_id for f in forecasts if not f.is_feasible]

    if all_feasible:
        honest_verdict = "all_feasible"
    elif some_feasible:
        honest_verdict = "partial_feasible"
    else:
        honest_verdict = "all_blocked"

    cpu_routing_recommendation = not all_feasible

    _log.info("Feasibility summary: %s", honest_verdict)
    _log.info("Blocking experiments: %s", blocking_experiments)
    _log.info("CPU routing recommended: %s", cpu_routing_recommendation)

    for f in forecasts:
        _log.info(
            "  %s: feasible=%s, required=%.1f GiB, available=%.1f GiB, headroom=%.1f GiB",
            f.exp_id, f.is_feasible, f.required_gb, f.available_gb, f.headroom_gb,
        )

    artifact = tmpl.build_result(
        {
            "conductor_vram_gb": _CONDUCTOR_VRAM_GPU_GB,
            "gpu_total_gb": _GPU_TOTAL_GB,
            "experiments_registered": len(_MILESTONE_38_EXPERIMENTS),
            "feasibility_results": [f.to_dict() for f in forecasts],
            "all_feasible": all_feasible,
            "blocking_experiments": blocking_experiments,
            "cpu_routing_recommendation": cpu_routing_recommendation,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )
    artifact["schema"] = "carnot.vram_budget_ledger.v1"

    _write(artifact)


def _write(artifact: dict) -> None:
    """Atomically write artifact to deliverable path."""
    out_path = Path(_DELIVERABLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, indent=2))
    tmp.replace(out_path)
    _log.info("Deliverable written: %s", out_path)


if __name__ == "__main__":
    main()
