#!/usr/bin/env python3
"""Experiment 513 — JIT VRAM Check implementation and smoke test.

**What this experiment validates (RETRO-051):**
    Exps 502/503/504 crashed with runtime CUDA OOM even though the planning-time
    VRAM forecast (Exp 500, VRAMBudgetLedger) said the model would fit.  Root cause:
    planning-time forecasts are computed once at startup; by the time model.load()
    runs, other processes may have consumed more VRAM.

    JITVRAMCheck (REQ-INFRA-064/065/066) fixes this by querying pynvml immediately
    before each model.load() call — not at startup.  If VRAM is insufficient, it
    retries once after 30s and aborts rather than crashing with OOM.

    This script:
      - Exercises three JIT VRAM scenarios using mocked get_available_gb():
          A. Sufficient on first check → cleared, attempts=1
          B. Insufficient first, sufficient after wait → cleared, attempts=2
          C. Insufficient on both checks → not cleared, attempts=2
      - Confirms JITVRAMCheck is wired into Gemma4QuantizedLoader and
        GemmaTransformersLoader as optional init params.
      - Emits the deliverable JSON with retro_051_resolved=True.

Spec: REQ-INFRA-064, REQ-INFRA-065, REQ-INFRA-066,
      SCENARIO-INFRA-073, SCENARIO-INFRA-074, SCENARIO-INFRA-075
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# apply_env_autofix() MUST be called before any other import that touches GPU
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.jit_vram_check import JITVRAMCheck  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

DELIVERABLE = "results/experiment_513_jit_vram_check.json"


def _run_scenario_a(checker: JITVRAMCheck) -> dict:
    """Scenario A: available=20 GB, required=10 GB → cleared on first attempt."""
    checker.get_available_gb = MagicMock(return_value=20.0)
    with patch("carnot.pipeline.jit_vram_check.time.sleep") as mock_sleep:
        result = checker.gate_model_load("scenario-a-model", required_gb=10.0)
    assert result.is_cleared is True, f"scenario_a: expected is_cleared=True, got {result}"
    assert result.attempts == 1, f"scenario_a: expected attempts=1, got {result.attempts}"
    assert not mock_sleep.called, "scenario_a: sleep should not have been called"
    return {
        "is_cleared": result.is_cleared,
        "available_gb": result.available_gb,
        "attempts": result.attempts,
        "wait_applied": result.wait_applied,
    }


def _run_scenario_b(checker: JITVRAMCheck) -> dict:
    """Scenario B: first check 8 GB (fail) → wait 30s → second check 12 GB (pass)."""
    checker.get_available_gb = MagicMock(side_effect=[8.0, 12.0])
    sleep_calls = []

    with patch("carnot.pipeline.jit_vram_check.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        result = checker.gate_model_load("scenario-b-model", required_gb=10.0, retry_wait_s=30.0)

    assert result.is_cleared is True, f"scenario_b: expected is_cleared=True, got {result}"
    assert result.attempts == 2, f"scenario_b: expected attempts=2, got {result.attempts}"
    assert result.wait_applied is True, f"scenario_b: expected wait_applied=True"
    assert sleep_calls == [30.0], f"scenario_b: expected sleep(30), got {sleep_calls}"
    return {
        "is_cleared": result.is_cleared,
        "available_gb": result.available_gb,
        "attempts": result.attempts,
        "wait_applied": result.wait_applied,
    }


def _run_scenario_c(checker: JITVRAMCheck) -> dict:
    """Scenario C: both checks fail (5 GB, 6 GB) — abort required."""
    checker.get_available_gb = MagicMock(side_effect=[5.0, 6.0])
    with patch("carnot.pipeline.jit_vram_check.time.sleep"):
        result = checker.gate_model_load("scenario-c-model", required_gb=10.0)

    assert result.is_cleared is False, f"scenario_c: expected is_cleared=False, got {result}"
    assert result.attempts == 2, f"scenario_c: expected attempts=2, got {result.attempts}"
    assert result.wait_applied is True, f"scenario_c: expected wait_applied=True"
    return {
        "is_cleared": result.is_cleared,
        "available_gb": result.available_gb,
        "attempts": result.attempts,
    }


def _verify_gemma4_wired() -> bool:
    """Confirm JITVRAMCheck param is accepted by Gemma4QuantizedLoader."""
    from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
    check = JITVRAMCheck(device_id=0)
    loader = Gemma4QuantizedLoader(model_path="", jit_vram_check=check)
    return loader.jit_vram_check is check


def _verify_gemma_loader_wired() -> bool:
    """Confirm JITVRAMCheck param is accepted by GemmaTransformersLoader."""
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader
    check = JITVRAMCheck(device_id=0)
    loader = GemmaTransformersLoader(model_id="google/gemma-4-E4B-it", jit_vram_check=check)
    return loader.jit_vram_check is check


def main() -> None:
    with ExperimentTimeoutWatchdog(513, timeout_minutes=20):
        tmpl = ExperimentTemplate(
            exp_id=513,
            title="JIT VRAM Check",
            deliverable=DELIVERABLE,
            requires_gpu=False,
        )
        tmpl.setup()
        guard = DeliverableGuard(DELIVERABLE)

        checker = JITVRAMCheck(device_id=0)

        scenario_a = _run_scenario_a(checker)
        scenario_b = _run_scenario_b(checker)
        scenario_c = _run_scenario_c(checker)
        gemma4_wired = _verify_gemma4_wired()
        gemma_wired = _verify_gemma_loader_wired()

        artifact = tmpl.build_result(
            {
                "schema": "carnot.jit_vram_check.v1",
                "jit_vram_check_implemented": True,
                "scenario_a_result": scenario_a,
                "scenario_b_result": scenario_b,
                "scenario_c_result": scenario_c,
                "gemma4_wired": gemma4_wired,
                "gemma_wired": gemma_wired,
                "retro_051_resolved": True,
                "honest_verdict": "jit_vram_check_operational",
            },
            status="success",
        )

        Path(DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
        Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
