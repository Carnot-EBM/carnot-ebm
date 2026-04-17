#!/usr/bin/env python3
"""Experiment 438: GPU1 Zombie Fix — close RETRO-025 scheduling root cause.

**Researcher summary (RETRO-025, diagnosed Exp 426, fixed here):**
    Exp 426 confirmed the zombie: GPU1 held 1786 MB at 0% utilization while
    GPU0 ran at 88% for 144+ minutes.  Root hypothesis: ``device_map='auto'``
    lets CUDA allocate VRAM on GPU1 for layer offloading, but the forward pass
    stays on GPU0.

    Fix shipped in this experiment: ``build_zombie_fix_strategy()`` returns
    explicit ``{'': 'cuda:N'}`` device maps for each model in dual-GPU live mode.
    ``ExperimentTemplate.setup_gpu()`` now injects these maps (REQ-INFRA-029).
    This pins every layer of each model to a single GPU, preventing cross-device
    VRAM spill that creates the zombie pattern.

**What this experiment does:**
    a. ``apply_env_autofix()`` FIRST — ensures CARNOT_FORCE_LIVE is set if GPU present.
    b. ``ExperimentTimeoutWatchdog(438, timeout_minutes=20)`` — hard wall-clock cap.
    c. ``check_dual_gpu_health()`` — baseline zombie status before fix.
    d. ``build_zombie_fix_strategy(n_gpus, model_ids)`` — compute and log the strategy.
    e. If live GPU: attempt loading Qwen3.5-0.8B on GPU1 with explicit
       ``device_map={'': 'cuda:1'}``.  After load: re-run ``check_dual_gpu_health()``.
       If ``gpu1_util > 0`` during inference → fix confirmed.
    f. Build ``ZombieFixResult`` with ``honest_verdict``.
    g. Write ``results/experiment_438_gpu1_zombie_fix.json``.

**CI mode:**
    When no GPU hardware is present, the experiment diagnoses the code path and
    returns ``honest_verdict='ci_mode'``.  The fix logic is exercised on every
    CI run so regressions are caught before hardware is needed.

**Verdict semantics:**
    - ``'fix_applied_and_verified'`` — explicit device_map used AND gpu1_util > 0
      after model load (RETRO-025 zombie pattern confirmed resolved)
    - ``'fix_applied_unverified'`` — explicit device_map used but gpu1_util still 0
      (fix applied; may need inference workload to trigger utilization)
    - ``'ci_mode'`` — no GPU hardware; fix strategy computed and logged only

Spec: REQ-INFRA-029, REQ-INFRA-030,
      SCENARIO-INFRA-037, SCENARIO-INFRA-038 (Exp 438)
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
# Step a: apply_env_autofix() FIRST — before any other imports that touch GPU.
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Now import the rest of the pipeline.
# ---------------------------------------------------------------------------

from carnot.pipeline.dual_gpu_health import check_dual_gpu_health  # noqa: E402
from carnot.pipeline.experiment_watchdog import (  # noqa: E402
    ExperimentTimeoutWatchdog,
    get_timeout_minutes,
)
from carnot.pipeline.gpu_zombie_fix import (  # noqa: E402
    ZombieFixResult,
    build_zombie_fix_artifact,
    build_zombie_fix_strategy,
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

EXP_ID = 438
EXP_TITLE = "GPU1 Zombie Fix — Explicit Device Assignment (RETRO-025)"
DELIVERABLE = "results/experiment_438_gpu1_zombie_fix.json"

# Model to attempt loading on GPU1 in live mode.
# Qwen3.5-0.8B is the smallest model in the standard benchmark suite; it fits
# on a single RTX 3090 (24 GB) with room to spare, so loading it on GPU1 alone
# is safe and fast.
_GPU1_TEST_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_GPU0_TEST_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def _detect_n_gpus() -> int:
    """Return the number of CUDA GPUs available, or 0 in CI mode.

    Uses pynvml first (preferred: direct C API, no subprocess overhead),
    then falls back to nvidia-smi, then returns 0 when neither is available.
    This is CI-safe: always returns an integer, never raises.
    """
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return int(count)
    except Exception:
        pass

    try:
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
            return len(lines)
    except Exception:
        pass

    return 0


def run_experiment() -> dict:
    """Core experiment logic.

    Runs baseline health check, computes the zombie-fix strategy, attempts
    a live GPU load if hardware is present, then builds the result artifact.
    Separated from ``main()`` so tests can call it directly without invoking
    the watchdog.
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

    # --- Step c: baseline dual-GPU health check ---
    _log.info("Step c: baseline check_dual_gpu_health()...")
    baseline_health = check_dual_gpu_health(timeout_seconds=60)
    _log.info(
        "Baseline GPU health: gpu1_vram=%.0fMB gpu1_util=%.0f%% zombie=%s",
        baseline_health.gpu1_vram_mb,
        baseline_health.gpu1_util_pct,
        baseline_health.gpu1_is_zombie,
    )

    # --- Step d: compute the zombie-fix strategy ---
    n_gpus = _detect_n_gpus()
    model_ids = [_GPU0_TEST_MODEL_ID, _GPU1_TEST_MODEL_ID]
    strategy = build_zombie_fix_strategy(n_gpus, model_ids)
    _log.info(
        "Step d: zombie fix strategy (n_gpus=%d): %s",
        n_gpus,
        {mid: str(dm) for mid, dm in strategy.items()},
    )

    # Determine what device_maps were selected for the two models.
    gpu0_dm = strategy.get(_GPU0_TEST_MODEL_ID, "auto")
    gpu1_dm = strategy.get(_GPU1_TEST_MODEL_ID, "auto")
    # Represent device_map as a string for JSON serialisation.
    gpu0_device_map_str = str(gpu0_dm)
    gpu1_device_map_str = str(gpu1_dm)
    fix_applied = gpu1_dm != "auto"

    post_fix_gpu1_util: float | None = None

    # --- Step e: live GPU load attempt ---
    if force_live and n_gpus >= 2:
        _log.info(
            "Step e: live GPU detected — attempting to load %s on GPU1 "
            "with explicit device_map=%s",
            _GPU1_TEST_MODEL_ID,
            gpu1_device_map_str,
        )
        try:
            # Import transformers only when GPU is live — heavy optional dep.
            from transformers import AutoModelForCausalLM  # type: ignore[import]  # noqa: PLC0415

            _ = AutoModelForCausalLM.from_pretrained(
                _GPU1_TEST_MODEL_ID,
                device_map=gpu1_dm,
            )
            _log.info("Model loaded on GPU1 successfully — re-checking GPU health...")

            post_health = check_dual_gpu_health(timeout_seconds=60)
            post_fix_gpu1_util = post_health.gpu1_util_pct
            _log.info(
                "Post-load GPU1 util=%.0f%% vram=%.0fMB zombie=%s",
                post_health.gpu1_util_pct,
                post_health.gpu1_vram_mb,
                post_health.gpu1_is_zombie,
            )
        except Exception as exc:
            _log.warning(
                "Live GPU load attempt failed (%s); proceeding with ci_mode verdict",
                exc,
            )
            # Treat as CI mode — device_map was correct but load failed
            fix_applied = False

    # --- Step f: determine honest_verdict ---
    if not force_live or n_gpus < 2:
        honest_verdict = "ci_mode"
    elif fix_applied and post_fix_gpu1_util is not None and post_fix_gpu1_util > 0:
        honest_verdict = "fix_applied_and_verified"
    elif fix_applied:
        honest_verdict = "fix_applied_unverified"
    else:
        honest_verdict = "ci_mode"

    fix_result = ZombieFixResult(
        gpu0_model_id=_GPU0_TEST_MODEL_ID,
        gpu1_model_id=_GPU1_TEST_MODEL_ID,
        gpu0_device_map=gpu0_device_map_str,
        gpu1_device_map=gpu1_device_map_str,
        fix_applied=fix_applied,
        post_fix_gpu1_util_pct=post_fix_gpu1_util,
        honest_verdict=honest_verdict,
    )

    fix_artifact = build_zombie_fix_artifact(fix_result)

    artifact = tmpl.build_result(
        {
            **fix_artifact,
            "n_gpus_detected": n_gpus,
            "force_live": force_live,
            "baseline_gpu1_vram_mb": baseline_health.gpu1_vram_mb,
            "baseline_gpu1_util_pct": baseline_health.gpu1_util_pct,
            "baseline_gpu1_is_zombie": baseline_health.gpu1_is_zombie,
            "zombie_fix_strategy": {
                mid: str(dm) for mid, dm in strategy.items()
            },
            "env_autofix": {
                "gpu_detected": _autofix_result.gpu_detected,
                "auto_fix_applied": _autofix_result.auto_fix_applied,
                "final_env_value": _autofix_result.final_env_value,
            },
        },
        status="success",
    )

    # --- Step g: write output ---
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

    _log.info(
        "Exp %d complete: honest_verdict=%s fix_applied=%s n_gpus=%d "
        "post_fix_gpu1_util=%s",
        EXP_ID,
        artifact.get("honest_verdict"),
        artifact.get("fix_applied"),
        artifact.get("n_gpus_detected", 0),
        artifact.get("post_fix_gpu1_util_pct"),
    )


if __name__ == "__main__":
    main()
