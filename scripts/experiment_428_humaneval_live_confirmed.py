#!/usr/bin/env python3
"""Experiment 428: HumanEval live benchmark confirmation — RETRO-022 fixed.

**Researcher summary:**
    Exps 369, 380, 411, 420 all attempted to confirm the +3.0pp live result from
    Exp 226 (pass@1: 19/164 → 24/164 on Gemma4-E4B-it) but each was blocked at
    Gate 0 due to RETRO-022: CARNOT_FORCE_LIVE=1 not propagating into the conductor
    subprocess.  RETRO-022 was mitigated by apply_env_autofix() (Exp 413), which
    self-injects the env var when GPU hardware is detected.  Exp 428 is the first
    run that can clear Gate 0 because apply_env_autofix() is called before any gate
    check.

**Why code verification outperforms arithmetic verification here:**
    ArithmeticExtractor found 0 violations on instruction-tuned models (Exp 328).
    CodeExtractor runs code against test cases and detects failures structurally —
    wrong output, runtime error, type mismatch.  No regex needed.  This gives
    VerifyRepairPipeline real signal to trigger repairs on genuinely wrong code.

**Gate sequence:**
    Gate 0: apply_env_autofix() + load Exp 413 preflight verdict (informational only).
    Gate 1: LiveGPUGate.require_live_or_blocked() — CARNOT_FORCE_LIVE=1 + live GPU.
    Gate 2: check_dual_gpu_health() — WARNING if GPU1 zombie (RETRO-025), not blocking.
    Gate 3: tmpl.setup_gpu() — model pre-warm + health check (blocking).
    Gate 4: _load_model_pipeline() — tokenizer + weights (blocking).

**Pipeline per problem (reuses Exp 369 helpers):**
    1. Generate code with Gemma4-E4B-it (GPU0).
    2. Run official test cases in a subprocess (10s timeout) — record pass/fail.
    3. If failed: CodeExtractor + VerifyRepairPipeline to attempt repair.
    4. Re-run official tests on repaired code — record final pass/fail.
    5. For solutions passing official tests: run PBT to detect unofficial bugs.

**Honest verdict rules (SCENARIO-BENCH-021):**
    - ``'code_verification_positive'``: inference_mode='live_gpu' AND signed_improvement>0
    - ``'no_improvement'``: live GPU, pipeline ran, but no net improvement
    - ``'blocked'``: any gate failed before inference began

**Baseline target (Exp 226 confirmed live):**
    pass_at_1_before > 0.116  →  pass_at_1_after > 0.146  (i.e. signed_improvement > 0.03)

**Output:** results/experiment_428_humaneval_live_confirmed.json

Usage:
    # Live mode (GPU required):
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_428_humaneval_live_confirmed.py

    # CI / no-GPU: apply_env_autofix detects no GPU → Gate 1 blocks immediately
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_428_humaneval_live_confirmed.py

Spec: REQ-BENCH-004, SCENARIO-BENCH-021, REQ-INFRA-021, REQ-INFRA-022
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow import from python/ and scripts/ without installation
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Gate 0a: apply_env_autofix() — MUST be called before any GPU-dependent code.
# This is the RETRO-022 mitigation: if GPU is present but CARNOT_FORCE_LIVE is
# absent (because the conductor subprocess didn't inherit it), inject it now so
# every downstream gate sees the correct env state.
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# Now safe to import GPU-dependent modules.
from carnot.pipeline.dual_gpu_health import check_dual_gpu_health  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

# Import shared helpers from Exp 369 — no duplication of tested code.
from experiment_369_humaneval_live import (  # noqa: E402
    HumanEvalResult369,
    _load_model_pipeline,
    _load_problems,
    _process_problem,
    build_humaneval_artifact_v2,
)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 428
EXP_TITLE = "HumanEval live benchmark confirmation — RETRO-022 fixed"
DELIVERABLE = "results/experiment_428_humaneval_live_confirmed.json"

_MODEL_IDS = ["google/gemma-4-E4B-it", "Qwen/Qwen2.5-0.5B"]

# Exp 226 baseline: 19/164 = 0.1158; target: 24/164 = 0.1463
_EXP226_BASELINE = 0.116
_EXP226_TARGET = 0.146

# Exp 413 preflight verdict file — Gate 0 informational check.
_EXP413_PREFLIGHT_PATH = _REPO_ROOT / "results" / "experiment_413_env_autofix.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_preflight_verdict() -> dict[str, Any]:
    """Load and return the Exp 413 preflight artifact for Gate 0 reporting.

    This is an informational gate only — if the file is missing or malformed,
    we return a sentinel dict so the artifact records what happened rather than
    crashing.  Gate 1 (LiveGPUGate) is the actual hard gate.

    Returns:
        Exp 413 artifact dict, or a minimal sentinel if loading fails.

    Spec: REQ-INFRA-021
    """
    try:
        return json.loads(_EXP413_PREFLIGHT_PATH.read_text())
    except Exception as exc:
        _log.warning(
            "_load_preflight_verdict: could not load %s (%s) — using sentinel",
            _EXP413_PREFLIGHT_PATH,
            exc,
        )
        return {
            "honest_verdict": "preflight_file_missing",
            "retro_022_resolved": False,
            "error": str(exc),
        }


def _write_artifact(tmpl: ExperimentTemplate, artifact: dict[str, Any]) -> None:
    """Write the experiment artifact JSON to disk (pretty-printed, indent=2).

    Creates the results/ directory if it does not already exist.  The output
    path is derived from the ExperimentTemplate deliverable field so the
    conductor can find it by its standard naming convention.

    Args:
        tmpl: ExperimentTemplate instance (provides _output_path).
        artifact: JSON-serializable dict to write.

    Spec: REQ-BENCH-004
    """
    tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", tmpl._output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 428: confirm Exp 226 +3.0pp HumanEval result live.

    Gate sequence (see module docstring for full rationale):
        Gate 0: apply_env_autofix() already called at import time; load Exp 413
                preflight verdict for informational recording.
        Gate 1: LiveGPUGate.require_live_or_blocked() — hard gate.
        Gate 2: check_dual_gpu_health() — WARNING if GPU1 zombie, not blocking.
        Gate 3: tmpl.setup_gpu() — model pre-warm, hard gate.
        Gate 4: _load_model_pipeline() — weight load, hard gate.

    The ExperimentTimeoutWatchdog (60-minute cap) wraps the full inference loop
    so a runaway experiment can't hold the GPU indefinitely (RETRO-003).

    Spec: REQ-BENCH-004, SCENARIO-BENCH-021, REQ-INFRA-021, REQ-INFRA-023
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # -----------------------------------------------------------------------
    # Gate 0: Informational — report autofix result and Exp 413 preflight.
    # apply_env_autofix() was called at module import time; we just log here.
    # -----------------------------------------------------------------------
    preflight = _load_preflight_verdict()
    _log.info(
        "Gate 0 (informational): autofix_applied=%s  retro_022_resolved=%s  "
        "exp413_verdict=%s  carnot_force_live_now=%s",
        _autofix_result.auto_fix_applied,
        preflight.get("retro_022_resolved"),
        preflight.get("honest_verdict"),
        _autofix_result.final_env_value,
    )

    # -----------------------------------------------------------------------
    # Gate 1: LiveGPUGate — CARNOT_FORCE_LIVE=1 + diagnose_live_gpu().
    # Writes a blocked artifact and returns immediately on failure.
    # -----------------------------------------------------------------------
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, _MODEL_IDS)
    if blocked is not None:
        _log.error("Gate 1 (LiveGPUGate) blocked Exp 428 — writing blocked artifact.")
        # Annotate with gate 0 info before writing so operators can diagnose.
        blocked["gate0_autofix_applied"] = _autofix_result.auto_fix_applied
        blocked["gate0_preflight_verdict"] = preflight.get("honest_verdict")
        _write_artifact(tmpl, blocked)
        return

    # -----------------------------------------------------------------------
    # Gate 2: check_dual_gpu_health() — WARNING ONLY (RETRO-025).
    # GPU1 zombie means half the VRAM is wasted and inference serialises to
    # GPU0 only.  We warn but do not block: even with a zombie GPU1 we can
    # still produce a live result (just slower, on GPU0 only).
    # -----------------------------------------------------------------------
    gpu_health = check_dual_gpu_health()
    if gpu_health.gpu1_is_zombie:
        _log.warning(
            "Gate 2: GPU1 zombie detected (RETRO-025 active) — "
            "gpu1_vram_mb=%.0f gpu1_util_pct=%.0f%%.  "
            "Inference will serialise to GPU0.  Continuing.",
            gpu_health.gpu1_vram_mb,
            gpu_health.gpu1_util_pct,
        )
    if gpu_health.temperature_warning:
        _log.warning(
            "Gate 2: temperature warning — batch_size_factor=%.2f "
            "(gpu0_temp=%.0fC gpu1_temp=%.0fC).",
            gpu_health.recommended_batch_size_factor,
            gpu_health.gpu0_temp_c,
            gpu_health.gpu1_temp_c,
        )

    # -----------------------------------------------------------------------
    # Gate 3: tmpl.setup_gpu() — model pre-warm + health check.
    # Two models: Gemma4-E4B-it on GPU0 (generation), Qwen on GPU1 (repair).
    # -----------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(
        [
            {"name": "Gemma4-E4B-it", "hf_id": _MODEL_IDS[0], "gpu": 0},
            {"name": "Qwen2.5-0.5B", "hf_id": _MODEL_IDS[1], "gpu": 1},
        ]
    )
    if not gpu_status["all_healthy"]:
        _log.error(
            "Gate 3 (setup_gpu) unhealthy — models=%s.  Writing blocked artifact.",
            gpu_status["models"],
        )
        artifact = tmpl.build_result(
            {
                "humaneval_schema": "carnot.humaneval_benchmark.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "setup_gpu health check failed",
                "n_problems": 0,
                "pass_at_1_before": 0.0,
                "pass_at_1_after": 0.0,
                "signed_improvement": 0.0,
                "pbt_bugs_found": 0,
                "gpu_status": gpu_status,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    inference_mode = "live_gpu"
    _log.info("Gates 1-3 passed — inference_mode=%s", inference_mode)

    # -----------------------------------------------------------------------
    # Gate 4: Load model weights (tokenizer + causal LM).
    # Only Gemma4-E4B-it is used for code generation; Qwen is loaded via
    # setup_gpu above as the repair assistant (GPU1).
    # -----------------------------------------------------------------------
    tokenizer, model, device, ok = _load_model_pipeline(
        hf_id=_MODEL_IDS[0], device=0
    )
    if not ok:
        _log.error("Gate 4 (_load_model_pipeline) failed — writing blocked artifact.")
        artifact = tmpl.build_result(
            {
                "humaneval_schema": "carnot.humaneval_benchmark.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "model load failed after GPU gate passed",
                "n_problems": 0,
                "pass_at_1_before": 0.0,
                "pass_at_1_after": 0.0,
                "signed_improvement": 0.0,
                "pbt_bugs_found": 0,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # -----------------------------------------------------------------------
    # Load problems.
    # -----------------------------------------------------------------------
    problems = _load_problems()
    _log.info("[Exp 428] %d problems loaded.", len(problems))

    # -----------------------------------------------------------------------
    # Inference loop — wrapped in a 60-minute watchdog (RETRO-003).
    # Checkpoint every 10 problems so partial results survive a watchdog kill.
    # -----------------------------------------------------------------------
    results: list[HumanEvalResult369] = []

    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=60,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        for i, problem in enumerate(problems):
            try:
                result = _process_problem(problem, tokenizer, model, device)
                results.append(result)
            except Exception as exc:
                _log.warning("[Exp 428] problem %d error: %r", i, exc)
                results.append(
                    HumanEvalResult369(
                        problem_id=problem.get("task_id", f"unknown/{i}"),
                        generated_code="",
                        passed_tests=False,
                        violations_found=0,
                        repair_attempted=False,
                        final_code="",
                        final_passed_tests=False,
                        pbt_bug_found=False,
                    )
                )

            if (i + 1) % 10 == 0:
                tmpl.checkpoint_save(
                    {
                        "completed": i + 1,
                        "partial_results": [asdict(r) for r in results],
                    },
                    step=i + 1,
                )

    humaneval_data = build_humaneval_artifact_v2(results, inference_mode)

    _log.info(
        "[Exp 428] pass@1_before=%.3f  pass@1_after=%.3f  "
        "signed_improvement=%+.3f  honest_verdict=%s  pbt_bugs=%d  "
        "exp226_baseline=%.3f  exp226_target=%.3f",
        humaneval_data["pass_at_1_before"],
        humaneval_data["pass_at_1_after"],
        humaneval_data["signed_improvement"],
        humaneval_data["honest_verdict"],
        humaneval_data["pbt_bugs_found"],
        _EXP226_BASELINE,
        _EXP226_TARGET,
    )

    artifact = tmpl.build_result(
        {
            **humaneval_data,
            "exp226_baseline_pass_at_1": _EXP226_BASELINE,
            "exp226_target_pass_at_1": _EXP226_TARGET,
            "gate0_autofix_applied": _autofix_result.auto_fix_applied,
            "gate0_preflight_verdict": preflight.get("honest_verdict"),
            "gate2_gpu1_zombie": gpu_health.gpu1_is_zombie,
            "gate2_temperature_warning": gpu_health.temperature_warning,
        },
        status="success",
    )
    _write_artifact(tmpl, artifact)


if __name__ == "__main__":
    main()
