#!/usr/bin/env python3
"""Experiment 440: Live HumanEval micro-benchmark (50 problems × 2 models).

**Researcher summary:**
    Exps 369/380/411/420/428 all produced scaffolding_only or blocked artifacts
    due to either RETRO-022 (CARNOT_FORCE_LIVE propagation) or excessive scope
    (164 problems requiring >45 min).  Exp 440 fixes both:

    1. apply_env_autofix() is called FIRST (RETRO-022 mitigation).
    2. Scope reduced to 50 problems × 2 models = 100 LLM calls ≈ 15–20 min —
       well inside the 45-minute watchdog (RETRO-026 mitigation).
    3. LongRunBenchmarkExecutor(batch_size=25) gives two 25-problem batches per
       model, enabling partial checkpoint recovery on any interruption.

**Why code verification works here (not arithmetic):**
    CodeExtractor uses execution — it actually runs the code and checks if tests
    pass.  It detects failures structurally (wrong output, runtime error, type
    mismatch).  No regex needed.  Instruction-tuned models produce valid Python
    that the extractor can run; ArithmeticExtractor found 0 violations on IT
    models (Exp 328) because regex pattern matching found nothing to grab.

**Gate sequence:**
    Gate 0: apply_env_autofix() — called at import time (RETRO-022).
    Gate 1: ExperimentTimeoutWatchdog(440, timeout_minutes=45) — outer budget cap.
    Gate 2: LiveGPUGate.require_live_or_blocked() — hard gate; no simulated fallback.
    Gate 3: check_dual_gpu_health() — WARNING if GPU1 zombie (RETRO-025), not blocking.
    Gate 4: tmpl.setup_gpu() — model pre-warm + health check (blocking).
    Gate 5: _load_model_pipeline() per model — tokenizer + weights (blocking).

**Pipeline per model (reuses Exp 369/428 helpers — no duplication):**
    1. Partition 50 problems into two 25-problem batches (LongRunBenchmarkExecutor).
    2. For each batch: run _process_problem() per problem (generate → test → repair → PBT).
    3. Assemble batch results into a MicroHumanEvalResult for this model.

**Honest verdict rules (SCENARIO-BENCH-027, SCENARIO-BENCH-028):**
    ``'code_verification_positive'``: inference_mode='live_gpu' AND signed_improvement > 0
    for at least one model.
    ``'code_no_improvement'``: live GPU, pipeline ran, but no model improved.
    ``'blocked'``: any gate failed before inference began.

**Output:** results/experiment_440_live_humaneval_micro.json

Usage:
    # Live mode (GPU required):
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_440_live_humaneval_micro.py

    # CI / no-GPU: apply_env_autofix detects no GPU → Gate 2 blocks immediately
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_440_live_humaneval_micro.py

Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import occurs.  Moving this below any torch/JAX import is a bug.
# See RETRO-022 for why this matters.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix, so CUDA env is already set)
# ---------------------------------------------------------------------------

import json
import logging
from dataclasses import asdict
from typing import Any

from carnot.pipeline.dual_gpu_health import check_dual_gpu_health  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.humaneval_micro import (  # noqa: E402
    MicroHumanEvalResult,
    build_micro_humaneval_artifact,
)
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

# Import ALL CodeExtractor and HumanEval helpers from Exp 369/428 — no duplication.
from experiment_369_humaneval_live import (  # noqa: E402
    HumanEvalResult369,
    _load_model_pipeline,
    _load_problems,
    _process_problem,
    compute_pass_at_1,
    compute_pass_at_1_after_repair,
)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 440
EXP_TITLE = "Live HumanEval micro-benchmark: 50 problems × 2 models"
DELIVERABLE = "results/experiment_440_live_humaneval_micro.json"

N_PROBLEMS = 50
BATCH_SIZE = 25  # two 25-problem batches per model

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen2.5-0.5B", "hf_id": "Qwen/Qwen2.5-0.5B", "gpu": 1},
]


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def _run_model_benchmark(
    model_spec: dict[str, Any],
    problems: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: str,
    executor: LongRunBenchmarkExecutor,
    exp_prefix: str,
) -> MicroHumanEvalResult:
    """Run 50 HumanEval problems for one model and return a MicroHumanEvalResult.

    **Detailed explanation for engineers:**
        Partitions the problem list into 25-problem batches using the provided
        LongRunBenchmarkExecutor.  Each batch is run under a 40-minute per-batch
        watchdog (well inside the 45-min outer cap).  Results are assembled into
        a LongRunBenchmarkResult, then aggregated into a MicroHumanEvalResult.

        pass_at_1_before / pass_at_1_after are computed via the Exp 369 helpers
        imported above — no replication of metric logic here.

    Args:
        model_spec: Dict with 'name', 'hf_id', 'gpu' keys.
        problems: List of problem dicts (task_id, prompt, entry_point, test_cases, test).
        tokenizer: Loaded tokenizer for this model.
        model: Loaded causal LM for this model.
        device: Device string (e.g. 'cuda:0').
        executor: LongRunBenchmarkExecutor (batch_size=25).
        exp_prefix: String prefix for checkpoint filenames (e.g. 'exp440_gemma').

    Returns:
        MicroHumanEvalResult capturing the full benchmark lifecycle for this model.

    Spec: REQ-BENCH-010, SCENARIO-BENCH-027
    """
    model_name = model_spec["name"]
    _log.info("[Exp 440] Starting benchmark for model=%s  n_problems=%d", model_name, len(problems))

    batches = executor.partition(problems)

    def _inference_fn(problem: dict[str, Any]) -> dict[str, Any]:
        # Returns a plain dict (JSON-serializable) so LongRunBenchmarkExecutor
        # can checkpoint results to disk between batches without a custom encoder.
        try:
            result = _process_problem(problem, tokenizer, model, device)
            return asdict(result)
        except Exception as exc:
            _log.warning("[Exp 440] problem %s error: %r", problem.get("task_id", "?"), exc)
            return asdict(HumanEvalResult369(
                problem_id=problem.get("task_id", "unknown"),
                generated_code="",
                passed_tests=False,
                violations_found=0,
                repair_attempted=False,
                final_code="",
                final_passed_tests=False,
                pbt_bug_found=False,
            ))

    completed_batches = []
    for batch in batches:
        batch = executor.run_batch(batch, _inference_fn, watchdog_timeout_minutes=40)
        executor.save_batch(batch, prefix=exp_prefix)
        completed_batches.append(batch)

    long_result = executor.assemble(completed_batches)
    # Reconstruct HumanEvalResult369 objects from the checkpointed dicts so the
    # Exp 369 metric helpers (compute_pass_at_1, etc.) get the correct types.
    all_results: list[HumanEvalResult369] = [
        HumanEvalResult369(**d) for d in long_result.all_results
    ]

    pass_before = compute_pass_at_1(all_results)
    pass_after = compute_pass_at_1_after_repair(all_results)
    signed_improvement = round(pass_after - pass_before, 6)
    pbt_bugs = sum(1 for r in all_results if r.pbt_bug_found)

    _log.info(
        "[Exp 440] model=%s  pass@1_before=%.3f  pass@1_after=%.3f  "
        "signed_improvement=%+.3f  pbt_bugs=%d  executor_verdict=%s",
        model_name, pass_before, pass_after, signed_improvement,
        pbt_bugs, long_result.honest_verdict,
    )

    return MicroHumanEvalResult(
        model_id=model_spec["hf_id"],
        n_problems=len(all_results),
        pass_at_1_before=pass_before,
        pass_at_1_after=pass_after,
        signed_improvement=signed_improvement,
        pbt_bugs_found=pbt_bugs,
        inference_mode="live_gpu",
    )


# ---------------------------------------------------------------------------
# Artifact write helper
# ---------------------------------------------------------------------------


def _write_artifact(tmpl: ExperimentTemplate, artifact: dict[str, Any]) -> None:
    """Write the experiment artifact JSON to disk (pretty-printed, indent=2).

    Creates the results/ directory if it does not already exist.  The output
    path is derived from the ExperimentTemplate deliverable field.

    Args:
        tmpl: ExperimentTemplate instance (provides _output_path).
        artifact: JSON-serializable dict to write.

    Spec: REQ-BENCH-010
    """
    tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", tmpl._output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 440: live HumanEval micro-benchmark (50 problems × 2 models).

    Gate sequence (see module docstring for full rationale):
        Gate 0: apply_env_autofix() already called at import time (RETRO-022).
        Gate 1: ExperimentTimeoutWatchdog(440, timeout_minutes=45) — outer budget cap.
        Gate 2: LiveGPUGate.require_live_or_blocked() — hard gate.
        Gate 3: check_dual_gpu_health() — WARNING ONLY if GPU1 zombie (RETRO-025).
        Gate 4: tmpl.setup_gpu() — model pre-warm, hard gate.
        Gate 5: _load_model_pipeline() per model — weight load, hard gate.

    Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # -----------------------------------------------------------------------
    # Gate 0: informational — log autofix status.
    # apply_env_autofix() was called at module import time.
    # -----------------------------------------------------------------------
    _log.info(
        "Gate 0 (informational): autofix_applied=%s  carnot_force_live_now=%s",
        _autofix_result.auto_fix_applied,
        _autofix_result.final_env_value,
    )

    # -----------------------------------------------------------------------
    # Gate 1: ExperimentTimeoutWatchdog — outer 45-minute cap (RETRO-003).
    # The watchdog is started here and stays active for the full inference loop.
    # -----------------------------------------------------------------------
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=45,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )
    watchdog.start()

    gpu_health = None  # initialized before try so it is visible in the finally / artifact block

    try:
        # -------------------------------------------------------------------
        # Gate 2: LiveGPUGate — CARNOT_FORCE_LIVE=1 + live GPU check.
        # -------------------------------------------------------------------
        blocked = LiveGPUGate.require_live_or_blocked(tmpl, [s["hf_id"] for s in MODEL_SPECS])
        if blocked is not None:
            _log.error("Gate 2 (LiveGPUGate) blocked Exp 440 — writing blocked artifact.")
            blocked["gate0_autofix_applied"] = _autofix_result.auto_fix_applied
            _write_artifact(tmpl, blocked)
            return

        # -------------------------------------------------------------------
        # Gate 3: check_dual_gpu_health() — WARNING ONLY (RETRO-025).
        # GPU1 zombie means half the VRAM is wasted; we warn but do not block.
        # -------------------------------------------------------------------
        gpu_health = check_dual_gpu_health()
        if gpu_health.gpu1_is_zombie:
            _log.warning(
                "Gate 3: GPU1 zombie detected (RETRO-025 active) — "
                "gpu1_vram_mb=%.0f gpu1_util_pct=%.0f%%.  "
                "Inference will serialise to GPU0.  Continuing.",
                gpu_health.gpu1_vram_mb,
                gpu_health.gpu1_util_pct,
            )
        if gpu_health.temperature_warning:
            _log.warning(
                "Gate 3: temperature warning — batch_size_factor=%.2f "
                "(gpu0_temp=%.0fC gpu1_temp=%.0fC).",
                gpu_health.recommended_batch_size_factor,
                gpu_health.gpu0_temp_c,
                gpu_health.gpu1_temp_c,
            )

        # -------------------------------------------------------------------
        # Gate 4: tmpl.setup_gpu() — model pre-warm + health check.
        # -------------------------------------------------------------------
        gpu_status = tmpl.setup_gpu(MODEL_SPECS)
        if not gpu_status["all_healthy"]:
            _log.error(
                "Gate 4 (setup_gpu) unhealthy — models=%s.  Writing blocked artifact.",
                gpu_status["models"],
            )
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.humaneval_micro.v1",
                    "inference_mode": "blocked",
                    "honest_verdict": "blocked",
                    "failure_reason": "setup_gpu health check failed",
                    "n_problems": 0,
                    "per_model_results": [],
                    "gpu_status": gpu_status,
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            return

        _log.info("Gates 2–4 passed — inference_mode=live_gpu")

        # -------------------------------------------------------------------
        # Load problems (50 total).
        # -------------------------------------------------------------------
        problems = _load_problems()[:N_PROBLEMS]
        _log.info("[Exp 440] %d problems loaded.", len(problems))

        # -------------------------------------------------------------------
        # Gate 5 + inference loop per model.
        # -------------------------------------------------------------------
        executor = LongRunBenchmarkExecutor(
            batch_size=BATCH_SIZE,
            checkpoint_dir=str(_REPO_ROOT / "results" / "batch_ckpt_exp440"),
        )

        micro_results: list[MicroHumanEvalResult] = []
        for model_spec in MODEL_SPECS:
            # Gate 5: load model weights (tokenizer + causal LM).
            tokenizer, model, device, ok = _load_model_pipeline(
                hf_id=model_spec["hf_id"], device=model_spec["gpu"]
            )
            if not ok:
                _log.error(
                    "Gate 5 (_load_model_pipeline) failed for %s — appending blocked result.",
                    model_spec["hf_id"],
                )
                micro_results.append(
                    MicroHumanEvalResult(
                        model_id=model_spec["hf_id"],
                        n_problems=0,
                        pass_at_1_before=0.0,
                        pass_at_1_after=0.0,
                        signed_improvement=0.0,
                        pbt_bugs_found=0,
                        inference_mode="blocked",
                    )
                )
                continue

            exp_prefix = f"exp440_{model_spec['name'].lower().replace('-', '_')}"
            result = _run_model_benchmark(
                model_spec=model_spec,
                problems=problems,
                tokenizer=tokenizer,
                model=model,
                device=device,
                executor=executor,
                exp_prefix=exp_prefix,
            )
            micro_results.append(result)

            # Checkpoint aggregate progress so partial results survive a kill.
            tmpl.checkpoint_save(
                {"completed_models": [asdict(r) for r in micro_results]},
                step=len(micro_results),
            )

    finally:
        watchdog.stop()

    # -----------------------------------------------------------------------
    # Assemble final artifact.
    # -----------------------------------------------------------------------
    humaneval_micro_data = build_micro_humaneval_artifact(micro_results)

    _log.info(
        "[Exp 440] honest_verdict=%s  inference_mode=%s  n_models=%d",
        humaneval_micro_data["honest_verdict"],
        humaneval_micro_data["inference_mode"],
        len(micro_results),
    )
    for r in micro_results:
        _log.info(
            "[Exp 440]   model=%s  pass@1_before=%.3f  pass@1_after=%.3f  "
            "signed_improvement=%+.3f  pbt_bugs=%d",
            r.model_id, r.pass_at_1_before, r.pass_at_1_after,
            r.signed_improvement, r.pbt_bugs_found,
        )

    artifact = tmpl.build_result(
        {
            **humaneval_micro_data,
            "gate0_autofix_applied": _autofix_result.auto_fix_applied,
            "gate3_gpu1_zombie": gpu_health.gpu1_is_zombie if gpu_health is not None else False,
            "gate3_temperature_warning": gpu_health.temperature_warning if gpu_health is not None else False,
        },
        status="success",
    )
    _write_artifact(tmpl, artifact)


if __name__ == "__main__":
    main()


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
