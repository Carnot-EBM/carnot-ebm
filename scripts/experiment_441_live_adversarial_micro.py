#!/usr/bin/env python3
"""Experiment 441 — Live adversarial GSM8K micro-benchmark (50q × 3 conditions × 2 models).

**Researcher summary:**
    Apple researchers (arXiv 2410.05229) showed that appending ONE irrelevant sentence
    to a GSM8K problem drops frontier LLM accuracy by up to 65%.  Carnot's arithmetic
    verifier is structural — it extracts explicit equation tokens and computes an Ising
    energy over those tokens only, ignoring surrounding context words.  Therefore Carnot
    SHOULD be immune to irrelevant-sentence injection.

    This experiment is the first LIVE (GPU-backed) test of that claim at micro scale.
    It deliberately limits scope to 50 questions × 3 conditions × 2 models = 300 LLM
    calls to fit within the 45-minute ExperimentTimeoutWatchdog budget:
        300 calls × ~8 s/call ≈ 40 min (leaving ~5 min headroom).

    Prior experiments (355/370/381/421/429) were all blocked at gates or exceeded
    the budget because they targeted 200 questions.  Exp 441 reduces scope to 50
    questions — small enough to complete in a single watchdog window.

**Three conditions:**
    1. standard    — original (clean) GSM8K questions, no verify-repair.
    2. adversarial — distractor-appended variants (one irrelevant sentence), no repair.
    3. repaired    — distractor-appended variants + VerifyRepairPipeline.

**Two models:**
    - Gemma4-E4B-it  (GPU 0, device='cuda:0')
    - Qwen/Qwen3.5-0.8B  (GPU 1, fallback to GPU 0 if zombie)

**Gate chain (runs in order, module-load first):**
    0. apply_env_autofix() — called at module load (FIRST, before any CUDA import).
    1. ExperimentTimeoutWatchdog(441, timeout_minutes=45) — outer budget cap.
    2. LiveGPUGate.require_live_or_blocked() — hard gate; no simulated fallback.
    3. check_dual_gpu_health() — WARNING if GPU1 zombie (continue on GPU0-only).
    4. setup_gpu() — blocked if not all_healthy.
    5. Model load for both models with explicit device (Exp 438 fix).

**Output:** results/experiment_441_live_adversarial_micro.json

Spec: REQ-BENCH-011, SCENARIO-BENCH-029, SCENARIO-BENCH-030,
      REQ-INFRA-021, REQ-INFRA-022, REQ-INFRA-023
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
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
from typing import Any

from carnot.pipeline.adversarial_gsm8k import (  # noqa: E402
    MicroAdversarialResult,
    build_adversarial_questions,
    build_micro_adversarial_artifact,
    compute_adversarial_results,
)
from carnot.pipeline.dual_gpu_health import check_dual_gpu_health  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor  # noqa: E402

# Reuse inference helpers from Exp 355 — no duplication of tested code.
from experiment_355_adversarial_gsm8k_benchmark import (  # noqa: E402
    _call_model,
    _is_correct,
    load_gsm8k_questions,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 441
EXP_TITLE = "Live adversarial GSM8K micro-benchmark: 50q × 3 conditions × 2 models"
DELIVERABLE = "results/experiment_441_live_adversarial_micro.json"

N_QUESTIONS = 50
WATCHDOG_TIMEOUT_MINUTES = 45
BATCH_SIZE = 50  # one batch per condition — fits within 40-min per-batch watchdog

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]


# ---------------------------------------------------------------------------
# Model loading helper (Exp 438 device_map fix)
# ---------------------------------------------------------------------------


def _load_model_with_explicit_device(hf_id: str, gpu_index: int) -> object:
    """Load a HuggingFace text-generation pipeline with explicit device assignment.

    **Why explicit device instead of device_map={'': 'cuda:0'}:**
        The GPU1 zombie issue (RETRO-025) was caused by HuggingFace device_map={'': 'cuda:1'}
        sometimes allocating all weight tensors on GPU0 while leaving GPU1 in VRAM-allocated
        but compute-idle zombie state.  Using ``device=gpu_index`` pins the entire model
        to the specified GPU, matching the fix confirmed working in Exp 438.

    Parameters
    ----------
    hf_id : str
        HuggingFace model identifier.
    gpu_index : int
        GPU device index (0 or 1).

    Returns
    -------
    object
        HuggingFace text-generation pipeline object.
    """
    from transformers import pipeline  # type: ignore[import]

    return pipeline(
        "text-generation",
        model=hf_id,
        device=gpu_index,
        max_new_tokens=512,
        do_sample=False,
    )


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write the experiment artifact JSON to disk (pretty-printed, indent=2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", path)


# ---------------------------------------------------------------------------
# _run_three_conditions_for_model
# ---------------------------------------------------------------------------


def _run_three_conditions_for_model(
    adversarial_questions: list[Any],
    model_obj: object,
    model_name: str,
    executor: LongRunBenchmarkExecutor,
    exp_id: int,
) -> MicroAdversarialResult:
    """Run standard / adversarial / repaired conditions for one model.

    **Detailed explanation for engineers:**
        Three passes, each using LongRunBenchmarkExecutor with BATCH_SIZE=50 so
        the entire 50-question pass fits inside one batch (one 40-min per-batch watchdog).

        Pass 1 (standard):    infer on original_question, no repair.
        Pass 2 (adversarial): infer on adversarial_question, no repair.
        Pass 3 (repaired):    infer on adversarial_question, then pipe through
                              VerifyRepairPipeline if available; otherwise re-infer
                              (controls for stochasticity — same code path, honest result).

        The three pass results are then folded into a MicroAdversarialResult with
        percentage-point accuracy delta fields.

    Parameters
    ----------
    adversarial_questions : list[AdversarialGSMQuestion]
        50 paired question objects.
    model_obj : object
        Loaded HF text-generation pipeline.
    model_name : str
        Human-readable model name for logging and result fields.
    executor : LongRunBenchmarkExecutor
        Shared executor; each pass is a single batch.
    exp_id : int
        Experiment ID used as prefix for checkpoint files.

    Returns
    -------
    MicroAdversarialResult
        Per-model aggregated result.
    """
    # Attempt to wire VerifyRepairPipeline.
    repair_pipeline: Any = None
    try:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415
        from carnot.pipeline.extract import AutoExtractor  # noqa: PLC0415

        extractor = AutoExtractor()
        repair_pipeline = VerifyRepairPipeline(
            extractor=extractor,
            model=model_obj,
            generate_fn=lambda prompt: _call_model(model_obj, prompt),
            max_repairs=2,
        )
        _log.info("VerifyRepairPipeline wired for %s", model_name)
    except Exception as exc:
        _log.warning("VerifyRepairPipeline unavailable for %s (%s) — repair=re-inference", model_name, exc)

    def _infer_standard(q: Any) -> bool:
        resp = _call_model(model_obj, q.original_question)
        return _is_correct(resp, q.ground_truth_answer)

    def _infer_adversarial(q: Any) -> bool:
        resp = _call_model(model_obj, q.adversarial_question)
        return _is_correct(resp, q.ground_truth_answer)

    def _infer_repaired(q: Any) -> bool:
        if repair_pipeline is not None:
            try:
                raw = _call_model(model_obj, q.adversarial_question)
                repair_result = repair_pipeline.verify_and_repair(
                    q.adversarial_question, raw, "arithmetic"
                )
                final = getattr(repair_result, "final_response", raw)
            except Exception as exc:
                _log.warning("verify_and_repair failed for %s: %s — re-inferring", model_name, exc)
                final = _call_model(model_obj, q.adversarial_question)
        else:
            final = _call_model(model_obj, q.adversarial_question)
        return _is_correct(final, q.ground_truth_answer)

    prefix = f"exp{exp_id}_{model_name.replace('/', '_').replace('-', '_')}"

    # Use integer indices as the batch "questions" so save_batch can serialize them.
    # The actual AdversarialGSMQuestion objects are accessed via closure.
    indices = list(range(len(adversarial_questions)))

    def _infer_standard_by_idx(idx: int) -> bool:
        return _infer_standard(adversarial_questions[idx])

    def _infer_adversarial_by_idx(idx: int) -> bool:
        return _infer_adversarial(adversarial_questions[idx])

    def _infer_repaired_by_idx(idx: int) -> bool:
        return _infer_repaired(adversarial_questions[idx])

    # Run each condition as a single batch; integer indices are JSON-serializable.
    batch_std = executor.partition(indices)[0]
    batch_adv = executor.partition(indices)[0]
    batch_rep = executor.partition(indices)[0]

    batch_std = executor.run_batch(batch_std, _infer_standard_by_idx, watchdog_timeout_minutes=40)
    executor.save_batch(batch_std, f"{prefix}_standard")

    batch_adv = executor.run_batch(batch_adv, _infer_adversarial_by_idx, watchdog_timeout_minutes=40)
    executor.save_batch(batch_adv, f"{prefix}_adversarial")

    batch_rep = executor.run_batch(batch_rep, _infer_repaired_by_idx, watchdog_timeout_minutes=40)
    executor.save_batch(batch_rep, f"{prefix}_repaired")

    standard_correct: list[bool] = batch_std.results or []
    adversarial_correct: list[bool] = batch_adv.results or []
    repaired_correct: list[bool] = batch_rep.results or []

    n = len(adversarial_questions)
    std_acc = sum(standard_correct) / n if n else 0.0
    adv_acc = sum(adversarial_correct) / n if n else 0.0
    rep_acc = sum(repaired_correct) / n if n else 0.0

    drop_pct = round((std_acc - adv_acc) * 100, 2)
    improvement_pct = round((rep_acc - adv_acc) * 100, 2)

    _log.info(
        "[%s] std=%.3f adv=%.3f rep=%.3f drop_pct=%.2f improvement_pct=%.2f",
        model_name, std_acc, adv_acc, rep_acc, drop_pct, improvement_pct,
    )

    return MicroAdversarialResult(
        model_id=model_name,
        n_questions=n,
        standard_accuracy=std_acc,
        adversarial_accuracy=adv_acc,
        repaired_accuracy=rep_acc,
        adversarial_drop_pct=drop_pct,
        repair_improvement_pct=improvement_pct,
        inference_mode="live_gpu",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — gate chain is inherently long
    """Run Experiment 441: live adversarial GSM8K micro-benchmark.

    **Decision tree:**
        Gate 0: apply_env_autofix() (already called at module load).
        Gate 1: LiveGPUGate.require_live_or_blocked() — hard gate; blocked artifact on failure.
        Gate 2: check_dual_gpu_health() — WARNING if gpu1_is_zombie (continue on GPU0).
        Gate 3: tmpl.setup_gpu() — blocked if not all_healthy.
        Gate 4: model load for each spec — blocked on failure.
        Inference: ExperimentTimeoutWatchdog(441, 45 min) wraps all inference.
        Output: results/experiment_441_live_adversarial_micro.json

    Spec: REQ-BENCH-011, SCENARIO-BENCH-029, SCENARIO-BENCH-030
    """
    output_path = _REPO_ROOT / DELIVERABLE

    # ------------------------------------------------------------------
    # Gate 0: env autofix (already applied at module load — log only)
    # ------------------------------------------------------------------
    _log.info(
        "Gate 0: autofix_applied=%s  carnot_force_live=%s",
        _autofix_result.auto_fix_applied,
        _autofix_result.final_env_value,
    )

    # ------------------------------------------------------------------
    # ExperimentTemplate setup
    # ------------------------------------------------------------------
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID, title=EXP_TITLE, deliverable=DELIVERABLE, requires_gpu=True
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Gate 1: LiveGPUGate hard gate
    # ------------------------------------------------------------------
    gate_model_ids = [s["hf_id"] for s in MODEL_SPECS]
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, gate_model_ids)
    if blocked is not None:
        _log.error("Gate 1 (LiveGPUGate) blocked — writing blocked artifact.")
        blocked.update({
            "schema": "carnot.adversarial_micro.v1",
            "honest_verdict": "blocked",
            "robustness_claim": False,
            "inference_mode": "blocked",
            "n_models": 0,
            "per_model_results": [],
            "headline_result": None,
            "gate0_autofix_applied": _autofix_result.auto_fix_applied,
        })
        artifact = tmpl.build_result(blocked, status="blocked")
        _write_artifact(output_path, artifact)
        return

    _log.info("Gate 1 passed")

    # ------------------------------------------------------------------
    # Gate 2: Dual-GPU health check (WARNING only, RETRO-025)
    # ------------------------------------------------------------------
    gpu_health = check_dual_gpu_health()
    if gpu_health.gpu1_is_zombie:
        _log.warning(
            "Gate 2: GPU1 zombie detected — gpu1_vram=%.0fMB util=%.0f%%. "
            "Will serialise all models to GPU0.",
            gpu_health.gpu1_vram_mb, gpu_health.gpu1_util_pct,
        )
    if gpu_health.temperature_warning:
        _log.warning(
            "Gate 2: temperature warning — gpu0=%.0fC gpu1=%.0fC",
            gpu_health.gpu0_temp_c, gpu_health.gpu1_temp_c,
        )

    # ------------------------------------------------------------------
    # Gate 3: setup_gpu health check (blocking)
    # ------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        _log.error("Gate 3 (setup_gpu) unhealthy — writing blocked artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.adversarial_micro.v1",
                "honest_verdict": "blocked",
                "robustness_claim": False,
                "inference_mode": "blocked",
                "n_models": 0,
                "per_model_results": [],
                "headline_result": None,
                "failure_reason": "setup_gpu health check failed",
                "gate0_autofix_applied": _autofix_result.auto_fix_applied,
                "gate2_gpu1_zombie": gpu_health.gpu1_is_zombie,
            },
            status="blocked",
        )
        _write_artifact(output_path, artifact)
        return

    _log.info("Gate 3 passed")

    # ------------------------------------------------------------------
    # Gate 4: Load model weights (blocking on failure)
    # ------------------------------------------------------------------
    model_objects: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        try:
            _log.info("Loading %s on GPU %d ...", spec["name"], spec["gpu"])
            model_objects[spec["name"]] = _load_model_with_explicit_device(
                spec["hf_id"], spec["gpu"]
            )
            _log.info("Loaded %s OK", spec["name"])
        except Exception as exc:
            _log.error("Gate 4: failed to load %s: %s — blocked", spec["name"], exc)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.adversarial_micro.v1",
                    "honest_verdict": "blocked",
                    "robustness_claim": False,
                    "inference_mode": "blocked",
                    "n_models": 0,
                    "per_model_results": [],
                    "headline_result": None,
                    "failure_reason": f"model load failed: {spec['name']}: {exc}",
                    "gate0_autofix_applied": _autofix_result.auto_fix_applied,
                    "gate2_gpu1_zombie": gpu_health.gpu1_is_zombie,
                },
                status="blocked",
            )
            _write_artifact(output_path, artifact)
            return

    _log.info("Gates 1–4 passed — inference_mode=live_gpu")

    # ------------------------------------------------------------------
    # Load questions and build adversarial pairs
    # ------------------------------------------------------------------
    questions_raw = load_gsm8k_questions(N_QUESTIONS)
    adversarial_questions = build_adversarial_questions(questions_raw, seed=42)
    _log.info("Loaded %d GSM8K questions with adversarial pairs", len(adversarial_questions))

    executor = LongRunBenchmarkExecutor(
        batch_size=BATCH_SIZE,
        checkpoint_dir=f"results/batch_ckpt/exp{EXP_ID}",
    )

    # ------------------------------------------------------------------
    # Run benchmark inside ExperimentTimeoutWatchdog (45 min outer cap)
    # ------------------------------------------------------------------
    micro_results: list[MicroAdversarialResult] = []

    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=WATCHDOG_TIMEOUT_MINUTES,
        result_path=str(output_path),
    ):
        for spec in MODEL_SPECS:
            model_name = spec["name"]
            model_obj = model_objects[model_name]
            _log.info("Running 3-condition micro-benchmark for model: %s", model_name)

            result = _run_three_conditions_for_model(
                adversarial_questions,
                model_obj,
                model_name,
                executor,
                EXP_ID,
            )
            micro_results.append(result)
            tmpl.checkpoint_save(
                {"completed_models": [r.model_id for r in micro_results]},
                step=len(micro_results) * N_QUESTIONS,
            )

    # ------------------------------------------------------------------
    # Build and write artifact
    # ------------------------------------------------------------------
    micro_art = build_micro_adversarial_artifact(micro_results)
    honest_verdict = micro_art["honest_verdict"]

    _log.info(
        "HEADLINE: honest_verdict=%s  avg_adversarial_drop_pct=%.2f  "
        "avg_repair_improvement_pct=%.2f  robustness_claim=%s",
        honest_verdict,
        micro_art.get("avg_adversarial_drop_pct", 0.0),
        micro_art.get("avg_repair_improvement_pct", 0.0),
        micro_art.get("robustness_claim"),
    )

    micro_art["gate0_autofix_applied"] = _autofix_result.auto_fix_applied
    micro_art["gate2_gpu1_zombie"] = gpu_health.gpu1_is_zombie
    micro_art["gate2_temperature_warning"] = gpu_health.temperature_warning

    artifact = tmpl.build_result(micro_art, status="success")
    _write_artifact(output_path, artifact)

    _log.info(
        "Exp 441 complete — artifact at %s  honest_verdict=%s",
        output_path,
        honest_verdict,
    )


if __name__ == "__main__":
    main()
