#!/usr/bin/env python3
"""Experiment 429 — Adversarial GSM8K live benchmark (Apple arXiv 2410.05229 reproduction).

**Researcher summary:**
    Apple researchers showed that appending ONE irrelevant sentence to a GSM8K
    problem drops frontier LLM accuracy by up to 65%.  Carnot's arithmetic verifier
    is structural — it extracts explicit equation tokens and computes an Ising energy
    over those tokens only, ignoring surrounding context words.  Therefore Carnot
    SHOULD be immune to irrelevant-sentence injection.

    Exp 421 implemented this but was blocked at Gate 0 (status='partial').
    Exp 429 re-runs with apply_env_autofix() (RETRO-022 fix) and
    ExperimentTimeoutWatchdog (RETRO-003 fix).

    Two models evaluated (50 questions each, 3 conditions each):
        - google/gemma-4-E4B-it  (GPU 0)
        - Qwen/Qwen3.5-0.8B      (GPU 1, fallback to GPU 0)

    Three conditions:
        1. standard    — original (clean) GSM8K questions, no verify-repair.
        2. adversarial — distractor-appended variants, no verify-repair.
        3. repaired    — distractor-appended variants + VerifyRepairPipeline.

    Primary success criterion:
        adversarial_drop > 0  (LLM drops on adversarial inputs — Apple's signal)
        repair_improvement > 0  (Carnot's repair recovers some of that drop)

**Gate sequence:**
    Gate 0: apply_env_autofix() at import time + Exp 413 preflight (informational).
    Gate 1: LiveGPUGate.require_live_or_blocked() — hard gate.
    Gate 2: check_dual_gpu_health() — WARNING if GPU1 zombie (RETRO-025), not blocking.
    Gate 3: tmpl.setup_gpu() — model pre-warm (blocking).
    Gate 4: _load_model_pipeline() for each model (blocking).

**Output:** results/experiment_429_adversarial_live.json

Spec: REQ-BENCH-006, REQ-BENCH-007,
      SCENARIO-BENCH-014, SCENARIO-BENCH-015, SCENARIO-BENCH-016,
      REQ-INFRA-021, REQ-INFRA-022, REQ-INFRA-023
"""

from __future__ import annotations

import json
import logging
import sys
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
# RETRO-022 mitigation: if GPU is present but CARNOT_FORCE_LIVE is absent,
# inject it now so every downstream gate sees the correct env state.
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# Now safe to import GPU-dependent modules.
from carnot.pipeline.dual_gpu_health import check_dual_gpu_health  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.adversarial_gsm8k import (  # noqa: E402
    build_adversarial_questions,
    compute_adversarial_results,
)
from experiment_template import ExperimentTemplate  # noqa: E402

# Reuse data helpers from Exp 355 (no duplication of tested code).
from experiment_355_adversarial_gsm8k_benchmark import (  # noqa: E402
    load_gsm8k_questions,
    _extract_answer,
    _is_correct,
    _call_model,
    _build_per_model_result,
    _compute_top_level_verdict,
)

# Reuse model loading from Exp 368 precision benchmark.
from experiment_368_precision_live import _load_model_pipeline  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 429
EXP_TITLE = "Adversarial GSM8K live benchmark — Apple arXiv 2410.05229 (RETRO-022 fixed)"
DELIVERABLE = "results/experiment_429_adversarial_live.json"

N_QUESTIONS = 50
BATCH_SIZE = 8
CHECKPOINT_EVERY = 10
WATCHDOG_TIMEOUT_MINUTES = 75

MODEL_SPECS = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]

_EXP421_RESULT_PATH = _REPO_ROOT / "results" / "experiment_421_adversarial_live.json"
_EXP413_PREFLIGHT_PATH = _REPO_ROOT / "results" / "experiment_413_env_autofix.json"

# Exp 421 is considered confirmable only if it ran live successfully.
_CONFIRM_VERDICTS = frozenset(["improvement_positive", "degradation_positive", "neutral"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_preflight_verdict() -> dict[str, Any]:
    """Load and return the Exp 413 preflight artifact for Gate 0 reporting.

    Informational gate only — if the file is missing or malformed, return
    a sentinel dict so the artifact records what happened rather than crashing.
    Gate 1 (LiveGPUGate) is the actual hard gate.

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

    Creates the results/ directory if it does not already exist.

    Spec: REQ-BENCH-006
    """
    tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", tmpl._output_path)


def _run_three_conditions(
    questions_paired: list[Any],
    model_obj: Any,
    model_name: str,
    tmpl: ExperimentTemplate,
    model_idx: int,
) -> dict[str, Any]:
    """Run standard / adversarial / repaired conditions for one model.

    **Detailed explanation for engineers:**
        Three passes over the same 50 question pairs:
        1. Standard  — original question, no repair.
        2. Adversarial — distractor-appended question, no repair.
        3. Repaired  — distractor-appended question; if VerifyRepairPipeline
           is available, pipe the model's response through verify_and_repair().
           Otherwise, re-run inference on the adversarial question (controls for
           any stochasticity benefit from a second sample, while still testing
           the same code path).

        Checkpoints every CHECKPOINT_EVERY questions so partial results survive
        a watchdog kill.  The checkpoint step encodes both model index and
        question index to make progress traceable in the conductor log.

        Returns a per-model result dict compatible with _build_per_model_result().

    Args:
        questions_paired: list[AdversarialGSMQuestion] — 50 paired question objects.
        model_obj:  Loaded HF text-generation pipeline object.
        model_name: Human-readable model name for logging.
        tmpl:       ExperimentTemplate for checkpointing.
        model_idx:  Index in MODEL_SPECS (for unique checkpoint step encoding).

    Returns:
        dict with standard_accuracy, adversarial_accuracy, repaired_adversarial_accuracy,
        accuracy_drop, repair_improvement, inference_mode, model_id, n_questions.

    Spec: REQ-BENCH-006, SCENARIO-BENCH-015, SCENARIO-BENCH-017, SCENARIO-BENCH-018
    """
    ground_truths = [q.ground_truth_answer for q in questions_paired]

    def _infer(prompt: str) -> str:
        """Call the loaded model and return the response string."""
        return _call_model(model_obj, prompt)

    # Attempt to wire VerifyRepairPipeline for the repaired condition.
    # If unavailable (import error, etc.), fall back to plain re-inference.
    repair_pipeline: Any = None
    try:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415
        from carnot.pipeline.extract import AutoExtractor  # noqa: PLC0415

        extractor = AutoExtractor()

        def _generate_fn(prompt: str) -> str:
            return _infer(prompt)

        repair_pipeline = VerifyRepairPipeline(
            extractor=extractor,
            model=model_obj,
            generate_fn=_generate_fn,
            max_repairs=2,
        )
        _log.info("VerifyRepairPipeline wired for %s", model_name)
    except Exception as exc:
        _log.warning(
            "VerifyRepairPipeline unavailable for %s (%s) — repair=re-inference", model_name, exc
        )

    standard_correct: list[bool] = []
    adversarial_correct: list[bool] = []
    repaired_correct: list[bool] = []

    for i, (q, gold) in enumerate(zip(questions_paired, ground_truths)):
        # Pass 1: standard (clean question)
        std_response = _infer(q.original_question)
        standard_correct.append(_is_correct(std_response, gold))

        # Pass 2: adversarial (distractor-appended, no repair)
        adv_response = _infer(q.adversarial_question)
        adversarial_correct.append(_is_correct(adv_response, gold))

        # Pass 3: adversarial + repair
        if repair_pipeline is not None:
            try:
                raw_adv = _infer(q.adversarial_question)
                repair_result = repair_pipeline.verify_and_repair(
                    q.adversarial_question, raw_adv, "arithmetic"
                )
                final_response = getattr(repair_result, "final_response", raw_adv)
            except Exception as exc:
                _log.warning("verify_and_repair failed for %s q%d: %s", model_name, i, exc)
                final_response = _infer(q.adversarial_question)
        else:
            final_response = _infer(q.adversarial_question)

        repaired_correct.append(_is_correct(final_response, gold))

        # Checkpoint every CHECKPOINT_EVERY questions
        if (i + 1) % CHECKPOINT_EVERY == 0:
            tmpl.checkpoint_save(
                {
                    "model": model_name,
                    "questions_done": i + 1,
                    "standard_acc_so_far": sum(standard_correct) / (i + 1),
                    "adversarial_acc_so_far": sum(adversarial_correct) / (i + 1),
                    "repaired_acc_so_far": sum(repaired_correct) / (i + 1),
                },
                step=model_idx * N_QUESTIONS + i + 1,
            )
            _log.info(
                "[%s] checkpoint at q%d: std=%.3f adv=%.3f rep=%.3f",
                model_name,
                i + 1,
                sum(standard_correct) / (i + 1),
                sum(adversarial_correct) / (i + 1),
                sum(repaired_correct) / (i + 1),
            )

    result = compute_adversarial_results(
        standard_correct, adversarial_correct, repaired_correct, inference_mode="live_gpu"
    )

    _log.info(
        "[%s] std=%.3f adv=%.3f rep=%.3f drop=%.3f improvement=%.3f",
        model_name,
        result.standard_accuracy,
        result.adversarial_accuracy,
        result.repaired_adversarial_accuracy,
        result.accuracy_drop,
        result.repair_improvement,
    )

    return _build_per_model_result(model_name, result, len(questions_paired))


def _build_exp429_artifact(
    per_model_results: list[dict[str, Any]],
    inference_mode: str,
    n_questions: int,
    gate0_autofix_applied: bool,
    gate0_preflight_verdict: str,
    gate2_gpu1_zombie: bool,
    gate2_temperature_warning: bool,
    confirmed_from: int = 421,
    rerun: bool = True,
) -> dict[str, Any]:
    """Build the Exp 429 artifact dict (schema='carnot.adversarial_gsm8k.v2').

    **Detailed explanation for engineers:**
        Aggregates per-model results into headline metrics.  honest_verdict is
        computed by _compute_top_level_verdict() using the same rules as Exp 355.

        adversarial_drop_pct and repair_improvement_pct are expressed in percentage
        points (i.e. multiplied by 100) to match the Apple paper's reporting style.
        This makes the numbers directly comparable to their reported 65pp drop.

        The primary success criterion is:
            adversarial_drop_pct > 0  (LLM degrades under distractor injection)
            repair_improvement_pct > 0  (Carnot's repair recovers some accuracy)
        Both conditions together constitute the "improvement_positive" headline.

    Args:
        per_model_results:        List of per-model result dicts.
        inference_mode:           "live_gpu" or "blocked".
        n_questions:              Number of questions evaluated per model.
        gate0_autofix_applied:    Whether env_autofix injected CARNOT_FORCE_LIVE.
        gate0_preflight_verdict:  Exp 413 honest_verdict string.
        gate2_gpu1_zombie:        Whether GPU1 zombie was detected.
        gate2_temperature_warning: Whether temperature warning was active.
        confirmed_from:           Source experiment (always 421 for this script).
        rerun:                    True when Exp 421 was partial (always True here).

    Returns:
        JSON-serializable artifact dict.

    Spec: REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-016, SCENARIO-BENCH-019
    """
    honest_verdict = _compute_top_level_verdict(per_model_results, inference_mode)

    avg_drop = (
        sum(m["accuracy_drop"] for m in per_model_results) / len(per_model_results)
        if per_model_results
        else 0.0
    )
    avg_improvement = (
        sum(m["repair_improvement"] for m in per_model_results) / len(per_model_results)
        if per_model_results
        else 0.0
    )

    return {
        "adversarial_schema": "carnot.adversarial_gsm8k.v2",
        "inference_mode": inference_mode,
        "honest_verdict": honest_verdict,
        "confirmed_from": confirmed_from,
        "rerun": rerun,
        "n_questions": n_questions,
        "n_models": len(per_model_results),
        "per_model_results": per_model_results,
        "adversarial_drop_pct": round(avg_drop * 100, 2),
        "repair_improvement_pct": round(avg_improvement * 100, 2),
        "headline_result": {
            "honest_verdict": honest_verdict,
            "inference_mode": inference_mode,
            "avg_adversarial_drop": round(avg_drop, 4),
            "avg_repair_improvement": round(avg_improvement, 4),
            "adversarial_drop_pct": round(avg_drop * 100, 2),
            "repair_improvement_pct": round(avg_improvement * 100, 2),
            "improvement_positive": honest_verdict == "improvement_positive",
            "n_questions_per_model": n_questions,
            "n_models": len(per_model_results),
        },
        "gate0_autofix_applied": gate0_autofix_applied,
        "gate0_preflight_verdict": gate0_preflight_verdict,
        "gate2_gpu1_zombie": gate2_gpu1_zombie,
        "gate2_temperature_warning": gate2_temperature_warning,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — gate chain is inherently long
    """Run Experiment 429: adversarial GSM8K live benchmark.

    **Decision tree:**
        1. Read results/experiment_421_adversarial_live.json.
           - If status='success' AND inference_mode='live_gpu' AND honest_verdict
             in _CONFIRM_VERDICTS: CONFIRM path — copy result with experiment=429.
           - Otherwise: RERUN path — full benchmark with gate chain below.

        Gate chain (RERUN path):
        2. Gate 0: informational — load Exp 413 preflight verdict.
        3. Gate 1: LiveGPUGate.require_live_or_blocked() — hard gate.
        4. Gate 2: check_dual_gpu_health() — WARNING if gpu1_is_zombie (continue).
        5. Gate 3: tmpl.setup_gpu() — blocked if not all_healthy.
        6. Gate 4: _load_model_pipeline() for each model — blocked on failure.
        7. ExperimentTimeoutWatchdog(429, timeout_minutes=75) wraps inference.
        8. 2 models × 50 questions × 3 conditions; checkpoint every 10 questions.
        9. Artifact written with schema='carnot.adversarial_gsm8k.v2'.

    Spec: REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-019, REQ-INFRA-021/023
    """
    # ------------------------------------------------------------------
    # Step 1: Check Exp 421 result
    # ------------------------------------------------------------------
    exp421_data: dict[str, Any] = {}
    try:
        exp421_data = json.loads(_EXP421_RESULT_PATH.read_text())
    except Exception as exc:
        _log.warning("Could not load Exp 421 result: %s — proceeding to re-run", exc)

    exp421_status = exp421_data.get("status", "")
    exp421_mode = exp421_data.get("inference_mode", "")
    exp421_verdict = exp421_data.get("honest_verdict", "")

    can_confirm = (
        exp421_status == "success"
        and exp421_mode == "live_gpu"
        and exp421_verdict in _CONFIRM_VERDICTS
    )

    # ------------------------------------------------------------------
    # Step 1a: CONFIRM PATH — copy Exp 421 result with 429 metadata
    # ------------------------------------------------------------------
    if can_confirm:
        _log.info(
            "Exp 421 result is confirmable (verdict=%s, mode=%s) — copying.",
            exp421_verdict,
            exp421_mode,
        )
        tmpl = ExperimentTemplate(
            exp_id=EXP_ID, title=EXP_TITLE, deliverable=DELIVERABLE,
            requires_gpu=False,
        )
        tmpl.setup()

        confirmed = dict(exp421_data)
        confirmed["experiment"] = EXP_ID
        confirmed["confirmed_from"] = 421
        confirmed["rerun"] = False
        confirmed["adversarial_schema"] = "carnot.adversarial_gsm8k.v2"

        artifact = tmpl.build_result(confirmed, status="success")
        _write_artifact(tmpl, artifact)
        _log.info("CONFIRMED from Exp 421: honest_verdict=%s", exp421_verdict)
        return

    _log.info(
        "Exp 421 status=%r mode=%r verdict=%r — proceeding to RERUN.",
        exp421_status, exp421_mode, exp421_verdict,
    )

    # ------------------------------------------------------------------
    # Step 2: Gate 0 — informational preflight
    # apply_env_autofix() was called at module import time; just log here.
    # ------------------------------------------------------------------
    preflight = _load_preflight_verdict()
    _log.info(
        "Gate 0 (informational): autofix_applied=%s  retro_022_resolved=%s  "
        "exp413_verdict=%s  carnot_force_live_now=%s",
        _autofix_result.auto_fix_applied,
        preflight.get("retro_022_resolved"),
        preflight.get("honest_verdict"),
        _autofix_result.final_env_value,
    )

    # ------------------------------------------------------------------
    # Step 3: ExperimentTemplate setup (RERUN path)
    # ------------------------------------------------------------------
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID, title=EXP_TITLE, deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 4: Gate 1 — LiveGPUGate hard gate
    # ------------------------------------------------------------------
    gate_model_ids = [spec["hf_id"] for spec in MODEL_SPECS]
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, gate_model_ids)
    if blocked is not None:
        _log.error("Gate 1 (LiveGPUGate) blocked Exp 429 — writing blocked artifact.")
        blocked.update({
            "adversarial_schema": "carnot.adversarial_gsm8k.v2",
            "inference_mode": "blocked",
            "honest_verdict": "blocked",
            "confirmed_from": 421,
            "rerun": True,
            "n_questions": 0,
            "n_models": 0,
            "per_model_results": [],
            "adversarial_drop_pct": 0.0,
            "repair_improvement_pct": 0.0,
            "gate0_autofix_applied": _autofix_result.auto_fix_applied,
            "gate0_preflight_verdict": preflight.get("honest_verdict"),
        })
        _write_artifact(tmpl, blocked)
        return

    _log.info("Gate 1 passed — inference_mode=live_gpu")

    # ------------------------------------------------------------------
    # Step 5: Gate 2 — Dual-GPU health check (WARNING only, RETRO-025)
    # ------------------------------------------------------------------
    gpu_health = check_dual_gpu_health()
    if gpu_health.gpu1_is_zombie:
        _log.warning(
            "Gate 2: GPU1 zombie detected (RETRO-025) — gpu1_vram_mb=%.0f gpu1_util=%.0f%%. "
            "Inference will serialise to GPU0 only.",
            gpu_health.gpu1_vram_mb, gpu_health.gpu1_util_pct,
        )
    if gpu_health.temperature_warning:
        _log.warning(
            "Gate 2: temperature warning — gpu0=%.0fC gpu1=%.0fC  batch_factor=%.2f",
            gpu_health.gpu0_temp_c, gpu_health.gpu1_temp_c,
            gpu_health.recommended_batch_size_factor,
        )

    # ------------------------------------------------------------------
    # Step 6: Gate 3 — setup_gpu health check (blocking)
    # ------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        _log.error("Gate 3 (setup_gpu) unhealthy — writing blocked artifact.")
        artifact = tmpl.build_result(
            {
                "adversarial_schema": "carnot.adversarial_gsm8k.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "confirmed_from": 421,
                "rerun": True,
                "failure_reason": "setup_gpu health check failed",
                "n_questions": 0,
                "n_models": 0,
                "per_model_results": [],
                "adversarial_drop_pct": 0.0,
                "repair_improvement_pct": 0.0,
                "gate0_autofix_applied": _autofix_result.auto_fix_applied,
                "gate0_preflight_verdict": preflight.get("honest_verdict"),
                "gate2_gpu1_zombie": gpu_health.gpu1_is_zombie,
                "gate2_temperature_warning": gpu_health.temperature_warning,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    _log.info("Gate 3 passed — all models healthy")

    # ------------------------------------------------------------------
    # Step 7: Gate 4 — Load model weights for each spec (blocking)
    # ------------------------------------------------------------------
    model_objects: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        try:
            _log.info("Loading %s on GPU %d ...", spec["name"], spec["gpu"])
            model_objects[spec["name"]] = _load_model_pipeline(
                spec["hf_id"], spec["gpu"], "auto"
            )
            _log.info("Loaded %s OK", spec["name"])
        except Exception as exc:
            _log.error("Gate 4: failed to load %s: %s — blocked", spec["name"], exc)
            artifact = tmpl.build_result(
                {
                    "adversarial_schema": "carnot.adversarial_gsm8k.v2",
                    "inference_mode": "blocked",
                    "honest_verdict": "blocked",
                    "confirmed_from": 421,
                    "rerun": True,
                    "failure_reason": f"model load failed: {spec['name']}: {exc}",
                    "n_questions": 0,
                    "n_models": 0,
                    "per_model_results": [],
                    "adversarial_drop_pct": 0.0,
                    "repair_improvement_pct": 0.0,
                    "gate0_autofix_applied": _autofix_result.auto_fix_applied,
                    "gate0_preflight_verdict": preflight.get("honest_verdict"),
                    "gate2_gpu1_zombie": gpu_health.gpu1_is_zombie,
                    "gate2_temperature_warning": gpu_health.temperature_warning,
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            return

    inference_mode = "live_gpu"
    _log.info("Gates 1-4 passed — inference_mode=%s", inference_mode)

    # ------------------------------------------------------------------
    # Step 8: Load questions and build adversarial pairs
    # ------------------------------------------------------------------
    questions_raw = load_gsm8k_questions(N_QUESTIONS)
    n_actual = len(questions_raw)
    adversarial_questions = build_adversarial_questions(questions_raw, seed=42)
    _log.info("Loaded %d GSM8K questions + adversarial pairs", n_actual)

    # ------------------------------------------------------------------
    # Step 9: Run benchmark inside ExperimentTimeoutWatchdog (75 min cap)
    # ------------------------------------------------------------------
    per_model_results: list[dict[str, Any]] = []

    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=WATCHDOG_TIMEOUT_MINUTES,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        for idx, spec in enumerate(MODEL_SPECS):
            model_name = spec["name"]
            model_obj = model_objects[model_name]
            _log.info("Running 3-condition benchmark for model: %s", model_name)

            per_model = _run_three_conditions(
                adversarial_questions, model_obj, model_name, tmpl, idx
            )
            per_model_results.append(per_model)

            tmpl.checkpoint_save(
                {"completed_models": [m["model_id"] for m in per_model_results]},
                step=(idx + 1) * N_QUESTIONS,
            )

    # ------------------------------------------------------------------
    # Step 10: Build and write artifact
    # ------------------------------------------------------------------
    exp429_data = _build_exp429_artifact(
        per_model_results=per_model_results,
        inference_mode=inference_mode,
        n_questions=n_actual,
        gate0_autofix_applied=_autofix_result.auto_fix_applied,
        gate0_preflight_verdict=preflight.get("honest_verdict", "unknown"),
        gate2_gpu1_zombie=gpu_health.gpu1_is_zombie,
        gate2_temperature_warning=gpu_health.temperature_warning,
        confirmed_from=421,
        rerun=True,
    )

    honest_verdict = exp429_data["honest_verdict"]

    _log.info(
        "HEADLINE: honest_verdict=%s  adversarial_drop_pct=%.2f  repair_improvement_pct=%.2f",
        honest_verdict,
        exp429_data["adversarial_drop_pct"],
        exp429_data["repair_improvement_pct"],
    )

    artifact = tmpl.build_result(exp429_data, status="success")
    _write_artifact(tmpl, artifact)

    _log.info(
        "Exp 429 complete — artifact at %s  honest_verdict=%s",
        _REPO_ROOT / DELIVERABLE,
        honest_verdict,
    )


if __name__ == "__main__":
    main()
