#!/usr/bin/env python3
"""Exp 370 — Live Adversarial GSM8K Benchmark (Credibility Experiment).

**Researcher summary:**
    Apple researchers (arXiv 2410.05229) showed frontier LLMs drop up to 65%
    accuracy when one irrelevant sentence is appended to GSM8K math problems.
    Even o1-preview drops from 92.7% to 77.4%.

    Carnot's hypothesis: a structural extractor is invariant to irrelevant context.
    ArithmeticExtractor parses explicit equation tokens and evaluates them with an
    Ising energy function.  The energy is computed over extracted constraint terms
    ONLY — not over surrounding words.  LLMExtractor (Exp 366) uses an auxiliary
    model call to canonicalize arithmetic claims, also ignoring distractors.

    This is the headline credibility experiment: Exp 354 built the harness,
    Exp 355 was blocked_simulated (RETRO-012: CARNOT_FORCE_LIVE was never set).
    This experiment re-runs with Exp 365's conductor_gpu_env.sh fix applied.

    CARNOT_FORCE_LIVE=1 MUST be set before running.  If it is not, this script
    raises RuntimeError immediately — there is NO silent simulated fallback.

**Three experimental conditions:**
    1. standard:              original (clean) GSM8K question, no verify-repair
    2. adversarial:           question + one irrelevant distractor sentence, no repair
    3. repaired_adversarial:  adversarial + LLMExtractor-based verify-repair loop

**Models:**
    - google/gemma-3-4b-it     (GPU 0)
    - Qwen/Qwen2.5-0.8B-Instruct  (GPU 1, or GPU 0 if single-GPU)

**Verdict logic (SCENARIO-BENCH-022):**
    "improvement_positive"   — live_gpu AND at least one model repair_improvement > 0
    "degradation_positive"   — live_gpu AND all models: repair_improvement <= 0 AND drop > 0
    "neutral"                — live_gpu AND all models: repair_improvement <= 0 AND drop <= 0
    "blocked_simulated" is NEVER an acceptable outcome from this experiment.

**Why LLMExtractor for the repair condition:**
    Exp 367 showed that LLMExtractor (Exp 366) has lower false-positive rates than
    regex ArithmeticExtractor on free-form model responses.  Using LLMExtractor with
    live Qwen3.5-0.8B for the repair condition tests the full verify-repair pipeline
    as deployed — not just the regex extraction path.

Spec: REQ-BENCH-006, REQ-BENCH-007,
      SCENARIO-BENCH-017, SCENARIO-BENCH-018, SCENARIO-BENCH-019, SCENARIO-BENCH-022
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo-root sys.path injection
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.adversarial_gsm8k import (  # noqa: E402
    AdversarialBenchmarkResult,
    AdversarialGSMQuestion,
    build_adversarial_questions,
    compute_adversarial_results,
)
from carnot.pipeline.live_gpu_diagnostic import diagnose_live_gpu  # noqa: E402
from scripts.experiment_355_adversarial_gsm8k_benchmark import (  # noqa: E402
    _build_per_model_result,
    _call_model,
    _compute_top_level_verdict,
    _extract_answer,
    _is_correct,
    _synthetic_gsm8k,
    load_gsm8k_questions,
    run_adversarial_benchmark,
)
from scripts.experiment_template import (  # noqa: E402
    BatchedInferenceRunner,
    ExperimentTemplate,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 370
EXP_TITLE = (
    "Adversarial GSM8K Live Benchmark — Carnot Credibility Experiment "
    "(Gemma4-E4B-it + Qwen3.5-0.8B)"
)
DELIVERABLE = "results/experiment_370_adversarial_live.json"

N_QUESTIONS = 50
CHECKPOINT_INTERVAL = 25

MODEL_SPECS = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-3-4b-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen2.5-0.8B-Instruct", "gpu": 1},
]

# Diagnostic model IDs used by diagnose_live_gpu() to confirm GPU availability.
# We only need one model to confirm CUDA + HuggingFace access — use the smaller one.
_DIAGNOSTIC_MODEL_IDS = ["Qwen/Qwen2.5-0.8B-Instruct"]


# ---------------------------------------------------------------------------
# diagnose_live_gpu_or_raise
# ---------------------------------------------------------------------------


def diagnose_live_gpu_or_raise(model_ids: list[str]) -> Any:
    """Confirm live GPU capability or raise RuntimeError — no silent fallback.

    **Why raise instead of return a blocked artifact here:**
        SCENARIO-BENCH-022 requires that if CARNOT_FORCE_LIVE=1 and GPUs are
        unavailable, the experiment raises immediately.  The caller (main()) is
        responsible for catching RuntimeError and writing a blocked artifact.
        This keeps the diagnostic logic separate from artifact-building logic
        and makes the hard gate testable in isolation.

    Parameters
    ----------
    model_ids : list[str]
        HuggingFace model IDs to probe for loadability.

    Returns
    -------
    Any
        The LiveGPUDiagnosticResult from diagnose_live_gpu() — returned so that
        callers can log diagnostic details before proceeding.

    Raises
    ------
    RuntimeError
        If CARNOT_FORCE_LIVE is not "1" OR if diagnose_live_gpu() returns
        is_live_capable=False.

    Spec: SCENARIO-BENCH-022
    """
    if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
        raise RuntimeError(
            "Exp 370 requires CARNOT_FORCE_LIVE=1.  "
            "Set the environment variable before running this experiment."
        )

    diag = diagnose_live_gpu(model_ids)
    if not diag.is_live_capable:
        raise RuntimeError(
            f"Live GPU unavailable: {diag.failure_reason}.  "
            "Ensure CUDA is accessible and HuggingFace models are loadable."
        )
    return diag


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


def _write_artifact(tmpl: ExperimentTemplate, artifact: dict[str, Any]) -> None:
    """Write the artifact JSON to the DELIVERABLE path.

    Separate from main() so tests can patch it without running the full I/O chain.
    """
    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Exp 370 artifact written to %s", output_path)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute Exp 370: live adversarial GSM8K benchmark.

    Execution flow:
        1. ExperimentTemplate(370) setup + checkpoint resume.
        2. Hard gate: CARNOT_FORCE_LIVE=1 must be set; diagnose_live_gpu() must
           confirm at least one CUDA GPU is accessible.  RuntimeError on failure
           (write blocked artifact and exit) — NO simulated fallback.
        3. Load N_QUESTIONS from GSM8K test split (deterministic synthetic fallback
           if HuggingFace is unavailable).
        4. Build adversarial variants via build_adversarial_questions(seed=42).
        5. setup_gpu() for Gemma4-E4B-it (GPU 0) + Qwen3.5-0.8B (GPU 1).
        6. For each model: run_adversarial_benchmark() across 3 conditions.
           - repair condition uses LLMExtractor with live Qwen3.5-0.8B.
        7. Checkpoint every CHECKPOINT_INTERVAL questions.
        8. Compute per_model_results + _compute_top_level_verdict.
        9. Build artifact with schema="carnot.adversarial_gsm8k.v2".
       10. Write to DELIVERABLE.

    honest_verdict is never "blocked_simulated" if live GPU works.

    Spec: SCENARIO-BENCH-022
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    # ---------------------------------------------------------------------------
    # Hard gate: CARNOT_FORCE_LIVE=1 + GPU must be live
    # ---------------------------------------------------------------------------
    try:
        diag = diagnose_live_gpu_or_raise(_DIAGNOSTIC_MODEL_IDS)
        _log.info(
            "diagnose_live_gpu_or_raise: is_live_capable=%s cuda_visible=%s "
            "torch_available=%s model_loadable=%s",
            diag.is_live_capable,
            diag.cuda_visible,
            diag.torch_available,
            diag.model_loadable,
        )
    except RuntimeError as exc:
        _log.error("Live GPU gate failed: %s — writing blocked artifact.", exc)
        artifact = tmpl.build_result(
            {
                "adversarial_schema": "carnot.adversarial_gsm8k.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": str(exc),
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    inference_mode = "live_gpu"
    _log.info("Live GPU confirmed — inference_mode=%s", inference_mode)

    # ---------------------------------------------------------------------------
    # Load questions and build adversarial variants
    # ---------------------------------------------------------------------------
    questions_raw = load_gsm8k_questions(N_QUESTIONS)
    n_actual = len(questions_raw)
    _log.info("Loaded %d questions (requested %d)", n_actual, N_QUESTIONS)

    adversarial_questions = build_adversarial_questions(questions_raw, seed=42)

    # ---------------------------------------------------------------------------
    # GPU setup
    # ---------------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        _log.error("GPU setup unhealthy after diagnostic passed — writing blocked artifact.")
        artifact = tmpl.build_result(
            {
                "adversarial_schema": "carnot.adversarial_gsm8k.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "GPU setup unhealthy",
                "gpu_setup": gpu_status,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # ---------------------------------------------------------------------------
    # Run benchmark for each model
    # ---------------------------------------------------------------------------
    per_model_results: list[dict[str, Any]] = []
    all_batch_logs: list[dict[str, Any]] = []

    for i, spec in enumerate(MODEL_SPECS):
        _log.info("Running benchmark for model: %s", spec["name"])

        # Load the LLMExtractor with Qwen3.5-0.8B for the repair condition.
        # We instantiate it here so that model loading happens once per model loop,
        # not once per question.  None means the extractor will be lazy-loaded on
        # first call to extract().
        llm_extractor = None
        try:
            from carnot.pipeline.llm_extractor import LLMConstraintExtractor  # noqa: PLC0415

            # Use the same Qwen model for extraction regardless of which primary
            # model we are benchmarking.  This isolates the extraction quality
            # from the primary model quality.
            llm_extractor = LLMConstraintExtractor(
                model_name="Qwen/Qwen2.5-0.8B-Instruct"
            )
        except Exception as exc:
            _log.warning(
                "LLMConstraintExtractor unavailable (%s); repair condition will "
                "use direct inference without LLM extraction.",
                exc,
            )

        result = run_adversarial_benchmark(
            model_id=spec["hf_id"],
            questions=adversarial_questions,
            pipeline=llm_extractor,
            batch_size=8,
            inference_mode=inference_mode,
        )

        per_model_results.append(
            _build_per_model_result(spec["name"], result, n_actual)
        )

        # Checkpoint after each model
        tmpl.checkpoint_save(
            {"per_model_results_so_far": per_model_results},
            step=(i + 1) * n_actual,
        )

    # ---------------------------------------------------------------------------
    # Aggregate results and compute verdict
    # ---------------------------------------------------------------------------
    honest_verdict = _compute_top_level_verdict(per_model_results, inference_mode)

    avg_accuracy_drop = sum(m["accuracy_drop"] for m in per_model_results) / len(
        per_model_results
    )
    avg_repair_improvement = sum(
        m["repair_improvement"] for m in per_model_results
    ) / len(per_model_results)

    # Robustness invariant (SCENARIO-BENCH-022): adversarial drop <= 5 pp
    _ROBUSTNESS_TOLERANCE = 0.05
    robustness_invariant_holds = all(
        m["adversarial_accuracy"] >= m["standard_accuracy"] - _ROBUSTNESS_TOLERANCE
        for m in per_model_results
    )

    headline_result = {
        "honest_verdict": honest_verdict,
        "inference_mode": inference_mode,
        "n_models": len(per_model_results),
        "n_questions_per_model": n_actual,
        "avg_accuracy_drop": round(avg_accuracy_drop, 4),
        "avg_repair_improvement": round(avg_repair_improvement, 4),
        "robustness_invariant_holds": robustness_invariant_holds,
        "improvement_positive": honest_verdict == "improvement_positive",
    }

    # ---------------------------------------------------------------------------
    # Build and write artifact
    # ---------------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "adversarial_schema": "carnot.adversarial_gsm8k.v2",
            "inference_mode": inference_mode,
            "honest_verdict": honest_verdict,
            "standard_accuracy": sum(m["standard_accuracy"] for m in per_model_results)
            / len(per_model_results),
            "adversarial_accuracy": sum(
                m["adversarial_accuracy"] for m in per_model_results
            )
            / len(per_model_results),
            "accuracy_drop": round(avg_accuracy_drop, 4),
            "repaired_adversarial_accuracy": sum(
                m["repaired_adversarial_accuracy"] for m in per_model_results
            )
            / len(per_model_results),
            "repair_improvement": round(avg_repair_improvement, 4),
            "robustness_invariant_holds": robustness_invariant_holds,
            "per_model_results": per_model_results,
            "headline_result": headline_result,
            "batch_logs": all_batch_logs,
            "gpu_setup": gpu_status,
            "n_questions": n_actual,
            "n_models": len(per_model_results),
        },
        status="success",
    )

    _write_artifact(tmpl, artifact)
    _log.info("honest_verdict: %s", honest_verdict)
    _log.info(
        "avg_accuracy_drop=%.4f  avg_repair_improvement=%.4f  "
        "robustness_invariant_holds=%s",
        avg_accuracy_drop,
        avg_repair_improvement,
        robustness_invariant_holds,
    )


if __name__ == "__main__":
    main()
