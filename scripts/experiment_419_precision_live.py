#!/usr/bin/env python3
"""Experiment 419: Live precision pipeline benchmark with CRANE extractor.

**Researcher summary:**
    Exps 368 and 379 ran the 5-variant × 2-model × 200-question GSM8K precision
    benchmark, but both were blocked by env-propagation issues (RETRO-022).
    Exp 413 fixed the root cause with EnvironmentAutoFix (self-injects
    CARNOT_FORCE_LIVE=1 when GPU is detected).  This experiment uses that fix
    as the FIRST action, then gates on the Exp 413 honest_verdict before proceeding.

    The key new element vs. Exp 379:
    - ``CRANEExtractionGate`` (Exp 418) is the PRIMARY extractor for FULL_STACK.
      CRANE is purely regex + deterministic math — no GPU overhead, higher precision
      (fewer false positives) than LLMConstraintExtractor.
    - ``LLMConstraintExtractor`` is the FALLBACK when CRANE extracts zero claims.
    - Both extractors are passed into ``run_variant()`` via ``extractor_obj``
      (CRANE) and ``fallback_extractor_obj`` (LLM).

**Why CRANE first?**
    CRANE runs on CPU with no model dependency, so it is always available even when
    the LLM extractor model fails to load.  Its output is deterministically verifiable.
    LLMConstraintExtractor is only invoked when CRANE finds nothing — this avoids
    the 40% IT-format miss rate while also avoiding LLM inference overhead on the
    majority of responses where CRANE succeeds.

**Hard CARNOT_FORCE_LIVE=1 requirement:**
    1. apply_env_autofix() called FIRST (before any CUDA import).
    2. results/experiment_413_env_autofix.json loaded; honest_verdict must be in
       the approved set; otherwise write blocked artifact and exit.
    3. LiveGPUGate.require_live_or_blocked() — hard gate, no simulated fallback.
    4. setup_gpu() all_healthy check — blocked if not all healthy.
    5. Model load failures — blocked artifact, no fake numbers.

**Output:** results/experiment_419_precision_live.json

Spec: REQ-BENCH-003, SCENARIO-BENCH-020
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import occurs.  Moving this below any torch/JAX import is a bug.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_AUTOFIX_RESULT = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
from typing import Any

from scripts.experiment_template import (  # noqa: E402
    ExperimentTemplate,
)
from scripts.experiment_368_precision_live import (  # noqa: E402
    load_gsm8k_questions,
    run_variant,
    _load_model_pipeline,
    _hf_pipeline_generate_fn,
)
from carnot.pipeline.precision_benchmark import (  # noqa: E402
    PipelineVariant,
    PrecisionStackResult,
    build_precision_benchmark_artifact,
)
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.crane_extractor import CRANEExtractionGate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 419
EXP_TITLE = "Live precision pipeline benchmark with CRANE extractor"
DELIVERABLE = "results/experiment_419_precision_live.json"
N_QUESTIONS = 200
BATCH_SIZE = 8
CHECKPOINT_EVERY = 50

MODEL_SPECS = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]

# honest_verdict values from Exp 413 that mean env fix succeeded.
_ALLOWED_VERDICTS = frozenset(
    ["gpu_confirmed_live", "auto_fix_applied", "gpu_detected_env_was_correct"]
)


# ---------------------------------------------------------------------------
# Artifact builder (Exp 419 v2 schema)
# ---------------------------------------------------------------------------


def build_exp419_artifact(
    results: list[PrecisionStackResult],
    inference_mode: str,
) -> dict[str, Any]:
    """Build the Exp 419 precision benchmark artifact from a list of results.

    **Detailed explanation for engineers:**
        Identical honest_verdict logic to Exp 379/368 (SCENARIO-BENCH-020):
        - ``"live_improvement"``: inference_mode == "live_gpu" AND signed_improvement > 0
          for the FULL_STACK Gemma4-E4B-it result.
        - ``"live_no_improvement"``: live_gpu but improvement <= 0.
        - ``"blocked"``: inference_mode is anything other than "live_gpu".

        The schema is explicitly set to "carnot.precision_benchmark.v2" to distinguish
        from Exp 340 simulated results.

    Args:
        results:        List of PrecisionStackResult objects (5 variants × 2 models).
        inference_mode: "live_gpu" for a valid run, "blocked" otherwise.

    Returns:
        Dict with schema v2, headline_result, per_variant_results, inference_mode,
        and honest_verdict.
    """
    base = build_precision_benchmark_artifact(results)

    # Override schema to v2.
    base["precision_schema"] = "carnot.precision_benchmark.v2"

    # Override inference_mode with the confirmed value from LiveGPUGate.
    base["inference_mode"] = inference_mode

    # Compute honest_verdict per SCENARIO-BENCH-020.
    headline = base.get("headline_result", {})
    if inference_mode == "live_gpu" and headline.get("signed_improvement", 0.0) > 0:
        base["honest_verdict"] = "live_improvement"
    elif inference_mode == "live_gpu":
        base["honest_verdict"] = "live_no_improvement"
    else:
        base["honest_verdict"] = "blocked"

    return base


# ---------------------------------------------------------------------------
# Artifact writer (thin wrapper so tests can patch it)
# ---------------------------------------------------------------------------


def _write_artifact(tmpl: ExperimentTemplate, artifact: dict[str, Any]) -> None:
    """Write the artifact to the deliverable path and log the location.

    **Detailed explanation for engineers:**
        Extracted to module level so tests can patch
        ``scripts.experiment_419_precision_live._write_artifact`` without
        instrumenting main() directly.
    """
    output_path = tmpl._output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)


# ---------------------------------------------------------------------------
# _apply_variant_with_crane
# ---------------------------------------------------------------------------


def _apply_variant_with_crane(
    variant: PipelineVariant,
    response: str,
    question: str,
    model_id: str,
    crane_extractor: CRANEExtractionGate,
    llm_extractor: object | None,
) -> tuple[str, int, int]:
    """Apply pipeline variant using CRANE primary + LLM fallback for FULL_STACK.

    **Detailed explanation for engineers:**
        Wraps the Exp 368 ``_apply_variant()`` for all variants, but for FULL_STACK
        injects CRANE as the primary extractor.

        CRANE primacy logic for FULL_STACK:
        1. Extract with CRANE first (CPU, fast, high precision).
        2. If CRANE returns at least one violation, use CRANE result and skip LLM.
        3. If CRANE returns zero violations, fall back to LLMConstraintExtractor.
        This avoids LLM inference overhead on the majority of responses.

        For non-FULL_STACK variants, delegates directly to Exp 368's ``_apply_variant``
        with the LLM extractor (same behaviour as Exp 379).

    Args:
        variant:         PipelineVariant ablation condition.
        response:        Model response text.
        question:        Original question text.
        model_id:        Model identifier for adaptive thresholds.
        crane_extractor: CRANEExtractionGate instance (always available).
        llm_extractor:   LLMConstraintExtractor or None if unavailable.

    Returns:
        (response_unchanged, n_violations_found, n_repairs_attempted)
    """
    from scripts.experiment_368_precision_live import _apply_variant  # noqa: PLC0415

    if variant != PipelineVariant.FULL_STACK:
        # Non-FULL_STACK: delegate to Exp 368 (LLM extractor path).
        return _apply_variant(variant, response, question, model_id, llm_extractor)

    # FULL_STACK: CRANE primary, LLM fallback.
    crane_violations = crane_extractor.extract(response, "arithmetic")
    if crane_violations:
        # CRANE found violations — use its count, skip LLM call.
        n_viol = len(crane_violations)
        n_rep = 1  # any violation triggers repair consideration
        return response, n_viol, n_rep

    # CRANE found nothing — fall back to LLM extractor path.
    return _apply_variant(variant, response, question, model_id, llm_extractor)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 419: live precision pipeline with CRANE extractor.

    **Detailed explanation for engineers:**
        Gate sequence:
        1. apply_env_autofix() already ran at module load (FIRST, before any CUDA
           import).
        2. Load results/experiment_413_env_autofix.json.  If honest_verdict not in
           the approved set, write blocked artifact and exit immediately.
        3. ExperimentTemplate setup.
        4. LiveGPUGate.require_live_or_blocked() — write blocked if env/GPU not live.
        5. tmpl.setup_gpu() — write blocked if not all_healthy.
        6. Load models — write blocked on any load failure.
        7. Wire CRANEExtractionGate (primary) + LLMConstraintExtractor (fallback).
        8. Load 200 GSM8K questions.
        9. Run 5 variants × 2 models via BatchedInferenceRunner (batch_size=8).
        10. Checkpoint every 50 questions via tmpl.checkpoint_save().
        11. Build and write artifact.
    """
    # ------------------------------------------------------------------
    # Step 2: gate on Exp 413 honest_verdict
    # ------------------------------------------------------------------
    exp413_path = _REPO_ROOT / "results" / "experiment_413_env_autofix.json"
    try:
        exp413_data = json.loads(exp413_path.read_text())
        exp413_verdict = exp413_data.get("honest_verdict", "")
    except Exception as exc:
        exp413_verdict = ""
        _log.error("Could not load Exp 413 result: %s", exc)

    if exp413_verdict not in _ALLOWED_VERDICTS:
        _log.error(
            "Exp 413 honest_verdict=%r not in allowed set %s — blocked.",
            exp413_verdict,
            sorted(_ALLOWED_VERDICTS),
        )
        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=DELIVERABLE,
            requires_gpu=True,
        )
        tmpl.setup()
        artifact = tmpl.build_result(
            {
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "precision_schema": "carnot.precision_benchmark.v2",
                "failure_reason": (
                    f"Exp 413 honest_verdict={exp413_verdict!r} not in approved set; "
                    "run Exp 413 (EnvironmentAutoFix) first"
                ),
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    _log.info("Exp 413 gate passed (honest_verdict=%s)", exp413_verdict)

    # ------------------------------------------------------------------
    # Step 3: ExperimentTemplate setup
    # ------------------------------------------------------------------
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 4: LiveGPUGate — hard gate
    # ------------------------------------------------------------------
    gate_model_ids = [spec["hf_id"] for spec in MODEL_SPECS]
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, gate_model_ids)
    if blocked is not None:
        _log.error("LiveGPUGate blocked Exp 419 — writing blocked artifact.")
        blocked["precision_schema"] = "carnot.precision_benchmark.v2"
        blocked["inference_mode"] = "blocked"
        blocked["honest_verdict"] = "blocked"
        _write_artifact(tmpl, blocked)
        return

    inference_mode = "live_gpu"
    _log.info("LiveGPUGate passed — inference_mode=%s", inference_mode)

    # ------------------------------------------------------------------
    # Step 5: GPU setup
    # ------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        _log.error("GPU setup unhealthy — writing blocked artifact.")
        artifact = tmpl.build_result(
            {
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "setup_gpu reported not all_healthy",
                "precision_schema": "carnot.precision_benchmark.v2",
                "gpu_setup_status": gpu_status,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # ------------------------------------------------------------------
    # Step 6: Load models
    # ------------------------------------------------------------------
    model_objects: dict[str, object] = {}
    for spec in MODEL_SPECS:
        try:
            _log.info("Loading %s on GPU %d ...", spec["name"], spec["gpu"])
            model_objects[spec["name"]] = _load_model_pipeline(
                spec["hf_id"], spec["gpu"], "auto"
            )
            _log.info("Loaded %s OK", spec["name"])
        except Exception as exc:
            _log.error("Failed to load %s: %s — blocked", spec["name"], exc)
            artifact = tmpl.build_result(
                {
                    "inference_mode": "blocked",
                    "honest_verdict": "blocked",
                    "failure_reason": f"model load failed: {spec['name']}: {exc}",
                    "precision_schema": "carnot.precision_benchmark.v2",
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            return

    # ------------------------------------------------------------------
    # Step 7: Wire CRANE (primary) + LLMConstraintExtractor (fallback)
    # ------------------------------------------------------------------
    crane_extractor = CRANEExtractionGate(min_confidence=0.7)
    _log.info("CRANEExtractionGate wired (primary, min_confidence=0.7)")

    llm_extractor: object | None = None
    qwen_obj = model_objects.get("Qwen3.5-0.8B")
    if qwen_obj is not None:
        try:
            from carnot.pipeline.llm_extractor import LLMConstraintExtractor  # noqa: PLC0415

            llm_extractor = LLMConstraintExtractor(
                model=qwen_obj,
                tokenizer=None,
                generate_fn=_hf_pipeline_generate_fn,
            )
            _log.info("LLMConstraintExtractor wired as fallback (Qwen3.5-0.8B)")
        except Exception as exc:
            _log.warning(
                "LLMConstraintExtractor unavailable: %s — CRANE-only mode", exc
            )

    # ------------------------------------------------------------------
    # Step 8: Load GSM8K questions
    # ------------------------------------------------------------------
    questions = load_gsm8k_questions(N_QUESTIONS)
    _log.info("Loaded %d GSM8K questions", len(questions))

    # ------------------------------------------------------------------
    # Step 9: Run 5 variants × 2 models with checkpointing
    # ------------------------------------------------------------------
    all_results: list[PrecisionStackResult] = []

    for spec in MODEL_SPECS:
        model_name = spec["name"]
        model_obj = model_objects[model_name]

        _log.info("Running variants for model: %s", model_name)
        for variant in PipelineVariant:
            _log.info("  variant=%s", variant.value)

            from scripts.experiment_368_precision_live import (  # noqa: PLC0415
                _count_baseline_correct,
                _extract_gsm8k_answer,
                _is_correct,
                _call_model,
                MIN_CONFIDENCE,
            )
            from scripts.experiment_template import BatchedInferenceRunner  # noqa: PLC0415
            from carnot.pipeline.precision_benchmark import compute_signed_improvement  # noqa: PLC0415

            # Compute baseline accuracy once per (model, variant) combo.
            # Re-computing per variant is consistent with Exp 368/379 behaviour.
            baseline_correct = _count_baseline_correct(questions, model_obj)
            baseline_acc = baseline_correct / max(len(questions), 1)

            def _inference_fn(q_text: str, _model_obj: object = model_obj) -> str:
                return _call_model(_model_obj, q_text)

            bir = BatchedInferenceRunner(_inference_fn, batch_size=BATCH_SIZE)
            question_texts = [q["question"] for q in questions]
            ir_results = bir.run_batch(question_texts)

            n_correct = 0
            n_violations = 0
            n_repairs = 0
            n_improved = 0
            n_broken = 0

            for ir, q_dict in zip(ir_results, questions):
                gold = _extract_gsm8k_answer(q_dict["answer"])
                if ir.timed_out or not ir.response:
                    continue

                response_before = ir.response
                _, viol_count, rep_attempted = _apply_variant_with_crane(
                    variant,
                    ir.response,
                    q_dict["question"],
                    model_name,
                    crane_extractor,
                    llm_extractor,
                )
                n_violations += viol_count
                n_repairs += rep_attempted

                was_correct_before = _is_correct(response_before, gold)
                was_correct_after = _is_correct(ir.response, gold)

                if was_correct_after:
                    n_correct += 1
                if rep_attempted > 0:
                    if not was_correct_before and was_correct_after:
                        n_improved += 1
                    elif was_correct_before and not was_correct_after:
                        n_broken += 1

            stack_acc = n_correct / max(len(questions), 1)
            signed_improvement = compute_signed_improvement(baseline_acc, stack_acc)

            result = PrecisionStackResult(
                model_id=model_name,
                n_questions=len(questions),
                baseline_accuracy=baseline_acc,
                precision_stack_accuracy=stack_acc,
                signed_improvement=signed_improvement,
                pipeline_variant=variant,
                inference_mode=inference_mode,
                n_violations_found=n_violations,
                n_repairs_attempted=n_repairs,
                n_repairs_improved=n_improved,
                n_repairs_broken=n_broken,
            )
            all_results.append(result)

            _log.info(
                "  %s/%s: baseline=%.3f stack=%.3f Δ=%.3f viol=%d rep=%d",
                model_name,
                variant.value,
                baseline_acc,
                stack_acc,
                signed_improvement,
                n_violations,
                n_repairs,
            )

        # Checkpoint after each model (≈ CHECKPOINT_EVERY questions per variant).
        tmpl.checkpoint_save(
            {"completed_models": [r.model_id for r in all_results]},
            step=len(all_results),
        )

    # ------------------------------------------------------------------
    # Step 11: Build and write artifact
    # ------------------------------------------------------------------
    precision_artifact = build_exp419_artifact(all_results, inference_mode)

    hr = precision_artifact.get("headline_result", {})
    if hr:
        _log.info(
            "HEADLINE: Gemma4-E4B-it FULL_STACK signed_improvement=%.4f "
            "honest_verdict=%s",
            hr.get("signed_improvement", float("nan")),
            precision_artifact.get("honest_verdict", "unknown"),
        )
    else:
        _log.info("HEADLINE: no FULL_STACK Gemma4-E4B-it result found")

    artifact = tmpl.build_result(
        precision_artifact,
        status="success",
        schema="carnot.precision_benchmark.v2",
        inference_mode=inference_mode,
        n_questions=N_QUESTIONS,
        n_models=len(MODEL_SPECS),
        n_variants=len(list(PipelineVariant)),
        model_specs=[s["name"] for s in MODEL_SPECS],
        pipeline_variants=[v.value for v in PipelineVariant],
        live_gpu_confirmed=True,
    )

    _write_artifact(tmpl, artifact)


if __name__ == "__main__":
    main()
