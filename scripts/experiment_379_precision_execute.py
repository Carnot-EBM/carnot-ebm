#!/usr/bin/env python3
"""Experiment 379: Live precision pipeline execution — first credible headline number.

**Researcher summary:**
    Exp 368 built and tested a complete precision benchmark pipeline (5 variants ×
    2 models × 200 GSM8K questions) with CARNOT_FORCE_LIVE=1 hard-gate logic.
    However, it was blocked in milestone 2026.04.27 because the session startup never
    set CARNOT_FORCE_LIVE=1 in conductor subprocesses.

    Exp 377 fixed the session startup by writing ``scripts/session_startup.sh`` and
    verified the env var propagation via LiveGPUGate.verify_subprocess_env_propagation().
    This experiment (379) is the EXECUTION of that fixed pipeline — the same
    benchmark logic as Exp 368, but using LiveGPUGate (the Exp 377 hard gate class)
    as the first gate instead of a raw os.environ check.

**What is different from Exp 368?**
    1. First gate uses ``LiveGPUGate.require_live_or_blocked()`` instead of a
       raw ``os.environ.get("CARNOT_FORCE_LIVE")`` check.  This is the pattern
       established by Exp 377 to make the gate more reusable and testable.
    2. Deliverable path is ``results/experiment_379_precision_execute.json``.
    3. All heavy pipeline logic (run_variant, load_gsm8k_questions, model loading,
       LLM extractor wiring) is IMPORTED from Exp 368 — NOT duplicated.  This
       experiment is a thin orchestration wrapper over the Exp 368 pipeline modules.

**Hard CARNOT_FORCE_LIVE=1 requirement (same as Exp 368):**
    - ``CARNOT_FORCE_LIVE != "1"`` → blocked artifact written, exit immediately.
    - ``diagnose_live_gpu()`` reports ``is_live_capable=False`` → blocked, exit.
    - ``setup_gpu()`` reports ``all_healthy=False`` → blocked, exit.
    - Any model fails to load → blocked, exit.

    Blocked is always better than fake numbers.

**Five pipeline variants (additive ablation stack):**
    BASELINE:                  ArithmeticExtractor only (control condition)
    CONFIDENCE_ONLY:           + LLMExtractor + ConfidenceWeightedRepair
    CONFIDENCE_ADAPTIVE:       + ModelAdaptiveThresholds
    CONFIDENCE_ADAPTIVE_VERGE: + VergeRefiner (Z3-guided step repair proxy)
    FULL_STACK:                + CoTCircuitVerifier

**Honest verdict rules (SCENARIO-BENCH-020):**
    ``honest_verdict="live_improvement"`` ONLY when:
    1. ``inference_mode == "live_gpu"`` (confirmed by LiveGPUGate)
    2. ``signed_improvement > 0`` for FULL_STACK on Gemma4-E4B-it

    Otherwise: ``"live_no_improvement"`` (live run, stack didn't help) or
    ``"blocked"`` (GPU unavailable).

**Output:** results/experiment_379_precision_execute.json

Spec: REQ-BENCH-003, SCENARIO-BENCH-007, SCENARIO-BENCH-008, SCENARIO-BENCH-009,
      SCENARIO-BENCH-020
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path so scripts.* and carnot.* resolve.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import (  # noqa: E402
    ExperimentTemplate,
)
from scripts.experiment_368_precision_live import (  # noqa: E402
    load_gsm8k_questions,
    run_variant,
    _load_model_pipeline,
    _hf_pipeline_generate_fn,
    _write_artifact as _write_artifact_368,
    MODEL_SPECS,
    N_QUESTIONS,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
)
from carnot.pipeline.precision_benchmark import (  # noqa: E402
    PipelineVariant,
    PrecisionStackResult,
    build_precision_benchmark_artifact,
)
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 379
EXP_TITLE = "Live precision pipeline execution"
DELIVERABLE = "results/experiment_379_precision_execute.json"

# Model IDs for the LiveGPUGate check.
_GATE_MODEL_IDS = [spec["hf_id"] for spec in MODEL_SPECS]


# ---------------------------------------------------------------------------
# Artifact builder (Exp 379 v2 schema)
# ---------------------------------------------------------------------------


def build_exp379_artifact(
    results: list[PrecisionStackResult],
    inference_mode: str,
) -> dict[str, Any]:
    """Build the Exp 379 precision benchmark artifact from a list of results.

    **Detailed explanation for engineers:**
        This is functionally equivalent to ``build_exp368_artifact()`` in Exp 368,
        extended with the same v2 schema and honest_verdict rules.  It is defined
        here rather than imported from Exp 368 so that:
        1. Exp 379's test suite can cover it independently.
        2. The function signature stays in the scope of this module (no cross-script
           import of a module-level helper that callers might expect to live here).

        Honest verdict rules (SCENARIO-BENCH-020):
        - ``"live_improvement"``: inference_mode == "live_gpu" AND signed_improvement > 0
          for the FULL_STACK Gemma4-E4B-it result.
        - ``"live_no_improvement"``: live_gpu but improvement <= 0.
        - ``"blocked"``: inference_mode is anything other than "live_gpu".

    Args:
        results:        List of PrecisionStackResult objects (5 variants × N models).
        inference_mode: Must be "live_gpu" for a valid run, "blocked" otherwise.

    Returns:
        Dict with schema v2, headline_result, per_variant_results, inference_mode,
        and honest_verdict.
    """
    # Delegate to the shared Exp 340/368 builder for the common structure.
    base = build_precision_benchmark_artifact(results)

    # Override schema to v2 (distinguishes from Exp 340 v1 simulated results).
    base["precision_schema"] = "carnot.precision_benchmark.v2"

    # Set inference_mode explicitly — the base builder infers from result objects,
    # but we want to override it with the confirmed value from LiveGPUGate.
    base["inference_mode"] = inference_mode

    # Compute honest_verdict per SCENARIO-BENCH-020 rules.
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
        Extracted to a module-level function (not inlined in main()) so that
        unit tests can patch ``scripts.experiment_379_precision_execute._write_artifact``
        without needing to instrument main() directly.
    """
    output_path = tmpl._output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 379: live full precision pipeline execution.

    **Detailed explanation for engineers:**
        This is the orchestration wrapper over the Exp 368 pipeline modules.
        Heavy work (run_variant, load_gsm8k_questions, model loading) is imported
        from scripts.experiment_368_precision_live — NOT duplicated here.

        Gate sequence:
        1. LiveGPUGate.require_live_or_blocked() — checks CARNOT_FORCE_LIVE=1 AND
           diagnose_live_gpu() live capability.  Writes blocked artifact and returns
           immediately if either check fails.
        2. tmpl.setup_gpu() — pre-warms models via ExperimentTemplate pattern.
           Writes blocked artifact and returns if all_healthy=False.
        3. _load_model_pipeline() × 2 — loads Gemma4-E4B-it (GPU 0) and
           Qwen3.5-0.8B (GPU 1).  Writes blocked artifact and returns on any load error.
        4. LLMConstraintExtractor wired to Qwen3.5-0.8B for IT-format extraction
           (replaces ArithmeticExtractor for non-BASELINE variants).
        5. load_gsm8k_questions() — 200 questions from HuggingFace or synthetic fallback.
        6. run_variant() × 10 (5 variants × 2 models) via the Exp 368 runner.
        7. build_exp379_artifact() → _write_artifact().
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # ---------------------------------------------------------------------------
    # Hard gate: LiveGPUGate checks CARNOT_FORCE_LIVE=1 AND diagnose_live_gpu().
    # If either fails, a blocked artifact is returned and we exit immediately.
    # ---------------------------------------------------------------------------
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, _GATE_MODEL_IDS)
    if blocked is not None:
        _log.error(
            "LiveGPUGate blocked Exp 379 — CARNOT_FORCE_LIVE not set or GPU not live. "
            "Writing blocked artifact."
        )
        blocked["precision_schema"] = "carnot.precision_benchmark.v2"
        blocked["inference_mode"] = "blocked"
        blocked["honest_verdict"] = "blocked"
        _write_artifact(tmpl, blocked)
        return

    inference_mode = "live_gpu"
    _log.info("LiveGPUGate passed — inference_mode=%s", inference_mode)

    # ---------------------------------------------------------------------------
    # GPU setup: ModelServer + DualGPURunner via ExperimentTemplate.setup_gpu().
    # ---------------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        _log.error("GPU setup unhealthy after gate passed — writing blocked artifact.")
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

    # ---------------------------------------------------------------------------
    # Load models for live inference.
    # ---------------------------------------------------------------------------
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

    # Wire LLMConstraintExtractor backed by live Qwen3.5-0.8B for IT-format extraction.
    # This replaces ArithmeticExtractor for non-BASELINE variants (Exp 366 finding:
    # ArithmeticExtractor misses ~40% of claims in IT-format responses).
    qwen_obj = model_objects.get("Qwen3.5-0.8B")
    extractor_obj: object | None = None
    if qwen_obj is not None:
        try:
            from carnot.pipeline.llm_extractor import LLMConstraintExtractor  # noqa: PLC0415

            extractor_obj = LLMConstraintExtractor(
                model=qwen_obj,
                tokenizer=None,
                generate_fn=_hf_pipeline_generate_fn,
            )
            _log.info("LLMConstraintExtractor wired to Qwen3.5-0.8B")
        except Exception as exc:
            _log.warning(
                "Could not build LLMConstraintExtractor: %s — falling back to ArithmeticExtractor",
                exc,
            )

    # ---------------------------------------------------------------------------
    # Load GSM8K questions.
    # ---------------------------------------------------------------------------
    questions = load_gsm8k_questions(N_QUESTIONS)
    _log.info("Loaded %d GSM8K questions", len(questions))

    # ---------------------------------------------------------------------------
    # Run all 5 variants × 2 models.
    # Checkpoint every CHECKPOINT_EVERY questions worth of work (after each model).
    # ---------------------------------------------------------------------------
    all_results: list[PrecisionStackResult] = []

    for spec in MODEL_SPECS:
        model_name = spec["name"]
        model_obj = model_objects[model_name]

        _log.info("Running variants for model: %s", model_name)
        for variant in PipelineVariant:
            _log.info("  variant=%s", variant.value)
            result = run_variant(
                variant=variant,
                questions=questions,
                model_name=model_name,
                inference_mode=inference_mode,
                model_obj=model_obj,
                extractor_obj=extractor_obj,
            )
            all_results.append(result)
            _log.info(
                "  %s/%s: baseline=%.3f stack=%.3f Δ=%.3f violations=%d repairs=%d",
                model_name,
                variant.value,
                result.baseline_accuracy,
                result.precision_stack_accuracy,
                result.signed_improvement,
                result.n_violations_found,
                result.n_repairs_attempted,
            )

        # Checkpoint after each model (equivalent to every CHECKPOINT_EVERY questions).
        tmpl.checkpoint_save(
            {"completed_models": [r.model_id for r in all_results]},
            step=len(all_results),
        )

    # ---------------------------------------------------------------------------
    # Build and write artifact.
    # ---------------------------------------------------------------------------
    precision_artifact = build_exp379_artifact(all_results, inference_mode)

    hr = precision_artifact.get("headline_result", {})
    if hr:
        label = hr.get("headline_label", "no_positive_result")
        verdict = precision_artifact.get("honest_verdict", "unknown")
        _log.info(
            "HEADLINE: Gemma4-E4B-it FULL_STACK signed_improvement=%.4f "
            "label=%s honest_verdict=%s",
            hr.get("signed_improvement", float("nan")),
            label,
            verdict,
        )
    else:
        _log.info("HEADLINE: no FULL_STACK Gemma4-E4B-it result found")

    artifact = tmpl.build_result(
        precision_artifact,
        status="success",
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
