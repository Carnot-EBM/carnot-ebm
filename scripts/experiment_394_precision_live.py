#!/usr/bin/env python3
"""Experiment 394: Live precision pipeline benchmark — first credible precision-stack numbers.

**Researcher summary:**
    Experiments 368 and 379 built and gated the full precision benchmark pipeline (5
    pipeline variants × 2 models × 200 GSM8K questions) but were blocked in milestones
    2026.05.27 and 2026.06.03 because the GPU node was offline.

    Exp 390 ran a GPU preflight that confirmed the hardware state for this milestone.
    This experiment (394) is gated on that preflight result: if Exp 390 did NOT return
    honest_verdict="gpu_confirmed_live", a blocked artifact is written immediately and
    the script exits.  No fake numbers, ever.

    If the preflight confirmed the GPU, the same benchmark logic as Exp 368/379 is
    executed (imported directly — not duplicated) and the results are recorded here.

**What is different from Exp 379?**
    1. First gate: load results/experiment_390_gpu_preflight.json and check
       honest_verdict == "gpu_confirmed_live".  This is a new preflight-based gate
       unique to Exp 394.  It is an outer gate BEFORE LiveGPUGate.
    2. Deliverable path is results/experiment_394_precision_live.json.
    3. Artifact schema is "carnot.precision_benchmark.v2" (same as Exp 379).
    4. All heavy pipeline logic is IMPORTED from Exp 368 — NOT duplicated.

**Hard gate sequence:**
    1. Load results/experiment_390_gpu_preflight.json — if honest_verdict !=
       "gpu_confirmed_live": write blocked artifact, exit immediately.
    2. LiveGPUGate.require_live_or_blocked() — CARNOT_FORCE_LIVE=1 AND
       diagnose_live_gpu() live capability.
    3. tmpl.setup_gpu() — model pre-warm via ExperimentTemplate.
    4. _load_model_pipeline() × 2 — Gemma4-E4B-it + Qwen3.5-0.8B.

    Blocked is always better than fake numbers.

**Five pipeline variants (additive ablation stack):**
    BASELINE:                  ArithmeticExtractor only (control condition)
    CONFIDENCE_ONLY:           + LLMExtractor + ConfidenceWeightedRepair
    CONFIDENCE_ADAPTIVE:       + ModelAdaptiveThresholds (auto-disables high-FP types)
    CONFIDENCE_ADAPTIVE_VERGE: + VergeRefiner (Z3-guided step repair proxy)
    FULL_STACK:                + CoTCircuitVerifier (all tiers active)

**Honest verdict rules (SCENARIO-BENCH-020):**
    "live_improvement":    inference_mode == "live_gpu" AND signed_improvement > 0
                           for the FULL_STACK Gemma4-E4B-it headline result.
    "live_no_improvement": live_gpu run but signed_improvement <= 0.
    "blocked":             GPU unavailable or preflight gate failed.

**Output:** results/experiment_394_precision_live.json

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
    _write_artifact as _write_artifact_368,  # noqa: F401  (re-exported for clarity)
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

EXP_ID = 394
EXP_TITLE = "Live precision pipeline benchmark — credible precision-stack numbers"
DELIVERABLE = "results/experiment_394_precision_live.json"

# Path to the Exp 390 GPU preflight result (relative to repo root).
GPU_PREFLIGHT_PATH = "results/experiment_390_gpu_preflight.json"

# Model IDs for the LiveGPUGate check — must match MODEL_SPECS from Exp 368.
_GATE_MODEL_IDS = [spec["hf_id"] for spec in MODEL_SPECS]


# ---------------------------------------------------------------------------
# Exp 390 preflight loader
# ---------------------------------------------------------------------------


def load_preflight_verdict(repo_root: Path) -> str | None:
    """Load the honest_verdict from the Exp 390 GPU preflight artifact.

    **Detailed explanation for engineers:**
        This is the outermost gate for Exp 394.  Exp 390 ran a GPU preflight
        check and recorded its findings in results/experiment_390_gpu_preflight.json.
        If that file is absent, unreadable, or its honest_verdict is not exactly
        "gpu_confirmed_live", this experiment must produce a blocked artifact and
        stop.

        The honest_verdict field is the authoritative signal from the preflight
        run.  Do not attempt to infer GPU state from other fields — use this value
        alone.

    Args:
        repo_root: Repository root path.  The preflight JSON is at
                   repo_root / GPU_PREFLIGHT_PATH.

    Returns:
        The string value of honest_verdict if present, or None if the file is
        missing, unreadable, or the key is absent.
    """
    path = repo_root / GPU_PREFLIGHT_PATH
    try:
        data = json.loads(path.read_text())
        return data.get("honest_verdict")
    except Exception as exc:
        _log.warning("Could not load Exp 390 preflight from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Artifact builder (Exp 394 v2 schema)
# ---------------------------------------------------------------------------


def build_exp394_artifact(
    results: list[PrecisionStackResult],
    inference_mode: str,
) -> dict[str, Any]:
    """Build the Exp 394 precision benchmark artifact from a list of results.

    **Detailed explanation for engineers:**
        Functionally equivalent to build_exp379_artifact() in Exp 379, extended
        with the same v2 schema and honest_verdict rules.  Defined here (rather than
        imported from Exp 379) so that:
        1. Exp 394's test suite can cover it independently.
        2. The function stays in the scope callers expect.

        Honest verdict rules (SCENARIO-BENCH-020):
        - "live_improvement":    inference_mode == "live_gpu" AND signed_improvement > 0
                                 for the FULL_STACK Gemma4-E4B-it headline result.
        - "live_no_improvement": live_gpu but improvement <= 0.
        - "blocked":             inference_mode is anything other than "live_gpu".

    Args:
        results:        List of PrecisionStackResult objects (5 variants × N models).
        inference_mode: "live_gpu" for a valid run; "blocked" otherwise.

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
        unit tests can patch scripts.experiment_394_precision_live._write_artifact
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
    """Run Experiment 394: live full precision pipeline benchmark.

    **Detailed explanation for engineers:**
        Gate sequence (any failure → blocked artifact → return):
        1. Load results/experiment_390_gpu_preflight.json and check
           honest_verdict == "gpu_confirmed_live".  Exp 390 ran a GPU preflight
           for this milestone; if it did not confirm the GPU we must not proceed.
        2. LiveGPUGate.require_live_or_blocked() — checks CARNOT_FORCE_LIVE=1 AND
           diagnose_live_gpu() live capability.
        3. tmpl.setup_gpu() — pre-warms models via ExperimentTemplate pattern.
        4. _load_model_pipeline() × 2 — Gemma4-E4B-it (GPU 0) + Qwen3.5-0.8B (GPU 1).
        5. LLMConstraintExtractor wired to Qwen3.5-0.8B for IT-format extraction.
        6. load_gsm8k_questions() — 200 questions from HuggingFace or synthetic fallback.
        7. run_variant() × 10 (5 variants × 2 models) via the Exp 368 runner.
        8. build_exp394_artifact() → _write_artifact().

        Heavy pipeline logic (run_variant, load_gsm8k_questions, model loading) is
        IMPORTED from scripts.experiment_368_precision_live — NOT duplicated here.
        This keeps Exp 394 as a thin orchestration wrapper.
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # ---------------------------------------------------------------------------
    # Gate 0: Check Exp 390 GPU preflight result.
    # If honest_verdict != "gpu_confirmed_live", blocked immediately.
    # ---------------------------------------------------------------------------
    preflight_verdict = load_preflight_verdict(tmpl._repo_root)
    if preflight_verdict != "gpu_confirmed_live":
        _log.error(
            "Exp 390 GPU preflight verdict is %r (expected 'gpu_confirmed_live') — "
            "writing blocked artifact.",
            preflight_verdict,
        )
        artifact = tmpl.build_result(
            {
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": (
                    f"Exp 390 GPU preflight honest_verdict={preflight_verdict!r} "
                    "is not 'gpu_confirmed_live'"
                ),
                "precision_schema": "carnot.precision_benchmark.v2",
                "preflight_verdict": preflight_verdict,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    _log.info("Exp 390 GPU preflight confirmed — honest_verdict='gpu_confirmed_live'")

    # ---------------------------------------------------------------------------
    # Gate 1: LiveGPUGate checks CARNOT_FORCE_LIVE=1 AND diagnose_live_gpu().
    # If either fails, a blocked artifact is returned and we exit immediately.
    # ---------------------------------------------------------------------------
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, _GATE_MODEL_IDS)
    if blocked is not None:
        _log.error(
            "LiveGPUGate blocked Exp 394 — CARNOT_FORCE_LIVE not set or GPU not live. "
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
    # Gate 2: GPU setup via ExperimentTemplate.setup_gpu().
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
    # Gate 3: Load models for live inference.
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
                "  %s/%s: baseline=%.3f stack=%.3f delta=%.3f violations=%d repairs=%d",
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
    precision_artifact = build_exp394_artifact(all_results, inference_mode)

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
