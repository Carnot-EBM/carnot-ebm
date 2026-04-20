#!/usr/bin/env python3
"""Experiment 410: Live precision pipeline benchmark — credible precision-stack numbers.

**Researcher summary:**
    Exp 404 preflight v2 must have returned honest_verdict='gpu_confirmed_live' for this
    experiment to proceed.  If the preflight verdict is anything else, this script writes
    a blocked artifact and exits immediately.  No simulated fallback is permitted.

    This experiment runs the full 5-variant × 2-model × 200-GSM8K benchmark that
    prior milestones blocked (GPU node offline, env propagation failures).  It produces
    Carnot's first credible precision-stack numbers IF the GPU prerequisites are met.

**What is different from Exp 379?**
    1. Reads results/experiment_404_preflight_v2.json first.  If honest_verdict is not
       'gpu_confirmed_live', writes a blocked artifact and exits with no inference.
    2. FULL_STACK variant uses CRANEExtractionGate (Exp 409) as the primary extractor,
       falling back to LLMExtractor (Exp 366) if CRANE is unavailable or errors out.
       This is a 1× inference call (CRANE) rather than the 2× call used in Exp 379.
    3. Schema is 'carnot.precision_benchmark.v2' (same as Exp 379 — consistent v2 tag).
    4. Deliverable: results/experiment_410_precision_live.json

**Hard gate sequence:**
    1. Load results/experiment_404_preflight_v2.json.
       honest_verdict != 'gpu_confirmed_live' → blocked artifact, exit immediately.
    2. LiveGPUGate.require_live_or_blocked() — checks CARNOT_FORCE_LIVE=1 AND live GPU.
    3. tmpl.setup_gpu() — pre-warms models.  all_healthy=False → blocked.
    4. Model load failures → blocked.

    Blocked is always better than fake numbers.

**Five pipeline variants (additive ablation stack):**
    BASELINE:                  ArithmeticExtractor only (control condition)
    CONFIDENCE_ONLY:           + LLMExtractor + ConfidenceWeightedRepair
    CONFIDENCE_ADAPTIVE:       + ModelAdaptiveThresholds
    CONFIDENCE_ADAPTIVE_VERGE: + VergeRefiner (Z3-guided step repair proxy)
    FULL_STACK:                + CRANEExtractionGate (with LLMExtractor fallback)

**Honest verdict rules (SCENARIO-BENCH-020):**
    'live_improvement'  : inference_mode=='live_gpu' AND signed_improvement>0 for
                          FULL_STACK on Gemma4-E4B-it.
    'live_no_improvement': live_gpu run but improvement <= 0.
    'blocked'           : GPU unavailable or preflight not gpu_confirmed_live.

**Output:** results/experiment_410_precision_live.json

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

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from scripts.experiment_368_precision_live import (  # noqa: E402
    load_gsm8k_questions,
    run_variant,
    _load_model_pipeline,
    _hf_pipeline_generate_fn,
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

EXP_ID = 410
EXP_TITLE = "Live precision pipeline benchmark"
DELIVERABLE = "results/experiment_410_precision_live.json"

#: Path to the Exp 404 preflight result that gates this experiment.
EXP_404_PREFLIGHT = "results/experiment_404_preflight_v2.json"

#: Required honest_verdict value in the Exp 404 preflight to proceed.
REQUIRED_PREFLIGHT_VERDICT = "gpu_confirmed_live"

#: Model IDs for the LiveGPUGate check.
_GATE_MODEL_IDS = [spec["hf_id"] for spec in MODEL_SPECS]


# ---------------------------------------------------------------------------
# Preflight gate
# ---------------------------------------------------------------------------


def load_preflight_verdict(repo_root: Path = _REPO_ROOT) -> str:
    """Load and return honest_verdict from the Exp 404 preflight JSON.

    **Detailed explanation for engineers:**
        Exp 404 ran a hardware preflight check and recorded its result in
        results/experiment_404_preflight_v2.json.  We read the 'honest_verdict'
        field from that file here.  If the file is missing or malformed we treat
        it as 'missing' (a blocking condition — never assume the GPU is ready
        when we cannot confirm it).

    Args:
        repo_root: Repository root directory.  Defaults to the directory two
                   levels above this script (i.e. the actual repo root).

    Returns:
        The 'honest_verdict' string from the preflight JSON, or 'missing' if
        the file cannot be read.
    """
    preflight_path = repo_root / EXP_404_PREFLIGHT
    try:
        data = json.loads(preflight_path.read_text())
        return str(data.get("honest_verdict", "missing"))
    except Exception as exc:  # noqa: BLE001 — treat any read error as missing
        _log.warning("Could not read preflight JSON at %s: %s", preflight_path, exc)
        return "missing"


# ---------------------------------------------------------------------------
# Artifact builder (Exp 410 schema)
# ---------------------------------------------------------------------------


def build_exp410_artifact(
    results: list[PrecisionStackResult],
    inference_mode: str,
) -> dict[str, Any]:
    """Build the Exp 410 precision benchmark artifact from a list of results.

    **Detailed explanation for engineers:**
        Functionally equivalent to the Exp 379 builder, extended for Exp 410's
        schema tag and honest_verdict rules.  Delegates to the shared Exp 340
        builder (build_precision_benchmark_artifact) for the common structure,
        then overrides schema to v2 and computes honest_verdict.

        Honest verdict rules (SCENARIO-BENCH-020):
        - 'live_improvement': inference_mode == 'live_gpu' AND signed_improvement > 0
          for the FULL_STACK Gemma4-E4B-it result.
        - 'live_no_improvement': live_gpu run but improvement <= 0.
        - 'blocked': inference_mode is anything other than 'live_gpu'.

    Args:
        results:        List of PrecisionStackResult objects (5 variants × N models).
        inference_mode: 'live_gpu' for a real run, 'blocked' otherwise.

    Returns:
        Dict with schema v2, headline_result, per_variant_results, inference_mode,
        and honest_verdict.
    """
    base = build_precision_benchmark_artifact(results)

    # Exp 410 uses the same v2 schema as Exp 379 (live results vs. v1 simulated).
    base["schema"] = "carnot.precision_benchmark.v2"
    base["precision_schema"] = "carnot.precision_benchmark.v2"

    # Override inference_mode with the value confirmed by LiveGPUGate.
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
        unit tests can patch 'scripts.experiment_410_precision_live._write_artifact'
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
    """Run Experiment 410: live full precision pipeline execution.

    **Detailed explanation for engineers:**
        Gate sequence:
        1. Read results/experiment_404_preflight_v2.json.
           honest_verdict != 'gpu_confirmed_live' → write blocked artifact, exit.
           This is a hard pre-check before any ExperimentTemplate setup.
        2. ExperimentTemplate(410, ...) + tmpl.setup().
        3. LiveGPUGate.require_live_or_blocked() — checks CARNOT_FORCE_LIVE=1 AND
           live GPU.  Returns blocked artifact on failure.
        4. tmpl.setup_gpu() — pre-warms Gemma4-E4B-it + Qwen3.5-0.8B.
           Blocked artifact if all_healthy=False.
        5. Model loading × 2.  Blocked artifact on any load failure.
        6. Wire FULL_STACK extractor: try CRANEExtractionGate (Exp 409) first;
           fall back to LLMConstraintExtractor (Exp 366) if CRANE unavailable.
        7. load_gsm8k_questions() — 200 questions.
        8. run_variant() × 10 (5 variants × 2 models).  Checkpoint every 50 questions.
        9. build_exp410_artifact() → _write_artifact().
    """
    from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: PLC0415
    from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: PLC0415

    apply_env_autofix()

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=40,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )
    _watchdog.start()

    # ---------------------------------------------------------------------------
    # Gate 1: Exp 404 preflight verdict.
    # Must be 'gpu_confirmed_live' or we write blocked and exit immediately.
    # ---------------------------------------------------------------------------
    preflight_verdict = load_preflight_verdict()
    if preflight_verdict != REQUIRED_PREFLIGHT_VERDICT:
        _log.error(
            "Exp 404 preflight honest_verdict=%r (expected 'gpu_confirmed_live') — "
            "writing blocked artifact.",
            preflight_verdict,
        )
        artifact = tmpl.build_result(
            {
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "precision_schema": "carnot.precision_benchmark.v2",
                "schema": "carnot.precision_benchmark.v2",
                "failure_reason": (
                    f"preflight blocked: honest_verdict={preflight_verdict!r}, "
                    f"required 'gpu_confirmed_live'"
                ),
                "preflight_verdict": preflight_verdict,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # ---------------------------------------------------------------------------
    # Gate 2: LiveGPUGate checks CARNOT_FORCE_LIVE=1 AND diagnose_live_gpu().
    # ---------------------------------------------------------------------------
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, _GATE_MODEL_IDS)
    if blocked is not None:
        _log.error(
            "LiveGPUGate blocked Exp 410 — CARNOT_FORCE_LIVE not set or GPU not live. "
            "Writing blocked artifact."
        )
        blocked["precision_schema"] = "carnot.precision_benchmark.v2"
        blocked["schema"] = "carnot.precision_benchmark.v2"
        blocked["inference_mode"] = "blocked"
        blocked["honest_verdict"] = "blocked"
        _write_artifact(tmpl, blocked)
        return

    inference_mode = "live_gpu"
    _log.info("LiveGPUGate passed — inference_mode=%s", inference_mode)

    # ---------------------------------------------------------------------------
    # Gate 3: GPU setup via ExperimentTemplate.setup_gpu().
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
                "schema": "carnot.precision_benchmark.v2",
                "gpu_setup_status": gpu_status,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # ---------------------------------------------------------------------------
    # Gate 4: Load models for live inference.
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
                    "schema": "carnot.precision_benchmark.v2",
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            return

    # ---------------------------------------------------------------------------
    # Wire extractors for FULL_STACK variant.
    # Primary: CRANEExtractionGate (Exp 409) — 1x inference call.
    # Fallback: LLMConstraintExtractor (Exp 366) if CRANE unavailable or errors.
    # ---------------------------------------------------------------------------
    qwen_obj = model_objects.get("Qwen3.5-0.8B")
    crane_extractor_obj: object | None = None
    extractor_obj: object | None = None

    # Try CRANE first (Exp 409).
    if qwen_obj is not None:
        try:
            from carnot.pipeline.crane_extractor import CRANEExtractionGate  # noqa: PLC0415

            crane_extractor_obj = CRANEExtractionGate(
                model=qwen_obj,
                tokenizer=None,
                generate_fn=_hf_pipeline_generate_fn,
            )
            _log.info("CRANEExtractionGate (Exp 409) wired to Qwen3.5-0.8B for FULL_STACK")
        except Exception as exc:
            _log.warning(
                "CRANEExtractionGate unavailable (%s) — will fall back to LLMExtractor",
                exc,
            )

    # Fall back to LLMConstraintExtractor (Exp 366) if CRANE not available.
    if crane_extractor_obj is not None:
        extractor_obj = crane_extractor_obj
    elif qwen_obj is not None:
        try:
            from carnot.pipeline.llm_extractor import LLMConstraintExtractor  # noqa: PLC0415

            extractor_obj = LLMConstraintExtractor(
                model=qwen_obj,
                tokenizer=None,
                generate_fn=_hf_pipeline_generate_fn,
            )
            _log.info(
                "LLMConstraintExtractor (Exp 366) wired to Qwen3.5-0.8B as FULL_STACK fallback"
            )
        except Exception as exc:
            _log.warning(
                "Could not build LLMConstraintExtractor: %s — FULL_STACK will use baseline extractor",
                exc,
            )

    # ---------------------------------------------------------------------------
    # Load GSM8K questions.
    # ---------------------------------------------------------------------------
    questions = load_gsm8k_questions(N_QUESTIONS)
    _log.info("Loaded %d GSM8K questions", len(questions))

    # ---------------------------------------------------------------------------
    # Run all 5 variants × 2 models.
    # Checkpoint every CHECKPOINT_EVERY questions (after each model's variants).
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

        # Checkpoint after each model (covers every CHECKPOINT_EVERY questions).
        tmpl.checkpoint_save(
            {"completed_models": [r.model_id for r in all_results]},
            step=len(all_results),
        )

    # ---------------------------------------------------------------------------
    # Build and write artifact.
    # ---------------------------------------------------------------------------
    precision_artifact = build_exp410_artifact(all_results, inference_mode)

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
    _watchdog.stop()
    tmpl.assert_deliverable_written()


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
