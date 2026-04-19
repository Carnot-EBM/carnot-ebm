#!/usr/bin/env python3
"""Experiment 427: Confirm or re-run Exp 419 live precision benchmark.

**Researcher summary:**
    Exp 419 ran the 5-variant × 2-model × 200-question GSM8K precision benchmark
    with CRANEExtractionGate as the primary extractor and LLMConstraintExtractor as
    fallback.  The run was interrupted after 144+ minutes: results/experiment_419_
    precision_live.json contains only ``{"experiment": 419, "status": "partial"}``.

    This experiment (427) either:
    (a) Confirms Exp 419 results — if status='success' AND inference_mode='live_gpu'
        AND honest_verdict in ('live_improvement', 'live_no_improvement'), copies
        the result with experiment=427, confirmed_from=419, rerun=False.
    (b) Re-runs the full benchmark — if status='partial' or inference_mode!='live_gpu',
        repeats the 5×2×200 run with the same helpers, adding two new guards:
        - ``ExperimentTimeoutWatchdog(427, timeout_minutes=90)`` (RETRO-003).
        - ``check_dual_gpu_health()`` warning when GPU1 is a zombie (RETRO-025).
        - ``crane_detection_rate`` metric (fraction of FULL_STACK questions where
          CRANE found at least one arithmetic violation).

**Why 90-minute watchdog (not 45)?**
    The 5×2×200 benchmark legitimately requires more wall-clock time than a typical
    45-minute experiment.  Exp 425's default is 45 min; here we set 90 min to allow
    the full run to complete while still preventing indefinite GPU occupation.

**Output:** results/experiment_427_precision_live_confirmed.json

Spec: REQ-BENCH-003, SCENARIO-BENCH-020
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import.  Moving this below any torch/JAX import is a bug.
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

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from scripts.experiment_419_precision_live import (  # noqa: E402
    build_exp419_artifact,
    _write_artifact,
    _apply_variant_with_crane,
    _ALLOWED_VERDICTS,
    MODEL_SPECS,
    N_QUESTIONS,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    EXP_TITLE as _EXP419_TITLE,
)
from scripts.experiment_368_precision_live import (  # noqa: E402
    load_gsm8k_questions,
    _load_model_pipeline,
    _hf_pipeline_generate_fn,
    _count_baseline_correct,
    _extract_gsm8k_answer,
    _is_correct,
    _call_model,
    MIN_CONFIDENCE,
)
from carnot.pipeline.precision_benchmark import (  # noqa: E402
    PipelineVariant,
    PrecisionStackResult,
    compute_signed_improvement,
)
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from carnot.pipeline.crane_extractor import CRANEExtractionGate  # noqa: E402
from carnot.pipeline.dual_gpu_health import check_dual_gpu_health  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 427
EXP_TITLE = "Confirm/re-run Exp 419 live precision benchmark with CRANE extractor"
DELIVERABLE = "results/experiment_427_precision_live_confirmed.json"
WATCHDOG_TIMEOUT_MINUTES = 90

_EXP419_RESULT_PATH = "results/experiment_419_precision_live.json"
_EXP413_RESULT_PATH = "results/experiment_413_env_autofix.json"

_CONFIRM_VERDICTS = frozenset(["live_improvement", "live_no_improvement"])


# ---------------------------------------------------------------------------
# New helper: crane_detection_rate
# ---------------------------------------------------------------------------


def compute_crane_detection_rate(crane_hits: list[bool]) -> float:
    """Compute fraction of questions where CRANE found at least one arithmetic violation.

    **Detailed explanation for engineers:**
        ``crane_hits`` is a list of booleans, one per question, recording whether
        CRANEExtractionGate.extract() returned a non-empty list for that question's
        model response.  A ``True`` entry means CRANE successfully identified a
        constraint violation before the LLM fallback was invoked.

        This rate measures CRANE's coverage on the FULL_STACK variant across all
        models:  a rate of 0.0 means CRANE never fired (all inference was carried
        by the LLM fallback) whereas 1.0 means CRANE covered every question.

    Args:
        crane_hits: Boolean list — True when CRANE found ≥1 violation for that question.

    Returns:
        Fraction in [0.0, 1.0]; 0.0 if the list is empty.
    """
    if not crane_hits:
        return 0.0
    return sum(1 for h in crane_hits if h) / len(crane_hits)


# ---------------------------------------------------------------------------
# Artifact builder for Exp 427
# ---------------------------------------------------------------------------


def build_exp427_artifact(
    results: list[PrecisionStackResult],
    inference_mode: str,
    crane_hits: list[bool],
    rerun: bool,
    confirmed_from: int = 419,
) -> dict[str, Any]:
    """Build Exp 427 artifact by extending Exp 419 artifact with confirmation metadata.

    **Detailed explanation for engineers:**
        Calls ``build_exp419_artifact()`` (imported from Exp 419) for all core
        fields, then overlays Exp 427-specific fields:

        - ``experiment``: 427 (overrides 419).
        - ``confirmed_from``: 419 — traceability to the original run.
        - ``rerun``: True when Exp 419 result was partial; False when copied verbatim.
        - ``crane_detection_rate``: fraction of FULL_STACK questions where CRANE
          found violations without needing the LLM fallback.

    Args:
        results:        5×2 PrecisionStackResult list (or empty for blocked).
        inference_mode: "live_gpu" or "blocked".
        crane_hits:     Per-question boolean list for CRANE coverage metric.
        rerun:          True when the benchmark was re-run (Exp 419 was partial).
        confirmed_from: Source experiment ID (always 419 for this experiment).

    Returns:
        Dict with all carnot.precision_benchmark.v2 fields plus 427-specific fields.
    """
    base = build_exp419_artifact(results, inference_mode)

    base["experiment"] = EXP_ID
    base["confirmed_from"] = confirmed_from
    base["rerun"] = rerun
    base["crane_detection_rate"] = compute_crane_detection_rate(crane_hits)

    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — gate chain is inherently long
    """Run Experiment 427: confirm or re-run Exp 419 live precision benchmark.

    **Detailed explanation for engineers:**
        Decision tree:
        1. Read results/experiment_419_precision_live.json.
           - If status='success' AND inference_mode='live_gpu' AND honest_verdict
             in ('live_improvement', 'live_no_improvement'): CONFIRM path — copy
             result with experiment=427, confirmed_from=419, rerun=False.
           - Otherwise: RERUN path — full benchmark re-run with gate chain below.

        Gate chain (RERUN path):
        2. Gate 0: Exp 413 honest_verdict must be in _ALLOWED_VERDICTS.
        3. Gate 1: LiveGPUGate.require_live_or_blocked() — no simulated fallback.
        4. Gate 2: check_dual_gpu_health() — WARNING if gpu1_is_zombie (continue;
           a zombie GPU1 only wastes memory, it does not corrupt results).
        5. Gate 3: tmpl.setup_gpu() — blocked if not all_healthy.
        6. Gate 4: Model load for Gemma4-E4B-it (GPU0) and Qwen3.5-0.8B (GPU1).
        7. ExperimentTimeoutWatchdog(427, timeout_minutes=90) wraps inference loop.
        8. 5 variants × 2 models × 200 GSM8K; checkpoint every 50 questions.
        9. crane_detection_rate computed from FULL_STACK per-question CRANE hits.
        10. Artifact written with schema='carnot.precision_benchmark.v2'.
    """
    # ------------------------------------------------------------------
    # Step 1: Check Exp 419 result
    # ------------------------------------------------------------------
    exp419_path = _REPO_ROOT / _EXP419_RESULT_PATH
    exp419_data: dict[str, Any] = {}
    try:
        exp419_data = json.loads(exp419_path.read_text())
    except Exception as exc:
        _log.warning("Could not load Exp 419 result: %s — proceeding to re-run", exc)

    exp419_status = exp419_data.get("status", "")
    exp419_mode = exp419_data.get("inference_mode", "")
    exp419_verdict = exp419_data.get("honest_verdict", "")

    can_confirm = (
        exp419_status == "success"
        and exp419_mode == "live_gpu"
        and exp419_verdict in _CONFIRM_VERDICTS
    )

    # ------------------------------------------------------------------
    # Step 1a: CONFIRM PATH — copy Exp 419 result with 427 metadata
    # ------------------------------------------------------------------
    if can_confirm:
        _log.info(
            "Exp 419 result is confirmable (verdict=%s, mode=%s) — copying.",
            exp419_verdict,
            exp419_mode,
        )
        tmpl = ExperimentTemplate(
            exp_id=EXP_ID, title=EXP_TITLE, deliverable=DELIVERABLE,
            requires_gpu=False,
        )
        tmpl.setup()

        confirmed = dict(exp419_data)
        confirmed["experiment"] = EXP_ID
        confirmed["confirmed_from"] = 419
        confirmed["rerun"] = False
        confirmed.setdefault("crane_detection_rate", 0.0)

        artifact = tmpl.build_result(confirmed, status="success")
        _write_artifact(tmpl, artifact)

        hr = confirmed.get("headline_result", {})
        _log.info(
            "CONFIRMED: honest_verdict=%s signed_improvement=%.4f",
            confirmed.get("honest_verdict"),
            hr.get("signed_improvement", float("nan")),
        )
        return

    _log.info(
        "Exp 419 status=%r mode=%r verdict=%r — proceeding to RERUN.",
        exp419_status,
        exp419_mode,
        exp419_verdict,
    )

    # ------------------------------------------------------------------
    # Step 2: Gate 0 — Exp 413 honest_verdict
    # ------------------------------------------------------------------
    exp413_path = _REPO_ROOT / _EXP413_RESULT_PATH
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
            exp_id=EXP_ID, title=EXP_TITLE, deliverable=DELIVERABLE,
            requires_gpu=True,
        )
        tmpl.setup()
        artifact = tmpl.build_result(
            {
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "precision_schema": "carnot.precision_benchmark.v2",
                "confirmed_from": 419,
                "rerun": True,
                "crane_detection_rate": 0.0,
                "failure_reason": (
                    f"Exp 413 honest_verdict={exp413_verdict!r} not in approved set; "
                    "run Exp 413 (EnvironmentAutoFix) first"
                ),
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    _log.info("Gate 0 passed (Exp 413 verdict=%s)", exp413_verdict)

    # ------------------------------------------------------------------
    # Step 3: ExperimentTemplate setup
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
        _log.error("LiveGPUGate blocked Exp 427 — writing blocked artifact.")
        blocked["precision_schema"] = "carnot.precision_benchmark.v2"
        blocked["inference_mode"] = "blocked"
        blocked["honest_verdict"] = "blocked"
        blocked["confirmed_from"] = 419
        blocked["rerun"] = True
        blocked["crane_detection_rate"] = 0.0
        _write_artifact(tmpl, blocked)
        return

    inference_mode = "live_gpu"
    _log.info("Gate 1 passed — inference_mode=%s", inference_mode)

    # ------------------------------------------------------------------
    # Step 5: Gate 2 — Dual-GPU health check (WARNING only, not blocking)
    # ------------------------------------------------------------------
    gpu_health = check_dual_gpu_health()
    if gpu_health.gpu1_is_zombie:
        _log.warning(
            "GPU1 is zombie (RETRO-025): VRAM allocated but compute=0. "
            "Exp 427 will use GPU0 only — throughput may be halved."
        )
    if gpu_health.temperature_warning:
        _log.warning(
            "Temperature warning: one or more GPUs exceed 80C. "
            "Batch size factor=%.2f",
            gpu_health.recommended_batch_size_factor,
        )

    # ------------------------------------------------------------------
    # Step 6: Gate 3 — setup_gpu health check
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
                "confirmed_from": 419,
                "rerun": True,
                "crane_detection_rate": 0.0,
                "gpu_setup_status": gpu_status,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # ------------------------------------------------------------------
    # Step 7: Gate 4 — Load models
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
                    "confirmed_from": 419,
                    "rerun": True,
                    "crane_detection_rate": 0.0,
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            return

    # ------------------------------------------------------------------
    # Step 8: Wire CRANE (primary) + LLMConstraintExtractor (fallback)
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
    # Step 9: Load GSM8K questions
    # ------------------------------------------------------------------
    questions = load_gsm8k_questions(N_QUESTIONS)
    _log.info("Loaded %d GSM8K questions", len(questions))

    # ------------------------------------------------------------------
    # Step 10: Run benchmark inside watchdog
    # ------------------------------------------------------------------
    all_results: list[PrecisionStackResult] = []
    crane_hits: list[bool] = []  # per-question CRANE detection for FULL_STACK

    with ExperimentTimeoutWatchdog(
        EXP_ID, timeout_minutes=WATCHDOG_TIMEOUT_MINUTES,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        from scripts.experiment_template import BatchedInferenceRunner  # noqa: PLC0415

        for spec in MODEL_SPECS:
            model_name = spec["name"]
            model_obj = model_objects[model_name]

            _log.info("Running variants for model: %s", model_name)
            for variant in PipelineVariant:
                _log.info("  variant=%s", variant.value)

                baseline_correct = _count_baseline_correct(questions, model_obj)
                baseline_acc = baseline_correct / max(len(questions), 1)

                def _inference_fn(
                    q_text: str, _model_obj: object = model_obj
                ) -> str:
                    return _call_model(_model_obj, q_text)

                from scripts.experiment_template import BatchedInferenceRunner  # noqa: PLC0415

                bir = BatchedInferenceRunner(_inference_fn, batch_size=BATCH_SIZE)
                question_texts = [q["question"] for q in questions]
                ir_results = bir.run_batch(question_texts)

                n_correct = 0
                n_violations = 0
                n_repairs = 0
                n_improved = 0
                n_broken = 0
                variant_crane_hits: list[bool] = []

                for q_idx, (ir, q_dict) in enumerate(zip(ir_results, questions)):
                    gold = _extract_gsm8k_answer(q_dict["answer"])
                    if ir.timed_out or not ir.response:
                        if variant == PipelineVariant.FULL_STACK:
                            variant_crane_hits.append(False)
                        continue

                    response_before = ir.response

                    # _apply_variant_with_crane returns (response, n_viol, n_rep).
                    # For FULL_STACK, n_viol > 0 means CRANE detected violations.
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

                    if variant == PipelineVariant.FULL_STACK:
                        variant_crane_hits.append(viol_count > 0)

                    was_correct_before = _is_correct(response_before, gold)
                    was_correct_after = _is_correct(ir.response, gold)

                    if was_correct_after:
                        n_correct += 1
                    if rep_attempted > 0:
                        if not was_correct_before and was_correct_after:
                            n_improved += 1
                        elif was_correct_before and not was_correct_after:
                            n_broken += 1

                    # Checkpoint every CHECKPOINT_EVERY questions to preserve
                    # progress in case the watchdog fires mid-variant.
                    if (q_idx + 1) % CHECKPOINT_EVERY == 0:
                        tmpl.checkpoint_save(
                            {
                                "completed_models": [r.model_id for r in all_results],
                                "current_model": model_name,
                                "current_variant": variant.value,
                                "questions_done": q_idx + 1,
                            },
                            step=len(all_results) * len(questions) + q_idx + 1,
                        )

                stack_acc = n_correct / max(len(questions), 1)
                signed_improvement = compute_signed_improvement(baseline_acc, stack_acc)

                if variant == PipelineVariant.FULL_STACK:
                    crane_hits.extend(variant_crane_hits)

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

            # Checkpoint after each model.
            tmpl.checkpoint_save(
                {"completed_models": [r.model_id for r in all_results]},
                step=len(all_results),
            )

    # ------------------------------------------------------------------
    # Step 11: Build and write artifact
    # ------------------------------------------------------------------
    precision_artifact = build_exp427_artifact(
        all_results, inference_mode, crane_hits, rerun=True, confirmed_from=419
    )

    hr = precision_artifact.get("headline_result", {})
    if hr:
        _log.info(
            "HEADLINE: Gemma4-E4B-it FULL_STACK signed_improvement=%.4f "
            "honest_verdict=%s crane_detection_rate=%.3f",
            hr.get("signed_improvement", float("nan")),
            precision_artifact.get("honest_verdict", "unknown"),
            precision_artifact.get("crane_detection_rate", 0.0),
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
