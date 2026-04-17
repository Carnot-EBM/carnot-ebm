"""MicroPrecisionResult and build_micro_precision_artifact for Exp 439.

**Researcher summary (Exp 439):**
    Exps 427/368/379 produced scaffolding_only results because 200q × 5 variants × 2 models
    = 2000 LLM calls exceeded the 45-minute watchdog budget.  Exp 439 fixes this by scoping
    down to 50q × 3 variants × 2 models = 300 calls ≈ 45 min — exactly fitting the budget.

    The three variants tested here are a simplified ablation designed to answer the
    key research question in the shortest possible live GPU window:
    - BASELINE: model output with no verification or repair.
    - CRANE_ONLY: add CRANEExtractionGate verification and one-shot repair.
    - FULL_STACK: add JitRLConstraintMemory threshold adaptation + energy-based gate.

    This module provides two things:
    1. ``MicroPrecisionResult`` — per-(model, variant) structured result.
    2. ``build_micro_precision_artifact`` — assembles a carnot.precision_micro.v1
       artifact from a list of MicroPrecisionResult objects.

**Why a separate module from precision_benchmark.py?**
    The existing PrecisionStackResult and PipelineVariant (precision_benchmark.py) use
    the 5-variant ablation from Exp 340.  The 3-variant micro-benchmark here has a
    different variant set (CRANE_ONLY is new; CONFIDENCE_ONLY/ADAPTIVE/VERGE are dropped
    to reduce scope).  Keeping these in separate modules avoids polluting the 5-variant
    schema with micro-benchmark-specific fields.

**Verdict semantics:**
    - ``'live_improvement'``    — all results are live_gpu AND best signed_improvement > 0
    - ``'live_no_improvement'`` — all results are live_gpu AND best signed_improvement <= 0
    - ``'blocked'``             — results empty OR any result is not live_gpu

    The inference_mode field in each result must be 'live_gpu' for any non-blocked verdict.
    Simulated results never produce a live verdict — this is the core honesty invariant.

Spec: REQ-BENCH-009, SCENARIO-BENCH-025, SCENARIO-BENCH-026 (Exp 439)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# MicroPrecisionResult
# ---------------------------------------------------------------------------


@dataclass
class MicroPrecisionResult:
    """Per-variant, per-model result from the live precision micro-benchmark.

    **Detailed explanation for engineers:**
        One MicroPrecisionResult is produced for each (model_id, variant) pair
        evaluated in Exp 439.  With 2 models × 3 variants = 6 results total.

        ``signed_improvement`` is ``variant_accuracy - baseline_accuracy``.
        Negative values are honest regression signals — never clamp or abs() this.
        The ``baseline_accuracy`` field is always the BASELINE variant accuracy for
        the same model, so ``signed_improvement`` is always comparable across models.

        ``crane_detection_rate`` is 0.0 for BASELINE (CRANE not run) and the
        fraction of questions where CRANEExtractionGate found at least one violation
        for CRANE_ONLY and FULL_STACK.  A rate of 0.0 for CRANE_ONLY means CRANE
        never fired; a rate of 1.0 means CRANE detected violations on every question.

        ``inference_mode`` must be 'live_gpu' for results that should be used in
        headline reporting.  'blocked' is used when the gate chain prevented inference.

    Fields
    ------
    model_id : str
        Human-readable model name (e.g. 'Gemma4-E4B-it', 'Qwen3.5-0.8B').
    variant : str
        Pipeline variant name: 'baseline', 'crane_only', or 'full_stack'.
    n_questions : int
        Number of GSM8K questions evaluated in this run.
    baseline_accuracy : float
        Accuracy of the BASELINE variant for this model.  Same value for all
        variants of the same model (the shared denominator).
    variant_accuracy : float
        Accuracy of THIS variant for this model.  Equal to baseline_accuracy
        when variant='baseline'.
    signed_improvement : float
        variant_accuracy - baseline_accuracy.  Positive = improvement, negative
        = regression.  Always 0.0 for the BASELINE variant itself.
    crane_detection_rate : float
        Fraction of questions where CRANE found at least one arithmetic violation.
        0.0 for BASELINE (CRANE not invoked).
    inference_mode : str
        'live_gpu' when model inference used real GPU hardware.  'blocked' when
        the gate chain prevented inference from running.

    Spec: REQ-BENCH-009, SCENARIO-BENCH-025
    """

    model_id: str
    variant: str
    n_questions: int
    baseline_accuracy: float
    variant_accuracy: float
    signed_improvement: float
    crane_detection_rate: float
    inference_mode: str


# ---------------------------------------------------------------------------
# _result_to_dict — internal serialization helper
# ---------------------------------------------------------------------------


def _result_to_dict(r: MicroPrecisionResult) -> dict[str, Any]:
    """Serialize a MicroPrecisionResult to a JSON-safe dict.

    All float fields are preserved as floats (never converted to int).
    The dict schema exactly mirrors the MicroPrecisionResult dataclass fields,
    so downstream tooling can round-trip the data without schema divergence.
    """
    return {
        "model_id": r.model_id,
        "variant": r.variant,
        "n_questions": r.n_questions,
        "baseline_accuracy": r.baseline_accuracy,
        "variant_accuracy": r.variant_accuracy,
        "signed_improvement": r.signed_improvement,
        "crane_detection_rate": r.crane_detection_rate,
        "inference_mode": r.inference_mode,
    }


# ---------------------------------------------------------------------------
# build_micro_precision_artifact
# ---------------------------------------------------------------------------


def build_micro_precision_artifact(results: list[MicroPrecisionResult]) -> dict[str, Any]:
    """Build a carnot.precision_micro.v1 artifact from micro-benchmark results.

    **Detailed explanation for engineers:**
        Assembles the JSON artifact from a flat list of MicroPrecisionResult objects
        (one per model × variant combination) and derives the honest_verdict:

        Verdict rules (first match wins):
        1. ``'blocked'``             — results list is empty (gate chain blocked run).
        2. ``'blocked'``             — any result has inference_mode != 'live_gpu'
                                       (simulated data must never become a headline claim).
        3. ``'live_improvement'``    — best signed_improvement among non-baseline variants > 0.
        4. ``'live_no_improvement'`` — best signed_improvement <= 0 (no improvement or regression).

        The headline_result is the single MicroPrecisionResult with the highest
        signed_improvement among non-baseline variants.  For the BASELINE variant
        itself (signed_improvement always 0.0), it is never selected as the headline
        unless no non-baseline results are present.

        The per_model_results list includes ALL results (all variants, all models)
        so downstream analysis can perform its own cuts.

    Parameters
    ----------
    results : list[MicroPrecisionResult]
        All (model, variant) results from the micro-benchmark.
        May be empty when the experiment was blocked before inference started.

    Returns
    -------
    dict
        JSON-serializable artifact with:
        - ``schema``            : 'carnot.precision_micro.v1'
        - ``honest_verdict``    : 'live_improvement', 'live_no_improvement', or 'blocked'
        - ``headline_result``   : serialized best non-baseline MicroPrecisionResult, or None
        - ``inference_mode``    : 'live_gpu' or 'blocked'
        - ``per_model_results`` : list of all serialized MicroPrecisionResult objects

    Spec: REQ-BENCH-009, SCENARIO-BENCH-026
    """
    # Rule 1: empty results → blocked (gate chain did not allow inference)
    if not results:
        return {
            "schema": "carnot.precision_micro.v1",
            "honest_verdict": "blocked",
            "headline_result": None,
            "inference_mode": "blocked",
            "per_model_results": [],
        }

    # Rule 2: any non-live result → blocked (simulated data must not become a headline)
    all_live = all(r.inference_mode == "live_gpu" for r in results)
    if not all_live:
        return {
            "schema": "carnot.precision_micro.v1",
            "honest_verdict": "blocked",
            "headline_result": None,
            "inference_mode": "blocked",
            "per_model_results": [_result_to_dict(r) for r in results],
        }

    # Rules 3 + 4: all live — find the best non-baseline signed_improvement
    non_baseline = [r for r in results if r.variant != "baseline"]
    # When no non-baseline results present (degenerate case), fall back to all results
    candidate_pool = non_baseline if non_baseline else results
    best = max(candidate_pool, key=lambda r: r.signed_improvement)

    if best.signed_improvement > 0.0:
        honest_verdict = "live_improvement"
    else:
        honest_verdict = "live_no_improvement"

    return {
        "schema": "carnot.precision_micro.v1",
        "honest_verdict": honest_verdict,
        "headline_result": _result_to_dict(best),
        "inference_mode": "live_gpu",
        "per_model_results": [_result_to_dict(r) for r in results],
    }


__all__ = [
    "MicroPrecisionResult",
    "build_micro_precision_artifact",
]
