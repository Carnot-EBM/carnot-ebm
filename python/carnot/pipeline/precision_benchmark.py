"""Precision stack benchmark data types and artifact builder (Exp 340).

**Researcher summary:**
    Experiments 332-336 implemented the combined precision stack (confidence-weighted
    repair, model-adaptive thresholds, VERGE iterative Z3 refinement, CoTCircuitVerifier)
    but measured those components only SYNTHETICALLY.  Exp 328 ran the live GPU benchmark
    but used the OLD pipeline (pre-332).

    This module provides the shared data types and artifact builder used by Exp 340 to
    answer the first honest measurement: does the combined precision stack actually help
    on real LLM output from instruction-tuned models (Gemma4-E4B-it, Qwen3.5-0.8B)?

    The five ablation conditions (PipelineVariant) let us see WHERE the improvement
    (or regression) comes from: adding each component one at a time, so we know which
    parts help and which parts hurt on live data.

**Detailed explanation for engineers:**
    This module is intentionally kept lightweight — pure data types and a builder
    function, no I/O, no LLM calls.  That makes it fully testable in CI without GPU.

    PipelineVariant:
        Five conditions tested in order (additive stack):
        - BASELINE: ArithmeticExtractor only (pre-Exp-332 pipeline)
        - CONFIDENCE_ONLY: + ConfidenceWeightedRepair (min_confidence=0.8)
        - CONFIDENCE_ADAPTIVE: + ModelAdaptiveThresholds (auto-disable high-FP types)
        - CONFIDENCE_ADAPTIVE_VERGE: + VergeRefiner (targeted Z3-guided step repair)
        - FULL_STACK: + CoTCircuitVerifier (all four components active)

    compute_signed_improvement:
        Returns stack_acc − baseline_acc WITHOUT clamping.  Negative values are honest
        signals that the stack made things WORSE — preserving these is critical for
        research integrity.  The Exp 328 baseline showed verify-repair was harmful
        (negative signed improvement) — we need to see the full picture.

    build_precision_benchmark_artifact:
        Builds the JSON artifact for Exp 340 results.  Extracts the FULL_STACK result
        for Gemma4-E4B-it as the headline result (the primary research question: did the
        combined stack help on the harder, more capable instruction-tuned model?).

    CI-safe simulated mode:
        When CARNOT_FORCE_LIVE is not set, the experiment runs with synthetic answers
        rather than loading real GPU models.  Every result produced in this mode has
        inference_mode="simulated" and the artifact carries honest_verdict="simulated_only"
        so downstream tooling never confuses synthetic results with live GPU results.

Spec: REQ-BENCH-003, SCENARIO-BENCH-007, SCENARIO-BENCH-008, SCENARIO-BENCH-009
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


# ---------------------------------------------------------------------------
# PipelineVariant enum
# ---------------------------------------------------------------------------


class PipelineVariant(str, enum.Enum):
    """The five ablation conditions for the precision stack benchmark.

    **Detailed explanation for engineers:**
        Each variant is an ADDITIVE layer on top of the previous one.  This
        lets us attribute improvements (or regressions) to individual components
        rather than measuring only the combined effect.

        BASELINE:
            ArithmeticExtractor only — the pre-Exp-332 pipeline with no confidence
            weighting.  This is the control condition.  Exp 328 measured this on
            live GPU and found verify-repair was harmful.

        CONFIDENCE_ONLY:
            Adds ConfidenceWeightedRepair (Exp 332, min_confidence=0.8).  The 86.7%
            synthetic FP reduction should translate to fewer repair regressions.

        CONFIDENCE_ADAPTIVE:
            Adds ModelAdaptiveThresholds (Exp 333).  For models whose constraint types
            have high observed FP rates (e.g. NL2Z3 range checks on Qwen3.5-0.8B),
            this layer auto-disables those constraint types.

        CONFIDENCE_ADAPTIVE_VERGE:
            Adds VergeRefiner (Exp 334).  Instead of whole-response repair, targets
            the specific reasoning step that Z3 found inconsistent.

        FULL_STACK:
            Adds CoTCircuitVerifier (Exp 336).  Catches structural dependency-chain
            errors that regex and Z3 miss.  This is the headline condition.

    Spec: REQ-BENCH-003
    """

    BASELINE = "baseline"
    CONFIDENCE_ONLY = "confidence_only"
    CONFIDENCE_ADAPTIVE = "confidence_adaptive"
    CONFIDENCE_ADAPTIVE_VERGE = "confidence_adaptive_verge"
    FULL_STACK = "full_stack"


# ---------------------------------------------------------------------------
# compute_signed_improvement
# ---------------------------------------------------------------------------


def compute_signed_improvement(baseline_acc: float, stack_acc: float) -> float:
    """Return the honest signed improvement of the precision stack over baseline.

    **Detailed explanation for engineers:**
        This is simply ``stack_acc - baseline_acc`` — no clamping, no absolute value.

        Preserving the sign is critical for research integrity:
        - Positive: the precision stack improved accuracy on this model/variant.
        - Negative: the precision stack made accuracy WORSE.  This is the result
          Exp 328 measured for verify-repair on the old pipeline — we MUST see it
          if it's still happening after the Exp 332-336 fixes.

        Callers that want a "was helpful" boolean can simply test
        ``compute_signed_improvement(...) > 0``.

    Args:
        baseline_acc: Accuracy of the BASELINE pipeline variant (ArithmeticExtractor only).
        stack_acc:    Accuracy of the precision-stack pipeline variant being compared.

    Returns:
        Float signed improvement (stack_acc − baseline_acc).  No clamping.

    Spec: REQ-BENCH-003, SCENARIO-BENCH-007
    """
    return stack_acc - baseline_acc


# ---------------------------------------------------------------------------
# PrecisionStackResult dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PrecisionStackResult:
    """Per-variant benchmark result for one (model, pipeline_variant) pair.

    **Detailed explanation for engineers:**
        One PrecisionStackResult is produced for each combination of model_id and
        PipelineVariant across the 200-question GSM8K subset.

        Fields:
            model_id:                Model identifier string, e.g. "Gemma4-E4B-it".
            n_questions:             Number of questions evaluated (target: 200).
            baseline_accuracy:       Accuracy of the BASELINE variant for this model.
                                     Constant across variants for the same model — used
                                     for consistent signed_improvement computation.
            precision_stack_accuracy: Accuracy of this pipeline_variant for this model.
            signed_improvement:      compute_signed_improvement(baseline, stack).
                                     Honest signed delta — may be negative.
            pipeline_variant:        Which of the five ablation conditions this result
                                     represents (PipelineVariant enum member).
            inference_mode:          "live_gpu" when running on real hardware with
                                     CARNOT_FORCE_LIVE=1; "simulated" in CI mode.
            n_violations_found:      How many violations were extracted from responses.
            n_repairs_attempted:     How many times repair was triggered.
            n_repairs_improved:      Repairs that increased accuracy (correct after repair).
            n_repairs_broken:        Repairs that decreased accuracy (broke a correct answer).

    Spec: REQ-BENCH-003, SCENARIO-BENCH-007
    """

    model_id: str
    n_questions: int
    baseline_accuracy: float
    precision_stack_accuracy: float
    signed_improvement: float
    pipeline_variant: PipelineVariant
    inference_mode: str
    n_violations_found: int = 0
    n_repairs_attempted: int = 0
    n_repairs_improved: int = 0
    n_repairs_broken: int = 0


# ---------------------------------------------------------------------------
# build_precision_benchmark_artifact
# ---------------------------------------------------------------------------

#: Target headline model — the harder instruction-tuned model (Gemma 4B).
_HEADLINE_MODEL = "Gemma4-E4B-it"


def build_precision_benchmark_artifact(
    results: list[PrecisionStackResult],
) -> dict[str, Any]:
    """Build the Exp 340 precision benchmark artifact from a list of results.

    **Detailed explanation for engineers:**
        Aggregates all per-variant results into the JSON artifact structure that
        Exp 340 writes to results/experiment_340_live_precision_benchmark.json.

        Headline result selection:
            The primary research question is whether the FULL_STACK variant helped
            on Gemma4-E4B-it (the harder, more capable instruction-tuned model).
            We look for the result with:
                pipeline_variant == PipelineVariant.FULL_STACK
                model_id == "Gemma4-E4B-it"
            If found and signed_improvement > 0, we set headline_label to
            "first_positive_live_it_result" — this is Carnot's first credible
            positive result on instruction-tuned models if it occurs.

        inference_mode resolution:
            If all results share the same inference_mode, that value is used.
            If the list is empty or modes are mixed, "unknown" is used.

        CI-safe honest_verdict:
            When all results are "simulated", the artifact carries
            honest_verdict="simulated_only" so consumers know these numbers are
            NOT from real GPU inference.

        all_results list:
            Every PrecisionStackResult is serialized via dataclasses.asdict()
            so the artifact is a complete record of all five variants × N models.

    Args:
        results: List of PrecisionStackResult objects from the benchmark run.

    Returns:
        Dict with schema, headline_result, inference_mode, honest_verdict,
        and all_results.

    Spec: REQ-BENCH-003, SCENARIO-BENCH-008, SCENARIO-BENCH-009
    """
    # --- Determine headline result (FULL_STACK on Gemma4-E4B-it) ---
    headline_result: dict[str, Any] = {}
    for r in results:
        if r.pipeline_variant == PipelineVariant.FULL_STACK and r.model_id == _HEADLINE_MODEL:
            headline_result = dataclasses.asdict(r)
            # Convert the enum value back to a string for JSON serialization.
            headline_result["pipeline_variant"] = r.pipeline_variant.value
            if r.signed_improvement > 0:
                headline_result["headline_label"] = "first_positive_live_it_result"
            break

    # --- Resolve inference_mode ---
    modes = {r.inference_mode for r in results}
    if len(modes) == 1:
        inference_mode = modes.pop()
    elif len(modes) == 0:
        inference_mode = "unknown"
    else:
        # Mixed modes — surface this explicitly so callers notice.
        inference_mode = "mixed"

    # --- honest_verdict for CI-safe runs ---
    honest_verdict: str | None = None
    if inference_mode == "simulated":
        honest_verdict = "simulated_only"

    # --- Serialize all results ---
    all_results_serialized = []
    for r in results:
        d = dataclasses.asdict(r)
        d["pipeline_variant"] = r.pipeline_variant.value
        all_results_serialized.append(d)

    artifact: dict[str, Any] = {
        "precision_schema": "carnot.precision_benchmark.v1",
        "headline_result": headline_result,
        "inference_mode": inference_mode,
        "all_results": all_results_serialized,
    }
    if honest_verdict is not None:
        artifact["honest_verdict"] = honest_verdict

    return artifact


__all__ = [
    "PipelineVariant",
    "PrecisionStackResult",
    "build_precision_benchmark_artifact",
    "compute_signed_improvement",
]
