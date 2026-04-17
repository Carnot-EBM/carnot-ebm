"""MicroHumanEvalResult and build_micro_humaneval_artifact for Exp 440.

**Researcher summary (Exp 440):**
    Exps 369/380/411/420/428 all produced scaffolding_only or blocked artifacts.
    Exp 440 fixes scope: 50 problems × 2 models = 100 LLM calls ≈ 15–20 min,
    well inside the 45-minute watchdog.  Each 50-problem run is split into two
    25-problem batches by LongRunBenchmarkExecutor to enable partial checkpointing.

    Why code verification specifically (not arithmetic)?  CodeExtractor uses
    execution — it literally runs the code and checks if tests pass.  This gives
    VerifyRepairPipeline real signal on genuinely wrong code.  Instruction-tuned
    models produce valid Python, so the extractor has concrete failures to work
    with.  ArithmeticExtractor returned 0 violations on IT models (Exp 328) because
    the regex pattern matching found nothing to grab.

    This module provides:
    1. ``MicroHumanEvalResult`` — per-model summary from a 50-problem run.
    2. ``build_micro_humaneval_artifact`` — assembles a carnot.humaneval_micro.v1
       artifact from a list of MicroHumanEvalResult objects.

**Verdict semantics:**
    - ``'code_verification_positive'`` — all results are live_gpu AND at least one
      model shows signed_improvement > 0.
    - ``'code_no_improvement'``        — all results are live_gpu AND no model shows
      signed_improvement > 0.
    - ``'blocked'``                    — results list is empty OR any result has
      inference_mode != 'live_gpu'.

    The inference_mode field in each result must be 'live_gpu' for any non-blocked
    verdict.  Simulated results never produce a live headline claim — this is the
    core honesty invariant from Exp 369/428.

Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# MicroHumanEvalResult
# ---------------------------------------------------------------------------


@dataclass
class MicroHumanEvalResult:
    """Per-model result from the live HumanEval micro-benchmark.

    **Detailed explanation for engineers:**
        One MicroHumanEvalResult is produced for each model evaluated in Exp 440.
        With 2 models × 50 problems = 100 LLM calls total.

        ``pass_at_1_before`` is the fraction of problems that passed all official
        tests on the FIRST code generation — before VerifyRepairPipeline is applied.
        This is the raw model baseline.

        ``pass_at_1_after`` is the fraction passing after the verify-repair loop.
        The signed delta ``signed_improvement = pass_at_1_after - pass_at_1_before``
        is the headline metric: positive means Carnot helped, negative means it hurt.

        ``pbt_bugs_found`` counts solutions that passed official tests but failed
        property-based testing (random argument fuzzing + idempotency checks).
        These are latent bugs not caught by the official HumanEval test cases.

        ``inference_mode`` must be 'live_gpu' for results used in headline reporting.
        'blocked' is only used when the gate chain prevented inference from running.

    Fields
    ------
    model_id : str
        Human-readable model name (e.g. 'google/gemma-4-E4B-it', 'Qwen/Qwen2.5-0.5B').
    n_problems : int
        Number of HumanEval problems evaluated in this run.
    pass_at_1_before : float
        Fraction of problems passing official tests on first generation (before repair).
    pass_at_1_after : float
        Fraction of problems passing official tests after verify-repair loop.
    signed_improvement : float
        pass_at_1_after - pass_at_1_before.  Positive = improvement, negative = regression.
    pbt_bugs_found : int
        Count of solutions that passed official tests but failed PBT probes.
    inference_mode : str
        'live_gpu' when model inference used real GPU hardware.  'blocked' when
        the gate chain prevented inference from running.

    Spec: REQ-BENCH-010, SCENARIO-BENCH-027
    """

    model_id: str
    n_problems: int
    pass_at_1_before: float
    pass_at_1_after: float
    signed_improvement: float
    pbt_bugs_found: int
    inference_mode: str


# ---------------------------------------------------------------------------
# _result_to_dict — internal serialization helper
# ---------------------------------------------------------------------------


def _result_to_dict(r: MicroHumanEvalResult) -> dict[str, Any]:
    """Serialize a MicroHumanEvalResult to a JSON-safe dict.

    All float fields are preserved as floats.  The dict schema exactly mirrors
    the MicroHumanEvalResult dataclass fields for round-trip fidelity.
    """
    return {
        "model_id": r.model_id,
        "n_problems": r.n_problems,
        "pass_at_1_before": r.pass_at_1_before,
        "pass_at_1_after": r.pass_at_1_after,
        "signed_improvement": r.signed_improvement,
        "pbt_bugs_found": r.pbt_bugs_found,
        "inference_mode": r.inference_mode,
    }


# ---------------------------------------------------------------------------
# build_micro_humaneval_artifact
# ---------------------------------------------------------------------------


def build_micro_humaneval_artifact(
    results: list[MicroHumanEvalResult],
) -> dict[str, Any]:
    """Build a carnot.humaneval_micro.v1 artifact from micro-benchmark results.

    **Detailed explanation for engineers:**
        Assembles the JSON artifact from a flat list of MicroHumanEvalResult
        objects (one per model) and derives the honest_verdict.

        Verdict rules (first match wins):
        1. ``'blocked'``                    — results list is empty (gate chain blocked run).
        2. ``'blocked'``                    — any result has inference_mode != 'live_gpu'
                                             (simulated data must never become a headline
                                             claim — this is the core honesty invariant).
        3. ``'code_verification_positive'`` — at least one model shows signed_improvement > 0.
        4. ``'code_no_improvement'``        — all models show signed_improvement <= 0.

        The ``headline_result`` is the single MicroHumanEvalResult with the highest
        signed_improvement.  None when the artifact is blocked.

    Parameters
    ----------
    results : list[MicroHumanEvalResult]
        Per-model results.  May be empty when the experiment was blocked before
        inference started.

    Returns
    -------
    dict
        JSON-serializable artifact with:
        - ``schema``          : 'carnot.humaneval_micro.v1'
        - ``honest_verdict``  : 'code_verification_positive', 'code_no_improvement',
                                or 'blocked'
        - ``headline_result`` : serialized best MicroHumanEvalResult, or None
        - ``inference_mode``  : 'live_gpu' or 'blocked'
        - ``per_model_results``: list of all serialized MicroHumanEvalResult objects

    Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028
    """
    # Rule 1: empty results → blocked (gate chain did not allow inference)
    if not results:
        return {
            "humaneval_micro_schema": "carnot.humaneval_micro.v1",
            "honest_verdict": "blocked",
            "headline_result": None,
            "inference_mode": "blocked",
            "per_model_results": [],
        }

    # Rule 2: any non-live result → blocked (simulated data must not become a headline)
    all_live = all(r.inference_mode == "live_gpu" for r in results)
    if not all_live:
        return {
            "humaneval_micro_schema": "carnot.humaneval_micro.v1",
            "honest_verdict": "blocked",
            "headline_result": None,
            "inference_mode": "blocked",
            "per_model_results": [_result_to_dict(r) for r in results],
        }

    # Rules 3 + 4: all results are live — find the best signed_improvement
    best = max(results, key=lambda r: r.signed_improvement)

    if best.signed_improvement > 0.0:
        honest_verdict = "code_verification_positive"
    else:
        honest_verdict = "code_no_improvement"

    return {
        "humaneval_micro_schema": "carnot.humaneval_micro.v1",
        "honest_verdict": honest_verdict,
        "headline_result": _result_to_dict(best),
        "inference_mode": "live_gpu",
        "per_model_results": [_result_to_dict(r) for r in results],
    }


__all__ = [
    "MicroHumanEvalResult",
    "build_micro_humaneval_artifact",
]
