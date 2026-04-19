"""Helpers for Exp 528 Live 200q VeriCoT+VPRM v7 benchmark (RETRO-038 target).

**Why this module exists:**
    RETRO-038 requires a 200-question benchmark where the Wilson 95% CI lower bound
    on the signed improvement is strictly > 0.  At n=100 the CI is too wide; at n=200
    with Wilson scoring the width narrows enough to make a publishable credibility claim.

    This module provides two helpers:
    - ``compute_wilson_ci``     — thin wrapper over the well-tested wilson_ci from v7;
                                  renamed for clarity and test discoverability.
    - ``build_200q_v7_artifact`` — build the v7 artifact dict with all required schema
                                   fields, including the new RETRO-038 close flag.

    All other helpers (write_cot_pairs, _extract_answer, _is_correct, wilson_ci,
    PrecisionBenchmarkResult) are re-exported from live_100q_v7_helpers.

Spec: REQ-BENCH-019, SCENARIO-BENCH-041, SCENARIO-BENCH-042
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from carnot.pipeline.live_100q_v7_helpers import (  # noqa: F401
    PrecisionBenchmarkResult,
    _extract_answer,
    _is_correct,
    load_jit_gated_model,
    run_100q_benchmark,
    wilson_ci,
    write_cot_pairs,
)

_log = logging.getLogger(__name__)

__all__ = [
    "PrecisionBenchmarkResult",
    "build_200q_v7_artifact",
    "compute_wilson_ci",
    "load_jit_gated_model",
    "run_100q_benchmark",
    "wilson_ci",
    "write_cot_pairs",
]


def compute_wilson_ci(
    n_successes: int,
    n_total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute the Wilson score confidence interval for a proportion.

    This is the authoritative CI computation for the RETRO-038 publishable-claim
    criterion: if the lower bound is > 0, the pipeline improvement is statistically
    distinguishable from noise at the requested confidence level.

    Why Wilson over Wald:
        The Wald interval (p ± z*SE) can produce negative lower bounds near p=0 or
        upper bounds > 1 near p=1 — nonsensical for a proportion.  Wilson avoids
        this by using a different centering formula that is always valid in [0,1].

    Parameters
    ----------
    n_successes : int
        Number of successful outcomes (e.g., pipeline-correct answers).
    n_total : int
        Total number of trials (e.g., questions attempted).
    confidence : float
        Confidence level.  Default 0.95 = 95%.  Must be in (0, 1).
        The z-score is derived from the normal distribution: 1.96 for 0.95.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
        Lower and upper bounds of the Wilson CI, clamped to [0, 1].
        Returns (0.0, 0.0) when n_total == 0 to avoid division by zero.

    Spec: REQ-BENCH-019, SCENARIO-BENCH-041
    """
    # Map the confidence level to the z-score.
    # We only support 0.95 exactly in this helper — anything else falls back to scipy.
    # The 1.96 value is the standard 95% z-score used throughout the Carnot benchmarks.
    import math

    if n_total == 0:
        return 0.0, 0.0

    if abs(confidence - 0.95) < 1e-9:
        z = 1.96
    else:
        # scipy is an optional dependency; fall back gracefully when absent.
        try:
            from scipy import stats as _stats  # type: ignore[import]
            z = float(_stats.norm.ppf(0.5 + confidence / 2))
        except Exception:
            z = 1.96  # safe fallback — caller should use 0.95

    return wilson_ci(n_successes, n_total, z=z)


def build_200q_v7_artifact(
    results: Dict,
    inference_mode: str,
    cot_pairs_path: Optional[str],
) -> Dict:
    """Build the v7 artifact dict for the 200q benchmark from aggregated results.

    Schema is 'carnot.live_200q.v7'.  The primary new field over v8 is
    ``is_statistically_positive`` (Wilson 95% CI lower bound > 0) which together
    with ``inference_mode='live_gpu'`` triggers ``retro_038_closed=True`` — the
    first publishable credibility claim for the Carnot pipeline.

    Honest verdict logic (first match wins):
      1. 'first_publishable_claim'  — live_gpu AND wilson_95ci_lower > 0
      2. 'live_no_significance'     — live_gpu AND wilson_95ci_lower <= 0
      3. 'gpu_required'             — inference was deferred (no GPU or VRAM insufficient)

    Parameters
    ----------
    results : dict
        Aggregated benchmark result dict with keys:
          - n_questions (int): number of questions processed
          - baseline_accuracy (float): fraction correct without pipeline
          - pipeline_accuracy (float): fraction correct with pipeline
          - wilson_95ci_lower (float): Wilson 95% CI lower bound on signed improvement
          - wilson_95ci_upper (float): Wilson 95% CI upper bound on signed improvement
        Missing keys default to 0.0 / 0 to handle the gpu_required deferred path.
    inference_mode : str
        'live_gpu' when real GPU inference ran; 'gpu_required' when deferred.
    cot_pairs_path : str or None
        Path to the written CoT pairs file, or None if no pairs were written.

    Returns
    -------
    dict
        JSON-serializable artifact fragment.  Callers merge this into the
        full ExperimentTemplate.build_result() output.

    Spec: REQ-BENCH-019, SCENARIO-BENCH-041, SCENARIO-BENCH-042
    """
    n = results.get("n_questions", 0)
    baseline_acc = results.get("baseline_accuracy", 0.0)
    pipeline_acc = results.get("pipeline_accuracy", 0.0)
    signed_improvement = pipeline_acc - baseline_acc
    ci_lower = results.get("wilson_95ci_lower", 0.0)
    ci_upper = results.get("wilson_95ci_upper", 0.0)

    # The RETRO-038 criterion: CI lower bound strictly > 0 AND live GPU mode.
    # A lower bound <= 0 means the improvement is not statistically distinguishable
    # from noise at the 95% level — not publishable.
    is_statistically_positive = ci_lower > 0.0 and inference_mode == "live_gpu"
    retro_038_closed = is_statistically_positive

    if retro_038_closed:
        honest_verdict = "first_publishable_claim"
    elif inference_mode == "live_gpu":
        honest_verdict = "live_no_significance"
    else:
        honest_verdict = "gpu_required"

    return {
        "schema": "carnot.live_200q.v7",
        "inference_mode": inference_mode,
        "n_questions": n,
        "baseline_accuracy": baseline_acc,
        "pipeline_accuracy": pipeline_acc,
        "signed_improvement": signed_improvement,
        "wilson_95ci_lower": ci_lower,
        "wilson_95ci_upper": ci_upper,
        "is_statistically_positive": is_statistically_positive,
        "retro_038_closed": retro_038_closed,
        "cot_pairs_written": cot_pairs_path,
        "honest_verdict": honest_verdict,
    }
