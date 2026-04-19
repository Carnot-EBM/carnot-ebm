"""Helpers for Exp 527 Live 100q Precision v8 benchmark (RETRO-053 fix applied).

**Why this module exists (eighth RETRO-033 attempt):**
    Exp 514 (v7) was blocked because the conductor injected CARNOT_FORCE_LIVE='0'
    as a placeholder default.  apply_env_autofix() checked only presence, not
    truthiness, so '0' satisfied the check and the experiment wrote a gpu_required
    artifact.  Exp 526 fixed apply_env_autofix() to treat '0'/'false'/'' as falsy
    (RETRO-053 fix).  Exp 527 is the first live benchmark run with that fix active.

    This module provides the two new helpers the task spec requires:
    - ``load_jit_gated_model``      — re-exported from live_100q_v7_helpers (same logic)
    - ``build_precision_v8_artifact`` — build the v8 artifact dict from benchmark results

    All other helpers (wilson_ci, write_cot_pairs, _extract_answer, _is_correct,
    run_100q_benchmark, PrecisionBenchmarkResult) are re-exported from the v7 helpers
    module so callers only need to import from this module.

**FOVER format** (same as Exp 442/514 schema):
    Each entry: {question: str, cot_text: str, correct: bool, model_id: str}

Spec: REQ-BENCH-054, REQ-BENCH-055,
      SCENARIO-BENCH-071, SCENARIO-BENCH-072, SCENARIO-BENCH-073, SCENARIO-BENCH-074
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

# Re-export all v7 helpers so callers only need this module.
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
    "build_precision_v8_artifact",
    "load_jit_gated_model",
    "run_100q_benchmark",
    "wilson_ci",
    "write_cot_pairs",
]


def build_precision_v8_artifact(
    results: Dict,
    inference_mode: str,
    cot_pairs_path: Optional[str],
) -> Dict:
    """Build the v8 artifact dict from aggregated benchmark results.

    The artifact uses schema='carnot.live_precision.v8' and captures the RETRO-053
    fix status alongside the standard headline metrics.  This is the source of truth
    for whether RETRO-033 is closed: retro_033_closed=True iff inference_mode is
    'live_gpu' AND pipeline_accuracy > baseline_accuracy.

    Honest verdict logic (first match wins):
      1. 'retro_033_closed'   — live_gpu AND pipeline_accuracy > baseline_accuracy
      2. 'live_no_improvement' — live_gpu AND no improvement
      3. 'gpu_required'        — inference was deferred (no GPU or VRAM insufficient)

    Parameters
    ----------
    results : dict
        Aggregated benchmark result dict with keys:
          - n_questions (int): number of questions processed
          - baseline_accuracy (float): fraction correct without pipeline
          - pipeline_accuracy (float): fraction correct with pipeline
          - wilson_95ci_lower (float): Wilson 95% CI lower bound on pipeline accuracy
          - wilson_95ci_upper (float): Wilson 95% CI upper bound on pipeline accuracy
        Missing keys default to 0.0 / 0 to handle deferred (gpu_required) paths.
    inference_mode : str
        'live_gpu' when real GPU inference ran; 'gpu_required' when deferred.
    cot_pairs_path : str or None
        Path to the written CoT pairs file, or None if no pairs were written.

    Returns
    -------
    dict
        JSON-serializable artifact fragment.  Callers should merge this into the
        full ExperimentTemplate.build_result() output.

    Spec: REQ-BENCH-054, REQ-BENCH-055, SCENARIO-BENCH-074
    """
    n = results.get("n_questions", 0)
    baseline_acc = results.get("baseline_accuracy", 0.0)
    pipeline_acc = results.get("pipeline_accuracy", 0.0)
    signed_improvement = pipeline_acc - baseline_acc
    is_positive = signed_improvement > 0
    ci_lower = results.get("wilson_95ci_lower", 0.0)
    ci_upper = results.get("wilson_95ci_upper", 0.0)

    retro_033_closed = is_positive and inference_mode == "live_gpu"

    if retro_033_closed:
        honest_verdict = "retro_033_closed"
    elif inference_mode == "live_gpu":
        honest_verdict = "live_no_improvement"
    else:
        honest_verdict = "gpu_required"

    return {
        "schema": "carnot.live_precision.v8",
        "inference_mode": inference_mode,
        "n_questions": n,
        "baseline_accuracy": baseline_acc,
        "pipeline_accuracy": pipeline_acc,
        "signed_improvement": signed_improvement,
        "wilson_95ci_lower": ci_lower,
        "wilson_95ci_upper": ci_upper,
        "is_positive": is_positive,
        "retro_033_closed": retro_033_closed,
        "cot_pairs_written": cot_pairs_path,
        "env_autofix_applied": True,
        "honest_verdict": honest_verdict,
    }
