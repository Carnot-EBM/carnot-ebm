"""Adversarial v5 result helpers for Exp 516 (RETRO-039 GSM-Symbolic adversarial benchmark).

**Researcher summary:**
    RETRO-039 has missed four consecutive milestones.  The thesis: Carnot's improvement should
    be LARGER under adversarial conditions because the Ising constraint verifier is independent
    of surface form.  A symbolic substitution that changes the phrasing while preserving
    arithmetic constraints does not fool the Ising sampler.

    This module provides two helpers used by Exp 516:

    1. ``compute_robustness_delta`` — computes the scalar that determines whether Carnot
       degrades LESS than the raw baseline under distractor injection.  A positive value
       confirms the RETRO-039 thesis.

    2. ``build_adversarial_v5_artifact`` — assembles the JSON artifact for Exp 516 with
       all required schema fields and an honest verdict.

**Detailed explanation for engineers:**
    The four-condition design (standard × {baseline, pipeline} crossed with adversarial ×
    {baseline, pipeline}) lets us compute two accuracy drops:

        baseline_drop = baseline_standard_accuracy - baseline_adversarial_accuracy
        pipeline_drop = pipeline_standard_accuracy - pipeline_adversarial_accuracy

    If the pipeline (Carnot) degrades LESS than the raw LLM under adversarial conditions,
    then ``pipeline_drop < baseline_drop``, which means
    ``robustness_delta = baseline_drop - pipeline_drop > 0``.

    A positive robustness_delta is the primary RETRO-039 credibility claim: Carnot's
    Ising verifier, which operates only on extracted constraint terms, is not distracted
    by irrelevant context sentences.

    honest_verdict logic:
        'thesis_confirmed'  — robustness_delta > 0 AND inference_mode == 'live_gpu'
        'thesis_rejected'   — robustness_delta <= 0 AND inference_mode == 'live_gpu'
        'gpu_required'      — inference_mode != 'live_gpu'

Spec: REQ-BENCH-052, REQ-BENCH-053,
      SCENARIO-BENCH-037, SCENARIO-BENCH-038
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "compute_robustness_delta",
    "build_adversarial_v5_artifact",
]


def compute_robustness_delta(
    baseline_std: float,
    baseline_adv: float,
    pipeline_std: float,
    pipeline_adv: float,
) -> float:
    """Compute how much MORE robust Carnot is compared to the raw baseline.

    **Detailed explanation for engineers:**
        We compare two accuracy drops under adversarial distractor injection:

            baseline_drop = baseline_std - baseline_adv
            pipeline_drop = pipeline_std - pipeline_adv

        robustness_delta = baseline_drop - pipeline_drop

        Positive result: Carnot's adversarial drop is SMALLER than the raw LLM's
        drop, confirming the RETRO-039 thesis.

        Zero or negative result: Carnot degrades at least as much as the baseline —
        thesis not confirmed for this model/condition.

        No clamping: negative values are honest research findings.

    Parameters
    ----------
    baseline_std : float
        Raw LLM accuracy on standard (non-adversarial) questions.
    baseline_adv : float
        Raw LLM accuracy on adversarial questions (distractor injected).
    pipeline_std : float
        Carnot pipeline accuracy on standard questions.
    pipeline_adv : float
        Carnot pipeline accuracy on adversarial questions.

    Returns
    -------
    float
        Positive = Carnot degrades less than the baseline (thesis confirmed).
        Zero or negative = thesis not confirmed for this condition.

    Spec: REQ-BENCH-052, SCENARIO-BENCH-037
    """
    baseline_drop = baseline_std - baseline_adv
    pipeline_drop = pipeline_std - pipeline_adv
    return baseline_drop - pipeline_drop


def build_adversarial_v5_artifact(
    results: dict[str, Any],
    inference_mode: str,
) -> dict[str, Any]:
    """Build the JSON artifact for the Exp 516 adversarial v5 benchmark.

    **Detailed explanation for engineers:**
        Assembles the structured dict that Exp 516 writes to
        ``results/experiment_516_gsm_symbolic_adversarial_v5.json``.

        Required keys in *results*:
            baseline_standard_accuracy  (float) — raw LLM, standard questions
            baseline_adversarial_accuracy (float) — raw LLM, adversarial questions
            pipeline_standard_accuracy  (float) — Carnot, standard questions
            pipeline_adversarial_accuracy (float) — Carnot, adversarial questions

        The function computes ``robustness_delta`` and ``retro_039_confirmed`` from
        these four values and ``inference_mode``.

        honest_verdict semantics:
            'thesis_confirmed'  — robustness_delta > 0 AND inference_mode == 'live_gpu'
            'thesis_rejected'   — robustness_delta <= 0 AND inference_mode == 'live_gpu'
            'gpu_required'      — inference_mode is anything other than 'live_gpu'

    Parameters
    ----------
    results : dict
        Must contain the four accuracy keys listed above.
        Additional keys (batch_log, n_questions, etc.) are passed through.
    inference_mode : str
        'live_gpu' when run on real hardware with CARNOT_FORCE_LIVE=1;
        any other string (e.g. 'simulated', 'gpu_required') for non-live paths.

    Returns
    -------
    dict
        Artifact ready for JSON serialization with schema='carnot.adversarial_v5.v1'.

    Spec: REQ-BENCH-053, SCENARIO-BENCH-038
    """
    baseline_std = results.get("baseline_standard_accuracy", 0.0)
    baseline_adv = results.get("baseline_adversarial_accuracy", 0.0)
    pipeline_std = results.get("pipeline_standard_accuracy", 0.0)
    pipeline_adv = results.get("pipeline_adversarial_accuracy", 0.0)

    robustness_delta = compute_robustness_delta(
        baseline_std, baseline_adv, pipeline_std, pipeline_adv
    )

    retro_039_confirmed = robustness_delta > 0 and inference_mode == "live_gpu"

    if inference_mode == "live_gpu":
        honest_verdict = "thesis_confirmed" if retro_039_confirmed else "thesis_rejected"
    else:
        honest_verdict = "gpu_required"

    artifact: dict[str, Any] = {
        "schema": "carnot.adversarial_v5.v1",
        "inference_mode": inference_mode,
        "baseline_standard_accuracy": baseline_std,
        "baseline_adversarial_accuracy": baseline_adv,
        "pipeline_standard_accuracy": pipeline_std,
        "pipeline_adversarial_accuracy": pipeline_adv,
        "robustness_delta": robustness_delta,
        "retro_039_confirmed": retro_039_confirmed,
        "honest_verdict": honest_verdict,
    }

    # Pass through any additional experiment data (batch_log, n_questions, etc.)
    for key, value in results.items():
        if key not in artifact:
            artifact[key] = value

    return artifact
