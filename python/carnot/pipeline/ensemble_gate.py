"""EnsembleGate — OR combination of InterWhen, HERMES v2, and Causal detection signals.

**Why this module exists (RETRO-070, Exp 643):**

    InterWhenMonitor (Exp 629), HERMES v2 live loop (Exp 641), and
    CausalReasoningVerifier (Exp 642) each catch a different error class:

        - InterWhenMonitor: arithmetic violations mid-generation (sentence-level)
        - HERMES v2: step-level hint injection during streaming generation
        - CausalReasoningVerifier: causal breaks between consecutive CoT steps

    Because the three detectors are orthogonal, their OR union covers violations
    that no single detector catches.  This module exposes compute_ensemble_hits()
    so the Exp 643 script and its tests can import the logic independently without
    running module-level experiment scaffolding code.

Spec: REQ-VERIFY-141, REQ-VERIFY-142,
      SCENARIO-VERIFY-186, SCENARIO-VERIFY-187
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carnot.pipeline.interwhen_monitor import InterWhenMonitor
    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier


def compute_ensemble_hits(
    responses: list[str],
    question_indices: list[int],
    interwhen_monitor: "InterWhenMonitor",
    causal_verifier: "CausalReasoningVerifier",
    hermes_tp_set: set[int],
) -> list[bool]:
    """Return one boolean per response: True if any of the three detectors fires.

    Why OR combination is the right gate for VR #17:
        Each of the three detectors covers a different violation class.  A response
        that passes InterWhen (no mid-step arithmetic errors) can still fail Causal
        (the numeric conclusion of step k is not used in step k+1).  The OR union
        maximises recall without duplicating per-response decisions.

    Args:
        responses: List of CoT response strings to evaluate (25 incorrect or 10 correct).
        question_indices: Original corpus index for each response (same length as responses).
                          Used to look up HERMES v2 TP indices from Exp 641.
        interwhen_monitor: Configured InterWhenMonitor instance (llm_caller may be None).
        causal_verifier: Configured CausalReasoningVerifier instance.
        hermes_tp_set: Set of question indices flagged as TP by HERMES v2 in Exp 641.
                       When Exp 641 did not store per-question indices, pass an empty set.
                       An empty set is conservative: no spurious TPs from guessing.

    Returns:
        List of booleans, one per response in input order.
    """
    hits: list[bool] = []
    for response, q_idx in zip(responses, question_indices):
        interwhen_hit = interwhen_monitor.any_violation(response)
        causal_hit = causal_verifier.any_violation(response)
        # hermes_hit is True only when Exp 641 stored a known TP index for this question.
        hermes_hit = q_idx in hermes_tp_set
        hits.append(interwhen_hit or hermes_hit or causal_hit)
    return hits
