"""Real-harness premise helpers (P0.1 v3 / exp3437).

**Why this module exists (plain-language summary):**
    Exp 3426 (the v2 premise test) tried to answer the load-bearing P0.1
    question — does energy-weighted voting beat plain majority-vote
    self-consistency at the *same* compute? — but its multi-sample harness was
    broken. Every k-sample condition (self-consistency, self-certainty BoN,
    energy-argmin, energy-weighted vote) returned 0.0 accuracy while greedy AR
    scored 0.75. The "energy matches self-consistency" verdict it emitted was a
    degenerate 0.0-vs-0.0 tie, NOT a measurement.

    The root cause was mechanical, not scientific: the experiment script
    requested per-token logprobs (`logprobs=1`) from a llama.cpp model that was
    created with `logits_all=False`. llama.cpp raises
    ``ValueError: logprobs is not supported for models created with
    logits_all=False`` in that situation, and the script's broad
    ``except Exception`` swallowed it, returning an empty string for every
    sampled candidate. Empty text extracts to a ``None`` answer, so every vote
    collapsed to nothing. The greedy condition survived only because it did not
    request logprobs.

    This module adds the *pure, deterministic, GPU-free* pieces v3 needs on top
    of the already-tested v2 surface:

      1. ``extract_candidate_answers`` — per-sample answer extraction that runs
         on EACH of the k sampled generations (the thing v2 effectively never
         exercised because every sample was empty). It is a thin, unit-tested
         wrapper over the shared ``extract_final_answer`` so the experiment
         script and the tests share one extraction code path.

      2. ``evaluate_sc_non_degenerate`` + ``ScDegeneracyGate`` — the step-0e
         NON-DEGENERATE-SC gate. Before any energy comparison is reported, the
         experiment samples a warm-up batch and asserts that majority-vote
         self-consistency is non-degenerate: its accuracy is at least the greedy
         accuracy AND strictly above an absolute floor (0.30 by default). A
         self-consistency baseline that scores 0.0 (or below greedy) is a
         broken harness, full stop — the gate makes the v2 0.0-tie impossible to
         ship.

      3. ``derive_premise_v3_verdict`` — the verdict mapping. If the
         non-degenerate gate failed, it returns the degenerate-harness blocked
         terminal (an honest finding: the harness still needs repair). Otherwise
         it delegates to the v2 verdict logic, whose PRIMARY comparison is
         energy-weighted vote vs self-consistency.

    The slow live-LLM I/O (loading the GGUF *with logits_all=True so logprobs
    work*, sampling candidates, training the energy substrate) stays in the
    experiment script; everything a reviewer needs to re-derive the verdict from
    saved per-problem outcomes lives here, GPU-free.

Spec: REQ-KONA-3437 (real-harness four-condition premise test),
SCENARIO-KONA-3437, SCENARIO-KONA-3437-DEGENERATE.
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export the shared, already-tested primitives so the experiment script and
# the v3 tests import one coherent surface. These are unchanged from v1/v2.
from carnot.phase3.energy_premise_v2 import (  # noqa: F401
    EnergyDescentResult,
    GsmProblem,
    PremiseV2Verdict,
    derive_premise_v2_verdict,
    energy_descent_select,
    energy_weighted_vote,
    extract_final_answer,
    is_correct,
    load_gsm8k_subset,
    majority_vote,
    mcnemar_test,
    mean_token_confidence,
    paired_bootstrap_ci,
    reproducibility_checksum,
    self_certainty_select,
)

# The terminal verdict emitted when the warm-up NON-DEGENERATE-SC gate fails.
# It starts with ``complete:`` so the conductor reconciler classifies it as a
# terminal honest finding (the harness still needs repair) rather than retrying.
DEGENERATE_VERDICT = (
    "complete: blocked_self_consistency_harness_degenerate_per_sample_extraction_broken"
)

# Default absolute floor for a non-degenerate self-consistency accuracy. GSM8K
# self-consistency for a competent SOTA model is well above this; 0.30 is a
# generous floor whose only job is to reject the 0.0 (all-empty) failure mode.
DEFAULT_SC_MIN_ABS = 0.30


def extract_candidate_answers(candidate_texts: list[str]) -> list[int | None]:
    """Extract the final integer answer from EACH of the k sampled generations.

    This is the function v2's harness never truly exercised: because every
    sampled generation came back as an empty string (the swallowed
    logprobs/logits_all error), ``extract_final_answer`` was always handed ``""``
    and always returned ``None``. Here we make the per-sample extraction an
    explicit, unit-tested step so a reviewer can confirm that a *non-empty*
    sampled generation does produce an answer.

    We deliberately reuse the shared ``extract_final_answer`` (canonical
    ``#### <n>`` coda first, last-standalone-number fallback) so the k sampled
    candidates and the single greedy candidate are scored by identical logic —
    any asymmetry there would itself be a harness bug. A ``None`` in the returned
    list means that specific generation produced no parseable number, which the
    vote aggregators already treat as a non-vote.
    """

    return [extract_final_answer(text) for text in candidate_texts]


@dataclass(frozen=True)
class ScDegeneracyGate:
    """Outcome of the step-0e NON-DEGENERATE-SC warm-up gate.

    ``passed`` is the load-bearing boolean: ``True`` only when majority-vote
    self-consistency on the warm-up batch is non-degenerate (at least as good as
    greedy AND strictly above the absolute floor). ``self_consistency_accuracy``
    / ``ar_greedy_accuracy`` are the two warm-up accuracies the gate compared,
    ``min_abs_threshold`` is the floor used, and ``reason`` is a short
    human-readable explanation (empty when the gate passed) so a failing run's
    artifact says *why* the harness is considered broken.
    """

    passed: bool
    self_consistency_accuracy: float
    ar_greedy_accuracy: float
    min_abs_threshold: float
    reason: str


def evaluate_sc_non_degenerate(
    ar_greedy_correct: list[bool],
    self_consistency_correct: list[bool],
    *,
    min_abs: float = DEFAULT_SC_MIN_ABS,
) -> ScDegeneracyGate:
    """Decide whether warm-up self-consistency is non-degenerate (the v2 fix).

    The exp3426 disaster was a self-consistency accuracy of exactly 0.0 sitting
    next to a greedy accuracy of 0.75 — a textbook impossibility (majority vote
    over k samples should *beat* a single greedy generation, never collapse to
    nothing). That can only happen when the per-sample answer extraction or the
    vote aggregation is broken. This gate encodes the textbook expectation as a
    hard precondition:

      * self-consistency accuracy MUST be **>= greedy accuracy** (majority vote
        does not lose to one greedy sample on GSM8K), and
      * self-consistency accuracy MUST be **strictly above** an absolute floor
        ``min_abs`` (a competent SOTA model clears 0.30 on GSM8K with ease; the
        floor exists only to reject the all-empty 0.0 case).

    Both correctness vectors come from the SAME warm-up problems (paired), so the
    comparison is apples-to-apples. Raises ``ValueError`` on empty or
    mismatched-length inputs because a gate computed on no data, or on misaligned
    pairs, would give a meaningless verdict.
    """

    if len(ar_greedy_correct) != len(self_consistency_correct):
        raise ValueError(
            "non-degenerate gate requires equal-length paired correctness vectors"
        )
    n = len(ar_greedy_correct)
    if n == 0:
        raise ValueError("non-degenerate gate requires a non-empty warm-up batch")

    ar_acc = sum(1 for c in ar_greedy_correct if c) / n
    sc_acc = sum(1 for c in self_consistency_correct if c) / n

    if sc_acc <= min_abs:
        return ScDegeneracyGate(
            passed=False,
            self_consistency_accuracy=sc_acc,
            ar_greedy_accuracy=ar_acc,
            min_abs_threshold=min_abs,
            reason=(
                f"self-consistency accuracy {sc_acc:.3f} <= floor {min_abs:.3f}: "
                "per-sample extraction or vote aggregation is broken (the exp3426 "
                "all-empty 0.0 failure mode)"
            ),
        )
    if sc_acc < ar_acc:
        return ScDegeneracyGate(
            passed=False,
            self_consistency_accuracy=sc_acc,
            ar_greedy_accuracy=ar_acc,
            min_abs_threshold=min_abs,
            reason=(
                f"self-consistency accuracy {sc_acc:.3f} < greedy accuracy "
                f"{ar_acc:.3f}: majority vote should never lose to one greedy "
                "sample on GSM8K — the multi-sample harness is degenerate"
            ),
        )
    return ScDegeneracyGate(
        passed=True,
        self_consistency_accuracy=sc_acc,
        ar_greedy_accuracy=ar_acc,
        min_abs_threshold=min_abs,
        reason="",
    )


def derive_premise_v3_verdict(
    sc_gate: ScDegeneracyGate,
    self_consistency_accuracy: float,
    energy_weighted_vote_accuracy: float,
    p_value: float,
    ci: tuple[float, float],
    *,
    direction: float,
    alpha: float = 0.05,
) -> PremiseV2Verdict:
    """Map the run to a terminal verdict, gated on a non-degenerate harness.

    The v3 contract is: NO energy-vs-self-consistency comparison may produce a
    success/refute verdict unless the warm-up harness was proven non-degenerate
    first. So:

      * If ``sc_gate.passed`` is ``False`` the harness is broken and we return
        the ``DEGENERATE_VERDICT`` honest finding regardless of the (meaningless)
        downstream accuracies. ``g1``/``g2`` are both ``False`` because nothing
        about energy was actually measured.
      * Otherwise we delegate to the v2 verdict logic, whose PRIMARY comparison
        is energy-weighted vote vs self-consistency (G1 non-inferiority, G2
        significant superiority). Reusing v2 keeps the gate semantics identical
        and the surface area small.
    """

    if not sc_gate.passed:
        return PremiseV2Verdict(DEGENERATE_VERDICT, False, False)
    return derive_premise_v2_verdict(
        self_consistency_accuracy,
        energy_weighted_vote_accuracy,
        p_value,
        ci,
        direction=direction,
        alpha=alpha,
    )
