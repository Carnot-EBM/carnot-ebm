"""Four-condition matched-compute premise helpers (P0.1 v2 / exp3426).

**Why this module exists (plain-language summary):**
    Exp 3312 (the v1 premise test) showed that *energy-descent selection* beat a
    single greedy autoregressive (AR) generation, but it LOST to plain
    *majority-vote self-consistency* over the same samples. That left the
    load-bearing question unanswered: does the energy function add ANYTHING over
    cheap majority voting at the *same compute budget*? If it does not, the whole
    "energy-descent reasoning is a better substrate" framing that motivates the
    Phase-3 / Kona foundation-model endgame is unsupported and should retire.

    This module holds the pure, deterministic, unit-testable pieces that v1 did
    not have: the two extra cheap baselines energy must beat at equal compute and
    the new verdict logic whose PRIMARY comparison is *energy-weighted vote vs
    majority-vote self-consistency*. The slow live-LLM I/O (loading the GGUF,
    sampling candidates, training the energy substrate) stays in the experiment
    script; everything a reviewer needs to re-derive the verdict from the saved
    per-problem outcomes lives here, GPU-free.

    Conditions under matched compute (all four aggregate the SAME k samples;
    greedy AR is the 1-sample floor):
      1. greedy AR                     — 1 greedy generation (the v1 baseline).
      2. self-consistency              — majority vote over k samples (PRIMARY
                                         control; from the v1 module).
      3. self-certainty Best-of-N      — pick the single sample with the highest
                                         mean token confidence (arXiv:2502.18581).
      4. energy                        — (a) energy-argmin select and
                                         (b) energy-WEIGHTED vote, softmax(-E/T)
                                         over answers (the EBM-CoT mechanism,
                                         arXiv:2511.07124). Condition under test.

Spec: REQ-KONA-3426 (four-condition matched-compute premise test),
SCENARIO-KONA-3426, SCENARIO-KONA-3426-BLOCKED.

What we are approximating (honest-heuristic disclosure, per CLAUDE.md Verifier
Authenticity Discipline): self-certainty in arXiv:2502.18581 is the average
KL divergence from a uniform distribution to the model's full predictive
distribution at each step, which requires the per-token logit vector. At
inference time through llama.cpp we cheaply have only the *chosen-token*
logprob, so ``mean_token_confidence`` uses the mean chosen-token probability as
a faithful monotone proxy for sequence confidence. This is a text/logprob
statistic, NOT a reconstruction of the paper's full-distribution KL, and the
selection ranking it induces is the proxy this experiment evaluates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Re-export the shared, already-tested primitives so the experiment script and
# the v2 tests import one coherent surface. These are unchanged from v1.
from carnot.phase3.energy_descent_premise import (  # noqa: F401
    EnergyDescentResult,
    GsmProblem,
    derive_premise_verdict,
    energy_descent_select,
    extract_final_answer,
    is_correct,
    load_gsm8k_subset,
    majority_vote,
    mcnemar_test,
    paired_bootstrap_ci,
    reproducibility_checksum,
)

# ---------------------------------------------------------------------------
# Condition 3: self-certainty Best-of-N (arXiv:2502.18581)
# ---------------------------------------------------------------------------


def mean_token_confidence(token_logprobs: list[float] | None) -> float:
    """Mean chosen-token probability for one generated sample.

    This is the cheap self-certainty proxy (see module docstring for the gap
    disclosure): given the per-token logprobs of the *chosen* tokens, we convert
    each to a probability ``exp(logprob)`` and average. A sample where the model
    was consistently confident (high chosen-token probability) scores near 1.0;
    a hesitant, high-entropy generation scores low. Returns ``0.0`` for an empty
    or missing logprob list so a candidate that produced no scorable tokens can
    never win the Best-of-N selection by default.
    """

    if not token_logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in token_logprobs if lp is not None and math.isfinite(lp)]
    if not probs:
        return 0.0
    return sum(probs) / len(probs)


def self_certainty_select(candidate_token_logprobs: list[list[float] | None]) -> int:
    """Best-of-N index by self-certainty: the most-confident of the k samples.

    This is the strongest *cheap* selector energy must beat — it spends zero
    extra generations (it reuses the same k samples the energy condition scores)
    and needs only the logprobs llama.cpp already returns. We pick the candidate
    with the highest ``mean_token_confidence``; ties break toward the
    earliest-listed candidate, which is deterministic given the input order.
    Raises ``ValueError`` on an empty candidate set because there is nothing to
    select.
    """

    if not candidate_token_logprobs:
        raise ValueError("self_certainty_select requires at least one candidate")
    scores = [mean_token_confidence(lp) for lp in candidate_token_logprobs]
    # max() with a key returns the first argmax on ties — the earliest candidate.
    return max(range(len(scores)), key=lambda i: scores[i])


# ---------------------------------------------------------------------------
# Condition 4b: energy-weighted vote (EBM-CoT calibration, arXiv:2511.07124)
# ---------------------------------------------------------------------------


def energy_weighted_vote(
    answers: list[int | None],
    energies: list[float],
    *,
    temperature: float = 1.0,
) -> int | None:
    """Aggregate the k candidate answers by softmax(-E/T) energy weighting.

    This mirrors EBM-CoT's latent calibration toward low-energy regions: instead
    of each candidate contributing one equal vote (plain majority / self-
    consistency), each contributes a weight ``softmax(-E_i / T)`` so that
    lower-energy (more correct-looking, per the learned verifier) candidates
    dominate. We sum the weights per *distinct extracted answer* and return the
    highest-weighted answer.

    This is the headline energy condition: it occupies the same niche as
    self-consistency (it aggregates votes, not picks a single sample) but lets
    the energy function reshape the vote. If it cannot beat — or even match —
    plain majority vote at the same compute, the energy substrate adds nothing.

    ``temperature`` controls how sharply energy reshapes the vote: ``T -> inf``
    recovers uniform weights (plain majority vote), small ``T`` approaches
    energy-argmin. We subtract the max logit before exponentiating for numerical
    stability. Null answers contribute no weight; ties break toward the
    earliest-appearing answer (deterministic). Returns ``None`` only when every
    candidate failed to produce an answer.
    """

    if len(answers) != len(energies):
        raise ValueError("energy_weighted_vote requires equal-length answers/energies")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Softmax over -E/T with the standard max-subtraction stabiliser.
    logits = [-e / temperature for e in energies]
    max_logit = max(logits)
    weights = [math.exp(lg - max_logit) for lg in logits]

    tallies: dict[int, float] = {}
    order: list[int] = []
    for answer, weight in zip(answers, weights, strict=True):
        if answer is None:
            continue
        if answer not in tallies:
            tallies[answer] = 0.0
            order.append(answer)
        tallies[answer] += weight
    if not order:
        return None
    # Highest summed weight; ties -> earliest-appearing answer.
    return max(order, key=lambda a: (tallies[a], -order.index(a)))


# ---------------------------------------------------------------------------
# v2 verdict: PRIMARY comparison is energy-weighted vote vs self-consistency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PremiseV2Verdict:
    """Terminal classification whose gates are stated against self-consistency.

    ``g1_energy_non_inferior`` — energy at least matches plain majority vote at
    equal compute (below this the premise is unsupported). ``g2_energy_adds_value``
    — energy SIGNIFICANTLY beats majority vote at matched compute, the first real
    justification for the Phase-3 endgame. ``verdict`` is the ``complete:``-prefixed
    string the conductor reconciler reads.
    """

    verdict: str
    g1_energy_non_inferior: bool
    g2_energy_adds_value: bool


def derive_premise_v2_verdict(
    self_consistency_accuracy: float,
    energy_weighted_vote_accuracy: float,
    p_value: float,
    ci: tuple[float, float],
    *,
    direction: float,
    alpha: float = 0.05,
) -> PremiseV2Verdict:
    """Map the PRIMARY (energy-vote vs self-consistency) comparison to a verdict.

    The gates follow the exp3426 spec exactly. The paired significance test and
    CI here are computed on the energy-weighted-vote vs self-consistency
    correctness vectors (NOT vs greedy AR — that comparison is reported only for
    continuity with exp3312):

      * **G2 ENERGY-ADDS-VALUE:** ``energy_weighted_vote`` is *strictly* more
        accurate than self-consistency AND the paired test is significant in
        energy's favour (``p < alpha``, positive direction, CI lower bound above
        0). This is the only outcome that justifies the foundation-model endgame.
      * **G1 ENERGY-NON-INFERIOR:** energy is *non-inferior* to self-consistency —
        its accuracy is at least self-consistency's, or any shortfall is not
        statistically significant (``p >= alpha``). Below this, energy adds
        nothing over plain sampling and the premise is unsupported.

    Every branch returns a ``complete:``-prefixed verdict because refutation is
    as terminal (and as publishable) as validation for this experiment.
    """

    significant = p_value < alpha
    g2 = (
        energy_weighted_vote_accuracy > self_consistency_accuracy
        and significant
        and direction > 0
        and ci[0] > 0.0
    )
    g1 = (energy_weighted_vote_accuracy >= self_consistency_accuracy) or (not significant)

    if g2:
        return PremiseV2Verdict(
            "complete: energy_beats_self_consistency_premise_validated", True, True
        )
    if g1:
        return PremiseV2Verdict(
            "complete: energy_matches_but_does_not_beat_self_consistency_at_equal_compute",
            True,
            False,
        )
    return PremiseV2Verdict(
        "complete: energy_below_self_consistency_premise_unsupported_retire_superiority_framing",
        False,
        False,
    )
