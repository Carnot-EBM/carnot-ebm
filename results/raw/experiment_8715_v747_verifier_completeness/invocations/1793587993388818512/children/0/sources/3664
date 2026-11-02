"""Cached six-condition energy-vote-vs-self-consistency scoring for the P0.1 premise test.

Spec: REQ-KONA-3449

WHY THIS MODULE EXISTS (plain-language version for engineers who are not EBM
specialists):

The whole Phase-3 / Kona "foundation model" bet rests on a single empirical
premise — that scoring candidate answers with an *energy function* (an objective
"how internally consistent is this reasoning?" number) is a BETTER way to pick the
right answer than the cheap, dumb baseline everyone already uses: **majority vote
over several sampled answers** (a.k.a. "self-consistency"). If energy cannot even
match plain majority vote at the same compute budget, then the fancy
energy-descent reasoning story adds nothing and the superiority framing should be
retired honestly.

This module does the SCORING half of that test. The expensive half — actually
asking a 26B language model to generate `k` candidate solutions per problem —
already happened in exp3448 and was cached to a JSONL file. Here we just read that
cache and run six cheap, fully-deterministic selection strategies over the same
candidates:

  1. greedy AR        — the model's single most-likely answer (1-sample floor)
  2. self-consistency — majority vote over the k samples (THE control to beat)
  3. self-certainty   — pick the sample the model itself was most confident about
  4. energy-argmin    — pick the single lowest-energy sample
  5. energy-weighted  — vote, but weight each sample by softmax(-energy) (headline)
  6. energy×SC hybrid — combine the majority-vote signal with the energy weighting

Because NO live model is invoked, this completes in seconds and CANNOT idle-time-
out (the failure that killed exp3437). The energy function used here is a
deterministic, parameter-free verifier ensemble (arithmetic-consistency + adjacent
-step contradiction) so the result is fully reproducible from the cache + a seed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# The verifier-energy substrate. Both are parameter-free, deterministic heuristics
# (honest heuristics per CLAUDE.md — they make no model-based claim). IsingVerifier
# scores arithmetic constraint violations in a reasoning step; EbmCotCalibrator
# scores adjacent-step claim-polarity contradictions across the whole trace. Lower
# energy = more internally consistent reasoning.
from carnot.verify.ebm_cot import EbmCotCalibrator
from carnot.verify.semantic_energy import IsingVerifier

# Splits a generation into reasoning "steps". GSM8K chain-of-thought traces are
# newline- and sentence-separated, so we break on newlines and sentence
# terminators and keep non-empty fragments.
_STEP_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[.;])\s+")


def extract_steps(text: str) -> list[str]:
    """Split a candidate generation into a list of reasoning-step strings.

    The energy scorers operate on a *list of steps*, not one blob of text:
    arithmetic violations are per-step, and the adjacent-contradiction energy
    needs the steps in order. We split on newlines and sentence terminators and
    drop empty fragments. A generation with no usable text yields an empty list,
    which the energy scorers treat as zero-energy (no evidence either way).

    Parameters
    ----------
    text : str
        The raw generated chain-of-thought.

    Returns
    -------
    list[str]
        Non-empty, stripped reasoning-step fragments in original order.
    """
    if not text:
        return []
    parts = [p.strip() for p in _STEP_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def candidate_energy(
    text: str,
    ising: IsingVerifier,
    ebmcot: EbmCotCalibrator,
    arithmetic_weight: float = 1.0,
    contradiction_weight: float = 1.0,
) -> float:
    """Score one candidate generation with the deterministic verifier-energy ensemble.

    This is the load-bearing "energy function" of the whole test. It is an
    ENSEMBLE of two parameter-free verifier energies, both "lower is better":

      * arithmetic energy   — fraction of `A op B = C` claims in the text that are
        numerically wrong (IsingVerifier). A trace that says "7 * 10 = 80" is
        penalised. This is the strongest signal for grade-school math.
      * contradiction energy — count of adjacent reasoning steps whose claim
        polarity flips (EbmCotCalibrator, the EBM-CoT heuristic). A trace that
        asserts something then negates it next step is penalised.

    We deliberately use parameter-free heuristics (no training, no labels) so the
    score is fully reproducible from the cached corpus alone, and so the test is
    not accused of winning by fitting the energy model to the answers.

    Parameters
    ----------
    text : str
        The candidate generation.
    ising : IsingVerifier
        Arithmetic constraint-violation scorer (re-used across candidates).
    ebmcot : EbmCotCalibrator
        Adjacent-step contradiction scorer (re-used across candidates).
    arithmetic_weight, contradiction_weight : float
        Fixed (un-tuned) ensemble weights, documented in the artifact's
        compute_parity_note so energy cannot be accused of winning by tuning.

    Returns
    -------
    float
        Scalar energy; lower means a more internally-consistent reasoning trace.
    """
    steps = extract_steps(text)
    # IsingVerifier scores a single step string; apply it to the whole text so it
    # catches every `A op B = C` claim regardless of how we split steps.
    arithmetic = ising.energy(text)
    contradiction = ebmcot.energy(steps)
    return arithmetic_weight * arithmetic + contradiction_weight * contradiction


def _valid_pairs(answers: list, energies: list[float]) -> tuple[list, list[float]]:
    """Drop samples whose extracted answer is None, keeping answer/energy aligned."""
    out_a: list = []
    out_e: list[float] = []
    for a, e in zip(answers, energies):
        if a is not None:
            out_a.append(a)
            out_e.append(e)
    return out_a, out_e


def majority_vote(answers: list, confidences: list[float] | None = None) -> object:
    """Self-consistency: return the most frequent extracted answer (the PRIMARY control).

    This is the baseline energy must beat. We count how often each distinct answer
    appears across the k samples and return the most common one. Ties are broken
    deterministically: first by summed sample confidence (if provided), then by
    first-appearance order — so the result never depends on dict iteration order.

    Parameters
    ----------
    answers : list
        Extracted answers (ints), one per sample; None entries are ignored.
    confidences : list[float] | None
        Optional per-sample confidence used only for deterministic tie-breaking.

    Returns
    -------
    object
        The majority answer, or None if there are no valid answers.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    tied = [a for a, c in counts.items() if c == top]
    if len(tied) == 1:
        return tied[0]
    # Deterministic tie-break: prefer the answer with the highest summed confidence.
    if confidences is not None:
        conf_sum = {a: 0.0 for a in tied}
        for a, c in zip(answers, confidences):
            if a in conf_sum:
                conf_sum[a] += c
        best = max(conf_sum.values())
        tied = [a for a in tied if conf_sum[a] == best]
    if len(tied) == 1:
        return tied[0]
    # Final tie-break: first appearance order in the original sample list.
    for a in answers:
        if a in tied:
            return a
    return tied[0]


def self_certainty_bon(answers: list, confidences: list[float]) -> object:
    """Self-certainty Best-of-N (arXiv:2502.18581): the answer the model was surest of.

    Pick the single sample with the highest mean-token log-probability (the
    model's own confidence) and return its answer. This is the strongest *cheap*
    selector — it uses information energy does not (the model's logprobs) — so it
    is a demanding non-energy comparator.

    Returns None if no sample has a valid answer.
    """
    best_a: object = None
    best_c = -math.inf
    for a, c in zip(answers, confidences):
        if a is None:
            continue
        if c > best_c:
            best_c = c
            best_a = a
    return best_a


def energy_argmin(answers: list, energies: list[float]) -> object:
    """Energy-argmin: return the answer of the single lowest-energy sample."""
    best_a: object = None
    best_e = math.inf
    for a, e in zip(answers, energies):
        if a is None:
            continue
        if e < best_e:
            best_e = e
            best_a = a
    return best_a


def _softmax_neg_energy(energies: list[float], temperature: float) -> list[float]:
    """Return softmax(-E / T) weights, numerically stabilised against overflow."""
    if not energies:
        return []
    scaled = [-e / temperature for e in energies]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    if total <= 0:
        n = len(exps)
        return [1.0 / n] * n
    return [x / total for x in exps]


def energy_weighted_vote(
    answers: list, energies: list[float], temperature: float = 1.0
) -> object:
    """Energy-weighted vote (EBM-CoT, arXiv:2511.07124) — THE headline condition.

    Instead of one-vote-per-sample (plain majority), each sample votes with weight
    `softmax(-energy / T)`: low-energy (internally consistent) samples count for
    more. We sum weights per distinct answer and return the heaviest. As
    `T -> inf` every weight becomes equal and this exactly recovers plain majority
    vote, so the premise under test is that a finite `T` reshapes the vote toward
    correctness. Ties break by first-appearance order for determinism.

    Returns None if no sample has a valid answer.
    """
    va, ve = _valid_pairs(answers, energies)
    if not va:
        return None
    weights = _softmax_neg_energy(ve, temperature)
    bucket: dict = {}
    for a, w in zip(va, weights):
        bucket[a] = bucket.get(a, 0.0) + w
    best = max(bucket.values())
    tied = [a for a in bucket if bucket[a] == best]
    if len(tied) == 1:
        return tied[0]
    for a in va:
        if a in tied:
            return a
    return tied[0]


def energy_sc_hybrid(
    answers: list,
    energies: list[float],
    confidences: list[float] | None = None,
    temperature: float = 1.0,
) -> object:
    """Energy×SC hybrid (arXiv:2510.14913): combine majority count with energy weight.

    The budget-aware-hybrid literature finds that combining a verifier signal with
    the sampling/voting signal beats either alone — mirroring the .317 Kona finding
    that only the hybrid (not pure energy) solves. We implement the combination by
    scoring each distinct answer with the SUM of two normalised signals:

      * normalised vote count   — how much self-consistency likes the answer
      * normalised energy weight — how much the energy-weighted vote likes it

    and returning the argmax. This degenerates to majority vote when energy is
    uninformative and to energy-weighted vote when votes are uniform, so it can
    only help. Ties break by first-appearance order.

    Returns None if no sample has a valid answer.
    """
    va, ve = _valid_pairs(answers, energies)
    if not va:
        return None
    counts = Counter(va)
    weights = _softmax_neg_energy(ve, temperature)
    energy_mass: dict = {}
    for a, w in zip(va, weights):
        energy_mass[a] = energy_mass.get(a, 0.0) + w
    total_count = sum(counts.values())
    total_mass = sum(energy_mass.values()) or 1.0
    score: dict = {}
    for a in counts:
        score[a] = counts[a] / total_count + energy_mass.get(a, 0.0) / total_mass
    best = max(score.values())
    tied = [a for a in score if score[a] == best]
    if len(tied) == 1:
        return tied[0]
    for a in va:
        if a in tied:
            return a
    return tied[0]


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> float:
    """Exact (binomial) McNemar p-value for two PAIRED binary classifiers.

    McNemar's test asks: among the problems where the two methods DISAGREE, is the
    split lopsided enough to be unlikely by chance? It uses only the discordant
    pairs (b=1/a=0 vs a=1/b=0) and an exact two-sided binomial test with p=0.5, so
    it is valid for small n. A small p means one method genuinely wins on the
    problems where they differ; a large p means the observed delta is plausibly
    noise.

    Parameters
    ----------
    a_correct, b_correct : list[bool]
        Per-problem correctness for method A and method B, paired (same order).

    Returns
    -------
    float
        Two-sided exact p-value in [0, 1]; 1.0 when there are no discordant pairs.
    """
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and (not b))
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    # Two-sided exact binomial: 2 * P(X <= k) capped at 1.0, with p = 0.5.
    cum = 0.0
    for i in range(0, k + 1):
        cum += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * cum)


def paired_bootstrap_ci(
    a_correct: list[bool],
    b_correct: list[bool],
    seed: int,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired bootstrap CI95 for the accuracy delta (A minus B) over the SAME problems.

    We resample problem indices with replacement (the SAME index applies to both
    methods — that is what makes it PAIRED), recompute mean(A) - mean(B) each time,
    and take the empirical 2.5th/97.5th percentiles. A CI that excludes zero is a
    significant delta; a CI straddling zero is not. Pairing removes per-problem
    difficulty variance, so it is far more sensitive than two independent CIs.

    Uses a deterministic seeded LCG so the result is reproducible without numpy.

    Parameters
    ----------
    a_correct, b_correct : list[bool]
        Paired per-problem correctness.
    seed : int
        RNG seed for reproducibility.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Two-sided significance level (0.05 -> 95% CI).

    Returns
    -------
    tuple[float, float]
        (low, high) bounds of the (mean_A - mean_B) accuracy delta.
    """
    n = len(a_correct)
    if n == 0:
        return (0.0, 0.0)
    diffs = [(1.0 if a else 0.0) - (1.0 if b else 0.0) for a, b in zip(a_correct, b_correct)]
    # Deterministic LCG (numerical-recipes constants) — no numpy dependency.
    state = (seed & 0xFFFFFFFF) or 1
    deltas: list[float] = []
    for _ in range(n_boot):
        acc = 0.0
        for _ in range(n):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            idx = state % n
            acc += diffs[idx]
        deltas.append(acc / n)
    deltas.sort()
    lo_i = int((alpha / 2.0) * n_boot)
    hi_i = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))
    return (deltas[lo_i], deltas[hi_i])


@dataclass
class ScoringResult:
    """All six condition accuracies plus the paired deltas and significance.

    Fields map directly onto the REQ-KONA-3449 artifact schema; the experiment
    script copies them into the JSON deliverable.
    """

    n_problems: int
    k_samples: int
    ar_greedy_accuracy: float
    self_consistency_accuracy: float
    self_certainty_bon_accuracy: float
    energy_argmin_accuracy: float
    energy_weighted_vote_accuracy: float
    energy_sc_hybrid_accuracy: float
    self_consistency_non_degenerate: bool
    degenerate_examples: list
    delta_energy_vs_self_consistency: float
    delta_hybrid_vs_self_consistency: float
    delta_energy_vs_greedy_ar: float
    paired_significance: dict


def _accuracy(preds: list, golds: list) -> float:
    """Fraction of predictions exactly equal to the gold answer."""
    if not golds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if p is not None and p == g) / len(golds)


def score_corpus(
    records: list[dict],
    *,
    seed: int,
    temperature: float = 1.0,
    n_boot: int = 10000,
    ising: IsingVerifier | None = None,
    ebmcot: EbmCotCalibrator | None = None,
) -> ScoringResult:
    """Score all six conditions over the cached corpus on the SAME paired problems.

    Each record is one problem with a gold answer, a greedy generation, and a list
    of `k` sampled generations (each carrying an extracted answer, raw text, and
    mean-token logprob). We compute, per problem, the prediction of each of the six
    strategies, then aggregate to per-condition accuracy and the PRIMARY paired
    deltas with McNemar + bootstrap significance.

    The NON-DEGENERATE-SC gate (self_consistency_non_degenerate) is set here over
    the FULL corpus: it requires SC accuracy >= greedy accuracy AND SC > 0.30. A
    degenerate gate means the per-sample answer extraction is broken (the exp3426
    0.0-tie bug) and no energy comparison should be trusted; we also capture three
    raw-answer examples for diagnosis.

    Parameters
    ----------
    records : list[dict]
        Cached corpus rows (problem_id, gold, greedy, samples, ...).
    seed : int
        Reproducibility seed for the bootstrap.
    temperature : float
        Fixed (un-tuned) softmax temperature for the energy-weighted conditions.
    n_boot : int
        Bootstrap resamples for the paired CI.
    ising, ebmcot : optional pre-built scorers (re-used across all candidates).

    Returns
    -------
    ScoringResult
    """
    ising = ising or IsingVerifier()
    ebmcot = ebmcot or EbmCotCalibrator()

    golds: list = []
    greedy_preds: list = []
    sc_preds: list = []
    certainty_preds: list = []
    eargmin_preds: list = []
    evote_preds: list = []
    hybrid_preds: list = []
    degenerate_examples: list = []

    for rec in records:
        gold = rec["gold"]
        golds.append(gold)
        greedy = rec.get("greedy") or {}
        greedy_preds.append(greedy.get("answer"))

        samples = rec.get("samples") or []
        answers = [s.get("answer") for s in samples]
        confidences = [
            s.get("mean_token_logprob")
            if s.get("mean_token_logprob") is not None
            else -math.inf
            for s in samples
        ]
        energies = [candidate_energy(s.get("text", ""), ising, ebmcot) for s in samples]

        sc_preds.append(majority_vote(answers, confidences))
        certainty_preds.append(self_certainty_bon(answers, confidences))
        eargmin_preds.append(energy_argmin(answers, energies))
        evote_preds.append(energy_weighted_vote(answers, energies, temperature))
        hybrid_preds.append(energy_sc_hybrid(answers, energies, confidences, temperature))

        if len(degenerate_examples) < 3:
            degenerate_examples.append(
                {
                    "problem_id": rec.get("problem_id"),
                    "gold": gold,
                    "greedy_answer": greedy.get("answer"),
                    "sample_answers": answers,
                }
            )

    ar_acc = _accuracy(greedy_preds, golds)
    sc_acc = _accuracy(sc_preds, golds)
    certainty_acc = _accuracy(certainty_preds, golds)
    eargmin_acc = _accuracy(eargmin_preds, golds)
    evote_acc = _accuracy(evote_preds, golds)
    hybrid_acc = _accuracy(hybrid_preds, golds)

    non_degenerate = (sc_acc >= ar_acc) and (sc_acc > 0.30)

    sc_correct = [p is not None and p == g for p, g in zip(sc_preds, golds)]
    evote_correct = [p is not None and p == g for p, g in zip(evote_preds, golds)]
    hybrid_correct = [p is not None and p == g for p, g in zip(hybrid_preds, golds)]

    primary = {
        "comparison": "energy_weighted_vote_vs_self_consistency",
        "mcnemar_exact_p": mcnemar_exact(sc_correct, evote_correct),
        "bootstrap_ci95": list(
            paired_bootstrap_ci(evote_correct, sc_correct, seed=seed, n_boot=n_boot)
        ),
    }
    hybrid_sig = {
        "comparison": "energy_sc_hybrid_vs_self_consistency",
        "mcnemar_exact_p": mcnemar_exact(sc_correct, hybrid_correct),
        "bootstrap_ci95": list(
            paired_bootstrap_ci(hybrid_correct, sc_correct, seed=seed, n_boot=n_boot)
        ),
    }

    return ScoringResult(
        n_problems=len(records),
        k_samples=max((len(r.get("samples") or []) for r in records), default=0),
        ar_greedy_accuracy=ar_acc,
        self_consistency_accuracy=sc_acc,
        self_certainty_bon_accuracy=certainty_acc,
        energy_argmin_accuracy=eargmin_acc,
        energy_weighted_vote_accuracy=evote_acc,
        energy_sc_hybrid_accuracy=hybrid_acc,
        self_consistency_non_degenerate=non_degenerate,
        degenerate_examples=degenerate_examples,
        delta_energy_vs_self_consistency=evote_acc - sc_acc,
        delta_hybrid_vs_self_consistency=hybrid_acc - sc_acc,
        delta_energy_vs_greedy_ar=evote_acc - ar_acc,
        paired_significance={"primary": primary, "hybrid": hybrid_sig},
    )


def derive_premise_v4_verdict(result: ScoringResult) -> str:
    """Map the scoring result to exactly one terminal verdict prefixed `complete:`.

    The gate ladder (per REQ-KONA-3449 acceptance gates):

      * G0 NON-DEGENERATE-SC: if self-consistency is degenerate (a broken harness,
        the exp3426 0.0-tie), no energy comparison is trustworthy.
      * G1 ENERGY-NON-INFERIOR: max(energy-weighted, hybrid) >= self-consistency.
        Below this, energy adds nothing over plain sampling.
      * G2 ENERGY-ADDS-VALUE: energy OR hybrid SIGNIFICANTLY beats self-consistency
        (positive delta AND paired McNemar p < 0.05) — the first real justification
        for the Phase-3 endgame, clearing the arXiv:2410.12608 bar.

    Returns
    -------
    str
        One of the five terminal `complete:` verdicts.
    """
    if not result.self_consistency_non_degenerate:
        return (
            "complete: blocked_self_consistency_harness_degenerate_"
            "per_sample_extraction_broken"
        )

    best_energy = max(
        result.energy_weighted_vote_accuracy, result.energy_sc_hybrid_accuracy
    )
    g1 = best_energy >= result.self_consistency_accuracy

    primary = result.paired_significance["primary"]
    hybrid = result.paired_significance["hybrid"]
    g2 = (
        result.delta_energy_vs_self_consistency > 0
        and primary["mcnemar_exact_p"] < 0.05
    ) or (
        result.delta_hybrid_vs_self_consistency > 0
        and hybrid["mcnemar_exact_p"] < 0.05
    )

    if g2:
        return "complete: energy_beats_self_consistency_premise_validated"
    if g1:
        return (
            "complete: energy_matches_but_does_not_beat_self_consistency_"
            "at_equal_compute"
        )
    return (
        "complete: energy_below_self_consistency_premise_unsupported_"
        "retire_superiority_framing"
    )
