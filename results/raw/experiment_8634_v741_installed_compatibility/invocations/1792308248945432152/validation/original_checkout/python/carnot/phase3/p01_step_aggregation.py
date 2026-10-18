"""Step-to-final aggregation functions for P0.1 AUROC gap analysis.

WHY THIS MODULE EXISTS:
exp3497 found that the FoVer 4-verifier ensemble detects step errors at
0.9131 AUROC but only reaches 0.601 AUROC on MATH final-answer correctness
when per-step energies are averaged (mean aggregation). This step-vs-final
gap of 0.138 is the core open question for exp3508.

Different aggregation functions route per-step energies into a single
per-candidate final score. The five functions below follow the catalogue in
arXiv:2508.01773 (PRM step-score routing) and arXiv:2504.16828 (ThinkPRM):

  mean:  arithmetic mean of per-step energies -- the current default in
         process_energy_per_step (exp3497 baseline, AUROC ~0.601).
  last:  only the final step's energy -- useful if MATH answers concentrate
         the error signal in the concluding step.
  min:   minimum per-step energy -- dominated by the cleanest step;
         in log-probability terms this is the "best path" heuristic.
  product: geometric mean of per-step energies -- emphasises ALL steps
         jointly; dominated by high-energy (suspicious) steps when
         energy values are > 1.
  uncertainty_weighted: each step is weighted by 1/(std of verifier signals +
         eps), so steps where the FoVer ensemble verifiers strongly agree
         carry more weight than steps where they disagree (uncertain).

All functions return a per-candidate energy value where LOWER means the
ensemble thinks the candidate is more correct (consistent with the rest of
the FoVer energy convention).

Spec: REQ-KONA-3508, SCENARIO-KONA-3508
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

# Tolerance for floating-point zero checks in geometric mean.
_LOG_EPS: float = 1e-9


def aggregate_step_energies(
    per_step_verifier_scores: Sequence[tuple[float, float, float]],
    method: str,
) -> float:
    """Aggregate per-step verifier-signal tuples into one candidate energy.

    Each element of ``per_step_verifier_scores`` is a 3-tuple
    ``(ising_energy, tier0r_score, tier0u_score)`` for one reasoning step.
    The per-step total energy is ``sum(ising, tier0r, tier0u)``.

    Lower aggregate energy means the ensemble considers the candidate more
    internally consistent -- higher score for final-correctness prediction
    (by negating this value before passing to binary_auroc).

    Parameters
    ----------
    per_step_verifier_scores:
        One (ising, tier0r, tier0u) tuple per step.  Empty list returns 0.0
        (no detectable step-level violation when there are no steps).
    method:
        One of ``'mean'``, ``'last'``, ``'min'``, ``'product'``,
        ``'uncertainty_weighted'``.

    Returns
    -------
    float
        Aggregate energy >= 0.  Lower = predicts correct.

    Raises
    ------
    ValueError
        If ``method`` is not one of the five supported names.
    """
    if not per_step_verifier_scores:
        return 0.0

    # Per-step totals: ising + tier0r + tier0u
    totals = [ising + tier0r + tier0u for ising, tier0r, tier0u in per_step_verifier_scores]

    if method == "mean":
        # Arithmetic mean -- the exp3497 baseline (without the trace-level
        # ebmcot contradiction term, which is a constant additive offset that
        # does not change between-candidate ranking).
        return sum(totals) / len(totals)

    if method == "last":
        # Only the energy of the last reasoning step.  Useful if MATH answer
        # errors concentrate in the concluding calculation step.
        return totals[-1]

    if method == "min":
        # Minimum per-step energy.  The "best path" heuristic: a candidate
        # whose weakest step is still clean is preferred.  For final-correctness
        # this captures "every step must be clean" less aggressively than product.
        return min(totals)

    if method == "product":
        # Geometric mean of per-step energies.  Penalises ANY high-energy step
        # severely; equivalent to the product aggregation from arXiv:2508.01773
        # but expressed as a mean to avoid underflow on long traces.
        # exp(mean(log(total_i + eps))) avoids log(0).
        log_mean = sum(math.log(t + _LOG_EPS) for t in totals) / len(totals)
        return math.exp(log_mean)

    if method == "uncertainty_weighted":
        # Weight each step by the RECIPROCAL of the standard deviation across
        # its three verifier signals.  Steps where verifiers strongly agree
        # (low std) carry more weight -- the "high-confidence step" heuristic
        # from ThinkPRM (arXiv:2504.16828).
        # stdev always succeeds with 3 values (the tuple shape is fixed).
        weights = [
            1.0 / (statistics.stdev([ising, tier0r, tier0u]) + 1e-6)
            for ising, tier0r, tier0u in per_step_verifier_scores
        ]
        total_weight = sum(weights)
        return sum(w * t for w, t in zip(weights, totals)) / total_weight

    raise ValueError(
        f"Unknown aggregation method {method!r}. "
        "Choose from: mean, last, min, product, uncertainty_weighted."
    )


_SUPPORTED_METHODS: tuple[str, ...] = (
    "mean",
    "last",
    "min",
    "product",
    "uncertainty_weighted",
)


def compute_per_step_verifier_scores(
    steps: list[str],
    verifiers: object,
) -> list[tuple[float, float, float]]:
    """Score each reasoning step with the three per-step FoVer verifiers.

    Returns one (ising, tier0r, tier0u) tuple per non-empty step.  Empty steps
    and the ``<think>``/``</think>`` delimiters that the MATH corpus includes
    are filtered out because they carry no reasoning content to score.

    Parameters
    ----------
    steps:
        Parsed reasoning-step strings (e.g., ``sample['reasoning_steps']``).
    verifiers:
        A ``_Verifiers`` instance from
        ``carnot.phase3.p01_trained_energy_reranker``.

    Returns
    -------
    list of (ising, tier0r, tier0u) tuples, one per scorable step.
    Returns an empty list if no scorable steps remain after filtering.
    """
    _SKIP = frozenset({"<think>", "</think>", ""})
    result: list[tuple[float, float, float]] = []
    for step in steps:
        stripped = step.strip()
        if stripped in _SKIP:
            continue
        result.append(
            (
                float(verifiers.ising.energy(stripped)),
                float(verifiers.tier0r.score(stripped)),
                float(verifiers.tier0u.score(stripped)),
            )
        )
    return result


def binary_auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC of ``scores`` as a binary classifier for ``labels``.

    Higher score predicts positive (label=1, i.e. correct).  For
    energy-as-predictor pass ``-energy`` so that lower energy implies a
    higher score and thus predicts correct.

    Returns 0.5 when the label set is degenerate (all positive or all
    negative) -- AUROC is undefined in that case.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5

    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def compute_aggregation_auroc(
    records: list[dict],
    verifiers: object,
    method: str,
) -> dict:
    """Compute final-correctness AUROC for one step-to-final aggregation method.

    Iterates over all records, scoring each candidate's steps with the FoVer
    verifiers and aggregating via ``method``.  Returns the AUROC plus diagnostic
    counts.

    Parameters
    ----------
    records:
        Problem dicts from the level-3 in-band corpus.  Each record must have:
        ``gold_answer`` (or ``gold_answer_norm``), and ``samples`` where each
        sample has ``reasoning_steps``, ``correct``, and optionally
        ``extracted_answer``.
    verifiers:
        A ``_Verifiers`` bundle (pre-built for reuse).
    method:
        One of the five supported aggregation names.

    Returns
    -------
    dict with keys:
        ``auroc``            -- float, the final-correctness AUROC
        ``n_candidates``     -- int, total scored candidates
        ``n_correct``        -- int, candidates with label=1
        ``n_empty_steps``    -- int, candidates with no scorable steps
        ``agg_scores``       -- list[float] raw aggregate energies (for diagnostics)
    """
    scores: list[float] = []  # -aggregate_energy (higher = predicts correct)
    labels: list[int] = []
    n_empty = 0
    raw_energies: list[float] = []

    for rec in records:
        samples = rec.get("samples") or []
        for s in samples:
            steps = s.get("reasoning_steps") or []
            verifier_scores = compute_per_step_verifier_scores(steps, verifiers)

            agg_e = aggregate_step_energies(verifier_scores, method)
            raw_energies.append(agg_e)
            scores.append(-agg_e)  # negate so higher = predicts correct

            if not verifier_scores:
                n_empty += 1

            label = 1 if s.get("correct") else 0
            labels.append(label)

    auroc = binary_auroc(scores, labels)
    n_correct = sum(labels)

    return {
        "auroc": auroc,
        "n_candidates": len(labels),
        "n_correct": n_correct,
        "n_empty_steps": n_empty,
        "agg_scores": raw_energies,
    }
