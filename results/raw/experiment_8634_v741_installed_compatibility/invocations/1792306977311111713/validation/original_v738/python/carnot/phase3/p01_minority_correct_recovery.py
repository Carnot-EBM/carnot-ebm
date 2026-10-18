"""P0.1 Minority-Correct Recovery analysis for the HEADROOM corpus.

Spec: REQ-KONA-3473

WHY THIS MODULE EXISTS:

exp3472 tests whether a PROCESS-AWARE step-level energy beats self-consistency
(SC) on a HEADROOM corpus (SC in [0.40, 0.78]). This module explains the
MECHANISM by answering three questions for each energy type:

  1. Does the energy predict correctness at the candidate level?
     (Spearman correlation, AUROC of -energy as a binary correctness classifier)

  2. Does the energy argmin pick the correct answer more often than SC?
     (within-problem argmin-correct rate)

  3. Among problems where the correct answer is NOT the SC majority answer
     (the "minority-correct" case), does the energy rank that correct answer
     first?

The third question is the CRUX. SC beats any selector at ceiling (GSM8K) because
the minority-correct fraction is near zero: almost every correct answer IS the
majority answer. The HEADROOM corpus changes this: SC leaves real errors on the
table, and those errors are exactly the minority-correct problems. If the energy
recovers those problems, it WILL beat SC — if not, the ceiling is in the energy,
not the benchmark.

Key terms (plain-English, for engineers who are not EBM specialists):

  minority-correct problem : A problem where the correct final answer is NOT
      chosen by simple majority vote over the k sampled solutions. These are the
      problems where a good selector can "save" what majority vote got wrong.

  minority_correct_fraction : count(minority-correct problems) /
      count(all problems). Near 0 = benchmark is at SC ceiling (like GSM8K).
      Near 1 = SC is barely better than random and any decent selector wins.

  minority_correct_recovery_rate : count(minority-correct problems where
      energy argmin = correct answer) / count(minority-correct problems).
      If this is > 0.5, the energy preferentially routes toward the correct
      answer on exactly the problems where SC fails — the direct causal
      explanation for the energy beating SC in exp3472.

  AUROC : Area Under the ROC Curve. 0.5 = random; 1.0 = perfect. Here it
      measures how well -energy separates correct from incorrect candidates
      across all candidates in the held-out set.

  Spearman ρ : Rank correlation. Negative means lower energy → more often
      correct (the theoretical prediction). Near 0 means energy is uninformative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# AUROC (no scipy dependency — O(n log n) sort-based implementation)
# ---------------------------------------------------------------------------


def binary_auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC of `scores` as a binary classifier for `labels`.

    Parameters
    ----------
    scores : list[float]
        Higher score = predicts positive (label=1). For energy-as-predictor
        pass **-energy** so that lower energy → higher score → predicts correct.
    labels : list[int]
        Binary ground-truth labels (1 = correct, 0 = incorrect).

    Returns
    -------
    float
        AUROC in [0, 1]. Returns 0.5 if there are no positive or no negative
        examples (degenerate case — cannot rank).

    Notes
    -----
    Uses the trapezoidal rule over the sorted-pairs formulation, which is
    equivalent to Wilcoxon-Mann-Whitney and handles ties correctly via
    averaging.  No external library required.
    """
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    if not pos_scores or not neg_scores:
        return 0.5  # degenerate: all same class, AUROC undefined

    # AUROC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    # Brute-force O(n*m) — fine for the candidate counts we have (~hundreds).
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between two equal-length sequences.

    Returns the Pearson correlation of the rank-transformed x and y.
    Tied values get the average rank.  Returns 0.0 for degenerate input
    (fewer than 2 observations).

    Parameters
    ----------
    x, y : list[float]
        The two sequences to correlate. Must have the same length.
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    def _ranks(seq: list[float]) -> list[float]:
        sorted_pairs = sorted(enumerate(seq), key=lambda t: t[1])
        rank_map = [0.0] * n
        i = 0
        while i < n:
            j = i
            # Find extent of the tied run.
            while j < n - 1 and sorted_pairs[j][1] == sorted_pairs[j + 1][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-based
            for k in range(i, j + 1):
                rank_map[sorted_pairs[k][0]] = avg_rank
            i = j + 1
        return rank_map

    rx = _ranks(x)
    ry = _ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_rx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_ry) ** 2 for b in ry))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


# ---------------------------------------------------------------------------
# Majority vote (SC prediction) — identical semantics to p01_energy_vote_scoring
# ---------------------------------------------------------------------------


def _majority_vote_answer(answers: list) -> object:
    """Return the most-frequent non-None answer; ties broken by first appearance."""
    counts: dict = {}
    order: list = []
    for a in answers:
        if a is None:
            continue
        if a not in counts:
            counts[a] = 0
            order.append(a)
        counts[a] += 1
    if not order:
        return None
    return max(order, key=lambda a: counts[a])


# ---------------------------------------------------------------------------
# Data structure for minority-correct analysis results
# ---------------------------------------------------------------------------


@dataclass
class MinorityCorrectResult:
    """Results of the minority-correct recovery analysis on a held-out split.

    Attributes
    ----------
    n_candidates : int
        Total candidates scored (across all problems in the held-out set).
    n_problems : int
        Number of distinct problems in the held-out set.
    process_energy_correctness_auroc : float
        AUROC of -process_energy as a binary correctness classifier.
        0.5 = random; > 0.5 = lower energy → more often correct.
    trained_energy_correctness_auroc : float
        AUROC of the trained reranker's P(correct) as a correctness classifier.
    process_energy_spearman : float
        Spearman ρ(process_energy, correctness_label).
        Negative = lower energy → more often correct.
    trained_energy_spearman : float
        Spearman ρ(trained_energy, correctness_label).
    within_problem_argmin_correct_rate_process : float
        Fraction of problems where the lowest-process-energy candidate is correct.
    minority_correct_fraction : float
        Fraction of problems where the SC majority answer is WRONG (= the
        correct answer is a minority vote).  This is the "headroom" the
        benchmark provides over GSM8K.
    minority_correct_recovery_rate_process : float
        Among minority-correct problems, fraction where the process-energy
        argmin picks the correct (minority) answer.  > 0.5 means the energy
        preferentially recovers the problems SC gets wrong — the mechanism
        that explains why the energy beats SC.
    minority_correct_recovery_rate_trained : float
        Same metric for the trained reranker energy.
    n_minority_correct_problems : int
        Raw count of minority-correct problems (denominator for recovery rate).
    """

    n_candidates: int
    n_problems: int
    process_energy_correctness_auroc: float
    trained_energy_correctness_auroc: float
    process_energy_spearman: float
    trained_energy_spearman: float
    within_problem_argmin_correct_rate_process: float
    minority_correct_fraction: float
    minority_correct_recovery_rate_process: float
    minority_correct_recovery_rate_trained: float
    n_minority_correct_problems: int


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------


def compute_minority_correct_recovery(
    records: list[dict],
    process_energies_per_problem: list[list[float]],
    trained_probas_per_problem: list[list[float]],
) -> MinorityCorrectResult:
    """Compute minority-correct recovery for the given held-out problems.

    Parameters
    ----------
    records : list[dict]
        Each record has:
          - "gold": str   — the ground-truth answer string
          - "samples": list[dict], each with:
              - "answer": str | None — the predicted answer
        The list is the HELD-OUT split (no train/test leakage).
    process_energies_per_problem : list[list[float]]
        For each problem, the per-candidate process (step-level FoVer) energy.
        Lower is better (more internally consistent steps).
        Parallel to records and to each problem's samples list.
    trained_probas_per_problem : list[list[float]]
        For each problem, the trained reranker's P(correct) per candidate.
        Higher is better. Trained energy = 1 - P(correct) (conceptually).
        Parallel to records and to each problem's samples list.

    Returns
    -------
    MinorityCorrectResult
        The full set of mechanism-level metrics.  See the class docstring for
        field-by-field semantics.
    """
    n_problems = len(records)
    all_process_scores: list[float] = []  # -process_energy (higher = predicts correct)
    all_trained_scores: list[float] = []  # trained P(correct)
    all_labels: list[int] = []

    process_argmin_correct: list[bool] = []
    sc_wrong: list[bool] = []  # True if SC is wrong (= minority-correct problem)
    process_recovers_minority: list[bool] = []
    trained_recovers_minority: list[bool] = []

    for rec, proc_es, trained_ps in zip(
        records, process_energies_per_problem, trained_probas_per_problem
    ):
        gold = str(rec["gold"]).strip()
        samples = rec.get("samples") or []
        answers = [str(s.get("answer", "")).strip() if s.get("answer") is not None else None
                   for s in samples]

        # Correctness labels for each candidate.
        correct = [int(a is not None and a == gold) for a in answers]

        # Accumulate candidate-level arrays for global AUROC / Spearman.
        # -process_energy: higher = predicts correct
        all_process_scores.extend(-e for e in proc_es)
        all_trained_scores.extend(trained_ps)
        all_labels.extend(correct)

        # Within-problem argmin-correct (process energy).
        if proc_es:
            best_idx = min(range(len(proc_es)), key=lambda i: proc_es[i])
            process_argmin_correct.append(
                answers[best_idx] is not None and answers[best_idx] == gold
                if best_idx < len(answers) else False
            )
        else:
            process_argmin_correct.append(False)

        # SC prediction for this problem.
        sc_ans = _majority_vote_answer(answers)
        sc_is_correct = sc_ans is not None and sc_ans == gold
        sc_wrong.append(not sc_is_correct)

        # Minority-correct recovery: only relevant when SC is wrong.
        if not sc_is_correct:
            # Process energy: does argmin-process pick the correct answer?
            if proc_es:
                pbest = min(range(len(proc_es)), key=lambda i: proc_es[i])
                process_recovers_minority.append(
                    answers[pbest] is not None and answers[pbest] == gold
                    if pbest < len(answers) else False
                )
            else:
                process_recovers_minority.append(False)

            # Trained energy: does argmax-P(correct) pick the correct answer?
            if trained_ps:
                tbest = max(range(len(trained_ps)), key=lambda i: trained_ps[i])
                trained_recovers_minority.append(
                    answers[tbest] is not None and answers[tbest] == gold
                    if tbest < len(answers) else False
                )
            else:
                trained_recovers_minority.append(False)

    n_candidates = len(all_labels)
    n_minority = sum(sc_wrong)

    proc_auroc = binary_auroc(all_process_scores, all_labels)
    trained_auroc = binary_auroc(all_trained_scores, all_labels)

    proc_spearman = spearman_correlation(
        [-s for s in all_process_scores],  # process_energy (not negated)
        [float(l) for l in all_labels],
    )
    trained_spearman = spearman_correlation(
        [1.0 - s for s in all_trained_scores],  # trained_energy = 1 - P(correct)
        [float(l) for l in all_labels],
    )

    argmin_correct_rate = (
        sum(process_argmin_correct) / n_problems if n_problems > 0 else 0.0
    )
    minority_fraction = n_minority / n_problems if n_problems > 0 else 0.0

    proc_recovery = (
        sum(process_recovers_minority) / n_minority if n_minority > 0 else 0.0
    )
    trained_recovery = (
        sum(trained_recovers_minority) / n_minority if n_minority > 0 else 0.0
    )

    return MinorityCorrectResult(
        n_candidates=n_candidates,
        n_problems=n_problems,
        process_energy_correctness_auroc=proc_auroc,
        trained_energy_correctness_auroc=trained_auroc,
        process_energy_spearman=proc_spearman,
        trained_energy_spearman=trained_spearman,
        within_problem_argmin_correct_rate_process=argmin_correct_rate,
        minority_correct_fraction=minority_fraction,
        minority_correct_recovery_rate_process=proc_recovery,
        minority_correct_recovery_rate_trained=trained_recovery,
        n_minority_correct_problems=n_minority,
    )
