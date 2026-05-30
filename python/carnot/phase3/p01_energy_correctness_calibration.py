"""Energy-correctness calibration audit for the P0.1 GSM8K cached corpus.

Spec: REQ-KONA-3450

WHY THIS MODULE EXISTS (plain-language explanation):

Exp 3449 showed that energy-based selection matches self-consistency (SC) on the
GSM8K corpus but does not beat it. The open question is WHY: is the energy
mechanism fundamentally broken (energy does not correlate with correctness at all),
or is it merely not strong enough to overcome the information advantage SC gets
from seeing multiple samples?

This module answers that question by treating the energy score as a binary
CLASSIFIER: does giving a candidate lower energy also make it more likely to be
correct? We measure three numbers:

  1. Spearman rank correlation (energy vs correctness across all candidates):
     A negative value means lower energy → more often correct, which is the
     hypothesis. A value near zero means energy carries no correctness signal.

  2. AUROC of -energy as a binary correctness classifier:
     This is the standard ROC metric. 0.5 = random; >0.5 = energy is informative;
     1.0 = energy perfectly predicts correctness. Values consistently at or below
     0.5 mean the energy mechanism is broken at its root.

  3. Within-problem argmin correct rate:
     For each problem, take the candidate with the LOWEST energy. Was it correct?
     This directly explains the exp3449 energy-argmin accuracy number — if 60% of
     argmin-energy picks are correct and 80% of majority-vote picks are correct,
     the energy selector is losing to majority vote AT THE CANDIDATE LEVEL.

The verifier ensemble used (IsingVerifier + EbmCotCalibrator) is the same
parameter-free, deterministic heuristic used in exp3449, so the calibration and
the selection experiment are grounded in the SAME energy function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from carnot.phase3.p01_energy_vote_scoring import (
    EbmCotCalibrator,
    IsingVerifier,
    candidate_energy,
    extract_steps,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CandidateRecord:
    """One scored candidate with its correctness label and energy.

    Parameters
    ----------
    problem_id : int | str
        Which GSM8K problem this candidate belongs to.
    text : str
        The raw candidate generation text.
    answer : int
        The numeric answer extracted from the candidate text.
    gold : int
        The ground-truth answer for this problem.
    energy : float
        Verifier-ensemble energy (lower = more internally consistent).
    """

    problem_id: int | str
    text: str
    answer: int
    gold: int
    energy: float

    @property
    def is_correct(self) -> bool:
        """True when the candidate's extracted answer matches the gold answer."""
        return self.answer == self.gold


@dataclass
class CalibrationResult:
    """Aggregated calibration metrics for the energy-correctness audit.

    Parameters
    ----------
    n_candidates : int
        Total candidates scored (all problems x k samples).
    n_problems : int
        Number of distinct problems in the corpus.
    energy_correctness_spearman : float
        Spearman rank correlation between energy and correctness (binary).
        Negative means lower energy → correct more often.
    energy_as_correctness_auroc : float
        AUROC when using -energy as a classifier for correctness.
        >0.5 means energy carries positive signal; 0.5 = chance.
    within_problem_argmin_correct_rate : float
        Fraction of problems where the lowest-energy candidate is correct.
    correct_mean_energy : float
        Mean energy of correct candidates.
    incorrect_mean_energy : float
        Mean energy of incorrect candidates.
    energy_gap : float
        incorrect_mean_energy - correct_mean_energy; positive means correct
        candidates have lower energy as expected.
    """

    n_candidates: int
    n_problems: int
    energy_correctness_spearman: float
    energy_as_correctness_auroc: float
    within_problem_argmin_correct_rate: float
    correct_mean_energy: float
    incorrect_mean_energy: float
    energy_gap: float


# ---------------------------------------------------------------------------
# Core scoring helpers
# ---------------------------------------------------------------------------


def compute_candidate_energies(
    problems: list[dict],
    arithmetic_weight: float = 1.0,
    contradiction_weight: float = 1.0,
) -> list[CandidateRecord]:
    """Score every cached candidate in the corpus with the verifier-energy ensemble.

    This is a read-only operation over the cached JSONL corpus — no live model is
    loaded. We instantiate IsingVerifier (arithmetic-violation scorer) and
    EbmCotCalibrator (adjacent-step contradiction scorer) once, then call
    candidate_energy() for every (problem, sample) pair.

    WHY PARAMETER-FREE: the energy function is deliberately not trained on the
    GSM8K answers. If we fitted the energy to correctness labels, the calibration
    test would be circular — of course a fitted classifier predicts correctly.
    Using the same un-tuned heuristics as exp3449 means any calibration signal
    (or its absence) is a genuine diagnostic of the mechanism, not an artefact of
    label leakage.

    Parameters
    ----------
    problems : list[dict]
        Decoded JSON records from data/p01_gsm8k_generations.jsonl. Each record
        has keys: problem_id, gold, samples (list with text, answer,
        mean_token_logprob fields).
    arithmetic_weight, contradiction_weight : float
        Passed through to candidate_energy() unchanged; same defaults as exp3449
        so the energies are directly comparable.

    Returns
    -------
    list[CandidateRecord]
        One entry per (problem, sample) pair, in corpus order.
    """
    ising = IsingVerifier()
    ebmcot = EbmCotCalibrator()
    records: list[CandidateRecord] = []

    for problem in problems:
        gold = int(problem["gold"])
        pid = problem["problem_id"]
        for sample in problem.get("samples") or []:
            text = sample.get("text") or ""
            answer = int(sample.get("answer") or 0)
            energy = candidate_energy(
                text,
                ising,
                ebmcot,
                arithmetic_weight=arithmetic_weight,
                contradiction_weight=contradiction_weight,
            )
            records.append(
                CandidateRecord(
                    problem_id=pid,
                    text=text,
                    answer=answer,
                    gold=gold,
                    energy=energy,
                )
            )

    return records


# ---------------------------------------------------------------------------
# Statistical helpers (no scipy dependency — minimal stdlib implementations)
# ---------------------------------------------------------------------------


def _ranks(values: list[float]) -> list[float]:
    """Compute average ranks for a list of floats (ties share average rank).

    This is the standard 'average' tiebreaking method used by scipy.stats.spearmanr.
    Ranking is 1-based. We implement it from stdlib to avoid scipy being a hard
    dependency in the module itself (scipy is used in tests for cross-validation).

    Parameters
    ----------
    values : list[float]
        Input sequence to rank.

    Returns
    -------
    list[float]
        Average ranks in the same index order as the input.
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient for two equal-length sequences.

    Correlation between energy values (x) and correctness labels (y, 0/1).
    A negative result means lower energy → correct more often (the hypothesis).

    This implements the Pearson correlation on the ranks, which is the standard
    definition of Spearman's ρ. Returns 0.0 when n < 2 or when either sequence
    has zero variance (all values identical).

    Parameters
    ----------
    x, y : list[float]
        Equal-length sequences. y is typically a binary 0/1 correctness label.

    Returns
    -------
    float
        Spearman ρ in [-1, 1]. Returns 0.0 for degenerate inputs.
    """
    n = len(x)
    if n < 2:
        return 0.0

    rx = _ranks(x)
    ry = _ranks(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    denom_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if denom_x < 1e-12 or denom_y < 1e-12:
        return 0.0

    return num / (denom_x * denom_y)


def binary_auroc(labels: list[int], scores: list[float]) -> float:
    """AUROC for a binary classifier using the Wilcoxon-Mann-Whitney statistic.

    For this task, scores = -energy (higher = more confident of correctness).
    labels = 1 for correct candidates, 0 for incorrect.

    AUROC = P(score(positive) > score(negative)) which equals the fraction of
    (positive, negative) pairs where the positive has a higher score. This is
    an O(n_pos * n_neg) computation; for n~300 it is negligible.

    Parameters
    ----------
    labels : list[int]
        Binary labels; 1 = correct, 0 = incorrect.
    scores : list[float]
        Classifier scores; higher = more likely correct. For energy we pass -energy.

    Returns
    -------
    float
        AUROC in [0, 1]. Returns 0.5 for degenerate inputs (no positives/negatives).
    """
    pos_scores = [s for l, s in zip(labels, scores) if l == 1]
    neg_scores = [s for l, s in zip(labels, scores) if l == 0]

    if not pos_scores or not neg_scores:
        return 0.5

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    wins = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5  # tied counts as half-win (standard convention)

    return wins / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Per-problem argmin metric
# ---------------------------------------------------------------------------


def within_problem_argmin_rate(records: list[CandidateRecord]) -> float:
    """Fraction of problems where the minimum-energy candidate is correct.

    For each problem we pick the candidate with the lowest energy (same as
    energy_argmin() in exp3449) and check whether its answer matches gold. This
    directly explains the energy-argmin accuracy reported in exp3449 — if the
    rate here matches that accuracy, the audit and the selection experiment are
    grounded consistently in the same energy function.

    Ties in energy are broken by taking the first occurrence in corpus order.

    Parameters
    ----------
    records : list[CandidateRecord]
        All scored candidates, as returned by compute_candidate_energies().

    Returns
    -------
    float
        Fraction of problems [0, 1] where argmin-energy pick is correct.
        Returns 0.0 if no problems are present.
    """
    # Group by problem_id
    problems: dict[int | str, list[CandidateRecord]] = {}
    for rec in records:
        problems.setdefault(rec.problem_id, []).append(rec)

    if not problems:
        return 0.0

    correct = 0
    for cands in problems.values():
        best = min(cands, key=lambda r: r.energy)
        if best.is_correct:
            correct += 1

    return correct / len(problems)


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------


def run_calibration_audit(problems: list[dict]) -> CalibrationResult:
    """Run the full energy-correctness calibration audit over the cached corpus.

    Steps:
      1. Score every (problem, sample) pair with the verifier-energy ensemble.
      2. Compute Spearman ρ between energy and correctness.
      3. Compute AUROC of -energy as a binary correctness classifier.
      4. Compute within-problem argmin-correct rate.
      5. Compute per-class mean energy and the energy gap.

    Parameters
    ----------
    problems : list[dict]
        Decoded JSONL records from data/p01_gsm8k_generations.jsonl.

    Returns
    -------
    CalibrationResult
        All calibration metrics in a single dataclass.
    """
    records = compute_candidate_energies(problems)

    energies = [r.energy for r in records]
    labels = [1 if r.is_correct else 0 for r in records]

    spearman = spearman_correlation(energies, [float(l) for l in labels])
    auroc = binary_auroc(labels, [-e for e in energies])  # -energy so higher = better
    argmin_rate = within_problem_argmin_rate(records)

    correct_energies = [r.energy for r in records if r.is_correct]
    incorrect_energies = [r.energy for r in records if not r.is_correct]

    correct_mean = sum(correct_energies) / len(correct_energies) if correct_energies else 0.0
    incorrect_mean = (
        sum(incorrect_energies) / len(incorrect_energies) if incorrect_energies else 0.0
    )
    energy_gap = incorrect_mean - correct_mean  # positive when correct < incorrect (expected)

    n_problems = len({r.problem_id for r in records})

    return CalibrationResult(
        n_candidates=len(records),
        n_problems=n_problems,
        energy_correctness_spearman=spearman,
        energy_as_correctness_auroc=auroc,
        within_problem_argmin_correct_rate=argmin_rate,
        correct_mean_energy=correct_mean,
        incorrect_mean_energy=incorrect_mean,
        energy_gap=energy_gap,
    )
