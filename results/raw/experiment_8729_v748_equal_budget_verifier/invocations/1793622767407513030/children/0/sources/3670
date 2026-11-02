"""Trained-energy-vs-correctness calibration for the P0.1 GSM8K corpus.

Spec: REQ-KONA-3461, SCENARIO-KONA-3461

WHY THIS MODULE EXISTS:

exp3450 measured the UNTRAINED verifier energy vs correctness and found AUROC = 0.516,
essentially chance. This module answers the follow-up: does a TRAINED logistic-regression
energy reranker (the one from exp3460) carry meaningful correctness signal?

We use the SAME 5-fold, problem-level cross-validation split as exp3460 so the
held-out candidates are identical to what exp3460 scored. For each fold we:

  1. Train a fresh EnergyReranker on the train-fold problems (same as exp3460).
  2. Predict P(correct) for every held-out candidate — that is the trained energy
     signal (higher P = lower conceptual energy).
  3. Also record the FoVer ensemble energy for each held-out candidate.

After collecting all held-out scores we compute three metrics for each energy:

  * Spearman rank correlation vs correctness label (negative = lower energy → correct).
  * AUROC of -energy as a binary correctness classifier (>0.5 = energy is informative).
  * Within-problem argmin correct rate (pick lowest-energy per problem; is it right?).

These three metrics explain WHY exp3460's energy-selection accuracy came out the way
it did: if AUROC is near chance (0.516), the energy cannot route correct answers to the
top regardless of the selection strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from carnot.phase3.p01_energy_correctness_calibration import (
    binary_auroc,
    spearman_correlation,
)
from carnot.phase3.p01_trained_energy_reranker import (
    TrainedEnergyReranker,
    _Verifiers,
    candidate_feature_vector,
    fover_candidate_energy,
    problem_kfold_indices,
)


@dataclass
class TrainedCalibrationResult:
    """Per-energy calibration metrics on the held-out CV candidates.

    Parameters
    ----------
    n_candidates_heldout : int
        Total held-out candidates scored (sum over all folds).
    trained_energy_correctness_auroc : float
        AUROC of P(correct) as a binary correctness classifier. Higher is better;
        0.5 = chance; 1.0 = perfect.
    trained_energy_correctness_spearman : float
        Spearman rank correlation between (1 - P(correct)) and the binary correctness
        label. Negative = lower trained-energy → correct more often (the hypothesis).
    fover_energy_correctness_auroc : float
        AUROC of -fover_energy as a binary correctness classifier.
    fover_energy_correctness_spearman : float
        Spearman ρ between fover_energy and correctness label.
    trained_energy_auroc_lift_over_untrained : float
        trained_energy_correctness_auroc minus 0.516 (the exp3450 untrained baseline).
        Positive means training fixed the energy's correctness-tracking ability.
    within_problem_argmin_correct_rate_trained : float
        Fraction of problems where the highest-P(correct) candidate (= lowest trained
        energy) is correct. Directly explains exp3460's trained-energy selection accuracy.
    within_problem_argmin_correct_rate_fover : float
        Fraction of problems where the lowest-FoVer-energy candidate is correct.
    n_problems_heldout : int
        Number of distinct problems in the held-out set.
    """

    n_candidates_heldout: int
    trained_energy_correctness_auroc: float
    trained_energy_correctness_spearman: float
    fover_energy_correctness_auroc: float
    fover_energy_correctness_spearman: float
    trained_energy_auroc_lift_over_untrained: float
    within_problem_argmin_correct_rate_trained: float
    within_problem_argmin_correct_rate_fover: float
    n_problems_heldout: int


UNTRAINED_AUROC_BASELINE = 0.516  # exp3450 reference value


def _within_problem_argmin_rate(
    problem_scores: dict[str, list[tuple[float, int]]],
    *,
    higher_is_better: bool,
) -> float:
    """Fraction of problems where the best-energy candidate is correct.

    Parameters
    ----------
    problem_scores : dict mapping problem_id -> list of (score, label) pairs
        score is either P(correct) for trained energy, or -fover_energy for FoVer.
    higher_is_better : bool
        When True, pick the candidate with the HIGHEST score (argmax).
        When False, pick the candidate with the LOWEST score (argmin).

    Returns
    -------
    float
        Fraction of problems [0, 1] where the picked candidate is correct (label=1).
    """
    if not problem_scores:
        return 0.0
    correct = 0
    for pairs in problem_scores.values():
        if not pairs:
            continue
        if higher_is_better:
            best_score, best_label = max(pairs, key=lambda p: p[0])
        else:
            best_score, best_label = min(pairs, key=lambda p: p[0])
        if best_label == 1:
            correct += 1
    return correct / len(problem_scores)


def compute_trained_calibration(
    records: list[dict],
    *,
    seed: int = 20260601,
    n_folds: int = 5,
    reranker_iter: int = 500,
    verifiers: _Verifiers | None = None,
) -> TrainedCalibrationResult:
    """Run the trained-energy calibration audit using the same CV split as exp3460.

    For every held-out candidate we record:
      - The trained reranker's P(correct) (the energy signal).
      - The FoVer ensemble energy (arithmetic + contradiction + Curry-Howard + logical).
      - The binary correctness label (1 if candidate answer == gold, else 0).

    We then compute AUROC and Spearman for each energy type, plus the within-problem
    argmin correct rate that directly explains exp3460's selection accuracy.

    This function uses the SAME parameters (seed, n_folds, reranker_iter) as
    exp3460's score_corpus_trained_cv() so the held-out split and model training
    are exactly reproducible from the same corpus.

    Parameters
    ----------
    records : list[dict]
        Usable rows from data/p01_gsm8k_generations.jsonl (same filtering as exp3460:
        gold present, greedy answer present, >= 5 samples).
    seed : int
        RNG seed for the fold split (should match exp3460's SEED).
    n_folds : int
        Number of cross-validation folds (should match exp3460's N_FOLDS).
    reranker_iter : int
        Gradient-descent iterations per fold's reranker (should match exp3460).
    verifiers : _Verifiers | None
        Pre-built verifier bundle. Instantiated once if None.

    Returns
    -------
    TrainedCalibrationResult
        All calibration metrics.
    """
    verifiers = verifiers or _Verifiers()
    n = len(records)

    # Pre-compute features, FoVer energies, answers, and labels for all candidates.
    # This mirrors score_corpus_trained_cv() exactly so the held-out scoring is
    # grounded in the same features the selection accuracy was computed from.
    feats: list[list[list[float]]] = []
    fover: list[list[float]] = []
    labels_all: list[list[int]] = []
    problem_ids: list[str] = []

    for rec in records:
        gold = rec["gold"]
        pid = str(rec.get("problem_id", ""))
        problem_ids.append(pid)
        samples = rec.get("samples") or []
        rec_feats: list[list[float]] = []
        rec_fover: list[float] = []
        rec_labels: list[int] = []
        for s in samples:
            text = s.get("text", "")
            mlp = s.get("mean_token_logprob")
            rec_feats.append(candidate_feature_vector(text, mlp, verifiers))
            rec_fover.append(fover_candidate_energy(text, verifiers))
            rec_labels.append(1 if s.get("answer") == gold else 0)
        feats.append(rec_feats)
        fover.append(rec_fover)
        labels_all.append(rec_labels)

    splits = problem_kfold_indices(n, n_folds, seed)

    # Collect per-candidate scores from the held-out folds.
    all_trained_scores: list[float] = []  # P(correct) for each held-out candidate
    all_fover_energies: list[float] = []  # FoVer energy for each held-out candidate
    all_labels: list[int] = []  # correctness label for each held-out candidate
    # For within-problem argmin rate, track per-problem scores.
    problem_trained: dict[str, list[tuple[float, int]]] = {}  # pid -> [(P_correct, label)]
    problem_fover: dict[str, list[tuple[float, int]]] = {}    # pid -> [(-fover_e, label)]

    for train_idx, test_idx in splits:
        # Train a fresh reranker on the train-fold problems.
        X_train: list[list[float]] = []
        y_train: list[int] = []
        for pi in train_idx:
            X_train.extend(feats[pi])
            y_train.extend(labels_all[pi])
        reranker = TrainedEnergyReranker(n_iter=reranker_iter)
        reranker.fit(X_train, y_train)

        # Score held-out candidates.
        for pi in test_idx:
            pid = problem_ids[pi]
            proba = reranker.predict_proba(feats[pi]) if feats[pi] else []
            for j, (p_correct, fe, lbl) in enumerate(
                zip(proba, fover[pi], labels_all[pi])
            ):
                all_trained_scores.append(p_correct)
                all_fover_energies.append(fe)
                all_labels.append(lbl)
                problem_trained.setdefault(pid, []).append((p_correct, lbl))
                problem_fover.setdefault(pid, []).append((-fe, lbl))

    n_candidates = len(all_labels)

    # AUROC: higher score = more likely correct.
    # For trained energy: P(correct) is already the score (higher = better).
    # For FoVer energy: -fover_energy (lower energy = better).
    trained_auroc = binary_auroc(all_labels, all_trained_scores)
    fover_auroc = binary_auroc(all_labels, [-e for e in all_fover_energies])

    # Spearman: correlate energy (not P(correct)) with label.
    # Trained energy = 1 - P(correct); negative spearman means lower energy → correct.
    trained_energies_for_spearman = [1.0 - p for p in all_trained_scores]
    trained_spearman = spearman_correlation(
        trained_energies_for_spearman, [float(l) for l in all_labels]
    )
    fover_spearman = spearman_correlation(
        all_fover_energies, [float(l) for l in all_labels]
    )

    # Within-problem argmin: for trained energy pick argmax(P(correct));
    # for FoVer energy pick argmin(fover_energy) i.e. argmax(-fover_energy).
    argmin_trained = _within_problem_argmin_rate(problem_trained, higher_is_better=True)
    argmin_fover = _within_problem_argmin_rate(problem_fover, higher_is_better=True)

    n_problems_heldout = len(problem_trained)

    return TrainedCalibrationResult(
        n_candidates_heldout=n_candidates,
        trained_energy_correctness_auroc=trained_auroc,
        trained_energy_correctness_spearman=trained_spearman,
        fover_energy_correctness_auroc=fover_auroc,
        fover_energy_correctness_spearman=fover_spearman,
        trained_energy_auroc_lift_over_untrained=trained_auroc - UNTRAINED_AUROC_BASELINE,
        within_problem_argmin_correct_rate_trained=argmin_trained,
        within_problem_argmin_correct_rate_fover=argmin_fover,
        n_problems_heldout=n_problems_heldout,
    )
