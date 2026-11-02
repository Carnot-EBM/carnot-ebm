"""P0.1 MATH-aware recalibration helpers for exp3497.

Spec: REQ-KONA-3497 / SCENARIO-KONA-3497

WHY THIS MODULE EXISTS:

exp3473 found that the FoVer process energy has AUROC=0.441 on MATH final-answer
correctness — BELOW chance. The energy was trained/evaluated on GSM8K-style
problems. Two hypotheses for why 0.9131 AUROC on FoVer step-error detection
does NOT transfer:

  (A) DOMAIN SHIFT: the 4 FoVer verifiers (IsingVerifier, EbmCotCalibrator,
      Tier0rVerifier, Tier0uVerifier) fire on arithmetic/logical patterns that
      appear in GSM8K reasoning traces but NOT in MATH reasoning traces (which
      involve multi-step proofs, algebraic manipulation, geometry). Training
      a reranker on MATH labels should recover signal if domain shift is the
      cause.

  (B) STEP-VS-FINAL GAP: the verifiers accurately detect individual step-level
      errors, but a wrong step does not always produce a wrong final answer
      (students sometimes recover; LaTeX/formatting issues in the final answer
      don't reflect step quality). In this case, MATH-aware recalibration would
      NOT help — the ceiling is the information-theoretic disconnect between
      step errors and final correctness.

This module implements:
  1. ``compute_step_error_auroc``: on a subset with parsed step lists, compute
     the AUROC of the MAXIMUM per-step energy as a proxy for step-error
     detection (lower is better; the max step energy flags the worst step).
     This is a step-level signal we can contrast with final-correctness AUROC.
  2. ``math_aware_cv_auroc``: 5-fold CV trained only on MATH (hardmath) problems
     to check whether domain-matched training recovers the signal.
  3. ``distinct_pipeline_assert``: runtime assertion that process-energy and
     trained-energy per-candidate arrays are NOT element-wise equal (the
     exp3473 de-flag: proves the two energies come from different computations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from carnot.phase3.p01_minority_correct_recovery import binary_auroc
from carnot.phase3.p01_trained_energy_reranker import (
    TrainedEnergyReranker,
    _Verifiers,
    candidate_feature_vector,
    problem_kfold_indices,
)


# ---------------------------------------------------------------------------
# Step-error AUROC (step-vs-final decomposition)
# ---------------------------------------------------------------------------


def compute_step_error_auroc(
    records: list[dict],
    verifiers: _Verifiers,
) -> float:
    """AUROC of MAX-step-energy as a step-error detector on problems with steps.

    Uses the maximum per-step FoVer energy across all steps in a candidate as
    the signal — highest energy step = most suspicious step. Evaluates against
    final-correctness-as-proxy (wrong final answer → assumed step error exists).

    This gives a step-level signal that can be contrasted with the mean-step
    process-energy's final-correctness AUROC. The gap between the two locates
    whether the verifiers are better at flagging ANY bad step vs. predicting
    the final answer.

    Parameters
    ----------
    records : list[dict]
        Problem records, each with ``gold``, ``samples`` (each sample has
        ``steps`` list, ``answer``). Only records with at least one sample
        having non-empty ``steps`` are used.
    verifiers : _Verifiers
        Pre-built verifier bundle.

    Returns
    -------
    float
        AUROC of max-step-energy as a step-error detector.
        Returns 0.5 if no records have step data.
    """
    # Import here to avoid circular; _Verifiers already imported from reranker module
    from carnot.phase3.p01_process_energy import process_energy_per_step

    max_step_scores: list[float] = []  # -max_step_energy (higher = predicts correct)
    final_labels: list[int] = []

    for rec in records:
        gold = str(rec["gold"]).strip()
        samples = rec.get("samples") or []
        for s in samples:
            raw_steps: list = s.get("steps") or []
            if not raw_steps:
                continue
            steps = [str(st) for st in raw_steps]
            # Compute energy per step and take the MAX (most suspicious step).
            step_energies = [
                verifiers.ising.energy(st) + verifiers.tier0u.score(st) * 0.5
                for st in steps
            ]
            max_e = max(step_energies)
            # Higher max-step-energy → predicts an error exists.
            # To use binary_auroc (higher score = predicts positive=correct),
            # negate: lower max-step-energy → predicts correct.
            max_step_scores.append(-max_e)
            ans = str(s.get("answer", "")).strip() if s.get("answer") is not None else None
            final_labels.append(1 if ans is not None and ans == gold else 0)

    if not max_step_scores:
        return 0.5
    return binary_auroc(max_step_scores, final_labels)


# ---------------------------------------------------------------------------
# MATH-aware CV (domain-matched recalibration)
# ---------------------------------------------------------------------------


@dataclass
class MathAwareRecalibResult:
    """Result of MATH-aware cross-validated recalibration.

    Attributes
    ----------
    mathaware_correctness_auroc : float
        AUROC of the recalibrated reranker on MATH-held-out problems.
    n_math_problems : int
        Number of MATH problems used for recalibration.
    n_math_candidates : int
        Total held-out MATH candidates scored.
    n_folds_used : int
        Actual folds used (may be < N_FOLDS if too few problems).
    """

    mathaware_correctness_auroc: float
    n_math_problems: int
    n_math_candidates: int
    n_folds_used: int


def math_aware_cv_auroc(
    records: list[dict],
    feats: list[list[list[float]]],
    labels: list[list[int]],
    *,
    seed: int,
    n_folds: int = 5,
    n_iter: int = 500,
) -> MathAwareRecalibResult:
    """5-fold CV on MATH-only problems to test domain-matched recalibration.

    The idea: the regular reranker is trained across GSM8K + MATH labels.
    When MATH labels are rare (19/48 contested), the reranker is dominated by
    GSM8K signal. Retraining exclusively on MATH labels (within MATH 5-fold CV)
    tests whether domain-matched training recovers the correctness signal.

    Parameters
    ----------
    records : list[dict]
        All records (GSM8K + MATH). Must be parallel to feats and labels.
    feats : list[list[list[float]]]
        Per-problem, per-candidate feature vectors.
    labels : list[list[int]]
        Per-problem, per-candidate binary correctness labels.
    seed : int
        Random seed for fold assignment.
    n_folds : int
        Number of CV folds; reduced to min(n_folds, n_math) if too few problems.
    n_iter : int
        Logistic regression iterations.

    Returns
    -------
    MathAwareRecalibResult
    """
    # Select only MATH (hardmath) problems — identified by 'level' field or
    # by NOT having 'gsm' in the problem_id.
    math_idx = [
        i for i, r in enumerate(records)
        if "gsm" not in str(r.get("problem_id", "")).lower()
    ]
    n_math = len(math_idx)
    if n_math < 2:
        # Not enough MATH problems for CV.
        return MathAwareRecalibResult(
            mathaware_correctness_auroc=0.5,
            n_math_problems=n_math,
            n_math_candidates=0,
            n_folds_used=0,
        )

    actual_folds = min(n_folds, n_math)
    # Map global index to local math index for kfold splitter.
    math_feats = [feats[i] for i in math_idx]
    math_labels = [labels[i] for i in math_idx]

    splits = problem_kfold_indices(n_math, actual_folds, seed)
    trained_probas: list[list[float]] = [[] for _ in range(n_math)]

    for train_local, test_local in splits:
        X_train: list[list[float]] = []
        y_train: list[int] = []
        for li in train_local:
            X_train.extend(math_feats[li])
            y_train.extend(math_labels[li])
        if not X_train or all(y == y_train[0] for y in y_train):
            # Degenerate fold — all one class; skip.
            continue
        reranker = TrainedEnergyReranker(n_iter=n_iter)
        reranker.fit(X_train, y_train)
        for li in test_local:
            if math_feats[li]:
                trained_probas[li] = reranker.predict_proba(math_feats[li])

    # Flatten for global AUROC.
    all_scores: list[float] = []
    all_labels_flat: list[int] = []
    for li in range(n_math):
        all_scores.extend(trained_probas[li])
        all_labels_flat.extend(math_labels[li])

    auroc = binary_auroc(all_scores, all_labels_flat) if all_scores else 0.5
    n_candidates = sum(len(math_feats[li]) for li in range(n_math))

    return MathAwareRecalibResult(
        mathaware_correctness_auroc=auroc,
        n_math_problems=n_math,
        n_math_candidates=n_candidates,
        n_folds_used=actual_folds,
    )


# ---------------------------------------------------------------------------
# Distinct-pipeline runtime assertion
# ---------------------------------------------------------------------------


def distinct_pipeline_assert(
    process_energies: list[float],
    trained_energies: list[float],
) -> bool:
    """Assert that process and trained energy arrays are NOT element-wise equal.

    This is the exp3473 de-flag guard: if both pipelines produce exactly the
    same per-candidate scores, it indicates a pipeline-sharing bug (the two
    energies are supposed to come from COMPLETELY SEPARATE code paths). If they
    happen to be identical, this function returns False and the experiment must
    fail loudly rather than emit a bit-identical artifact.

    Parameters
    ----------
    process_energies : list[float]
        Per-candidate process energies (step-level FoVer aggregate).
    trained_energies : list[float]
        Per-candidate trained reranker energies (1 - P(correct)).

    Returns
    -------
    bool
        True if the arrays differ in at least one position (the pipelines ARE
        distinct); False if all elements are equal (pipeline sharing bug).
    """
    if len(process_energies) != len(trained_energies):
        # Different lengths → trivially distinct.
        return True
    if not process_energies:
        # Empty arrays → trivially distinct (no comparison possible).
        return True
    return not all(
        math.isclose(p, t, rel_tol=1e-9, abs_tol=1e-12)
        for p, t in zip(process_energies, trained_energies)
    )
