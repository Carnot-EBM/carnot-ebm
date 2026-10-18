"""P0.1 v5 — Trained energy reranker vs self-consistency on held-out GSM8K.

**Researcher summary:**
    This module is the SCORING substrate for the decisive P0.1 premise test
    (REQ-KONA-3460). Its predecessor, exp3449, showed that an UNTRAINED,
    parameter-free verifier-energy ensemble does NOT beat majority-vote
    self-consistency on GSM8K final-answer selection — the energy-weighted vote
    just degenerated onto plain majority vote, and a follow-up calibration audit
    (exp3450) measured energy-vs-correctness AUROC at 0.516, essentially chance.

    The reasoning literature says the fix is a *trained* energy: arXiv:2505.14999
    (EORM) trains a lightweight energy reward model on outcome-correctness labels
    and gets a real lift on math reasoning. The never-asked question this module
    answers: does a TRAINED outcome-label energy reranker, evaluated on a
    HELD-OUT problem-level split, MATCH or BEAT self-consistency at matched
    compute? Because it consumes the already-cached corpus and trains only a tiny
    logistic-regression reranker, it runs in seconds and cannot idle-timeout.

**Why this design (for engineers who are not EBM specialists):**
    * We never load a live LLM. The corpus (`data/p01_gsm8k_generations.jsonl`)
      already contains, per GSM8K problem, a greedy generation plus `k` sampled
      generations, each with its extracted numeric answer and per-token
      logprobs. All we do is *score* those cached candidates with different
      selection strategies and compare held-out accuracy.

    * The "energy" of a candidate is computed from cheap, deterministic verifier
      heuristics (arithmetic-violation energy, adjacent-step contradiction
      energy, Curry-Howard type-violation score, logical-inconsistency score)
      plus the model's own mean-token logprob. The UNTRAINED predecessor summed
      a fixed subset of these with weight 1.0 — and it did not track correctness.
      Here we instead *train* a small logistic regression to map the feature
      vector to P(correct), using the outcome-correctness label as the target.

    * The single most important methodological guard is the **problem-level
      held-out split**. If we trained and evaluated on candidates from the same
      problem, the reranker could memorise that problem's answer and the win
      would be leakage, not generalisation. So we split BY PROBLEM ID into K
      folds: a problem's candidates are entirely in train OR entirely in the
      held-out test fold, never both. Feature standardisation (mean/std) is fit
      on the TRAIN fold only and applied to the held-out fold — the same
      leakage discipline.

    * **Matched compute.** Every selection condition consumes the SAME `k`
      cached generations. The trained reranker adds only feature extraction plus
      a logistic dot product per candidate — no extra samples. We report the
      reranker parameter count so "energy wins by spending more compute" is
      ruled out by accounting, not by hope.

Spec: REQ-KONA-3460, SCENARIO-KONA-3460, SCENARIO-KONA-3460-BLOCKED
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# The deterministic verifier-energy heuristics + the proven paired-significance
# machinery are shared with the UNTRAINED predecessor (exp3449). We re-use them
# verbatim so the trained-vs-untrained comparison differs ONLY in the reranker.
from carnot.phase3.p01_energy_vote_scoring import (
    candidate_energy,
    extract_steps,
    majority_vote,
    mcnemar_exact,
    paired_bootstrap_ci,
    self_certainty_bon,
)
from carnot.verify.ebm_cot import EbmCotCalibrator
from carnot.verify.semantic_energy import IsingVerifier
from carnot.verify.tier0r_curry_howard import Tier0rVerifier
from carnot.verify.tier0u_logical_consistency import Tier0uVerifier

# Feature vector layout (kept explicit so the artifact + tests can name each
# dimension). All features are "higher = more suspicious" EXCEPT mean_logprob
# (higher = model more confident); the logistic regression learns the signs.
FEATURE_NAMES: tuple[str, ...] = (
    "arithmetic_energy",  # IsingVerifier: fraction of wrong `A op B = C` claims
    "contradiction_energy",  # EbmCotCalibrator: adjacent-step polarity flips
    "curry_howard_score",  # Tier0rVerifier: type-violation penalty in [0, 1]
    "logical_inconsistency",  # Tier0uVerifier: self-inconsistency in [0, 1]
    "mean_logprob",  # model's own per-token confidence (higher = surer)
    "log_n_steps",  # log(1 + number of reasoning steps): trace length proxy
)
N_FEATURES: int = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Per-candidate features + FoVer energy
# ---------------------------------------------------------------------------
@dataclass
class _Verifiers:
    """Bundle of the four reusable deterministic verifiers (built once)."""

    ising: IsingVerifier = field(default_factory=IsingVerifier)
    ebmcot: EbmCotCalibrator = field(default_factory=EbmCotCalibrator)
    tier0r: Tier0rVerifier = field(default_factory=Tier0rVerifier)
    tier0u: Tier0uVerifier = field(default_factory=Tier0uVerifier)


def candidate_feature_vector(
    text: str,
    mean_logprob: float | None,
    verifiers: _Verifiers,
) -> list[float]:
    """Extract the fixed-length feature vector for one cached candidate.

    Each feature is a cheap, deterministic signal about whether the reasoning
    trace is internally consistent. The trained reranker learns how to combine
    them into a calibrated P(correct); none of them peeks at the gold answer, so
    there is no label leakage at the feature level.

    Parameters
    ----------
    text : str
        The candidate generation (chain of thought + final answer).
    mean_logprob : float | None
        Mean per-token logprob from the generator; None -> a finite fallback so
        the feature is always defined (we use a large-magnitude negative value,
        i.e. "very unconfident", because a missing logprob is not evidence of
        confidence).
    verifiers : _Verifiers
        Pre-built verifier bundle reused across all candidates.

    Returns
    -------
    list[float]
        Feature vector in the order of ``FEATURE_NAMES``.
    """
    steps = extract_steps(text)
    arithmetic = verifiers.ising.energy(text)
    contradiction = verifiers.ebmcot.energy(steps)
    curry = verifiers.tier0r.score(text)
    logical = verifiers.tier0u.score(text)
    conf = mean_logprob if mean_logprob is not None else -10.0
    log_steps = math.log1p(len(steps))
    return [arithmetic, contradiction, curry, logical, conf, log_steps]


def fover_candidate_energy(text: str, verifiers: _Verifiers) -> float:
    """Aggregate the four FoVer step-error verifiers into one candidate energy.

    Lower energy = the FoVer ensemble considers the trace more internally
    correct. This routes the step-error verifier ensemble (the one that reaches
    0.9131 AUROC on step-error detection) into FINAL-ANSWER selection: we pick
    the candidate whose chain the ensemble likes most. We sum the four
    "higher = worse" signals (arithmetic, contradiction, Curry-Howard, logical);
    we do NOT include logprob here because that is the self-certainty baseline's
    signal, kept separate so FoVer-energy is a pure verifier comparator.
    """
    steps = extract_steps(text)
    return (
        verifiers.ising.energy(text)
        + verifiers.ebmcot.energy(steps)
        + verifiers.tier0r.score(text)
        + verifiers.tier0u.score(text)
    )


# ---------------------------------------------------------------------------
# The small trained energy reranker (logistic regression)
# ---------------------------------------------------------------------------
class TrainedEnergyReranker:
    """A lightweight logistic-regression energy reranker (EORM-style).

    This is deliberately tiny: ``N_FEATURES`` weights + 1 bias. It maps a
    candidate feature vector to P(correct) via the logistic function. Training
    minimises L2-regularised cross-entropy with full-batch gradient descent —
    fully deterministic (zero-initialised weights, fixed iteration count), so the
    run reproduces exactly from the corpus + config.

    The reranker is the ONLY thing that differs from the UNTRAINED exp3449
    predecessor. Keeping it this small is the whole point: it cannot win by
    being a big model, so a held-out win (if any) is attributable to *learning
    the right combination of cheap signals from outcome labels*, which is exactly
    the EORM (arXiv:2505.14999) claim.
    """

    def __init__(self, n_iter: int = 500, lr: float = 0.5, l2: float = 1e-3) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.l2 = l2
        self.weights = np.zeros(N_FEATURES, dtype=np.float64)
        self.bias = 0.0
        # Train-fold standardisation statistics (the leakage guard).
        self._mu = np.zeros(N_FEATURES, dtype=np.float64)
        self._sigma = np.ones(N_FEATURES, dtype=np.float64)
        self._fitted = False

    @property
    def n_params(self) -> int:
        """Total trainable parameters: one weight per feature plus the bias."""
        return N_FEATURES + 1

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        """Apply the stored (train-fold) mean/std to a feature matrix."""
        return (X - self._mu) / self._sigma

    def fit(self, X: list[list[float]], y: list[int]) -> TrainedEnergyReranker:
        """Train on TRAIN-fold candidate features and outcome-correctness labels.

        Standardisation statistics are computed HERE, on the training features
        only, so no information from the held-out fold leaks into the model.
        """
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if Xa.ndim != 2 or Xa.shape[1] != N_FEATURES:
            raise ValueError(f"expected (_, {N_FEATURES}) feature matrix, got {Xa.shape}")
        self._mu = Xa.mean(axis=0)
        sigma = Xa.std(axis=0)
        # Guard against zero-variance columns (a constant feature has std 0).
        sigma[sigma < 1e-8] = 1.0
        self._sigma = sigma
        Z = self._standardise(Xa)
        n = Z.shape[0]
        for _ in range(self.n_iter):
            logits = Z @ self.weights + self.bias
            p = 1.0 / (1.0 + np.exp(-logits))
            err = p - ya
            grad_w = Z.T @ err / n + self.l2 * self.weights
            grad_b = float(err.mean())
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        self._fitted = True
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Return P(correct) for each candidate feature vector (held-out fold)."""
        if not self._fitted:
            raise RuntimeError("reranker not fitted")
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.size == 0:
            return []
        Z = self._standardise(Xa)
        logits = Z @ self.weights + self.bias
        return [float(1.0 / (1.0 + math.exp(-z))) for z in logits]


# ---------------------------------------------------------------------------
# Problem-level K-fold split (the leakage guard)
# ---------------------------------------------------------------------------
def problem_kfold_indices(
    n_problems: int, n_folds: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Deterministic K-fold split BY PROBLEM index — never split a problem's samples.

    We shuffle the problem indices with a seeded RNG, then deal them round-robin
    into ``n_folds`` buckets. Each fold's held-out set is one bucket; its train
    set is every other problem. Because we partition problems (not samples), a
    problem's `k` candidates are always entirely in train or entirely in
    held-out — the leakage guard that makes any trained-energy win defensible.

    Returns a list of (train_problem_indices, test_problem_indices) tuples; the
    test sets are disjoint and cover all problems exactly once.
    """
    if n_problems <= 0:
        return []
    n_folds = max(2, min(n_folds, n_problems))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(n_problems))
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for i, p in enumerate(order):
        buckets[i % n_folds].append(int(p))
    splits: list[tuple[list[int], list[int]]] = []
    for f in range(n_folds):
        test = sorted(buckets[f])
        train = sorted(idx for b in range(n_folds) if b != f for idx in buckets[b])
        splits.append((train, test))
    return splits


# ---------------------------------------------------------------------------
# Trained-energy selection conditions
# ---------------------------------------------------------------------------
def _valid_pairs(answers: list, scores: list[float]) -> tuple[list, list[float]]:
    """Drop samples whose extracted answer is None, keeping answer/score aligned."""
    out_a: list = []
    out_s: list[float] = []
    for a, s in zip(answers, scores):
        if a is not None:
            out_a.append(a)
            out_s.append(s)
    return out_a, out_s


def trained_energy_weighted_vote(answers: list, proba_correct: list[float]) -> object:
    """Vote over the k samples weighted by the trained reranker's P(correct).

    Each sample votes with weight = its trained P(correct); we sum weights per
    distinct answer and return the heaviest. This is THE headline condition: a
    well-calibrated reranker up-weights samples that are actually correct, so the
    vote should be reshaped toward the right answer relative to plain majority.
    Ties break by first-appearance order for determinism.
    """
    va, vs = _valid_pairs(answers, proba_correct)
    if not va:
        return None
    bucket: dict = {}
    for a, w in zip(va, vs):
        bucket[a] = bucket.get(a, 0.0) + w
    best = max(bucket.values())
    tied = [a for a in bucket if bucket[a] == best]
    if len(tied) == 1:
        return tied[0]
    for a in va:
        if a in tied:
            return a
    return tied[0]


def trained_energy_sc_hybrid(
    answers: list, proba_correct: list[float]
) -> object:
    """Trained-energy x SC hybrid (arXiv:2510.14913).

    Combine the self-consistency signal (how many samples gave the answer) with
    the trained-energy signal (summed P(correct) for the answer), each
    normalised, and return the argmax. This degenerates to majority vote when the
    reranker is uninformative and to the trained vote when votes are uniform, so
    it can only help. Ties break by first-appearance order.
    """
    va, vs = _valid_pairs(answers, proba_correct)
    if not va:
        return None
    counts: dict = {}
    mass: dict = {}
    for a, w in zip(va, vs):
        counts[a] = counts.get(a, 0) + 1
        mass[a] = mass.get(a, 0.0) + w
    total_count = sum(counts.values())
    total_mass = sum(mass.values()) or 1.0
    score: dict = {}
    for a in counts:
        score[a] = counts[a] / total_count + mass[a] / total_mass
    best = max(score.values())
    tied = [a for a in score if score[a] == best]
    if len(tied) == 1:
        return tied[0]
    for a in va:
        if a in tied:
            return a
    return tied[0]


def fover_energy_argmin(answers: list, fover_energies: list[float]) -> object:
    """Return the answer of the single lowest FoVer-energy candidate."""
    best_a: object = None
    best_e = math.inf
    for a, e in zip(answers, fover_energies):
        if a is None:
            continue
        if e < best_e:
            best_e = e
            best_a = a
    return best_a


# ---------------------------------------------------------------------------
# Cross-validated corpus scoring
# ---------------------------------------------------------------------------
@dataclass
class TrainedScoringResult:
    """All six held-out condition accuracies plus deltas, significance, metadata.

    Fields map directly onto the REQ-KONA-3460 artifact schema; the experiment
    script copies them into the JSON deliverable.
    """

    n_problems_heldout: int
    k_samples: int
    reranker_param_count: int
    train_test_split_note: str
    self_consistency_non_degenerate: bool
    degenerate_examples: list
    ar_greedy_accuracy: float
    self_consistency_accuracy: float
    self_certainty_bon_accuracy: float
    fover_energy_argmin_accuracy: float
    trained_energy_weighted_vote_accuracy: float
    trained_energy_sc_hybrid_accuracy: float
    delta_trained_energy_vs_self_consistency: float
    delta_fover_energy_vs_self_consistency: float
    delta_hybrid_vs_self_consistency: float
    paired_significance: dict


def _accuracy(preds: list, golds: list) -> float:
    """Fraction of predictions exactly equal to the gold answer."""
    if not golds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if p is not None and p == g) / len(golds)


def score_corpus_trained_cv(
    records: list[dict],
    *,
    seed: int,
    n_folds: int = 5,
    n_boot: int = 10000,
    reranker_iter: int = 500,
    verifiers: _Verifiers | None = None,
) -> TrainedScoringResult:
    """Train per-fold rerankers and score six conditions on held-out problems.

    For each fold we (1) build candidate feature vectors + outcome labels for the
    train problems, (2) fit a fresh ``TrainedEnergyReranker`` with train-fold-only
    standardisation, (3) predict P(correct) on the held-out problems' candidates,
    and (4) record each held-out problem's prediction under every condition.
    Because the folds' test sets partition the problems, every problem is scored
    exactly once as held-out, so the reported accuracies cover the whole corpus
    with zero train/test leakage.

    The NON-DEGENERATE-SC gate is computed over the FULL corpus (self-consistency
    accuracy >= greedy AND > 0.30); a degenerate gate means per-sample answer
    extraction is broken (the exp3426 0.0-tie) and no energy comparison should be
    trusted.

    Parameters
    ----------
    records : list[dict]
        Cached corpus rows (problem_id, gold, greedy, samples, ...).
    seed : int
        Reproducibility seed for the fold split and the bootstrap.
    n_folds : int
        Number of cross-validation folds (problem-level).
    n_boot : int
        Bootstrap resamples for the paired CIs.
    reranker_iter : int
        Gradient-descent iterations for each fold's reranker.
    verifiers : _Verifiers | None
        Pre-built verifier bundle reused across all candidates.

    Returns
    -------
    TrainedScoringResult
    """
    verifiers = verifiers or _Verifiers()
    n = len(records)

    # Pre-compute per-candidate features, FoVer energies, untrained energy, and
    # labels ONCE (deterministic, corpus-only) so folds share the same features.
    feats: list[list[list[float]]] = []
    labels: list[list[int]] = []
    fover: list[list[float]] = []
    answers_all: list[list] = []
    confidences_all: list[list[float]] = []
    for rec in records:
        gold = rec["gold"]
        samples = rec.get("samples") or []
        rec_feats: list[list[float]] = []
        rec_labels: list[int] = []
        rec_fover: list[float] = []
        rec_answers: list = []
        rec_conf: list[float] = []
        for s in samples:
            text = s.get("text", "")
            mlp = s.get("mean_token_logprob")
            rec_feats.append(candidate_feature_vector(text, mlp, verifiers))
            rec_labels.append(1 if s.get("answer") == gold else 0)
            rec_fover.append(fover_candidate_energy(text, verifiers))
            rec_answers.append(s.get("answer"))
            rec_conf.append(mlp if mlp is not None else -math.inf)
        feats.append(rec_feats)
        labels.append(rec_labels)
        fover.append(rec_fover)
        answers_all.append(rec_answers)
        confidences_all.append(rec_conf)

    splits = problem_kfold_indices(n, n_folds, seed)
    effective_folds = len(splits)

    # Per-problem held-out predictions (index -> answer) for the trained conditions.
    trained_vote_pred: list = [None] * n
    hybrid_pred: list = [None] * n
    param_count = TrainedEnergyReranker().n_params

    for train_idx, test_idx in splits:
        X_train: list[list[float]] = []
        y_train: list[int] = []
        for pi in train_idx:
            X_train.extend(feats[pi])
            y_train.extend(labels[pi])
        reranker = TrainedEnergyReranker(n_iter=reranker_iter)
        # A degenerate train fold (all-correct or all-incorrect) still trains;
        # logistic regression simply learns a near-constant — handled gracefully.
        reranker.fit(X_train, y_train)
        for pi in test_idx:
            proba = reranker.predict_proba(feats[pi]) if feats[pi] else []
            trained_vote_pred[pi] = trained_energy_weighted_vote(answers_all[pi], proba)
            hybrid_pred[pi] = trained_energy_sc_hybrid(answers_all[pi], proba)

    # Non-trained conditions are corpus-only (no fold dependence): compute per
    # problem directly so every problem is covered exactly once (held-out).
    golds: list = [rec["gold"] for rec in records]
    greedy_pred: list = [(rec.get("greedy") or {}).get("answer") for rec in records]
    sc_pred: list = [
        majority_vote(answers_all[i], confidences_all[i]) for i in range(n)
    ]
    certainty_pred: list = [
        self_certainty_bon(answers_all[i], confidences_all[i]) for i in range(n)
    ]
    fover_pred: list = [fover_energy_argmin(answers_all[i], fover[i]) for i in range(n)]

    ar_acc = _accuracy(greedy_pred, golds)
    sc_acc = _accuracy(sc_pred, golds)
    certainty_acc = _accuracy(certainty_pred, golds)
    fover_acc = _accuracy(fover_pred, golds)
    trained_vote_acc = _accuracy(trained_vote_pred, golds)
    hybrid_acc = _accuracy(hybrid_pred, golds)

    non_degenerate = (sc_acc >= ar_acc) and (sc_acc > 0.30)
    degenerate_examples = [
        {
            "problem_id": records[i].get("problem_id"),
            "gold": golds[i],
            "greedy_answer": greedy_pred[i],
            "sample_answers": answers_all[i],
        }
        for i in range(min(3, n))
    ]

    sc_correct = [p is not None and p == g for p, g in zip(sc_pred, golds)]
    tv_correct = [p is not None and p == g for p, g in zip(trained_vote_pred, golds)]
    fv_correct = [p is not None and p == g for p, g in zip(fover_pred, golds)]
    hy_correct = [p is not None and p == g for p, g in zip(hybrid_pred, golds)]

    def _sig(method_correct: list[bool], label: str) -> dict:
        return {
            "comparison": f"{label}_vs_self_consistency",
            "mcnemar_exact_p": mcnemar_exact(sc_correct, method_correct),
            "bootstrap_ci95": list(
                paired_bootstrap_ci(method_correct, sc_correct, seed=seed, n_boot=n_boot)
            ),
        }

    paired_significance = {
        "trained_energy": _sig(tv_correct, "trained_energy_weighted_vote"),
        "fover_energy": _sig(fv_correct, "fover_energy_argmin"),
        "hybrid": _sig(hy_correct, "trained_energy_sc_hybrid"),
    }

    split_note = (
        f"problem-level {effective_folds}-fold CV (seed={seed}); each problem's "
        f"{max((len(r.get('samples') or []) for r in records), default=0)} samples are "
        "entirely in train OR held-out, never split; feature standardisation fit on "
        "train fold only. All accuracies are on held-out problems."
    )

    return TrainedScoringResult(
        n_problems_heldout=n,
        k_samples=max((len(r.get("samples") or []) for r in records), default=0),
        reranker_param_count=param_count,
        train_test_split_note=split_note,
        self_consistency_non_degenerate=non_degenerate,
        degenerate_examples=degenerate_examples,
        ar_greedy_accuracy=ar_acc,
        self_consistency_accuracy=sc_acc,
        self_certainty_bon_accuracy=certainty_acc,
        fover_energy_argmin_accuracy=fover_acc,
        trained_energy_weighted_vote_accuracy=trained_vote_acc,
        trained_energy_sc_hybrid_accuracy=hybrid_acc,
        delta_trained_energy_vs_self_consistency=trained_vote_acc - sc_acc,
        delta_fover_energy_vs_self_consistency=fover_acc - sc_acc,
        delta_hybrid_vs_self_consistency=hybrid_acc - sc_acc,
        paired_significance=paired_significance,
    )


def derive_v5_verdict(result: TrainedScoringResult) -> str:
    """Map the trained-reranker result to exactly one `complete:` terminal verdict.

    Gate ladder (per REQ-KONA-3460 acceptance gates):

      * G0 NON-DEGENERATE-SC: if self-consistency is degenerate (the exp3426
        0.0-tie broken harness), no energy comparison is trustworthy.
      * G1 TRAINED-ENERGY-NON-INFERIOR: max(trained-vote, hybrid, FoVer-argmin)
        >= self-consistency. The floor the UNTRAINED energy failed.
      * G2 TRAINED-ENERGY-ADDS-VALUE: the trained-vote OR hybrid delta is
        positive AND its paired McNemar p < 0.05 — a trained energy SIGNIFICANTLY
        beats majority vote at matched compute.
    """
    if not result.self_consistency_non_degenerate:
        return (
            "complete: blocked_self_consistency_harness_degenerate_"
            "per_sample_extraction_broken"
        )

    best = max(
        result.trained_energy_weighted_vote_accuracy,
        result.trained_energy_sc_hybrid_accuracy,
        result.fover_energy_argmin_accuracy,
    )
    g1 = best >= result.self_consistency_accuracy

    trained_sig = result.paired_significance["trained_energy"]
    hybrid_sig = result.paired_significance["hybrid"]
    g2 = (
        result.delta_trained_energy_vs_self_consistency > 0
        and trained_sig["mcnemar_exact_p"] < 0.05
    ) or (
        result.delta_hybrid_vs_self_consistency > 0
        and hybrid_sig["mcnemar_exact_p"] < 0.05
    )

    if g2:
        return (
            "complete: trained_energy_beats_self_consistency_"
            "phase3_premise_validated"
        )
    if g1:
        return (
            "complete: trained_energy_matches_but_does_not_beat_"
            "self_consistency_at_equal_compute"
        )
    return (
        "complete: even_trained_energy_below_self_consistency_"
        "selection_premise_refuted_on_this_substrate"
    )
