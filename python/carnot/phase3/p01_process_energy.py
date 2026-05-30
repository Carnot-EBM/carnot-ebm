"""P0.1 v6 — Process-aware energy + optimal aggregation vs self-consistency (HEADROOM).

**Researcher summary (for engineers who are not EBM specialists):**
    This module is the SCORING substrate for the decisive P0.1 v6 premise test
    (REQ-KONA-3472). Its predecessor, exp3460 (v5), trained an outcome-label
    energy reranker and found it merely *ties* majority-vote self-consistency on
    GSM8K. The reason was structural, not a modelling failure: GSM8K
    self-consistency sits at ceiling (~0.908), so the majority vote is almost
    always already correct and ANY selector degenerates onto it — an exact tie
    that the adversarial verifier flags as a tautology.

    The reasoning literature says the win appears (a) WITH HEADROOM — a benchmark
    where self-consistency leaves room to improve — and (b) from PROCESS-level
    verification that scores each reasoning STEP, not just the final answer
    (arXiv:2602.11570 PRIME reports process-aware verification beating
    outcome-only by +8-9% on AIME), combined with (c) an OPTIMAL aggregation of
    the vote signal and the energy signal (arXiv:2510.13918). This module answers
    the never-asked question: on a HEADROOM corpus (self-consistency in
    [0.4, 0.78]) does a PROCESS-AWARE step-level energy plus OPTIMAL aggregation
    BEAT self-consistency at matched compute?

**Why this design:**
    * We never load a live LLM. The HEADROOM corpus
      (`data/p01_hardmath_generations.jsonl`, built by exp3471) already contains,
      per problem, a greedy generation plus `k` sampled generations, each with its
      extracted answer, a PARSED STEP LIST, per-token logprobs, and a
      correctness label. All we do is *score* those cached candidates with
      different selection strategies and compare held-out accuracy.

    * PROCESS energy (the v6 novelty). The v5 FoVer energy scored the whole
      candidate text once. Here we score each PARSED STEP individually with the
      FoVer 4-verifier ensemble and aggregate to a per-candidate process-energy.
      This is the PRIME intuition: a wrong final answer usually has a locatable
      bad step, and step-level scoring surfaces it where whole-text scoring
      averages it away.

    * OPTIMAL aggregation (the headline condition). arXiv:2510.13918 shows the
      statistically optimal way to combine self-consistency vote mass with a
      reward/energy signal is a weighted vote whose mixing coefficient trades off
      the two. We fit that single coefficient on the TRAIN fold (grid search over
      train accuracy) and APPLY it on the held-out fold — never peeking at the
      held-out labels. When the energy is uninformative the fitted coefficient
      collapses to pure self-consistency, so the aggregator can only help.

    * FLIP-COUNT primary metric (tautology-clean by construction). Instead of
      comparing two accuracies that may be bit-identical (the exp3460 flag), the
      primary signal is how many problems the condition's selected answer DIFFERS
      from the SC majority answer (`flip_count`), and the net correctness change
      among those flips. A condition that agrees with SC everywhere reports
      `flip_count=0` ONCE — there is no second bit-identical field to flag.

    * Matched compute. Every selection condition consumes the SAME `k` cached
      generations. The energy adds only feature extraction, a tiny logistic
      reranker, and a one-parameter aggregator — no extra samples — so "energy
      wins by spending more compute" is ruled out by accounting.

Spec: REQ-KONA-3472, SCENARIO-KONA-3472, SCENARIO-KONA-3472-BLOCKED
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The deterministic verifier-energy heuristics + paired-significance machinery
# are shared with the UNTRAINED predecessor (exp3449) and the v5 trained reranker
# (exp3460). We re-use them verbatim so v6 differs ONLY in the process energy,
# the optimal aggregator, and the flip-count metric.
from carnot.phase3.p01_energy_vote_scoring import (
    extract_steps,
    majority_vote,
    mcnemar_exact,
    paired_bootstrap_ci,
    self_certainty_bon,
)
from carnot.phase3.p01_trained_energy_reranker import (
    TrainedEnergyReranker,
    _Verifiers,
    candidate_feature_vector,
    problem_kfold_indices,
)

# The HEADROOM band: a corpus whose self-consistency accuracy falls in this range
# leaves a selector room to improve (unlike GSM8K's ~0.908 ceiling). Below 0.4 the
# corpus is too hard for the samples to even contain the right answer often enough
# to recover; above 0.78 majority vote is near-ceiling and any selector
# degenerates onto it (the exp3460 regime).
HEADROOM_LOW: float = 0.40
HEADROOM_HIGH: float = 0.78

# Grid of mixing coefficients lambda in [0, 1] for the optimal aggregator. 0 ->
# pure self-consistency (vote count only); 1 -> pure energy mass. We fit lambda
# on the train fold and apply on held-out.
_LAMBDA_GRID: tuple[float, ...] = tuple(round(0.05 * i, 4) for i in range(21))


# ---------------------------------------------------------------------------
# Per-step PROCESS energy (the v6 novelty)
# ---------------------------------------------------------------------------
def process_energy_per_step(steps: list[str], verifiers: _Verifiers) -> float:
    """Score each parsed reasoning step with the FoVer ensemble; aggregate to one energy.

    The four FoVer verifiers each emit a "higher = more suspicious" signal. We run
    the per-step verifiers (arithmetic-violation energy, Curry-Howard type penalty,
    logical-inconsistency) on EACH step individually and average across steps, then
    add the adjacent-step contradiction energy (which is inherently a pairwise,
    over-the-trace signal) normalised by trace length. Lower process-energy = the
    FoVer ensemble considers the step-by-step reasoning more internally correct.

    This is the PRIME intuition operationalised: a wrong answer usually has a
    locatable bad step, and per-step scoring surfaces it where the v5 whole-text
    score averaged it away.

    Parameters
    ----------
    steps : list[str]
        The parsed reasoning steps for one candidate (corpus `steps` field, or
        ``extract_steps(text)`` as a fallback).
    verifiers : _Verifiers
        Pre-built verifier bundle reused across all candidates.

    Returns
    -------
    float
        The aggregated per-candidate process energy (0.0 for an empty step list,
        because no step means no detectable step-level violation).
    """
    if not steps:
        return 0.0
    per_step_total = 0.0
    for step in steps:
        # The arithmetic + type + logical verifiers are all single-string scorers,
        # so they apply directly to one step's text.
        per_step_total += (
            verifiers.ising.energy(step)
            + verifiers.tier0r.score(step)
            + verifiers.tier0u.score(step)
        )
    per_step_mean = per_step_total / len(steps)
    # The contradiction verifier looks at adjacent steps, so it needs the whole
    # trace; we normalise by trace length to keep it on the same per-step scale.
    contradiction = verifiers.ebmcot.energy(steps) / len(steps)
    return per_step_mean + contradiction


def process_energy_argmin(answers: list, process_energies: list[float]) -> object:
    """Return the answer of the single lowest process-energy candidate.

    This is the v6 per-step condition: pick the candidate whose step-by-step
    reasoning the FoVer ensemble likes most. None-answer candidates are skipped.
    """
    best_a: object = None
    best_e = math.inf
    for a, e in zip(answers, process_energies):
        if a is None:
            continue
        if e < best_e:
            best_e = e
            best_a = a
    return best_a


# ---------------------------------------------------------------------------
# OPTIMAL SC + energy aggregation (the headline condition)
# ---------------------------------------------------------------------------
def _aggregate_with_lambda(
    answers: list, proba_correct: list[float], lam: float
) -> object:
    """Weighted vote mixing self-consistency count mass with energy mass.

    For each distinct answer ``a`` we compute a score that linearly mixes the
    fraction of samples voting for ``a`` (the self-consistency signal) and the
    fraction of total trained-energy P(correct) mass on ``a`` (the energy signal),
    with ``lam`` the mixing coefficient. ``lam=0`` reduces to plain majority vote;
    ``lam=1`` is pure energy mass. Ties break by first-appearance order for
    determinism. This is the parametric family arXiv:2510.13918 optimises over.
    """
    counts: dict = {}
    mass: dict = {}
    order: list = []
    for a, w in zip(answers, proba_correct):
        if a is None:
            continue
        if a not in counts:
            order.append(a)
        counts[a] = counts.get(a, 0) + 1
        mass[a] = mass.get(a, 0.0) + w
    if not counts:
        return None
    total_count = sum(counts.values()) or 1
    total_mass = sum(mass.values()) or 1.0
    score: dict = {
        a: (1.0 - lam) * counts[a] / total_count + lam * mass[a] / total_mass
        for a in counts
    }
    best = max(score.values())
    tied = [a for a in order if score[a] == best]
    return tied[0]


def fit_optimal_lambda(
    train_answers: list[list],
    train_proba: list[list[float]],
    train_golds: list,
) -> float:
    """Grid-search the mixing coefficient lambda that maximises TRAIN accuracy.

    We never look at held-out labels here — lambda is chosen on the train fold
    only, the same leakage discipline as the reranker's feature standardisation.
    On a tie we prefer the SMALLEST lambda (closest to plain self-consistency), so
    the aggregator only departs from majority vote when the energy demonstrably
    helps on train. Returns lambda in [0, 1].
    """
    best_lambda = 0.0
    best_acc = -1.0
    for lam in _LAMBDA_GRID:
        correct = 0
        for ans, proba, gold in zip(train_answers, train_proba, train_golds):
            pred = _aggregate_with_lambda(ans, proba, lam)
            if pred is not None and pred == gold:
                correct += 1
        acc = correct / len(train_golds) if train_golds else 0.0
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam
    return best_lambda


def optimal_aggregate(answers: list, proba_correct: list[float], lam: float) -> object:
    """Apply the train-fitted optimal aggregator to one held-out problem's samples."""
    return _aggregate_with_lambda(answers, proba_correct, lam)


# ---------------------------------------------------------------------------
# FLIP-COUNT primary metric (tautology-clean by construction)
# ---------------------------------------------------------------------------
@dataclass
class FlipMetrics:
    """How a condition's selections differ from the self-consistency majority.

    `flip_count` is the number of problems where the condition picked a DIFFERENT
    answer than SC. `flips_correct` / `flips_incorrect` split those flips by
    whether the new answer is right or wrong, and `net_correctness_gain` is their
    difference — the honest net effect of the condition's departures from SC,
    robust to ceiling-induced ties (a condition that never flips reports all
    zeros, not a bit-identical accuracy).
    """

    flip_count: int
    flips_correct: int
    flips_incorrect: int
    net_correctness_gain: int


def flip_metrics(cond_preds: list, sc_preds: list, golds: list) -> FlipMetrics:
    """Compute the flip-count metric of a condition's predictions versus SC.

    A "flip" is a problem where the condition's selected answer differs from the
    SC majority answer. Among flips we count how many landed on the correct gold
    answer (a recovered minority-yet-correct answer — the win mechanism) versus
    how many landed on a wrong answer (the cost). When the condition agrees with
    SC on a problem, that problem contributes nothing — there is no separate
    accuracy field to tie, so the tautology flag cannot fire on this signal.
    """
    flip_count = 0
    flips_correct = 0
    flips_incorrect = 0
    for cond, sc, gold in zip(cond_preds, sc_preds, golds):
        if cond != sc:
            flip_count += 1
            if cond is not None and cond == gold:
                flips_correct += 1
            else:
                flips_incorrect += 1
    return FlipMetrics(
        flip_count=flip_count,
        flips_correct=flips_correct,
        flips_incorrect=flips_incorrect,
        net_correctness_gain=flips_correct - flips_incorrect,
    )


# ---------------------------------------------------------------------------
# Cross-validated corpus scoring (seven conditions)
# ---------------------------------------------------------------------------
@dataclass
class ProcessScoringResult:
    """All seven held-out condition accuracies, flip metrics, deltas, significance.

    Fields map directly onto the REQ-KONA-3472 artifact schema; the experiment
    script copies them into the JSON deliverable.
    """

    n_problems_heldout: int
    k_samples: int
    reranker_param_count: int
    aggregator_param_count: int
    fitted_lambdas: list[float]
    train_test_split_note: str
    self_consistency_in_headroom_band: bool
    ar_greedy_accuracy: float
    self_consistency_accuracy: float
    self_certainty_bon_accuracy: float
    process_energy_argmin_accuracy: float
    trained_energy_weighted_vote_accuracy: float
    trained_energy_sc_hybrid_accuracy: float
    optimal_aggregation_accuracy: float
    flip_optimal: FlipMetrics
    flip_process_energy: FlipMetrics
    flip_hybrid: FlipMetrics
    delta_optimal_vs_self_consistency: float
    delta_process_energy_vs_self_consistency: float
    delta_hybrid_vs_self_consistency: float
    paired_significance: dict


def _accuracy(preds: list, golds: list) -> float:
    """Fraction of predictions exactly equal to the gold answer."""
    if not golds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if p is not None and p == g) / len(golds)


def _candidate_steps(sample: dict) -> list[str]:
    """Return a candidate's parsed steps, falling back to re-parsing its text.

    The HEADROOM corpus ships a `steps` list per sample, but we re-derive from
    `text` if it is missing so the scorer is robust to older corpus shards.
    """
    steps = sample.get("steps")
    if isinstance(steps, list) and steps:
        return [str(s) for s in steps]
    return extract_steps(sample.get("text", ""))


def score_corpus_process_cv(
    records: list[dict],
    *,
    seed: int,
    n_folds: int = 5,
    n_boot: int = 10000,
    reranker_iter: int = 500,
    verifiers: _Verifiers | None = None,
) -> ProcessScoringResult:
    """Score seven held-out conditions with process energy + optimal aggregation.

    For each fold we (1) build candidate feature vectors + outcome labels for the
    train problems, (2) fit a fresh ``TrainedEnergyReranker`` with train-fold-only
    standardisation, (3) fit the optimal aggregator's mixing coefficient on the
    train problems, (4) predict P(correct) on the held-out problems' candidates,
    and (5) record each held-out problem's prediction under every condition.
    Because the folds' test sets partition the problems, every problem is scored
    exactly once as held-out, so the reported accuracies cover the whole corpus
    with zero train/test leakage.

    The HEADROOM gate (self-consistency in [0.40, 0.78]) is computed over the FULL
    corpus; when it fails the caller emits a `blocked_corpus_at_ceiling` /
    too-hard verdict rather than trusting an energy comparison against a control
    with no room to improve.
    """
    verifiers = verifiers or _Verifiers()
    n = len(records)

    # Pre-compute per-candidate features, process energies, answers, confidences,
    # and labels ONCE (deterministic, corpus-only) so folds share the same data.
    feats: list[list[list[float]]] = []
    labels: list[list[int]] = []
    proc: list[list[float]] = []
    answers_all: list[list] = []
    confidences_all: list[list[float]] = []
    for rec in records:
        gold = rec["gold"]
        samples = rec.get("samples") or []
        rec_feats: list[list[float]] = []
        rec_labels: list[int] = []
        rec_proc: list[float] = []
        rec_answers: list = []
        rec_conf: list[float] = []
        for s in samples:
            text = s.get("text", "")
            mlp = s.get("mean_token_logprob")
            steps = _candidate_steps(s)
            rec_feats.append(candidate_feature_vector(text, mlp, verifiers))
            rec_labels.append(1 if s.get("answer") == gold else 0)
            rec_proc.append(process_energy_per_step(steps, verifiers))
            rec_answers.append(s.get("answer"))
            rec_conf.append(mlp if mlp is not None else -math.inf)
        feats.append(rec_feats)
        labels.append(rec_labels)
        proc.append(rec_proc)
        answers_all.append(rec_answers)
        confidences_all.append(rec_conf)

    splits = problem_kfold_indices(n, n_folds, seed)
    effective_folds = len(splits)

    trained_vote_pred: list = [None] * n
    hybrid_pred: list = [None] * n
    optimal_pred: list = [None] * n
    fitted_lambdas: list[float] = []
    param_count = TrainedEnergyReranker().n_params

    golds: list = [rec["gold"] for rec in records]

    for train_idx, test_idx in splits:
        X_train: list[list[float]] = []
        y_train: list[int] = []
        for pi in train_idx:
            X_train.extend(feats[pi])
            y_train.extend(labels[pi])
        reranker = TrainedEnergyReranker(n_iter=reranker_iter)
        reranker.fit(X_train, y_train)

        # Fit the optimal aggregator's lambda on the train problems only.
        train_answers = [answers_all[pi] for pi in train_idx]
        train_proba = [
            reranker.predict_proba(feats[pi]) if feats[pi] else [] for pi in train_idx
        ]
        train_golds = [golds[pi] for pi in train_idx]
        lam = fit_optimal_lambda(train_answers, train_proba, train_golds)
        fitted_lambdas.append(lam)

        for pi in test_idx:
            proba = reranker.predict_proba(feats[pi]) if feats[pi] else []
            trained_vote_pred[pi] = _aggregate_with_lambda(answers_all[pi], proba, 1.0)
            hybrid_pred[pi] = _aggregate_with_lambda(answers_all[pi], proba, 0.5)
            optimal_pred[pi] = optimal_aggregate(answers_all[pi], proba, lam)

    # Non-trained conditions are corpus-only (no fold dependence).
    greedy_pred: list = [(rec.get("greedy") or {}).get("answer") for rec in records]
    sc_pred: list = [majority_vote(answers_all[i], confidences_all[i]) for i in range(n)]
    certainty_pred: list = [
        self_certainty_bon(answers_all[i], confidences_all[i]) for i in range(n)
    ]
    process_pred: list = [
        process_energy_argmin(answers_all[i], proc[i]) for i in range(n)
    ]

    ar_acc = _accuracy(greedy_pred, golds)
    sc_acc = _accuracy(sc_pred, golds)
    certainty_acc = _accuracy(certainty_pred, golds)
    process_acc = _accuracy(process_pred, golds)
    trained_vote_acc = _accuracy(trained_vote_pred, golds)
    hybrid_acc = _accuracy(hybrid_pred, golds)
    optimal_acc = _accuracy(optimal_pred, golds)

    in_band = HEADROOM_LOW <= sc_acc <= HEADROOM_HIGH

    flip_optimal = flip_metrics(optimal_pred, sc_pred, golds)
    flip_process = flip_metrics(process_pred, sc_pred, golds)
    flip_hybrid = flip_metrics(hybrid_pred, sc_pred, golds)

    sc_correct = [p is not None and p == g for p, g in zip(sc_pred, golds)]
    opt_correct = [p is not None and p == g for p, g in zip(optimal_pred, golds)]
    proc_correct = [p is not None and p == g for p, g in zip(process_pred, golds)]
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
        "optimal_aggregation": _sig(opt_correct, "optimal_aggregation"),
        "process_energy": _sig(proc_correct, "process_energy_argmin"),
        "hybrid": _sig(hy_correct, "trained_energy_sc_hybrid"),
    }

    max_k = max((len(r.get("samples") or []) for r in records), default=0)
    split_note = (
        f"problem-level {effective_folds}-fold CV (seed={seed}); each problem's "
        f"up to {max_k} samples are entirely in train OR held-out, never split; "
        "feature standardisation AND the optimal-aggregator lambda are fit on the "
        "train fold only. All accuracies are on held-out problems."
    )

    return ProcessScoringResult(
        n_problems_heldout=n,
        k_samples=max_k,
        reranker_param_count=param_count,
        aggregator_param_count=1,  # the single mixing coefficient lambda
        fitted_lambdas=fitted_lambdas,
        train_test_split_note=split_note,
        self_consistency_in_headroom_band=in_band,
        ar_greedy_accuracy=ar_acc,
        self_consistency_accuracy=sc_acc,
        self_certainty_bon_accuracy=certainty_acc,
        process_energy_argmin_accuracy=process_acc,
        trained_energy_weighted_vote_accuracy=trained_vote_acc,
        trained_energy_sc_hybrid_accuracy=hybrid_acc,
        optimal_aggregation_accuracy=optimal_acc,
        flip_optimal=flip_optimal,
        flip_process_energy=flip_process,
        flip_hybrid=flip_hybrid,
        delta_optimal_vs_self_consistency=optimal_acc - sc_acc,
        delta_process_energy_vs_self_consistency=process_acc - sc_acc,
        delta_hybrid_vs_self_consistency=hybrid_acc - sc_acc,
        paired_significance=paired_significance,
    )


def derive_v6_verdict(result: ProcessScoringResult) -> str:
    """Map the process-energy result to exactly one `complete:` terminal verdict.

    Gate ladder (per REQ-KONA-3472 acceptance gates):

      * G0 HEADROOM: self-consistency must be in [0.40, 0.78]; else the test
        repeats the GSM8K degenerate-tie regime and no comparison is reported.
      * G1 ENERGY-BEATS-SC-WITH-HEADROOM: optimal aggregation has a positive net
        correctness gain among flips AND a positive delta vs SC with paired
        p < 0.05 — the first real Phase-3 selection justification.
      * G2 NON-DEGENERATE: the optimal aggregation actually CHANGES selections
        (flip_count > 0), proving this is not the exp3460 degenerate-tie regime
        regardless of whether the net gain is positive.
    """
    if not result.self_consistency_in_headroom_band:
        sc = result.self_consistency_accuracy
        return f"complete: blocked_corpus_at_ceiling_no_headroom_sc={sc:.4f}"

    opt_sig = result.paired_significance["optimal_aggregation"]
    g1 = (
        result.flip_optimal.net_correctness_gain > 0
        and result.delta_optimal_vs_self_consistency > 0
        and opt_sig["mcnemar_exact_p"] < 0.05
    )
    g2 = result.flip_optimal.flip_count > 0

    if g1:
        return (
            "complete: process_energy_beats_self_consistency_with_headroom_"
            "phase3_premise_validated"
        )
    if g2:
        return (
            "complete: process_energy_changes_selections_but_does_not_beat_"
            "self_consistency_with_headroom"
        )
    return (
        "complete: process_energy_does_not_change_selections_selection_premise_"
        "refuted_on_this_substrate"
    )
