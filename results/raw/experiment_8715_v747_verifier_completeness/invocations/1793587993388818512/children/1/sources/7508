#!/usr/bin/env python3
"""Exp 3519 — P0.1 Route 2: Fix the consensus-trap collapse in the energy reranker.

WHY this experiment exists:
  exp3507 (v9) reported process_energy_argmin_accuracy == self_consistency_accuracy
  for ALL 7 conditions, flip_count=0, lambdas all=0.  The adversarial verifier
  flagged 5 TAUTOLOGY critical warnings.

  Root-cause diagnosis: the v9 reranker fell into the "consensus trap" — all three
  sources of collapse happened together:
    (a) The SC majority-vote indicator was included as a feature, making it collinear
        with the target: a 0/1 binary indicator that already predicts ~65% of
        outcomes dominates the logistic regression and drives all energy weights to
        0 via L2 regularization shrinkage.
    (b) No StandardScaler: raw energy scores (floating-point, wide range) and the
        SC indicator (0/1 binary) have incompatible scales, so L2 regularization
        penalizes the energy weights disproportionately hard.
    (c) Over-regularization: the default C=1.0 is too small given the scale mismatch;
        with all features at incompatible scales, the regularizer pushes everything
        toward the majority baseline.

  This experiment (v10) applies all three fixes:
    1. NEVER include the SC majority indicator as a feature — energy features only.
    2. Wrap LogisticRegression in a Pipeline with StandardScaler as step 1.
    3. Use C=100 (weak regularization) and sample weights 3.0 for problems where
       SC is WRONG on the training fold (minority-recovery bias).

Experiment number: 3519
Spec: REQ-KONA-3519, SCENARIO-KONA-3519

INFERENCE SUBSTRATE: verifier_ensemble_against_cached_candidates — no live model;
duration floor 1 s.

Run:
  cd /home/ianblenke/github.com/ianblenke/carnot && \
    JAX_PLATFORMS=cpu .venv/bin/python \
    scripts/experiment_3519_p01_route2_energy_reranker_fix_consensus_trap_v10.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

CORPUS_PATH = REPO_ROOT / "data" / "p01_difficulty_matched_generations.jsonl"
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3519_p01_route2_energy_reranker_fix_consensus_trap_v10.json"
)

# Collapse root cause — documented here and echoed into the artifact.
COLLAPSE_ROOT_CAUSE = (
    "The v9 reranker collapsed because the SC majority-vote indicator was included "
    "as a feature (collinear with the target, dominates L2 regression), compounded "
    "by absent StandardScaler (binary indicator vs float energy at incompatible scales "
    "causes disproportionate L2 shrinkage on energy weights) and over-regularization "
    "(C=1.0 with scale mismatch drives all lambdas to zero)."
)

MIN_PROBLEMS = 40
MIN_SAMPLES_PER_PROBLEM = 4
HEADROOM_LOW = 0.40
HEADROOM_HIGH = 0.80
N_BOOT = 1000

_START_TIME = time.time()
_START_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_level3_records(corpus_path: Path) -> list[dict]:
    """Load and filter the corpus to usable level-3 records.

    WHY level-3 only: the purpose-built corpus was constructed so that
    aggregate SC over level-3 lands in [0.40, 0.80] — the headroom band
    where an energy-based selector has room to beat majority vote.  Problems
    below MIN_SAMPLES_PER_PROBLEM are excluded because the per-problem
    correctness estimate is too noisy to train the reranker.
    """
    records: list[dict] = []
    with open(corpus_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Keep level-3 records (int or string comparison)
            if str(rec.get("level", "")).strip() != "3":
                continue
            gold = rec.get("gold_answer_norm") or rec.get("gold_answer")
            if gold is None:
                continue
            samples = rec.get("samples") or []
            if len(samples) < MIN_SAMPLES_PER_PROBLEM:
                continue
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# SC majority vote
# ---------------------------------------------------------------------------

def build_sc_majority(records: list[dict]) -> list[tuple[str | None, bool]]:
    """Return list of (majority_answer, is_correct) per problem.

    WHY: the SC majority answer is the PRIMARY CONTROL condition; comparing
    all other conditions against it gives the flip-count signal that is
    tautology-clean by construction (flip_count=0 is detectable precisely
    because SC is the reference, not a feature in the model).
    """
    results: list[tuple[str | None, bool]] = []
    for rec in records:
        gold = rec.get("gold_answer_norm") or rec.get("gold_answer")
        answers = [
            s.get("extracted_answer_norm") or s.get("extracted_answer")
            for s in (rec.get("samples") or [])
        ]
        valid = [a for a in answers if a is not None]
        if not valid:
            results.append((None, False))
            continue
        voted: str = Counter(valid).most_common(1)[0][0]
        results.append((voted, voted == gold))
    return results


# ---------------------------------------------------------------------------
# Process energy proxy (no live model)
# ---------------------------------------------------------------------------

def compute_process_energy(records: list[dict]) -> list[list[float]]:
    """Compute a scalar process-energy proxy for each generation of each problem.

    WHY lower energy = higher quality: we use -n_steps as the proxy because
    concise reasoning chains tend to be more reliable (fewer steps = tighter
    argument = lower energy).  We normalize within each problem so energies
    are on [0, 1] per problem.

    WHY NOT to include mean_token_logprob: the corpus has no logprob values
    (all None), so we fall back to the step-count proxy.  The proxy is
    intentionally simple so it cannot accidentally encode SC majority agreement.
    """
    per_problem: list[list[float]] = []
    for rec in records:
        samples = rec.get("samples") or []
        raw: list[float] = []
        for s in samples:
            n_steps = s.get("n_steps") or len(s.get("reasoning_steps") or []) or 1
            # Energy = step count (more steps → higher energy, i.e. worse)
            raw.append(float(n_steps))
        if not raw:
            per_problem.append([])
            continue
        mn, mx = min(raw), max(raw)
        span = mx - mn
        if span < 1e-9:
            # All generations have equal step count → uniform energy; add
            # tiny positional jitter so the argmin selects deterministically
            normalized = [float(i) / max(len(raw), 1) for i in range(len(raw))]
        else:
            normalized = [(v - mn) / span for v in raw]
        per_problem.append(normalized)
    return per_problem


# ---------------------------------------------------------------------------
# Feature extraction (THE FIX: no SC indicator)
# ---------------------------------------------------------------------------

def _extract_features(
    records: list[dict],
    energies: list[list[float]],
    sc_majority: list[tuple[str | None, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, labels, and problem indices for the reranker.

    Returns:
        X: shape (total_generations, n_features) — energy features ONLY.
        y: shape (total_generations,) — 1 if this generation's answer is correct.
        problem_idx: shape (total_generations,) — which problem each row belongs to.

    THE FIX (applied here):
        Feature vector = [process_energy, n_steps_normalized, answer_len_normalized]
        We NEVER include the SC majority indicator as a feature.  Including it
        creates perfect collinearity with the regression target (SC indicator
        already predicts ~65% of outcomes on this corpus), which causes L2
        regularization to shrink all energy weights to zero.

    WHY three features:
        - process_energy: the main hypothesis (step-level energy correlates
          with correctness in a way SC does not capture)
        - n_steps_normalized: redundant with energy but provides a signal
          on a different scale (ensures StandardScaler has something to do)
        - answer_len_normalized: weak proxy for over-generation (very long
          answers tend to hedge or hallucinate)
    """
    rows_X: list[list[float]] = []
    rows_y: list[int] = []
    rows_prob: list[int] = []

    for i, (rec, energies_i) in enumerate(zip(records, energies)):
        samples = rec.get("samples") or []
        gold = rec.get("gold_answer_norm") or rec.get("gold_answer")
        for j, (s, e_j) in enumerate(zip(samples, energies_i)):
            n_steps = float(
                s.get("n_steps") or len(s.get("reasoning_steps") or []) or 1
            )
            ans = s.get("extracted_answer_norm") or s.get("extracted_answer") or ""
            ans_len = float(len(str(ans)))
            # Correct = 1 if this sample's answer matches gold
            correct_flag = s.get("correct")
            if correct_flag is None:
                sample_ans = s.get("extracted_answer_norm") or s.get("extracted_answer")
                correct_flag = (sample_ans == gold) if sample_ans is not None else False
            y_i = 1 if correct_flag else 0
            # Features: [process_energy, n_steps, answer_len]
            # DO NOT add SC indicator here.
            rows_X.append([e_j, n_steps, ans_len])
            rows_y.append(y_i)
            rows_prob.append(i)

    if not rows_X:
        return np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    X = np.array(rows_X, dtype=float)
    y = np.array(rows_y, dtype=int)
    prob_idx = np.array(rows_prob, dtype=int)
    return X, y, prob_idx


# ---------------------------------------------------------------------------
# Reranker (THE FIX: StandardScaler + C=100 + sample weights)
# ---------------------------------------------------------------------------

def fit_energy_reranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None = None,
) -> Any:
    """Fit the FIXED energy reranker pipeline.

    THE FIX (applied here):
        1. StandardScaler as step 1 — normalizes energy features to zero-mean,
           unit-variance so L2 regularization treats each feature fairly.
        2. C=100 (weak regularization) — avoids over-shrinkage when true signal
           in energy features is small relative to noise.
        3. sample_weight support — caller passes weight=3.0 for SC-wrong problems
           (minority-recovery bias) to break the majority-vote baseline's grip.

    WHY this pipeline vs plain LogisticRegression:
        The old code used sklearn.linear_model.LogisticRegression(C=1.0) directly
        on raw features with no scaling.  Step-count values (range ~1..50) and
        answer-length values (range ~1..200) at incompatible scales cause the
        L2 penalty to shrink the larger-scale weights harder.  StandardScaler
        equalizes the penalty.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=100, max_iter=500, random_state=42)),
        ]
    )
    if w_train is not None and len(w_train) == len(y_train):
        pipeline.fit(X_train, y_train, clf__sample_weight=w_train)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# 7-condition scoring
# ---------------------------------------------------------------------------

def score_7_conditions(
    records: list[dict],
    reranker: Any,
    energies: list[list[float]] | None = None,
    sc_majority: list[tuple[str | None, bool]] | None = None,
    train_mask: np.ndarray | None = None,
) -> dict[str, list[str | None]]:
    """Score all 7 conditions for each problem, returning the selected answer.

    Conditions:
      1. greedy — first generation or the one with is_greedy flag
      2. sc — majority vote
      3. self_certainty_bon — best-of-N by mean logprob (or SC if no logprobs)
      4. process_energy_argmin — argmin of normalized step-count energy
      5. trained_energy_vote — FIXED reranker selects by highest P(correct) score
      6. sc_energy_hybrid — 0.5*SC_probability + 0.5*energy_score per candidate
      7. optimal_aggregation — trained with reranker on this fold's TRAIN split

    WHY: computing all 7 conditions lets us attribute any accuracy gain to the
    specific mechanism (flip analysis), not just to an uncontrolled variable.
    """
    if energies is None:
        energies = compute_process_energy(records)
    if sc_majority is None:
        sc_majority = build_sc_majority(records)

    cond: dict[str, list[str | None]] = {
        "greedy": [],
        "sc": [],
        "self_certainty_bon": [],
        "process_energy_argmin": [],
        "trained_energy_vote": [],
        "sc_energy_hybrid": [],
        "optimal_aggregation": [],
    }

    # Extract features for the entire set (needed for reranker + hybrid)
    X_all, _, prob_idx_all = _extract_features(records, energies, sc_majority)

    for i, (rec, e_i, sc_i) in enumerate(zip(records, energies, sc_majority)):
        samples = rec.get("samples") or []
        gold = rec.get("gold_answer_norm") or rec.get("gold_answer")

        # 1. greedy
        greedy = rec.get("greedy") or {}
        greedy_ans = greedy.get("extracted_answer_norm") or greedy.get("extracted_answer")
        if greedy_ans is None and samples:
            greedy_ans = (
                samples[0].get("extracted_answer_norm")
                or samples[0].get("extracted_answer")
            )
        cond["greedy"].append(greedy_ans)

        # 2. sc
        cond["sc"].append(sc_i[0])

        # answers list
        answers = [
            s.get("extracted_answer_norm") or s.get("extracted_answer")
            for s in samples
        ]

        # 3. self_certainty_bon — best by logprob; fall back to SC if no logprobs
        logprobs = [s.get("mean_token_logprob") for s in samples]
        if any(lp is not None for lp in logprobs):
            best_lp_idx = max(
                range(len(logprobs)),
                key=lambda j: (logprobs[j] if logprobs[j] is not None else float("-inf")),
            )
            cond["self_certainty_bon"].append(answers[best_lp_idx] if answers else None)
        else:
            cond["self_certainty_bon"].append(sc_i[0])  # fall back to SC

        # 4. process_energy_argmin — pick min energy (best quality)
        if e_i:
            best_e_idx = int(np.argmin(e_i))
            cond["process_energy_argmin"].append(answers[best_e_idx] if answers else None)
        else:
            cond["process_energy_argmin"].append(sc_i[0])

        # 5. trained_energy_vote — highest P(correct) from the FIXED reranker
        rows_for_i = np.where(prob_idx_all == i)[0]
        if len(rows_for_i) > 0 and reranker is not None:
            X_i = X_all[rows_for_i]
            try:
                probs_i = reranker.predict_proba(X_i)[:, 1]
                best_r_idx = int(np.argmax(probs_i))
                sel_answer = answers[best_r_idx] if answers else None
            except Exception:
                sel_answer = sc_i[0]
        else:
            sel_answer = sc_i[0]
        cond["trained_energy_vote"].append(sel_answer)

        # 6. sc_energy_hybrid — 0.5*SC_prob + 0.5*(1-energy) per candidate
        #    SC probability for candidate c = count(c in answers) / len(answers)
        if answers and e_i:
            answer_counts = Counter(a for a in answers if a is not None)
            total_valid = sum(answer_counts.values())
            hybrid_scores: list[float] = []
            for j, ans in enumerate(answers):
                sc_prob = answer_counts.get(ans, 0) / max(total_valid, 1)
                energy_score = 1.0 - (e_i[j] if j < len(e_i) else 0.5)
                hybrid_scores.append(0.5 * sc_prob + 0.5 * energy_score)
            best_h_idx = int(np.argmax(hybrid_scores))
            cond["sc_energy_hybrid"].append(answers[best_h_idx])
        else:
            cond["sc_energy_hybrid"].append(sc_i[0])

        # 7. optimal_aggregation — same as trained_energy_vote (uses the fold reranker)
        cond["optimal_aggregation"].append(cond["trained_energy_vote"][-1])

    return cond


# ---------------------------------------------------------------------------
# Flip metrics
# ---------------------------------------------------------------------------

def compute_flip_metrics(
    cond_selections: list[str | None],
    sc_selections: list[str | None],
    labels: list[bool],
) -> dict[str, int]:
    """Compute flip-count and net correctness gain of cond vs SC.

    WHY flip metrics are the PRIMARY signal (not a raw accuracy delta):
    Accuracy differences can be noise-driven at n=50-80; flip counts are
    exact integers that directly show HOW MANY problems the reranker is
    treating differently from the SC baseline.  A delta=0.01 at n=49
    with flip_count=0 is a degeneracy finding, not a 1pp improvement.
    """
    flip_count = 0
    flips_correct = 0
    flips_incorrect = 0
    for cond_ans, sc_ans, correct in zip(cond_selections, sc_selections, labels):
        if cond_ans != sc_ans:
            flip_count += 1
            if correct and cond_ans == correct:
                # This path doesn't apply since 'correct' is a bool, not the answer
                pass
    # Recompute properly: labels here are booleans for cond_selections correctness
    flip_count = 0
    flips_correct = 0
    flips_incorrect = 0
    for cond_ans, sc_ans, cond_correct in zip(cond_selections, sc_selections, labels):
        if cond_ans != sc_ans:
            flip_count += 1
            if cond_correct:
                flips_correct += 1
            else:
                flips_incorrect += 1
    return {
        "flip_count": flip_count,
        "flips_correct": flips_correct,
        "flips_incorrect": flips_incorrect,
        "net_correctness_gain": flips_correct - flips_incorrect,
    }


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def compute_mcnemar_significance(
    cond_correct: list[bool],
    sc_correct: list[bool],
    seed: int,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """McNemar exact test + bootstrap CI95.

    WHY McNemar: paired binary outcomes (same problem tested under two conditions);
    McNemar is the standard test for matched-pairs binary data.  We use exact
    form (scipy chi2 is approximate; we compute the exact binomial directly when
    discordant cells are small).
    """
    import scipy.stats as ss

    n = len(cond_correct)
    assert len(sc_correct) == n

    # Discordant cells
    n01 = sum(1 for c, s in zip(cond_correct, sc_correct) if not c and s)  # SC right, cond wrong
    n10 = sum(1 for c, s in zip(cond_correct, sc_correct) if c and not s)  # cond right, SC wrong

    # Exact McNemar: binomial p-value that n10 > n01 under H0: p=0.5
    total_discordant = n01 + n10
    if total_discordant == 0:
        mcnemar_p = 1.0
    else:
        # Two-sided exact binomial test: P(X >= n10 | X~Bin(n01+n10, 0.5))
        mcnemar_p = float(
            2.0 * min(
                ss.binom.cdf(min(n10, n01), total_discordant, 0.5),
                1.0 - ss.binom.cdf(max(n10, n01) - 1, total_discordant, 0.5),
                0.5,
            )
        )

    # Bootstrap CI95
    rng = np.random.default_rng(seed)
    cond_arr = np.array(cond_correct, dtype=float)
    sc_arr = np.array(sc_correct, dtype=float)
    boot_deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas.append(float(cond_arr[idx].mean() - sc_arr[idx].mean()))
    boot_deltas_sorted = sorted(boot_deltas)
    lo = boot_deltas_sorted[int(0.025 * n_boot)]
    hi = boot_deltas_sorted[int(0.975 * n_boot)]

    return {"mcnemar_p": mcnemar_p, "bootstrap_ci95": [lo, hi]}


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------

def build_artifact(fields: dict[str, Any]) -> dict[str, Any]:
    """Build and return the result artifact dict with all required fields.

    WHY: centralizing the artifact schema in one function ensures that blocked
    paths and scored paths emit the same key set, so downstream capstone tasks
    never cascade-block on a missing field.
    """
    required_keys = [
        "experiment", "run_date", "honest_verdict", "inference_substrate",
        "collapse_root_cause", "reranker_makes_distinct_selections",
        "level3_n", "self_consistency_accuracy",
        "greedy_accuracy", "self_certainty_bon_accuracy",
        "process_energy_argmin_accuracy", "trained_energy_vote_accuracy",
        "sc_energy_hybrid_accuracy", "optimal_aggregation_accuracy",
        "flip_count_trained_vs_sc", "flip_count_process_vs_sc",
        "flip_count_optimal_vs_sc",
        "flips_correct_optimal", "flips_incorrect_optimal",
        "net_correctness_gain_optimal",
        "delta_optimal_vs_self_consistency",
        "paired_significance", "random_seed",
        "reproducibility_checksum", "duration_s",
        "preconditions_checked",
        "acceptance_gates",
    ]
    base: dict[str, Any] = {
        "experiment": 3519,
        "run_date": datetime.now(timezone.utc).isoformat(),
        "honest_verdict": "complete: unknown",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "collapse_root_cause": COLLAPSE_ROOT_CAUSE,
        "reranker_makes_distinct_selections": False,
        "level3_n": 0,
        "self_consistency_accuracy": None,
        "greedy_accuracy": None,
        "self_certainty_bon_accuracy": None,
        "process_energy_argmin_accuracy": None,
        "trained_energy_vote_accuracy": None,
        "sc_energy_hybrid_accuracy": None,
        "optimal_aggregation_accuracy": None,
        "flip_count_trained_vs_sc": 0,
        "flip_count_process_vs_sc": 0,
        "flip_count_optimal_vs_sc": 0,
        "flips_correct_optimal": 0,
        "flips_incorrect_optimal": 0,
        "net_correctness_gain_optimal": 0,
        "delta_optimal_vs_self_consistency": None,
        "paired_significance": {"mcnemar_p": 1.0, "bootstrap_ci95": [0.0, 0.0]},
        "random_seed": 0,
        "reproducibility_checksum": "",
        "duration_s": 0.0,
        "preconditions_checked": [],
        "acceptance_gates": {"G0_nondegeneracy": False, "G1_energy_beats_sc": False},
    }
    base.update(fields)
    for k in required_keys:
        if k not in base:
            base[k] = None
    return base


# ---------------------------------------------------------------------------
# Main scoring loop (cross-validation)
# ---------------------------------------------------------------------------

def _run_cv(
    records: list[dict],
    energies: list[list[float]],
    sc_majority: list[tuple[str | None, bool]],
    seed: int,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run stratified K-fold cross-validation over all 7 conditions.

    WHY cross-validation instead of train/test split: with only 40-80 problems,
    a single split wastes half the data for training the reranker.  K-fold CV
    produces held-out predictions for every problem and ensures no problem's
    answer leaks into its own evaluation.
    """
    from sklearn.model_selection import StratifiedKFold

    n = len(records)
    n_folds_actual = min(n_folds, n // 8) or 2

    X_all, y_all, prob_idx_all = _extract_features(records, energies, sc_majority)

    # Labels for stratification: use SC correctness (binary)
    sc_correct_arr = np.array([sc[1] for sc in sc_majority], dtype=int)

    skf = StratifiedKFold(n_splits=n_folds_actual, shuffle=True, random_state=seed)

    # Collect held-out predictions per problem
    gold_labels: list[bool] = [
        bool((rec.get("gold_answer_norm") or rec.get("gold_answer")) is not None)
        for rec in records
    ]  # will be overwritten with actual correctness below

    # We need per-problem predictions across ALL conditions
    pred_greedy: list[str | None] = [None] * n
    pred_sc: list[str | None] = [None] * n
    pred_certainty_bon: list[str | None] = [None] * n
    pred_process_argmin: list[str | None] = [None] * n
    pred_trained: list[str | None] = [None] * n
    pred_hybrid: list[str | None] = [None] * n
    pred_optimal: list[str | None] = [None] * n
    gold_answers: list[str | None] = []

    for rec in records:
        gold_answers.append(rec.get("gold_answer_norm") or rec.get("gold_answer"))

    for fold_train, fold_test in skf.split(np.arange(n), sc_correct_arr):
        # Train reranker on fold_train problems
        train_rows = np.isin(prob_idx_all, fold_train)
        X_tr, y_tr = X_all[train_rows], y_all[train_rows]
        prob_tr = prob_idx_all[train_rows]

        # Sample weights: 3.0 where SC was wrong on this training problem
        w_tr = np.ones(len(y_tr), dtype=float)
        for j, pidx in enumerate(prob_tr):
            if not sc_majority[pidx][1]:  # SC wrong on this problem
                w_tr[j] = 3.0

        reranker = fit_energy_reranker(X_tr, y_tr, w_tr)

        # Score test problems
        test_records = [records[i] for i in fold_test]
        test_energies = [energies[i] for i in fold_test]
        test_sc = [sc_majority[i] for i in fold_test]

        cond = score_7_conditions(
            test_records,
            reranker,
            energies=test_energies,
            sc_majority=test_sc,
        )

        for local_j, global_i in enumerate(fold_test):
            pred_greedy[global_i] = cond["greedy"][local_j]
            pred_sc[global_i] = cond["sc"][local_j]
            pred_certainty_bon[global_i] = cond["self_certainty_bon"][local_j]
            pred_process_argmin[global_i] = cond["process_energy_argmin"][local_j]
            pred_trained[global_i] = cond["trained_energy_vote"][local_j]
            pred_hybrid[global_i] = cond["sc_energy_hybrid"][local_j]
            pred_optimal[global_i] = cond["optimal_aggregation"][local_j]

    def _acc(preds: list[str | None]) -> float:
        correct = sum(1 for p, g in zip(preds, gold_answers) if p == g and g is not None)
        return correct / max(n, 1)

    def _correct_arr(preds: list[str | None]) -> list[bool]:
        return [p == g and g is not None for p, g in zip(preds, gold_answers)]

    sc_correct_list = _correct_arr(pred_sc)
    opt_correct_list = _correct_arr(pred_optimal)
    trained_correct_list = _correct_arr(pred_trained)
    process_correct_list = _correct_arr(pred_process_argmin)

    flip_trained = compute_flip_metrics(pred_trained, pred_sc, trained_correct_list)
    flip_process = compute_flip_metrics(pred_process_argmin, pred_sc, process_correct_list)
    flip_optimal = compute_flip_metrics(pred_optimal, pred_sc, opt_correct_list)

    sc_acc = _acc(pred_sc)
    opt_acc = _acc(pred_optimal)

    sig = compute_mcnemar_significance(opt_correct_list, sc_correct_list, seed=seed)

    reranker_distinct = flip_trained["flip_count"] > 0 or flip_optimal["flip_count"] > 0

    g0 = reranker_distinct
    g1 = (
        g0
        and flip_optimal["net_correctness_gain"] > 0
        and (opt_acc - sc_acc) > 0
        and sig["mcnemar_p"] < 0.05
    )

    return {
        "greedy_accuracy": round(_acc(pred_greedy), 6),
        "self_consistency_accuracy": round(sc_acc, 6),
        "self_certainty_bon_accuracy": round(_acc(pred_certainty_bon), 6),
        "process_energy_argmin_accuracy": round(_acc(pred_process_argmin), 6),
        "trained_energy_vote_accuracy": round(_acc(pred_trained), 6),
        "sc_energy_hybrid_accuracy": round(_acc(pred_hybrid), 6),
        "optimal_aggregation_accuracy": round(opt_acc, 6),
        "flip_count_trained_vs_sc": flip_trained["flip_count"],
        "flip_count_process_vs_sc": flip_process["flip_count"],
        "flip_count_optimal_vs_sc": flip_optimal["flip_count"],
        "flips_correct_optimal": flip_optimal["flips_correct"],
        "flips_incorrect_optimal": flip_optimal["flips_incorrect"],
        "net_correctness_gain_optimal": flip_optimal["net_correctness_gain"],
        "delta_optimal_vs_self_consistency": round(opt_acc - sc_acc, 6),
        "paired_significance": sig,
        "reranker_makes_distinct_selections": reranker_distinct,
        "acceptance_gates": {"G0_nondegeneracy": g0, "G1_energy_beats_sc": g1},
        "n_folds_used": n_folds_actual,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full experiment: load corpus, check preconditions, score 7 conditions."""
    start = time.time()
    preconditions: list[dict] = []

    # Compute seed from corpus content (NOT experiment number 3519)
    if CORPUS_PATH.exists():
        corpus_bytes = CORPUS_PATH.read_bytes()
        seed = int(hashlib.sha256(corpus_bytes).hexdigest()[:8], 16) % (2**31)
        checksum_input = corpus_bytes + json.dumps(
            {
                "exp": 3519,
                "headroom_low": HEADROOM_LOW,
                "headroom_high": HEADROOM_HIGH,
                "min_problems": MIN_PROBLEMS,
                "min_samples": MIN_SAMPLES_PER_PROBLEM,
            },
            sort_keys=True,
        ).encode()
        checksum = hashlib.sha256(checksum_input).hexdigest()[:16]
    else:
        seed = 0
        checksum = "no_corpus"

    # PRECONDITION 1: corpus file must exist
    corpus_ok = CORPUS_PATH.exists()
    preconditions.append(
        {"resource": "level3_corpus_file", "available": corpus_ok}
    )
    if not corpus_ok:
        artifact = build_artifact(
            {
                "honest_verdict": "complete: blocked_corpus_file_missing",
                "duration_s": round(time.time() - start, 3),
                "preconditions_checked": preconditions,
                "random_seed": seed,
                "reproducibility_checksum": checksum,
            }
        )
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))
        print(artifact["honest_verdict"])
        return

    # PRECONDITION 2: load level-3 records
    records = load_level3_records(CORPUS_PATH)
    level3_n = len(records)
    preconditions.append(
        {
            "resource": "level3_corpus_size",
            "available": level3_n >= MIN_PROBLEMS,
            "level3_n": level3_n,
            "min_required": MIN_PROBLEMS,
        }
    )
    if level3_n < MIN_PROBLEMS:
        artifact = build_artifact(
            {
                "honest_verdict": f"complete: blocked_level3_corpus_too_small_n={level3_n}",
                "level3_n": level3_n,
                "duration_s": round(time.time() - start, 3),
                "preconditions_checked": preconditions,
                "random_seed": seed,
                "reproducibility_checksum": checksum,
            }
        )
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))
        print(artifact["honest_verdict"])
        return

    # PRECONDITION 3: sklearn available
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.pipeline import Pipeline  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False
    preconditions.append({"resource": "sklearn", "available": sklearn_ok})
    if not sklearn_ok:
        artifact = build_artifact(
            {
                "honest_verdict": "complete: blocked_sklearn_not_installed",
                "level3_n": level3_n,
                "duration_s": round(time.time() - start, 3),
                "preconditions_checked": preconditions,
                "random_seed": seed,
                "reproducibility_checksum": checksum,
            }
        )
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))
        print(artifact["honest_verdict"])
        return

    # Compute SC and headroom check
    sc_majority = build_sc_majority(records)
    sc_acc_overall = sum(1 for _, ok in sc_majority if ok) / max(level3_n, 1)
    in_band = HEADROOM_LOW <= sc_acc_overall <= HEADROOM_HIGH
    preconditions.append(
        {
            "resource": "sc_in_headroom_band",
            "available": in_band,
            "sc_accuracy": round(sc_acc_overall, 4),
            "headroom_band": [HEADROOM_LOW, HEADROOM_HIGH],
        }
    )

    # Compute process energies
    energies = compute_process_energy(records)

    # Cross-validation scoring
    cv_result = _run_cv(records, energies, sc_majority, seed=seed, n_folds=5)

    # NON-DEGENERACY CHECK
    if not cv_result["reranker_makes_distinct_selections"]:
        verdict = "complete: blocked_reranker_still_degenerate_consensus_trap"
    elif cv_result["acceptance_gates"]["G1_energy_beats_sc"]:
        verdict = (
            "complete: process_energy_beats_self_consistency_in_band"
            "_phase3_premise_validated"
        )
    elif cv_result["acceptance_gates"]["G0_nondegeneracy"]:
        verdict = (
            "complete: energy_makes_distinct_selections_but_does_not_beat"
            "_self_consistency_in_band"
        )
    else:
        verdict = "complete: blocked_reranker_still_degenerate_consensus_trap"

    artifact = build_artifact(
        {
            "honest_verdict": verdict,
            "level3_n": level3_n,
            "duration_s": round(time.time() - start, 3),
            "preconditions_checked": preconditions,
            "random_seed": seed,
            "reproducibility_checksum": checksum,
            **cv_result,
        }
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))

    print(
        f"DONE: {verdict}\n"
        f"  level3_n={level3_n} sc={sc_acc_overall:.4f} in_band={in_band}\n"
        f"  greedy={cv_result['greedy_accuracy']:.4f} "
        f"sc={cv_result['self_consistency_accuracy']:.4f}\n"
        f"  process_argmin={cv_result['process_energy_argmin_accuracy']:.4f} "
        f"trained={cv_result['trained_energy_vote_accuracy']:.4f} "
        f"optimal={cv_result['optimal_aggregation_accuracy']:.4f}\n"
        f"  flip_trained={cv_result['flip_count_trained_vs_sc']} "
        f"flip_process={cv_result['flip_count_process_vs_sc']} "
        f"flip_optimal={cv_result['flip_count_optimal_vs_sc']}\n"
        f"  G0_nondegen={cv_result['acceptance_gates']['G0_nondegeneracy']} "
        f"G1_beat_sc={cv_result['acceptance_gates']['G1_energy_beats_sc']}\n"
        f"  dur={artifact['duration_s']}s"
    )


if __name__ == "__main__":
    main()
