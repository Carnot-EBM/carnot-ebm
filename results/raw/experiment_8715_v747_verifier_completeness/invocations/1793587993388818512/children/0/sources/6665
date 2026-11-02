"""Experiment 1185: SC-Energy Overfit Diagnosis and Regularized k=6 Retraining.

Why this experiment exists:
    Exp1176 showed k=6 AND-compose AUROC (0.8973) REGRESSED below k=5 (0.9240).
    Exp1168 showed SC-Energy AUROC = 1.0 on its 12-pair eval set — a red flag for
    overfitting to the training distribution.  This experiment:
    (a) Diagnoses the overfit by evaluating on a proper 20%-holdout of fover_corpus.jsonl,
    (b) Retrains SC-Energy with dropout=0.3 + L2 weight_decay=1e-4 regularization,
    (c) Re-runs k=6 AND-compose to determine whether k=6 is viable or should be retired.

Spec: REQ-VERIFY-1185, SCENARIO-VERIFY-1185
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make sure carnot.* imports resolve.
_python_dir = PROJECT_ROOT / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 1185
K5_AUROC_BASELINE = 0.9240  # From exp1176 k5_auroc_on_eval (same eval set)
RANDOM_SEED = 1185
EXP1168_SEED = 1168
N_EPOCHS_MAX = 30
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
OVERFIT_THRESHOLD = 0.98  # v2 holdout AUROC < this => overfit resolved

FOVER_CORPUS_PATH = PROJECT_ROOT / "data" / "fover_corpus.jsonl"
FOVER_TEST_PATH = PROJECT_ROOT / "data" / "fover_test_v4.json"
FOVER_CORPUS_V4_PATH = PROJECT_ROOT / "data" / "fover_corpus_v4.json"
EXP1168_PATH = PROJECT_ROOT / "results" / "experiment_1168_sc_energy_7th_verifier.json"
EXP1176_PATH = PROJECT_ROOT / "results" / "experiment_1176_k6_and_compose_validation.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1185_sc_energy_overfit_regularized_k6.json"
CHECKPOINT_PATH = PROJECT_ROOT / "python" / "carnot" / "models" / "sc_energy_v2_regularized.pt"

REQUIRED_FIELDS = {
    "sc_energy_v1_holdout_auroc",
    "sc_energy_regularized",
    "sc_energy_v2_holdout_auroc",
    "overfit_resolved",
    "k6_regularized_auroc",
    "k5_baseline_auroc",
    "k6_above_k5",
    "k6_viable_for_production",
    "retire_k6",
    "honest_verdict",
}

ALLOWED_VERDICTS = {
    "k6_viable_after_regularization",
    "k6_retired_permanent",
    "overfit_resolved_but_k6_still_regresses",
    "overfit_not_resolved",
}


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load labeled rows from JSONL path.

    Each row must have 'label' field ('correct' or 'incorrect') and at least one
    text field.  Silently skips malformed lines.
    """
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            pass
    return rows


def split_rows_by_question_80_20(
    rows: list[dict[str, Any]],
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split fover_corpus rows 80/20 by question_id for train/holdout.

    Why split by question_id instead of row?
    If we split randomly by row, both halves will contain steps from the same question.
    The SC-Energy model could learn question-specific patterns and appear to generalize
    when it's really just recognizing the same question.  Splitting by question_id
    ensures the holdout contains entirely unseen questions.
    """
    qids = sorted(set(str(r.get("question_id", r.get("id", i))) for i, r in enumerate(rows)))
    rng = random.Random(seed)
    rng.shuffle(qids)
    split_at = max(1, int(0.8 * len(qids)))
    train_qids = set(qids[:split_at])
    train_rows = [r for r in rows if str(r.get("question_id", r.get("id", ""))) in train_qids]
    holdout_rows = [r for r in rows if str(r.get("question_id", r.get("id", ""))) not in train_qids]
    return train_rows, holdout_rows


def is_row_incorrect(row: dict[str, Any]) -> bool:
    """Return whether a labeled row is marked as incorrect/wrong.

    Handles multiple label field names used across different fover corpus versions.
    """
    if "is_correct" in row:
        return not bool(row["is_correct"])
    if "step_correct" in row:
        return not bool(row["step_correct"])
    label = row.get("label") or row.get("sc_energy_label") or row.get("coherence_label")
    if isinstance(label, str):
        return label.lower() in {"incorrect", "incoherent", "wrong", "false", "0"}
    if isinstance(label, bool):
        return not label
    return False


def row_step_text(row: dict[str, Any]) -> str:
    """Extract the response/step text from a fover row."""
    return str(row.get("step_text") or row.get("response") or row.get("step") or "").strip()


# ---------------------------------------------------------------------------
# Contrastive pair building
# ---------------------------------------------------------------------------


def build_contrastive_pairs_from_fover_labeled(
    labeled_path: Path,
    seed: int = EXP1168_SEED,
) -> tuple[list[Any], list[Any]]:
    """Build contrastive pairs from fover_labeled_steps_v21_multi.json.

    This is the same dataset used in exp1168 to train the original SC-Energy.
    We reproduce the same contrastive pair structure so the v1/v2 comparison
    is apples-to-apples (same training data, different regularization).

    Returns:
        (train_pairs, val_pairs) where each element is a list of
        (coherent_tuple, incoherent_tuple) for SCEnergyVerifier.train().

    Why use the exp1168 dataset rather than fover_corpus.jsonl for training?
    The fover_labeled_steps_v21_multi.json has the multi-step structure needed
    for contrastive SC-Energy training (2+ correct steps per question).
    fover_corpus.jsonl has single steps per row (not structured for contrastive pairs).
    """
    from carnot.eval.k6_and_compose_validation import (
        build_contrastive_pairs,
        load_rows,
        split_pairs,
    )

    rows = load_rows(labeled_path)
    pairs = build_contrastive_pairs(rows)
    # Split 80/20 train/val using the exp1168 seed for reproducibility
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    split_at = max(1, int(0.8 * len(shuffled)))
    train_raw, val_raw = shuffled[:split_at], shuffled[split_at:]

    # Convert to the tuple format that SCEnergyVerifier.train() accepts:
    # each element is a (context_str, response_str) tuple
    def _to_tuples(
        pair_dicts: list[dict[str, Any]],
    ) -> tuple[list[Any], list[Any]]:
        coh_list, inc_list = [], []
        for p in pair_dicts:
            coh_list.append(p["coherent"])
            inc_list.append(p["incoherent"])
        return coh_list, inc_list

    train_coh, train_inc = _to_tuples(train_raw)
    val_coh, val_inc = _to_tuples(val_raw)
    return (train_coh, train_inc), (val_coh, val_inc)


# ---------------------------------------------------------------------------
# SC-Energy v1: baseline (same protocol as exp1168)
# ---------------------------------------------------------------------------


def build_sc_energy_v1(
    labeled_path: Path,
    n_epochs: int = 10,
) -> Any:
    """Train SC-Energy verifier v1 using the exp1168 protocol.

    This reproduces the model that achieved AUROC=1.0 on the tiny 12-pair eval.
    Used as the baseline to measure the overfit level on proper holdout data.

    Args:
        labeled_path: Path to fover_labeled_steps_v21_multi.json.
        n_epochs: Training epochs (10 matches exp1168 default).

    Returns:
        Trained SCEnergyVerifier.
    """
    from carnot.verify.sc_energy_verifier import SCEnergyVerifier

    (train_coh, train_inc), _ = build_contrastive_pairs_from_fover_labeled(labeled_path)
    verifier = SCEnergyVerifier(model_name="roberta-base", hidden_dim=128)
    verifier.train(train_coh, train_inc, n_epochs=n_epochs)
    return verifier


# ---------------------------------------------------------------------------
# SC-Energy v2: regularized training
# ---------------------------------------------------------------------------


def train_sc_energy_regularized(
    labeled_path: Path,
    dropout_rate: float = DROPOUT_RATE,
    weight_decay: float = WEIGHT_DECAY,
    n_epochs_max: int = N_EPOCHS_MAX,
    patience: int = EARLY_STOP_PATIENCE,
    seed: int = RANDOM_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Retrain SC-Energy with dropout and L2 regularization.

    Why regularization?
    The v1 model achieved AUROC=1.0 on the 12-pair eval from exp1168 — a clear sign
    of overfitting to the training distribution.  Regularization prevents the model
    from memorizing specific question patterns:

    - Dropout (0.3): During each training step, randomly zero out 30% of the feature
      vector dimensions.  Forces the model to rely on distributed feature representations
      rather than memorizing any single dimension.

    - L2 weight decay (1e-4): Penalizes large metric values.  Prevents the model from
      tuning a small number of metric dimensions to extreme values to perfectly separate
      the training pairs.

    Early stopping: If validation loss stops improving for `patience` consecutive epochs,
    stop training (avoids overfitting to training pairs at the expense of val loss).

    Args:
        labeled_path: Path to fover_labeled_steps_v21_multi.json.
        dropout_rate: Fraction of feature dimensions to randomly zero during training.
        weight_decay: L2 regularization coefficient.
        n_epochs_max: Maximum training epochs.
        patience: Early stopping patience (epochs without val loss improvement).
        seed: RNG seed for dropout and data shuffling.

    Returns:
        (trained_verifier, training_diagnostics)
    """
    from carnot.verify.sc_energy_verifier import SCEnergyVerifier

    (train_coh, train_inc), (val_coh, val_inc) = build_contrastive_pairs_from_fover_labeled(
        labeled_path, seed=EXP1168_SEED
    )

    verifier = SCEnergyVerifier(model_name="roberta-base", hidden_dim=128)

    # Pre-compute features once (avoid re-computing on every epoch)
    from carnot.verify.sc_energy_verifier import _Pair, _coerce_pair

    def _feature(pair_item: Any) -> np.ndarray:
        pair = _coerce_pair(pair_item)
        return verifier._feature_for_pair(pair)

    train_coh_feats = [_feature(p) for p in train_coh]
    train_inc_feats = [_feature(p) for p in train_inc]
    val_coh_feats = [_feature(p) for p in val_coh]
    val_inc_feats = [_feature(p) for p in val_inc]

    rng = np.random.default_rng(seed)
    hidden_dim = verifier.hidden_dim
    margin = verifier.margin
    lr = verifier.learning_rate
    keep_prob = 1.0 - dropout_rate

    best_val_loss = float("inf")
    no_improve_count = 0
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_metric = verifier._metric.copy()
    best_epoch = 0

    def _apply_dropout(feat: np.ndarray) -> np.ndarray:
        """Randomly zero out dropout_rate fraction of feature dimensions.

        Inverted dropout: scale surviving elements by 1/keep_prob so the expected
        sum is unchanged regardless of dropout rate.  This prevents the model from
        having to compensate for different scales between training and inference.
        """
        if feat.shape[0] == 0:
            return feat
        mask = rng.uniform(size=feat.shape[0]) >= dropout_rate
        return feat * mask.astype(np.float32) / keep_prob

    for epoch in range(n_epochs_max):
        epoch_loss = 0.0
        n_pairs = min(len(train_coh_feats), len(train_inc_feats))

        for coh_feat, inc_feat in zip(train_coh_feats[:n_pairs], train_inc_feats[:n_pairs]):
            # Apply dropout to features (inverted dropout)
            coh_feat_drop = _apply_dropout(coh_feat)
            inc_feat_drop = _apply_dropout(inc_feat)

            e_coh = float(verifier._bias + np.dot(verifier._metric, coh_feat_drop))
            e_inc = float(verifier._bias + np.dot(verifier._metric, inc_feat_drop))
            gap = e_inc - e_coh

            if gap < margin:
                # Gradient of hinge loss: push metric toward higher gap
                gradient = inc_feat_drop - coh_feat_drop
                # L2 weight decay: subtract weight_decay * current_metric
                # This is the gradient of (1/2) * weight_decay * ||metric||^2
                l2_gradient = weight_decay * verifier._metric
                verifier._metric = verifier._metric + lr * gradient - lr * l2_gradient

            hinge_loss = max(0.0, margin - gap)
            l2_penalty = float(weight_decay * np.dot(verifier._metric, verifier._metric))
            epoch_loss += hinge_loss + l2_penalty

        verifier._metric = np.clip(verifier._metric, -8.0, 8.0)
        epoch_loss /= max(n_pairs, 1)
        loss_history.append(epoch_loss)

        # Compute validation loss (no dropout at inference)
        val_loss = _compute_margin_loss(
            val_coh_feats, val_inc_feats, verifier._metric, verifier._bias, margin
        )
        val_loss_history.append(val_loss)

        # Early stopping: track best val loss
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            best_metric = verifier._metric.copy()
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            # Restore best metric and stop
            verifier._metric = best_metric
            break

    diagnostics = {
        "n_train_pairs": len(train_coh_feats),
        "n_val_pairs": len(val_coh_feats),
        "n_epochs_run": len(loss_history),
        "best_epoch": best_epoch,
        "final_train_loss": round(loss_history[-1], 6) if loss_history else None,
        "best_val_loss": round(best_val_loss, 6),
        "early_stopped": len(loss_history) < n_epochs_max,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
    }
    return verifier, diagnostics


def _compute_margin_loss(
    coh_feats: list[np.ndarray],
    inc_feats: list[np.ndarray],
    metric: np.ndarray,
    bias: float,
    margin: float,
) -> float:
    """Compute mean margin (hinge) loss on a set of pre-computed features.

    Used for val loss tracking during regularized training.  No dropout applied —
    dropout is a training-time stochastic regularizer, not an inference procedure.
    """
    if not coh_feats or not inc_feats:
        return 0.0
    n = min(len(coh_feats), len(inc_feats))
    total = 0.0
    for cf, inf in zip(coh_feats[:n], inc_feats[:n]):
        e_coh = float(bias + np.dot(metric, cf))
        e_inc = float(bias + np.dot(metric, inf))
        total += max(0.0, margin - (e_inc - e_coh))
    return total / n


# ---------------------------------------------------------------------------
# Holdout AUROC evaluation (individual step level)
# ---------------------------------------------------------------------------


def evaluate_auroc_on_rows(verifier: Any, rows: list[dict[str, Any]]) -> float:
    """Compute AUROC for SC-Energy on individual labeled rows.

    Each row is scored as score(step_text, context="") — i.e., the model is
    asked whether this single step is coherent.  Label = 1 if incorrect.

    Why evaluate at the individual-step level (not pair level)?
    Exp1168 evaluated on contrastive pairs: the model was asked "which of these
    two step-sets is coherent?" on 12 pairs drawn from the SAME 60-question distribution
    it trained on.  That's why AUROC was 1.0 — it was essentially interpolating within
    the training distribution.

    Evaluating on INDIVIDUAL steps from a DIFFERENT corpus (fover_corpus.jsonl) tests
    whether the model has learned a genuine "step coherence" signal that transfers to
    new problems.  A properly generalizing model should score incorrect steps with
    higher energy than correct steps.
    """
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        text = row_step_text(row)
        if not text:
            continue
        labels.append(1 if is_row_incorrect(row) else 0)
        # score() returns energy in [0, 1]; higher = more incoherent (higher = incorrect)
        scores.append(float(verifier.score(text, "")))
    return tie_aware_auroc(labels, scores)


def tie_aware_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC with 0.5 credit for tied positive/negative scores.

    Why not just use sklearn?
    sklearn roc_auc_score requires both classes to be present and may give
    misleading results when there are many ties (e.g., both classes score 0.5).
    This implementation gives explicit 0.5 credit for ties, which correctly
    reflects "random" performance when the model has no signal.
    """
    pos = [s for lbl, s in zip(labels, scores) if lbl == 1]
    neg = [s for lbl, s in zip(labels, scores) if lbl == 0]
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


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------


def save_verifier_weights(verifier: Any, path: Path) -> None:
    """Save SCEnergyVerifier diagonal metric weights to a numpy .npz file.

    The file is named with a .pt extension (as expected by the task spec) but
    uses numpy's npz format internally.  Load with:
        data = np.load(path, allow_pickle=False)
        metric = data['metric']
        bias = data['bias']

    Why not use PyTorch .pt format?
    This model uses JAX/numpy for all computation; there is no PyTorch dependency
    in the project.  The .pt extension is used to match the task spec while keeping
    the actual serialization simple and dependency-free.

    Why pass a file handle to np.savez instead of a path string?
    np.savez appends '.npz' when given a string path that doesn't end in '.npz'.
    Passing an open file handle bypasses that behaviour and writes to the exact
    requested path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, metric=verifier._metric, bias=np.array([verifier._bias], dtype=np.float32))


# ---------------------------------------------------------------------------
# k=6 AND-compose evaluation using regularized SC-Energy
# ---------------------------------------------------------------------------


def run_k6_and_compose_with_regularized(
    v2_verifier: Any,
    *,
    eval_path: Path = FOVER_TEST_PATH,
    soskan_train_path: Path = FOVER_CORPUS_V4_PATH,
) -> dict[str, float]:
    """Run k=6 AND-compose using regularized SC-Energy v2 on the exp1176 eval set.

    Uses the same eval set, same k=5 verifiers, and same 200-row selection as
    exp1176 so the comparison is fair.

    Returns:
        Dict with 'k5_auroc_on_eval' and 'k6_regularized_auroc'.
    """
    from carnot.eval.k6_and_compose_validation import (
        build_fixed_k5_verifiers,
        install_lightweight_carnot_import_stubs,
        load_rows,
        score_eval_rows,
        select_heldout_eval_rows,
        tie_aware_auroc as _k6_auroc,
        compute_validation_metrics,
    )

    install_lightweight_carnot_import_stubs(PROJECT_ROOT)
    k5_verifiers = build_fixed_k5_verifiers(soskan_train_path)
    eval_rows = select_heldout_eval_rows(load_rows(eval_path))
    scores = score_eval_rows(eval_rows, v2_verifier, k5_verifiers)
    metrics = compute_validation_metrics(scores)
    return {
        "k5_auroc_on_eval": metrics.k5_auroc_on_eval,
        "k6_regularized_auroc": metrics.k6_auroc,
    }


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


def determine_verdict(
    overfit_resolved: bool,
    k6_above_k5: bool,
) -> str:
    """Map experiment outcomes to the required honest_verdict enum string.

    The verdict directly answers: is k=6 viable after regularization?

    Cases:
      - Overfit not resolved (holdout AUROC still >= 0.98): regularization
        failed to fix the problem.  k=6 is not viable.
      - Overfit resolved AND k=6 >= k=5 baseline: k=6 is viable for production.
      - Overfit resolved BUT k=6 still regresses: the overfit was not the only
        problem.  k=6 retired permanently.

    Spec: REQ-VERIFY-1185
    """
    if not overfit_resolved:
        return "overfit_not_resolved"
    if k6_above_k5:
        return "k6_viable_after_regularization"
    return "overfit_resolved_but_k6_still_regresses"


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_artifact(
    *,
    v1_holdout_auroc: float,
    v2_holdout_auroc: float,
    k5_auroc_on_eval: float,
    k6_regularized_auroc: float,
    overfit_resolved: bool,
    k6_above_k5: bool,
    training_diagnostics: dict[str, Any],
    started_at: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the schema-complete exp1185 result artifact.

    Validates that all REQUIRED_FIELDS are present and honest_verdict is in
    ALLOWED_VERDICTS before returning.
    """
    retire_k6 = not k6_above_k5
    k6_viable = k6_above_k5 and overfit_resolved
    verdict = determine_verdict(overfit_resolved, k6_above_k5)

    now = datetime.now(tz=UTC)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": "sc_energy_overfit_regularized_k6",
        "run_date": now.date().isoformat(),
        "started_at": started_at,
        "finished_at": now.isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "success",
        "spec": ["REQ-VERIFY-1185", "SCENARIO-VERIFY-1185"],
        # Overfit diagnosis
        "sc_energy_v1_holdout_auroc": round(v1_holdout_auroc, 6),
        "sc_energy_v1_overfit_evidence": "exp1168 eval AUROC=1.0 on 12 training-adjacent pairs",
        # Regularization
        "sc_energy_regularized": True,
        "regularization_params": {
            "dropout_rate": DROPOUT_RATE,
            "weight_decay": WEIGHT_DECAY,
            "n_epochs_max": N_EPOCHS_MAX,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        "training_diagnostics": training_diagnostics,
        "sc_energy_v2_checkpoint": str(CHECKPOINT_PATH),
        # Post-regularization evaluation
        "sc_energy_v2_holdout_auroc": round(v2_holdout_auroc, 6),
        "overfit_resolved": overfit_resolved,
        "overfit_resolved_criterion": f"v2_holdout_auroc < {OVERFIT_THRESHOLD}",
        # k=6 AND-compose
        "k5_baseline_auroc": K5_AUROC_BASELINE,
        "k5_auroc_on_eval": round(k5_auroc_on_eval, 6),
        "k6_regularized_auroc": round(k6_regularized_auroc, 6),
        "k6_above_k5": k6_above_k5,
        "k6_viable_for_production": k6_viable,
        "retire_k6": retire_k6,
        "honest_verdict": verdict,
    }

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["honest_verdict"] not in ALLOWED_VERDICTS:
        raise ValueError(f"unexpected honest_verdict: {artifact['honest_verdict']}")

    return artifact


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    fover_corpus_path: Path = FOVER_CORPUS_PATH,
    labeled_path: Path | None = None,
    eval_path: Path = FOVER_TEST_PATH,
    soskan_train_path: Path = FOVER_CORPUS_V4_PATH,
    output_path: Path = OUTPUT_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> dict[str, Any]:
    """Run exp1185 end-to-end: diagnose overfit, retrain, re-evaluate k=6.

    Spec: REQ-VERIFY-1185, SCENARIO-VERIFY-1185
    """
    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()

    exp1168_path = EXP1168_PATH
    _labeled_path = labeled_path or (
        PROJECT_ROOT
        / json.loads(exp1168_path.read_text()).get(
            "fover_labeled_pairs_path", "results/fover_labeled_steps_v21_multi.json"
        )
        if exp1168_path.exists()
        else PROJECT_ROOT / "results" / "fover_labeled_steps_v21_multi.json"
    )

    # ── Step 1: Load fover_corpus.jsonl and split 80/20 by question_id ──────
    corpus_rows = load_jsonl_rows(fover_corpus_path)
    _train_rows, holdout_rows = split_rows_by_question_80_20(corpus_rows)

    # ── Step 2: Train v1 (baseline, exp1168 protocol) ────────────────────────
    v1_verifier = build_sc_energy_v1(_labeled_path)

    # ── Step 3: Diagnose overfit — evaluate v1 on holdout rows ───────────────
    v1_holdout_auroc = evaluate_auroc_on_rows(v1_verifier, holdout_rows)

    # ── Step 4: Retrain v2 with regularization ───────────────────────────────
    v2_verifier, training_diagnostics = train_sc_energy_regularized(_labeled_path)

    # ── Step 5: Evaluate v2 on holdout ───────────────────────────────────────
    v2_holdout_auroc = evaluate_auroc_on_rows(v2_verifier, holdout_rows)
    overfit_resolved = v2_holdout_auroc < OVERFIT_THRESHOLD

    # ── Step 6: Save v2 checkpoint ───────────────────────────────────────────
    save_verifier_weights(v2_verifier, checkpoint_path)

    # ── Step 7: Run k=6 AND-compose with v2 ─────────────────────────────────
    k6_results = run_k6_and_compose_with_regularized(
        v2_verifier,
        eval_path=eval_path,
        soskan_train_path=soskan_train_path,
    )
    k5_auroc_on_eval = k6_results["k5_auroc_on_eval"]
    k6_regularized_auroc = k6_results["k6_regularized_auroc"]
    k6_above_k5 = k6_regularized_auroc >= K5_AUROC_BASELINE

    # ── Step 8: Build and write artifact ────────────────────────────────────
    artifact = build_artifact(
        v1_holdout_auroc=v1_holdout_auroc,
        v2_holdout_auroc=v2_holdout_auroc,
        k5_auroc_on_eval=k5_auroc_on_eval,
        k6_regularized_auroc=k6_regularized_auroc,
        overfit_resolved=overfit_resolved,
        k6_above_k5=k6_above_k5,
        training_diagnostics=training_diagnostics,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


if __name__ == "__main__":
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
