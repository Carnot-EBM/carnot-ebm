#!/usr/bin/env python3
"""Experiment: Per-token EBM for hallucination detection via rejection sampling.

WHY PER-TOKEN: Previous experiments (9-12) used mean-pooled sequence activations,
yielding only ~42 training examples that overfit badly (94% cal, 35% test).
The collect_token_activations.py script extracts one activation per generated token,
giving us 5000+ examples. This script trains a Gibbs EBM on those per-token
activations and evaluates whether it separates correct from hallucinated tokens
better than logprob-based rejection sampling (+10% from experiment 13).

ARCHITECTURE: Gibbs [1024 -> 128 -> 32 -> 1] via NCE.
  - 1024 = PCA reduction from 2048-dim Qwen3-0.6B hidden states
  - Two hidden layers (128, 32) to capture nonlinear activation patterns
  - NCE training: correct tokens = low energy, hallucinated tokens = high energy

EVALUATION:
  1. Calibration accuracy: does energy separate correct vs wrong tokens on train?
  2. Test accuracy: does it generalize to held-out tokens?
  3. Rejection sampling: for each question, generate N candidates, pick lowest
     mean per-token energy. Compare vs logprob baseline (+10% from exp 13).

Usage:
    python scripts/experiment_per_token_ebm.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
PCA_DIM = 1024          # Reduce 2048-dim activations to 1024 via PCA
HIDDEN_DIMS = [128, 32] # Gibbs hidden layers
N_EPOCHS = 200          # NCE training epochs
LEARNING_RATE = 0.005   # Gradient descent step size
TRAIN_FRACTION = 0.8    # 80/20 train/test split by question
N_CANDIDATES = 5        # Candidates per question for rejection sampling
TEMPERATURE = 0.8       # Sampling temperature for candidate generation
SEED = 42


def load_token_activations(path: str) -> dict[str, np.ndarray]:
    """Load per-token activations from safetensors file.

    Returns dict with keys: token_ids, activations, labels, question_ids.
    Each array has N elements (one per generated token across all questions).
    """
    from safetensors.numpy import load_file

    data = load_file(path)
    logger.info(
        "Loaded %d tokens, dim=%d, correct=%d, wrong=%d",
        data["activations"].shape[0],
        data["activations"].shape[1],
        int(data["labels"].sum()),
        int((1 - data["labels"]).sum()),
    )
    return data


def split_by_question(
    data: dict[str, np.ndarray], train_frac: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split data into train/test by question_id to avoid leakage.

    Tokens from the same question always go to the same split.
    This prevents the model from memorizing question-specific patterns
    and leaking train information into the test set.
    """
    rng = np.random.default_rng(seed)
    question_ids = np.unique(data["question_ids"])
    rng.shuffle(question_ids)

    n_train = int(len(question_ids) * train_frac)
    train_qids = set(question_ids[:n_train].tolist())
    test_qids = set(question_ids[n_train:].tolist())

    train_mask = np.isin(data["question_ids"], list(train_qids))
    test_mask = np.isin(data["question_ids"], list(test_qids))

    train = {k: v[train_mask] for k, v in data.items()}
    test = {k: v[test_mask] for k, v in data.items()}

    logger.info(
        "Split: %d train tokens (%d questions), %d test tokens (%d questions)",
        train_mask.sum(), len(train_qids), test_mask.sum(), len(test_qids),
    )
    return train, test


def fit_pca(activations: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training activations.

    Returns (mean, components) where:
      - mean: shape (D,) — the centroid of the training data
      - components: shape (n_components, D) — the top principal axes

    To project new data: projected = (x - mean) @ components.T
    """
    mean = activations.mean(axis=0)
    centered = activations - mean
    # Truncated SVD: only need top n_components singular vectors
    # For N >> D this is efficient; for N < D we'd use the other form
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    variance_explained = _S[:n_components] ** 2 / (_S ** 2).sum()
    logger.info(
        "PCA: %d -> %d dims, variance explained: %.1f%%",
        activations.shape[1], n_components, 100 * variance_explained.sum(),
    )
    return mean, components


def train_ebm(
    correct_acts: np.ndarray,
    wrong_acts: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
):
    """Train Gibbs EBM via NCE on PCA-projected per-token activations.

    NCE setup:
      - "data" = correct token activations (should get LOW energy)
      - "noise" = hallucinated token activations (should get HIGH energy)
      - Loss pushes correct tokens to low energy, wrong tokens to high energy

    Returns: (trained_model, project_fn)
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    pca_dim = pca_components.shape[0]

    def project(x: np.ndarray) -> jnp.ndarray:
        """Project raw activations to PCA space."""
        return jnp.array(((x - pca_mean) @ pca_components.T).astype(np.float32))

    # Project training data
    correct_pca = jnp.array(project(correct_acts))
    wrong_pca = jnp.array(project(wrong_acts))

    # Balance classes for NCE: use equal number of correct and wrong tokens.
    # If imbalanced, subsample the larger class.
    min_n = min(len(correct_pca), len(wrong_pca))
    logger.info("Balanced NCE: %d correct, %d wrong -> %d each", len(correct_pca), len(wrong_pca), min_n)

    rng = np.random.default_rng(SEED)
    if len(correct_pca) > min_n:
        idx = rng.choice(len(correct_pca), min_n, replace=False)
        correct_pca = correct_pca[idx]
    if len(wrong_pca) > min_n:
        idx = rng.choice(len(wrong_pca), min_n, replace=False)
        wrong_pca = wrong_pca[idx]

    # Create Gibbs model: input_dim=PCA_DIM, hidden=[128, 32], SiLU activation
    config = GibbsConfig(input_dim=pca_dim, hidden_dims=HIDDEN_DIMS, activation="silu")
    key = jrandom.PRNGKey(SEED)
    model = GibbsModel(config, key=key)

    # Extract mutable params for gradient descent.
    # The Gibbs model stores params as attributes; we pack/unpack for jax.grad.
    def get_params(m):
        return {
            "layers": [(w, b) for w, b in m.layers],
            "output_weight": m.output_weight,
            "output_bias": m.output_bias,
        }

    def set_params(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    def loss_fn(params):
        """NCE loss: correct tokens = data (low energy), wrong = noise (high energy)."""
        old = get_params(model)
        set_params(model, params)
        result = nce_loss(model, correct_pca, wrong_pca)
        set_params(model, old)
        return result

    # Training loop: vanilla gradient descent
    params = get_params(model)
    losses = []
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        loss_val = loss_fn(params)
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - LEARNING_RATE * g, params, grads)

        if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
            losses.append(float(loss_val))
            logger.info("  epoch %3d/%d  loss=%.4f", epoch, N_EPOCHS, float(loss_val))

    set_params(model, params)
    train_time = time.time() - t0
    logger.info("Training done in %.1fs, final loss=%.4f", train_time, losses[-1])

    return model, project


def evaluate_classification(model, project_fn, activations, labels, split_name):
    """Evaluate per-token classification: does energy separate correct from wrong?

    Computes:
      - Mean energy for correct vs wrong tokens
      - Energy gap (want: correct < wrong)
      - Classification accuracy at optimal threshold
    """
    import jax.numpy as jnp

    projected = project_fn(activations)
    energies = np.array([float(model.energy(projected[i])) for i in range(len(projected))])

    correct_mask = labels == 1
    wrong_mask = labels == 0

    mean_correct = energies[correct_mask].mean() if correct_mask.any() else 0.0
    mean_wrong = energies[wrong_mask].mean() if wrong_mask.any() else 0.0
    gap = mean_wrong - mean_correct

    # Threshold at midpoint between means
    threshold = (mean_correct + mean_wrong) / 2.0
    # Correct tokens should have energy <= threshold (low energy = good)
    # Wrong tokens should have energy > threshold (high energy = bad)
    tp = (energies[wrong_mask] > threshold).sum() if wrong_mask.any() else 0
    tn = (energies[correct_mask] <= threshold).sum() if correct_mask.any() else 0
    total = correct_mask.sum() + wrong_mask.sum()
    accuracy = (tp + tn) / total if total > 0 else 0.0

    print(f"\n  {split_name} Classification:")
    print(f"    Correct tokens: {correct_mask.sum()}, mean energy: {mean_correct:.4f}")
    print(f"    Wrong tokens:   {wrong_mask.sum()}, mean energy: {mean_wrong:.4f}")
    print(f"    Energy gap:     {gap:.4f} ({'good' if gap > 0 else 'BAD: wrong < correct'})")
    print(f"    Threshold:      {threshold:.4f}")
    print(f"    Accuracy:       {accuracy:.1%} ({tp+tn}/{total})")

    return {
        "accuracy": accuracy,
        "gap": gap,
        "mean_correct_energy": mean_correct,
        "mean_wrong_energy": mean_wrong,
        "threshold": threshold,
        "n_correct": int(correct_mask.sum()),
        "n_wrong": int(wrong_mask.sum()),
    }


def evaluate_rejection_sampling(
    model, project_fn, data, test_qids,
):
    """Rejection sampling: for each test question, score candidates by mean energy.

    Since we don't have a live LLM here, we simulate rejection sampling using
    the collected token activations. For each test question:
      1. Get all tokens for that question
      2. Compute mean per-token energy (the sequence-level EBM score)
      3. The "candidate" with lowest mean energy would be selected

    We evaluate: does the EBM correctly identify questions with correct answers
    (low mean energy) vs questions with wrong answers (high mean energy)?
    """
    question_ids = data["question_ids"]
    labels = data["labels"]
    activations = data["activations"]

    results = []
    for qid in test_qids:
        mask = question_ids == qid
        q_acts = activations[mask]
        q_labels = labels[mask]

        if len(q_acts) == 0:
            continue

        # Project and compute per-token energies
        projected = project_fn(q_acts)
        energies = np.array([float(model.energy(projected[i])) for i in range(len(projected))])

        # Mean energy for this question's tokens
        mean_energy = energies.mean()
        # All tokens for a question share the same label (sequence-level)
        is_correct = q_labels[0] == 1

        results.append({
            "question_id": int(qid),
            "mean_energy": float(mean_energy),
            "is_correct": bool(is_correct),
            "n_tokens": len(q_acts),
        })

    return results


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: Per-Token EBM for Hallucination Detection")
    print(f"Architecture: Gibbs [{PCA_DIM} -> {' -> '.join(map(str, HIDDEN_DIMS))} -> 1]")
    print(f"Training: NCE, {N_EPOCHS} epochs, lr={LEARNING_RATE}")
    print(f"Split: {TRAIN_FRACTION:.0%} train / {1-TRAIN_FRACTION:.0%} test (by question)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load per-token activations
    # ------------------------------------------------------------------
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations.safetensors")
    data_path = os.path.abspath(data_path)

    if not os.path.exists(data_path):
        print(f"\nERROR: {data_path} not found.")
        print("Run first: python scripts/collect_token_activations.py")
        return 1

    data = load_token_activations(data_path)
    n_total = data["activations"].shape[0]
    hidden_dim = data["activations"].shape[1]
    n_questions = len(np.unique(data["question_ids"]))

    print(f"\nDataset: {n_total} tokens from {n_questions} questions, dim={hidden_dim}")
    print(f"  Correct tokens: {int(data['labels'].sum())}")
    print(f"  Wrong tokens:   {int((1 - data['labels']).sum())}")

    # ------------------------------------------------------------------
    # Step 2: Train/test split by question (no leakage)
    # ------------------------------------------------------------------
    train, test = split_by_question(data, TRAIN_FRACTION, SEED)

    # ------------------------------------------------------------------
    # Step 3: Fit PCA on training activations, then train Gibbs EBM
    # ------------------------------------------------------------------
    pca_dim = min(PCA_DIM, hidden_dim, train["activations"].shape[0])
    if pca_dim < PCA_DIM:
        logger.info("PCA dim capped at %d (limited by data or hidden dim)", pca_dim)

    pca_mean, pca_components = fit_pca(train["activations"], pca_dim)

    # Separate train tokens by label for NCE
    train_correct = train["activations"][train["labels"] == 1]
    train_wrong = train["activations"][train["labels"] == 0]

    print(f"\nTraining EBM on {len(train_correct)} correct + {len(train_wrong)} wrong tokens...")
    model, project_fn = train_ebm(train_correct, train_wrong, pca_mean, pca_components)

    # ------------------------------------------------------------------
    # Step 4: Evaluate token-level classification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL CLASSIFICATION")
    print("=" * 60)

    train_metrics = evaluate_classification(
        model, project_fn, train["activations"], train["labels"], "Train"
    )
    test_metrics = evaluate_classification(
        model, project_fn, test["activations"], test["labels"], "Test"
    )

    # ------------------------------------------------------------------
    # Step 5: Evaluate question-level rejection sampling
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUESTION-LEVEL REJECTION SAMPLING (mean per-token energy)")
    print("=" * 60)

    test_qids = np.unique(test["question_ids"])
    rej_results = evaluate_rejection_sampling(model, project_fn, test, test_qids)

    if not rej_results:
        print("\n  No test questions available for rejection sampling.")
        return 1

    # Sort questions by mean energy: correct answers should have lower energy
    rej_results.sort(key=lambda r: r["mean_energy"])

    # Compute question-level accuracy using same threshold approach:
    # correct questions (label=1) should have low mean energy
    correct_energies = [r["mean_energy"] for r in rej_results if r["is_correct"]]
    wrong_energies = [r["mean_energy"] for r in rej_results if not r["is_correct"]]

    if correct_energies and wrong_energies:
        mean_correct_q = np.mean(correct_energies)
        mean_wrong_q = np.mean(wrong_energies)
        gap_q = mean_wrong_q - mean_correct_q
        threshold_q = (mean_correct_q + mean_wrong_q) / 2.0

        tp_q = sum(1 for e in wrong_energies if e > threshold_q)
        tn_q = sum(1 for e in correct_energies if e <= threshold_q)
        total_q = len(correct_energies) + len(wrong_energies)
        acc_q = (tp_q + tn_q) / total_q

        print(f"\n  Correct questions: {len(correct_energies)}, mean energy: {mean_correct_q:.4f}")
        print(f"  Wrong questions:   {len(wrong_energies)}, mean energy: {mean_wrong_q:.4f}")
        print(f"  Energy gap:        {gap_q:.4f}")
        print(f"  Question-level accuracy: {acc_q:.1%} ({tp_q+tn_q}/{total_q})")

        # Simulated rejection sampling: if we had N candidates per question,
        # the EBM would rank them by mean energy and pick the lowest.
        # Here we assess: does the energy ordering correlate with correctness?
        # A "fix" = wrong answer gets high energy (would be rejected).
        # A "regression" = correct answer gets high energy (would be wrongly rejected).
        n_fixes = sum(1 for r in rej_results if not r["is_correct"] and r["mean_energy"] > threshold_q)
        n_regressions = sum(1 for r in rej_results if r["is_correct"] and r["mean_energy"] > threshold_q)
        print(f"  Fixes (wrong -> rejected): {n_fixes}")
        print(f"  Regressions (correct -> rejected): {n_regressions}")
        print(f"  Net improvement: {n_fixes - n_regressions:+d}")
    else:
        print("\n  WARNING: Need both correct and wrong questions in test set.")
        acc_q = 0.0
        gap_q = 0.0

    # ------------------------------------------------------------------
    # Step 6: Comparison with logprob baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON WITH LOGPROB BASELINE")
    print("=" * 60)

    logprob_improvement = 0.10  # +10% from experiment 13
    ebm_cal_acc = train_metrics["accuracy"]
    ebm_test_acc = test_metrics["accuracy"]
    overfit_gap = ebm_cal_acc - ebm_test_acc

    print(f"\n  Logprob rejection (exp 13):  +10% improvement over greedy")
    print(f"  EBM calibration accuracy:    {ebm_cal_acc:.1%}")
    print(f"  EBM test accuracy:           {ebm_test_acc:.1%}")
    print(f"  Overfit gap:                 {overfit_gap:.1%} (cal - test)")
    if correct_energies and wrong_energies:
        print(f"  Question-level accuracy:     {acc_q:.1%}")

    if overfit_gap > 0.20:
        print("\n  WARNING: >20% overfit gap. Model may need regularization.")
    if ebm_test_acc > 0.60:
        print("  PROMISING: Test accuracy > 60% -- EBM is learning real signal.")
    if ebm_test_acc > 0.70:
        print("  STRONG: Test accuracy > 70% -- EBM likely outperforms logprob baseline.")

    # ------------------------------------------------------------------
    # Step 7: Summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset:               {n_total} tokens, {n_questions} questions")
    print(f"  Architecture:          Gibbs [{pca_dim} -> {' -> '.join(map(str, HIDDEN_DIMS))} -> 1]")
    print(f"  Training:              NCE, {N_EPOCHS} epochs, lr={LEARNING_RATE}")
    print(f"  Calibration accuracy:  {ebm_cal_acc:.1%}")
    print(f"  Test accuracy:         {ebm_test_acc:.1%}")
    print(f"  Train energy gap:      {train_metrics['gap']:.4f}")
    print(f"  Test energy gap:       {test_metrics['gap']:.4f}")
    if correct_energies and wrong_energies:
        print(f"  Question-level acc:    {acc_q:.1%}")
    print(f"  Logprob baseline:      +10% (experiment 13)")

    verdict = "INCONCLUSIVE"
    if ebm_test_acc > 0.70 and test_metrics["gap"] > 0:
        verdict = "SUCCESS: Per-token EBM shows strong generalization"
    elif ebm_test_acc > 0.55 and test_metrics["gap"] > 0:
        verdict = "PARTIAL: Per-token EBM shows some signal, needs tuning"
    elif test_metrics["gap"] <= 0:
        verdict = "FAILURE: EBM cannot separate correct from wrong on test set"
    print(f"\n  VERDICT: {verdict}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
