#!/usr/bin/env python3
"""Train per-token EBM on activation datasets.

IMPORTANT: Existing QA data (Qwen3-0.6B) and TruthfulQA data (Qwen3.5-0.8B)
are from different models with different representation spaces. They should NOT
be combined for training. This script trains on each source separately.

Previous results:
  - 1,860 tokens (Qwen3-0.6B):  cal=87.8%, test=71.8%
  - 26,800 tokens (Qwen3-0.6B): cal=~90%, test=84.1%
  - TruthfulQA (~27K tokens, Qwen3.5-0.8B): ?

Usage:
    .venv/bin/python scripts/train_per_token_ebm_combined.py [--source qa|tqa|both]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from safetensors.numpy import load_file

from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss

# Data files
MERGED_QWEN35 = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_qwen35_merged.safetensors")
COMBINED = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_combined.safetensors")
QA_ONLY = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_large.safetensors")


def evaluate(ebm, correct, wrong, label=""):
    """Compute accuracy and energy gap."""
    c_e = [float(ebm.energy(correct[i])) for i in range(len(correct))]
    w_e = [float(ebm.energy(wrong[i])) for i in range(len(wrong))]

    thresh = (sum(c_e) / len(c_e) + sum(w_e) / len(w_e)) / 2
    tp = sum(1 for e in w_e if e > thresh)
    tn = sum(1 for e in c_e if e <= thresh)
    acc = (tp + tn) / (len(c_e) + len(w_e))
    gap = sum(w_e) / len(w_e) - sum(c_e) / len(c_e)
    print(f"  {label}: acc={acc:.1%}, gap={gap:.4f}, thresh={thresh:.4f}")
    return acc, gap


def load_data(source="tqa"):
    """Load activations, filtering by source model.

    source="qa":  QA dataset only (Qwen3-0.6B)
    source="tqa": TruthfulQA only (Qwen3.5-0.8B)
    source="both": all data (WARNING: different model spaces)
    """
    if source == "merged":
        if not os.path.exists(MERGED_QWEN35):
            raise FileNotFoundError(f"Run merge_activations_qwen35.py first: {MERGED_QWEN35}")
        data = load_file(MERGED_QWEN35)
        activations = data["activations"]
        labels = data["labels"]
        model_name = "Qwen3.5-0.8B (QA + TruthfulQA)"
        mask = np.ones(len(labels), dtype=bool)
        print(f"Source: {source} ({model_name})")
        print(f"Tokens: {len(labels)}, dim={activations.shape[1]}")
        print(f"Correct: {int(labels.sum())}, Wrong: {int(len(labels) - labels.sum())}")
        return jnp.array(activations), labels

    # Legacy: load from combined or QA-only files
    if os.path.exists(COMBINED):
        data = load_file(COMBINED)
    elif os.path.exists(QA_ONLY):
        data = load_file(QA_ONLY)
    else:
        raise FileNotFoundError("No activation data found")

    activations = data["activations"]
    labels = data["labels"]
    question_ids = data["question_ids"]

    if source == "qa":
        mask = question_ids < 10000
        model_name = "Qwen3-0.6B"
    elif source == "tqa":
        mask = question_ids >= 10000
        model_name = "Qwen3.5-0.8B"
    else:
        mask = np.ones(len(labels), dtype=bool)
        model_name = "MIXED (Qwen3-0.6B + Qwen3.5-0.8B)"
        print("  WARNING: Combining activations from different models!")

    activations = jnp.array(activations[mask])
    labels = labels[mask]

    print(f"Source: {source} ({model_name})")
    print(f"Tokens: {len(labels)}, dim={activations.shape[1]}")
    print(f"Correct: {int(labels.sum())}, Wrong: {int(len(labels) - labels.sum())}")
    return activations, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["qa", "tqa", "both", "merged"], default="merged",
                        help="Data source: qa (Qwen3-0.6B), tqa (Qwen3.5-0.8B), both (mixed models), merged (QA+TQA from Qwen3.5-0.8B)")
    args = parser.parse_args()

    activations, labels = load_data(args.source)

    correct_mask = np.array(labels == 1)
    wrong_mask = np.array(labels == 0)
    correct_acts = activations[correct_mask]
    wrong_acts = activations[wrong_mask]

    min_n = min(len(correct_acts), len(wrong_acts))
    # Shuffle before truncating to mix QA and TruthfulQA tokens
    rng = np.random.default_rng(42)
    c_idx = rng.permutation(len(correct_acts))[:min_n]
    w_idx = rng.permutation(len(wrong_acts))[:min_n]
    correct_acts = correct_acts[c_idx]
    wrong_acts = wrong_acts[w_idx]
    print(f"Balanced: {min_n} each (shuffled)")

    split = int(min_n * 0.8)
    train_correct = correct_acts[:split]
    train_wrong = wrong_acts[:split]
    test_correct = correct_acts[split:]
    test_wrong = wrong_acts[split:]
    print(f"Train: {split} each, Test: {min_n - split} each")

    # Architecture: same as best from architecture search
    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=1024, hidden_dims=[256, 64], activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_params(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_params(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    def loss_fn(params):
        old = get_params(ebm)
        set_params(ebm, params)
        r = nce_loss(ebm, train_correct, train_wrong)
        set_params(ebm, old)
        return r

    params = get_params(ebm)
    lr = 0.005
    n_epochs = 300
    print(f"\nTraining: {n_epochs} epochs, lr={lr}")

    start = time.time()
    best_loss = float("inf")
    for epoch in range(n_epochs):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        if epoch % 50 == 0:
            set_params(ebm, params)
            loss = float(nce_loss(ebm, train_correct, train_wrong))
            best_loss = min(best_loss, loss)
            print(f"  Epoch {epoch}: loss={loss:.4f}")
    set_params(ebm, params)
    elapsed = time.time() - start
    print(f"Training done in {elapsed:.1f}s")

    # Evaluate
    print("\nEvaluation:")
    n_eval = min(500, len(train_correct))
    train_acc, train_gap = evaluate(ebm, train_correct[:n_eval], train_wrong[:n_eval], "Train")
    test_acc, test_gap = evaluate(ebm, test_correct, test_wrong, "Test")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"RESULTS: Per-Token EBM on Combined Dataset")
    print(sep)
    print(f"  Previous results:")
    print(f"    1,860 tokens:   cal=87.8%, test=71.8%")
    print(f"    26,800 tokens:  cal=~90%, test=84.1%")
    print(f"  This run ({activations.shape[0]} tokens):")
    print(f"    Train accuracy: {train_acc:.1%}")
    print(f"    Test accuracy:  {test_acc:.1%}")
    print(f"    Train gap:      {train_gap:.4f}")
    print(f"    Test gap:       {test_gap:.4f}")
    print(sep)

    if test_acc > 0.841:
        print(f"IMPROVEMENT: {test_acc:.1%} > 84.1% (more diverse data helped!)")
    elif test_acc > 0.8:
        print(f"SIMILAR: {test_acc:.1%} ~ 84.1%")
    else:
        print(f"REGRESSION: {test_acc:.1%} < 84.1% (domain shift?)")


if __name__ == "__main__":
    main()
