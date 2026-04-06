#!/usr/bin/env python3
"""Train per-token EBM on 26K+ token activations.

Previous: 1860 tokens → 71.8% test accuracy.
This run: 26,800 tokens → ?

Usage:
    .venv/bin/python scripts/train_per_token_ebm_large.py
"""

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

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_large.safetensors")


def main():
    data = load_file(DATA_FILE)
    activations = jnp.array(data["activations"])
    labels = data["labels"]
    print(f"Dataset: {activations.shape[0]} tokens, dim={activations.shape[1]}")
    print(f"Correct: {labels.sum()}, Wrong: {len(labels) - labels.sum()}")

    correct_mask = labels == 1
    wrong_mask = labels == 0
    correct_acts = activations[correct_mask]
    wrong_acts = activations[wrong_mask]

    min_n = min(len(correct_acts), len(wrong_acts))
    correct_acts = correct_acts[:min_n]
    wrong_acts = wrong_acts[:min_n]
    print(f"Balanced: {min_n} each")

    split = int(min_n * 0.8)
    train_correct = correct_acts[:split]
    train_wrong = wrong_acts[:split]
    test_correct = correct_acts[split:]
    test_wrong = wrong_acts[split:]
    print(f"Train: {split} each, Test: {min_n - split} each")

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
    print(f"Training: {n_epochs} epochs, lr={lr}")

    start = time.time()
    for epoch in range(n_epochs):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        if epoch % 50 == 0:
            set_params(ebm, params)
            loss = float(nce_loss(ebm, train_correct, train_wrong))
            print(f"  Epoch {epoch}: loss={loss:.4f}")
    set_params(ebm, params)
    elapsed = time.time() - start
    print(f"Training done in {elapsed:.1f}s")

    # Evaluate
    n_eval = min(500, len(train_correct))
    train_c_e = [float(ebm.energy(train_correct[i])) for i in range(n_eval)]
    train_w_e = [float(ebm.energy(train_wrong[i])) for i in range(n_eval)]
    test_c_e = [float(ebm.energy(test_correct[i])) for i in range(len(test_correct))]
    test_w_e = [float(ebm.energy(test_wrong[i])) for i in range(len(test_wrong))]

    train_thresh = (sum(train_c_e) / len(train_c_e) + sum(train_w_e) / len(train_w_e)) / 2
    test_thresh = (sum(test_c_e) / len(test_c_e) + sum(test_w_e) / len(test_w_e)) / 2

    train_tp = sum(1 for e in train_w_e if e > train_thresh)
    train_tn = sum(1 for e in train_c_e if e <= train_thresh)
    train_acc = (train_tp + train_tn) / (len(train_c_e) + len(train_w_e))

    test_tp = sum(1 for e in test_w_e if e > test_thresh)
    test_tn = sum(1 for e in test_c_e if e <= test_thresh)
    test_acc = (test_tp + test_tn) / (len(test_c_e) + len(test_w_e))

    train_gap = sum(train_w_e) / len(train_w_e) - sum(train_c_e) / len(train_c_e)
    test_gap = sum(test_w_e) / len(test_w_e) - sum(test_c_e) / len(test_c_e)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"RESULTS: Per-Token EBM on {activations.shape[0]} tokens")
    print(sep)
    print(f"  Previous (1860 tokens):  cal=87.8%, test=71.8%")
    print(f"  This run ({activations.shape[0]} tokens):")
    print(f"    Train accuracy: {train_acc:.1%}")
    print(f"    Test accuracy:  {test_acc:.1%}")
    print(f"    Train gap:      {train_gap:.4f}")
    print(f"    Test gap:       {test_gap:.4f}")
    print(sep)

    if test_acc > 0.718:
        print(f"IMPROVEMENT: {test_acc:.1%} > 71.8% (more data helped!)")
    elif test_acc > 0.7:
        print(f"SIMILAR: {test_acc:.1%} ~ 71.8%")
    else:
        print(f"NO IMPROVEMENT: {test_acc:.1%} <= 71.8%")


if __name__ == "__main__":
    main()
