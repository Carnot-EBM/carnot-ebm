#!/usr/bin/env python3
"""Experiment 35: Activation normalization for cross-domain/model transfer.

Experiments 26 (cross-model) and 31 (cross-domain) both failed because raw
activations are model/domain-specific. Can normalization fix this?

Tests three normalization strategies:
  1. Z-score: subtract mean, divide by std (per-dimension across the dataset)
  2. L2-normalize: unit-norm each activation vector
  3. Whitening: PCA-based decorrelation + scaling

For each, we test:
  - Same-domain performance (should stay similar)
  - Cross-domain transfer (train MMLU → test TruthfulQA)
  - Cross-model transfer (train Qwen3.5-0.8B → test on another 1024-dim model)

REQ: Roadmap v5, Phase 1, Experiment 35
SCENARIO: Does normalization enable the domain/model transfer that raw activations can't?

Usage:
    .venv/bin/python scripts/experiment_35_activation_normalization.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def train_and_eval(activations, labels, input_dim, label):
    """Train EBM and evaluate."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    acts = jnp.array(activations)
    correct = acts[labels == 1]
    wrong = acts[labels == 0]
    min_n = min(len(correct), len(wrong))
    if min_n < 20:
        print(f"  {label}: SKIP — {min_n} samples")
        return 0.5

    rng = np.random.default_rng(42)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct[:split], wrong[:split]
    vc, vw = correct[split:], wrong[split:]

    hdims = [256, 64] if input_dim <= 1024 else [512, 128]
    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=input_dim, hidden_dims=hdims, activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_p(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm)

    def loss_fn(p):
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, tc, tw)
        set_p(ebm, old)
        return r

    for _ in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm, params)

    n = min(300, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n)]
    we = [float(ebm.energy(vw[i])) for i in range(n)]
    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    acc = (tp + tn) / (len(ce) + len(we))
    print(f"  {label:40s}: {acc:.1%}")
    return acc


def train_on_eval_on(train_acts, train_labels, eval_acts, eval_labels, input_dim, label):
    """Train on one dataset, evaluate on another."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    tc_all = jnp.array(train_acts[train_labels == 1])
    tw_all = jnp.array(train_acts[train_labels == 0])
    min_t = min(len(tc_all), len(tw_all))
    rng = np.random.default_rng(42)
    tc = tc_all[rng.permutation(len(tc_all))[:min_t]]
    tw = tw_all[rng.permutation(len(tw_all))[:min_t]]
    split = int(min_t * 0.8)
    tc, tw = tc[:split], tw[:split]

    hdims = [256, 64] if input_dim <= 1024 else [512, 128]
    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=input_dim, hidden_dims=hdims, activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_p(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm)

    def loss_fn(p):
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, tc, tw)
        set_p(ebm, old)
        return r

    for _ in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm, params)

    ec = jnp.array(eval_acts[eval_labels == 1])
    ew = jnp.array(eval_acts[eval_labels == 0])
    min_e = min(len(ec), len(ew), 300)
    rng2 = np.random.default_rng(99)
    ec = ec[rng2.permutation(len(ec))[:min_e]]
    ew = ew[rng2.permutation(len(ew))[:min_e]]

    ce = [float(ebm.energy(ec[i])) for i in range(min_e)]
    we = [float(ebm.energy(ew[i])) for i in range(min_e)]
    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    acc = (tp + tn) / (len(ce) + len(we))
    print(f"  {label:40s}: {acc:.1%}")
    return acc


def normalize_zscore(acts):
    """Z-score normalization: zero mean, unit variance per dimension."""
    mean = acts.mean(axis=0)
    std = acts.std(axis=0) + 1e-8
    return (acts - mean) / std, mean, std


def normalize_l2(acts):
    """L2 normalize each activation vector to unit norm."""
    norms = np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8
    return acts / norms


def normalize_pca_whiten(acts, n_components=None):
    """PCA whitening: decorrelate and scale to unit variance."""
    mean = acts.mean(axis=0)
    centered = acts - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]
    # Whiten
    whitening = eigvecs / np.sqrt(eigvals + 1e-8)
    return centered @ whitening, mean, whitening


def main() -> int:
    from safetensors.numpy import load_file

    print("=" * 70)
    print("EXPERIMENT 35: Activation Normalization for Cross-Domain Transfer")
    print("=" * 70)

    start = time.time()

    # Load multi-dataset activations
    multi_file = os.path.join(DATA_DIR, "token_activations_qwen3508b_multi.safetensors")
    multi = load_file(multi_file)
    acts = multi["activations"]
    labels = multi["labels"]
    source_ids = multi["source_ids"]
    hidden_dim = acts.shape[1]

    sources = {0: "halueval", 1: "mmlu", 2: "simpleqa", 3: "truthfulqa"}
    print(f"  Multi-dataset: {len(labels)} tokens, dim={hidden_dim}")
    for sid, name in sources.items():
        mask = source_ids == sid
        print(f"    {name}: {int(mask.sum())} tokens ({int(labels[mask].sum())} correct)")

    # --- Part 1: Same-domain with different normalizations ---
    print(f"\n{'='*70}")
    print("PART 1: Same-domain accuracy under normalization")
    print(f"{'='*70}")

    tqa_mask = source_ids == 3
    tqa_acts = acts[tqa_mask]
    tqa_labels = labels[tqa_mask]

    print("\n  TruthfulQA:")
    train_and_eval(tqa_acts, tqa_labels, hidden_dim, "Raw (baseline)")

    tqa_z, _, _ = normalize_zscore(tqa_acts)
    train_and_eval(tqa_z, tqa_labels, hidden_dim, "Z-score normalized")

    tqa_l2 = normalize_l2(tqa_acts)
    train_and_eval(tqa_l2, tqa_labels, hidden_dim, "L2 normalized")

    tqa_w, _, _ = normalize_pca_whiten(tqa_acts, n_components=256)
    train_and_eval(tqa_w, tqa_labels, 256, "PCA whitened (256-dim)")

    # --- Part 2: Cross-domain transfer with normalization ---
    print(f"\n{'='*70}")
    print("PART 2: Cross-domain transfer (train MMLU → test TruthfulQA)")
    print(f"{'='*70}")

    mmlu_mask = source_ids == 1
    mmlu_acts = acts[mmlu_mask]
    mmlu_labels = labels[mmlu_mask]

    print("\n  Raw:")
    train_on_eval_on(mmlu_acts, mmlu_labels, tqa_acts, tqa_labels, hidden_dim, "MMLU → TruthfulQA (raw)")

    print("  Z-score (fit on combined, transform separately):")
    all_z, mean_all, std_all = normalize_zscore(acts)
    mmlu_z = (mmlu_acts - mean_all) / (std_all + 1e-8)
    tqa_z2 = (tqa_acts - mean_all) / (std_all + 1e-8)
    train_on_eval_on(mmlu_z, mmlu_labels, tqa_z2, tqa_labels, hidden_dim, "MMLU → TruthfulQA (z-score)")

    print("  L2:")
    mmlu_l2 = normalize_l2(mmlu_acts)
    tqa_l2_2 = normalize_l2(tqa_acts)
    train_on_eval_on(mmlu_l2, mmlu_labels, tqa_l2_2, tqa_labels, hidden_dim, "MMLU → TruthfulQA (L2)")

    # --- Part 3: Cross-model transfer with normalization ---
    print(f"\n{'='*70}")
    print("PART 3: Cross-model transfer (same hidden dim, different models)")
    print(f"{'='*70}")

    # Load two 1024-dim models
    qwen_file = os.path.join(DATA_DIR, "token_activations_qwen35_nothink.safetensors")
    lfm_file = os.path.join(DATA_DIR, "token_activations_lfm25-350m_nothink.safetensors")

    if os.path.exists(qwen_file) and os.path.exists(lfm_file):
        qwen = load_file(qwen_file)
        lfm = load_file(lfm_file)

        print("\n  Raw:")
        train_on_eval_on(qwen["activations"], qwen["labels"],
                         lfm["activations"], lfm["labels"], 1024,
                         "Qwen3.5 → LFM2.5 (raw)")

        print("  Z-score (fit on combined):")
        combined = np.concatenate([qwen["activations"], lfm["activations"]])
        mean_c = combined.mean(axis=0)
        std_c = combined.std(axis=0) + 1e-8
        qwen_z = (qwen["activations"] - mean_c) / std_c
        lfm_z = (lfm["activations"] - mean_c) / std_c
        train_on_eval_on(qwen_z, qwen["labels"], lfm_z, lfm["labels"], 1024,
                         "Qwen3.5 → LFM2.5 (z-score)")

        print("  L2:")
        qwen_l2 = normalize_l2(qwen["activations"])
        lfm_l2 = normalize_l2(lfm["activations"])
        train_on_eval_on(qwen_l2, qwen["labels"], lfm_l2, lfm["labels"], 1024,
                         "Qwen3.5 → LFM2.5 (L2)")
    else:
        print("  SKIP — need both Qwen3.5 and LFM2.5 nothink data")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 35 COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
