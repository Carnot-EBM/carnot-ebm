#!/usr/bin/env python3
"""Experiment 26: Cross-model EBM transfer.

Tests whether an EBM trained on one model's activations can detect
hallucinations in a different model with the same hidden dimension.

If the hallucination "direction" in activation space is universal across
architectures, a single EBM could work for any model of matching dimension.

Groups tested:
  - 1024-dim: LFM2.5-350M <-> Qwen3.5-0.8B
  - 1536-dim: Gemma4-E2B <-> Gemma4-E2B-it
  - 2048-dim: Bonsai-1.7B <-> LFM2.5-1.2B <-> Qwen3.5-2B

Usage:
    .venv/bin/python scripts/experiment_26_cross_model_transfer.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_activations(short_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load activations and labels for a model."""
    from safetensors.numpy import load_file

    # Try nothink variant first, then generic
    for suffix in [f"_{short_id}_nothink", f"_{short_id}"]:
        path = os.path.join(DATA_DIR, f"token_activations{suffix}.safetensors")
        if os.path.exists(path):
            d = load_file(path)
            return d["activations"], d["labels"]

    raise FileNotFoundError(f"No activation data for {short_id}")


def train_ebm_on(activations: np.ndarray, labels: np.ndarray, hidden_dim: int, seed: int = 42):
    """Train a Gibbs EBM and return it."""
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
        return None

    rng = np.random.default_rng(seed)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct[:split], wrong[:split]

    if hidden_dim <= 1024:
        hdims = [256, 64]
    elif hidden_dim <= 2048:
        hdims = [512, 128]
    else:
        hdims = [1024, 256]

    key = jrandom.PRNGKey(seed)
    config = GibbsConfig(input_dim=hidden_dim, hidden_dims=hdims, activation="silu")
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
    return ebm


def evaluate_ebm(ebm, activations: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Evaluate EBM on a dataset. Returns (accuracy, energy_gap)."""
    import jax.numpy as jnp

    correct = activations[labels == 1]
    wrong = activations[labels == 0]
    min_n = min(len(correct), len(wrong))
    if min_n < 10:
        return 0.5, 0.0

    rng = np.random.default_rng(99)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]

    # Use test split (last 20%)
    split = int(min_n * 0.8)
    vc, vw = correct[split:], wrong[split:]

    n_eval = min(300, len(vc))
    ce = [float(ebm.energy(jnp.array(vc[i]))) for i in range(n_eval)]
    we = [float(ebm.energy(jnp.array(vw[i]))) for i in range(n_eval)]

    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    acc = (tp + tn) / (len(ce) + len(we))
    gap = np.mean(we) - np.mean(ce)
    return acc, gap


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 26: Cross-Model EBM Transfer")
    print("=" * 70)

    # Define groups by hidden dimension
    groups = {
        1024: [
            ("lfm25-350m", "LFM2.5-350M"),
            ("qwen35", "Qwen3.5-0.8B"),
        ],
        1536: [
            ("gemma4-e2b", "Gemma4-E2B (base)"),
            ("gemma4-e2b-it", "Gemma4-E2B-it"),
        ],
        2048: [
            ("bonsai-17b", "Bonsai-1.7B"),
            ("lfm25-12b", "LFM2.5-1.2B"),
            ("qwen35-2b", "Qwen3.5-2B"),
        ],
    }

    all_results = []
    start = time.time()

    for dim, models in groups.items():
        print(f"\n{'='*70}")
        print(f"DIMENSION GROUP: {dim}")
        print(f"{'='*70}")

        # Load all datasets in this group
        data = {}
        for short_id, label in models:
            try:
                acts, labels = load_activations(short_id)
                data[short_id] = (acts, labels, label)
                print(f"  Loaded {label}: {len(labels)} tokens")
            except FileNotFoundError:
                print(f"  SKIP {label}: no data")

        if len(data) < 2:
            print("  Need at least 2 models for transfer test")
            continue

        # Train EBM on each model, evaluate on all others
        ebms = {}
        for short_id in data:
            acts, labels, label = data[short_id]
            print(f"\n  Training EBM on {label}...")
            ebm = train_ebm_on(acts, labels, dim)
            if ebm is None:
                print(f"    SKIP: insufficient data")
                continue
            ebms[short_id] = ebm

            # Self-evaluation
            acc, gap = evaluate_ebm(ebm, acts, labels)
            print(f"    Self:     {acc:.1%} (gap={gap:.4f})")

        # Cross-evaluation matrix
        print(f"\n  --- Transfer Matrix (dim={dim}) ---")
        header = f"  {'Trained on':<20s}"
        for short_id in data:
            _, _, label = data[short_id]
            header += f" {label[:15]:>15s}"
        print(header)
        print("  " + "-" * (20 + 16 * len(data)))

        for train_id in ebms:
            _, _, train_label = data[train_id]
            row = f"  {train_label:<20s}"
            for eval_id in data:
                acts, labels, _ = data[eval_id]
                acc, gap = evaluate_ebm(ebms[train_id], acts, labels)
                marker = " *" if train_id == eval_id else ""
                row += f" {acc:>13.1%}{marker}"
                all_results.append({
                    "dim": dim,
                    "trained_on": train_label,
                    "evaluated_on": data[eval_id][2],
                    "accuracy": acc,
                    "gap": gap,
                    "is_self": train_id == eval_id,
                })
            print(row)

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 26 SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*70}")

    # Compute transfer rates
    self_accs = [r["accuracy"] for r in all_results if r["is_self"]]
    cross_accs = [r["accuracy"] for r in all_results if not r["is_self"]]

    if self_accs and cross_accs:
        mean_self = np.mean(self_accs)
        mean_cross = np.mean(cross_accs)
        transfer_rate = mean_cross / mean_self

        print(f"  Mean self-accuracy:    {mean_self:.1%}")
        print(f"  Mean cross-accuracy:   {mean_cross:.1%}")
        print(f"  Transfer rate:         {transfer_rate:.1%}")
        print(f"  Transfer penalty:      {mean_self - mean_cross:.1%}")

        # Same-family vs cross-family
        same_family = [r for r in all_results if not r["is_self"]
                       and r["trained_on"].split()[0] == r["evaluated_on"].split()[0]]
        cross_family = [r for r in all_results if not r["is_self"]
                        and r["trained_on"].split()[0] != r["evaluated_on"].split()[0]]

        if same_family:
            print(f"  Same-family transfer:  {np.mean([r['accuracy'] for r in same_family]):.1%}")
        if cross_family:
            print(f"  Cross-family transfer: {np.mean([r['accuracy'] for r in cross_family]):.1%}")

        if transfer_rate > 0.85:
            print(f"\n  VERDICT: ✅ Strong transfer ({transfer_rate:.0%}) — hallucination direction may be universal!")
        elif transfer_rate > 0.7:
            print(f"\n  VERDICT: ⚠️ Partial transfer ({transfer_rate:.0%}) — some shared structure")
        else:
            print(f"\n  VERDICT: ❌ Poor transfer ({transfer_rate:.0%}) — model-specific representations")

    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
