#!/usr/bin/env python3
"""Experiment 60: Scale Contrastive Divergence training to 100+ variable SAT.

Exp 50 trained Ising couplings via CD on 10-var SAT (100 parameters). This
experiment scales that approach to 50, 100, and 200 variables (up to 40K
parameters) and measures whether learned couplings outperform both random
baselines and hand-coded SAT-to-Ising mappings at generating satisfying
assignments.

**Why this matters:**
    Hand-coded SAT-to-Ising (experiment 39) encodes each clause's penalty
    structure exactly, but the resulting energy landscape may have many local
    minima that trap samplers. CD-trained couplings learn the *empirical*
    distribution of satisfying assignments, potentially smoothing the landscape
    and making sampling easier — especially as problem size grows and the
    hand-coded landscape becomes rougher.

**What's different from Exp 50:**
    1. Training data comes from the parallel sampler (hand-coded Ising +
       annealing), not brute-force random search — the only way to get
       satisfying assignments at 100+ vars in reasonable time.
    2. L1 regularization on couplings to prevent overfitting (10K+ params
       from only 5K samples).
    3. Tests three problem sizes (50, 100, 200) to show scaling trends.
    4. Compares three methods: CD-trained, hand-coded, and random.

**Training data generation strategy:**
    For large SAT instances, random search finds almost no satisfying
    assignments. Instead we use the hand-coded Ising mapping + parallel
    annealing sampler to generate near-satisfying assignments as training
    data. This is bootstrapping: we use the hand-coded model to generate
    data, then train a CD model that may sample *better* than the
    hand-coded model it was bootstrapped from.

Usage:
    .venv/bin/python scripts/experiment_60_scale_cd_training.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from experiment_39_thrml_sat import random_3sat, sat_to_ising, check_assignment
from carnot.samplers.parallel_ising import (
    ParallelIsingSampler,
    AnnealingSchedule,
    sat_to_coupling_matrix,
)


def generate_training_data(
    clauses: list[list[int]],
    n_vars: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate near-satisfying assignments using the hand-coded Ising + parallel sampler.

    For small instances (n_vars <= 20), brute-force random search can find
    satisfying assignments. For larger instances, we rely on the parallel
    Ising sampler with annealing to produce high-quality (near-satisfying)
    assignments as training data.

    Args:
        clauses: SAT clauses from random_3sat.
        n_vars: Number of variables.
        n_samples: Number of training samples to collect.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_collected, n_vars), float32 in {0, 1}.
        May return fewer than n_samples if filtering is strict.
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    # Build the hand-coded Ising model for this SAT instance.
    biases_vec, weights_vec, _ = sat_to_ising(clauses, n_vars)
    biases, J = sat_to_coupling_matrix(biases_vec, weights_vec, n_vars)

    # Use aggressive annealing to find low-energy (near-satisfying) states.
    # We sample more than needed and filter for quality.
    oversample_factor = 4
    sampler = ParallelIsingSampler(
        n_warmup=min(2000, n_vars * 20),
        n_samples=n_samples * oversample_factor,
        steps_per_sample=max(20, n_vars // 5),
        schedule=AnnealingSchedule(beta_init=0.1, beta_final=10.0),
        use_checkerboard=True,
    )

    key = jrandom.PRNGKey(seed)
    samples = sampler.sample(key, biases, J, beta=10.0)
    samples_np = np.asarray(samples)

    # Score each sample by clause satisfaction.
    n_clauses = len(clauses)
    scored = []
    for i in range(samples_np.shape[0]):
        assignment = {v + 1: bool(samples_np[i, v]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        scored.append((sat, i))

    # Keep the best assignments as training data.
    scored.sort(key=lambda x: -x[0])
    top_indices = [idx for _, idx in scored[:n_samples]]
    data = samples_np[top_indices].astype(np.float32)

    # Report quality of training data.
    top_sats = [s for s, _ in scored[:n_samples]]
    mean_pct = np.mean(top_sats) / n_clauses * 100
    best_pct = max(top_sats) / n_clauses * 100
    print(f"    Training data: {len(data)} samples, "
          f"mean {mean_pct:.1f}% sat, best {best_pct:.1f}% sat")

    return data


def train_ising_cd_l1(
    data: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.01,
    beta: float = 2.0,
    cd_steps: int = 1,
    l1_lambda: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train an Ising model via Contrastive Divergence with L1 regularization.

    Same CD training as Exp 50's train_ising_cd, but adds L1 penalty on
    couplings to prevent overfitting when parameter count (N^2) exceeds
    sample count. L1 encourages sparse couplings, which matches the
    structure of SAT problems (each clause only involves 3 variables).

    Args:
        data: Training data, shape (n_samples, n_vars), float {0,1}.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        beta: Inverse temperature for sampling.
        cd_steps: Number of Gibbs steps per CD negative phase.
        l1_lambda: L1 regularization strength on couplings.

    Returns:
        Tuple of (biases, coupling_matrix, losses).
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    n_samples, n_vars = data.shape

    # Initialize parameters — small random couplings, zero biases.
    rng = np.random.default_rng(42)
    biases = np.zeros(n_vars, dtype=np.float32)
    J = rng.normal(0, 0.01, (n_vars, n_vars)).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    # Positive phase statistics (from data) — computed once since data is fixed.
    data_jax = jnp.array(data)
    spins_data = 2.0 * data_jax - 1.0  # Map {0,1} -> {-1,+1}.
    pos_bias_moments = jnp.mean(spins_data, axis=0)
    pos_weight_moments = jnp.mean(
        jnp.einsum("bi,bj->bij", spins_data, spins_data), axis=0
    )

    # Configure the sampler for the negative phase.
    sampler = ParallelIsingSampler(
        n_warmup=cd_steps * 10,
        n_samples=n_samples,
        steps_per_sample=cd_steps,
        schedule=None,
        use_checkerboard=True,
    )

    losses = []
    for epoch in range(n_epochs):
        key = jrandom.PRNGKey(epoch)

        # Negative phase: sample from current model.
        b_jax = jnp.array(biases)
        J_jax = jnp.array(J)
        model_samples = sampler.sample(key, b_jax, J_jax, beta=beta)

        spins_model = 2.0 * model_samples.astype(jnp.float32) - 1.0
        neg_bias_moments = jnp.mean(spins_model, axis=0)
        neg_weight_moments = jnp.mean(
            jnp.einsum("bi,bj->bij", spins_model, spins_model), axis=0
        )

        # CD gradient.
        grad_b = -beta * (pos_bias_moments - neg_bias_moments)
        grad_J = -beta * (pos_weight_moments - neg_weight_moments)

        # L1 regularization gradient: d/dJ |J| = sign(J).
        l1_grad = l1_lambda * np.sign(J)

        # Update parameters.
        biases -= lr * np.array(grad_b)
        J -= lr * (np.array(grad_J) + l1_grad)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

        recon_error = float(jnp.mean((pos_bias_moments - neg_bias_moments) ** 2))
        losses.append(recon_error)

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            sparsity = np.mean(np.abs(J) < 0.01) * 100
            print(f"    Epoch {epoch:3d}: recon={recon_error:.6f}, "
                  f"sparsity={sparsity:.0f}%")

    return biases, J, losses


def evaluate_method(
    biases: np.ndarray,
    J: np.ndarray,
    clauses: list[list[int]],
    n_vars: int,
    beta: float = 10.0,
    n_eval_samples: int = 200,
    label: str = "",
) -> dict:
    """Evaluate an Ising model by sampling and checking SAT quality.

    Args:
        biases: Bias vector, shape (n_vars,).
        J: Coupling matrix, shape (n_vars, n_vars).
        clauses: SAT clauses to evaluate against.
        n_vars: Number of variables.
        beta: Inverse temperature for evaluation sampling.
        n_eval_samples: Number of samples to draw.
        label: Name for this method (for printing).

    Returns:
        Dict with best_pct, mean_pct, n_perfect, best_sat, n_clauses.
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    sampler = ParallelIsingSampler(
        n_warmup=min(2000, n_vars * 10),
        n_samples=n_eval_samples,
        steps_per_sample=max(20, n_vars // 5),
        schedule=AnnealingSchedule(beta_init=0.5, beta_final=beta),
        use_checkerboard=True,
    )

    samples = sampler.sample(
        jrandom.PRNGKey(12345),
        jnp.array(biases, dtype=jnp.float32),
        jnp.array(J, dtype=jnp.float32),
        beta=beta,
    )
    samples_np = np.asarray(samples)

    n_clauses = len(clauses)
    sat_counts = []
    for i in range(samples_np.shape[0]):
        assignment = {v + 1: bool(samples_np[i, v]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    best = max(sat_counts)
    mean_pct = np.mean(sat_counts) / n_clauses * 100
    best_pct = best / n_clauses * 100
    n_perfect = sum(1 for s in sat_counts if s == n_clauses)

    if label:
        print(f"    {label:12s}: mean={mean_pct:.1f}%, "
              f"best={best}/{n_clauses} ({best_pct:.1f}%), "
              f"perfect={n_perfect}/{n_eval_samples}")

    return {
        "best_sat": best,
        "n_clauses": n_clauses,
        "best_pct": best_pct,
        "mean_pct": mean_pct,
        "n_perfect": n_perfect,
    }


def random_baseline(clauses: list[list[int]], n_vars: int, n_tries: int = 500) -> dict:
    """Evaluate random assignment baseline."""
    rng = np.random.default_rng(42)
    n_clauses = len(clauses)
    sat_counts = []
    for _ in range(n_tries):
        assignment = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    best = max(sat_counts)
    mean_pct = np.mean(sat_counts) / n_clauses * 100
    best_pct = best / n_clauses * 100

    print(f"    {'Random':12s}: mean={mean_pct:.1f}%, "
          f"best={best}/{n_clauses} ({best_pct:.1f}%)")

    return {
        "best_sat": best,
        "n_clauses": n_clauses,
        "best_pct": best_pct,
        "mean_pct": mean_pct,
        "n_perfect": 0,
    }


def main() -> int:
    import jax

    print("=" * 70)
    print("EXPERIMENT 60: Scale CD Training to 100+ Variable SAT")
    print("  CD-trained Ising vs hand-coded vs random at 50/100/200 vars")
    print(f"  JAX backend: {jax.default_backend()}")
    print("=" * 70)

    start = time.time()

    # Test sizes: below phase transition (alpha ~ 3.5) so solutions exist.
    # At alpha=4.26 (phase transition), finding training data is hard.
    # We use alpha=3.5 to ensure enough satisfying assignments for CD training.
    test_cases = [
        (50,  int(50 * 3.5)),
        (100, int(100 * 3.5)),
        (200, int(200 * 3.5)),
    ]

    all_results = []

    for n_vars, n_clauses in test_cases:
        sep = "-" * 70
        print(f"\n{sep}")
        print(f"  {n_vars} vars, {n_clauses} clauses ({n_vars*n_vars} coupling params)")
        print(sep)

        # Generate a held-out SAT instance for evaluation (different seed).
        clauses_train = random_3sat(n_vars, n_clauses, seed=42 + n_vars)
        clauses_eval = random_3sat(n_vars, n_clauses, seed=999 + n_vars)

        # --- Step 1: Generate training data via hand-coded Ising sampler ---
        print(f"\n  [1/4] Generating training data ({5000} samples)...")
        t0 = time.time()
        data = generate_training_data(clauses_train, n_vars, n_samples=5000, seed=42)
        data_time = time.time() - t0
        print(f"    Data generation: {data_time:.1f}s")

        if data.shape[0] < 100:
            print(f"    WARNING: Only {data.shape[0]} training samples. Skipping.")
            continue

        # --- Step 2: Train CD model with L1 ---
        print(f"\n  [2/4] Training CD model (200 epochs, L1={0.001})...")
        t0 = time.time()
        # Scale learning rate down for larger problems to avoid divergence.
        lr = 0.05 / (n_vars / 50)
        biases_cd, J_cd, losses = train_ising_cd_l1(
            data, n_epochs=200, lr=lr, beta=2.0, cd_steps=1, l1_lambda=0.001,
        )
        train_time = time.time() - t0
        print(f"    Training: {train_time:.1f}s, final recon={losses[-1]:.6f}")

        # --- Step 3: Build hand-coded model for comparison ---
        print(f"\n  [3/4] Evaluating on training instance...")
        biases_hc_vec, weights_hc_vec, _ = sat_to_ising(clauses_train, n_vars)
        biases_hc, J_hc = sat_to_coupling_matrix(biases_hc_vec, weights_hc_vec, n_vars)
        biases_hc_np = np.array(biases_hc)
        J_hc_np = np.array(J_hc)

        res_cd_train = evaluate_method(
            biases_cd, J_cd, clauses_train, n_vars, label="CD-trained")
        res_hc_train = evaluate_method(
            biases_hc_np, J_hc_np, clauses_train, n_vars, label="Hand-coded")
        res_rand_train = random_baseline(clauses_train, n_vars)

        # --- Step 4: Evaluate on held-out instance ---
        print(f"\n  [4/4] Evaluating on held-out instance (generalization)...")
        biases_hc2_vec, weights_hc2_vec, _ = sat_to_ising(clauses_eval, n_vars)
        biases_hc2, J_hc2 = sat_to_coupling_matrix(
            biases_hc2_vec, weights_hc2_vec, n_vars)
        biases_hc2_np = np.array(biases_hc2)
        J_hc2_np = np.array(J_hc2)

        # CD model was trained on clauses_train — test on clauses_eval.
        res_cd_eval = evaluate_method(
            biases_cd, J_cd, clauses_eval, n_vars, label="CD-trained")
        # Hand-coded uses the eval instance's own Ising mapping (oracle).
        res_hc_eval = evaluate_method(
            biases_hc2_np, J_hc2_np, clauses_eval, n_vars, label="Hand-coded*")
        res_rand_eval = random_baseline(clauses_eval, n_vars)

        all_results.append({
            "n_vars": n_vars,
            "n_clauses": n_clauses,
            "n_params": n_vars * n_vars,
            "data_time": data_time,
            "train_time": train_time,
            "final_loss": losses[-1],
            "train": {
                "cd": res_cd_train,
                "handcoded": res_hc_train,
                "random": res_rand_train,
            },
            "eval": {
                "cd": res_cd_eval,
                "handcoded": res_hc_eval,
                "random": res_rand_eval,
            },
        })

    # --- Summary ---
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 60 RESULTS ({elapsed:.0f}s)")
    print(sep)

    # Training instance results.
    print(f"\n  Training Instance (same SAT used for CD training):")
    print(f"  {'Vars':>6s} {'Params':>8s} {'CD mean':>9s} {'HC mean':>9s} "
          f"{'Rand mean':>10s} {'CD best':>9s} {'HC best':>9s}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['n_vars']:>6d} {r['n_params']:>8d} "
              f"{r['train']['cd']['mean_pct']:>8.1f}% "
              f"{r['train']['handcoded']['mean_pct']:>8.1f}% "
              f"{r['train']['random']['mean_pct']:>9.1f}% "
              f"{r['train']['cd']['best_pct']:>8.1f}% "
              f"{r['train']['handcoded']['best_pct']:>8.1f}%")

    # Held-out instance results.
    print(f"\n  Held-out Instance (generalization — CD never saw this SAT):")
    print(f"  {'Vars':>6s} {'CD mean':>9s} {'HC* mean':>10s} "
          f"{'Rand mean':>10s} {'CD best':>9s} {'HC* best':>10s}")
    print(f"  {'-'*56}")
    for r in all_results:
        print(f"  {r['n_vars']:>6d} "
              f"{r['eval']['cd']['mean_pct']:>8.1f}% "
              f"{r['eval']['handcoded']['mean_pct']:>9.1f}% "
              f"{r['eval']['random']['mean_pct']:>9.1f}% "
              f"{r['eval']['cd']['best_pct']:>8.1f}% "
              f"{r['eval']['handcoded']['best_pct']:>9.1f}%")
    print(f"  (* Hand-coded uses the eval instance's own Ising mapping — oracle baseline)")

    # Timing.
    print(f"\n  Timing:")
    for r in all_results:
        print(f"    {r['n_vars']} vars: data={r['data_time']:.1f}s, "
              f"train={r['train_time']:.1f}s")

    # Verdict: does CD improve over random, and how does it compare to hand-coded?
    if all_results:
        cd_advantages = []
        for r in all_results:
            cd_over_rand = (r["train"]["cd"]["mean_pct"]
                           - r["train"]["random"]["mean_pct"])
            cd_advantages.append(cd_over_rand)

        mean_cd_advantage = np.mean(cd_advantages)
        scaling_trend = cd_advantages[-1] - cd_advantages[0] if len(cd_advantages) > 1 else 0

        print(f"\n  CD advantage over random (training instance): "
              f"{[f'{a:+.1f}%' for a in cd_advantages]}")
        print(f"  Mean CD advantage: {mean_cd_advantage:+.1f}%")

        if mean_cd_advantage > 5:
            print(f"\n  VERDICT: ✅ CD training produces useful Ising couplings at scale")
        elif mean_cd_advantage > 1:
            print(f"\n  VERDICT: ⚠️ Modest improvement from CD training")
        else:
            print(f"\n  VERDICT: ❌ CD training does not significantly beat random at this scale")

        if scaling_trend > 2:
            print(f"  SCALING: ✅ CD advantage grows with problem size ({scaling_trend:+.1f}%)")
        elif scaling_trend < -2:
            print(f"  SCALING: ❌ CD advantage shrinks with problem size ({scaling_trend:+.1f}%)")
        else:
            print(f"  SCALING: ➡️ CD advantage roughly constant across sizes ({scaling_trend:+.1f}%)")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
