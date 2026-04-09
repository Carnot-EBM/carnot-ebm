#!/usr/bin/env python3
"""Experiment 50: Learn Ising couplings from satisfying assignments.

Train an Ising model using Contrastive Divergence to reproduce a distribution
of satisfying assignments for a SAT problem. The learned couplings should
assign low energy to satisfying assignments and high energy to violating ones.

This implements the same gradient estimator as thrml's estimate_kl_grad but
uses the parallel sampler for the negative phase (model sampling), making it
compatible with ROCm and 100x+ faster.

The training loop:
  1. Positive phase: compute statistics from data (known satisfying assignments)
  2. Negative phase: sample from the model using parallel Gibbs
  3. Update: ΔJ = -β(⟨s_i s_j⟩_data - ⟨s_i s_j⟩_model)
             Δb = -β(⟨s_i⟩_data - ⟨s_i⟩_model)

After training, the model should:
  - Assign low energy to satisfying assignments
  - Generate new satisfying assignments via sampling
  - Generalize to unseen SAT instances of the same structure

Usage:
    .venv/bin/python scripts/experiment_50_learn_ising.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def generate_sat_data(n_vars: int, n_clauses: int, n_samples: int, seed: int = 42) -> tuple:
    """Generate satisfying assignments for a random 3-SAT instance via brute force / heuristic.

    Returns (clauses, data) where data is shape (n_samples, n_vars) boolean.
    """
    from experiment_39_thrml_sat import random_3sat, check_assignment

    clauses = random_3sat(n_vars, n_clauses, seed=seed)

    # Generate satisfying assignments via random search.
    rng = np.random.default_rng(seed)
    data = []
    attempts = 0
    while len(data) < n_samples and attempts < n_samples * 1000:
        assignment = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
        sat, total = check_assignment(clauses, assignment)
        if sat == total:
            data.append([assignment[i + 1] for i in range(n_vars)])
        attempts += 1

    if len(data) < n_samples:
        # For hard instances, accept near-satisfying assignments.
        while len(data) < n_samples:
            assignment = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
            sat, total = check_assignment(clauses, assignment)
            if sat >= total - 1:
                data.append([assignment[i + 1] for i in range(n_vars)])
            attempts += 1
            if attempts > n_samples * 10000:
                break

    return clauses, np.array(data[:n_samples], dtype=np.float32)


def train_ising_cd(
    data: np.ndarray,
    n_epochs: int = 100,
    lr: float = 0.01,
    beta: float = 1.0,
    cd_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Train an Ising model via Contrastive Divergence.

    Args:
        data: Training data, shape (n_samples, n_vars), float {0,1}.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        beta: Inverse temperature.
        cd_steps: Number of Gibbs steps for the negative phase (CD-k).

    Returns:
        Tuple of (biases, coupling_matrix).
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.parallel_ising import ParallelIsingSampler, AnnealingSchedule

    n_samples, n_vars = data.shape

    # Initialize parameters.
    rng = np.random.default_rng(42)
    biases = np.zeros(n_vars, dtype=np.float32)
    J = rng.normal(0, 0.01, (n_vars, n_vars)).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    # Positive phase statistics (from data) — compute once.
    data_jax = jnp.array(data)
    # Convert {0,1} to {-1,+1} for moment computation.
    spins_data = 2.0 * data_jax - 1.0
    pos_bias_moments = jnp.mean(spins_data, axis=0)  # ⟨σ_i⟩_data
    pos_weight_moments = jnp.mean(
        jnp.einsum("bi,bj->bij", spins_data, spins_data), axis=0
    )  # ⟨σ_i σ_j⟩_data

    # Training loop.
    sampler = ParallelIsingSampler(
        n_warmup=cd_steps * 10,
        n_samples=n_samples,
        steps_per_sample=cd_steps,
        schedule=None,  # Constant temperature.
        use_checkerboard=True,
    )

    losses = []
    for epoch in range(n_epochs):
        key = jrandom.PRNGKey(epoch)

        # Negative phase: sample from the model.
        b_jax = jnp.array(biases)
        J_jax = jnp.array(J)
        model_samples = sampler.sample(key, b_jax, J_jax, beta=beta)

        # Convert to {-1,+1}.
        spins_model = 2.0 * model_samples.astype(jnp.float32) - 1.0
        neg_bias_moments = jnp.mean(spins_model, axis=0)
        neg_weight_moments = jnp.mean(
            jnp.einsum("bi,bj->bij", spins_model, spins_model), axis=0
        )

        # Gradient update (CD rule).
        grad_b = -beta * (pos_bias_moments - neg_bias_moments)
        grad_J = -beta * (pos_weight_moments - neg_weight_moments)

        # Update parameters.
        biases -= lr * np.array(grad_b)
        J -= lr * np.array(grad_J)
        J = (J + J.T) / 2.0  # Keep symmetric.
        np.fill_diagonal(J, 0.0)

        # Compute reconstruction error (how well model samples match data).
        recon_error = float(jnp.mean((pos_bias_moments - neg_bias_moments) ** 2))
        losses.append(recon_error)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch:3d}: recon_error={recon_error:.6f}")

    return biases, J, losses


def evaluate_model(
    biases: np.ndarray,
    J: np.ndarray,
    clauses: list,
    n_vars: int,
    beta: float = 1.0,
) -> dict:
    """Evaluate a trained Ising model: generate samples and check SAT quality."""
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.parallel_ising import ParallelIsingSampler
    from experiment_39_thrml_sat import check_assignment

    sampler = ParallelIsingSampler(
        n_warmup=500, n_samples=100, steps_per_sample=20,
        use_checkerboard=True,
    )

    samples = sampler.sample(
        jrandom.PRNGKey(999),
        jnp.array(biases), jnp.array(J), beta=beta,
    )

    sat_counts = []
    for s_idx in range(samples.shape[0]):
        assignment = {v + 1: bool(samples[s_idx, v]) for v in range(n_vars)}
        sat, total = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    n_clauses = len(clauses)
    best = max(sat_counts)
    mean_pct = np.mean(sat_counts) / n_clauses * 100
    best_pct = best / n_clauses * 100
    n_perfect = sum(1 for s in sat_counts if s == n_clauses)

    return {
        "best_sat": best,
        "n_clauses": n_clauses,
        "best_pct": best_pct,
        "mean_pct": mean_pct,
        "n_perfect": n_perfect,
        "n_samples": len(sat_counts),
    }


def main() -> int:
    import jax
    print("=" * 70)
    print("EXPERIMENT 50: Learn Ising Couplings from SAT Data")
    print("  Contrastive Divergence training on satisfying assignments")
    print(f"  JAX backend: {jax.default_backend()}")
    print("=" * 70)

    start = time.time()

    # Use a small SAT instance where we can find many satisfying assignments.
    n_vars = 10
    n_clauses = 20  # Below phase transition → many solutions.
    n_data = 200

    print(f"\n--- Generating training data: {n_vars} vars, {n_clauses} clauses ---")
    clauses, data = generate_sat_data(n_vars, n_clauses, n_data, seed=42)
    print(f"  Found {data.shape[0]} satisfying assignments")

    if data.shape[0] < 10:
        print("  Not enough data to train. Exiting.")
        return 1

    # Random baseline: sample and check.
    from experiment_39_thrml_sat import check_assignment
    rng = np.random.default_rng(42)
    rand_sats = []
    for _ in range(100):
        ra = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
        sat, _ = check_assignment(clauses, ra)
        rand_sats.append(sat)
    print(f"  Random baseline: {np.mean(rand_sats)/n_clauses*100:.1f}% mean, {max(rand_sats)}/{n_clauses} best")

    # Train the Ising model.
    print(f"\n--- Training Ising model (CD-1, 100 epochs) ---")
    biases, J, losses = train_ising_cd(data, n_epochs=100, lr=0.05, beta=2.0, cd_steps=1)

    # Evaluate: generate new assignments from the trained model.
    print(f"\n--- Evaluating trained model ---")
    result = evaluate_model(biases, J, clauses, n_vars, beta=2.0)

    print(f"  Generated samples: {result['n_samples']}")
    print(f"  Best SAT: {result['best_sat']}/{result['n_clauses']} ({result['best_pct']:.1f}%)")
    print(f"  Mean SAT: {result['mean_pct']:.1f}%")
    print(f"  Perfect solutions: {result['n_perfect']}/{result['n_samples']}")

    # Also test on a DIFFERENT random instance (generalization).
    print(f"\n--- Generalization test (new SAT instance, same structure) ---")
    from experiment_39_thrml_sat import random_3sat
    clauses_new = random_3sat(n_vars, n_clauses, seed=99)
    result_gen = evaluate_model(biases, J, clauses_new, n_vars, beta=2.0)
    print(f"  Best SAT: {result_gen['best_sat']}/{result_gen['n_clauses']} ({result_gen['best_pct']:.1f}%)")
    print(f"  Mean SAT: {result_gen['mean_pct']:.1f}%")

    # Summary.
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 50 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"  Training data: {data.shape[0]} satisfying assignments")
    print(f"  Final recon error: {losses[-1]:.6f}")
    print(f"  Trained model SAT quality: {result['mean_pct']:.1f}% mean, {result['best_pct']:.1f}% best")
    print(f"  Random baseline:           {np.mean(rand_sats)/n_clauses*100:.1f}% mean")
    print(f"  Generalization:            {result_gen['mean_pct']:.1f}% mean")

    improvement = result["mean_pct"] - np.mean(rand_sats) / n_clauses * 100
    if result["n_perfect"] > 0 and improvement > 5:
        print(f"\n  VERDICT: ✅ Learned Ising generates satisfying assignments!")
    elif improvement > 2:
        print(f"\n  VERDICT: ⚠️ Modest improvement (+{improvement:.1f}%) over random")
    else:
        print(f"\n  VERDICT: ❌ No significant improvement over random")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
