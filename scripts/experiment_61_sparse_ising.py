#!/usr/bin/env python3
"""Experiment 61: Sparse Ising CD — learn only clause-graph edges.

Full coupling for 500 vars is 250K parameters. This experiment exploits the
fact that SAT clauses only connect ~3 variables each, so the clause graph is
sparse (~5% density for 3-SAT at alpha=3.5). We mask the CD gradient to only
update couplings for edges that appear in at least one clause, reducing the
effective parameter count by ~20x while preserving the problem structure.

**Why sparse beats dense at scale:**
    Dense CD with 500 vars has 250K parameters but typically only 5K-10K
    training samples. This massive over-parameterization leads to overfitting:
    the model memorizes training data instead of learning the SAT structure.
    L1 regularization (Exp 60) helps but is a soft constraint — it pushes
    irrelevant couplings toward zero but doesn't eliminate them. Hard sparsity
    (masking) eliminates them exactly, which:
    1. Reduces effective parameters by ~20x (only clause-graph edges).
    2. Prevents the model from "hallucinating" correlations between unrelated
       variables.
    3. Makes training faster (fewer gradients to compute, though the mask
       application is O(N^2) regardless).
    4. Should improve generalization to unseen SAT instances of the same
       structure because the learned couplings correspond to actual clause
       connections.

**What this experiment tests:**
    1. Sparse CD: mask grad_J to only update clause-graph edges.
    2. Compare dense CD (Exp 60 style) vs sparse CD vs hand-coded Ising.
    3. Generalization: train on seed=42, evaluate on seed=99.
    4. Scaling: 200, 500, 1000 variables.

Usage:
    .venv/bin/python scripts/experiment_61_sparse_ising.py
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


def build_clause_edge_mask(
    clauses: list[list[int]],
    n_vars: int,
) -> np.ndarray:
    """Build a binary mask indicating which variable pairs share a clause.

    For each clause (l1, l2, l3), the three variable pairs (|l1|, |l2|),
    (|l1|, |l3|), (|l2|, |l3|) are marked as connected. The resulting mask
    is symmetric with zero diagonal.

    Args:
        clauses: SAT clauses from random_3sat. Each clause is a list of
            signed integers where |lit| is the variable index (1-based)
            and the sign indicates negation.
        n_vars: Number of variables.

    Returns:
        Boolean mask of shape (n_vars, n_vars). mask[i, j] = True means
        variables i and j appear together in at least one clause. The
        density of this mask is typically ~5% for random 3-SAT at
        alpha=3.5, because each clause adds at most 3 edges out of
        N*(N-1)/2 possible.
    """
    mask = np.zeros((n_vars, n_vars), dtype=np.float32)
    for clause in clauses:
        # Extract 0-based variable indices from the clause.
        var_indices = [abs(lit) - 1 for lit in clause]
        # Mark all pairs in this clause as connected.
        for a in range(len(var_indices)):
            for b in range(a + 1, len(var_indices)):
                vi, vj = var_indices[a], var_indices[b]
                if vi != vj:  # Guard against degenerate clauses.
                    mask[vi, vj] = 1.0
                    mask[vj, vi] = 1.0
    return mask


def compute_mask_density(mask: np.ndarray) -> float:
    """Compute the fraction of non-zero off-diagonal entries in the mask.

    Returns a percentage (0-100).
    """
    n = mask.shape[0]
    off_diag_count = n * (n - 1)
    nonzero_count = np.count_nonzero(mask)
    return nonzero_count / off_diag_count * 100 if off_diag_count > 0 else 0.0


def generate_training_data(
    clauses: list[list[int]],
    n_vars: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate near-satisfying assignments using the hand-coded Ising + parallel sampler.

    Same strategy as Exp 60: bootstrap training data from the hand-coded
    Ising model via parallel annealing, then filter for the best-quality
    assignments. This is the only feasible way to get satisfying assignments
    at 500+ variables where random search finds essentially nothing.

    Args:
        clauses: SAT clauses.
        n_vars: Number of variables.
        n_samples: Desired number of training samples.
        seed: Random seed.

    Returns:
        Array of shape (n_collected, n_vars), float32 in {0, 1}.
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    biases_vec, weights_vec, _ = sat_to_ising(clauses, n_vars)
    biases, J = sat_to_coupling_matrix(biases_vec, weights_vec, n_vars)

    # Keep oversample modest to avoid OOM on CPU. For large problems,
    # warmup and sample counts are capped so JAX scan compilation
    # stays within memory limits (~16GB).
    oversample_factor = 2
    sampler = ParallelIsingSampler(
        n_warmup=min(500, n_vars * 5),
        n_samples=min(n_samples * oversample_factor, 4000),
        steps_per_sample=max(10, min(n_vars // 10, 50)),
        schedule=AnnealingSchedule(beta_init=0.1, beta_final=10.0),
        use_checkerboard=True,
    )

    key = jrandom.PRNGKey(seed)
    samples = sampler.sample(key, biases, J, beta=10.0)
    samples_np = np.asarray(samples)

    n_clauses = len(clauses)
    scored = []
    for i in range(samples_np.shape[0]):
        assignment = {v + 1: bool(samples_np[i, v]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        scored.append((sat, i))

    scored.sort(key=lambda x: -x[0])
    top_indices = [idx for _, idx in scored[:n_samples]]
    data = samples_np[top_indices].astype(np.float32)

    top_sats = [s for s, _ in scored[:n_samples]]
    mean_pct = np.mean(top_sats) / n_clauses * 100
    best_pct = max(top_sats) / n_clauses * 100
    print(f"    Training data: {len(data)} samples, "
          f"mean {mean_pct:.1f}% sat, best {best_pct:.1f}% sat")

    return data


def train_ising_cd_sparse(
    data: np.ndarray,
    edge_mask: np.ndarray | None = None,
    n_epochs: int = 200,
    lr: float = 0.01,
    beta: float = 2.0,
    cd_steps: int = 1,
    l1_lambda: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train an Ising model via CD with optional sparsity mask on couplings.

    When edge_mask is provided, the gradient for J is element-wise multiplied
    by the mask BEFORE the parameter update. This means couplings for non-edge
    variable pairs are never updated from their initial value (zero), so the
    learned model only has couplings where SAT clauses connect variables.

    The mask does NOT affect the L1 regularization gradient — L1 only acts on
    non-zero couplings anyway, and masked entries stay at zero.

    Args:
        data: Training data, shape (n_samples, n_vars), float {0,1}.
        edge_mask: Binary mask, shape (n_vars, n_vars). If None, train dense
            (equivalent to Exp 60's train_ising_cd_l1).
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

    # Initialize: small random couplings, zero biases.
    rng = np.random.default_rng(42)
    biases = np.zeros(n_vars, dtype=np.float32)
    J = rng.normal(0, 0.01, (n_vars, n_vars)).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    # If sparse, zero out non-edge couplings from the start so
    # the model begins from the correct sparsity pattern.
    if edge_mask is not None:
        J *= edge_mask

    # Positive phase statistics (computed once from data).
    data_jax = jnp.array(data)
    spins_data = 2.0 * data_jax - 1.0
    pos_bias_moments = jnp.mean(spins_data, axis=0)
    pos_weight_moments = jnp.mean(
        jnp.einsum("bi,bj->bij", spins_data, spins_data), axis=0
    )

    # Cap the negative-phase sampler to avoid OOM. We need n_samples
    # model samples per epoch — but for large models, use fewer samples
    # and compensate with more epochs.
    neg_samples = min(n_samples, 1000)
    sampler = ParallelIsingSampler(
        n_warmup=cd_steps * 10,
        n_samples=neg_samples,
        steps_per_sample=cd_steps,
        schedule=None,
        use_checkerboard=True,
    )

    losses = []
    for epoch in range(n_epochs):
        key = jrandom.PRNGKey(epoch)

        b_jax = jnp.array(biases)
        J_jax = jnp.array(J)
        model_samples = sampler.sample(key, b_jax, J_jax, beta=beta)

        spins_model = 2.0 * model_samples.astype(jnp.float32) - 1.0
        neg_bias_moments = jnp.mean(spins_model, axis=0)
        neg_weight_moments = jnp.mean(
            jnp.einsum("bi,bj->bij", spins_model, spins_model), axis=0
        )

        # CD gradient.
        grad_J = -beta * (pos_weight_moments - neg_weight_moments)

        # Apply sparsity mask to gradient BEFORE update. This is the key
        # difference from Exp 60: non-edge couplings receive zero gradient
        # and thus stay at zero forever.
        if edge_mask is not None:
            grad_J = np.array(grad_J) * edge_mask
        else:
            grad_J = np.array(grad_J)

        grad_b = -beta * (pos_bias_moments - neg_bias_moments)

        # L1 regularization on couplings (only affects non-zero entries).
        l1_grad = l1_lambda * np.sign(J)

        # Update parameters.
        biases -= lr * np.array(grad_b)
        J -= lr * (grad_J + l1_grad)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

        # Re-enforce mask after symmetrization (floating point can leak).
        if edge_mask is not None:
            J *= edge_mask

        recon_error = float(jnp.mean((pos_bias_moments - neg_bias_moments) ** 2))
        losses.append(recon_error)

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            sparsity = np.mean(np.abs(J) < 0.01) * 100
            n_nonzero = np.count_nonzero(np.abs(J) >= 0.01)
            print(f"    Epoch {epoch:3d}: recon={recon_error:.6f}, "
                  f"sparsity={sparsity:.0f}%, nonzero={n_nonzero}")

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
    """Evaluate an Ising model by sampling and checking SAT satisfaction.

    Draws samples from the model at the given inverse temperature and
    checks what fraction of clauses each sample satisfies. Reports mean
    and best satisfaction percentages.

    Args:
        biases: Bias vector, shape (n_vars,).
        J: Coupling matrix, shape (n_vars, n_vars).
        clauses: SAT clauses to evaluate against.
        n_vars: Number of variables.
        beta: Inverse temperature for evaluation sampling.
        n_eval_samples: Number of samples to draw.
        label: Display name for this method.

    Returns:
        Dict with best_pct, mean_pct, n_perfect, best_sat, n_clauses.
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    sampler = ParallelIsingSampler(
        n_warmup=min(500, n_vars * 5),
        n_samples=min(n_eval_samples, 200),
        steps_per_sample=max(10, min(n_vars // 10, 50)),
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
        print(f"    {label:15s}: mean={mean_pct:.1f}%, "
              f"best={best}/{n_clauses} ({best_pct:.1f}%), "
              f"perfect={n_perfect}/{n_eval_samples}")

    return {
        "best_sat": best,
        "n_clauses": n_clauses,
        "best_pct": best_pct,
        "mean_pct": mean_pct,
        "n_perfect": n_perfect,
    }


def random_baseline(
    clauses: list[list[int]],
    n_vars: int,
    n_tries: int = 500,
) -> dict:
    """Evaluate random assignment baseline.

    For large SAT instances, random assignments satisfy ~7/8 of clauses
    (each 3-SAT clause is violated by exactly 1 of 8 possible assignments
    to its 3 variables). This gives an expected ~87.5% baseline.
    """
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

    print(f"    {'Random':15s}: mean={mean_pct:.1f}%, "
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
    print("EXPERIMENT 61: Sparse Ising CD — Clause-Graph Edge Masking")
    print("  Only learn couplings for edges in the SAT clause graph (~5% density)")
    print("  Compare: sparse CD vs dense CD vs hand-coded vs random")
    print(f"  JAX backend: {jax.default_backend()}")
    print("=" * 70)

    start = time.time()

    # Problem sizes: 200, 500, 1000 variables.
    # alpha=3.5 (below phase transition) so satisfying assignments exist.
    # Full coupling params: N^2. Sparse params: ~3*n_clauses (each clause
    # adds up to 3 edges).
    test_cases = [
        (200,  int(200 * 3.5)),
        (500,  int(500 * 3.5)),
        (1000, int(1000 * 3.5)),
    ]

    all_results = []

    for n_vars, n_clauses in test_cases:
        sep = "-" * 70
        print(f"\n{sep}")
        n_dense_params = n_vars * (n_vars - 1) // 2
        print(f"  {n_vars} vars, {n_clauses} clauses "
              f"(dense: {n_dense_params} params)")
        print(sep)

        # Generate SAT instances: train on seed=42, test on seed=99.
        clauses_train = random_3sat(n_vars, n_clauses, seed=42)
        clauses_test = random_3sat(n_vars, n_clauses, seed=99)

        # --- Build sparsity mask from the training clause graph ---
        mask = build_clause_edge_mask(clauses_train, n_vars)
        density = compute_mask_density(mask)
        n_sparse_params = int(np.count_nonzero(mask)) // 2  # Symmetric.
        print(f"  Clause-graph density: {density:.1f}%, "
              f"sparse params: {n_sparse_params} "
              f"(vs dense: {n_dense_params}, "
              f"ratio: {n_sparse_params / n_dense_params * 100:.1f}%)")

        # --- Step 1: Generate training data ---
        # Scale samples with problem size but cap to keep runtime reasonable
        # on CPU. JAX scan compilation is O(n_samples * n_vars^2) in memory.
        n_train_samples = min(2000, max(500, n_vars * 2))
        print(f"\n  [1/5] Generating training data ({n_train_samples} samples)...")
        t0 = time.time()
        data = generate_training_data(
            clauses_train, n_vars, n_samples=n_train_samples, seed=42
        )
        data_time = time.time() - t0
        print(f"    Data generation: {data_time:.1f}s")

        if data.shape[0] < 100:
            print(f"    WARNING: Only {data.shape[0]} training samples. Skipping.")
            continue

        # Scale learning rate and epochs for problem size.
        lr = 0.05 / (n_vars / 50)
        # Fewer epochs for larger problems to keep runtime manageable.
        n_epochs = max(100, 200 - n_vars // 10)

        # --- Step 2: Train SPARSE CD model ---
        print(f"\n  [2/5] Training SPARSE CD model "
              f"({n_epochs} epochs, {n_sparse_params} params)...")
        t0 = time.time()
        biases_sparse, J_sparse, losses_sparse = train_ising_cd_sparse(
            data, edge_mask=mask, n_epochs=n_epochs, lr=lr,
            beta=2.0, cd_steps=1, l1_lambda=0.001,
        )
        sparse_time = time.time() - t0
        print(f"    Sparse training: {sparse_time:.1f}s, "
              f"final recon={losses_sparse[-1]:.6f}")

        # --- Step 3: Train DENSE CD model (Exp 60 style, for comparison) ---
        # For 1000 vars, dense training is very slow. Use fewer epochs.
        dense_epochs = max(50, n_epochs // 2)
        print(f"\n  [3/5] Training DENSE CD model "
              f"({dense_epochs} epochs, {n_dense_params} params)...")
        t0 = time.time()
        biases_dense, J_dense, losses_dense = train_ising_cd_sparse(
            data, edge_mask=None, n_epochs=dense_epochs, lr=lr,
            beta=2.0, cd_steps=1, l1_lambda=0.001,
        )
        dense_time = time.time() - t0
        print(f"    Dense training: {dense_time:.1f}s, "
              f"final recon={losses_dense[-1]:.6f}")

        # --- Step 4: Build hand-coded model ---
        biases_hc_vec, weights_hc_vec, _ = sat_to_ising(clauses_train, n_vars)
        biases_hc, J_hc = sat_to_coupling_matrix(biases_hc_vec, weights_hc_vec, n_vars)
        biases_hc_np = np.array(biases_hc)
        J_hc_np = np.array(J_hc)

        # --- Step 4a: Evaluate on TRAINING instance ---
        print(f"\n  [4/5] Evaluating on training instance (seed=42)...")
        res_sparse_train = evaluate_method(
            biases_sparse, J_sparse, clauses_train, n_vars, label="Sparse CD")
        res_dense_train = evaluate_method(
            biases_dense, J_dense, clauses_train, n_vars, label="Dense CD")
        res_hc_train = evaluate_method(
            biases_hc_np, J_hc_np, clauses_train, n_vars, label="Hand-coded")
        res_rand_train = random_baseline(clauses_train, n_vars)

        # --- Step 5: Evaluate on HELD-OUT instance (seed=99) ---
        print(f"\n  [5/5] Evaluating on held-out instance (seed=99, generalization)...")
        # Hand-coded uses the test instance's own mapping (oracle).
        biases_hc2_vec, weights_hc2_vec, _ = sat_to_ising(clauses_test, n_vars)
        biases_hc2, J_hc2 = sat_to_coupling_matrix(
            biases_hc2_vec, weights_hc2_vec, n_vars
        )
        biases_hc2_np = np.array(biases_hc2)
        J_hc2_np = np.array(J_hc2)

        res_sparse_test = evaluate_method(
            biases_sparse, J_sparse, clauses_test, n_vars, label="Sparse CD")
        res_dense_test = evaluate_method(
            biases_dense, J_dense, clauses_test, n_vars, label="Dense CD")
        res_hc_test = evaluate_method(
            biases_hc2_np, J_hc2_np, clauses_test, n_vars, label="Hand-coded*")
        res_rand_test = random_baseline(clauses_test, n_vars)

        all_results.append({
            "n_vars": n_vars,
            "n_clauses": n_clauses,
            "n_dense_params": n_dense_params,
            "n_sparse_params": n_sparse_params,
            "mask_density_pct": density,
            "data_time": data_time,
            "sparse_time": sparse_time,
            "dense_time": dense_time,
            "sparse_final_loss": losses_sparse[-1],
            "dense_final_loss": losses_dense[-1],
            "train": {
                "sparse": res_sparse_train,
                "dense": res_dense_train,
                "handcoded": res_hc_train,
                "random": res_rand_train,
            },
            "test": {
                "sparse": res_sparse_test,
                "dense": res_dense_test,
                "handcoded": res_hc_test,
                "random": res_rand_test,
            },
        })

    # --- Summary ---
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 61 RESULTS ({elapsed:.0f}s)")
    print(sep)

    # Sparsity stats.
    print(f"\n  Sparsity Analysis:")
    print(f"  {'Vars':>6s} {'Dense':>8s} {'Sparse':>8s} {'Density':>8s} {'Ratio':>8s}")
    print(f"  {'-'*42}")
    for r in all_results:
        print(f"  {r['n_vars']:>6d} {r['n_dense_params']:>8d} "
              f"{r['n_sparse_params']:>8d} {r['mask_density_pct']:>7.1f}% "
              f"{r['n_sparse_params'] / r['n_dense_params'] * 100:>7.1f}%")

    # Training instance results.
    print(f"\n  Training Instance (seed=42):")
    print(f"  {'Vars':>6s} {'Sparse':>9s} {'Dense':>9s} {'HC':>9s} {'Random':>9s}")
    print(f"  {'-'*45}")
    for r in all_results:
        print(f"  {r['n_vars']:>6d} "
              f"{r['train']['sparse']['mean_pct']:>8.1f}% "
              f"{r['train']['dense']['mean_pct']:>8.1f}% "
              f"{r['train']['handcoded']['mean_pct']:>8.1f}% "
              f"{r['train']['random']['mean_pct']:>8.1f}%")

    # Held-out instance results.
    print(f"\n  Held-out Instance (seed=99, generalization):")
    print(f"  {'Vars':>6s} {'Sparse':>9s} {'Dense':>9s} {'HC*':>9s} {'Random':>9s}")
    print(f"  {'-'*45}")
    for r in all_results:
        print(f"  {r['n_vars']:>6d} "
              f"{r['test']['sparse']['mean_pct']:>8.1f}% "
              f"{r['test']['dense']['mean_pct']:>8.1f}% "
              f"{r['test']['handcoded']['mean_pct']:>8.1f}% "
              f"{r['test']['random']['mean_pct']:>8.1f}%")
    print(f"  (* Hand-coded uses the test instance's own Ising mapping — oracle)")

    # Best satisfaction (what matters for SAT solving).
    print(f"\n  Best Satisfaction (training instance):")
    print(f"  {'Vars':>6s} {'Sparse':>9s} {'Dense':>9s} {'HC':>9s}")
    print(f"  {'-'*36}")
    for r in all_results:
        print(f"  {r['n_vars']:>6d} "
              f"{r['train']['sparse']['best_pct']:>8.1f}% "
              f"{r['train']['dense']['best_pct']:>8.1f}% "
              f"{r['train']['handcoded']['best_pct']:>8.1f}%")

    # Timing comparison.
    print(f"\n  Timing:")
    for r in all_results:
        speedup = r["dense_time"] / r["sparse_time"] if r["sparse_time"] > 0 else 0
        print(f"    {r['n_vars']} vars: data={r['data_time']:.1f}s, "
              f"sparse={r['sparse_time']:.1f}s, dense={r['dense_time']:.1f}s "
              f"(sparse {speedup:.1f}x faster)")

    # Verdict.
    if all_results:
        sparse_advantages = []
        for r in all_results:
            sparse_over_dense = (r["train"]["sparse"]["mean_pct"]
                                 - r["train"]["dense"]["mean_pct"])
            sparse_advantages.append(sparse_over_dense)

        sparse_over_rand = []
        for r in all_results:
            adv = (r["train"]["sparse"]["mean_pct"]
                   - r["train"]["random"]["mean_pct"])
            sparse_over_rand.append(adv)

        gen_gap_sparse = []
        gen_gap_dense = []
        for r in all_results:
            gen_gap_sparse.append(
                r["train"]["sparse"]["mean_pct"] - r["test"]["sparse"]["mean_pct"]
            )
            gen_gap_dense.append(
                r["train"]["dense"]["mean_pct"] - r["test"]["dense"]["mean_pct"]
            )

        mean_sparse_advantage = np.mean(sparse_advantages)
        mean_sparse_over_rand = np.mean(sparse_over_rand)
        mean_gen_gap_sparse = np.mean(gen_gap_sparse)
        mean_gen_gap_dense = np.mean(gen_gap_dense)

        print(f"\n  Sparse CD vs Dense CD (train): "
              f"{[f'{a:+.1f}%' for a in sparse_advantages]}")
        print(f"  Sparse CD over Random (train): "
              f"{[f'{a:+.1f}%' for a in sparse_over_rand]}")
        print(f"  Generalization gap (train - test): "
              f"sparse={[f'{g:.1f}%' for g in gen_gap_sparse]}, "
              f"dense={[f'{g:.1f}%' for g in gen_gap_dense]}")

        # Sparsity verdict.
        if mean_sparse_advantage > 2:
            print(f"\n  SPARSITY: ✅ Sparse CD outperforms dense CD "
                  f"by {mean_sparse_advantage:+.1f}% mean")
        elif mean_sparse_advantage > -1:
            print(f"\n  SPARSITY: ➡️ Sparse CD matches dense CD "
                  f"({mean_sparse_advantage:+.1f}%) with far fewer params")
        else:
            print(f"\n  SPARSITY: ❌ Sparse CD underperforms dense CD "
                  f"by {mean_sparse_advantage:+.1f}%")

        # Generalization verdict.
        if mean_gen_gap_sparse < mean_gen_gap_dense:
            print(f"  GENERALIZATION: ✅ Sparse generalizes better "
                  f"(gap: {mean_gen_gap_sparse:.1f}% vs dense {mean_gen_gap_dense:.1f}%)")
        else:
            print(f"  GENERALIZATION: ➡️ Similar generalization "
                  f"(sparse gap: {mean_gen_gap_sparse:.1f}%, "
                  f"dense gap: {mean_gen_gap_dense:.1f}%)")

        # Overall usefulness.
        if mean_sparse_over_rand > 5:
            print(f"  VERDICT: ✅ Sparse CD produces useful Ising couplings at scale")
        elif mean_sparse_over_rand > 1:
            print(f"  VERDICT: ⚠️ Modest improvement from sparse CD")
        else:
            print(f"  VERDICT: ❌ Sparse CD does not significantly beat random")

        # Scaling trend.
        if len(sparse_advantages) >= 2:
            trend = sparse_advantages[-1] - sparse_advantages[0]
            if trend > 2:
                print(f"  SCALING: ✅ Sparse advantage grows with size ({trend:+.1f}%)")
            elif trend < -2:
                print(f"  SCALING: ❌ Sparse advantage shrinks ({trend:+.1f}%)")
            else:
                print(f"  SCALING: ➡️ Sparse advantage stable ({trend:+.1f}%)")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
