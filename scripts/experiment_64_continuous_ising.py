#!/usr/bin/env python3
"""Experiment 64: Continuous Ising Relaxation — Bridge discrete spins to continuous optimization.

**Researcher summary:**
    Relaxes binary Ising spins s ∈ {0,1}^n to continuous s ∈ [0,1]^n and uses
    gradient descent (via jax.grad) to minimize the Ising energy. Final continuous
    solution is rounded to binary. This bridges discrete EBM sampling (thrml/TSU)
    with continuous latent-space reasoning (Kona-style).

**Detailed explanation for engineers:**
    All previous Carnot Ising experiments use discrete sampling: each spin is 0 or 1,
    and we use Gibbs sampling (sequential or parallel) to find low-energy states.
    This works, but sampling is inherently stochastic and scales poorly for large
    problems where the energy landscape has many local minima.

    Continuous relaxation is a classic technique from combinatorial optimization:
    1. Replace discrete variables s_i ∈ {0,1} with continuous s_i ∈ [0,1]
    2. The Ising energy E(s) = -(b^T s + s^T J s) is now a smooth function
    3. Use gradient descent to find the minimum: s ← s - lr * ∇E(s)
    4. Clip to [0,1] after each step (projected gradient descent)
    5. Round final solution to binary (threshold 0.5)

    The challenge: rounding can destroy optimality. A continuous solution of s_i = 0.5
    contributes half-weight to interactions, but rounding to 0 or 1 changes the energy
    discontinuously. Three strategies address this:

    (a) **Soft rounding (sigmoid annealing):** Replace s with sigmoid(alpha * (s - 0.5))
        where alpha starts small (smooth) and increases (approaches hard rounding).
        The gradient flows through the sigmoid, so the optimizer "knows" about rounding.

    (b) **Penalty term:** Add lambda * ||s - round(s)||^2 to the energy. This pushes
        continuous values toward 0 or 1, making the final rounding less disruptive.
        Lambda increases over time (start smooth, end binary).

    (c) **Straight-through rounding:** Round s to binary for the forward pass (energy
        evaluation) but use the continuous gradient for the backward pass. This is the
        "straight-through estimator" from neural network quantization. In JAX, we
        implement this with jax.lax.stop_gradient tricks.

    We compare all three strategies against:
    - ParallelIsingSampler (discrete Gibbs with simulated annealing)
    - Random baseline (best of N random binary assignments)

    This experiment answers: can continuous optimization match or beat stochastic
    sampling for finding low-energy Ising states? If yes, it opens the door to
    gradient-based reasoning on EBM energy landscapes.

Usage:
    .venv/bin/python scripts/experiment_64_continuous_ising.py
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

import jax
import jax.numpy as jnp
import jax.random as jrandom


# ---------------------------------------------------------------------------
# Ising energy function (continuous domain)
# ---------------------------------------------------------------------------


def ising_energy(s: jax.Array, biases: jax.Array, J: jax.Array) -> jax.Array:
    """Compute Ising energy for (possibly continuous) spin vector.

    **Detailed explanation for engineers:**
        The standard Ising energy for boolean spins is:
            E(s) = -(b^T s + s^T J s)
        where b is the bias vector and J is the symmetric coupling matrix.
        Negative energy = good (low energy configurations are preferred).

        For continuous s ∈ [0,1]^n, this is a smooth quadratic function.
        The gradient ∇E = -(b + (J + J^T) s) = -(b + 2*J s) for symmetric J.
        JAX computes this automatically via jax.grad.

    Args:
        s: Spin vector, shape (n,). Values in [0,1] for continuous, {0,1} for discrete.
        biases: Bias vector, shape (n,).
        J: Symmetric coupling matrix, shape (n, n), zero diagonal.

    Returns:
        Scalar energy (lower is better).
    """
    return -(jnp.dot(biases, s) + jnp.dot(s, J @ s))


# ---------------------------------------------------------------------------
# Strategy A: Plain gradient descent with clip
# ---------------------------------------------------------------------------


def continuous_ising_solve(
    biases: jax.Array,
    J: jax.Array,
    n_steps: int,
    lr: float,
    beta: float,
    key: jax.Array,
) -> jax.Array:
    """Continuous Ising relaxation with projected gradient descent.

    **Detailed explanation for engineers:**
        The simplest continuous relaxation: initialize s randomly in [0,1],
        take gradient steps to minimize E(s), clip to [0,1] after each step,
        and round to binary at the end.

        The beta parameter scales the energy (higher beta = sharper landscape,
        more sensitive to small changes). We multiply the energy by beta before
        taking gradients, equivalent to using lr/beta as the effective step size
        but with better numerical properties.

    Args:
        biases: Bias vector, shape (n,).
        J: Coupling matrix, shape (n, n).
        n_steps: Number of gradient descent steps.
        lr: Learning rate for gradient descent.
        beta: Inverse temperature (scales the energy landscape).
        key: JAX PRNG key for random initialization.

    Returns:
        Binary spin vector, shape (n,), values in {0, 1}.
    """
    n = biases.shape[0]
    # Initialize uniformly in [0,1].
    s = jrandom.uniform(key, (n,), minval=0.0, maxval=1.0)

    # Gradient of the scaled energy with respect to s.
    def scaled_energy(s_val):
        return beta * ising_energy(s_val, biases, J)

    grad_fn = jax.grad(scaled_energy)

    # Gradient descent loop using jax.lax.fori_loop for efficiency.
    # (No Python overhead per step -- runs entirely on accelerator.)
    def step_fn(_, s_val):
        g = grad_fn(s_val)
        s_val = s_val - lr * g
        # Project back to [0,1] (projected gradient descent).
        return jnp.clip(s_val, 0.0, 1.0)

    s = jax.lax.fori_loop(0, n_steps, step_fn, s)

    # Round to binary.
    return (s > 0.5).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Strategy B: Soft rounding via sigmoid annealing
# ---------------------------------------------------------------------------


def continuous_ising_sigmoid(
    biases: jax.Array,
    J: jax.Array,
    n_steps: int,
    lr: float,
    beta: float,
    key: jax.Array,
    alpha_init: float = 1.0,
    alpha_final: float = 20.0,
) -> jax.Array:
    """Continuous Ising with sigmoid annealing (soft → hard rounding).

    **Detailed explanation for engineers:**
        Instead of hard-clipping to [0,1] and rounding at the end, we
        parameterize the spins as s = sigmoid(alpha * z) where z is an
        unconstrained real vector. As alpha increases from alpha_init to
        alpha_final, the sigmoid sharpens:
        - alpha=1: smooth, gradients flow everywhere
        - alpha=20: nearly binary, close to hard rounding

        The key advantage: the optimizer "sees" the rounding during training
        because the sigmoid is differentiable. The gradients tell the optimizer
        how to adjust z so that after rounding, the energy is low.

        This is analogous to temperature annealing in simulated annealing,
        but in the continuous optimization domain.

    Args:
        biases: Bias vector, shape (n,).
        J: Coupling matrix, shape (n, n).
        n_steps: Number of gradient descent steps.
        lr: Learning rate.
        beta: Inverse temperature for energy scaling.
        key: JAX PRNG key.
        alpha_init: Initial sigmoid sharpness (soft).
        alpha_final: Final sigmoid sharpness (hard).

    Returns:
        Binary spin vector, shape (n,).
    """
    n = biases.shape[0]
    # Initialize z near zero so sigmoid(z) ≈ 0.5 (maximum uncertainty).
    z = jrandom.normal(key, (n,)) * 0.1

    def energy_with_sigmoid(z_val, alpha):
        s_val = jax.nn.sigmoid(alpha * z_val)
        return beta * ising_energy(s_val, biases, J)

    def step_fn(step, z_val):
        # Linearly anneal alpha from alpha_init to alpha_final.
        frac = step / jnp.maximum(n_steps - 1, 1)
        alpha = alpha_init + frac * (alpha_final - alpha_init)
        g = jax.grad(energy_with_sigmoid)(z_val, alpha)
        return z_val - lr * g

    z = jax.lax.fori_loop(0, n_steps, step_fn, z)

    # Final hard rounding.
    s = jax.nn.sigmoid(alpha_final * z)
    return (s > 0.5).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Strategy C: Penalty term ||s - round(s)||^2
# ---------------------------------------------------------------------------


def continuous_ising_penalty(
    biases: jax.Array,
    J: jax.Array,
    n_steps: int,
    lr: float,
    beta: float,
    key: jax.Array,
    lam_init: float = 0.0,
    lam_final: float = 10.0,
) -> jax.Array:
    """Continuous Ising with integrality penalty (push toward binary).

    **Detailed explanation for engineers:**
        Adds a penalty term that punishes non-binary values:
            E_total(s) = beta * E_ising(s) + lambda * sum_i s_i * (1 - s_i)

        The term s_i * (1 - s_i) equals 0 when s_i ∈ {0, 1} and peaks at 0.25
        when s_i = 0.5. So the penalty pushes all spins toward binary values.

        Lambda increases linearly from lam_init to lam_final over the
        optimization. Early steps (low lambda) focus on finding a good
        continuous solution; late steps (high lambda) push it toward binary.

        Note: we use s*(1-s) instead of ||s - round(s)||^2 because round()
        has zero gradient everywhere. s*(1-s) is smoothly differentiable and
        achieves the same effect.

    Args:
        biases: Bias vector, shape (n,).
        J: Coupling matrix, shape (n, n).
        n_steps: Number of gradient descent steps.
        lr: Learning rate.
        beta: Inverse temperature.
        key: JAX PRNG key.
        lam_init: Initial penalty weight (0 = no penalty).
        lam_final: Final penalty weight.

    Returns:
        Binary spin vector, shape (n,).
    """
    n = biases.shape[0]
    s = jrandom.uniform(key, (n,), minval=0.0, maxval=1.0)

    def penalized_energy(s_val, lam):
        e_ising = beta * ising_energy(s_val, biases, J)
        # Integrality penalty: s*(1-s) = 0 at binary, 0.25 at midpoint.
        penalty = lam * jnp.sum(s_val * (1.0 - s_val))
        return e_ising + penalty

    def step_fn(step, s_val):
        frac = step / jnp.maximum(n_steps - 1, 1)
        lam = lam_init + frac * (lam_final - lam_init)
        g = jax.grad(penalized_energy)(s_val, lam)
        s_val = s_val - lr * g
        return jnp.clip(s_val, 0.0, 1.0)

    s = jax.lax.fori_loop(0, n_steps, step_fn, s)
    return (s > 0.5).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Strategy D: Straight-through rounding
# ---------------------------------------------------------------------------


def continuous_ising_straight_through(
    biases: jax.Array,
    J: jax.Array,
    n_steps: int,
    lr: float,
    beta: float,
    key: jax.Array,
) -> jax.Array:
    """Continuous Ising with straight-through estimator for rounding.

    **Detailed explanation for engineers:**
        The straight-through estimator (STE) is a trick from neural network
        quantization: in the forward pass, we round s to binary; in the
        backward pass, we pretend the rounding didn't happen and pass the
        gradient straight through.

        In JAX, we implement this as:
            s_hard = (s > 0.5).astype(float)
            s_ste = s + jax.lax.stop_gradient(s_hard - s)

        Forward: s_ste = s_hard (binary)
        Backward: d(s_ste)/d(s) = 1 (gradient flows through s, not s_hard)

        This means the energy is always evaluated at binary values (realistic),
        but the optimizer can still adjust the continuous values smoothly.

    Args:
        biases: Bias vector, shape (n,).
        J: Coupling matrix, shape (n, n).
        n_steps: Number of gradient descent steps.
        lr: Learning rate.
        beta: Inverse temperature.
        key: JAX PRNG key.

    Returns:
        Binary spin vector, shape (n,).
    """
    n = biases.shape[0]
    s = jrandom.uniform(key, (n,), minval=0.0, maxval=1.0)

    def ste_energy(s_val):
        # Straight-through: round in forward, pass gradient in backward.
        s_hard = (s_val > 0.5).astype(jnp.float32)
        s_ste = s_val + jax.lax.stop_gradient(s_hard - s_val)
        return beta * ising_energy(s_ste, biases, J)

    grad_fn = jax.grad(ste_energy)

    def step_fn(_, s_val):
        g = grad_fn(s_val)
        s_val = s_val - lr * g
        return jnp.clip(s_val, 0.0, 1.0)

    s = jax.lax.fori_loop(0, n_steps, step_fn, s)
    return (s > 0.5).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Multi-restart wrapper
# ---------------------------------------------------------------------------


def multi_restart_solve(
    solve_fn,
    biases: jax.Array,
    J: jax.Array,
    n_restarts: int,
    key: jax.Array,
    **kwargs,
) -> jax.Array:
    """Run a continuous solver multiple times and return the best solution.

    **Detailed explanation for engineers:**
        Continuous optimization can get stuck in local minima. Running multiple
        restarts with different random initializations and picking the best
        result is a simple but effective way to improve solution quality.
        This mirrors the ParallelIsingSampler collecting multiple samples.

    Args:
        solve_fn: One of the continuous_ising_* functions.
        biases: Bias vector.
        J: Coupling matrix.
        n_restarts: Number of random restarts.
        key: JAX PRNG key.
        **kwargs: Additional arguments passed to solve_fn.

    Returns:
        Best binary spin vector (lowest energy).
    """
    best_s = None
    best_energy = float("inf")

    keys = jrandom.split(key, n_restarts)
    for i in range(n_restarts):
        s = solve_fn(biases, J, key=keys[i], **kwargs)
        e = float(ising_energy(s, biases, J))
        if e < best_energy:
            best_energy = e
            best_s = s

    return best_s


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def spins_to_assignment(spins: jax.Array, n_vars: int) -> dict[int, bool]:
    """Convert a spin vector to a SAT variable assignment dict."""
    spins_np = np.asarray(spins)
    return {v + 1: bool(spins_np[v] > 0.5) for v in range(n_vars)}


def evaluate_method(
    name: str,
    solve_fn,
    clauses: list,
    biases: jax.Array,
    J: jax.Array,
    n_vars: int,
    key: jax.Array,
    n_restarts: int = 50,
    **kwargs,
) -> dict:
    """Run a solver method and evaluate SAT satisfaction.

    Returns a dict with method name, satisfied count, percentage, and wall time.
    """
    t0 = time.time()
    best_s = multi_restart_solve(
        solve_fn, biases, J, n_restarts=n_restarts, key=key, **kwargs
    )
    elapsed = time.time() - t0

    assignment = spins_to_assignment(best_s, n_vars)
    satisfied, total = check_assignment(clauses, assignment)
    pct = satisfied / total * 100

    return {
        "name": name,
        "satisfied": satisfied,
        "total": total,
        "pct": pct,
        "time": elapsed,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 64: Continuous Ising Relaxation")
    print("  Bridge discrete sampling ↔ continuous gradient optimization")
    print("  Strategies: plain GD, sigmoid anneal, penalty, straight-through")
    print("=" * 70)

    start = time.time()

    # SAT benchmark sizes: 50, 100, 200, 500 variables.
    # Clause ratio ~4.26 is the satisfiability phase transition for 3-SAT.
    test_cases = [
        (50, 4.26),
        (100, 4.26),
        (200, 4.26),
        (500, 4.26),
    ]

    # Shared hyperparameters for continuous methods.
    # Tuned for reasonable performance across problem sizes.
    gd_params = {"n_steps": 1000, "lr": 0.01, "beta": 5.0}
    sig_params = {
        "n_steps": 1000,
        "lr": 0.05,
        "beta": 5.0,
        "alpha_init": 1.0,
        "alpha_final": 20.0,
    }
    pen_params = {
        "n_steps": 1000,
        "lr": 0.01,
        "beta": 5.0,
        "lam_init": 0.0,
        "lam_final": 10.0,
    }
    ste_params = {"n_steps": 1000, "lr": 0.01, "beta": 5.0}

    all_results = []

    for n_vars, ratio in test_cases:
        n_clauses = int(n_vars * ratio)
        print(f"\n{'='*60}")
        print(f"  {n_vars} variables, {n_clauses} clauses (ratio={ratio})")
        print(f"{'='*60}")

        # Generate SAT instance and convert to Ising.
        clauses = random_3sat(n_vars, n_clauses, seed=42 + n_vars)
        biases_vec, weights_vec, _ = sat_to_ising(clauses, n_vars)
        biases, J = sat_to_coupling_matrix(biases_vec, weights_vec, n_vars)

        key = jrandom.PRNGKey(n_vars + 64)
        n_restarts = 50
        size_results = {"n_vars": n_vars, "n_clauses": n_clauses, "methods": []}

        # --- Method 1: Plain continuous GD ---
        k1, k2, k3, k4, k5 = jrandom.split(key, 5)
        r = evaluate_method(
            "Plain GD",
            continuous_ising_solve,
            clauses,
            biases,
            J,
            n_vars,
            k1,
            n_restarts=n_restarts,
            **gd_params,
        )
        print(f"  Plain GD:       {r['satisfied']}/{r['total']} ({r['pct']:.1f}%) in {r['time']:.1f}s")
        size_results["methods"].append(r)

        # --- Method 2: Sigmoid annealing ---
        r = evaluate_method(
            "Sigmoid anneal",
            continuous_ising_sigmoid,
            clauses,
            biases,
            J,
            n_vars,
            k2,
            n_restarts=n_restarts,
            **sig_params,
        )
        print(f"  Sigmoid anneal: {r['satisfied']}/{r['total']} ({r['pct']:.1f}%) in {r['time']:.1f}s")
        size_results["methods"].append(r)

        # --- Method 3: Penalty term ---
        r = evaluate_method(
            "Penalty",
            continuous_ising_penalty,
            clauses,
            biases,
            J,
            n_vars,
            k3,
            n_restarts=n_restarts,
            **pen_params,
        )
        print(f"  Penalty:        {r['satisfied']}/{r['total']} ({r['pct']:.1f}%) in {r['time']:.1f}s")
        size_results["methods"].append(r)

        # --- Method 4: Straight-through ---
        r = evaluate_method(
            "Straight-thru",
            continuous_ising_straight_through,
            clauses,
            biases,
            J,
            n_vars,
            k4,
            n_restarts=n_restarts,
            **ste_params,
        )
        print(f"  Straight-thru:  {r['satisfied']}/{r['total']} ({r['pct']:.1f}%) in {r['time']:.1f}s")
        size_results["methods"].append(r)

        # --- Method 5: Parallel Gibbs sampling ---
        t0 = time.time()
        schedule = AnnealingSchedule(beta_init=0.1, beta_final=10.0, schedule_type="linear")
        sampler = ParallelIsingSampler(
            n_warmup=min(2000, n_vars * 10),
            n_samples=n_restarts,
            steps_per_sample=max(10, n_vars // 10),
            schedule=schedule,
            use_checkerboard=True,
        )
        samples = sampler.sample(k5, biases, J, beta=10.0)
        gibbs_time = time.time() - t0

        samples_np = np.asarray(samples)
        gibbs_best_sat = 0
        for s_idx in range(samples_np.shape[0]):
            assignment = {v + 1: bool(samples_np[s_idx, v]) for v in range(n_vars)}
            sat, _ = check_assignment(clauses, assignment)
            gibbs_best_sat = max(gibbs_best_sat, sat)

        gibbs_pct = gibbs_best_sat / n_clauses * 100
        gibbs_r = {
            "name": "Parallel Gibbs",
            "satisfied": gibbs_best_sat,
            "total": n_clauses,
            "pct": gibbs_pct,
            "time": gibbs_time,
        }
        print(f"  Parallel Gibbs: {gibbs_best_sat}/{n_clauses} ({gibbs_pct:.1f}%) in {gibbs_time:.1f}s")
        size_results["methods"].append(gibbs_r)

        # --- Method 6: Random baseline ---
        t0 = time.time()
        rng = np.random.default_rng(42 + n_vars)
        rand_best_sat = 0
        for _ in range(max(n_restarts, 500)):
            ra = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
            sat, _ = check_assignment(clauses, ra)
            rand_best_sat = max(rand_best_sat, sat)
        rand_time = time.time() - t0

        rand_pct = rand_best_sat / n_clauses * 100
        rand_r = {
            "name": "Random",
            "satisfied": rand_best_sat,
            "total": n_clauses,
            "pct": rand_pct,
            "time": rand_time,
        }
        print(f"  Random (500):   {rand_best_sat}/{n_clauses} ({rand_pct:.1f}%) in {rand_time:.1f}s")
        size_results["methods"].append(rand_r)

        all_results.append(size_results)

    # --- Summary table ---
    elapsed = time.time() - start
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"EXPERIMENT 64 RESULTS ({elapsed:.0f}s)")
    print(sep)

    # Header row with method names.
    method_names = ["Plain GD", "Sigmoid", "Penalty", "STE", "Gibbs", "Random"]
    header = f"{'Vars':>5s} {'Cls':>5s}"
    for mn in method_names:
        header += f" {mn:>10s}"
    print(header)
    print("-" * len(header))

    for sr in all_results:
        row = f"{sr['n_vars']:>5d} {sr['n_clauses']:>5d}"
        for m in sr["methods"]:
            row += f" {m['pct']:>9.1f}%"
        print(row)

    # --- Timing table ---
    print(f"\nTiming (seconds):")
    header2 = f"{'Vars':>5s}"
    for mn in method_names:
        header2 += f" {mn:>10s}"
    print(header2)
    print("-" * len(header2))

    for sr in all_results:
        row = f"{sr['n_vars']:>5d}"
        for m in sr["methods"]:
            row += f" {m['time']:>9.1f}s"
        print(row)

    # --- Analysis: which continuous method is best? ---
    print(f"\n--- Analysis ---")

    # For each problem size, rank the methods.
    for sr in all_results:
        n_vars = sr["n_vars"]
        methods_sorted = sorted(sr["methods"], key=lambda m: -m["pct"])
        best = methods_sorted[0]
        random_pct = sr["methods"][-1]["pct"]  # Random is last
        gibbs_pct = sr["methods"][-2]["pct"]   # Gibbs is second to last

        best_continuous = max(
            (m for m in sr["methods"] if m["name"] not in ("Parallel Gibbs", "Random")),
            key=lambda m: m["pct"],
        )

        delta_vs_random = best_continuous["pct"] - random_pct
        delta_vs_gibbs = best_continuous["pct"] - gibbs_pct

        print(
            f"  {n_vars:>4d} vars: best continuous = {best_continuous['name']} "
            f"({best_continuous['pct']:.1f}%), "
            f"vs Gibbs {delta_vs_gibbs:+.1f}%, vs Random {delta_vs_random:+.1f}%"
        )

    # --- Overall verdict ---
    # Check if any continuous method consistently beats Gibbs.
    continuous_wins = 0
    total_comparisons = 0
    for sr in all_results:
        gibbs_pct = next(m["pct"] for m in sr["methods"] if m["name"] == "Parallel Gibbs")
        for m in sr["methods"]:
            if m["name"] not in ("Parallel Gibbs", "Random"):
                total_comparisons += 1
                if m["pct"] > gibbs_pct:
                    continuous_wins += 1

    print(f"\n  Continuous beats Gibbs: {continuous_wins}/{total_comparisons} comparisons")

    if continuous_wins > total_comparisons // 2:
        print(f"\n  VERDICT: Continuous relaxation outperforms discrete Gibbs sampling.")
        print(f"  Gradient-based optimization is a viable path for EBM reasoning.")
    elif continuous_wins > 0:
        print(f"\n  VERDICT: Mixed results. Continuous relaxation competitive in some cases.")
        print(f"  Hybrid approaches (continuous init → Gibbs refinement) worth exploring.")
    else:
        print(f"\n  VERDICT: Discrete Gibbs sampling dominates continuous relaxation.")
        print(f"  Stochastic exploration handles the rugged SAT landscape better.")
        print(f"  However, continuous methods may excel on smoother energy landscapes.")

    # Always note the rounding gap.
    print(f"\n  NOTE: The rounding gap (continuous → binary) is the key bottleneck.")
    print(f"  Sigmoid annealing addresses this by making rounding differentiable.")
    print(f"  Future work: combine continuous warmstart with discrete refinement.")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
