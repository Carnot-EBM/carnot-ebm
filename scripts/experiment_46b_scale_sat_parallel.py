#!/usr/bin/env python3
"""Experiment 46b: Scale SAT to 1000+ variables using parallel Ising sampler.

The parallel Ising sampler (ParallelIsingSampler) is 183x faster than thrml's
sequential per-spin Gibbs loop. This experiment uses it to tackle SAT instances
far larger than thrml can handle in reasonable time:

    500, 1000, 2000, 5000 variables at the SAT phase transition (alpha ~ 4.26).

At these sizes, random search's coverage degrades exponentially while the
parallel thermodynamic sampler's GPU-vectorized matrix-vector updates explore
the energy landscape efficiently. The checkerboard decomposition keeps
sampling quality high, and simulated annealing (beta 0.1 -> 10.0) avoids
early trapping in local minima.

Key difference from experiment_46 (which uses thrml):
    - Uses ParallelIsingSampler directly (no thrml dependency)
    - Uses sat_to_coupling_matrix to build the dense J matrix
    - Can handle 5000+ variables in reasonable wall-clock time

Usage:
    .venv/bin/python scripts/experiment_46b_scale_sat_parallel.py
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


def main() -> int:
    import jax
    import jax.random as jrandom

    print("=" * 70)
    print("EXPERIMENT 46b: Scale SAT to 1000+ Vars (Parallel Ising Sampler)")
    print("  Uses ParallelIsingSampler (183x faster than thrml)")
    print("  Checkerboard Gibbs + simulated annealing (beta 0.1 -> 10.0)")
    print("=" * 70)

    start = time.time()

    # Test sizes: 500 to 5000 variables at the SAT phase transition.
    # alpha ~ 4.26 is where random 3-SAT undergoes a satisfiability phase
    # transition -- instances here are hardest for both random and structured
    # solvers.
    test_cases = [
        (500, 4.26),
        (1000, 4.26),
        (2000, 4.26),
        (5000, 4.26),
    ]

    results = []

    for n_vars, ratio in test_cases:
        n_clauses = int(n_vars * ratio)
        print(f"\n--- {n_vars} vars, {n_clauses} clauses ---")

        # Step 1: Generate random 3-SAT instance.
        clauses = random_3sat(n_vars, n_clauses, seed=42 + n_vars)

        # Step 2: Convert SAT to Ising parameters (flat vectors).
        biases_vec, weights_vec, _ = sat_to_ising(clauses, n_vars)

        # Step 3: Convert flat vectors to dense coupling matrix for the
        # parallel sampler. The matrix J is symmetric with zero diagonal,
        # shape (n_vars, n_vars).
        biases, J = sat_to_coupling_matrix(biases_vec, weights_vec, n_vars)

        # Step 4: Configure and run the parallel sampler.
        # Sampling parameters scale with problem size:
        #   - n_warmup: more warmup for larger problems (capped at 2000)
        #   - steps_per_sample: longer decorrelation for larger state spaces
        #   - annealing: start hot (beta=0.1, broad exploration) and cool to
        #     beta=10.0 (focused low-energy sampling)
        n_warmup = min(2000, n_vars * 10)
        n_samples = 50
        steps_per = max(10, n_vars // 10)

        schedule = AnnealingSchedule(
            beta_init=0.1,
            beta_final=10.0,
            schedule_type="linear",
        )

        sampler = ParallelIsingSampler(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per,
            schedule=schedule,
            use_checkerboard=True,
        )

        t0 = time.time()
        key = jrandom.PRNGKey(n_vars + 200)
        samples = sampler.sample(key, biases, J, beta=10.0)
        # samples shape: (n_samples, n_vars), boolean
        sample_time = time.time() - t0

        # Step 5: Evaluate -- find the best assignment from sampler output.
        n_got = samples.shape[0]
        # Convert JAX array to numpy for check_assignment.
        samples_np = np.asarray(samples)

        best_sat = 0
        for s_idx in range(n_got):
            assignment = {
                v + 1: bool(samples_np[s_idx, v]) for v in range(n_vars)
            }
            sat, _ = check_assignment(clauses, assignment)
            best_sat = max(best_sat, sat)

        # Step 6: Random baseline -- 500 independent random tries.
        rng = np.random.default_rng(42)
        rand_best = 0
        n_random_tries = 500
        for _ in range(n_random_tries):
            ra = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
            sat, _ = check_assignment(clauses, ra)
            rand_best = max(rand_best, sat)

        t_pct = best_sat / n_clauses * 100
        r_pct = rand_best / n_clauses * 100

        print(f"  parallel: {best_sat}/{n_clauses} ({t_pct:.1f}%) in {sample_time:.1f}s")
        print(f"  random ({n_random_tries} tries): {rand_best}/{n_clauses} ({r_pct:.1f}%)")
        print(f"  delta: {t_pct - r_pct:+.1f}%")

        results.append({
            "n_vars": n_vars,
            "n_clauses": n_clauses,
            "parallel": best_sat,
            "random": rand_best,
            "total": n_clauses,
            "time": sample_time,
        })

    # Summary table.
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 46b RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"{'Vars':>6s} {'Clauses':>8s} {'Parallel':>10s} {'Random':>8s} {'Delta':>8s} {'Time':>8s}")
    print("-" * 52)

    for r in results:
        t = r["parallel"] / r["total"] * 100
        ra = r["random"] / r["total"] * 100
        print(
            f"{r['n_vars']:>6d} {r['n_clauses']:>8d} "
            f"{t:>9.1f}% {ra:>7.1f}% {t - ra:>+7.1f}% {r['time']:>7.1f}s"
        )

    # Verdict: does the parallel sampler's advantage grow with problem size?
    if len(results) >= 3:
        small_delta = (
            results[0]["parallel"] / results[0]["total"]
            - results[0]["random"] / results[0]["total"]
        )
        large_delta = (
            results[-1]["parallel"] / results[-1]["total"]
            - results[-1]["random"] / results[-1]["total"]
        )
        if large_delta > small_delta:
            print(
                f"\n  VERDICT: ✅ Parallel Ising advantage grows with problem size"
            )
            print(
                f"           Small delta: {small_delta * 100:+.1f}%  "
                f"Large delta: {large_delta * 100:+.1f}%"
            )
        else:
            print(
                f"\n  VERDICT: ⚠️ Parallel Ising advantage does not clearly grow with size"
            )
            print(
                f"           Small delta: {small_delta * 100:+.1f}%  "
                f"Large delta: {large_delta * 100:+.1f}%"
            )

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
