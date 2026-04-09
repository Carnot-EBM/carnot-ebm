#!/usr/bin/env python3
"""Experiment 46: Scale thrml SAT solver to 500+ variables.

At 50 variables, thrml beat random by 2-4%. This experiment tests
larger instances where random search's coverage degrades exponentially
while thermodynamic sampling follows the energy landscape.

Tests: 100, 200, 300, 500 variables at the SAT phase transition (α≈4.26).

Usage:
    .venv/bin/python scripts/experiment_46_scale_sat.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from experiment_39_thrml_sat import random_3sat, sat_to_ising, check_assignment


def main() -> int:
    import jax.numpy as jnp
    import jax.random as jrandom
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

    print("=" * 70)
    print("EXPERIMENT 46: Scale SAT to 500+ Variables")
    print("=" * 70)

    start = time.time()

    test_cases = [
        (50, 4.26),
        (100, 4.26),
        (200, 4.26),
        (300, 4.26),
        (500, 4.26),
    ]

    results = []

    for n_vars, ratio in test_cases:
        n_clauses = int(n_vars * ratio)
        print(f"\n--- {n_vars} vars, {n_clauses} clauses ---")

        clauses = random_3sat(n_vars, n_clauses, seed=42 + n_vars)
        biases, weights, _ = sat_to_ising(clauses, n_vars)

        nodes = [SpinNode() for _ in range(n_vars)]
        edges = [(nodes[i], nodes[j]) for i in range(n_vars) for j in range(i + 1, n_vars)]

        model = IsingEBM(
            nodes=nodes, edges=edges,
            biases=jnp.array(biases, dtype=jnp.float32),
            weights=jnp.array(weights, dtype=jnp.float32),
            beta=jnp.array(10.0),
        )

        free_blocks = [Block([nodes[i]]) for i in range(n_vars)]
        program = IsingSamplingProgram(model, free_blocks, [])
        init_state = hinton_init(jrandom.PRNGKey(n_vars), model, free_blocks, ())

        # Scale sampling with problem size
        n_warmup = min(2000, n_vars * 20)
        n_samples = 50
        steps_per = max(10, n_vars // 5)
        schedule = SamplingSchedule(n_warmup, n_samples, steps_per)

        t0 = time.time()
        samples = sample_states(
            jrandom.PRNGKey(n_vars + 100), program, schedule,
            init_state, [], free_blocks,
        )
        sample_time = time.time() - t0

        # Best from thrml
        n_got = samples[0].shape[0]
        best_sat = 0
        for s_idx in range(n_got):
            assignment = {v + 1: bool(samples[v][s_idx, 0]) for v in range(n_vars)}
            sat, _ = check_assignment(clauses, assignment)
            best_sat = max(best_sat, sat)

        # Best from random (more tries for fair comparison)
        rng = np.random.default_rng(42)
        rand_best = 0
        n_random_tries = max(n_got, 500)
        for _ in range(n_random_tries):
            ra = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
            sat, _ = check_assignment(clauses, ra)
            rand_best = max(rand_best, sat)

        t_pct = best_sat / n_clauses * 100
        r_pct = rand_best / n_clauses * 100

        print(f"  thrml: {best_sat}/{n_clauses} ({t_pct:.1f}%) in {sample_time:.1f}s")
        print(f"  random ({n_random_tries} tries): {rand_best}/{n_clauses} ({r_pct:.1f}%)")
        print(f"  delta: {t_pct - r_pct:+.1f}%")

        results.append({
            "n_vars": n_vars,
            "n_clauses": n_clauses,
            "thrml": best_sat,
            "random": rand_best,
            "total": n_clauses,
            "time": sample_time,
        })

    # Summary
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 46 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"{'Vars':>6s} {'Clauses':>8s} {'thrml':>8s} {'Random':>8s} {'Delta':>8s} {'Time':>8s}")
    print("-" * 50)

    for r in results:
        t = r["thrml"] / r["total"] * 100
        ra = r["random"] / r["total"] * 100
        print(f"{r['n_vars']:>6d} {r['n_clauses']:>8d} {t:>7.1f}% {ra:>7.1f}% {t-ra:>+7.1f}% {r['time']:>7.1f}s")

    # Does advantage grow with size?
    if len(results) >= 3:
        small_delta = results[0]["thrml"] / results[0]["total"] - results[0]["random"] / results[0]["total"]
        large_delta = results[-1]["thrml"] / results[-1]["total"] - results[-1]["random"] / results[-1]["total"]
        if large_delta > small_delta:
            print(f"\n  VERDICT: ✅ thrml advantage grows with problem size")
        else:
            print(f"\n  VERDICT: ⚠️ thrml advantage does not clearly grow with size")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
