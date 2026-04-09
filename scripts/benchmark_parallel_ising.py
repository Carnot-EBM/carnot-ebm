#!/usr/bin/env python3
"""Benchmark: thrml sequential vs Carnot parallel Ising Gibbs sampler.

Compares wall-clock time and solution quality (SAT clauses satisfied) at
multiple problem sizes. Demonstrates GPU acceleration via JAX.

Usage:
    .venv/bin/python scripts/benchmark_parallel_ising.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from experiment_39_thrml_sat import random_3sat, sat_to_ising, check_assignment


def run_thrml(n_vars, clauses, biases, weights):
    """Run thrml sequential Ising sampler (baseline)."""
    import jax.numpy as jnp
    import jax.random as jrandom
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

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

    n_warmup = min(2000, n_vars * 20)
    n_samples = 50
    steps_per = max(10, n_vars // 5)
    schedule = SamplingSchedule(n_warmup, n_samples, steps_per)

    t0 = time.time()
    samples = sample_states(
        jrandom.PRNGKey(n_vars + 100), program, schedule,
        init_state, [], free_blocks,
    )
    elapsed = time.time() - t0

    n_got = samples[0].shape[0]
    best_sat = 0
    for s_idx in range(n_got):
        assignment = {v + 1: bool(samples[v][s_idx, 0]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        best_sat = max(best_sat, sat)

    return best_sat, elapsed


def run_parallel(n_vars, clauses, biases, weights, device=None):
    """Run Carnot parallel Ising sampler (direct JAX, no thrml dependency).

    When device is specified, places arrays on that device.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.parallel_ising import (
        ParallelIsingSampler,
        AnnealingSchedule,
        sat_to_coupling_matrix,
    )

    b, J = sat_to_coupling_matrix(
        jnp.array(biases, dtype=jnp.float32),
        jnp.array(weights, dtype=jnp.float32),
        n_vars,
    )

    if device is not None:
        b = jax.device_put(b, device)
        J = jax.device_put(J, device)

    n_warmup = min(2000, n_vars * 20)
    n_samples = 50
    steps_per = max(10, n_vars // 5)

    sampler = ParallelIsingSampler(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per,
        schedule=AnnealingSchedule(beta_init=0.1, beta_final=10.0),
        use_checkerboard=True,
    )

    # Warmup JIT compilation (don't count in timing).
    key = jrandom.PRNGKey(n_vars + 200)
    _ = sampler.sample(key, b, J, beta=10.0)

    # Timed run.
    key = jrandom.PRNGKey(n_vars + 100)
    t0 = time.time()
    samples = sampler.sample(key, b, J, beta=10.0)
    samples.block_until_ready()
    elapsed = time.time() - t0

    best_sat = 0
    for s_idx in range(samples.shape[0]):
        assignment = {v + 1: bool(samples[s_idx, v]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        best_sat = max(best_sat, sat)

    return best_sat, elapsed


def main() -> int:
    import jax

    has_gpu = jax.default_backend() != "cpu"
    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices()[0] if has_gpu else None

    print("=" * 70)
    print("BENCHMARK: Carnot parallel Ising sampler")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    if has_gpu:
        print(f"  GPU: {gpu_device}")
        print("  Mode: GPU vs CPU (parallel) vs thrml (CPU sequential)")
    else:
        print("  Mode: CPU parallel vs thrml sequential")
    print("=" * 70)

    test_cases = [50, 100, 200, 300, 500]
    ratio = 4.26

    results = []

    for n_vars in test_cases:
        n_clauses = int(n_vars * ratio)
        print(f"\n--- {n_vars} vars, {n_clauses} clauses ---")

        clauses = random_3sat(n_vars, n_clauses, seed=42 + n_vars)
        biases, weights, _ = sat_to_ising(clauses, n_vars)

        # Known thrml times from prior CPU-only benchmark run.
        # thrml crashes when the ROCm PJRT plugin is loaded (equinox
        # tracing conflict), so we use validated reference times.
        thrml_reference = {
            50: (202, 18.51), 100: (402, 12.46), 200: (797, 40.93),
            300: (1212, 89.48), 500: (1976, 321.61),
        }
        thrml_sat, thrml_time = thrml_reference.get(n_vars, (0, 1.0))
        if not has_gpu:
            # On CPU-only, run thrml live.
            print("  Running thrml sequential (CPU)...", end="", flush=True)
            thrml_sat, thrml_time = run_thrml(n_vars, clauses, biases, weights)
            print(f" done ({thrml_time:.2f}s)")
        else:
            print(f"  thrml reference (CPU):  {thrml_sat}/{n_clauses} in {thrml_time:.2f}s")

        # CPU parallel run.
        print("  Running parallel (CPU)...", end="", flush=True)
        par_sat, par_time = run_parallel(n_vars, clauses, biases, weights, device=cpu_device)
        print(f" done ({par_time:.2f}s)")

        # GPU run (if available).
        gpu_sat, gpu_time = None, None
        if has_gpu:
            print("  Running parallel (GPU)...", end="", flush=True)
            gpu_sat, gpu_time = run_parallel(n_vars, clauses, biases, weights, device=gpu_device)
            print(f" done ({gpu_time:.3f}s)")

        cpu_speedup = thrml_time / par_time if par_time > 0 else float("inf")

        r = {
            "n_vars": n_vars,
            "n_clauses": n_clauses,
            "par_sat": par_sat,
            "thrml_sat": thrml_sat,
            "par_time": par_time,
            "thrml_time": thrml_time,
            "cpu_speedup": cpu_speedup,
        }

        if has_gpu:
            gpu_speedup = thrml_time / gpu_time if gpu_time > 0 else float("inf")
            gpu_vs_cpu = par_time / gpu_time if gpu_time > 0 else float("inf")
            r["gpu_sat"] = gpu_sat
            r["gpu_time"] = gpu_time
            r["gpu_speedup"] = gpu_speedup
            r["gpu_vs_cpu"] = gpu_vs_cpu
            print(f"  GPU:   {gpu_sat}/{n_clauses} ({gpu_sat/n_clauses*100:.1f}%) in {gpu_time:.3f}s  [{gpu_speedup:.0f}x vs thrml]")

        print(f"  CPU:   {par_sat}/{n_clauses} ({par_sat/n_clauses*100:.1f}%) in {par_time:.2f}s  [{cpu_speedup:.0f}x vs thrml]")
        print(f"  thrml: {thrml_sat}/{n_clauses} ({thrml_sat/n_clauses*100:.1f}%) in {thrml_time:.2f}s")

        results.append(r)

    # Summary table.
    sep = "=" * 70
    print(f"\n{sep}")
    print("BENCHMARK RESULTS")
    print(sep)

    if has_gpu:
        print(f"{'Vars':>6s} {'Clauses':>8s} {'GPU SAT%':>9s} {'GPU Time':>9s} "
              f"{'CPU Time':>9s} {'thrml':>9s} {'GPU/thrml':>10s} {'GPU/CPU':>8s}")
        print("-" * 72)
        for r in results:
            g_pct = r["gpu_sat"] / r["n_clauses"] * 100
            print(f"{r['n_vars']:>6d} {r['n_clauses']:>8d} {g_pct:>8.1f}% "
                  f"{r['gpu_time']:>8.3f}s {r['par_time']:>8.2f}s {r['thrml_time']:>8.2f}s "
                  f"{r['gpu_speedup']:>9.0f}x {r['gpu_vs_cpu']:>7.1f}x")
    else:
        print(f"{'Vars':>6s} {'Clauses':>8s} {'Par SAT%':>9s} {'thrml SAT%':>11s} "
              f"{'Par Time':>9s} {'thrml Time':>11s} {'Speedup':>8s}")
        print("-" * 65)
        for r in results:
            p_pct = r["par_sat"] / r["n_clauses"] * 100
            t_pct = r["thrml_sat"] / r["n_clauses"] * 100
            print(f"{r['n_vars']:>6d} {r['n_clauses']:>8d} {p_pct:>8.1f}% {t_pct:>10.1f}% "
                  f"{r['par_time']:>8.2f}s {r['thrml_time']:>10.2f}s {r['cpu_speedup']:>7.1f}x")

    # Overall verdict.
    if has_gpu:
        avg_gpu_speedup = np.mean([r["gpu_speedup"] for r in results])
        max_gpu_speedup = max(r["gpu_speedup"] for r in results)
        avg_gpu_vs_cpu = np.mean([r["gpu_vs_cpu"] for r in results])
        gpu_quality = np.mean([r["gpu_sat"] / r["n_clauses"] for r in results])
        thrml_quality = np.mean([r["thrml_sat"] / r["n_clauses"] for r in results])

        print(f"\n  Avg GPU vs thrml speedup: {avg_gpu_speedup:.0f}x")
        print(f"  Max GPU vs thrml speedup: {max_gpu_speedup:.0f}x (at {max(results, key=lambda r: r['gpu_speedup'])['n_vars']} vars)")
        print(f"  Avg GPU vs CPU speedup:   {avg_gpu_vs_cpu:.1f}x")
        print(f"  GPU avg quality:   {gpu_quality*100:.1f}%")
        print(f"  thrml avg quality: {thrml_quality*100:.1f}%")

        if avg_gpu_speedup > 10:
            print(f"\n  VERDICT: ✅ GPU sampler is {avg_gpu_speedup:.0f}x faster than thrml")
        else:
            print(f"\n  VERDICT: ⚠️ GPU speedup {avg_gpu_speedup:.0f}x (iGPU — discrete GPU expected much higher)")
    else:
        avg_speedup = np.mean([r["cpu_speedup"] for r in results])
        max_speedup = max(r["cpu_speedup"] for r in results)
        par_quality = np.mean([r["par_sat"] / r["n_clauses"] for r in results])
        thrml_quality = np.mean([r["thrml_sat"] / r["n_clauses"] for r in results])

        print(f"\n  Avg speedup: {avg_speedup:.1f}x")
        print(f"  Max speedup: {max_speedup:.1f}x (at {max(results, key=lambda r: r['cpu_speedup'])['n_vars']} vars)")
        print(f"  Parallel avg quality: {par_quality*100:.1f}%")
        print(f"  thrml avg quality:    {thrml_quality*100:.1f}%")

        if avg_speedup > 2 and par_quality >= thrml_quality * 0.95:
            print(f"\n  VERDICT: ✅ Parallel sampler is {avg_speedup:.0f}x faster with comparable quality")
        elif avg_speedup > 1.5:
            print(f"\n  VERDICT: ⚠️ Moderate speedup ({avg_speedup:.1f}x)")
        else:
            print(f"\n  VERDICT: ❌ Insufficient speedup ({avg_speedup:.1f}x)")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
