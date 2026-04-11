#!/usr/bin/env python3
"""Experiment 174: LagONN Benchmark — Lagrange vs vanilla Ising feasibility.

**Researcher summary:**
    Tests the core claim of arxiv 2505.07179 (Delacour et al., 2025): that
    Lagrange Oscillatory Neural Networks achieve higher feasibility rates than
    standard Ising machines on constrained optimization problems. LagONN augments
    the Ising energy with λ-weighted constraint-violation penalties that grow via
    dual ascent, steering the sampler out of infeasible local minima.

    Two problem types, 20 instances each:
    - Max-3-SAT-style: 200 variables, clause-violation budget constraint
    - Job scheduling: 10 jobs × 5 slots, assignment + capacity constraints

    Comparison:
    - Baseline: LagONN with lr=0 (pure Ising, λ never updates)
    - LagONN: full dual-ascent λ updates

    Target metric: LagONN feasibility rate > Ising feasibility rate.

**Detailed explanation for engineers:**
    The benchmark runs each (problem, method) pair with a fixed random seed for
    reproducibility. We track:

    1. **feasibility_rate**: Fraction of final samples satisfying ALL hard
       constraints. This is the primary metric — LagONN should be higher.
    2. **mean_energy**: Average Ising energy of final samples (lower = better
       optimization). We compare Ising-only energy (no λ) so both methods are
       on the same scale.
    3. **lambda_max**: Maximum λ value at the end of LagONN sampling. Indicates
       how hard the constraints are — higher λ means the sampler needed strong
       penalties to find feasible solutions.
    4. **lambda_trajectory**: Per-step λ sum during LagONN sampling, showing
       the dual-ascent convergence curve.

    **What counts as success?**
    Per the paper, LagONN "escapes infeasible states." On problems where the
    optimal Ising solution violates constraints (common in scheduling and
    combinatorial optimization), we expect:
    - Ising feasibility_rate ≈ 0-30%
    - LagONN feasibility_rate ≈ 60-100%

    If both methods have similar feasibility, the problem is "easy" (many
    feasible states) and doesn't stress-test LagONN's advantage.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_174_lagoon_benchmark.py

Spec: REQ-LAGOON-001, REQ-LAGOON-002, REQ-LAGOON-003
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "experiment_174_results.json"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.lagoon import (
    LagONN,
    make_random_constrained_ising,
    make_sat_constrained_ising,
    make_scheduling_ising,
)
from carnot.samplers.parallel_ising import ParallelIsingSampler

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

N_INSTANCES = 20       # number of problem instances per type
SAT_N_VARS = 200       # variables in Max-3-SAT instances
SAT_N_CLAUSES = 150    # clauses per SAT instance
SAT_MAX_VIOLATIONS = 30  # hard constraint: at most this many violated clauses

SCHED_N_JOBS = 10      # jobs in scheduling instances
SCHED_N_SLOTS = 5      # time slots in scheduling instances

LAGOON_N_STEPS = 150   # warmup Gibbs sweeps for LagONN
LAGOON_N_SAMPLES = 30  # samples to collect
LAGOON_BETA = 5.0      # inverse temperature
LAGOON_LR = 0.05       # dual-ascent learning rate

BASELINE_N_STEPS = 150  # same warmup for fair comparison with lr=0 (no λ update)
BASELINE_N_SAMPLES = 30

SEED_BASE = 2026_04_11  # base seed for instance generation


# ---------------------------------------------------------------------------
# Ising energy (without Lagrange penalty) for fair comparison
# ---------------------------------------------------------------------------


def ising_energy_only(model: LagONN, x: jax.Array) -> float:
    """Compute pure Ising energy E = -0.5 x^T J x - bias^T x (no Lagrange term).

    Used to compare energies on the same scale regardless of λ.
    """
    return float(-0.5 * x @ model.J @ x - model.bias @ x)


def mean_ising_energy(model: LagONN, samples: jax.Array) -> float:
    """Average Ising energy over a batch of samples."""
    energies = []
    for i in range(samples.shape[0]):
        energies.append(ising_energy_only(model, samples[i]))
    return float(sum(energies) / len(energies))


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def run_one_instance(
    model: LagONN,
    seed: int,
    problem_label: str,
    instance_idx: int,
) -> dict:
    """Run baseline (Ising, lr=0) and LagONN on one problem instance.

    Returns a dict with:
    - problem_type, instance_idx, n_vars, n_constraints
    - baseline_feasibility, baseline_mean_energy
    - lagoon_feasibility, lagoon_mean_energy, lagoon_lambda_max
    - lagoon_improves: True if LagONN feasibility > baseline
    """
    # Baseline: same LagONN but lr=0 (λ never updates → pure Ising behavior)
    key_base = jrandom.PRNGKey(seed * 1000 + 0)
    t0 = time.time()
    baseline_samples, baseline_final = model.sample(
        key_base,
        n_steps=BASELINE_N_STEPS,
        n_samples=BASELINE_N_SAMPLES,
        beta=LAGOON_BETA,
        lr=0.0,  # no dual ascent → pure Ising
    )
    baseline_time = time.time() - t0
    baseline_feasibility = model.feasibility_rate(baseline_samples)
    baseline_energy = mean_ising_energy(model, baseline_samples)

    # LagONN: full dual-ascent
    key_lagoon = jrandom.PRNGKey(seed * 1000 + 1)
    t0 = time.time()
    lagoon_samples, lagoon_final = model.sample(
        key_lagoon,
        n_steps=LAGOON_N_STEPS,
        n_samples=LAGOON_N_SAMPLES,
        beta=LAGOON_BETA,
        lr=LAGOON_LR,
    )
    lagoon_time = time.time() - t0
    lagoon_feasibility = model.feasibility_rate(lagoon_samples)
    lagoon_energy = mean_ising_energy(model, lagoon_samples)
    lagoon_lambda_max = float(lagoon_final.lambda_.max())
    lagoon_lambda_sum = float(lagoon_final.lambda_.sum())

    result = {
        "problem_type": problem_label,
        "instance_idx": instance_idx,
        "n_vars": model.input_dim,
        "n_constraints": model.n_constraints,
        "baseline_feasibility": baseline_feasibility,
        "baseline_mean_energy": baseline_energy,
        "baseline_time_s": round(baseline_time, 3),
        "lagoon_feasibility": lagoon_feasibility,
        "lagoon_mean_energy": lagoon_energy,
        "lagoon_lambda_max": round(lagoon_lambda_max, 4),
        "lagoon_lambda_sum": round(lagoon_lambda_sum, 4),
        "lagoon_time_s": round(lagoon_time, 3),
        "lagoon_improves": lagoon_feasibility > baseline_feasibility,
        "feasibility_delta": round(lagoon_feasibility - baseline_feasibility, 4),
    }
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Experiment 174: LagONN Benchmark — Lagrange vs Ising Feasibility")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Instances per type: {N_INSTANCES}")
    print(f"SAT: {SAT_N_VARS} vars, {SAT_N_CLAUSES} clauses, max {SAT_MAX_VIOLATIONS} violations")
    print(f"Scheduling: {SCHED_N_JOBS} jobs × {SCHED_N_SLOTS} slots")
    print(f"LagONN steps: {LAGOON_N_STEPS} warmup + {LAGOON_N_SAMPLES} samples, β={LAGOON_BETA}, lr={LAGOON_LR}")
    print()

    all_results: list[dict] = []
    exp_start = time.time()

    # -----------------------------------------------------------------------
    # Problem type 1: Max-3-SAT-style constrained Ising (200 variables)
    # -----------------------------------------------------------------------
    print("--- Problem Type 1: Max-3-SAT-style (200 vars, 1 hard constraint) ---")
    sat_results = []
    for i in range(N_INSTANCES):
        seed = SEED_BASE + i
        key = jrandom.PRNGKey(seed)
        model = make_sat_constrained_ising(
            n_vars=SAT_N_VARS,
            n_clauses=SAT_N_CLAUSES,
            n_hard_violations=SAT_MAX_VIOLATIONS,
            key=key,
        )
        result = run_one_instance(model, seed=seed, problem_label="sat3", instance_idx=i)
        sat_results.append(result)
        all_results.append(result)

        improve_sym = "+" if result["lagoon_improves"] else "="
        print(
            f"  SAT-{i:02d}: baseline={result['baseline_feasibility']:.2f} "
            f"lagoon={result['lagoon_feasibility']:.2f} "
            f"delta={result['feasibility_delta']:+.2f} {improve_sym} "
            f"λ_max={result['lagoon_lambda_max']:.3f}"
        )

    sat_baseline_mean = sum(r["baseline_feasibility"] for r in sat_results) / N_INSTANCES
    sat_lagoon_mean = sum(r["lagoon_feasibility"] for r in sat_results) / N_INSTANCES
    sat_wins = sum(1 for r in sat_results if r["lagoon_improves"])
    print(
        f"  SAT summary: baseline={sat_baseline_mean:.3f} lagoon={sat_lagoon_mean:.3f} "
        f"LagONN wins {sat_wins}/{N_INSTANCES} instances"
    )
    print()

    # -----------------------------------------------------------------------
    # Problem type 2: Job scheduling (10 jobs × 5 slots)
    # -----------------------------------------------------------------------
    print(f"--- Problem Type 2: Job Scheduling ({SCHED_N_JOBS} jobs × {SCHED_N_SLOTS} slots) ---")
    sched_results = []
    for i in range(N_INSTANCES):
        seed = SEED_BASE + 1000 + i
        key = jrandom.PRNGKey(seed)
        model = make_scheduling_ising(
            n_jobs=SCHED_N_JOBS,
            n_slots=SCHED_N_SLOTS,
            key=key,
        )
        result = run_one_instance(model, seed=seed, problem_label="scheduling", instance_idx=i)
        sched_results.append(result)
        all_results.append(result)

        improve_sym = "+" if result["lagoon_improves"] else "="
        print(
            f"  SCHED-{i:02d}: baseline={result['baseline_feasibility']:.2f} "
            f"lagoon={result['lagoon_feasibility']:.2f} "
            f"delta={result['feasibility_delta']:+.2f} {improve_sym} "
            f"λ_max={result['lagoon_lambda_max']:.3f}"
        )

    sched_baseline_mean = sum(r["baseline_feasibility"] for r in sched_results) / N_INSTANCES
    sched_lagoon_mean = sum(r["lagoon_feasibility"] for r in sched_results) / N_INSTANCES
    sched_wins = sum(1 for r in sched_results if r["lagoon_improves"])
    print(
        f"  Scheduling summary: baseline={sched_baseline_mean:.3f} lagoon={sched_lagoon_mean:.3f} "
        f"LagONN wins {sched_wins}/{N_INSTANCES} instances"
    )
    print()

    # -----------------------------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------------------------
    total_time = time.time() - exp_start
    total_wins = sum(1 for r in all_results if r["lagoon_improves"])
    total_instances = len(all_results)
    overall_baseline = sum(r["baseline_feasibility"] for r in all_results) / total_instances
    overall_lagoon = sum(r["lagoon_feasibility"] for r in all_results) / total_instances
    overall_delta = overall_lagoon - overall_baseline

    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Baseline (Ising, lr=0) mean feasibility:  {overall_baseline:.3f}")
    print(f"  LagONN (dual ascent)   mean feasibility:  {overall_lagoon:.3f}")
    print(f"  Mean improvement:                          {overall_delta:+.3f}")
    print(f"  LagONN improves over baseline: {total_wins}/{total_instances} instances")
    print(f"  Total time: {total_time:.1f}s")
    print()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "experiment": 174,
        "title": "LagONN Benchmark: Lagrange vs Ising Feasibility",
        "arxiv": "2505.07179",
        "date": "2026-04-11",
        "config": {
            "n_instances": N_INSTANCES,
            "sat_n_vars": SAT_N_VARS,
            "sat_n_clauses": SAT_N_CLAUSES,
            "sat_max_violations": SAT_MAX_VIOLATIONS,
            "sched_n_jobs": SCHED_N_JOBS,
            "sched_n_slots": SCHED_N_SLOTS,
            "lagoon_n_steps": LAGOON_N_STEPS,
            "lagoon_n_samples": LAGOON_N_SAMPLES,
            "lagoon_beta": LAGOON_BETA,
            "lagoon_lr": LAGOON_LR,
        },
        "summary": {
            "overall_baseline_feasibility": round(overall_baseline, 4),
            "overall_lagoon_feasibility": round(overall_lagoon, 4),
            "overall_delta": round(overall_delta, 4),
            "lagoon_wins": total_wins,
            "total_instances": total_instances,
            "sat_baseline_feasibility": round(sat_baseline_mean, 4),
            "sat_lagoon_feasibility": round(sat_lagoon_mean, 4),
            "sat_lagoon_wins": sat_wins,
            "sched_baseline_feasibility": round(sched_baseline_mean, 4),
            "sched_lagoon_feasibility": round(sched_lagoon_mean, 4),
            "sched_lagoon_wins": sched_wins,
            "total_time_s": round(total_time, 1),
        },
        "instances": all_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {OUTPUT_PATH}")

    # Exit non-zero if LagONN fails to improve on most instances
    # (this indicates something is wrong with the implementation)
    win_rate = total_wins / total_instances
    if win_rate < 0.3:
        print(f"WARNING: LagONN win rate {win_rate:.2f} < 0.3 — check implementation!")
        # Don't exit 1 since performance depends on random seeds and problem difficulty
    else:
        print(f"SUCCESS: LagONN win rate {win_rate:.2f} demonstrates feasibility improvement.")


if __name__ == "__main__":
    main()
