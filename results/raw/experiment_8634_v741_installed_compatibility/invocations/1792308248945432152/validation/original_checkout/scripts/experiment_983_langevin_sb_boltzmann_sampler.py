"""Experiment 983: Langevin Stochastic Boltzmann (LSB) Sampler Benchmark.

Implements and benchmarks the LSB sampler from arXiv 2512.02323 against the
existing ParallelIsingSampler on 5 constraint problems.

Verdict logic:
  lsb_speedup_ratio = ising_wall_time_s / lsb_wall_time_s
  > 1.0  → lsb_faster_set_default
  0.8-1.0 → lsb_comparable_optional
  < 0.8  → lsb_slower_optional
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.samplers.lsb_sampler import LangevinSBSampler
from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

RESULT_PATH = Path("results/experiment_983_langevin_sb_boltzmann_sampler.json")

# ──────────────────────────────────────────────────────────────────────────────
# Constraint problems (5 problems matching Exp 285 parallelism benchmark style)
# ──────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(285)


def _make_problem(
    n: int, density: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random Ising constraint problem.

    Args:
        n: Number of spins.
        density: Fraction of off-diagonal couplings that are nonzero.
        rng: NumPy random generator.

    Returns:
        (biases, couplings) as float32 numpy arrays.
    """
    biases = rng.normal(0, 1, n).astype(np.float32)
    J_upper = (rng.random((n, n)) < density) * rng.normal(0, 0.5, (n, n))
    J = (J_upper + J_upper.T).astype(np.float32)
    np.fill_diagonal(J, 0.0)
    return biases, J


PROBLEMS = [
    _make_problem(n=20, density=0.3, rng=RNG),
    _make_problem(n=40, density=0.2, rng=RNG),
    _make_problem(n=80, density=0.15, rng=RNG),
    _make_problem(n=60, density=0.25, rng=RNG),
    _make_problem(n=100, density=0.1, rng=RNG),
]

BETA = 8.0
N_SAMPLES = 30
N_WARMUP = 300
STEPS_PER_SAMPLE = 10
TARGET_ENERGY_FRACTION = 0.5  # convergence = samples with energy in bottom 50%


# ──────────────────────────────────────────────────────────────────────────────
# Energy computation
# ──────────────────────────────────────────────────────────────────────────────


def ising_energy(spins: np.ndarray, biases: np.ndarray, J: np.ndarray) -> float:
    """Compute Ising energy E = -b^T s - s^T J s for boolean spins."""
    s = spins.astype(np.float32)
    return float(-biases @ s - s @ J @ s)


def mean_best_energy(samples: np.ndarray, biases: np.ndarray, J: np.ndarray) -> float:
    """Mean energy across the best 25% of samples (measures convergence quality)."""
    energies = np.array([ising_energy(s, biases, J) for s in samples])
    n_best = max(1, len(energies) // 4)
    return float(np.mean(np.sort(energies)[:n_best]))


def count_convergence_steps(
    biases: np.ndarray,
    J: np.ndarray,
    sampler_cls: str,
    beta: float,
    target_fraction: float,
    max_steps: int = 2000,
    step_size: int = 50,
) -> int:
    """Estimate iterations needed to reach target energy quality.

    Runs the sampler with increasing n_warmup until the best-25% mean energy
    falls below target_fraction * initial_energy. Returns the first n_warmup
    at which this threshold is reached, or max_steps if never reached.

    Args:
        biases: Bias vector.
        J: Coupling matrix.
        sampler_cls: "lsb" or "ising".
        beta: Inverse temperature.
        target_fraction: Energy reduction target (0.5 = 50% below random init).
        max_steps: Maximum warmup steps to try.
        step_size: Increment in warmup steps per trial.

    Returns:
        Number of warmup iterations when convergence was first detected.
    """
    n = biases.shape[0]
    # Random baseline energy.
    rng = np.random.default_rng(42)
    random_samples = rng.integers(0, 2, size=(20, n)).astype(bool)
    baseline_energy = float(np.mean([ising_energy(s, biases, J) for s in random_samples]))
    target_energy = baseline_energy * target_fraction

    for n_steps in range(step_size, max_steps + 1, step_size):
        if sampler_cls == "lsb":
            sampler = LangevinSBSampler(
                lr=0.05,
                beta=beta,
                n_warmup=n_steps,
                n_samples=N_SAMPLES,
                steps_per_sample=STEPS_PER_SAMPLE,
                use_cem=True,
                seed=42,
            )
            samples = sampler.run_sampler(
                jrandom.PRNGKey(42),
                jnp.asarray(biases),
                jnp.asarray(J),
                beta,
            )
            samples_np = np.asarray(samples)
        else:  # "ising"
            sampler = ParallelIsingSampler(
                n_warmup=n_steps,
                n_samples=N_SAMPLES,
                steps_per_sample=STEPS_PER_SAMPLE,
                schedule=AnnealingSchedule(beta_init=0.5, beta_final=beta),
                use_checkerboard=True,
            )
            samples_jax = sampler.sample(
                jrandom.PRNGKey(42),
                jnp.asarray(biases),
                jnp.asarray(J),
                beta=beta,
            )
            samples_np = np.asarray(samples_jax)

        best_energy = mean_best_energy(samples_np, biases, J)
        if best_energy <= target_energy:
            return n_steps

    return max_steps


# ──────────────────────────────────────────────────────────────────────────────
# Wall-clock timing
# ──────────────────────────────────────────────────────────────────────────────


def time_sampler(sampler_cls: str, beta: float) -> float:
    """Total wall-clock time for N_SAMPLES on all 5 constraint problems."""
    total = 0.0
    for biases, J in PROBLEMS:
        n = biases.shape[0]
        b_jax = jnp.asarray(biases)
        J_jax = jnp.asarray(J)

        if sampler_cls == "lsb":
            sampler = LangevinSBSampler(
                lr=0.05,
                beta=beta,
                n_warmup=N_WARMUP,
                n_samples=N_SAMPLES,
                steps_per_sample=STEPS_PER_SAMPLE,
                use_cem=True,
                seed=42,
            )
            # Warm up JIT first run (discard time).
            sampler.run_sampler(jrandom.PRNGKey(0), b_jax, J_jax, beta)
            t0 = time.perf_counter()
            result = sampler.run_sampler(jrandom.PRNGKey(1), b_jax, J_jax, beta)
            _ = np.asarray(result)  # force device→host transfer
            total += time.perf_counter() - t0
        else:
            sampler = ParallelIsingSampler(
                n_warmup=N_WARMUP,
                n_samples=N_SAMPLES,
                steps_per_sample=STEPS_PER_SAMPLE,
                schedule=AnnealingSchedule(beta_init=0.5, beta_final=beta),
                use_checkerboard=True,
            )
            # Warm up.
            sampler.sample(jrandom.PRNGKey(0), b_jax, J_jax, beta=beta)
            t0 = time.perf_counter()
            result = sampler.sample(jrandom.PRNGKey(1), b_jax, J_jax, beta=beta)
            _ = np.asarray(result)
            total += time.perf_counter() - t0

    return total


# ──────────────────────────────────────────────────────────────────────────────
# Run tests
# ──────────────────────────────────────────────────────────────────────────────


def run_tests() -> bool:
    """Run the LSB unit test suite. Returns True if all pass."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/python/test_lsb_sampler.py",
            "-v",
            "--tb=short",
            # pyproject.toml addopts injects --cov=python/carnot globally which
            # will always fail the project-wide coverage gate when running only
            # one test file. We pass -p no:cov to suppress that check entirely
            # and rely on the pass/fail count from the test run itself.
            # Override addopts to strip the project-wide --cov gate; we only
            # care about pass/fail for the LSB tests themselves here.
            "--override-ini=addopts=-v",
        ],
        capture_output=True,
        text=True,
    )
    # Print last portion for readability.
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)
    if result.stderr:
        print(result.stderr[-500:])
    # pytest exits 0 only when all tests pass; if coverage gate fires that
    # also sets a non-zero exit but "-p no:cov" disables it.
    return result.returncode == 0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("Experiment 983: Langevin Stochastic Boltzmann Sampler Benchmark")
    print("=" * 70)

    # 1. Convergence iterations.
    print("\n[1/4] Measuring convergence iterations (5 problems × 2 samplers)...")
    lsb_iters = []
    ising_iters = []
    for i, (biases, J) in enumerate(PROBLEMS):
        n = biases.shape[0]
        print(f"  Problem {i + 1} (n={n})...")
        lsb_it = count_convergence_steps(biases, J, "lsb", BETA, TARGET_ENERGY_FRACTION)
        ising_it = count_convergence_steps(biases, J, "ising", BETA, TARGET_ENERGY_FRACTION)
        lsb_iters.append(lsb_it)
        ising_iters.append(ising_it)
        print(f"    LSB: {lsb_it} steps,  Ising: {ising_it} steps")

    # 2. Wall-clock time.
    print("\n[2/4] Measuring wall-clock time (2 warmup runs each)...")
    lsb_time = time_sampler("lsb", BETA)
    ising_time = time_sampler("ising", BETA)
    print(f"  LSB wall time: {lsb_time:.4f}s")
    print(f"  Ising wall time: {ising_time:.4f}s")

    # 3. Compute speedup ratio.
    # speedup > 1 means LSB is faster (ising took longer).
    lsb_speedup_ratio = ising_time / max(lsb_time, 1e-9)
    print(f"  LSB speedup ratio: {lsb_speedup_ratio:.3f}x")

    # 4. Run tests.
    print("\n[3/4] Running unit tests...")
    unit_tests_pass = run_tests()
    print(f"  Tests passed: {unit_tests_pass}")

    # 5. Determine verdict.
    if lsb_speedup_ratio > 1.0:
        honest_verdict = "lsb_faster_set_default"
        lsb_set_as_default = True
    elif lsb_speedup_ratio >= 0.8:
        honest_verdict = "lsb_comparable_optional"
        lsb_set_as_default = False
    else:
        honest_verdict = "lsb_slower_optional"
        lsb_set_as_default = False

    print(f"\n[4/4] Verdict: {honest_verdict}")

    # Build result artifact.
    result = {
        "experiment": "exp983-langevin-sb-boltzmann-sampler",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": "1.0",
        "lsb_convergence_iterations": lsb_iters,
        "ising_convergence_iterations": ising_iters,
        "lsb_wall_time_s": round(lsb_time, 6),
        "ising_wall_time_s": round(ising_time, 6),
        "lsb_speedup_ratio": round(lsb_speedup_ratio, 4),
        "lsb_set_as_default": lsb_set_as_default,
        "unit_tests_pass": unit_tests_pass,
        "honest_verdict": honest_verdict,
        "benchmark_params": {
            "n_problems": 5,
            "problem_sizes": [p[0].shape[0] for p in PROBLEMS],
            "beta": BETA,
            "n_samples": N_SAMPLES,
            "n_warmup": N_WARMUP,
            "steps_per_sample": STEPS_PER_SAMPLE,
            "target_energy_fraction": TARGET_ENERGY_FRACTION,
            "lsb_lr": 0.05,
            "lsb_use_cem": True,
        },
        "reference": "arXiv 2512.02323 -- Langevin Stochastic Boltzmann sampling",
    }

    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nResult written to: {RESULT_PATH}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
