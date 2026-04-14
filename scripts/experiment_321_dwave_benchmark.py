#!/usr/bin/env python3
"""Experiment 321: D-Wave Neal vs CPU Ising Benchmark on Constraint Problems.

CONTEXT:
    Exp 320 introduced DWaveSampler as a quantum-inspired backend using D-Wave's
    Ocean SDK. This experiment benchmarks DWaveSampler(neal), DWaveSampler(tabu),
    and the existing CpuBackend (ParallelIsingSampler) on planted-solution Ising
    constraint problems at three scales: 128, 512, and 2048 spins.

    A "planted-solution" Ising problem encodes a known optimal spin configuration
    s* as the global energy minimum. This gives us a ground-truth reference so we
    can measure *success rate* (fraction of samples within a tolerance of the true
    optimum) in addition to raw energy and wall-clock time.

    **Why planted solutions are the right benchmark for Carnot:**
        In Carnot's verify-repair pipeline, each logical constraint maps to an
        Ising edge. Satisfying all constraints == reaching the ground state. By
        planting the optimal answer, we can measure whether each sampler reliably
        finds it — directly simulating Carnot's real workload.

    **Planted-solution construction:**
        1. Choose a random ground state s* ∈ {0,1}^n.
        2. Map to ±1: σ*_i = 2·s*_i − 1.
        3. Set biases b_i = B·σ*_i  (positive bias toward planted value).
        4. For randomly chosen sparse edge set E, set J_ij = C·σ*_i·σ*_j
           (positive coupling rewards alignment with planted spins).
        5. The ground state energy is E(s*) = −B·n − C·|E|
           (every bias and coupling term is at its minimum simultaneously).

    **Samplers compared:**
        - CpuBackend:      ParallelIsingSampler via JAX parallel Gibbs + annealing
        - DWaveSampler(neal): SimulatedAnnealingSampler from dwave-neal
        - DWaveSampler(tabu): TabuSampler (fast deterministic heuristic)

    **Hypothesis being tested:**
        Quantum-inspired annealing (Neal SA) finds better solutions than parallel
        Gibbs (CPU JAX) for constraint verification problems at ≥512 spins, at the
        cost of higher latency per sample.

Deliverable: results/experiment_321_results.json

Spec: REQ-SAMPLE-003, REQ-SAMPLE-007, SCENARIO-SAMPLE-007
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Repo root injection (scripts/ can import python/ without install)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

# ---------------------------------------------------------------------------
# Optional tabu shim: dwave-samplers >= 1.0 moved TabuSampler out of the
# standalone 'tabu' package into 'dwave.samplers'. The existing dwave_sampler.py
# still imports 'from tabu import TabuSampler'. This shim lets the legacy import
# work transparently when dwave-samplers is installed but the old tabu package
# is not.
# ---------------------------------------------------------------------------

if "tabu" not in sys.modules:
    try:
        from dwave.samplers import TabuSampler as _TabuSampler  # type: ignore[import-untyped]
        import types as _types

        _tabu_shim = _types.ModuleType("tabu")
        _tabu_shim.TabuSampler = _TabuSampler  # type: ignore[attr-defined]
        sys.modules["tabu"] = _tabu_shim
    except ImportError:
        pass  # Tabu will be unavailable; benchmark handles this gracefully.

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 321
TITLE: str = "D-Wave Neal vs CPU Ising Benchmark"
DEFAULT_OUTPUT: str = "results/experiment_321_results.json"

# Problem configurations (n_spins, density, n_steps, n_samples).
# Density controls sparsity of the random coupling graph:
#   128 spins:  0.20 density ≈  1270 edges  (moderately dense)
#   512 spins:  0.05 density ≈  6528 edges  (sparse)
#   2048 spins: 0.01 density ≈ 20971 edges  (very sparse)
PROBLEM_CONFIGS: list[dict[str, Any]] = [
    {"n_spins": 128,  "density": 0.20, "n_steps": 1000, "n_samples": 20},
    {"n_spins": 512,  "density": 0.05, "n_steps":  500, "n_samples": 10},
    {"n_spins": 2048, "density": 0.01, "n_steps":  200, "n_samples":  5},
]

# Number of independent random problems per size (for statistical stability).
N_TRIALS: int = 3

# Inverse temperature for annealing.
BETA: float = 10.0

# Bias strength in the planted problem (how strongly each spin is encouraged
# to match the planted state).
BIAS_STRENGTH: float = 2.0

# Coupling strength between aligned spin pairs in the planted graph.
COUPLING_STRENGTH: float = 1.0

# Success tolerance: a sample counts as "successful" if its energy is within
# TOLERANCE_FRAC of the ground state energy relative to the total energy scale.
TOLERANCE_FRAC: float = 0.05  # 5% of ground state energy depth


# ---------------------------------------------------------------------------
# Pure utility functions (testable without samplers)
# ---------------------------------------------------------------------------


def make_planted_ising_problem(
    n_spins: int,
    density: float,
    rng: np.random.Generator,
    bias_strength: float = BIAS_STRENGTH,
    coupling_strength: float = COUPLING_STRENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a planted-solution Ising problem with known global minimum.

    **Detailed explanation for engineers:**
        The Ising energy function is:
            E(s) = -sum_i b_i * s_i  -  sum_{i,j} J_ij * s_i * s_j
        We choose a random binary ground state s* ∈ {0,1}^n, then set
        biases and couplings so that every term in E(s*) is at its minimum.

        Ground state construction:
            sigma*_i = 2*s*_i - 1  (map {0,1} → {-1,+1})
            b_i     = B * sigma*_i  → contributes -B*|sigma*_i| = -B to E(s*)
            J_ij    = C * sigma*_i * sigma*_j  → contributes -C to E(s*) for edge (i,j)

        With all terms minimized simultaneously, E(s*) = -B*n - C*|E| is the
        global minimum of E(s). Any other configuration will have at least one
        unsatisfied term.

        The coupling graph is a random sparse Erdős-Rényi subgraph of the
        complete graph on n vertices. Upper-triangle indices are sampled without
        replacement at the requested density.

    Args:
        n_spins:          Number of spins (problem size).
        density:          Fraction of all n*(n-1)/2 possible edges to include.
                          Should be << 1 for large n to keep problems tractable.
        rng:              NumPy random Generator for reproducibility.
        bias_strength:    B — magnitude of per-spin bias toward planted state.
        coupling_strength: C — magnitude of inter-spin coupling.

    Returns:
        Tuple of (biases, couplings, planted_state, ground_energy) where:
            biases:        shape (n_spins,), float64
            couplings:     shape (n_spins, n_spins), float64, symmetric, zero-diag
            planted_state: shape (n_spins,), bool — the global minimum
            ground_energy: float — energy at planted_state (guaranteed global min)
    """
    # Step 1: Choose random planted state.
    planted = rng.integers(0, 2, size=n_spins).astype(bool)
    sigma_star = 2.0 * planted.astype(np.float64) - 1.0  # ±1 encoding

    # Step 2: Biases aligned with planted state.
    biases = bias_strength * sigma_star

    # Step 3: Sparse coupling graph.
    # Enumerate all upper-triangle (i, j) pairs and sample without replacement.
    row_idx, col_idx = np.triu_indices(n_spins, k=1)
    n_possible = len(row_idx)
    n_edges = max(1, int(n_possible * density))
    chosen = rng.choice(n_possible, size=n_edges, replace=False)

    # Step 4: Set coupling values for chosen edges.
    couplings = np.zeros((n_spins, n_spins), dtype=np.float64)
    ii, jj = row_idx[chosen], col_idx[chosen]
    # J_ij = C * sigma*_i * sigma*_j: positive if spins aligned in planted state.
    vals = coupling_strength * sigma_star[ii] * sigma_star[jj]
    couplings[ii, jj] = vals
    couplings[jj, ii] = vals  # symmetric

    # Step 5: Compute exact ground state energy.
    # E(s*) = -b·s* - s*^T J s*
    ground_energy = float(
        -biases @ planted.astype(np.float64)
        - planted.astype(np.float64) @ couplings @ planted.astype(np.float64)
    )

    return biases, couplings, planted, ground_energy


def compute_ising_energies(
    biases: np.ndarray,
    couplings: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Compute Ising energy for each sample in a batch.

    **Detailed explanation for engineers:**
        The Ising energy for a single spin configuration s ∈ {0,1}^n is:
            E(s) = -b·s - s^T J s

        For a batch of samples (each row is one configuration), we vectorize
        by computing the full matrix of local fields: samples @ J, then
        element-wise multiply by samples and sum. This is O(n_samples * n^2)
        but benefits from BLAS-level matrix multiply performance.

    Args:
        biases:    Bias vector, shape (n_spins,).
        couplings: Symmetric coupling matrix, shape (n_spins, n_spins).
        samples:   Boolean array, shape (n_samples, n_spins).

    Returns:
        Energy vector, shape (n_samples,).
    """
    s = np.asarray(samples, dtype=np.float64)
    # Local field for each sample: (n_samples, n_spins)
    local_fields = s @ couplings
    # Quadratic energy term: sum_j J_ij * s_j for each i, then dot with s_i
    quadratic = np.sum(local_fields * s, axis=1)
    # Linear energy term
    linear = s @ biases
    return -linear - quadratic


def compute_success_rate(
    energies: np.ndarray,
    ground_energy: float,
    tolerance_frac: float = TOLERANCE_FRAC,
) -> float:
    """Fraction of samples within tolerance_frac of the ground state energy.

    **Detailed explanation for engineers:**
        "Success" means the sample's energy is within a relative tolerance of
        the planted ground state. The tolerance is expressed as a fraction of
        the energy gap from zero to the ground state:
            threshold = ground_energy * (1 - tolerance_frac)
        Since ground_energy < 0, multiplying by (1 - frac) makes it slightly
        less negative — i.e., we accept samples up to frac*|ground_energy|
        above the true minimum.

        Example: ground_energy = -100, tolerance_frac = 0.05
            → threshold = -100 * (1 - 0.05) = -95
            → samples with E ≤ -95 count as successful

    Args:
        energies:       Array of sample energies, shape (n_samples,).
        ground_energy:  Energy at the planted ground state (negative).
        tolerance_frac: Fraction of ground state depth to accept as "success".

    Returns:
        Float in [0, 1]: fraction of samples meeting the success criterion.
    """
    if ground_energy >= 0.0:
        # Degenerate case (should not happen with B,C > 0).
        return float(np.mean(energies <= 0.0))
    threshold = ground_energy * (1.0 - tolerance_frac)
    return float(np.mean(energies <= threshold))


def aggregate_trial_results(
    all_energies: list[np.ndarray],
    ground_energy: float,
    elapsed_list: list[float],
    tolerance_frac: float = TOLERANCE_FRAC,
) -> dict[str, Any]:
    """Combine results from multiple trials into summary statistics.

    **Detailed explanation for engineers:**
        Pools all samples from all trials to compute aggregate statistics.
        Per-trial timing is averaged since each trial runs the same workload.

        mean_best_energy: average of the best (lowest) energy found per trial
        std_best_energy:  standard deviation of best energy across trials
        mean_energy:      average energy across ALL samples in ALL trials
        std_energy:       standard deviation across all pooled samples
        success_rate:     fraction of ALL pooled samples near ground state
        mean_time_s:      mean wall-clock time per trial (seconds)
        n_total_samples:  total samples collected across all trials

    Args:
        all_energies:   List of energy arrays, one per trial.
        ground_energy:  Energy at the planted ground state.
        elapsed_list:   Wall-clock seconds per trial.
        tolerance_frac: Success tolerance fraction.

    Returns:
        Dict with summary statistics.
    """
    pooled = np.concatenate(all_energies)
    best_per_trial = np.array([float(e.min()) for e in all_energies])

    return {
        "mean_best_energy": float(np.mean(best_per_trial)),
        "std_best_energy": float(np.std(best_per_trial)),
        "mean_energy": float(np.mean(pooled)),
        "std_energy": float(np.std(pooled)),
        "success_rate": compute_success_rate(pooled, ground_energy, tolerance_frac),
        "mean_time_s": float(np.mean(elapsed_list)),
        "n_total_samples": int(len(pooled)),
    }


# ---------------------------------------------------------------------------
# Sampler runner
# ---------------------------------------------------------------------------


def run_one_sampler(
    sampler: Any,
    biases: np.ndarray,
    couplings: np.ndarray,
    n_samples: int,
    n_steps: int,
    beta: float,
) -> tuple[np.ndarray, float]:
    """Run a single sampler call and return (samples_bool, elapsed_seconds).

    **Detailed explanation for engineers:**
        Both DWaveSampler and CpuBackend implement the SamplerBackend protocol
        with the same minimize_energy signature, so this function works for all
        three backends (cpu, neal, tabu). The returned samples are converted to
        numpy bool arrays for uniform downstream processing.

    Args:
        sampler:   Any object with minimize_energy(biases, couplings, n_samples,
                   n_steps, beta) -> bool array.
        biases:    Bias vector, shape (n_spins,).
        couplings: Symmetric coupling matrix, shape (n_spins, n_spins).
        n_samples: Number of independent samples to draw.
        n_steps:   Annealing/sweep steps.
        beta:      Final inverse temperature.

    Returns:
        (samples, elapsed_seconds) where samples is bool array (n_samples, n_spins).
    """
    t0 = time.perf_counter()
    raw = sampler.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    elapsed = time.perf_counter() - t0

    # Normalize to numpy bool (CpuBackend returns JAX DeviceArray).
    samples = np.asarray(raw, dtype=bool)
    return samples, elapsed


def benchmark_problem_size(
    cfg: dict[str, Any],
    seed: int,
    samplers: dict[str, Any],
) -> dict[str, Any]:
    """Benchmark all samplers on N_TRIALS random problems of one size.

    **Detailed explanation for engineers:**
        For each trial, a fresh planted-solution problem is generated with a
        unique seed. Every sampler is run on the same problem instance in the
        same trial. This controls for problem difficulty variation: we compare
        the samplers on identical inputs.

        The first trial for CpuBackend may include JAX JIT compilation overhead.
        This is noted in the results metadata. Users should inspect per-trial
        timing to see the JIT warm-up effect.

    Args:
        cfg:      Problem config dict (n_spins, density, n_steps, n_samples).
        seed:     Base random seed; trial k uses seed + k.
        samplers: Dict mapping name -> sampler object.

    Returns:
        Dict with problem metadata and per-sampler summary statistics.
    """
    n_spins = cfg["n_spins"]
    density = cfg["density"]
    n_steps = cfg["n_steps"]
    n_samples = cfg["n_samples"]

    per_sampler_energies: dict[str, list[np.ndarray]] = {name: [] for name in samplers}
    per_sampler_elapsed: dict[str, list[float]] = {name: [] for name in samplers}

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(seed + trial)
        biases, couplings, _planted, ground_energy = make_planted_ising_problem(
            n_spins, density, rng
        )

        for name, sampler in samplers.items():
            try:
                samples, elapsed = run_one_sampler(
                    sampler, biases, couplings, n_samples, n_steps, BETA
                )
                energies = compute_ising_energies(biases, couplings, samples)
                per_sampler_energies[name].append(energies)
                per_sampler_elapsed[name].append(elapsed)
                print(
                    f"  n={n_spins} trial={trial+1}/{N_TRIALS} {name}: "
                    f"best_E={energies.min():.2f} (ground={ground_energy:.2f}) "
                    f"t={elapsed:.3f}s"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  n={n_spins} trial={trial+1}/{N_TRIALS} {name}: ERROR {exc}")
                per_sampler_energies[name].append(np.array([0.0]))
                per_sampler_elapsed[name].append(float("nan"))

    # Compute ground energy for the last problem (representative).
    # Recompute ground energy for reporting: use trial 0 for consistency.
    rng0 = np.random.default_rng(seed)
    _b, _J, _p, representative_ground_energy = make_planted_ising_problem(
        n_spins, density, rng0
    )

    sampler_summaries: dict[str, dict[str, Any]] = {}
    for name in samplers:
        sampler_summaries[name] = aggregate_trial_results(
            per_sampler_energies[name],
            representative_ground_energy,
            per_sampler_elapsed[name],
        )

    # Determine which sampler found the best (lowest mean best energy).
    winner = min(sampler_summaries, key=lambda k: sampler_summaries[k]["mean_best_energy"])

    return {
        "n_spins": n_spins,
        "density": density,
        "n_steps": n_steps,
        "n_samples_per_trial": n_samples,
        "n_trials": N_TRIALS,
        "beta": BETA,
        "representative_ground_energy": representative_ground_energy,
        "samplers": sampler_summaries,
        "best_sampler_by_energy": winner,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_path: str = DEFAULT_OUTPUT) -> None:
    """Run the full D-Wave vs CPU benchmark and write results JSON.

    **Detailed explanation for engineers:**
        1. Attempt to build all three sampler backends. If a backend is
           unavailable (import error), it is skipped and noted in metadata.
        2. For each problem size (128, 512, 2048 spins), run N_TRIALS random
           planted problems and record per-sampler statistics.
        3. Write a results JSON with the same schema as other Carnot experiments
           (experiment, title, run_date, schema, metadata, results, conclusion).
        4. Print a summary table showing energy gap ratio and speed comparison.
    """
    print(f"Experiment {EXPERIMENT}: {TITLE}")
    print("=" * 60)

    # --- Build samplers ---
    from carnot.samplers.backend import CpuBackend
    from carnot.samplers.dwave_sampler import DWaveSampler

    samplers: dict[str, Any] = {}

    cpu = CpuBackend(seed=42)
    samplers["cpu"] = cpu
    print(f"[OK] CpuBackend (ParallelIsingSampler / JAX)")

    try:
        neal_sampler = DWaveSampler(mode="neal")
        samplers["dwave_neal"] = neal_sampler
        print(f"[OK] DWaveSampler(neal) — local SimulatedAnnealingSampler")
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] DWaveSampler(neal) unavailable: {exc}")

    try:
        tabu_sampler = DWaveSampler(mode="tabu")
        samplers["dwave_tabu"] = tabu_sampler
        print(f"[OK] DWaveSampler(tabu) — local TabuSampler")
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] DWaveSampler(tabu) unavailable: {exc}")

    print(f"\nRunning {N_TRIALS} trials per size with beta={BETA}, "
          f"tolerance={TOLERANCE_FRAC*100:.0f}% of ground energy\n")

    # --- Run benchmarks ---
    results: list[dict[str, Any]] = []
    base_seed = 321_000

    for cfg in PROBLEM_CONFIGS:
        n = cfg["n_spins"]
        print(f"--- n_spins={n} (density={cfg['density']}, "
              f"n_steps={cfg['n_steps']}, n_samples={cfg['n_samples']}) ---")
        row = benchmark_problem_size(cfg, seed=base_seed + n, samplers=samplers)
        results.append(row)
        print()

    # --- Build conclusion ---
    conclusion = _build_conclusion(results)
    print("=== CONCLUSION ===")
    print(conclusion["summary"])

    # --- Write artifact ---
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": TITLE,
        "run_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"),
        "schema": {"artifact": "carnot.dwave_cpu_benchmark.v1"},
        "metadata": {
            "n_trials": N_TRIALS,
            "beta": BETA,
            "bias_strength": BIAS_STRENGTH,
            "coupling_strength": COUPLING_STRENGTH,
            "tolerance_frac": TOLERANCE_FRAC,
            "samplers_tested": list(samplers.keys()),
            "note": (
                "All backends use minimize_energy() with a linear annealing schedule "
                "from beta=0.1 to beta=10.0. CpuBackend first-trial timing includes "
                "JAX JIT compilation overhead. Tabu requires dwave-samplers or the "
                "standalone tabu package."
            ),
        },
        "results": results,
        "conclusion": conclusion,
    }

    out_path = Path(_REPO_ROOT) / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nResults written to {out_path}")


def _build_conclusion(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize which sampler won at each size and overall.

    **Detailed explanation for engineers:**
        We compare samplers on two dimensions:
        1. Solution quality: mean_best_energy / representative_ground_energy
           (ratio closer to 1.0 == better; ground energy is negative so
           we want this ratio to be ≥ 0.95 for 5% slack)
        2. Speed: mean_time_s (lower is better)

        The "energy gap ratio" = mean_best_energy / ground_energy measures how
        close the best found sample is to the planted optimal. A ratio of 1.0
        means the ground state was found in every trial.
    """
    per_size_winners = []
    best_quality_overall: dict[str, float] = {}

    for row in results:
        n = row["n_spins"]
        ge = row["representative_ground_energy"]
        samp = row["samplers"]

        size_info: dict[str, Any] = {"n_spins": n}
        for name, stats in samp.items():
            ratio = stats["mean_best_energy"] / ge if ge != 0.0 else float("nan")
            size_info[name] = {
                "energy_gap_ratio": round(ratio, 4),
                "success_rate": round(stats["success_rate"], 4),
                "mean_time_s": round(stats["mean_time_s"], 4),
            }
            best_quality_overall[name] = best_quality_overall.get(name, 0.0) + ratio

        per_size_winners.append(size_info)

    # Overall quality winner: highest average energy_gap_ratio across all sizes.
    # (Higher ratio = sampler found states closer to the planted minimum.)
    n_sizes = len(results)
    if n_sizes > 0 and best_quality_overall:
        # Normalize by number of sizes so absent samplers don't rank high.
        available = [k for k in best_quality_overall if n_sizes > 0]
        quality_winner = max(available, key=lambda k: best_quality_overall[k] / n_sizes)
    else:
        quality_winner = "unknown"

    # Speed winner: sampler with lowest mean_time_s averaged over all sizes
    # (only for samplers that appeared in all size tiers).
    speed_totals: dict[str, float] = {}
    speed_counts: dict[str, int] = {}
    for row in results:
        for name, stats in row["samplers"].items():
            t = stats["mean_time_s"]
            if not (t != t):  # skip NaN
                speed_totals[name] = speed_totals.get(name, 0.0) + t
                speed_counts[name] = speed_counts.get(name, 0) + 1
    speed_winner = min(
        speed_totals, key=lambda k: speed_totals[k] / speed_counts[k], default="unknown"
    )

    lines = [
        f"Quality winner (closest to ground state): {quality_winner}",
        f"Speed winner (lowest mean time per trial): {speed_winner}",
    ]
    for row in per_size_winners:
        n = row["n_spins"]
        for name, stats in row.items():
            if name == "n_spins":
                continue
            lines.append(
                f"  n={n} {name}: "
                f"energy_gap_ratio={stats['energy_gap_ratio']:.4f}, "
                f"success_rate={stats['success_rate']:.2%}, "
                f"time={stats['mean_time_s']:.3f}s"
            )

    return {
        "quality_winner": quality_winner,
        "speed_winner": speed_winner,
        "per_size": per_size_winners,
        "summary": "\n".join(lines),
    }


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT
    main(output)
