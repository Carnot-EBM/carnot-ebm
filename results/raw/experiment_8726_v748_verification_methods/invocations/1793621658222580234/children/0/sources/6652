#!/usr/bin/env python3
"""Experiment 1149: KV260 v5 DC-continuous Ising diagnostic.

Exp 1134 found that KV260 v4 sparse parallel Glauber improved over the
earlier FPGA mismatch but still missed the KL < 0.05 gate. This script tests
the next candidate in software first: relax spins to continuous [-1, +1],
run a DC/proximal update on the Ising energy, threshold back to {-1, +1},
and compare the resulting empirical distribution against exact CPU Gibbs on
small N=8 matrices.

Spec: REQ-HW-042, SCENARIO-HW-042.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLE = REPO_ROOT / "results" / "experiment_1149_kv260_v5_dc_continuous_diagnostic.json"
EXP1134_ARTIFACT = REPO_ROOT / "results" / "experiment_1134_kv260_v4_parameter_tuning.json"
SAMPLER_SIM_PATH = REPO_ROOT / "python" / "carnot" / "hardware" / "sampler_sim.py"
V4_SPEC_PATH = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v4_spec.md"

EXPERIMENT_ID = 1149
TITLE = "KV260 Ising v5 — DC-Continuous Relaxation Diagnostic (Software Parity)"
ALGORITHM = "dc_continuous_relaxation"

N_SPINS = 8
K_NEIGHBORS = 2
BETA = 2.0
KL_THRESHOLD = 0.05

KL_V4_BEST_PRIOR = 0.1128
KL_V4_WITH_SELF_ADAPTIVE_PRIOR = 31.893

MATRIX_SEEDS = (1134, 1135, 1136)
ALPHA_GRID = (0.01, 0.02, 0.05, 0.1)
DEFAULT_N_RESTARTS = 4096
DEFAULT_MAX_ITER = 300
DEFAULT_TOLERANCE = 1e-7
V4_N_RECORD = 4096
V4_BURN_IN_SWEEPS = 250

HONEST_VERDICTS = {
    "kl_below_threshold_v5_viable",
    "kl_improved_not_threshold",
    "kl_unchanged_topology_wall",
    "algorithm_diverged",
}

REQUIRED_ARTIFACT_FIELDS = {
    "kl_v4_best_prior",
    "kl_v4_with_self_adaptive_prior",
    "algorithm",
    "kl_v5_best",
    "kl_v5_below_threshold",
    "kl_improvement_over_v4",
    "energy_time_accuracy_reported",
    "n_j_matrices_tested",
    "kv260_v5_diagnostic_complete",
    "rtl_recommendation",
    "honest_verdict",
}


@dataclass(frozen=True)
class DCConfig:
    """Numerical settings for one vectorized DC-continuous restart batch."""

    n_restarts: int = DEFAULT_N_RESTARTS
    max_iter: int = DEFAULT_MAX_ITER
    alpha: float = 0.05
    tolerance: float = DEFAULT_TOLERANCE
    seed: int = 1149


def _utc_now_iso() -> str:  # pragma: no cover - exercised by the script entrypoint
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_v4_prior_artifact(path: Path = EXP1134_ARTIFACT) -> dict[str, Any]:
    """Read the exp1134 artifact so the v5 comparison is anchored to v4."""
    return json.loads(path.read_text())


def _load_sampler_sim():
    """Import the existing pure-NumPy v4 simulator by file path."""
    spec = importlib.util.spec_from_file_location("sampler_sim", SAMPLER_SIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sampler_sim", mod)
    spec.loader.exec_module(mod)
    return mod


def all_spin_states(n_spins: int) -> np.ndarray:
    """Enumerate every {-1,+1} configuration using bit i as spin i."""
    states = np.empty((2**n_spins, n_spins), dtype=np.int8)
    for idx in range(2**n_spins):
        states[idx] = [1 if (idx >> i) & 1 else -1 for i in range(n_spins)]
    return states


def dc_spectral_split(j_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return PSD matrices (J_plus, J_minus) such that J = J_plus - J_minus."""
    j_sym = np.asarray(j_matrix, dtype=np.float64)
    j_sym = 0.5 * (j_sym + j_sym.T)
    eigenvalues, eigenvectors = np.linalg.eigh(j_sym)
    j_plus = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    j_minus = (eigenvectors * np.clip(-eigenvalues, 0.0, None)) @ eigenvectors.T
    return 0.5 * (j_plus + j_plus.T), 0.5 * (j_minus + j_minus.T)


def dc_proximal_step(
    state: np.ndarray,
    j_plus: np.ndarray,
    j_minus: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Run one clipped DC/proximal step for E(s) = -0.5 * s.T @ J @ s.

    With J = J_plus - J_minus, the algebraically consistent DC step for the
    Ising energy is equivalent to `s + alpha * J @ s`, written through the
    split as `s - alpha * J_minus @ s + alpha * J_plus @ s`.
    """
    state_f = np.asarray(state, dtype=np.float64)
    updated = state_f - alpha * (j_minus @ state_f) + alpha * (j_plus @ state_f)
    return np.clip(updated, -1.0, 1.0)


def threshold_spins(state: np.ndarray) -> np.ndarray:
    """Threshold continuous spins to {-1,+1}, mapping exact zero to +1."""
    return np.where(np.asarray(state) >= 0.0, 1, -1).astype(np.int8)


def ising_energy(j_matrix: np.ndarray, spins: np.ndarray) -> float:
    """Compute E(s) = -0.5 * s.T @ J @ s for one spin vector."""
    s = np.asarray(spins, dtype=np.float64).ravel()
    return float(-0.5 * s @ np.asarray(j_matrix, dtype=np.float64) @ s)


def constraint_satisfaction_accuracy(j_matrix: np.ndarray, spins: np.ndarray) -> float:
    """Return percent of non-zero pairwise Ising constraints satisfied."""
    j = np.asarray(j_matrix, dtype=np.float64)
    s = np.asarray(spins, dtype=np.float64).ravel()
    upper_i, upper_j = np.triu_indices_from(j, k=1)
    weights = j[upper_i, upper_j]
    active = np.abs(weights) > 1e-12
    if not np.any(active):
        return 100.0
    satisfied = weights[active] * s[upper_i[active]] * s[upper_j[active]] > 0.0
    return float(100.0 * np.mean(satisfied))


def configurations_to_indices(configs: np.ndarray) -> np.ndarray:
    """Pack a (T, N) spin array into integer state indices."""
    arr = np.asarray(configs)
    weights = np.array([1 << i for i in range(arr.shape[1])], dtype=np.int64)
    return (arr > 0).astype(np.int64) @ weights


def cpu_gibbs_distribution(j_matrix: np.ndarray, beta: float = BETA) -> np.ndarray:
    """Compute exact Boltzmann probabilities over all 2**N configurations."""
    n_spins = np.asarray(j_matrix).shape[0]
    if n_spins > 20:
        raise ValueError("exact CPU Gibbs enumeration is limited to N<=20")  # pragma: no cover
    states = all_spin_states(n_spins).astype(np.float64)
    energies = -0.5 * np.einsum("bi,ij,bj->b", states, j_matrix, states)
    log_probs = -beta * energies
    log_probs -= float(np.max(log_probs))
    probs = np.exp(log_probs)
    return probs / probs.sum()


def kl_against_cpu_gibbs(samples: np.ndarray, j_matrix: np.ndarray, beta: float = BETA) -> float:
    """Compute KL(empirical samples || exact CPU Gibbs) with Laplace smoothing."""
    n_spins = np.asarray(j_matrix).shape[0]
    n_states = 2**n_spins
    indices = configurations_to_indices(samples)
    counts = np.bincount(indices, minlength=n_states).astype(np.float64)
    empirical = (counts + 1.0) / (counts.sum() + n_states)
    gibbs = np.clip(cpu_gibbs_distribution(j_matrix, beta), 1e-300, None)
    return float(np.sum(empirical * np.log(empirical / gibbs)))


def build_exp1134_seeded_j_matrices(
    n_spins: int = N_SPINS,
    seeds: tuple[int, ...] = MATRIX_SEEDS,
) -> list[np.ndarray]:
    """Build three deterministic K=2 signed-ring J matrices from exp1134 seeds.

    Exp 1134 did not persist raw J matrices in its artifact. To keep this
    diagnostic reproducible and tied to the prior run, we use its experiment seed
    family to generate three signed weighted rings with the same small-N K=2 shape
    used by exp1122/1134.
    """
    matrices: list[np.ndarray] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_spins)
        magnitudes = rng.uniform(0.75, 1.25, size=n_spins)
        j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
        for i in range(n_spins):
            j = (i + 1) % n_spins
            value = float(signs[i] * magnitudes[i])
            j_matrix[i, j] = value
            j_matrix[j, i] = value
        matrices.append(j_matrix)
    return matrices


def sparse_tables_from_j(
    j_matrix: np.ndarray,
    *,
    k_neighbors: int = K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a fixed-degree sparse dense J matrix into v4 neighbour tables."""
    j = np.asarray(j_matrix, dtype=np.float64)
    nbr_idx = np.zeros((j.shape[0], k_neighbors), dtype=np.int32)
    j_sparse = np.zeros((j.shape[0], k_neighbors), dtype=np.float64)
    for row in range(j.shape[0]):
        cols = np.flatnonzero(np.abs(j[row]) > 1e-12)
        if len(cols) != k_neighbors:
            raise ValueError(
                "each row must have exactly k_neighbors non-zero entries"
            )  # pragma: no cover
        cols = np.sort(cols)
        nbr_idx[row] = cols
        j_sparse[row] = j[row, cols]
    return nbr_idx, j_sparse


def dense_from_sparse_tables(nbr_idx: np.ndarray, j_sparse: np.ndarray) -> np.ndarray:
    """Reconstruct a dense J matrix from neighbour index and value tables."""
    dense = np.zeros((nbr_idx.shape[0], nbr_idx.shape[0]), dtype=np.float64)
    for row in range(nbr_idx.shape[0]):
        dense[row, nbr_idx[row]] = j_sparse[row]
    return dense


def _dc_batch_step(
    states: np.ndarray,
    j_plus: np.ndarray,
    j_minus: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Vectorized DC step for a batch of restart states."""
    j_effective = j_plus - j_minus
    return np.clip(states + alpha * (states @ j_effective.T), -1.0, 1.0)


def _mean_energy(j_matrix: np.ndarray, samples: np.ndarray) -> float:
    """Mean Ising energy over a batch of thresholded samples."""
    samples_f = np.asarray(samples, dtype=np.float64)
    energies = -0.5 * np.einsum("bi,ij,bj->b", samples_f, j_matrix, samples_f)
    return float(np.mean(energies))


def _mean_constraint_accuracy(j_matrix: np.ndarray, samples: np.ndarray) -> float:
    """Mean constraint-satisfaction percent over thresholded samples."""
    return float(np.mean([constraint_satisfaction_accuracy(j_matrix, row) for row in samples]))


def run_dc_measurement(
    j_matrix: np.ndarray,
    *,
    matrix_id: str,
    config: DCConfig,
    beta: float = BETA,
) -> dict[str, Any]:
    """Run vectorized DC-continuous restarts and return KL + EDDP metrics."""
    start = time.time()
    j_plus, j_minus = dc_spectral_split(j_matrix)
    rng = np.random.default_rng(config.seed)
    states = rng.uniform(-1.0, 1.0, size=(config.n_restarts, j_matrix.shape[0]))
    converged = np.zeros(config.n_restarts, dtype=bool)
    iterations = np.full(config.n_restarts, config.max_iter, dtype=np.int32)

    for iteration in range(1, config.max_iter + 1):
        updated = _dc_batch_step(states, j_plus, j_minus, config.alpha)
        deltas = np.max(np.abs(updated - states), axis=1)
        newly_converged = (~converged) & (deltas < config.tolerance)
        iterations[newly_converged] = iteration
        converged |= newly_converged
        states = updated
        if bool(np.all(converged)):
            break

    samples = threshold_spins(states)
    wall_clock = time.time() - start
    return {
        "matrix_id": matrix_id,
        "alpha": float(config.alpha),
        "kl_v5_vs_cpu_gibbs": kl_against_cpu_gibbs(samples, j_matrix, beta),
        "energy_at_convergence": _mean_energy(j_matrix, samples),
        "wall_clock_s": round(wall_clock, 6),
        "final_accuracy": _mean_constraint_accuracy(j_matrix, samples),
        "convergence_fraction": float(np.mean(converged)),
        "iterations_mean": float(np.mean(iterations)),
        "n_restarts": int(config.n_restarts),
        "samples": samples,
    }


def measurement_for_artifact(measurement: dict[str, Any]) -> dict[str, Any]:
    """Drop NumPy sample arrays so a measurement can be JSON-serialized."""
    return {key: value for key, value in measurement.items() if key != "samples"}


def run_v5_alpha_grid(
    matrices: list[np.ndarray],
    *,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    base_config: DCConfig = DCConfig(),
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    """Sweep DC alpha and return best alpha, best measurements, and summary rows."""
    best_alpha = alpha_grid[0]
    best_measurements: list[dict[str, Any]] = []
    per_alpha_summary: list[dict[str, Any]] = []
    best_mean_kl = float("inf")
    for alpha_idx, alpha in enumerate(alpha_grid):
        measurements = [
            run_dc_measurement(
                j_matrix,
                matrix_id=f"exp1134_seeded_j{matrix_idx}",
                config=DCConfig(
                    n_restarts=base_config.n_restarts,
                    max_iter=base_config.max_iter,
                    alpha=alpha,
                    tolerance=base_config.tolerance,
                    seed=base_config.seed + alpha_idx * 100 + matrix_idx,
                ),
            )
            for matrix_idx, j_matrix in enumerate(matrices)
        ]
        mean_kl = float(np.mean([m["kl_v5_vs_cpu_gibbs"] for m in measurements]))
        per_alpha_summary.append(
            {
                "alpha": float(alpha),
                "mean_kl_v5": mean_kl,
                "mean_energy_at_convergence": float(
                    np.mean([m["energy_at_convergence"] for m in measurements])
                ),
                "mean_final_accuracy": float(np.mean([m["final_accuracy"] for m in measurements])),
            }
        )
        if mean_kl < best_mean_kl:
            best_mean_kl = mean_kl
            best_alpha = alpha
            best_measurements = measurements
    return float(best_alpha), best_measurements, per_alpha_summary


def run_v4_sparse_measurement(
    j_matrix: np.ndarray,
    *,
    n_record: int = V4_N_RECORD,
    burn_in: int = V4_BURN_IN_SWEEPS,
    seed: int = 1149,
) -> dict[str, Any]:
    """Run the existing v4 sparse inertia simulator on one J matrix."""
    sim_mod = _load_sampler_sim()
    problem = sim_mod.IsingProblem(
        n_spins=j_matrix.shape[0],
        J=np.asarray(j_matrix, dtype=np.float64),
        h=np.zeros(j_matrix.shape[0], dtype=np.float64),
        beta=BETA,
    )
    nbr_idx, j_sparse = sparse_tables_from_j(j_matrix, k_neighbors=K_NEIGHBORS)
    sampler = sim_mod.SparseInertiaIsingSamplerV4(
        n_spins=j_matrix.shape[0],
        k_neighbors=K_NEIGHBORS,
        alpha_ema=0.1,
        beta_temperature=BETA,
        seed=seed,
        mode="stochastic",
    )
    start = time.time()
    samples = sampler.sample(
        nbr_idx=nbr_idx, j_sparse=j_sparse, n_steps=n_record, burn_in_sweeps=burn_in
    )
    return {
        "kl_v4_sparse_vs_cpu_gibbs": float(sim_mod.kl_against_true_gibbs(samples, problem)),
        "wall_clock_s": round(time.time() - start, 6),
        "n_record": int(n_record),
        "burn_in_sweeps": int(burn_in),
    }


def energy_time_accuracy_is_reported(measurements: list[dict[str, Any]]) -> bool:
    """Return True iff every v5 measurement carries the EDDP metric trio."""
    required = {"energy_at_convergence", "wall_clock_s", "final_accuracy"}
    return all(required <= set(measurement) for measurement in measurements)


def classify_verdict(
    kl_v5_best: float,
    *,
    diverged: bool,
    kl_v4_best_prior: float = KL_V4_BEST_PRIOR,
    kl_threshold: float = KL_THRESHOLD,
) -> str:
    """Map the v5 KL result to the allowed honest verdict vocabulary."""
    if diverged or not np.isfinite(kl_v5_best):
        return "algorithm_diverged"
    if kl_v5_best < kl_threshold:
        return "kl_below_threshold_v5_viable"
    if kl_v5_best < kl_v4_best_prior:
        return "kl_improved_not_threshold"
    return "kl_unchanged_topology_wall"


def rtl_recommendation_for_verdict(verdict: str) -> str:
    """Produce the v5 RTL recommendation implied by the diagnostic verdict."""
    if verdict == "kl_below_threshold_v5_viable":
        return (
            "Implement DC-continuous v5 RTL as fixed-point spectral-transform/proximal-update "
            "pipeline with sign threshold readout; keep CPU Gibbs parity tests as the acceptance gate."
        )
    if verdict == "kl_improved_not_threshold":
        return (
            "Prototype v5 RTL only with a stochastic readout or Metropolis correction after the "
            "DC-continuous step; deterministic thresholding improved KL but still misses the gate."
        )
    if verdict == "kl_unchanged_topology_wall":
        return (
            "Keep v5 in software research; use sequential Gibbs for KL-correct RTL until a "
            "DC-continuous design includes a Boltzmann-correct stochastic correction layer."
        )
    return "Do not implement v5 RTL from this run; first fix the DC update stability diagnostic."


def build_artifact(
    *,
    best_alpha: float,
    best_measurements: list[dict[str, Any]],
    per_alpha_summary: list[dict[str, Any]],
    v4_measurements: list[dict[str, Any]],
    duration_s: float,
    run_date: str,
) -> dict[str, Any]:
    """Build the JSON-serializable exp1149 artifact."""
    serial_measurements = [measurement_for_artifact(m) for m in best_measurements]
    kl_v5_best = float(np.mean([m["kl_v5_vs_cpu_gibbs"] for m in best_measurements]))
    diverged = any(not np.isfinite(m["kl_v5_vs_cpu_gibbs"]) for m in best_measurements)
    verdict = classify_verdict(kl_v5_best, diverged=diverged)
    artifact = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "schema": "kv260_v5_dc_continuous_diagnostic_v1",
        "run_date": run_date,
        "algorithm": ALGORITHM,
        "kl_threshold": KL_THRESHOLD,
        "kl_v4_best_prior": KL_V4_BEST_PRIOR,
        "kl_v4_with_self_adaptive_prior": KL_V4_WITH_SELF_ADAPTIVE_PRIOR,
        "kl_v5_best": kl_v5_best,
        "kl_v5_below_threshold": bool(kl_v5_best < KL_THRESHOLD),
        "kv260_v5_kl_below_threshold": bool(kl_v5_best < KL_THRESHOLD),
        "kl_improvement_over_v4": float(KL_V4_BEST_PRIOR - kl_v5_best),
        "best_alpha": float(best_alpha),
        "n_j_matrices_tested": len(best_measurements),
        "n_spins": N_SPINS,
        "k_neighbors_small_n": K_NEIGHBORS,
        "v4_spec_topology": "KV260 v4 target N=128, K=16; exp1134 parity used N=8, K=2",
        "j_matrix_source": (
            "exp1134 artifact did not persist raw matrices; exp1149 uses three deterministic "
            "signed weighted K=2 ring matrices generated from seeds 1134, 1135, 1136."
        ),
        "per_alpha_summary": per_alpha_summary,
        "per_matrix_v5": serial_measurements,
        "per_matrix_v4_sparse": v4_measurements,
        "kl_v4_sparse_measured_mean": float(
            np.mean([m["kl_v4_sparse_vs_cpu_gibbs"] for m in v4_measurements])
        ),
        "energy_at_convergence": float(
            np.mean([m["energy_at_convergence"] for m in best_measurements])
        ),
        "wall_clock_s": float(np.sum([m["wall_clock_s"] for m in best_measurements])),
        "final_accuracy": float(np.mean([m["final_accuracy"] for m in best_measurements])),
        "energy_time_accuracy_reported": energy_time_accuracy_is_reported(best_measurements),
        "kv260_v5_diagnostic_complete": True,
        "honest_verdict": verdict,
        "rtl_recommendation": rtl_recommendation_for_verdict(verdict),
        "v4_spec_path": str(V4_SPEC_PATH.relative_to(REPO_ROOT)),
        "duration_s": round(duration_s, 3),
    }
    return artifact


def write_artifact(artifact: dict[str, Any], path: Path = DELIVERABLE) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main() -> int:  # pragma: no cover - exercised by the required script run
    start = time.time()
    print(f"[exp{EXPERIMENT_ID}] starting {ALGORITHM}")
    prior = load_v4_prior_artifact()
    print(
        f"[exp{EXPERIMENT_ID}] exp1134 prior KL={prior['kl_v4_best']:.4f}, "
        f"self-adaptive KL={prior['kl_v4_with_self_adaptive']:.3f}"
    )

    matrices = build_exp1134_seeded_j_matrices()
    best_alpha, best_measurements, per_alpha_summary = run_v5_alpha_grid(matrices)
    v4_measurements = []
    for idx, j_matrix in enumerate(matrices):
        measurement = run_v4_sparse_measurement(j_matrix, seed=1149 + idx)
        measurement["matrix_id"] = f"exp1134_seeded_j{idx}"
        v4_measurements.append(measurement)

    artifact = build_artifact(
        best_alpha=best_alpha,
        best_measurements=best_measurements,
        per_alpha_summary=per_alpha_summary,
        v4_measurements=v4_measurements,
        duration_s=time.time() - start,
        run_date=_utc_now_iso(),
    )
    write_artifact(artifact)
    print(f"wrote {DELIVERABLE}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kl_v5_best: {artifact['kl_v5_best']:.6f}")
    print(f"kl_improvement_over_v4: {artifact['kl_improvement_over_v4']:.6f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
