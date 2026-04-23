#!/usr/bin/env python3
"""Experiment 751: D-Wave Neal SamplerBackend Validation.

**What this experiment does:**
    Validates the SamplerBackend abstraction by adding a third concrete
    implementation (DWaveNealBackend) and comparing solution quality and
    speed against the existing CpuBackend (parallel Gibbs).

    D-Wave's neal uses simulated annealing (SA): starts at high temperature
    and gradually lowers it, accepting uphill moves probabilistically.  This
    is fundamentally different from Gibbs sampling (which always takes the
    conditional optimum).  SA may escape local minima that Gibbs gets stuck
    in, potentially finding lower-energy configurations for dense constraint
    graphs.

**What we test:**
    - 20 synthetic IsingModel instances (n=50 spins, sparsity=0.3)
    - Both backends run on the same instances
    - We compare: final_energy (lower = better), wall_time_s

**Expected honest_verdicts:**
    - "neal_better_energy"     — SA finds >5% lower energy than Gibbs
    - "neal_comparable_energy" — within 5% (typical for random problems)
    - "neal_worse_energy"      — Gibbs wins by >5% (unlikely but possible)
    - "blocked_on_dependency"  — dwave-ocean-sdk not installed

Spec: REQ-SAMPLE-017, REQ-SAMPLE-018, SCENARIO-SAMPLE-030, SCENARIO-SAMPLE-031
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

# Ensure repo root is on sys.path so we can import carnot and scripts modules.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.models.ising import IsingConfig, IsingModel  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.samplers.backend import CpuBackend  # noqa: E402
from carnot.samplers.dwave_neal_backend import DWaveNealBackend  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

DELIVERABLE = "results/experiment_751_dwave_neal_backend.json"
N_PROBLEMS = 20
N_SPINS = 50
SPARSITY = 0.3
N_SAMPLES = 100  # reads per backend call


def generate_ising_model(key: jnp.ndarray, seed_offset: int = 0) -> IsingModel:
    """Generate a random IsingModel with controlled sparsity.

    **Why synthetic problems instead of real GSM8K violations:**
        The FR11EventBus violation events are not always persisted in a
        format that is directly loadable as IsingModel instances.  Synthetic
        problems with known sparsity give a clean, reproducible benchmark
        that focuses on the backend comparison rather than data-loading
        complexity.  n=50 spins with sparsity=0.3 gives ~375 non-zero
        couplings — dense enough to create frustrated cycles where SA has
        an advantage over Gibbs.

    Spec: SCENARIO-SAMPLE-030
    """
    k1, k2, k3 = jrandom.split(key, 3)

    # Random coupling matrix with controlled sparsity.
    J_raw = np.array(jrandom.normal(k1, (N_SPINS, N_SPINS)))
    mask = np.array(jrandom.uniform(k2, (N_SPINS, N_SPINS))) < SPARSITY
    J = J_raw * mask
    # Enforce symmetry and zero diagonal.
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    h = np.array(jrandom.normal(k3, (N_SPINS,)))

    # Build an IsingModel and manually set coupling/bias so we control the problem.
    config = IsingConfig(input_dim=N_SPINS, coupling_init="zeros")
    model = IsingModel(config)
    model.coupling = jnp.asarray(J, dtype=jnp.float32)
    model.bias = jnp.asarray(h, dtype=jnp.float32)

    return model


def compute_energy_improvement_pct(
    mean_energy_gibbs: float,
    mean_energy_neal: float,
) -> float:
    """Compute energy improvement percentage (positive = neal finds lower energy).

    **Formula:**
        improvement = (gibbs_energy - neal_energy) / |gibbs_energy| * 100

    Lower energy is better, so if neal finds lower energy than Gibbs,
    (gibbs - neal) > 0 and the percentage is positive.  Division by
    |gibbs_energy| normalises to problem scale.

    Spec: REQ-SAMPLE-018
    """
    if mean_energy_gibbs == 0.0:
        return 0.0
    return (mean_energy_gibbs - mean_energy_neal) / abs(mean_energy_gibbs) * 100.0


def run_experiment(tmpl: ExperimentTemplate) -> dict:
    """Core experiment logic: compare DWaveNealBackend vs CpuBackend on 20 problems.

    Separated from main() so tests can call it with a mock ExperimentTemplate.

    Spec: REQ-SAMPLE-017, REQ-SAMPLE-018, SCENARIO-SAMPLE-030, SCENARIO-SAMPLE-031
    """
    # Check if dwave-ocean-sdk / neal is available.
    neal_backend = DWaveNealBackend(num_reads=N_SAMPLES, num_sweeps=1000, beta_range=(0.1, 5.0))

    if not neal_backend.available:
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_on_dependency",
                "n_problems": 0,
                "n_spins": N_SPINS,
                "n_samples": N_SAMPLES,
                "install_command": "pip install dwave-ocean-sdk",
                "mean_energy_neal": None,
                "mean_energy_gibbs": None,
                "energy_improvement_pct": None,
                "wall_time_s_neal": None,
                "wall_time_s_gibbs": None,
            },
            status="blocked",
        )
        (Path(_REPO) / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        return artifact

    cpu_backend = CpuBackend(seed=42)
    key = jrandom.PRNGKey(0)

    neal_energies: list[float] = []
    gibbs_energies: list[float] = []
    neal_times: list[float] = []
    gibbs_times: list[float] = []

    for i in range(N_PROBLEMS):
        key, subkey = jrandom.split(key)
        model = generate_ising_model(subkey, seed_offset=i)

        J_np = np.asarray(model.coupling)
        h_np = np.asarray(model.bias)

        # --- Neal (D-Wave SA) ---
        t0 = time.perf_counter()
        neal_result = neal_backend.sample(model)
        neal_times.append(time.perf_counter() - t0)
        neal_energies.append(neal_result.energy)

        # --- Gibbs (CpuBackend) ---
        t0 = time.perf_counter()
        # minimize_energy signature: (biases, couplings, n_samples, n_steps, beta)
        gibbs_samples = cpu_backend.minimize_energy(
            h_np, J_np, n_samples=N_SAMPLES, n_steps=1000, beta=10.0
        )
        gibbs_times.append(time.perf_counter() - t0)

        # Find lowest-energy Gibbs sample using IsingModel.energy.
        gibbs_best_energy = min(
            float(model.energy(jnp.asarray(gibbs_samples[s], dtype=jnp.float32)))
            for s in range(len(gibbs_samples))
        )
        gibbs_energies.append(gibbs_best_energy)

    mean_energy_neal = float(np.mean(neal_energies))
    mean_energy_gibbs = float(np.mean(gibbs_energies))
    energy_improvement_pct = compute_energy_improvement_pct(mean_energy_gibbs, mean_energy_neal)

    if energy_improvement_pct > 5.0:
        honest_verdict = "neal_better_energy"
    elif abs(energy_improvement_pct) <= 5.0:
        honest_verdict = "neal_comparable_energy"
    else:
        honest_verdict = "neal_worse_energy"

    artifact = tmpl.build_result(
        {
            "mean_energy_neal": mean_energy_neal,
            "mean_energy_gibbs": mean_energy_gibbs,
            "energy_improvement_pct": energy_improvement_pct,
            "wall_time_s_neal": float(np.mean(neal_times)),
            "wall_time_s_gibbs": float(np.mean(gibbs_times)),
            "n_problems": N_PROBLEMS,
            "n_spins": N_SPINS,
            "n_samples": N_SAMPLES,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )
    (Path(_REPO) / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
    return artifact


def main() -> None:
    """Entry point: set up template/watchdog, run experiment, assert deliverable."""
    tmpl = ExperimentTemplate(
        751,
        "D-Wave Neal SamplerBackend Validation",
        DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(
        751,
        timeout_minutes=30,
        result_path=str(Path(_REPO) / DELIVERABLE),
    ):
        run_experiment(tmpl)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
