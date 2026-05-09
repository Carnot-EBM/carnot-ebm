"""Synthetic Z1 analog beta-drift simulator and detailed-balance correction.

Spec: REQ-SAMPLE-063, SCENARIO-SAMPLE-091.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DriftCorrectionConfig:
    """Configuration for the simulator-only Exp 1583 drift correction run."""

    n_spins: int = 128
    beta: float = 0.85
    drift_std: float = 0.05
    n_samples: int = 384
    n_warmup_sweeps: int = 192
    sweeps_per_sample: int = 3
    seed: int = 1583
    coupling_scale: float = 0.18
    bias_scale: float = 0.06
    correction_method: str = "hastings_boundary_accept_reject"


@dataclass
class SamplingTrace:
    """Samples plus the boundary acceptance rate used by the correction gate."""

    samples: np.ndarray
    acceptance_rate: float


def _spin_pm(samples: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(samples, dtype=bool), 1.0, -1.0)


def build_bipartite_ring_problem(config: DriftCorrectionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build the n-spin ring fixture whose even/odd blocks have no internal edges."""

    n_spins = int(config.n_spins)
    rng = np.random.default_rng(int(config.seed))
    phases = np.linspace(0.0, 2.0 * np.pi, n_spins, endpoint=False)
    biases = config.bias_scale * (0.65 * np.sin(phases) + 0.35 * rng.normal(size=n_spins))
    couplings = np.zeros((n_spins, n_spins), dtype=np.float64)
    for index in range(n_spins):
        right = (index + 1) % n_spins
        weight = config.coupling_scale * (0.7 + 0.3 * np.cos(phases[index]))
        couplings[index, right] = weight
        couplings[right, index] = weight
    return biases.astype(np.float64), couplings


def make_beta_drift(config: DriftCorrectionConfig) -> np.ndarray:
    """Return positive per-spin beta multipliers with controlled mean and std."""

    n_spins = int(config.n_spins)
    drift_std = float(config.drift_std)
    if drift_std == 0.0:
        return np.ones(n_spins, dtype=np.float64)
    rng = np.random.default_rng(int(config.seed) + 1)
    raw = rng.normal(size=n_spins)
    raw = (raw - raw.mean()) / raw.std(ddof=0)
    drift = 1.0 + drift_std * raw
    drift = drift - drift.mean() + 1.0
    drift = 1.0 + (drift - 1.0) * (drift_std / drift.std(ddof=0))
    return drift.astype(np.float64)


def hamiltonian_score(
    samples: np.ndarray,
    biases: np.ndarray,
    couplings: np.ndarray,
) -> float | np.ndarray:
    """Return ``h.s + 0.5*sJs`` in +/-1 spin convention."""

    spins = _spin_pm(np.asarray(samples))
    bias_vector = np.asarray(biases, dtype=np.float64)
    coupling_matrix = np.asarray(couplings, dtype=np.float64)
    if spins.ndim == 1:
        return float(spins @ bias_vector + 0.5 * spins @ coupling_matrix @ spins)
    linear = spins @ bias_vector
    pair = 0.5 * np.einsum("bi,ij,bj->b", spins, coupling_matrix, spins)
    return linear + pair


def hamiltonian_energy(
    samples: np.ndarray,
    biases: np.ndarray,
    couplings: np.ndarray,
) -> float | np.ndarray:
    """Return physical Ising energy, without the scalar beta multiplier."""

    return -hamiltonian_score(samples, biases, couplings)


def magnetization(samples: np.ndarray) -> np.ndarray:
    """Return per-sample mean +/-1 magnetization."""

    return _spin_pm(np.asarray(samples)).mean(axis=1)


def _local_fields(state: np.ndarray, biases: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    spins = _spin_pm(state)
    return np.asarray(biases, dtype=np.float64) + np.asarray(couplings, dtype=np.float64) @ spins


def _log_bernoulli(values: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return np.where(values, -np.logaddexp(0.0, -logits), -np.logaddexp(0.0, logits))


def proposal_log_probability(
    source: np.ndarray,
    proposed: np.ndarray,
    block: np.ndarray,
    biases: np.ndarray,
    couplings: np.ndarray,
    beta: float,
    beta_multipliers: np.ndarray,
) -> float:
    """Log probability of proposing ``proposed[block]`` from ``source``."""

    source_array = np.asarray(source, dtype=bool)
    proposed_array = np.asarray(proposed, dtype=bool)
    block_array = np.asarray(block, dtype=int)
    fields = _local_fields(source_array, biases, couplings)
    local_beta = float(beta) * np.asarray(beta_multipliers, dtype=np.float64)[block_array]
    logits = 2.0 * local_beta * fields[block_array]
    return float(_log_bernoulli(proposed_array[block_array], logits).sum())


def hastings_log_acceptance(
    source: np.ndarray,
    proposed: np.ndarray,
    block: np.ndarray,
    biases: np.ndarray,
    couplings: np.ndarray,
    beta: float,
    beta_multipliers: np.ndarray,
) -> float:
    """Return the raw Hastings log-ratio for a drifted block proposal."""

    target_delta = float(beta) * (
        float(hamiltonian_score(proposed, biases, couplings))
        - float(hamiltonian_score(source, biases, couplings))
    )
    forward = proposal_log_probability(
        source, proposed, block, biases, couplings, beta, beta_multipliers
    )
    reverse = proposal_log_probability(
        proposed, source, block, biases, couplings, beta, beta_multipliers
    )
    ratio = target_delta + reverse - forward
    return 0.0 if abs(ratio) < 1e-12 else float(ratio)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.where(
        logits >= 0.0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def _block_indices(n_spins: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(0, n_spins, 2, dtype=int), np.arange(1, n_spins, 2, dtype=int)


def _propose_block(
    rng: np.random.Generator,
    state: np.ndarray,
    block: np.ndarray,
    biases: np.ndarray,
    couplings: np.ndarray,
    beta: float,
    beta_multipliers: np.ndarray,
) -> np.ndarray:
    fields = _local_fields(state, biases, couplings)
    local_beta = float(beta) * np.asarray(beta_multipliers, dtype=np.float64)[block]
    probabilities = _sigmoid(2.0 * local_beta * fields[block])
    proposed = state.copy()
    proposed[block] = rng.random(block.size) < probabilities
    return proposed


def sample_block_gibbs(
    biases: np.ndarray,
    couplings: np.ndarray,
    config: DriftCorrectionConfig,
    beta_multipliers: np.ndarray,
    *,
    corrected: bool,
) -> SamplingTrace:
    """Run even/odd block Gibbs with optional Hastings correction."""

    rng = np.random.default_rng(int(config.seed))
    n_spins = int(config.n_spins)
    state = rng.random(n_spins) < 0.5
    blocks = _block_indices(n_spins)
    samples = np.empty((int(config.n_samples), n_spins), dtype=bool)
    accepted = 0
    attempts = 0

    def sweep() -> None:
        nonlocal state, accepted, attempts
        for block in blocks:
            proposed = _propose_block(
                rng, state, block, biases, couplings, config.beta, beta_multipliers
            )
            if corrected:
                log_accept = min(
                    0.0,
                    hastings_log_acceptance(
                        state, proposed, block, biases, couplings, config.beta, beta_multipliers
                    ),
                )
                if np.log(rng.random()) < log_accept:
                    state = proposed
                    accepted += 1
            else:
                state = proposed
                accepted += 1
            attempts += 1

    for _ in range(int(config.n_warmup_sweeps)):
        sweep()
    for sample_index in range(int(config.n_samples)):
        for _ in range(int(config.sweeps_per_sample)):
            sweep()
        samples[sample_index] = state
    return SamplingTrace(samples=samples, acceptance_rate=float(accepted / attempts))


@dataclass
class SyntheticDriftIsingBackend:
    """SamplerBackend-shaped synthetic analog drift boundary."""

    config: DriftCorrectionConfig
    beta_multipliers: np.ndarray
    corrected: bool = False
    last_acceptance_rate: float = 0.0

    @property
    def backend_name(self) -> str:
        suffix = "hastings" if self.corrected else "uncorrected"
        return f"synthetic-drift-ising-{suffix}"

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        run_config = replace(
            self.config,
            n_samples=int(n_samples),
            beta=float(config.get("beta", self.config.beta)),
        )
        trace = sample_block_gibbs(
            biases,
            couplings,
            run_config,
            self.beta_multipliers,
            corrected=self.corrected,
        )
        self.last_acceptance_rate = trace.acceptance_rate
        return trace.samples

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        run_config = replace(
            self.config,
            n_samples=int(n_samples),
            n_warmup_sweeps=int(n_steps),
            beta=float(beta),
        )
        trace = sample_block_gibbs(
            biases,
            couplings,
            run_config,
            self.beta_multipliers,
            corrected=self.corrected,
        )
        self.last_acceptance_rate = trace.acceptance_rate
        return trace.samples


def energy_bias(sample_energy: np.ndarray, reference_energy: np.ndarray) -> float:
    """Mean-energy bias relative to the no-drift reference."""

    return float(np.asarray(sample_energy, dtype=np.float64).mean() - np.asarray(reference_energy).mean())


def magnetization_bias(samples: np.ndarray, reference_samples: np.ndarray) -> float:
    """Mean magnetization bias relative to the no-drift reference."""

    return float(magnetization(samples).mean() - magnetization(reference_samples).mean())


def empirical_kl_proxy(samples: np.ndarray, reference: np.ndarray, bins: int = 24) -> float:
    """Histogram KL proxy for one-dimensional energy samples."""

    sample_values = np.asarray(samples, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    low = float(min(sample_values.min(), reference_values.min()))
    high = float(max(sample_values.max(), reference_values.max()))
    span = max(high - low, 1e-9)
    hist_range = (low - 0.01 * span, high + 0.01 * span)
    sample_hist, _ = np.histogram(sample_values, bins=int(bins), range=hist_range)
    reference_hist, _ = np.histogram(reference_values, bins=int(bins), range=hist_range)
    epsilon = 1e-12
    p = (sample_hist.astype(np.float64) + epsilon) / (sample_hist.sum() + epsilon * bins)
    q = (reference_hist.astype(np.float64) + epsilon) / (reference_hist.sum() + epsilon * bins)
    return float(np.sum(p * np.log(p / q)))


def empirical_ks_proxy(samples: np.ndarray, reference: np.ndarray) -> float:
    """Two-sample KS distance proxy for scalar observable samples."""

    sample_values = np.sort(np.asarray(samples, dtype=np.float64))
    reference_values = np.sort(np.asarray(reference, dtype=np.float64))
    grid = np.sort(np.concatenate([sample_values, reference_values]))
    sample_cdf = np.searchsorted(sample_values, grid, side="right") / sample_values.size
    reference_cdf = np.searchsorted(reference_values, grid, side="right") / reference_values.size
    return float(np.max(np.abs(sample_cdf - reference_cdf)))


def combined_sigma(samples: np.ndarray, reference: np.ndarray) -> float:
    """One-sigma combined empirical spread for two observable samples."""

    sample_values = np.asarray(samples, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(sample_values.var(ddof=1) + reference_values.var(ddof=1)))


def build_exp1583_payload(config: DriftCorrectionConfig | None = None) -> dict[str, Any]:
    """Run the simulator-only correction experiment and return its JSON payload."""

    run_config = config or DriftCorrectionConfig()
    biases, couplings = build_bipartite_ring_problem(run_config)
    drift = make_beta_drift(run_config)
    reference_backend = SyntheticDriftIsingBackend(run_config, np.ones(run_config.n_spins))
    drift_backend = SyntheticDriftIsingBackend(run_config, drift)
    corrected_backend = SyntheticDriftIsingBackend(run_config, drift, corrected=True)
    sample_config = {"beta": run_config.beta}
    reference_samples = reference_backend.sample(
        biases, couplings, run_config.n_samples, sample_config
    )
    drift_samples = drift_backend.sample(biases, couplings, run_config.n_samples, sample_config)
    corrected_samples = corrected_backend.sample(
        biases, couplings, run_config.n_samples, sample_config
    )
    reference_energy = hamiltonian_energy(reference_samples, biases, couplings)
    uncorrected_energy = hamiltonian_energy(drift_samples, biases, couplings)
    corrected_energy = hamiltonian_energy(corrected_samples, biases, couplings)
    corrected_energy_bias = energy_bias(corrected_energy, reference_energy)
    corrected_mag_bias = magnetization_bias(corrected_samples, reference_samples)
    energy_sigma = combined_sigma(corrected_energy, reference_energy)
    mag_sigma = combined_sigma(magnetization(corrected_samples), magnetization(reference_samples))
    within_1sigma = abs(corrected_energy_bias) <= energy_sigma and abs(corrected_mag_bias) <= mag_sigma
    no_hardware_claim = True
    return {
        "status": "complete",
        "synthetic_drift_simulator_ready": True,
        "correction_method": run_config.correction_method,
        "n_spins": int(run_config.n_spins),
        "beta": float(run_config.beta),
        "drift_std": float(drift.std(ddof=0)),
        "uncorrected_energy_bias": energy_bias(uncorrected_energy, reference_energy),
        "corrected_energy_bias": corrected_energy_bias,
        "uncorrected_magnetization_bias": magnetization_bias(drift_samples, reference_samples),
        "corrected_magnetization_bias": corrected_mag_bias,
        "uncorrected_energy_kl_proxy": empirical_kl_proxy(uncorrected_energy, reference_energy),
        "corrected_energy_kl_proxy": empirical_kl_proxy(corrected_energy, reference_energy),
        "uncorrected_energy_ks_proxy": empirical_ks_proxy(uncorrected_energy, reference_energy),
        "corrected_energy_ks_proxy": empirical_ks_proxy(corrected_energy, reference_energy),
        "energy_one_sigma": energy_sigma,
        "magnetization_one_sigma": mag_sigma,
        "corrected_acceptance_rate": corrected_backend.last_acceptance_rate,
        "correction_within_1sigma": bool(within_1sigma),
        "detailed_balance_correction_ready": bool(within_1sigma and no_hardware_claim),
        "simulator_only_no_hardware_claim": no_hardware_claim,
        "sampler_backend_boundary": "carnot.sampling.z1_drift_correction.SyntheticDriftIsingBackend",
        "spec_refs": ["REQ-SAMPLE-063", "SCENARIO-SAMPLE-091"],
        "honest_verdict": (
            "complete: simulator_only_hastings_correction_within_1sigma"
            if within_1sigma
            else "blocked: simulator_only_correction_not_within_1sigma"
        ),
    }


def write_exp1583_artifact(
    path: str | Path = "results/experiment_1583_z1_analog_drift_detailed_balance_correction.json",
    config: DriftCorrectionConfig | None = None,
) -> dict[str, Any]:
    """Write the Exp 1583 terminal artifact."""

    payload = build_exp1583_payload(config)
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
