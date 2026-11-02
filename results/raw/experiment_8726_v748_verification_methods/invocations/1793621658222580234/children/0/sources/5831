"""Run matched single-site Gibbs and temporal exchange Ising schedules.

The temporal arm follows the local field in arXiv:2608.21753. It stores the
previous completed configuration as separate state. That state stays fixed for
one random-order sweep, which prevents an update from reading a mixture of two
temporal configurations. This is a CPU simulator. It does not model p-bit
circuits or any physical sampler.

Spec: REQ-SAMPLE-097, SCENARIO-SAMPLE-097, SCENARIO-SAMPLE-098.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

import numpy as np


ARMS = (
    "ordinary_gibbs",
    "temporal_exchange",
    "temporal_exchange_zero_coupling",
)


class TemporalExchangeInputError(ValueError):
    """Identify an Ising input whose probability semantics are ambiguous."""


@dataclass
class TemporalExchangeState:
    """Keep the current configuration and prior sweep as distinct arrays."""

    current: np.ndarray
    previous: np.ndarray
    sweep_order: np.ndarray
    sweep_position: int = 0
    update_count: int = 0


@dataclass(frozen=True)
class SamplingResult:
    """Return samples and work receipts through one common arm interface."""

    samples: np.ndarray
    energy_trace: np.ndarray
    magnetization_trace: np.ndarray
    best_state: np.ndarray
    best_energy: float
    optimum_hitting_updates: int | None
    update_count: int
    collection_update_counts: np.ndarray
    final_state: TemporalExchangeState
    trajectory_sha256: str
    wall_time_s: float


def _bipolar_state(values: Any, size: int | None = None) -> np.ndarray:
    """Copy one finite -1/+1 vector so caller mutations cannot alter a run."""

    state = np.asarray(values, dtype=np.int8)
    if state.ndim != 1 or (size is not None and state.size != size):
        raise TemporalExchangeInputError("state must be a one-dimensional vector of the graph size")
    if state.size == 0 or not np.all(np.isin(state, (-1, 1))):
        raise TemporalExchangeInputError("state values must be bipolar -1/+1")
    return state.copy()


def _validate_problem(
    biases: Any,
    couplings: Any,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the finite symmetric Ising law before probability arithmetic."""

    b = np.asarray(biases, dtype=np.float64)
    matrix = np.asarray(couplings, dtype=np.float64)
    value = float(temperature)
    if b.ndim != 1 or b.size == 0 or not np.all(np.isfinite(b)):
        raise TemporalExchangeInputError("biases must be one finite non-empty vector")
    if matrix.shape != (b.size, b.size) or not np.all(np.isfinite(matrix)):
        raise TemporalExchangeInputError("couplings must be one finite square graph matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise TemporalExchangeInputError("couplings must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1.0e-12):
        raise TemporalExchangeInputError("coupling diagonal must be zero")
    if not math.isfinite(value) or value <= 0.0:
        raise TemporalExchangeInputError("temperature must be finite and positive")
    return b, matrix, value


def initialize_temporal_state(current: Any, previous: Any) -> TemporalExchangeState:
    """Initialize both configurations without silently copying one over the other."""

    current_state = _bipolar_state(current)
    previous_state = _bipolar_state(previous, current_state.size)
    return TemporalExchangeState(
        current=current_state,
        previous=previous_state,
        sweep_order=np.arange(current_state.size, dtype=np.int64),
    )


def _logistic(value: float) -> float:
    """Evaluate a logistic probability without overflowing at strong fields."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _conditional_from_validated(
    biases: np.ndarray,
    couplings: np.ndarray,
    current: np.ndarray,
    previous: np.ndarray,
    site: int,
    temperature: float,
    temporal_coupling: float,
) -> float:
    """Evaluate the declared local field after the outer contract is valid."""

    spatial_field = float(biases[site] + couplings[site] @ current)
    total_field = spatial_field + float(temporal_coupling) * float(previous[site])
    return _logistic(2.0 * total_field / temperature)


def conditional_probability_plus(
    biases: Any,
    couplings: Any,
    current: Any,
    previous: Any,
    *,
    site: int,
    temperature: float,
    temporal_coupling: float,
) -> float:
    """Return the probability that one selected spin becomes +1."""

    b, matrix, thermal = _validate_problem(biases, couplings, temperature)
    current_state = _bipolar_state(current, b.size)
    previous_state = _bipolar_state(previous, b.size)
    if not isinstance(site, (int, np.integer)) or not 0 <= int(site) < b.size:
        raise TemporalExchangeInputError("site must index one graph spin")
    coupling = float(temporal_coupling)
    if not math.isfinite(coupling):
        raise TemporalExchangeInputError("temporal coupling must be finite")
    return _conditional_from_validated(
        b,
        matrix,
        current_state,
        previous_state,
        int(site),
        thermal,
        coupling,
    )


def attempt_single_site_update(
    state: TemporalExchangeState,
    biases: Any,
    couplings: Any,
    *,
    temperature: float,
    temporal_coupling: float,
    generator: np.random.Generator,
) -> int:
    """Attempt one draw and return the updated site for work accounting."""

    b, matrix, thermal = _validate_problem(biases, couplings, temperature)
    if state.current.shape != (b.size,) or state.previous.shape != (b.size,):
        raise TemporalExchangeInputError("temporal state does not match the graph size")
    if state.sweep_position == 0:
        state.sweep_order = np.asarray(generator.permutation(b.size), dtype=np.int64)
    site = int(state.sweep_order[state.sweep_position])
    probability = _conditional_from_validated(
        b,
        matrix,
        state.current,
        state.previous,
        site,
        thermal,
        float(temporal_coupling),
    )
    state.current[site] = 1 if generator.random() < probability else -1
    state.update_count += 1
    state.sweep_position += 1
    if state.sweep_position == b.size:
        state.previous = state.current.copy()
        state.sweep_position = 0
    return site


def ising_energy(state: Any, biases: Any, couplings: Any) -> float:
    """Compute the spatial Ising energy without the diagnostic temporal term."""

    b = np.asarray(biases, dtype=np.float64)
    matrix = np.asarray(couplings, dtype=np.float64)
    spin = _bipolar_state(state, b.size)
    if matrix.shape != (b.size, b.size):
        raise TemporalExchangeInputError("couplings must match the bias vector")
    return float(-b @ spin - 0.5 * spin @ matrix @ spin)


def enumerate_target_distribution(
    biases: Any,
    couplings: Any,
    temperature: float,
    *,
    maximum_states: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate the exact spatial Boltzmann law for one bounded graph."""

    b, matrix, thermal = _validate_problem(biases, couplings, temperature)
    state_count = 1 << b.size
    if state_count > int(maximum_states):
        raise TemporalExchangeInputError(
            f"exact state count {state_count} exceeds bound {int(maximum_states)}"
        )
    indices = np.arange(state_count, dtype=np.uint64)[:, None]
    bit_positions = np.arange(b.size, dtype=np.uint64)[None, :]
    states = np.where(((indices >> bit_positions) & 1) == 1, 1, -1).astype(np.int8)
    energies = -states @ b - 0.5 * np.einsum("bi,ij,bj->b", states, matrix, states)
    log_weights = -energies / thermal
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    probabilities = weights / float(np.sum(weights))
    return states, probabilities, np.asarray(energies, dtype=np.float64)


def _trajectory_digest(samples: np.ndarray, energy_trace: np.ndarray) -> str:
    """Bind deterministic trajectory evidence without measured wall time."""

    payload = np.asarray(samples, dtype=np.int8).tobytes(order="C")
    payload += np.asarray(energy_trace, dtype="<f8").tobytes(order="C")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sample_ising(
    biases: Any,
    couplings: Any,
    *,
    current: Any,
    previous: Any,
    temperature: float,
    arm: str,
    temporal_coupling: float,
    seed: int,
    burn_in_sweeps: int,
    n_samples: int,
    sweeps_per_sample: int = 1,
    optimum_energy: float | None = None,
) -> SamplingResult:
    """Run one arm with work measured only as attempted single-spin draws."""

    b, matrix, thermal = _validate_problem(biases, couplings, temperature)
    if arm not in ARMS:
        raise TemporalExchangeInputError(f"unknown arm: {arm}")
    coupling = float(temporal_coupling)
    if arm in {"ordinary_gibbs", "temporal_exchange_zero_coupling"} and coupling != 0.0:
        raise TemporalExchangeInputError(f"{arm} requires zero coupling")
    if not math.isfinite(coupling):
        raise TemporalExchangeInputError("temporal coupling must be finite")
    if int(burn_in_sweeps) < 0 or int(n_samples) <= 0 or int(sweeps_per_sample) <= 0:
        raise TemporalExchangeInputError(
            "burn-in must be nonnegative and collection counts positive"
        )

    state = initialize_temporal_state(current, previous)
    if state.current.size != b.size:
        raise TemporalExchangeInputError("initial state pair does not match the graph size")
    generator = np.random.default_rng(int(seed))
    burn_updates = int(burn_in_sweeps) * b.size
    interval_updates = int(sweeps_per_sample) * b.size
    total_updates = burn_updates + int(n_samples) * interval_updates
    samples = np.empty((int(n_samples), b.size), dtype=np.int8)
    energy_trace = np.empty(int(n_samples), dtype=np.float64)
    magnetization_trace = np.empty(int(n_samples), dtype=np.float64)
    collection_updates = np.empty(int(n_samples), dtype=np.int64)

    spatial_energy = ising_energy(state.current, b, matrix)
    best_energy = spatial_energy
    best_state = state.current.copy()
    hit = (
        0
        if optimum_energy is not None and spatial_energy <= float(optimum_energy) + 1.0e-12
        else None
    )
    sample_index = 0
    started = time.perf_counter()
    for _attempt in range(total_updates):
        if state.sweep_position == 0:
            state.sweep_order = np.asarray(generator.permutation(b.size), dtype=np.int64)
        site = int(state.sweep_order[state.sweep_position])
        old_spin = int(state.current[site])
        spatial_field = float(b[site] + matrix[site] @ state.current)
        total_field = spatial_field + coupling * float(state.previous[site])
        probability = _logistic(2.0 * total_field / thermal)
        new_spin = 1 if generator.random() < probability else -1
        if new_spin != old_spin:
            state.current[site] = new_spin
            spatial_energy += 2.0 * old_spin * spatial_field
        state.update_count += 1
        state.sweep_position += 1
        if state.sweep_position == b.size:
            state.previous = state.current.copy()
            state.sweep_position = 0
        if spatial_energy < best_energy - 1.0e-12:
            best_energy = spatial_energy
            best_state = state.current.copy()
        if (
            optimum_energy is not None
            and hit is None
            and spatial_energy <= float(optimum_energy) + 1.0e-12
        ):
            hit = state.update_count
        after_burn = state.update_count - burn_updates
        if after_burn > 0 and after_burn % interval_updates == 0:
            spatial_energy = ising_energy(state.current, b, matrix)
            samples[sample_index] = state.current
            energy_trace[sample_index] = spatial_energy
            magnetization_trace[sample_index] = float(np.mean(state.current))
            collection_updates[sample_index] = state.update_count
            sample_index += 1
    wall_time = time.perf_counter() - started
    if sample_index != int(n_samples):  # pragma: no cover - arithmetic loop invariant.
        raise RuntimeError("matched sample collection did not fill its frozen budget")
    return SamplingResult(
        samples=samples,
        energy_trace=energy_trace,
        magnetization_trace=magnetization_trace,
        best_state=best_state,
        best_energy=float(best_energy),
        optimum_hitting_updates=hit,
        update_count=state.update_count,
        collection_update_counts=collection_updates,
        final_state=state,
        trajectory_sha256=_trajectory_digest(samples, energy_trace),
        wall_time_s=float(wall_time),
    )
