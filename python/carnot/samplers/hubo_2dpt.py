"""CPU HUBO/p-spin sampling with beta/penalty parallel tempering.

Spec refs: REQ-SAMPLE-5116, SCENARIO-SAMPLE-5116.

The sampler in this file is intentionally small and exact-checkable. It targets
direct high-order parity energies from Exp 5102, keeps every state as binary
bits, and evaluates p-spin products directly instead of reducing the problem to
pairwise QUBO gadgets. That makes it a CPU algorithm reference, not a hardware
claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
import math
from typing import Any

import numpy as np

from carnot.experiment_5102_hubo_pspin_direct_energy import (
    build_hubo_encoding,
    build_instance_family,
)


@dataclass(frozen=True)
class HuboTerm:
    """One p-spin monomial over binary variables using bipolar products.

    Binary bit ``0`` maps to spin ``+1`` and bit ``1`` maps to spin ``-1``.
    This is the same convention used by the Exp 5102 direct parity encoding.
    """

    variables: tuple[int, ...]
    coefficient: float


@dataclass(frozen=True)
class HuboProblem:
    """Tiny direct HUBO/p-spin constraint problem for CPU reference sampling."""

    name: str
    family: str
    n_vars: int
    constraint_constant: float
    constraint_terms: tuple[HuboTerm, ...]
    objective_constant: float = 0.0
    objective_terms: tuple[HuboTerm, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class ExactHuboEnumeration:
    """Exhaustive optimum and energy histogram for a tiny HUBO instance."""

    optimum_energy: float
    optimal_states: tuple[tuple[int, ...], ...]
    energy_distribution: Mapping[float, int]
    all_states: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe exact-enumeration evidence."""

        return {
            "optimum_energy": self.optimum_energy,
            "optimal_states": [list(state) for state in self.optimal_states],
            "n_optimal_states": len(self.optimal_states),
            "energy_distribution": [
                {"energy": energy, "count": count}
                for energy, count in sorted(self.energy_distribution.items())
            ],
            "n_states_enumerated": len(self.all_states),
        }


@dataclass(frozen=True)
class SwapStats:
    """Attempt and acceptance counters for one swap axis."""

    attempts: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Return accepted / attempted, using zero for an unused axis."""

        if self.attempts == 0:
            return 0.0
        return float(self.accepted / self.attempts)

    def with_attempt(self, *, accepted: bool) -> "SwapStats":
        """Return a new counter with one additional swap attempt."""

        return SwapStats(
            attempts=self.attempts + 1,
            accepted=self.accepted + int(accepted),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe swap counters."""

        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "acceptance_rate": round(self.acceptance_rate, 6),
        }


@dataclass(frozen=True)
class HuboRunResult:
    """Sampler result measured against the target high-penalty HUBO energy."""

    algorithm: str
    best_energy: float
    final_energy: float
    best_state: tuple[int, ...]
    final_state: tuple[int, ...]
    energy_trace: tuple[float, ...]
    swap_stats: Mapping[str, SwapStats]
    beta_grid: tuple[float, ...]
    penalty_grid: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-safe sampler output."""

        return {
            "algorithm": self.algorithm,
            "best_energy": self.best_energy,
            "final_energy": self.final_energy,
            "best_state": list(self.best_state),
            "final_state": list(self.final_state),
            "energy_trace": list(self.energy_trace),
            "swap_stats": {
                axis: stats.as_dict()
                for axis, stats in sorted(self.swap_stats.items())
            },
            "beta_grid": list(self.beta_grid),
            "penalty_grid": list(self.penalty_grid),
        }


@dataclass(frozen=True)
class AdaptiveHuboRunResult:
    """Adaptive 2D PT result with residual and ladder telemetry.

    The adaptive run is still a CPU reference. The extra fields make the
    sampler auditable: they show how the inverse-temperature ladder moved, how
    far the best state stayed above the exact optimum, and whether the local
    swap ratio remains reversible when read forward and backward.
    """

    algorithm: str
    best_energy: float
    final_energy: float
    best_state: tuple[int, ...]
    final_state: tuple[int, ...]
    energy_trace: tuple[float, ...]
    residual_energy_trace: tuple[float, ...]
    swap_stats: Mapping[str, SwapStats]
    beta_pair_swap_stats: tuple[SwapStats, ...]
    beta_grid_initial: tuple[float, ...]
    beta_grid_final: tuple[float, ...]
    beta_grid_history: tuple[tuple[float, ...], ...]
    penalty_grid: tuple[float, ...]
    adaptation_history: tuple[Mapping[str, Any], ...]
    round_trip_proxy: Mapping[str, Any]
    detailed_balance_sanity: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-safe adaptive sampler output."""

        return {
            "algorithm": self.algorithm,
            "best_energy": self.best_energy,
            "final_energy": self.final_energy,
            "best_state": list(self.best_state),
            "final_state": list(self.final_state),
            "energy_trace": list(self.energy_trace),
            "residual_energy_trace": list(self.residual_energy_trace),
            "swap_stats": {
                axis: stats.as_dict()
                for axis, stats in sorted(self.swap_stats.items())
            },
            "beta_pair_swap_stats": [
                {"pair_index": pair_index, **stats.as_dict()}
                for pair_index, stats in enumerate(self.beta_pair_swap_stats)
            ],
            "beta_grid_initial": list(self.beta_grid_initial),
            "beta_grid_final": list(self.beta_grid_final),
            "beta_grid_history": [list(grid) for grid in self.beta_grid_history],
            "penalty_grid": list(self.penalty_grid),
            "adaptation_history": [dict(row) for row in self.adaptation_history],
            "round_trip_proxy": dict(self.round_trip_proxy),
            "detailed_balance_sanity": dict(self.detailed_balance_sanity),
        }


@dataclass(frozen=True)
class Hubo2DPTConfig:
    """Configuration for the CPU HUBO beta/penalty replica grid."""

    beta_grid: tuple[float, ...] = (0.35, 0.8, 1.6, 3.0)
    penalty_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    sweeps: int = 48
    swap_interval: int = 1

    def __post_init__(self) -> None:
        """Validate grids before any random state is drawn."""

        _validate_positive_grid(self.beta_grid, "beta_grid")
        _validate_positive_grid(self.penalty_grid, "penalty_grid")
        if self.sweeps < 1:
            raise ValueError("sweeps must be at least 1")
        if self.swap_interval < 1:
            raise ValueError("swap_interval must be at least 1")

    @property
    def target_penalty(self) -> float:
        """Return the penalty used for final utility and exact checks."""

        return max(self.penalty_grid)


@dataclass(frozen=True)
class AdaptiveHubo2DPTConfig:
    """Configuration for adaptive inverse-temperature HUBO 2D PT.

    The penalty grid stays fixed while beta spacings are adapted from recent
    adjacent-swap acceptance. Keeping endpoints fixed prevents the adaptive run
    from quietly changing the hot and cold distribution it is being compared
    against.
    """

    initial_beta_grid: tuple[float, ...] = (0.35, 0.8, 1.6, 3.0)
    penalty_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    sweeps: int = 48
    swap_interval: int = 1
    adaptation_interval: int = 6
    target_acceptance: float = 0.45
    adaptation_learning_rate: float = 0.5
    min_beta_gap: float = 0.05

    def __post_init__(self) -> None:
        """Validate adaptive settings before any random state is drawn."""

        _validate_positive_grid(self.initial_beta_grid, "initial_beta_grid")
        _validate_positive_grid(self.penalty_grid, "penalty_grid")
        if len(self.initial_beta_grid) < 2:
            raise ValueError("initial_beta_grid must contain at least two betas")
        if self.sweeps < 1:
            raise ValueError("sweeps must be at least 1")
        if self.swap_interval < 1:
            raise ValueError("swap_interval must be at least 1")
        if self.adaptation_interval < 1:
            raise ValueError("adaptation_interval must be at least 1")
        if not 0.0 < self.target_acceptance < 1.0:
            raise ValueError("target_acceptance must be in (0, 1)")
        if self.adaptation_learning_rate < 0.0:
            raise ValueError("adaptation_learning_rate must be nonnegative")
        if self.min_beta_gap <= 0.0:
            raise ValueError("min_beta_gap must be positive")

    @property
    def target_penalty(self) -> float:
        """Return the penalty used for final utility and exact checks."""

        return max(self.penalty_grid)


class Hubo2DParallelTemperingSampler:
    """Serial CPU 2D parallel tempering over beta and constraint penalty."""

    def __init__(self, config: Hubo2DPTConfig | None = None) -> None:
        """Store the immutable sampler configuration."""

        self.config = config or Hubo2DPTConfig()

    def run(self, problem: HuboProblem, *, seed: int) -> HuboRunResult:
        """Run a 2D beta/penalty replica grid with adjacent-axis swaps."""

        rng = np.random.default_rng(seed)
        beta_grid = self.config.beta_grid
        penalty_grid = self.config.penalty_grid
        states = rng.integers(
            0,
            2,
            size=(len(beta_grid), len(penalty_grid), problem.n_vars),
            dtype=np.int8,
        )
        energies = _slot_energies(problem, states, penalty_grid)
        best_energy, best_state = _best_target_state(
            problem,
            states,
            target_penalty=self.config.target_penalty,
        )
        beta_stats = SwapStats()
        penalty_stats = SwapStats()
        trace: list[float] = []

        for sweep in range(1, self.config.sweeps + 1):
            for beta_index, beta in enumerate(beta_grid):
                for penalty_index, penalty in enumerate(penalty_grid):
                    states[beta_index, penalty_index] = _gibbs_sweep(
                        problem,
                        states[beta_index, penalty_index],
                        beta=beta,
                        penalty=penalty,
                        rng=rng,
                    )
                    energies[beta_index, penalty_index] = evaluate_hubo_energy(
                        problem,
                        states[beta_index, penalty_index],
                        penalty=penalty,
                    )

            if sweep % self.config.swap_interval == 0:
                states, energies, beta_stats = _swap_beta_axis(
                    problem,
                    states,
                    energies,
                    beta_grid=beta_grid,
                    penalty_grid=penalty_grid,
                    stats=beta_stats,
                    rng=rng,
                    phase=(sweep // self.config.swap_interval) % 2,
                )
                states, energies, penalty_stats = _swap_penalty_axis(
                    problem,
                    states,
                    energies,
                    beta_grid=beta_grid,
                    penalty_grid=penalty_grid,
                    stats=penalty_stats,
                    rng=rng,
                    phase=(sweep // self.config.swap_interval) % 2,
                )

            current_best, current_state = _best_target_state(
                problem,
                states,
                target_penalty=self.config.target_penalty,
            )
            if current_best < best_energy:
                best_energy = current_best
                best_state = current_state
            trace.append(best_energy)

        cold_beta_index = len(beta_grid) - 1
        target_penalty_index = len(penalty_grid) - 1
        final_state = tuple(int(value) for value in states[cold_beta_index, target_penalty_index])
        final_energy = evaluate_hubo_energy(problem, final_state, penalty=self.config.target_penalty)
        return HuboRunResult(
            algorithm="two_d_beta_penalty_pt",
            best_energy=best_energy,
            final_energy=final_energy,
            best_state=best_state,
            final_state=final_state,
            energy_trace=tuple(trace),
            swap_stats={
                "beta_axis": beta_stats,
                "penalty_axis": penalty_stats,
            },
            beta_grid=beta_grid,
            penalty_grid=penalty_grid,
        )


class AdaptiveHubo2DParallelTemperingSampler:
    """Serial CPU 2D PT with deterministic adaptive beta-ladder updates."""

    def __init__(self, config: AdaptiveHubo2DPTConfig | None = None) -> None:
        """Store the immutable adaptive sampler configuration."""

        self.config = config or AdaptiveHubo2DPTConfig()

    def run(
        self,
        problem: HuboProblem,
        *,
        seed: int,
        exact_optimum_energy: float,
    ) -> AdaptiveHuboRunResult:
        """Run adaptive 2D PT and record exact-residual telemetry."""

        rng = np.random.default_rng(seed)
        beta_grid = np.asarray(self.config.initial_beta_grid, dtype=np.float64)
        penalty_grid = self.config.penalty_grid
        states = rng.integers(
            0,
            2,
            size=(len(beta_grid), len(penalty_grid), problem.n_vars),
            dtype=np.int8,
        )
        energies = _slot_energies(problem, states, penalty_grid)
        best_energy, best_state = _best_target_state(
            problem,
            states,
            target_penalty=self.config.target_penalty,
        )
        beta_stats = SwapStats()
        penalty_stats = SwapStats()
        beta_pair_totals = [SwapStats() for _ in range(len(beta_grid) - 1)]
        beta_pair_window = [SwapStats() for _ in range(len(beta_grid) - 1)]
        beta_tags = np.tile(np.arange(len(beta_grid), dtype=np.int16)[:, None], (1, len(penalty_grid)))
        min_seen = beta_tags.copy()
        max_seen = beta_tags.copy()
        accepted_up_moves = 0
        accepted_down_moves = 0
        trace: list[float] = []
        residual_trace: list[float] = []
        beta_history: list[tuple[float, ...]] = [_round_grid(beta_grid)]
        adaptation_history: list[Mapping[str, Any]] = []

        for sweep in range(1, self.config.sweeps + 1):
            for beta_index, beta in enumerate(beta_grid):
                for penalty_index, penalty in enumerate(penalty_grid):
                    states[beta_index, penalty_index] = _gibbs_sweep(
                        problem,
                        states[beta_index, penalty_index],
                        beta=float(beta),
                        penalty=penalty,
                        rng=rng,
                    )
                    energies[beta_index, penalty_index] = evaluate_hubo_energy(
                        problem,
                        states[beta_index, penalty_index],
                        penalty=penalty,
                    )

            if sweep % self.config.swap_interval == 0:
                (
                    states,
                    energies,
                    beta_tags,
                    beta_stats,
                    beta_pair_totals,
                    beta_pair_window,
                    accepted_up_moves,
                    accepted_down_moves,
                ) = _adaptive_swap_beta_axis(
                    problem,
                    states,
                    energies,
                    beta_tags,
                    beta_grid=tuple(float(value) for value in beta_grid),
                    penalty_grid=penalty_grid,
                    stats=beta_stats,
                    pair_totals=beta_pair_totals,
                    pair_window=beta_pair_window,
                    accepted_up_moves=accepted_up_moves,
                    accepted_down_moves=accepted_down_moves,
                    rng=rng,
                    phase=(sweep // self.config.swap_interval) % 2,
                )
                states, energies, penalty_stats = _swap_penalty_axis(
                    problem,
                    states,
                    energies,
                    beta_grid=tuple(float(value) for value in beta_grid),
                    penalty_grid=penalty_grid,
                    stats=penalty_stats,
                    rng=rng,
                    phase=(sweep // self.config.swap_interval) % 2,
                )
                min_seen = np.minimum(min_seen, beta_tags)
                max_seen = np.maximum(max_seen, beta_tags)

            if sweep % self.config.adaptation_interval == 0 and sweep < self.config.sweeps:
                pair_rates = tuple(stats.acceptance_rate for stats in beta_pair_window)
                previous_grid = _round_grid(beta_grid)
                beta_grid = np.asarray(
                    adapt_inverse_temperature_ladder(
                        previous_grid,
                        pair_acceptance_rates=pair_rates,
                        target_acceptance=self.config.target_acceptance,
                        learning_rate=self.config.adaptation_learning_rate,
                        min_beta_gap=self.config.min_beta_gap,
                    ),
                    dtype=np.float64,
                )
                updated_grid = _round_grid(beta_grid)
                adaptation_history.append(
                    {
                        "sweep": sweep,
                        "previous_beta_grid": list(previous_grid),
                        "pair_acceptance_rates": [_round_metric(value) for value in pair_rates],
                        "updated_beta_grid": list(updated_grid),
                        "monotonic_order_preserved": _is_strictly_increasing(updated_grid),
                    }
                )
                beta_history.append(updated_grid)
                beta_pair_window = [SwapStats() for _ in beta_pair_window]

            current_best, current_state = _best_target_state(
                problem,
                states,
                target_penalty=self.config.target_penalty,
            )
            if current_best < best_energy:
                best_energy = current_best
                best_state = current_state
            trace.append(best_energy)
            residual_trace.append(_round_energy(best_energy - exact_optimum_energy))

        cold_beta_index = len(beta_grid) - 1
        target_penalty_index = len(penalty_grid) - 1
        final_state = tuple(int(value) for value in states[cold_beta_index, target_penalty_index])
        detailed_balance = _detailed_balance_sanity_for_slots(
            problem,
            states,
            beta_grid=tuple(float(value) for value in beta_grid),
            penalty_grid=penalty_grid,
            accepted_up_moves=accepted_up_moves,
            accepted_down_moves=accepted_down_moves,
        )
        return AdaptiveHuboRunResult(
            algorithm="adaptive_two_d_beta_penalty_pt",
            best_energy=best_energy,
            final_energy=evaluate_hubo_energy(problem, final_state, penalty=self.config.target_penalty),
            best_state=best_state,
            final_state=final_state,
            energy_trace=tuple(trace),
            residual_energy_trace=tuple(residual_trace),
            swap_stats={
                "beta_axis": beta_stats,
                "penalty_axis": penalty_stats,
            },
            beta_pair_swap_stats=tuple(beta_pair_totals),
            beta_grid_initial=self.config.initial_beta_grid,
            beta_grid_final=_round_grid(beta_grid),
            beta_grid_history=tuple(beta_history),
            penalty_grid=penalty_grid,
            adaptation_history=tuple(adaptation_history),
            round_trip_proxy=_round_trip_proxy(min_seen, max_seen, accepted_up_moves, accepted_down_moves),
            detailed_balance_sanity=detailed_balance,
        )


def build_synthetic_hubo_families() -> tuple[HuboProblem, ...]:
    """Return tiny parity HUBO families reused from the exact Exp 5102 encoder."""

    problems: list[HuboProblem] = []
    for instance in build_instance_family():
        encoding = build_hubo_encoding(instance)
        family = (
            "sat_high_order_parity"
            if "frustrated" not in instance.instance_id
            else "frustrated_high_order_parity"
        )
        terms = tuple(
            HuboTerm(variables=tuple(term), coefficient=float(coefficient))
            for term, coefficient in sorted(encoding.terms.items())
        )
        problems.append(
            HuboProblem(
                name=instance.instance_id,
                family=family,
                n_vars=instance.n_vars,
                constraint_constant=float(encoding.constant),
                constraint_terms=terms,
                description=instance.description,
            )
        )
    return tuple(problems)


def evaluate_hubo_energy(
    problem: HuboProblem,
    state: Sequence[int] | np.ndarray,
    *,
    penalty: float,
) -> float:
    """Evaluate objective plus ``penalty * constraint`` for one binary state."""

    if penalty < 0.0:
        raise ValueError("penalty must be nonnegative")
    bits = tuple(int(value) for value in state)
    if len(bits) != problem.n_vars:
        raise ValueError("state length does not match problem variables")
    objective = problem.objective_constant + _evaluate_terms(problem.objective_terms, bits)
    constraint = problem.constraint_constant + _evaluate_terms(problem.constraint_terms, bits)
    return _round_energy(objective + penalty * constraint)


def exact_enumerate(problem: HuboProblem, *, penalty: float) -> ExactHuboEnumeration:
    """Enumerate every binary state for a tiny HUBO instance."""

    all_states = tuple(itertools.product((0, 1), repeat=problem.n_vars))
    energies = {
        state: evaluate_hubo_energy(problem, state, penalty=penalty)
        for state in all_states
    }
    optimum = min(energies.values())
    optimal_states = tuple(
        sorted(state for state, energy in energies.items() if energy == optimum)
    )
    distribution = Counter(energies.values())
    return ExactHuboEnumeration(
        optimum_energy=optimum,
        optimal_states=optimal_states,
        energy_distribution=dict(sorted(distribution.items())),
        all_states=all_states,
    )


def metropolis_swap_log_acceptance(
    problem: HuboProblem,
    left_state: Sequence[int],
    right_state: Sequence[int],
    *,
    beta_left: float,
    penalty_left: float,
    beta_right: float,
    penalty_right: float,
) -> float:
    """Return the log Metropolis ratio for swapping two fixed parameter slots."""

    left_current = evaluate_hubo_energy(problem, left_state, penalty=penalty_left)
    right_current = evaluate_hubo_energy(problem, right_state, penalty=penalty_right)
    left_after = evaluate_hubo_energy(problem, right_state, penalty=penalty_left)
    right_after = evaluate_hubo_energy(problem, left_state, penalty=penalty_right)
    return (
        -beta_left * left_after
        - beta_right * right_after
        + beta_left * left_current
        + beta_right * right_current
    )


def run_unguided_gibbs(
    problem: HuboProblem,
    *,
    seed: int,
    beta: float,
    penalty: float,
    sweeps: int,
) -> HuboRunResult:
    """Run a single fixed-parameter Gibbs chain as the unguided baseline."""

    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if sweeps < 1:
        raise ValueError("sweeps must be at least 1")
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, size=problem.n_vars, dtype=np.int8)
    best_state = tuple(int(value) for value in state)
    best_energy = evaluate_hubo_energy(problem, best_state, penalty=penalty)
    trace: list[float] = []

    for _ in range(sweeps):
        state = _gibbs_sweep(problem, state, beta=beta, penalty=penalty, rng=rng)
        energy = evaluate_hubo_energy(problem, state, penalty=penalty)
        if energy < best_energy:
            best_energy = energy
            best_state = tuple(int(value) for value in state)
        trace.append(best_energy)

    final_state = tuple(int(value) for value in state)
    return HuboRunResult(
        algorithm="unguided_gibbs",
        best_energy=best_energy,
        final_energy=evaluate_hubo_energy(problem, final_state, penalty=penalty),
        best_state=best_state,
        final_state=final_state,
        energy_trace=tuple(trace),
        swap_stats={
            "beta_axis": SwapStats(),
            "penalty_axis": SwapStats(),
        },
        beta_grid=(beta,),
        penalty_grid=(penalty,),
    )


def run_beta_parallel_tempering(
    problem: HuboProblem,
    *,
    seed: int,
    beta_grid: Sequence[float],
    penalty: float,
    sweeps: int,
    swap_interval: int = 1,
) -> HuboRunResult:
    """Run one-dimensional beta parallel tempering at fixed penalty."""

    beta_tuple = tuple(float(value) for value in beta_grid)
    _validate_positive_grid(beta_tuple, "beta_grid")
    if sweeps < 1:
        raise ValueError("sweeps must be at least 1")
    if swap_interval < 1:
        raise ValueError("swap_interval must be at least 1")

    rng = np.random.default_rng(seed)
    states = rng.integers(0, 2, size=(len(beta_tuple), problem.n_vars), dtype=np.int8)
    energies = np.asarray(
        [evaluate_hubo_energy(problem, state, penalty=penalty) for state in states],
        dtype=np.float64,
    )
    best_energy, best_state = _best_target_state_1d(problem, states, target_penalty=penalty)
    beta_stats = SwapStats()
    trace: list[float] = []

    for sweep in range(1, sweeps + 1):
        for beta_index, beta in enumerate(beta_tuple):
            states[beta_index] = _gibbs_sweep(
                problem,
                states[beta_index],
                beta=beta,
                penalty=penalty,
                rng=rng,
            )
            energies[beta_index] = evaluate_hubo_energy(problem, states[beta_index], penalty=penalty)
        if sweep % swap_interval == 0:
            phase = (sweep // swap_interval) % 2
            for left in range(phase, len(beta_tuple) - 1, 2):
                right = left + 1
                log_accept = metropolis_swap_log_acceptance(
                    problem,
                    states[left],
                    states[right],
                    beta_left=beta_tuple[left],
                    penalty_left=penalty,
                    beta_right=beta_tuple[right],
                    penalty_right=penalty,
                )
                accepted = _accept_log_ratio(log_accept, rng)
                beta_stats = beta_stats.with_attempt(accepted=accepted)
                if accepted:
                    states[[left, right]] = states[[right, left]]
                    energies[[left, right]] = energies[[right, left]]

        current_best, current_state = _best_target_state_1d(problem, states, target_penalty=penalty)
        if current_best < best_energy:
            best_energy = current_best
            best_state = current_state
        trace.append(best_energy)

    final_state = tuple(int(value) for value in states[-1])
    return HuboRunResult(
        algorithm="beta_pt",
        best_energy=best_energy,
        final_energy=evaluate_hubo_energy(problem, final_state, penalty=penalty),
        best_state=best_state,
        final_state=final_state,
        energy_trace=tuple(trace),
        swap_stats={
            "beta_axis": beta_stats,
            "penalty_axis": SwapStats(),
        },
        beta_grid=beta_tuple,
        penalty_grid=(penalty,),
    )


def adapt_inverse_temperature_ladder(
    beta_grid: Sequence[float],
    *,
    pair_acceptance_rates: Sequence[float],
    target_acceptance: float = 0.45,
    learning_rate: float = 0.5,
    min_beta_gap: float = 0.05,
) -> tuple[float, ...]:
    """Return an ordered beta ladder adapted from adjacent-pair acceptance.

    Large gaps suppress swaps, while tiny gaps waste replicas on nearly
    identical distributions. The update expands gaps whose recent acceptance
    is above target and shrinks gaps whose acceptance is below target, then
    rescales the interior points so the original hot and cold endpoints remain
    fixed for a fair comparison against the fixed-grid baseline.
    """

    beta_tuple = tuple(float(value) for value in beta_grid)
    _validate_positive_grid(beta_tuple, "beta_grid")
    if len(beta_tuple) < 2:
        raise ValueError("beta_grid must contain at least two betas")
    rates = tuple(float(value) for value in pair_acceptance_rates)
    if len(rates) != len(beta_tuple) - 1:
        raise ValueError("pair_acceptance_rates must have one value per adjacent beta pair")
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError("pair_acceptance_rates must be in [0, 1]")
    if not 0.0 < target_acceptance < 1.0:
        raise ValueError("target_acceptance must be in (0, 1)")
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be nonnegative")
    if min_beta_gap <= 0.0:
        raise ValueError("min_beta_gap must be positive")

    span = beta_tuple[-1] - beta_tuple[0]
    gap_count = len(beta_tuple) - 1
    gap_floor = min(float(min_beta_gap), span / (2.0 * gap_count))
    old_gaps = np.diff(np.asarray(beta_tuple, dtype=np.float64))
    rate_array = np.asarray(rates, dtype=np.float64)
    scaled_gaps = old_gaps * np.exp(learning_rate * (rate_array - target_acceptance))
    weights = np.maximum(scaled_gaps, 1e-12)
    flexible_span = span - gap_floor * gap_count
    new_gaps = gap_floor + flexible_span * weights / float(np.sum(weights))
    updated = np.concatenate(
        (
            np.asarray([beta_tuple[0]], dtype=np.float64),
            beta_tuple[0] + np.cumsum(new_gaps),
        )
    )
    updated[0] = beta_tuple[0]
    updated[-1] = beta_tuple[-1]
    return _round_grid(updated)


def _validate_positive_grid(values: Sequence[float], name: str) -> None:
    if len(values) < 1:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{name} values must be positive")
    if tuple(values) != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted ascending")


def _evaluate_terms(terms: Sequence[HuboTerm], bits: Sequence[int]) -> float:
    total = 0.0
    for term in terms:
        product = 1.0
        for variable in term.variables:
            product *= 1.0 if bits[variable] == 0 else -1.0
        total += term.coefficient * product
    return total


def _round_energy(value: float) -> float:
    return round(float(value), 12)


def _gibbs_sweep(
    problem: HuboProblem,
    state: np.ndarray,
    *,
    beta: float,
    penalty: float,
    rng: np.random.Generator,
) -> np.ndarray:
    next_state = np.asarray(state, dtype=np.int8).copy()
    for variable in range(problem.n_vars):
        candidate_zero = next_state.copy()
        candidate_one = next_state.copy()
        candidate_zero[variable] = 0
        candidate_one[variable] = 1
        energy_zero = evaluate_hubo_energy(problem, candidate_zero, penalty=penalty)
        energy_one = evaluate_hubo_energy(problem, candidate_one, penalty=penalty)
        probability_one = _logistic(-beta * (energy_one - energy_zero))
        next_state[variable] = 1 if rng.random() < probability_one else 0
    return next_state


def _logistic(value: float) -> float:
    clipped = min(60.0, max(-60.0, value))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _slot_energies(
    problem: HuboProblem,
    states: np.ndarray,
    penalty_grid: Sequence[float],
) -> np.ndarray:
    energies = np.zeros(states.shape[:2], dtype=np.float64)
    for beta_index in range(states.shape[0]):
        for penalty_index, penalty in enumerate(penalty_grid):
            energies[beta_index, penalty_index] = evaluate_hubo_energy(
                problem,
                states[beta_index, penalty_index],
                penalty=penalty,
            )
    return energies


def _best_target_state(
    problem: HuboProblem,
    states: np.ndarray,
    *,
    target_penalty: float,
) -> tuple[float, tuple[int, ...]]:
    best_energy: float | None = None
    best_state: tuple[int, ...] | None = None
    for state in states.reshape((-1, problem.n_vars)):
        state_tuple = tuple(int(value) for value in state)
        energy = evaluate_hubo_energy(problem, state_tuple, penalty=target_penalty)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_state = state_tuple
    if best_energy is None or best_state is None:
        raise ValueError("state grid must not be empty")
    return best_energy, best_state


def _best_target_state_1d(
    problem: HuboProblem,
    states: np.ndarray,
    *,
    target_penalty: float,
) -> tuple[float, tuple[int, ...]]:
    best_energy: float | None = None
    best_state: tuple[int, ...] | None = None
    for state in states:
        state_tuple = tuple(int(value) for value in state)
        energy = evaluate_hubo_energy(problem, state_tuple, penalty=target_penalty)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_state = state_tuple
    if best_energy is None or best_state is None:
        raise ValueError("state ladder must not be empty")
    return best_energy, best_state


def _swap_beta_axis(
    problem: HuboProblem,
    states: np.ndarray,
    energies: np.ndarray,
    *,
    beta_grid: Sequence[float],
    penalty_grid: Sequence[float],
    stats: SwapStats,
    rng: np.random.Generator,
    phase: int,
) -> tuple[np.ndarray, np.ndarray, SwapStats]:
    for penalty_index, penalty in enumerate(penalty_grid):
        for left in range(phase, len(beta_grid) - 1, 2):
            right = left + 1
            log_accept = metropolis_swap_log_acceptance(
                problem,
                states[left, penalty_index],
                states[right, penalty_index],
                beta_left=beta_grid[left],
                penalty_left=penalty,
                beta_right=beta_grid[right],
                penalty_right=penalty,
            )
            accepted = _accept_log_ratio(log_accept, rng)
            stats = stats.with_attempt(accepted=accepted)
            if accepted:
                states[[left, right], penalty_index] = states[[right, left], penalty_index]
                energies[[left, right], penalty_index] = energies[[right, left], penalty_index]
    return states, energies, stats


def _adaptive_swap_beta_axis(
    problem: HuboProblem,
    states: np.ndarray,
    energies: np.ndarray,
    beta_tags: np.ndarray,
    *,
    beta_grid: Sequence[float],
    penalty_grid: Sequence[float],
    stats: SwapStats,
    pair_totals: list[SwapStats],
    pair_window: list[SwapStats],
    accepted_up_moves: int,
    accepted_down_moves: int,
    rng: np.random.Generator,
    phase: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SwapStats, list[SwapStats], list[SwapStats], int, int]:
    for penalty_index, penalty in enumerate(penalty_grid):
        for left in range(phase, len(beta_grid) - 1, 2):
            right = left + 1
            log_accept = metropolis_swap_log_acceptance(
                problem,
                states[left, penalty_index],
                states[right, penalty_index],
                beta_left=beta_grid[left],
                penalty_left=penalty,
                beta_right=beta_grid[right],
                penalty_right=penalty,
            )
            accepted = _accept_log_ratio(log_accept, rng)
            stats = stats.with_attempt(accepted=accepted)
            pair_totals[left] = pair_totals[left].with_attempt(accepted=accepted)
            pair_window[left] = pair_window[left].with_attempt(accepted=accepted)
            if accepted:
                states[[left, right], penalty_index] = states[[right, left], penalty_index]
                energies[[left, right], penalty_index] = energies[[right, left], penalty_index]
                beta_tags[[left, right], penalty_index] = beta_tags[[right, left], penalty_index]
                accepted_up_moves += 1
                accepted_down_moves += 1
    return (
        states,
        energies,
        beta_tags,
        stats,
        pair_totals,
        pair_window,
        accepted_up_moves,
        accepted_down_moves,
    )


def _swap_penalty_axis(
    problem: HuboProblem,
    states: np.ndarray,
    energies: np.ndarray,
    *,
    beta_grid: Sequence[float],
    penalty_grid: Sequence[float],
    stats: SwapStats,
    rng: np.random.Generator,
    phase: int,
) -> tuple[np.ndarray, np.ndarray, SwapStats]:
    for beta_index, beta in enumerate(beta_grid):
        for lower in range(phase, len(penalty_grid) - 1, 2):
            upper = lower + 1
            left_state = states[beta_index, lower].copy()
            right_state = states[beta_index, upper].copy()
            log_accept = metropolis_swap_log_acceptance(
                problem,
                left_state,
                right_state,
                beta_left=beta,
                penalty_left=penalty_grid[lower],
                beta_right=beta,
                penalty_right=penalty_grid[upper],
            )
            accepted = _accept_log_ratio(log_accept, rng)
            stats = stats.with_attempt(accepted=accepted)
            if accepted:
                states[beta_index, lower] = right_state
                states[beta_index, upper] = left_state
                energies[beta_index, lower] = evaluate_hubo_energy(
                    problem,
                    states[beta_index, lower],
                    penalty=penalty_grid[lower],
                )
                energies[beta_index, upper] = evaluate_hubo_energy(
                    problem,
                    states[beta_index, upper],
                    penalty=penalty_grid[upper],
                )
    return states, energies, stats


def _accept_log_ratio(log_accept: float, rng: np.random.Generator) -> bool:
    if log_accept >= 0.0:
        return True
    return bool(math.log(max(rng.random(), 1e-300)) < log_accept)


def _detailed_balance_sanity_for_slots(
    problem: HuboProblem,
    states: np.ndarray,
    *,
    beta_grid: Sequence[float],
    penalty_grid: Sequence[float],
    accepted_up_moves: int,
    accepted_down_moves: int,
) -> dict[str, Any]:
    max_abs = 0.0
    checks = 0
    for penalty_index, penalty in enumerate(penalty_grid):
        for left in range(len(beta_grid) - 1):
            right = left + 1
            forward = metropolis_swap_log_acceptance(
                problem,
                states[left, penalty_index],
                states[right, penalty_index],
                beta_left=beta_grid[left],
                penalty_left=penalty,
                beta_right=beta_grid[right],
                penalty_right=penalty,
            )
            backward = metropolis_swap_log_acceptance(
                problem,
                states[right, penalty_index],
                states[left, penalty_index],
                beta_left=beta_grid[left],
                penalty_left=penalty,
                beta_right=beta_grid[right],
                penalty_right=penalty,
            )
            max_abs = max(max_abs, abs(forward + backward))
            checks += 1
    net_flow = abs(int(accepted_up_moves) - int(accepted_down_moves))
    return {
        "checks": checks,
        "local_log_ratio_antisymmetry_max_abs": _round_metric(max_abs),
        "trajectory_forward_moves": int(accepted_up_moves),
        "trajectory_backward_moves": int(accepted_down_moves),
        "trajectory_net_flow_abs": net_flow,
        "passed": bool(checks > 0 and max_abs <= 1e-9 and net_flow == 0),
    }


def _round_trip_proxy(
    min_seen: np.ndarray,
    max_seen: np.ndarray,
    accepted_up_moves: int,
    accepted_down_moves: int,
) -> dict[str, Any]:
    spans = np.asarray(max_seen - min_seen, dtype=np.float64)
    denominator = max(1, int(min_seen.shape[0]) - 1)
    span_fraction = spans / float(denominator)
    return {
        "tag_count": int(spans.size),
        "full_span_tag_count": int(np.sum(spans >= denominator)),
        "mean_beta_span_fraction": _round_metric(float(np.mean(span_fraction))),
        "trajectory_forward_moves": int(accepted_up_moves),
        "trajectory_backward_moves": int(accepted_down_moves),
        "trajectory_net_flow_abs": abs(int(accepted_up_moves) - int(accepted_down_moves)),
    }


def _round_grid(values: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    return tuple(_round_metric(float(value)) for value in values)


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _is_strictly_increasing(values: Sequence[float]) -> bool:
    return all(right > left for left, right in zip(values, values[1:]))
