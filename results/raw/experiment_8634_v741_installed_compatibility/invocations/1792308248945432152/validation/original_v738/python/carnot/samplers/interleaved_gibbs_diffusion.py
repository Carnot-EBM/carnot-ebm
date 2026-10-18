"""Interleaved Gibbs Diffusion sampler for q=2 Potts MAX-3-SAT.

**Researcher summary:**
    This module keeps a boolean SAT assignment as discrete q=2 Potts states
    while also maintaining continuous per-state logits.  Each IGD sweep first
    diffuses the logits with Gaussian noise, then performs a full Gibbs pass
    over the discrete variables using the MAX-3-SAT conditional energy plus the
    current logit bias.  A sequential Gibbs baseline is provided for paired
    benchmark comparisons.

**Detailed explanation for engineers:**
    A MAX-3-SAT assignment is naturally discrete: every variable is either
    false or true.  IGD adds a continuous companion state, here a two-column
    logit matrix.  The logits let the chain carry soft momentum between
    discrete updates without making the final assignment non-boolean.  The
    continuous noise injection is deliberately simple and CPU-only; it is meant
    to test the mixed-state sampling contract before introducing learned
    denoisers or hardware backends.

Spec: REQ-IGD-1961, REQ-IGD-1961-1, REQ-IGD-1961-2, REQ-IGD-1961-3,
      REQ-IGD-1961-4, REQ-IGD-1961-5, SCENARIO-IGD-1961
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SPEC_REFS = [
    "REQ-IGD-1961",
    "REQ-IGD-1961-1",
    "REQ-IGD-1961-2",
    "REQ-IGD-1961-3",
    "REQ-IGD-1961-4",
    "REQ-IGD-1961-5",
    "SCENARIO-IGD-1961",
]


@dataclass(frozen=True)
class Max3SatInstance:
    """Synthetic MAX-3-SAT problem with q=2 Potts assignment semantics.

    State convention: Potts state ``0`` means boolean false and state ``1``
    means boolean true.  Clauses use signed one-based literals, so ``-3`` means
    variable three appears negated.  Lower energy means fewer unsatisfied
    clauses.

    Spec: REQ-IGD-1961-3
    """

    num_variables: int
    clauses: np.ndarray
    planted_assignment: np.ndarray
    q: int = 2

    def __post_init__(self) -> None:
        clauses = np.asarray(self.clauses, dtype=np.int64)
        planted = np.asarray(self.planted_assignment, dtype=np.int64)
        if self.num_variables < 3:
            raise ValueError("num_variables must be >= 3")  # pragma: no cover
        if self.q != 2:
            raise ValueError("MAX-3-SAT Potts encoding requires q=2")  # pragma: no cover
        if clauses.ndim != 2 or clauses.shape[1] != 3:
            raise ValueError("clauses must have shape (num_clauses, 3)")  # pragma: no cover
        if planted.shape != (self.num_variables,):
            raise ValueError("planted_assignment has wrong shape")  # pragma: no cover
        if np.any((planted < 0) | (planted >= self.q)):
            raise ValueError("planted_assignment must contain q=2 Potts states")  # pragma: no cover
        if np.any(clauses == 0) or np.max(np.abs(clauses)) > self.num_variables:
            raise ValueError("clauses contain invalid literal indices")  # pragma: no cover
        object.__setattr__(self, "clauses", clauses)
        object.__setattr__(self, "planted_assignment", planted)

    @property
    def num_clauses(self) -> int:
        """Number of three-literal clauses in the instance."""
        return int(self.clauses.shape[0])

    def count_satisfied(self, state: np.ndarray) -> int:
        """Count clauses satisfied by a q=2 Potts assignment."""
        assignment = _validate_state(state, self.num_variables, self.q)
        satisfied = 0
        for clause in self.clauses:
            if any(_literal_satisfied(int(literal), assignment) for literal in clause):
                satisfied += 1
        return satisfied

    def energy(self, state: np.ndarray) -> int:
        """Return MAX-3-SAT energy, defined as unsatisfied clause count."""
        return self.num_clauses - self.count_satisfied(state)


@dataclass(frozen=True)
class SamplerRun:
    """Result from an IGD or baseline Gibbs chain.

    Spec: REQ-IGD-1961-4
    """

    sampler_name: str
    final_state: np.ndarray
    final_logits: np.ndarray
    satisfaction_history: list[int]
    energy_history: list[int]
    continuous_noise_norms: list[float]
    target_satisfied: int

    @property
    def best_satisfied(self) -> int:
        """Best MAX-3-SAT satisfaction reached by the chain."""
        return int(max(self.satisfaction_history))

    @property
    def mixing_time(self) -> int | None:
        """First sweep at or above the target satisfaction threshold."""
        for sweep, satisfied in enumerate(self.satisfaction_history):
            if satisfied >= self.target_satisfied:
                return int(sweep)
        return None  # pragma: no cover

    @property
    def convergence_rate(self) -> float:
        """Average satisfied-clause gain per sweep."""
        n_steps = len(self.satisfaction_history) - 1
        if n_steps <= 0:
            return 0.0  # pragma: no cover
        delta = self.satisfaction_history[-1] - self.satisfaction_history[0]
        return float(delta / n_steps)

    def to_metrics(self) -> dict[str, Any]:
        """Convert run statistics to JSON-safe benchmark metrics."""
        return {
            "sampler": self.sampler_name,
            "initial_satisfied": int(self.satisfaction_history[0]),
            "final_satisfied": int(self.satisfaction_history[-1]),
            "best_satisfied": self.best_satisfied,
            "target_satisfied": int(self.target_satisfied),
            "mixing_time_sweeps": self.mixing_time,
            "convergence_rate": self.convergence_rate,
            "finite_logits": bool(np.all(np.isfinite(self.final_logits))),
            "continuous_noise_injections": len(self.continuous_noise_norms),
            "mean_noise_norm": float(np.mean(self.continuous_noise_norms))
            if self.continuous_noise_norms
            else 0.0,
            "satisfaction_history": [int(value) for value in self.satisfaction_history],
            "energy_history": [int(value) for value in self.energy_history],
            "final_state": [int(value) for value in self.final_state],
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the deterministic Exp 1961 benchmark."""

    num_variables: int = 12
    num_clauses: int = 42
    n_sweeps: int = 28
    seed: int = 1961
    beta: float = 3.5
    noise_scale: float = 0.15
    logit_coupling: float = 0.7
    target_satisfaction_ratio: float = 0.95


@dataclass(frozen=True)
class InterleavedGibbsDiffusionSampler:
    """Mixed continuous-discrete Gibbs sampler for q=2 Potts MAX-3-SAT.

    Spec: REQ-IGD-1961-1, REQ-IGD-1961-2
    """

    beta: float = 3.5
    noise_scale: float = 0.15
    logit_coupling: float = 0.7
    seed: int = 1961

    def sample(
        self,
        problem: Max3SatInstance,
        n_sweeps: int,
        init_state: np.ndarray | None = None,
    ) -> SamplerRun:
        """Run IGD sweeps and return discrete, continuous, and convergence state."""
        rng = np.random.default_rng(self.seed)
        state = _initial_state(problem, rng, init_state)
        logits = potts_one_hot(state, problem.q)
        target = _target_satisfied(problem, 0.95)
        satisfaction_history = [problem.count_satisfied(state)]
        energy_history = [problem.energy(state)]
        noise_norms: list[float] = []

        for _ in range(n_sweeps):
            logits, noise_norm = self._inject_continuous_noise(rng, logits, state, problem.q)
            noise_norms.append(noise_norm)
            state = self._discrete_sweep(problem, state, logits, rng)
            satisfaction_history.append(problem.count_satisfied(state))
            energy_history.append(problem.energy(state))

        return SamplerRun(
            sampler_name="interleaved_gibbs_diffusion",
            final_state=state,
            final_logits=logits,
            satisfaction_history=satisfaction_history,
            energy_history=energy_history,
            continuous_noise_norms=noise_norms,
            target_satisfied=target,
        )

    def _inject_continuous_noise(
        self,
        rng: np.random.Generator,
        logits: np.ndarray,
        state: np.ndarray,
        q: int,
    ) -> tuple[np.ndarray, float]:
        """Diffuse logits around the current Potts one-hot encoding."""
        centered_one_hot = 2.0 * potts_one_hot(state, q) - 1.0
        noise = rng.normal(loc=0.0, scale=self.noise_scale, size=logits.shape)
        next_logits = 0.8 * logits + 0.2 * centered_one_hot + noise
        return next_logits, float(np.linalg.norm(noise))

    def _discrete_sweep(
        self,
        problem: Max3SatInstance,
        state: np.ndarray,
        logits: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Update every discrete Potts variable using conditional SAT energies."""
        updated = state.copy()
        for variable in rng.permutation(problem.num_variables):
            scores = []
            for candidate in range(problem.q):
                trial = updated.copy()
                trial[int(variable)] = candidate
                scores.append(-self.beta * problem.energy(trial) + self.logit_coupling * logits[variable, candidate])
            updated[int(variable)] = _draw_category(rng, _softmax(np.asarray(scores, dtype=np.float64)))
        return updated


@dataclass(frozen=True)
class SequentialGibbsSampler:
    """Discrete-only sequential Gibbs baseline for q=2 Potts MAX-3-SAT.

    Spec: REQ-IGD-1961-4
    """

    beta: float = 3.5
    seed: int = 1962

    def sample(
        self,
        problem: Max3SatInstance,
        n_sweeps: int,
        init_state: np.ndarray | None = None,
    ) -> SamplerRun:
        """Run baseline Gibbs sweeps without any continuous logit diffusion."""
        rng = np.random.default_rng(self.seed)
        state = _initial_state(problem, rng, init_state)
        target = _target_satisfied(problem, 0.95)
        satisfaction_history = [problem.count_satisfied(state)]
        energy_history = [problem.energy(state)]

        for _ in range(n_sweeps):
            for variable in range(problem.num_variables):
                scores = []
                for candidate in range(problem.q):
                    trial = state.copy()
                    trial[variable] = candidate
                    scores.append(-self.beta * problem.energy(trial))
                state[variable] = _draw_category(rng, _softmax(np.asarray(scores, dtype=np.float64)))
            satisfaction_history.append(problem.count_satisfied(state))
            energy_history.append(problem.energy(state))

        return SamplerRun(
            sampler_name="sequential_gibbs",
            final_state=state,
            final_logits=potts_one_hot(state, problem.q),
            satisfaction_history=satisfaction_history,
            energy_history=energy_history,
            continuous_noise_norms=[],
            target_satisfied=target,
        )


def generate_synthetic_max3sat(
    num_variables: int = 12,
    num_clauses: int = 42,
    seed: int = 1961,
) -> Max3SatInstance:
    """Generate a deterministic planted-solution MAX-3-SAT benchmark instance.

    Spec: REQ-IGD-1961-3
    """
    rng = np.random.default_rng(seed)
    if num_variables < 3:
        raise ValueError("num_variables must be >= 3")  # pragma: no cover
    planted = rng.integers(0, 2, size=int(num_variables), dtype=np.int64)
    clauses = []
    for _ in range(int(num_clauses)):
        variables = rng.choice(num_variables, size=3, replace=False)
        signs = rng.choice(np.array([-1, 1], dtype=np.int64), size=3)
        literals = signs * (variables + 1)
        if not any(_literal_satisfied(int(literal), planted) for literal in literals):
            repair_idx = int(rng.integers(0, 3))
            variable = int(variables[repair_idx])
            signs[repair_idx] = 1 if planted[variable] == 1 else -1
            literals = signs * (variables + 1)
        clauses.append(literals)
    return Max3SatInstance(
        num_variables=int(num_variables),
        clauses=np.asarray(clauses, dtype=np.int64),
        planted_assignment=planted,
    )


def potts_one_hot(state: np.ndarray, q: int = 2) -> np.ndarray:
    """Return one-hot q-state Potts encoding for an integer state vector."""
    assignment = _validate_state(state, len(state), q)
    encoded = np.zeros((assignment.size, int(q)), dtype=np.float64)
    encoded[np.arange(assignment.size), assignment] = 1.0
    return encoded


def run_max3sat_benchmark(config: BenchmarkConfig = BenchmarkConfig()) -> dict[str, Any]:
    """Run IGD and sequential Gibbs on the same deterministic MAX-3-SAT instance.

    Spec: REQ-IGD-1961-4, REQ-IGD-1961-5, SCENARIO-IGD-1961
    """
    problem = generate_synthetic_max3sat(
        num_variables=config.num_variables,
        num_clauses=config.num_clauses,
        seed=config.seed,
    )
    init_rng = np.random.default_rng(config.seed + 100)
    init_state = init_rng.integers(0, problem.q, size=problem.num_variables, dtype=np.int64)
    target_satisfied = _target_satisfied(problem, config.target_satisfaction_ratio)

    igd = InterleavedGibbsDiffusionSampler(
        beta=config.beta,
        noise_scale=config.noise_scale,
        logit_coupling=config.logit_coupling,
        seed=config.seed + 1,
    )
    sequential = SequentialGibbsSampler(beta=config.beta, seed=config.seed + 2)
    igd_run = _with_target(igd.sample(problem, config.n_sweeps, init_state), target_satisfied)
    sequential_run = _with_target(sequential.sample(problem, config.n_sweeps, init_state), target_satisfied)

    igd_metrics = igd_run.to_metrics()
    sequential_metrics = sequential_run.to_metrics()
    mixing_delta = _mixing_delta(igd_run.mixing_time, sequential_run.mixing_time)
    convergence_delta = float(igd_run.convergence_rate - sequential_run.convergence_rate)
    best_delta = int(igd_run.best_satisfied - sequential_run.best_satisfied)
    improved = best_delta >= 0 and (mixing_delta is None or mixing_delta >= 0)

    return {
        "experiment_id": "1961",
        "title": "Interleaved Gibbs Diffusion MAX-3-SAT Potts Benchmark",
        "spec_refs": SPEC_REFS,
        "problem": {
            "name": "synthetic_planted_max_3sat",
            "num_variables": problem.num_variables,
            "num_clauses": problem.num_clauses,
            "clause_width": 3,
            "q": problem.q,
            "seed": config.seed,
            "target_satisfied": target_satisfied,
            "planted_satisfied": problem.count_satisfied(problem.planted_assignment),
        },
        "samplers": {
            "igd": {
                "beta": config.beta,
                "noise_scale": config.noise_scale,
                "logit_coupling": config.logit_coupling,
                "seed": config.seed + 1,
                "uses_continuous_noise": True,
            },
            "sequential_gibbs": {
                "beta": config.beta,
                "seed": config.seed + 2,
                "uses_continuous_noise": False,
            },
        },
        "metrics": {
            "igd": igd_metrics,
            "sequential_gibbs": sequential_metrics,
        },
        "comparison": {
            "best_satisfied_delta": best_delta,
            "mixing_time_delta_sweeps": mixing_delta,
            "convergence_rate_delta": convergence_delta,
        },
        "honest_verdict": "igd_mixed_sampler_benchmark_complete"
        if improved
        else "igd_mixed_sampler_no_baseline_improvement",
    }


def _literal_satisfied(literal: int, state: np.ndarray) -> bool:
    variable = abs(literal) - 1
    value = bool(state[variable])
    return value if literal > 0 else not value


def _validate_state(state: np.ndarray, num_variables: int, q: int) -> np.ndarray:
    assignment = np.asarray(state, dtype=np.int64)
    if assignment.shape != (int(num_variables),):
        raise ValueError("state has wrong shape")  # pragma: no cover
    if np.any((assignment < 0) | (assignment >= int(q))):
        raise ValueError("state contains invalid Potts values")  # pragma: no cover
    return assignment


def _initial_state(
    problem: Max3SatInstance,
    rng: np.random.Generator,
    init_state: np.ndarray | None,
) -> np.ndarray:
    if init_state is None:
        return rng.integers(0, problem.q, size=problem.num_variables, dtype=np.int64)
    return _validate_state(init_state, problem.num_variables, problem.q).copy()


def _target_satisfied(problem: Max3SatInstance, ratio: float) -> int:
    return int(np.ceil(float(ratio) * problem.num_clauses))


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / np.sum(weights)


def _draw_category(rng: np.random.Generator, probabilities: np.ndarray) -> int:
    cumulative = np.cumsum(probabilities)
    selected = int(np.searchsorted(cumulative, rng.random(), side="right"))
    return min(selected, probabilities.size - 1)


def _with_target(run: SamplerRun, target_satisfied: int) -> SamplerRun:
    return SamplerRun(
        sampler_name=run.sampler_name,
        final_state=run.final_state,
        final_logits=run.final_logits,
        satisfaction_history=run.satisfaction_history,
        energy_history=run.energy_history,
        continuous_noise_norms=run.continuous_noise_norms,
        target_satisfied=target_satisfied,
    )


def _mixing_delta(igd_mixing: int | None, sequential_mixing: int | None) -> int | None:
    if igd_mixing is None or sequential_mixing is None:
        return None
    return int(sequential_mixing - igd_mixing)
