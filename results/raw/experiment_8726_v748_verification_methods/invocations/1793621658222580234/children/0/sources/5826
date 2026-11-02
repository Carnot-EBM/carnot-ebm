"""Hard-BRS and Soft-Gibbs residual rejection sampling.

Spec: REQ-SAMPLE-059, SCENARIO-SAMPLE-087.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1565_soft_gibbs_residual_implementation.json"
)
N_STEP_VALUES = (10, 100, 1000, 10000)
BETA_VALUES = (0.5, 1.0, 2.0, 5.0)
DECAY_CONFIRMATION_BETA = 5.0
DECAY_ABS_TOLERANCE = 0.08
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "soft_gibbs_residual_implemented",
    "hard_brs_acceptance_rate",
    "soft_brs_decay_confirmed",
    "min_violation_state_found",
    "z_beta_curve",
    "honest_verdict",
}

SpinState = tuple[int, ...]
Verifier = Callable[[SpinState], bool]
PriorSampler = Callable[[], Sequence[int] | np.ndarray]


@dataclass(frozen=True)
class HardBRSResult:
    """Result of a finite-proposal hard rejection-sampling run."""

    total_steps: int
    accepted_samples: tuple[SpinState, ...]

    @property
    def accepted_count(self) -> int:
        """Number of proposals accepted into the hard residual set."""

        return len(self.accepted_samples)

    @property
    def acceptance_rate(self) -> float:
        """Accepted proposals divided by all proposals."""

        return self.accepted_count / self.total_steps if self.total_steps else 0.0


@dataclass(frozen=True)
class SoftBRSResult(HardBRSResult):
    """Result of a finite-proposal Soft-Gibbs residual sampling run."""

    beta: float
    proposal_violation_trace: tuple[int, ...]
    acceptance_probability_trace: tuple[float, ...]
    accepted_violation_trace: tuple[int, ...]

    @property
    def best_violation(self) -> int | None:
        """Lowest residual violation count among accepted proposals."""

        return min(self.accepted_violation_trace) if self.accepted_violation_trace else None


class LatentSignPrior:
    """Uniform spin prior produced by pushing continuous latent samples through sign."""

    def __init__(self, *, n: int = 8, seed: int = 1565) -> None:
        self.n = int(n)
        self._rng = np.random.default_rng(int(seed))

    def __call__(self) -> SpinState:
        """Draw z ~ Normal(0, I) and return sgn(z) in {-1, +1}^n."""

        latent = self._rng.standard_normal(self.n)
        return tuple(int(value) for value in np.where(latent >= 0.0, 1, -1))


def hard_brs(
    prior_sampler: PriorSampler,
    accept_set_S: Callable[[SpinState], bool] | Container[SpinState],
    n_steps: int,
) -> HardBRSResult:
    """Run finite Hard-BRS from a caller-supplied prior sampler.

    Each step proposes one `y ~ mu`; a proposal is retained only when it is in
    the hard accept set `S`.
    """

    accepted: list[SpinState] = []
    total_steps = int(n_steps)
    for _ in range(total_steps):
        state = _coerce_state(prior_sampler())
        if _accepts(accept_set_S, state):
            accepted.append(state)
    return HardBRSResult(total_steps=total_steps, accepted_samples=tuple(accepted))


def soft_brs(
    prior_sampler: PriorSampler,
    verifiers: Iterable[Verifier],
    beta: float,
    n_steps: int,
    *,
    rng: Any | None = None,
) -> SoftBRSResult:
    """Run finite Soft-BRS with Soft-Gibbs residual acceptance.

    The residual count is `V(y) = sum_i 1{verifier_i(y) is false}` and the
    acceptance probability is `A(y) = exp(-beta * V(y))`.
    """

    verifier_tuple = tuple(verifiers)
    random_source = np.random.default_rng(0) if rng is None else rng
    accepted: list[SpinState] = []
    accepted_violations: list[int] = []
    proposal_violations: list[int] = []
    acceptance_probabilities: list[float] = []
    total_steps = int(n_steps)
    beta_value = float(beta)

    for _ in range(total_steps):
        state = _coerce_state(prior_sampler())
        violations = violation_count(state, verifier_tuple)
        accept_probability = math.exp(-beta_value * violations)
        proposal_violations.append(violations)
        acceptance_probabilities.append(accept_probability)
        if float(random_source.random()) < accept_probability:
            accepted.append(state)
            accepted_violations.append(violations)

    return SoftBRSResult(
        total_steps=total_steps,
        accepted_samples=tuple(accepted),
        beta=beta_value,
        proposal_violation_trace=tuple(proposal_violations),
        acceptance_probability_trace=tuple(acceptance_probabilities),
        accepted_violation_trace=tuple(accepted_violations),
    )


def enumerate_spin_states(n: int) -> tuple[SpinState, ...]:
    """Enumerate all {-1, +1} spin states for an exact finite prior audit."""

    return tuple(tuple(int(value) for value in state) for state in itertools.product((-1, 1), repeat=int(n)))


def contradictory_verifiers() -> tuple[Verifier, Verifier, Verifier]:
    """Return the Exp 1565 verifier triple with empty hard intersection."""

    def requires_y1_positive(y: SpinState) -> bool:
        return y[0] == 1

    def requires_y1_negative(y: SpinState) -> bool:
        return y[0] == -1

    def requires_y2_or_y3_positive(y: SpinState) -> bool:
        return y[1] == 1 or y[2] == 1

    return (requires_y1_positive, requires_y1_negative, requires_y2_or_y3_positive)


def violation_count(state: SpinState, verifiers: Iterable[Verifier]) -> int:
    """Count failed verifier predicates for one spin state."""

    return sum(0 if verifier(state) else 1 for verifier in verifiers)


def exact_violation_distribution(
    states: Iterable[SpinState],
    verifiers: Iterable[Verifier],
) -> dict[int, int]:
    """Return exact counts by residual violation count over supplied states."""

    verifier_tuple = tuple(verifiers)
    counts = Counter(violation_count(state, verifier_tuple) for state in states)
    return dict(sorted(counts.items()))


def exact_z_beta(
    states: Iterable[SpinState],
    verifiers: Iterable[Verifier],
    beta: float,
) -> float:
    """Compute exact `Z_beta = E_mu[exp(-beta V(y))]` for a finite uniform prior."""

    verifier_tuple = tuple(verifiers)
    violations = [violation_count(state, verifier_tuple) for state in states]
    return float(np.mean(np.exp(-float(beta) * np.asarray(violations, dtype=np.float64))))


def run_experiment(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    decay_trials: int = 4096,
) -> dict[str, Any]:
    """Run Exp 1565 and write the terminal Soft-Gibbs residual artifact."""

    states = enumerate_spin_states(n=8)
    verifiers = contradictory_verifiers()
    accept_set = lambda state: all(verifier(state) for verifier in verifiers)
    exact_counts = exact_violation_distribution(states, verifiers)
    min_violation = min(exact_counts)

    hard_runs = [
        hard_brs(LatentSignPrior(n=8, seed=156500 + n_steps), accept_set, n_steps)
        for n_steps in N_STEP_VALUES
    ]
    hard_by_n = [
        {
            "n_steps": run.total_steps,
            "accepted_count": run.accepted_count,
            "empirical_acceptance_rate": run.acceptance_rate,
        }
        for run in hard_runs
    ]

    z_beta_curve: list[dict[str, float]] = []
    soft_by_beta: list[dict[str, Any]] = []
    min_violation_source: SoftBRSResult | None = None
    for beta in BETA_VALUES:
        per_n: list[dict[str, float | int | None]] = []
        final_run: SoftBRSResult | None = None
        for n_steps in N_STEP_VALUES:
            run = soft_brs(
                LatentSignPrior(n=8, seed=156510 + int(beta * 1000) + n_steps),
                verifiers,
                beta,
                n_steps,
                rng=np.random.default_rng(156520 + int(beta * 1000) + n_steps),
            )
            final_run = run
            per_n.append(
                {
                    "n_steps": run.total_steps,
                    "accepted_count": run.accepted_count,
                    "empirical_acceptance_rate": run.acceptance_rate,
                    "best_violation": run.best_violation,
                }
            )
        assert final_run is not None
        if beta == 2.0:
            min_violation_source = final_run
        z_beta_curve.append(
            {
                "beta": float(beta),
                "empirical_acceptance_rate": final_run.acceptance_rate,
                "exact_z_beta": exact_z_beta(states, verifiers, beta),
            }
        )
        soft_by_beta.append({"beta": float(beta), "runs": per_n})

    assert min_violation_source is not None
    min_violation_distribution = min_violation_state_distribution(
        min_violation_source.accepted_samples,
        verifiers,
        min_violation=min_violation,
    )
    subopt_rows, decay_confirmed = _subopt_decay_rows(
        states=states,
        verifiers=verifiers,
        beta=DECAY_CONFIRMATION_BETA,
        n_values=N_STEP_VALUES,
        n_trials=int(decay_trials),
    )

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": 1565,
            "schema": "soft_gibbs_residual_brs_v1",
            "spec_refs": ["REQ-SAMPLE-059", "SCENARIO-SAMPLE-087"],
            "n": 8,
            "n_step_values": list(N_STEP_VALUES),
            "beta_values": list(BETA_VALUES),
            "decay_confirmation_beta": DECAY_CONFIRMATION_BETA,
            "decay_trials": int(decay_trials),
        },
        "status": "complete",
        "soft_gibbs_residual_implemented": True,
        "hard_brs_acceptance_rate": max(run.acceptance_rate for run in hard_runs),
        "hard_brs_by_n": hard_by_n,
        "soft_brs_by_beta": soft_by_beta,
        "soft_brs_decay_confirmed": decay_confirmed,
        "empirical_subopt_over_time": subopt_rows,
        "min_violation_state_found": bool(min_violation_distribution),
        "min_violation_count": min_violation,
        "exact_violation_distribution": {str(key): value for key, value in exact_counts.items()},
        "min_violation_state_distribution": min_violation_distribution,
        "z_beta_curve": z_beta_curve,
        "honest_verdict": (
            "complete: soft_gibbs_residual_operational_hard_brs_empty_intersection_falsified"
        ),
    }
    validate_artifact(artifact)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def min_violation_state_distribution(
    samples: Iterable[SpinState],
    verifiers: Iterable[Verifier],
    *,
    min_violation: int,
) -> list[dict[str, Any]]:
    """Return observed distribution over accepted states with minimum residual violation."""

    verifier_tuple = tuple(verifiers)
    counts = Counter(
        state
        for state in samples
        if violation_count(state, verifier_tuple) == int(min_violation)
    )
    total = sum(counts.values())
    return [
        {
            "state": list(state),
            "count": count,
            "probability": count / total,
            "violation_count": int(min_violation),
        }
        for state, count in sorted(counts.items())
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required terminal fields for Exp 1565."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if not str(artifact["honest_verdict"]).startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    if float(artifact["hard_brs_acceptance_rate"]) != 0.0:
        raise ValueError("hard_brs_acceptance_rate must be 0.0")
    if not artifact["soft_gibbs_residual_implemented"]:
        raise ValueError("soft_gibbs_residual_implemented must be true")
    if not artifact["soft_brs_decay_confirmed"]:
        raise ValueError("soft_brs_decay_confirmed must be true")
    if not artifact["min_violation_state_found"]:
        raise ValueError("min_violation_state_found must be true")
    if not artifact["z_beta_curve"]:
        raise ValueError("z_beta_curve must not be empty")
    for row in artifact["z_beta_curve"]:
        if {"beta", "empirical_acceptance_rate"} - set(row):
            raise ValueError("z_beta_curve rows require beta and empirical_acceptance_rate")


def _coerce_state(value: Sequence[int] | np.ndarray) -> SpinState:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("prior_sampler must return a 1D state")
    return tuple(int(item) for item in array.tolist())


def _accepts(
    accept_set_S: Callable[[SpinState], bool] | Container[SpinState],
    state: SpinState,
) -> bool:
    if callable(accept_set_S):
        return bool(accept_set_S(state))
    return state in accept_set_S


def _subopt_decay_rows(
    *,
    states: tuple[SpinState, ...],
    verifiers: Iterable[Verifier],
    beta: float,
    n_values: tuple[int, ...],
    n_trials: int,
) -> tuple[list[dict[str, float | int]], bool]:
    empirical_curve = _estimate_no_acceptance_curve(
        states=states,
        verifiers=verifiers,
        beta=beta,
        n_values=n_values,
        n_trials=n_trials,
        seed=156530 + int(beta * 1000),
    )
    z_beta = exact_z_beta(states, verifiers, beta)
    rows = []
    for n_steps in n_values:
        theoretical = (1.0 - z_beta) ** n_steps
        empirical = empirical_curve[n_steps]
        rows.append(
            {
                "beta": float(beta),
                "n_steps": int(n_steps),
                "empirical_subopt": empirical,
                "theoretical_decay": theoretical,
                "abs_error": abs(empirical - theoretical),
            }
        )
    return rows, all(row["abs_error"] <= DECAY_ABS_TOLERANCE for row in rows)


def _estimate_no_acceptance_curve(
    *,
    states: tuple[SpinState, ...],
    verifiers: Iterable[Verifier],
    beta: float,
    n_values: tuple[int, ...],
    n_trials: int,
    seed: int,
) -> dict[int, float]:
    verifier_tuple = tuple(verifiers)
    violations = np.asarray(
        [violation_count(state, verifier_tuple) for state in states],
        dtype=np.float64,
    )
    acceptance_probabilities = np.exp(-float(beta) * violations)
    unresolved = np.ones(int(n_trials), dtype=bool)
    checkpoints = set(int(value) for value in n_values)
    max_steps = max(checkpoints)
    rng = np.random.default_rng(int(seed))
    curve: dict[int, float] = {}

    for step in range(1, max_steps + 1):
        indices = rng.integers(0, len(states), size=int(n_trials))
        accepted = rng.random(int(n_trials)) < acceptance_probabilities[indices]
        unresolved &= ~accepted
        if step in checkpoints:
            curve[step] = float(np.mean(unresolved))
    return curve
