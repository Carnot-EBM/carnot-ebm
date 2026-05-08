"""Exp 1562 DT-BRAIN-CORRELATIONS k-sweep verification.

This script extends the 2026-05-08 Deep Think BRAIN-correlation check from the
original `k=4` probe to `k in {4, 8, 12, 15}` at `n=16`. Each run enumerates
all `2^16 = 65,536` binary states exactly, optimizes reverse KL
`KL(q || pi_beta)`, and writes the terminal Exp 1562 artifact.

Spec refs: REQ-VERIFY-1562, SCENARIO-VERIFY-1562.
Reference: docs/research-notes/iclr26-deep-think-responses.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1562_brain_linear_ar_k_sweep_extended.json"
)

EXPERIMENT_ID = 1562
RUN_DATE = "20260508"
SCHEMA = "brain_linear_ar_k_sweep_v1"
TERMINAL_VERDICT_PREFIX = "complete:"
K15_AR_KL_GATE = 0.1
VALIDATION_RATIO_GATE = 10.0
FALSIFICATION_RATIO_GATE = 5.0
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "brain_linear_ar_rescue_validated",
    "kl_by_k_by_parameterization",
    "factorized_vs_ar_ratio_at_k15",
    "made_required_at_k15",
    "phase_3_recommendation",
    "honest_verdict",
}

KlRow = dict[str, float | None]
KlByK = dict[int, KlRow]


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for the exact BRAIN-correlation k-sweep."""

    n: int = 16
    k_values: tuple[int, ...] = (4, 8, 12, 15)
    constraints_per_k: int = 10
    beta: float = 2.0
    seed: int = 42
    maxiter: int = 500
    made_hidden_units: int = 32
    made_steps: int = 2_000
    made_learning_rate: float = 0.03

    @property
    def factorized_parameter_count(self) -> int:
        """Return the factorized-Bernoulli parameter count."""

        return self.n

    @property
    def linear_ar_parameter_count(self) -> int:
        """Return the strictly lower-triangular Linear-AR parameter count."""

        return self.n + self.n * (self.n - 1) // 2


@dataclass(frozen=True)
class BrainCorrelationProblem:
    """Exact finite-state target distribution for one `k` value."""

    config: SweepConfig
    k: int
    states: np.ndarray
    log_pi: np.ndarray
    energy: np.ndarray
    constraints: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class OptimizationResult:
    """Minimal optimizer result persisted in the Exp 1562 artifact."""

    parameterization: str
    kl: float
    success: bool
    iterations: int
    message: str


def enumerate_states(n: int) -> np.ndarray:
    """Enumerate every binary state as a float matrix with shape `(2**n, n)`."""

    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)


def generate_constraints(
    *,
    n: int,
    k: int,
    count: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Generate deterministic AND-composition verifier constraints.

    `np.random.RandomState` intentionally preserves the original one-off
    script's `np.random.seed(42)` semantics for the `k=4` baseline.
    """

    rng = np.random.RandomState(seed)
    return tuple(
        (
            rng.choice(n, k, replace=False).astype(np.int64),
            rng.randint(0, 2, k).astype(np.int64),
        )
        for _ in range(count)
    )


def build_problem(config: SweepConfig, *, k: int) -> BrainCorrelationProblem:
    """Build the exact target Boltzmann distribution for one `k`."""

    states = enumerate_states(config.n)
    constraints = generate_constraints(
        n=config.n,
        k=k,
        count=config.constraints_per_k,
        seed=config.seed,
    )
    energy = np.zeros(states.shape[0], dtype=np.float64)
    for indices, target in constraints:
        energy -= np.all(states[:, indices] == target, axis=1)

    log_weights = -float(config.beta) * energy
    log_pi = log_weights - _logsumexp(log_weights)
    return BrainCorrelationProblem(
        config=config,
        k=k,
        states=states,
        log_pi=log_pi,
        energy=energy,
        constraints=constraints,
    )


def optimize_factorized(
    problem: BrainCorrelationProblem,
    *,
    maxiter: int,
) -> OptimizationResult:
    """Optimize the `n`-parameter factorized Bernoulli reverse KL."""

    initial = np.zeros(problem.config.n, dtype=np.float64)
    result = minimize(
        lambda params: _factorized_objective_and_grad(params, problem),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    return OptimizationResult(
        parameterization="factorized",
        kl=float(result.fun),
        success=bool(result.success),
        iterations=int(result.nit),
        message=str(result.message),
    )


def optimize_linear_ar(
    problem: BrainCorrelationProblem,
    *,
    maxiter: int,
) -> OptimizationResult:
    """Optimize the Linear-AR reverse KL with all pairwise past-bit weights."""

    initial = np.zeros(problem.config.linear_ar_parameter_count, dtype=np.float64)
    result = minimize(
        lambda params: _linear_ar_objective_and_grad(params, problem),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    return OptimizationResult(
        parameterization="linear_ar",
        kl=float(result.fun),
        success=bool(result.success),
        iterations=int(result.nit),
        message=str(result.message),
    )


def optimize_made(
    problem: BrainCorrelationProblem,
    *,
    hidden_units: int,
    steps: int,
    learning_rate: float,
) -> OptimizationResult:  # pragma: no cover - Exp 1562 seed does not need MADE.
    """Optimize a one-hidden-layer MADE fallback when Linear-AR misses the gate."""

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import optax

    states = jnp.asarray(problem.states, dtype=jnp.float32)
    log_pi = jnp.asarray(problem.log_pi, dtype=jnp.float32)
    key = jax.random.PRNGKey(problem.config.seed + 10_000 + problem.k)
    input_key, hidden_key, output_key = jax.random.split(key, 3)
    n = problem.config.n
    degrees = jnp.asarray([(idx % max(1, n - 1)) for idx in range(hidden_units)])
    input_indices = jnp.arange(n)
    output_indices = jnp.arange(n)
    input_mask = (input_indices[:, None] <= degrees[None, :]).astype(jnp.float32)
    output_mask = (degrees[:, None] < output_indices[None, :]).astype(jnp.float32)
    params = {
        "w1": 0.01 * jax.random.normal(input_key, (n, hidden_units)),
        "b1": jnp.zeros((hidden_units,), dtype=jnp.float32),
        "w2": 0.01 * jax.random.normal(hidden_key, (hidden_units, n)),
        "b2": jnp.zeros((n,), dtype=jnp.float32),
        "direct": 0.01 * jax.random.normal(output_key, (n, n)),
    }
    direct_mask = jnp.tril(jnp.ones((n, n), dtype=jnp.float32), k=-1)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(current: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        hidden = jnp.tanh(states @ (current["w1"] * input_mask) + current["b1"])
        logits = (
            hidden @ (current["w2"] * output_mask)
            + states @ (current["direct"] * direct_mask).T
            + current["b2"]
        )
        log_q = jnp.sum(
            states * jax.nn.log_sigmoid(logits)
            + (1.0 - states) * jax.nn.log_sigmoid(-logits),
            axis=1,
        )
        q = jnp.exp(log_q)
        return jnp.sum(q * (log_q - log_pi))

    grad_fn = jax.value_and_grad(loss_fn)
    best = float("inf")
    for _ in range(steps):
        value, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        best = min(best, float(value))

    return OptimizationResult(
        parameterization="made_optional",
        kl=best,
        success=bool(np.isfinite(best)),
        iterations=int(steps),
        message="adam_exact_enumeration",
    )


def compute_kl_for_k(
    config: SweepConfig,
    k: int,
    *,
    include_made: bool,
) -> KlRow:
    """Compute all requested KL values for one `k`."""

    problem = build_problem(config, k=k)
    factorized = optimize_factorized(problem, maxiter=config.maxiter)
    linear_ar = optimize_linear_ar(problem, maxiter=config.maxiter)
    made_kl: float | None = None
    if include_made:
        made = optimize_made(
            problem,
            hidden_units=config.made_hidden_units,
            steps=config.made_steps,
            learning_rate=config.made_learning_rate,
        )
        made_kl = made.kl
    return {
        "factorized": _round_float(factorized.kl),
        "linear_ar": _round_float(linear_ar.kl),
        "made_optional": None if made_kl is None else _round_float(made_kl),
    }


def run_k_sweep(
    config: SweepConfig,
    *,
    optimizer: Callable[..., KlRow] | None = None,
) -> KlByK:
    """Run the full k-sweep, invoking MADE only when the `k=15` AR gate fails."""

    selected_optimizer = optimizer or compute_kl_for_k
    kl_by_k = {
        k: _normalise_kl_row(selected_optimizer(config, k, include_made=False))
        for k in config.k_values
    }
    if float(kl_by_k[15]["linear_ar"] or float("inf")) > K15_AR_KL_GATE:
        kl_by_k = {
            k: _normalise_kl_row(selected_optimizer(config, k, include_made=True))
            for k in config.k_values
        }
    return kl_by_k


def build_artifact(
    *,
    config: SweepConfig,
    kl_by_k: Mapping[int | str, Mapping[str, float | None]],
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1562 artifact."""

    normalised = _normalise_kl_by_k(kl_by_k, expected_k_values=config.k_values)
    k15 = normalised[15]
    factorized_k15 = _required_float(k15, "factorized")
    linear_ar_k15 = _required_float(k15, "linear_ar")
    made_k15 = k15.get("made_optional")
    made_required = linear_ar_k15 > K15_AR_KL_GATE
    ratio = factorized_k15 / max(linear_ar_k15, 1e-12)
    best_k15 = min(
        value
        for value in (factorized_k15, linear_ar_k15, made_k15)
        if value is not None
    )
    recommendation = _phase_3_recommendation(
        ratio=ratio,
        linear_ar_k15=linear_ar_k15,
        made_k15=made_k15,
    )
    linear_ar_validated = (
        recommendation == "linear_ar_sufficient"
        and ratio >= VALIDATION_RATIO_GATE
        and best_k15 <= K15_AR_KL_GATE
    )
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "spec_refs": ["REQ-VERIFY-1562", "SCENARIO-VERIFY-1562"],
            "dt_predicate": "DT-BRAIN-CORRELATIONS",
            "n": config.n,
            "constraints_per_k": config.constraints_per_k,
            "beta": config.beta,
            "seed": config.seed,
            "factorized_parameter_count": config.factorized_parameter_count,
            "linear_ar_parameter_count": config.linear_ar_parameter_count,
            "made_hidden_units": config.made_hidden_units,
        },
        "status": "complete",
        "brain_linear_ar_rescue_validated": bool(linear_ar_validated),
        "kl_by_k_by_parameterization": {
            str(k): _normalise_kl_row(normalised[k]) for k in config.k_values
        },
        "factorized_vs_ar_ratio_at_k15": _round_float(ratio),
        "best_parameterization_kl_at_k15": _round_float(best_k15),
        "made_required_at_k15": bool(made_required),
        "phase_3_recommendation": recommendation,
        "honest_verdict": _honest_verdict(recommendation, ratio, best_k15),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required terminal fields for Exp 1562."""

    missing = REQUIRED_ARTIFACT_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("status") != "complete":
        raise ValueError("status must be complete")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_VERDICT_PREFIX):
        raise ValueError("honest_verdict must begin with complete:")
    recommendation = artifact.get("phase_3_recommendation")
    if recommendation not in {"linear_ar_sufficient", "made_required", "brain_dropped"}:
        raise ValueError(f"invalid phase_3_recommendation: {recommendation!r}")


def run_experiment(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    config: SweepConfig = SweepConfig(),
    sweep_runner: Callable[[SweepConfig], KlByK] = run_k_sweep,
) -> dict[str, Any]:
    """Run the sweep and write the terminal deliverable JSON."""

    kl_by_k = sweep_runner(config)
    artifact = build_artifact(config=config, kl_by_k=kl_by_k)
    return _write_json(output_path, artifact)


def _factorized_objective_and_grad(
    params: np.ndarray,
    problem: BrainCorrelationProblem,
) -> tuple[float, np.ndarray]:
    probabilities = np.clip(expit(params), 1e-12, 1.0 - 1e-12)
    log_q = (
        problem.states @ np.log(probabilities)
        + (1.0 - problem.states) @ np.log1p(-probabilities)
    )
    q = np.exp(log_q)
    centered_cost = log_q - problem.log_pi + 1.0
    objective = float(np.sum(q * (log_q - problem.log_pi)))
    grad = (q * centered_cost) @ (problem.states - probabilities)
    return objective, grad.astype(np.float64)


def _linear_ar_objective_and_grad(
    params: np.ndarray,
    problem: BrainCorrelationProblem,
) -> tuple[float, np.ndarray]:
    n = problem.config.n
    bias = params[:n]
    weights = _lower_triangular_weights(params[n:], n)
    logits = problem.states @ weights.T + bias
    probabilities = np.clip(expit(logits), 1e-12, 1.0 - 1e-12)
    log_q = np.sum(
        problem.states * np.log(probabilities)
        + (1.0 - problem.states) * np.log1p(-probabilities),
        axis=1,
    )
    q = np.exp(log_q)
    centered_q = q * (log_q - problem.log_pi + 1.0)
    residual = problem.states - probabilities
    bias_grad = centered_q @ residual
    weight_grad_matrix = (centered_q[:, None] * residual).T @ problem.states
    flat_weight_grad = np.array(
        [weight_grad_matrix[i, j] for i, j in _linear_ar_pairs(n)],
        dtype=np.float64,
    )
    objective = float(np.sum(q * (log_q - problem.log_pi)))
    return objective, np.concatenate([bias_grad, flat_weight_grad])


def _lower_triangular_weights(flat: np.ndarray, n: int) -> np.ndarray:
    weights = np.zeros((n, n), dtype=np.float64)
    for flat_index, (row, column) in enumerate(_linear_ar_pairs(n)):
        weights[row, column] = flat[flat_index]
    return weights


def _linear_ar_pairs(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((row, column) for row in range(1, n) for column in range(row))


def _phase_3_recommendation(
    *,
    ratio: float,
    linear_ar_k15: float,
    made_k15: float | None,
) -> str:
    if ratio < FALSIFICATION_RATIO_GATE:
        return "brain_dropped"
    if ratio >= VALIDATION_RATIO_GATE and linear_ar_k15 <= K15_AR_KL_GATE:
        return "linear_ar_sufficient"
    if linear_ar_k15 > K15_AR_KL_GATE and made_k15 is not None and made_k15 <= K15_AR_KL_GATE:
        return "made_required"
    return "brain_dropped"


def _honest_verdict(recommendation: str, ratio: float, best_k15: float) -> str:
    if recommendation == "linear_ar_sufficient":
        return (
            "complete: Linear-AR rescue validated at k=15 "
            f"(factorized/AR ratio={ratio:.2f}x, best_KL={best_k15:.6f})"
        )
    if recommendation == "made_required":
        return (
            "complete: Linear-AR alone missed the KL gate; MADE satisfied k=15 "
            f"(factorized/AR ratio={ratio:.2f}x, best_KL={best_k15:.6f})"
        )
    return (
        "complete: falsified BRAIN+Linear-AR rescue widening; "
        f"factorized/AR ratio={ratio:.2f}x and best_KL={best_k15:.6f}"
    )


def _normalise_kl_by_k(
    kl_by_k: Mapping[int | str, Mapping[str, float | None]],
    *,
    expected_k_values: tuple[int, ...],
) -> KlByK:
    normalised = {int(k): _normalise_kl_row(row) for k, row in kl_by_k.items()}
    missing = set(expected_k_values) - normalised.keys()
    if missing:
        raise ValueError(f"missing k values: {sorted(missing)}")
    return normalised


def _normalise_kl_row(row: Mapping[str, float | None]) -> KlRow:
    return {
        "factorized": _round_float(_required_float(row, "factorized")),
        "linear_ar": _round_float(_required_float(row, "linear_ar")),
        "made_optional": (
            None
            if row.get("made_optional") is None
            else _round_float(float(row["made_optional"]))
        ),
    }


def _required_float(row: Mapping[str, float | None], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"missing KL value for {key}")
    return float(value)


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:  # pragma: no cover - CLI glue.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DELIVERABLE_PATH),
        help="Path for the terminal Exp 1562 JSON artifact.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=SweepConfig.maxiter,
        help="Maximum L-BFGS iterations for factorized and Linear-AR optimizers.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - exercised through run_experiment in tests.
    args = _parse_args()
    config = SweepConfig(maxiter=args.maxiter)
    artifact = run_experiment(output_path=args.output, config=config)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
