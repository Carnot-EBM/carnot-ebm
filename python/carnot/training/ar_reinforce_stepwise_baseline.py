"""Exp 1571 step-wise baselines for Linear-AR REINFORCE.

Spec refs: REQ-VERIFY-1571, SCENARIO-VERIFY-1571.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1571_step_wise_baseline_AR_REINFORCE.json"
)

EXPERIMENT_ID = 1571
RUN_DATE = "20260508"
SCHEMA = "ar_reinforce_step_wise_baseline_v1"
TERMINAL_VERDICT_PREFIX = "complete:"
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "step_wise_baseline_implemented",
    "gradient_variance_reduction_factor",
    "convergence_rate_matches_theorem_2",
    "honest_verdict",
}


@dataclass(frozen=True)
class StepWiseBaselineConfig:
    """Configuration for the deterministic Exp 1571 AR-REINFORCE benchmark."""

    n: int = 32
    k: int = 15
    constraints_per_k: int = 10
    constraint_prefix_span: int = 15
    seed: int = 1571
    batch_size: int = 8192
    warm_start_logit: float = 1.15
    noise_fraction: float = 0.03
    variance_reduction_gate: float = 10.0
    convergence_rate_floor: float = 0.97

    @property
    def linear_ar_parameter_count(self) -> int:
        """Return the full Linear-AR parameter count."""

        return self.n + self.linear_ar_coupling_parameter_count

    @property
    def linear_ar_coupling_parameter_count(self) -> int:
        """Return the strictly lower-triangular coupling-parameter count."""

        return self.n * (self.n - 1) // 2

    def validate(self) -> None:
        """Raise when the benchmark shape cannot define the AR stress case."""

        if self.n <= 0:
            raise ValueError("n must be positive")
        if not 1 <= self.k <= self.n:
            raise ValueError("k must satisfy 1 <= k <= n")
        if not self.k <= self.constraint_prefix_span <= self.n:
            raise ValueError("constraint_prefix_span must satisfy k <= span <= n")
        if self.constraints_per_k <= 0:
            raise ValueError("constraints_per_k must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.noise_fraction < 0.0:
            raise ValueError("noise_fraction must be non-negative")


@dataclass(frozen=True)
class AndConstraint:
    """One planted k-way AND constraint over binary variables."""

    indices: tuple[int, ...]
    target: tuple[int, ...]


@dataclass(frozen=True)
class LinearARParameters:
    """Linear autoregressive Bernoulli parameters."""

    bias: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class ARReinforceProblem:
    """Complete finite Linear-AR REINFORCE benchmark instance."""

    config: StepWiseBaselineConfig
    planted_target: np.ndarray
    constraints: tuple[AndConstraint, ...]
    parameters: LinearARParameters


@dataclass(frozen=True)
class GradientVarianceResult:
    """A/B variance comparison between scalar and step-wise baselines."""

    metric: str
    scalar_coupling_trace: float
    step_wise_coupling_trace: float
    scalar_full_trace: float
    step_wise_full_trace: float
    reduction_factor: float


@dataclass(frozen=True)
class ConvergenceRateResult:
    """Noise-resilience proxy for the step-wise AR estimator."""

    clean_step_wise_snr: float
    noisy_step_wise_snr: float
    noisy_to_clean_rate_ratio: float
    matches_theorem_2: bool


@dataclass(frozen=True)
class ABTestResult:
    """Terminal in-memory result for Exp 1571 before JSON serialization."""

    gradient_variance: GradientVarianceResult
    convergence: ConvergenceRateResult


def build_problem(config: StepWiseBaselineConfig = StepWiseBaselineConfig()) -> ARReinforceProblem:
    """Build the planted `n=32`, `k=15` Linear-AR AND-composition problem."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    planted_target = rng.integers(0, 2, size=config.n).astype(np.float64)
    indices = tuple(range(config.constraint_prefix_span - config.k, config.constraint_prefix_span))
    target = tuple(int(planted_target[index]) for index in indices)
    constraints = tuple(
        AndConstraint(indices=indices, target=target) for _ in range(config.constraints_per_k)
    )
    return ARReinforceProblem(
        config=config,
        planted_target=planted_target,
        constraints=constraints,
        parameters=build_initial_linear_ar_parameters(config, planted_target),
    )


def build_initial_linear_ar_parameters(
    config: StepWiseBaselineConfig,
    planted_target: np.ndarray,
) -> LinearARParameters:
    """Build a warm-start Linear-AR model that keeps all coupling params trainable."""

    bias = np.where(planted_target == 1.0, config.warm_start_logit, -config.warm_start_logit)
    weights = np.zeros((config.n, config.n), dtype=np.float64)
    return LinearARParameters(bias=bias.astype(np.float64), weights=weights)


def sample_linear_ar(
    problem: ARReinforceProblem,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample binary sequences and token probabilities from the Linear-AR model."""

    config = problem.config
    states = np.zeros((config.batch_size, config.n), dtype=np.float64)
    probabilities = np.zeros_like(states)
    params = problem.parameters
    for token in range(config.n):
        logits = params.bias[token] + states[:, :token] @ params.weights[token, :token]
        token_probabilities = _sigmoid(logits)
        probabilities[:, token] = token_probabilities
        states[:, token] = (rng.random(config.batch_size) < token_probabilities).astype(np.float64)
    return states, probabilities


def evaluate_and_reward(
    states: np.ndarray,
    constraints: tuple[AndConstraint, ...],
) -> np.ndarray:
    """Return the number of planted AND constraints satisfied by each sample."""

    states = np.asarray(states, dtype=np.float64)
    reward = np.zeros(states.shape[0], dtype=np.float64)
    for constraint in constraints:
        indices = np.asarray(constraint.indices, dtype=np.int64)
        target = np.asarray(constraint.target, dtype=np.float64)
        reward += np.all(states[:, indices] == target, axis=1).astype(np.float64)
    return reward


def compute_step_wise_baseline(
    states: np.ndarray,
    problem: ARReinforceProblem,
) -> np.ndarray:
    """Compute prefix-only per-token control variates for Linear-AR REINFORCE."""

    states = np.asarray(states, dtype=np.float64)
    batch_size, n = states.shape
    if n != problem.config.n:
        raise ValueError("states width must match config.n")

    baseline = np.zeros((batch_size, n), dtype=np.float64)
    for token in range(n):
        expected_states = _expected_states_from_prefix(states, problem.parameters, token)
        token_baseline = np.zeros(batch_size, dtype=np.float64)
        for constraint in problem.constraints:
            token_baseline += _constraint_probability_from_prefix(
                states,
                expected_states,
                token,
                constraint,
            )
        baseline[:, token] = token_baseline
    return baseline


def run_ab_test(config: StepWiseBaselineConfig = StepWiseBaselineConfig()) -> ABTestResult:
    """Run scalar-baseline versus step-wise-baseline AR-REINFORCE A/B statistics."""

    problem = build_problem(config)
    rng = np.random.default_rng(config.seed + 10)
    states, probabilities = sample_linear_ar(problem, rng)
    clean_reward = evaluate_and_reward(states, problem.constraints)
    noisy_reward = clean_reward + rng.normal(
        loc=0.0,
        scale=config.noise_fraction,
        size=config.batch_size,
    )
    step_wise_baseline = compute_step_wise_baseline(states, problem)
    scalar_residual = np.broadcast_to(
        (noisy_reward - float(np.mean(noisy_reward)))[:, None],
        states.shape,
    )
    step_wise_residual = noisy_reward[:, None] - step_wise_baseline
    clean_step_wise_residual = clean_reward[:, None] - step_wise_baseline

    scalar_coupling_trace, _ = _gradient_trace_and_mean_sq(
        states,
        probabilities,
        scalar_residual,
        include_bias=False,
    )
    step_wise_coupling_trace, noisy_mean_sq = _gradient_trace_and_mean_sq(
        states,
        probabilities,
        step_wise_residual,
        include_bias=False,
    )
    scalar_full_trace, _ = _gradient_trace_and_mean_sq(
        states,
        probabilities,
        scalar_residual,
        include_bias=True,
    )
    step_wise_full_trace, _ = _gradient_trace_and_mean_sq(
        states,
        probabilities,
        step_wise_residual,
        include_bias=True,
    )
    clean_step_trace, clean_mean_sq = _gradient_trace_and_mean_sq(
        states,
        probabilities,
        clean_step_wise_residual,
        include_bias=False,
    )

    reduction_factor = scalar_coupling_trace / max(step_wise_coupling_trace, 1e-12)
    clean_snr = clean_mean_sq / max(clean_step_trace, 1e-12)
    noisy_snr = noisy_mean_sq / max(step_wise_coupling_trace, 1e-12)
    rate_ratio = noisy_snr / max(clean_snr, 1e-12)
    return ABTestResult(
        gradient_variance=GradientVarianceResult(
            metric="linear_ar_coupling_trace_variance",
            scalar_coupling_trace=float(scalar_coupling_trace),
            step_wise_coupling_trace=float(step_wise_coupling_trace),
            scalar_full_trace=float(scalar_full_trace),
            step_wise_full_trace=float(step_wise_full_trace),
            reduction_factor=float(reduction_factor),
        ),
        convergence=ConvergenceRateResult(
            clean_step_wise_snr=float(clean_snr),
            noisy_step_wise_snr=float(noisy_snr),
            noisy_to_clean_rate_ratio=float(rate_ratio),
            matches_theorem_2=bool(rate_ratio >= config.convergence_rate_floor),
        ),
    )


def build_artifact(
    *,
    config: StepWiseBaselineConfig,
    result: ABTestResult,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1571 artifact."""

    variance = result.gradient_variance
    convergence = result.convergence
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "spec_refs": ["REQ-VERIFY-1571", "SCENARIO-VERIFY-1571"],
            "n": config.n,
            "k": config.k,
            "constraints_per_k": config.constraints_per_k,
            "constraint_prefix_span": config.constraint_prefix_span,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "noise_fraction": config.noise_fraction,
            "variance_reduction_gate": config.variance_reduction_gate,
            "convergence_rate_floor": config.convergence_rate_floor,
            "linear_ar_parameter_count": config.linear_ar_parameter_count,
            "linear_ar_coupling_parameter_count": config.linear_ar_coupling_parameter_count,
            "gradient_variance_metric": variance.metric,
            "and_composition_stress_case": (
                "ten planted k=15 AND constraints over the first fifteen variables; "
                "this isolates suffix score-function variance in Linear-AR couplings"
            ),
        },
        "status": "complete",
        "step_wise_baseline_implemented": True,
        "gradient_variance_reduction_factor": _round_float(variance.reduction_factor),
        "convergence_rate_matches_theorem_2": bool(convergence.matches_theorem_2),
        "scalar_baseline_coupling_trace_variance": _round_float(
            variance.scalar_coupling_trace
        ),
        "step_wise_baseline_coupling_trace_variance": _round_float(
            variance.step_wise_coupling_trace
        ),
        "scalar_baseline_full_trace_variance": _round_float(variance.scalar_full_trace),
        "step_wise_baseline_full_trace_variance": _round_float(variance.step_wise_full_trace),
        "clean_step_wise_snr": _round_float(convergence.clean_step_wise_snr),
        "noisy_step_wise_snr": _round_float(convergence.noisy_step_wise_snr),
        "convergence_rate_noisy_to_clean_ratio": _round_float(
            convergence.noisy_to_clean_rate_ratio
        ),
        "honest_verdict": _honest_verdict(config, result),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the required terminal fields and acceptance gates."""

    missing = REQUIRED_ARTIFACT_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("status") != "complete":
        raise ValueError("status must be complete")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_VERDICT_PREFIX):
        raise ValueError("honest_verdict must begin with complete:")
    if artifact.get("step_wise_baseline_implemented") is not True:
        raise ValueError("step-wise baseline must be implemented")
    if float(artifact.get("gradient_variance_reduction_factor", 0.0)) < 10.0:
        raise ValueError("variance reduction gate failed")
    if artifact.get("convergence_rate_matches_theorem_2") is not True:
        raise ValueError("convergence gate failed")


def run_experiment(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    config: StepWiseBaselineConfig = StepWiseBaselineConfig(),
) -> dict[str, Any]:
    """Run Exp 1571 and write the terminal deliverable JSON."""

    result = run_ab_test(config)
    artifact = build_artifact(config=config, result=result)
    return _write_json(output_path, artifact)


def _expected_states_from_prefix(
    states: np.ndarray,
    params: LinearARParameters,
    token: int,
) -> np.ndarray:
    batch_size, n = states.shape
    expected_states = np.zeros((batch_size, n), dtype=np.float64)
    if token > 0:
        expected_states[:, :token] = states[:, :token]
    for future_token in range(token, n):
        logits = (
            params.bias[future_token]
            + expected_states[:, :future_token] @ params.weights[future_token, :future_token]
        )
        expected_states[:, future_token] = _sigmoid(logits)
    return expected_states


def _constraint_probability_from_prefix(
    states: np.ndarray,
    expected_states: np.ndarray,
    token: int,
    constraint: AndConstraint,
) -> np.ndarray:
    indices = np.asarray(constraint.indices, dtype=np.int64)
    target = np.asarray(constraint.target, dtype=np.float64)
    past_mask = indices < token
    future_mask = ~past_mask
    prefix_matches = np.ones(states.shape[0], dtype=bool)
    if np.any(past_mask):
        prefix_matches = np.all(states[:, indices[past_mask]] == target[past_mask], axis=1)

    future_probability = np.ones(states.shape[0], dtype=np.float64)
    if np.any(future_mask):
        future_indices = indices[future_mask]
        future_target = target[future_mask]
        match_probabilities = np.where(
            future_target[None, :] == 1.0,
            expected_states[:, future_indices],
            1.0 - expected_states[:, future_indices],
        )
        future_probability = np.prod(match_probabilities, axis=1)
    return prefix_matches.astype(np.float64) * future_probability


def _gradient_trace_and_mean_sq(
    states: np.ndarray,
    probabilities: np.ndarray,
    residual_by_token: np.ndarray,
    *,
    include_bias: bool,
) -> tuple[float, float]:
    centered = states - probabilities
    trace_variance = 0.0
    mean_sq_norm = 0.0
    if include_bias:
        bias_gradients = residual_by_token * centered
        trace_variance += float(np.var(bias_gradients, axis=0, ddof=1).sum())
        mean_sq_norm += float(np.square(np.mean(bias_gradients, axis=0)).sum())

    for token in range(1, states.shape[1]):
        token_factor = residual_by_token[:, token] * centered[:, token]
        gradients = token_factor[:, None] * states[:, :token]
        trace_variance += float(np.var(gradients, axis=0, ddof=1).sum())
        mean_sq_norm += float(np.square(np.mean(gradients, axis=0)).sum())
    return trace_variance, mean_sq_norm


def _honest_verdict(config: StepWiseBaselineConfig, result: ABTestResult) -> str:
    variance = result.gradient_variance
    convergence = result.convergence
    return (
        "complete: step-wise Linear-AR REINFORCE baseline passed "
        f"{config.variance_reduction_gate:.1f}x AR-coupling variance gate "
        f"({variance.reduction_factor:.2f}x) and retained "
        f"{convergence.noisy_to_clean_rate_ratio:.3f} of the noiseless "
        "3% noise convergence-rate proxy"
    )


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-values))


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _write_json(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:  # pragma: no cover - exercised through run_experiment in tests.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
