"""Exp 1578 BRAIN REINFORCE training-dynamics audit.

Spec refs: REQ-VERIFY-1578, SCENARIO-VERIFY-1578.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1578_brain_reinforce_training_dynamics_at_k15.json"
)
RESEARCH_NOTE_PATH = (
    PROJECT_ROOT / "docs" / "research-notes" / "brain-reinforce-training-dynamics-k15.md"
)

EXPERIMENT_ID = 1578
RUN_DATE = "20260508"
SCHEMA = "brain_reinforce_training_dynamics_k15_v1"
TERMINAL_VERDICT_PREFIX = "complete:"
FACTOR_STARVATION_REAL = "factorized gradient starvation real"
STARVATION_OVERSTATED = "starvation overstated"
BOTH_INADEQUATE = "both parameterizations inadequate"
ALLOWED_VERDICTS = {
    FACTOR_STARVATION_REAL,
    STARVATION_OVERSTATED,
    BOTH_INADEQUATE,
}
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "factorized_gradient_active_fraction_first_1000",
    "linear_ar_gradient_active_fraction_first_1000",
    "factorized_final_kl",
    "linear_ar_final_kl",
    "factorized_converged",
    "linear_ar_converged",
    "brain_training_dynamics_verdict_ready",
    "paper_v6_brain_recommendation",
    "honest_verdict",
}


@dataclass(frozen=True)
class TrainingDynamicsConfig:
    """Configuration for the finite-state BRAIN k=15 REINFORCE audit."""

    n: int = 16
    k: int = 15
    beta: float = 2.0
    constraints_per_k: int = 10
    constraint_seed: int = 42
    sampling_seed: int = 1578
    batch_size: int = 512
    max_iterations: int = 50_000
    min_iterations: int = 1_000
    checkpoint_interval: int = 1_000
    factorized_learning_rate: float = 0.01
    linear_ar_learning_rate: float = 0.01
    convergence_kl_threshold: float = 0.005
    gradient_active_threshold: float = 1e-12
    active_fraction_window: int = 1_000
    starvation_active_fraction_floor: float = 0.20

    @property
    def factorized_parameter_count(self) -> int:
        """Return the number of factorized Bernoulli logits."""

        return self.n

    @property
    def linear_ar_parameter_count(self) -> int:
        """Return the Linear-AR bias plus lower-triangular coupling count."""

        return self.n + self.n * (self.n - 1) // 2

    def validate(self) -> None:
        """Raise when the audit configuration cannot define the benchmark."""

        if self.n <= 0:
            raise ValueError("n must be positive")
        if not 1 <= self.k <= self.n:
            raise ValueError("k must satisfy 1 <= k <= n")
        if self.constraints_per_k <= 0:
            raise ValueError("constraints_per_k must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.min_iterations < 0:
            raise ValueError("min_iterations must be non-negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")


@dataclass(frozen=True)
class AndConstraint:
    """One k-way AND equality constraint over binary variables."""

    indices: tuple[int, ...]
    target: tuple[int, ...]


@dataclass(frozen=True)
class BrainReinforceProblem:
    """Exact finite-state BRAIN target distribution."""

    config: TrainingDynamicsConfig
    states: np.ndarray
    rewards: np.ndarray
    log_pi: np.ndarray
    constraints: tuple[AndConstraint, ...]

    def initial_factorized_kl(self) -> float:
        """Return exact KL for the uniform factorized q initialization."""

        return exact_factorized_kl(np.zeros(self.config.n, dtype=np.float64), self)

    def initial_linear_ar_kl(self) -> float:
        """Return exact KL for the uniform Linear-AR q initialization."""

        bias = np.zeros(self.config.n, dtype=np.float64)
        weights = np.zeros((self.config.n, self.config.n), dtype=np.float64)
        return exact_linear_ar_kl(bias, weights, self)


@dataclass(frozen=True)
class CheckpointMetric:
    """Metrics captured at one training checkpoint."""

    iteration: int
    kl: float
    gradient_l2: float
    marginal_escape_from_half: float


@dataclass(frozen=True)
class TrainingTrace:
    """Training trace for one q parameterization."""

    parameterization: str
    checkpoints: tuple[CheckpointMetric, ...]
    gradient_active_fraction_first_1000: float
    convergence_iteration: int | None
    wall_time_s: float
    iterations_run: int

    @property
    def final_kl(self) -> float:
        """Return the last tracked exact KL, or infinity for incomplete traces."""

        if not self.checkpoints:
            return float("inf")
        return float(self.checkpoints[-1].kl)

    def converged(self, threshold: float) -> bool:
        """Return whether the trace has reached the configured KL threshold."""

        return self.convergence_iteration is not None and self.final_kl <= threshold


def build_problem(config: TrainingDynamicsConfig = TrainingDynamicsConfig()) -> BrainReinforceProblem:
    """Build the deterministic finite-state BRAIN k-way AND target."""

    config.validate()
    states = np.asarray(list(itertools.product([0, 1], repeat=config.n)), dtype=np.float64)
    constraints = generate_constraints(config)
    rewards = evaluate_reward(states, constraints)
    log_weights = float(config.beta) * rewards
    log_pi = log_weights - _logsumexp(log_weights)
    return BrainReinforceProblem(
        config=config,
        states=states,
        rewards=rewards,
        log_pi=log_pi,
        constraints=constraints,
    )


def generate_constraints(config: TrainingDynamicsConfig) -> tuple[AndConstraint, ...]:
    """Generate deterministic random AND-composition constraints."""

    rng = np.random.RandomState(config.constraint_seed)
    constraints: list[AndConstraint] = []
    for _ in range(config.constraints_per_k):
        indices = tuple(int(value) for value in rng.choice(config.n, config.k, replace=False))
        target = tuple(int(value) for value in rng.randint(0, 2, config.k))
        constraints.append(AndConstraint(indices=indices, target=target))
    return tuple(constraints)


def evaluate_reward(
    states: np.ndarray,
    constraints: tuple[AndConstraint, ...],
) -> np.ndarray:
    """Return how many AND constraints each binary state satisfies."""

    states = np.asarray(states, dtype=np.float64)
    reward = np.zeros(states.shape[0], dtype=np.float64)
    for constraint in constraints:
        indices = np.asarray(constraint.indices, dtype=np.int64)
        target = np.asarray(constraint.target, dtype=np.float64)
        reward += np.all(states[:, indices] == target, axis=1).astype(np.float64)
    return reward


def train_factorized(
    problem: BrainReinforceProblem,
    config: TrainingDynamicsConfig,
) -> TrainingTrace:
    """Train factorized Bernoulli q with scalar-baseline REINFORCE."""

    rng = np.random.default_rng(config.sampling_seed)
    logits = np.zeros(config.n, dtype=np.float64)
    start = time.perf_counter()
    initial = _factorized_checkpoint(0, logits, problem, gradient_l2=0.0)
    checkpoints = [initial]
    convergence_iteration = 0 if initial.kl <= config.convergence_kl_threshold else None
    active_count = 0
    iterations_run = 0

    for iteration in range(1, config.max_iterations + 1):
        gradient, gradient_l2 = _factorized_reinforce_gradient(logits, problem, config, rng)
        logits -= config.factorized_learning_rate * gradient
        iterations_run = iteration
        if iteration <= config.active_fraction_window and gradient_l2 > config.gradient_active_threshold:
            active_count += 1
        if iteration % config.checkpoint_interval == 0:
            checkpoints.append(_factorized_checkpoint(iteration, logits, problem, gradient_l2))
            if (
                convergence_iteration is None
                and checkpoints[-1].kl <= config.convergence_kl_threshold
            ):
                convergence_iteration = iteration
            if iteration >= config.min_iterations and convergence_iteration is not None:
                break

    denominator = max(1, min(config.active_fraction_window, iterations_run))
    return TrainingTrace(
        parameterization="factorized",
        checkpoints=tuple(checkpoints),
        gradient_active_fraction_first_1000=float(active_count / denominator),
        convergence_iteration=convergence_iteration,
        wall_time_s=float(time.perf_counter() - start),
        iterations_run=iterations_run,
    )


def train_linear_ar(
    problem: BrainReinforceProblem,
    config: TrainingDynamicsConfig,
) -> TrainingTrace:
    """Train Linear-AR Bernoulli q with scalar-baseline REINFORCE."""

    rng = np.random.default_rng(config.sampling_seed + 1)
    bias = np.zeros(config.n, dtype=np.float64)
    weights = np.zeros((config.n, config.n), dtype=np.float64)
    start = time.perf_counter()
    initial = _linear_ar_checkpoint(0, bias, weights, problem, gradient_l2=0.0)
    checkpoints = [initial]
    convergence_iteration = 0 if initial.kl <= config.convergence_kl_threshold else None
    active_count = 0
    iterations_run = 0

    for iteration in range(1, config.max_iterations + 1):
        bias_gradient, weight_gradient, gradient_l2 = _linear_ar_reinforce_gradient(
            bias,
            weights,
            problem,
            config,
            rng,
        )
        bias -= config.linear_ar_learning_rate * bias_gradient
        weights -= config.linear_ar_learning_rate * weight_gradient
        iterations_run = iteration
        if iteration <= config.active_fraction_window and gradient_l2 > config.gradient_active_threshold:
            active_count += 1
        if iteration % config.checkpoint_interval == 0:
            checkpoints.append(
                _linear_ar_checkpoint(iteration, bias, weights, problem, gradient_l2)
            )
            if (
                convergence_iteration is None
                and checkpoints[-1].kl <= config.convergence_kl_threshold
            ):
                convergence_iteration = iteration
            if iteration >= config.min_iterations and convergence_iteration is not None:
                break

    denominator = max(1, min(config.active_fraction_window, iterations_run))
    return TrainingTrace(
        parameterization="linear_ar",
        checkpoints=tuple(checkpoints),
        gradient_active_fraction_first_1000=float(active_count / denominator),
        convergence_iteration=convergence_iteration,
        wall_time_s=float(time.perf_counter() - start),
        iterations_run=iterations_run,
    )


def exact_factorized_kl(logits: np.ndarray, problem: BrainReinforceProblem) -> float:
    """Compute exact finite-state `KL(q || pi_beta)` for factorized q."""

    probabilities = _sigmoid(logits)
    log_q = _factorized_log_prob(problem.states, probabilities)
    q = np.exp(log_q)
    return float(np.sum(q * (log_q - problem.log_pi)))


def exact_linear_ar_kl(
    bias: np.ndarray,
    weights: np.ndarray,
    problem: BrainReinforceProblem,
) -> float:
    """Compute exact finite-state `KL(q || pi_beta)` for Linear-AR q."""

    log_q = _linear_ar_log_prob(problem.states, bias, weights)
    q = np.exp(log_q)
    return float(np.sum(q * (log_q - problem.log_pi)))


def classify_training_dynamics(
    *,
    config: TrainingDynamicsConfig,
    factorized: TrainingTrace,
    linear_ar: TrainingTrace,
) -> str:
    """Select the registered training-dynamics verdict."""

    factorized_converged = factorized.converged(config.convergence_kl_threshold)
    linear_ar_converged = linear_ar.converged(config.convergence_kl_threshold)
    if not factorized_converged and not linear_ar_converged:
        return BOTH_INADEQUATE
    if (
        not factorized_converged
        and linear_ar_converged
        and factorized.gradient_active_fraction_first_1000
        < config.starvation_active_fraction_floor
        and linear_ar.gradient_active_fraction_first_1000
        >= config.starvation_active_fraction_floor
    ):
        return FACTOR_STARVATION_REAL
    return STARVATION_OVERSTATED


def build_artifact(
    *,
    config: TrainingDynamicsConfig,
    factorized: TrainingTrace,
    linear_ar: TrainingTrace,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1578 artifact."""

    verdict = classify_training_dynamics(
        config=config,
        factorized=factorized,
        linear_ar=linear_ar,
    )
    paper_recommendation = _paper_v6_recommendation(verdict)
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "spec_refs": ["REQ-VERIFY-1578", "SCENARIO-VERIFY-1578"],
            "n": config.n,
            "k": config.k,
            "beta": config.beta,
            "constraints_per_k": config.constraints_per_k,
            "constraint_seed": config.constraint_seed,
            "sampling_seed": config.sampling_seed,
            "batch_size": config.batch_size,
            "max_iterations": config.max_iterations,
            "min_iterations": config.min_iterations,
            "checkpoint_interval": config.checkpoint_interval,
            "convergence_kl_threshold": config.convergence_kl_threshold,
            "factorized_parameter_count": config.factorized_parameter_count,
            "linear_ar_parameter_count": config.linear_ar_parameter_count,
        },
        "status": "complete",
        "factorized_gradient_active_fraction_first_1000": _round_float(
            factorized.gradient_active_fraction_first_1000
        ),
        "linear_ar_gradient_active_fraction_first_1000": _round_float(
            linear_ar.gradient_active_fraction_first_1000
        ),
        "factorized_final_kl": _round_float(factorized.final_kl),
        "linear_ar_final_kl": _round_float(linear_ar.final_kl),
        "factorized_converged": factorized.converged(config.convergence_kl_threshold),
        "linear_ar_converged": linear_ar.converged(config.convergence_kl_threshold),
        "factorized_convergence_iteration": factorized.convergence_iteration,
        "linear_ar_convergence_iteration": linear_ar.convergence_iteration,
        "factorized_wall_time_s": _round_float(factorized.wall_time_s),
        "linear_ar_wall_time_s": _round_float(linear_ar.wall_time_s),
        "factorized_iterations_run": factorized.iterations_run,
        "linear_ar_iterations_run": linear_ar.iterations_run,
        "factorized_trace": _trace_to_dicts(factorized),
        "linear_ar_trace": _trace_to_dicts(linear_ar),
        "brain_training_dynamics_verdict": verdict,
        "brain_training_dynamics_verdict_ready": bool(
            factorized.checkpoints and linear_ar.checkpoints and verdict in ALLOWED_VERDICTS
        ),
        "paper_v6_brain_recommendation": paper_recommendation,
        "honest_verdict": _honest_verdict(verdict, factorized, linear_ar),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the terminal artifact schema and registered verdict."""

    missing = REQUIRED_ARTIFACT_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("status") != "complete":
        raise ValueError("status must be complete")
    honest_verdict = str(artifact.get("honest_verdict", ""))
    if not honest_verdict.startswith(TERMINAL_VERDICT_PREFIX):
        raise ValueError("honest_verdict must begin with complete:")
    if artifact.get("brain_training_dynamics_verdict_ready") is not True:
        raise ValueError("brain_training_dynamics_verdict_ready must be true")
    if not any(verdict in honest_verdict for verdict in ALLOWED_VERDICTS):
        raise ValueError("honest_verdict must include an allowed verdict")


def run_experiment(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    research_note_path: str | Path = RESEARCH_NOTE_PATH,
    config: TrainingDynamicsConfig = TrainingDynamicsConfig(),
) -> dict[str, Any]:
    """Run Exp 1578 and write the JSON artifact plus research note."""

    problem = build_problem(config)
    factorized = train_factorized(problem, config)
    linear_ar = train_linear_ar(problem, config)
    artifact = build_artifact(config=config, factorized=factorized, linear_ar=linear_ar)
    _write_json(output_path, artifact)
    _write_research_note(research_note_path, artifact)
    return artifact


def _factorized_reinforce_gradient(
    logits: np.ndarray,
    problem: BrainReinforceProblem,
    config: TrainingDynamicsConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    probabilities = _sigmoid(logits)
    samples = (rng.random((config.batch_size, config.n)) < probabilities).astype(np.float64)
    rewards = evaluate_reward(samples, problem.constraints)
    log_q = _factorized_log_prob(samples, probabilities)
    advantage = _center(log_q - float(config.beta) * rewards)
    gradient = np.mean(advantage[:, None] * (samples - probabilities), axis=0)
    return gradient, float(np.linalg.norm(gradient))


def _linear_ar_reinforce_gradient(
    bias: np.ndarray,
    weights: np.ndarray,
    problem: BrainReinforceProblem,
    config: TrainingDynamicsConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    samples, probabilities = _sample_linear_ar(bias, weights, config, rng)
    rewards = evaluate_reward(samples, problem.constraints)
    log_q = _bernoulli_log_prob(samples, probabilities)
    advantage = _center(log_q - float(config.beta) * rewards)
    residual = samples - probabilities
    bias_gradient = np.mean(advantage[:, None] * residual, axis=0)
    weight_gradient = ((advantage[:, None] * residual).T @ samples) / float(config.batch_size)
    weight_gradient *= np.tril(np.ones_like(weight_gradient), k=-1)
    gradient_l2 = float(
        np.sqrt(np.sum(bias_gradient * bias_gradient) + np.sum(weight_gradient * weight_gradient))
    )
    return bias_gradient, weight_gradient, gradient_l2


def _sample_linear_ar(
    bias: np.ndarray,
    weights: np.ndarray,
    config: TrainingDynamicsConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.zeros((config.batch_size, config.n), dtype=np.float64)
    probabilities = np.zeros_like(samples)
    for token in range(config.n):
        logits = bias[token] + samples[:, :token] @ weights[token, :token]
        token_probabilities = _sigmoid(logits)
        probabilities[:, token] = token_probabilities
        samples[:, token] = (rng.random(config.batch_size) < token_probabilities).astype(
            np.float64
        )
    return samples, probabilities


def _factorized_checkpoint(
    iteration: int,
    logits: np.ndarray,
    problem: BrainReinforceProblem,
    gradient_l2: float,
) -> CheckpointMetric:
    probabilities = _sigmoid(logits)
    return CheckpointMetric(
        iteration=iteration,
        kl=exact_factorized_kl(logits, problem),
        gradient_l2=float(gradient_l2),
        marginal_escape_from_half=float(np.max(np.abs(probabilities - 0.5))),
    )


def _linear_ar_checkpoint(
    iteration: int,
    bias: np.ndarray,
    weights: np.ndarray,
    problem: BrainReinforceProblem,
    gradient_l2: float,
) -> CheckpointMetric:
    log_q = _linear_ar_log_prob(problem.states, bias, weights)
    q = np.exp(log_q)
    marginals = q @ problem.states
    return CheckpointMetric(
        iteration=iteration,
        kl=float(np.sum(q * (log_q - problem.log_pi))),
        gradient_l2=float(gradient_l2),
        marginal_escape_from_half=float(np.max(np.abs(marginals - 0.5))),
    )


def _factorized_log_prob(states: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return states @ np.log(probabilities) + (1.0 - states) @ np.log1p(-probabilities)


def _linear_ar_log_prob(states: np.ndarray, bias: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = states @ weights.T + bias
    return _bernoulli_log_prob(states, _sigmoid(logits))


def _bernoulli_log_prob(states: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.sum(
        states * np.log(probabilities) + (1.0 - states) * np.log1p(-probabilities),
        axis=1,
    )


def _center(values: np.ndarray) -> np.ndarray:
    return values - float(np.mean(values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _trace_to_dicts(trace: TrainingTrace) -> list[dict[str, float | int]]:
    return [
        {
            "iteration": point.iteration,
            "kl": _round_float(point.kl),
            "gradient_l2": _round_float(point.gradient_l2),
            "marginal_escape_from_half": _round_float(point.marginal_escape_from_half),
        }
        for point in trace.checkpoints
    ]


def _paper_v6_recommendation(verdict: str) -> str:
    if verdict == FACTOR_STARVATION_REAL:
        return "paper_v6: cite factorized gradient starvation but keep Linear-AR as the control"
    if verdict == BOTH_INADEQUATE:
        return "paper_v6: drop both factorized Bernoulli and Linear-AR BRAIN parameterizations"
    return "paper_v6: treat BRAIN gradient-starvation as overstated at k=15"


def _honest_verdict(
    verdict: str,
    factorized: TrainingTrace,
    linear_ar: TrainingTrace,
) -> str:
    return (
        f"complete: {verdict}; "
        f"factorized_final_KL={factorized.final_kl:.6f}, "
        f"linear_AR_final_KL={linear_ar.final_kl:.6f}"
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _write_research_note(path: str | Path, artifact: dict[str, Any]) -> None:
    note_path = Path(path)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "\n".join(
            [
                "# BRAIN REINFORCE Training Dynamics at k=15",
                "",
                f"Experiment: {EXPERIMENT_ID}",
                f"Status: {artifact['status']}",
                f"Verdict: {artifact['honest_verdict']}",
                f"Paper v6 recommendation: {artifact['paper_v6_brain_recommendation']}",
                "",
                "## Summary",
                "",
                (
                    "The audit trains factorized Bernoulli and Linear-AR q_theta "
                    "with scalar-baseline REINFORCE against the exact finite-state "
                    "BRAIN target at n=16, k=15, beta=2.0."
                ),
                "",
                "## Metrics",
                "",
                (
                    f"- Factorized active fraction first 1000: "
                    f"{artifact['factorized_gradient_active_fraction_first_1000']}"
                ),
                (
                    f"- Linear-AR active fraction first 1000: "
                    f"{artifact['linear_ar_gradient_active_fraction_first_1000']}"
                ),
                f"- Factorized final KL: {artifact['factorized_final_kl']}",
                f"- Linear-AR final KL: {artifact['linear_ar_final_kl']}",
                "",
                "## Paper-v6 Recommendation",
                "",
                str(artifact["paper_v6_brain_recommendation"]),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _round_float(value: float) -> float:
    return round(float(value), 6)


def main() -> None:  # pragma: no cover - CLI glue.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
