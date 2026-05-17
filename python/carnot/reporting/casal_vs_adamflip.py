"""Exp 2246 CASAL vs AdamFLIP constraint-violation benchmark.

Spec: REQ-SAMPLE-2246, SCENARIO-SAMPLE-2246.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carnot.phase3.continuous_ebm import ContinuousEBM

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_2246_casal_vs_adamflip.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE

EXPERIMENT = "2246_casal_vs_adamflip"
SCHEMA = "casal_vs_adamflip_v1"
RANDOM_SEED = 2246
N_SAMPLES = 100
ADAMFLIP_TRAINING_STEPS = 120
ADAMFLIP_LEARNING_RATE = 0.035
MCMC_STEPS = 48
MCMC_LR = 0.035
MCMC_NOISE_SCALE = 0.20
SOFT_PENALTY_WEIGHT = 1.40
CASAL_STEPS = 16
CASAL_STEP_SIZE = 0.025
CASAL_NOISE_SCALE = 0.20

CASAL_MODULE = "carnot.samplers.casal"
ADAMFLIP_MODULE = "carnot.training.adamflip"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "casal_validated",
    "casal_violation_mean",
    "adamflip_violation_mean",
    "n_samples",
    "random_seed",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": ("Terminal-prefix required. Use complete: if CASAL violation <= AdamFLIP/2."),
    "casal_validated": "Boolean gate for exp2251 capstone.",
    "casal_violation_mean": (
        "Primary comparison metric; expected much lower under hard-constraint sampling."
    ),
    "adamflip_violation_mean": ("Baseline for comparison; needed so the ratio is auditable."),
    "n_samples": "Must be >= 100 for statistical significance on constraint violation claims.",
    "random_seed": "Reproducibility: both regimes use the same seed for fair comparison.",
}


@dataclass(frozen=True)
class ConstraintBenchmark:
    """Deterministic 3D continuous EBM benchmark for Exp 2246."""

    model: ContinuousEBM
    constraint_matrix: np.ndarray
    constraint_target: np.ndarray


@dataclass(frozen=True)
class AdamFLIPTrainingResult:
    """AdamFLIP-trained EBM parameters and concise convergence diagnostics."""

    model: ContinuousEBM
    initial_violation_mean: float
    final_violation_mean: float
    final_unconstrained_minimum: np.ndarray


def check_preconditions(
    *,
    casal_module: str = CASAL_MODULE,
    adamflip_module: str = ADAMFLIP_MODULE,
) -> list[JsonDict]:
    """REQ-SAMPLE-2246-1: import CASAL and AdamFLIP before benchmark execution."""

    checks: list[JsonDict] = []
    for label, module_name in (
        ("casal_import", casal_module),
        ("adamflip_import", adamflip_module),
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - blocker artifact records import cause.
            checks.append(
                {
                    "name": label,
                    "module": module_name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            checks.append({"name": label, "module": module_name, "status": "passed"})
    return checks


def build_benchmark() -> ConstraintBenchmark:
    """REQ-SAMPLE-2246-2: build a 3D continuous EBM with two equality constraints."""

    stiffness = np.asarray(
        [
            [1.80, 0.20, -0.10],
            [0.20, 1.50, 0.15],
            [-0.10, 0.15, 1.30],
        ],
        dtype=np.float64,
    )
    base_bias = np.asarray([0.90, -0.40, 0.35], dtype=np.float64)
    constraint_matrix = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    constraint_target = np.asarray([0.10, -0.25], dtype=np.float64)
    model = ContinuousEBM(variables=3, coupling=-stiffness, bias=base_bias)
    return ConstraintBenchmark(
        model=model,
        constraint_matrix=constraint_matrix,
        constraint_target=constraint_target,
    )


def train_adamflip_parameters(
    benchmark: ConstraintBenchmark,
    *,
    n_steps: int = ADAMFLIP_TRAINING_STEPS,
    learning_rate: float = ADAMFLIP_LEARNING_RATE,
) -> AdamFLIPTrainingResult:
    """REQ-SAMPLE-2246-3: train EBM bias parameters with AdamFLIP residual feedback."""

    _, jnp = _load_jax_x64()
    from carnot.training.adamflip import AdamFLIP

    stiffness = -np.asarray(benchmark.model.coupling, dtype=np.float64)
    bias = np.asarray(benchmark.model.bias, dtype=np.float64).copy()
    optimizer = AdamFLIP(learning_rate=learning_rate)

    initial_minimum = np.linalg.solve(stiffness, bias)
    initial_violation = _constraint_abs_residuals(benchmark, initial_minimum).mean()

    final_minimum = initial_minimum
    final_violation = float(initial_violation)
    for _ in range(n_steps):
        final_minimum = np.linalg.solve(stiffness, bias)
        residual = _constraint_residuals(benchmark, final_minimum)
        feedback = np.asarray(
            optimizer.update(jnp.asarray(residual, dtype=jnp.float64)),
            dtype=np.float64,
        )
        bias = bias - benchmark.constraint_matrix.T @ feedback
        final_violation = float(np.mean(np.abs(residual)))

    final_minimum = np.linalg.solve(stiffness, bias)
    final_violation = float(_constraint_abs_residuals(benchmark, final_minimum).mean())
    trained_model = ContinuousEBM(
        variables=benchmark.model.variables,
        coupling=np.asarray(benchmark.model.coupling, dtype=np.float64),
        bias=bias,
    )
    return AdamFLIPTrainingResult(
        model=trained_model,
        initial_violation_mean=float(initial_violation),
        final_violation_mean=final_violation,
        final_unconstrained_minimum=final_minimum,
    )


def run_benchmark(
    *,
    n_samples: int = N_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run the 100-sample CASAL-vs-AdamFLIP comparison and return an artifact body."""

    if n_samples < N_SAMPLES:
        raise ValueError(f"Exp 2246 requires at least {N_SAMPLES} samples")

    benchmark = build_benchmark()
    training = train_adamflip_parameters(benchmark)
    initial_states = _draw_initial_states(n_samples, random_seed)
    adamflip_states = sample_adamflip_soft_penalty_mcmc(
        benchmark,
        training.model,
        initial_states,
        random_seed=random_seed,
    )
    casal_states = sample_casal_primal_dual(
        benchmark,
        training.model,
        initial_states,
        random_seed=random_seed,
    )

    adamflip_summary = summarize_regime(benchmark, training.model, adamflip_states)
    casal_summary = summarize_regime(benchmark, training.model, casal_states)

    casal_validated = casal_summary["violation_mean"] <= adamflip_summary["violation_mean"] / 2.0
    if casal_validated:
        honest_verdict = "complete: CASAL violation <= AdamFLIP/2 on 100 shared-seed samples"
    else:
        honest_verdict = "incomplete: CASAL violation > AdamFLIP/2 on 100 shared-seed samples"

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": _run_date(),
        "status": "success",
        "honest_verdict": honest_verdict,
        "casal_validated": casal_validated,
        "casal_violation_mean": casal_summary["violation_mean"],
        "adamflip_violation_mean": adamflip_summary["violation_mean"],
        "casal_max_constraint_violation": casal_summary["max_constraint_violation"],
        "adamflip_max_constraint_violation": adamflip_summary["max_constraint_violation"],
        "casal_energy_mean": casal_summary["energy_mean"],
        "adamflip_energy_mean": adamflip_summary["energy_mean"],
        "max_constraint_violation": {
            "casal": casal_summary["max_constraint_violation"],
            "adamflip": adamflip_summary["max_constraint_violation"],
        },
        "energy_mean": {
            "casal": casal_summary["energy_mean"],
            "adamflip": adamflip_summary["energy_mean"],
        },
        "n_samples": n_samples,
        "random_seed": random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "benchmark": {
            "energy": "ContinuousEBM quadratic E(x) = -0.5*x^T*J*x - h^T*x",
            "variables": benchmark.model.variables,
            "constraint_count": int(benchmark.constraint_matrix.shape[0]),
            "constraint_matrix": benchmark.constraint_matrix.tolist(),
            "constraint_target": benchmark.constraint_target.tolist(),
            "base_bias": benchmark.model.bias.tolist(),
            "adamflip_trained_bias": training.model.bias.tolist(),
            "coupling": benchmark.model.coupling.tolist(),
        },
        "adamflip_training": {
            "iterations": ADAMFLIP_TRAINING_STEPS,
            "learning_rate": ADAMFLIP_LEARNING_RATE,
            "initial_violation_mean": training.initial_violation_mean,
            "final_violation_mean": training.final_violation_mean,
            "final_unconstrained_minimum": training.final_unconstrained_minimum.tolist(),
            "post_training_sampler": "soft_penalty_mcmc",
            "soft_penalty_weight": SOFT_PENALTY_WEIGHT,
        },
        "regimes": {
            "A_adamflip_soft_penalty_mcmc": {
                "n_samples": n_samples,
                "mcmc_steps": MCMC_STEPS,
                "step_size": MCMC_LR,
                "noise_scale": MCMC_NOISE_SCALE,
                **adamflip_summary,
            },
            "B_casal_primal_dual": {
                "n_samples": n_samples,
                "casal_steps": CASAL_STEPS,
                "step_size": CASAL_STEP_SIZE,
                "noise_scale": CASAL_NOISE_SCALE,
                **casal_summary,
            },
        },
    }


def sample_adamflip_soft_penalty_mcmc(
    benchmark: ConstraintBenchmark,
    model: ContinuousEBM,
    initial_states: np.ndarray,
    *,
    random_seed: int,
) -> np.ndarray:
    """Regime A: MCMC over AdamFLIP-trained parameters with soft penalties."""

    rng = np.random.default_rng(random_seed)
    states = np.asarray(initial_states, dtype=np.float64).copy()
    stiffness = -np.asarray(model.coupling, dtype=np.float64)
    bias = np.asarray(model.bias, dtype=np.float64)
    noise_std = MCMC_NOISE_SCALE * np.sqrt(2.0 * MCMC_LR)

    for step in range(MCMC_STEPS):
        residual = _constraint_residuals(benchmark, states)
        grad_energy = states @ stiffness.T - bias
        grad_penalty = SOFT_PENALTY_WEIGHT * residual @ benchmark.constraint_matrix
        temp = 0.5 * (1.0 + np.cos(np.pi * step / max(MCMC_STEPS - 1, 1)))
        noise = noise_std * temp * rng.standard_normal(states.shape)
        states = states - MCMC_LR * (grad_energy + grad_penalty) + noise

    return states


def sample_casal_primal_dual(
    benchmark: ConstraintBenchmark,
    model: ContinuousEBM,
    initial_states: np.ndarray,
    *,
    random_seed: int,
) -> np.ndarray:
    """Regime B: batched CASAL primal-dual sampling on the same energy."""

    _, jnp = _load_jax_x64()
    from carnot.samplers.casal import CASALSampler

    coupling = jnp.asarray(model.coupling, dtype=jnp.float64)
    bias = jnp.asarray(model.bias, dtype=jnp.float64)
    constraint_matrix = jnp.asarray(benchmark.constraint_matrix, dtype=jnp.float64)
    constraint_target = jnp.asarray(benchmark.constraint_target, dtype=jnp.float64)

    def energy_fn(states: Any) -> Any:
        return -0.5 * jnp.sum(states * (states @ coupling.T)) - jnp.sum(states * bias)

    def residual_fn(states: Any) -> Any:
        return jnp.ravel(states @ constraint_matrix.T - constraint_target)

    sampler = CASALSampler(
        constraints=residual_fn,
        step_size=CASAL_STEP_SIZE,
        dual_step_size=0.8,
        n_steps=CASAL_STEPS,
        seed=random_seed,
        noise_scale=CASAL_NOISE_SCALE,
        projection_steps=2,
        projection_damping=0.0,
        penalty_weight=1.0,
    )
    sampled = sampler.sample(jnp.asarray(initial_states, dtype=jnp.float64), energy_fn)
    return np.asarray(sampled, dtype=np.float64)


def summarize_regime(
    benchmark: ConstraintBenchmark,
    model: ContinuousEBM,
    states: np.ndarray,
) -> JsonDict:
    """Return violation and energy statistics for one sample set."""

    residual_abs = _constraint_abs_residuals(benchmark, states)
    energies = np.asarray([continuous_ebm_energy(model, state) for state in states])
    return {
        "violation_mean": float(residual_abs.mean()),
        "max_constraint_violation": float(residual_abs.max()),
        "constraint_violation_mean_by_constraint": residual_abs.mean(axis=0).tolist(),
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std(ddof=0)),
    }


def continuous_ebm_energy(model: ContinuousEBM, state: np.ndarray) -> float:
    """Evaluate the ContinuousEBM quadratic energy for one state."""

    x = np.asarray(state, dtype=np.float64)
    coupling = np.asarray(model.coupling, dtype=np.float64)
    bias = np.asarray(model.bias, dtype=np.float64)
    return float(-0.5 * x @ coupling @ x - bias @ x)


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    casal_module: str = CASAL_MODULE,
    adamflip_module: str = ADAMFLIP_MODULE,
    n_samples: int = N_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run Exp 2246 and write the JSON deliverable."""

    output = Path(output_path)
    preconditions = check_preconditions(
        casal_module=casal_module,
        adamflip_module=adamflip_module,
    )
    failed = next((item for item in preconditions if item["status"] != "passed"), None)
    if failed is not None:
        verdict = (
            "blocked_casal_missing"
            if failed["name"] == "casal_import"
            else "blocked_adamflip_missing"
        )
        artifact = _blocked_artifact(
            honest_verdict=verdict,
            preconditions=preconditions,
            random_seed=random_seed,
        )
    else:
        artifact = run_benchmark(n_samples=n_samples, random_seed=random_seed)
        artifact["preconditions_checked"] = preconditions

    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the subset of Exp 2246 schema needed by downstream gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")

    for field, principle in FIELD_PRINCIPLES.items():
        if artifact.get("field_principles", {}).get(field) != principle:
            raise ValueError(f"missing or incorrect field principle for {field}")

    if not isinstance(artifact["casal_validated"], bool):
        raise ValueError("casal_validated must be a boolean")
    if not isinstance(artifact["random_seed"], int):
        raise ValueError("random_seed must be an integer")

    if artifact.get("status") == "blocked":
        if artifact["honest_verdict"] not in {
            "blocked_casal_missing",
            "blocked_adamflip_missing",
        }:
            raise ValueError("blocked artifacts must use the requested blocker verdict")
        if artifact["n_samples"] != 0:
            raise ValueError("blocked artifacts must not claim samples")
        return

    if artifact["n_samples"] < N_SAMPLES:
        raise ValueError(f"n_samples must be at least {N_SAMPLES}")

    numeric_fields = (
        "casal_violation_mean",
        "adamflip_violation_mean",
        "casal_max_constraint_violation",
        "adamflip_max_constraint_violation",
        "casal_energy_mean",
        "adamflip_energy_mean",
    )
    for field in numeric_fields:
        value = artifact.get(field)
        if not isinstance(value, int | float) or not np.isfinite(value):
            raise ValueError(f"{field} must be finite")

    expected_gate = artifact["casal_violation_mean"] <= artifact["adamflip_violation_mean"] / 2.0
    if artifact["casal_validated"] is not expected_gate:
        raise ValueError("casal_validated does not match the half-violation gate")

    if expected_gate and not artifact["honest_verdict"].startswith("complete:"):
        raise ValueError("passing artifacts must use a complete: verdict")
    if not expected_gate and not artifact["honest_verdict"].startswith("incomplete:"):
        raise ValueError("failing artifacts must use an incomplete: verdict")


def _blocked_artifact(
    *,
    honest_verdict: str,
    preconditions: list[JsonDict],
    random_seed: int,
) -> JsonDict:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": _run_date(),
        "status": "blocked",
        "honest_verdict": honest_verdict,
        "casal_validated": False,
        "casal_violation_mean": None,
        "adamflip_violation_mean": None,
        "casal_max_constraint_violation": None,
        "adamflip_max_constraint_violation": None,
        "casal_energy_mean": None,
        "adamflip_energy_mean": None,
        "max_constraint_violation": {"casal": None, "adamflip": None},
        "energy_mean": {"casal": None, "adamflip": None},
        "n_samples": 0,
        "random_seed": random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "regimes": {},
    }


def _constraint_residuals(
    benchmark: ConstraintBenchmark,
    states: np.ndarray,
) -> np.ndarray:
    state_arr = np.asarray(states, dtype=np.float64)
    return state_arr @ benchmark.constraint_matrix.T - benchmark.constraint_target


def _constraint_abs_residuals(
    benchmark: ConstraintBenchmark,
    states: np.ndarray,
) -> np.ndarray:
    return np.abs(_constraint_residuals(benchmark, states))


def _draw_initial_states(n_samples: int, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    return rng.standard_normal((n_samples, 3))


def _load_jax_x64() -> tuple[Any, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return jax, jnp


def _run_date() -> str:
    return dt.date.today().strftime("%Y%m%d")


def _write_json(output_path: Path, artifact: JsonDict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Artifact path for results/{OUTPUT_FILE}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_experiment(output_path=args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
