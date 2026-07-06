"""Exp 5301 deterministic EBT spectral step-control diagnostic.

Spec refs: REQ-INFER-5301, SCENARIO-INFER-5301.

This module is a small stability certificate, not a language-model quality
claim.  It uses a three-dimensional sharpened quadratic as a stand-in for the
continuous latent energy minimized inside an EBT-style inner loop.  The sharp
axis makes the local curvature large enough that a fixed aggressive step
overshoots, while a spectral estimate of the largest Hessian eigenvalue gives a
plain reason to shrink the step before accepting it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any


JsonDict = dict[str, Any]
Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]

RUN_DATE = "20260706"
RANDOM_SEED = 5301
EXPERIMENT_ID = "exp5301-ebt-spectral-step-control-diagnostic-v484"
SCHEMA = "carnot.experiment_5301.ebt_spectral_step_control_diagnostic.v484"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5301_ebt_spectral_step_control_diagnostic_v484.json"
)
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
SPEC_REFS = ("REQ-INFER-5301", "SCENARIO-INFER-5301")
TERMINAL_PREFIXES = ("complete:", "null:", "blocked_")

POWER_ITERATIONS = 24
MAX_STEPS = 8
FIXED_CONSERVATIVE_ALPHA = 0.008
FIXED_AGGRESSIVE_ALPHA = 0.03
ADAPTIVE_INITIAL_FACTOR = 2.4
ADAPTIVE_BACKTRACK_FACTOR = 0.5
STABILITY_LIMIT_FACTOR = 2.0
DIVERGENCE_INCREASE_FACTOR = 1.05
CONVERGENCE_ENERGY_RATIO = 0.02

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5301 verdict; starts with complete:, null:, or blocked_ "
        "and states whether spectral step-control is usable."
    ),
    "inference_substrate": (
        "Must be offline_deterministic_certificate_no_llm because the diagnostic "
        "uses only a CPU-local analytic energy fixture and no LLM inference."
    ),
    "spectral_control_ready": (
        "True only when the conservative policy decreases energy, the aggressive "
        "policy detects divergence, and adaptive spectral control recovers without "
        "claiming LLM quality."
    ),
    "lambda_max_estimates": (
        "Largest local Hessian eigenvalue estimates from deterministic "
        "power-iteration Hessian-vector products for every descent step."
    ),
    "alpha_policy_results": (
        "Per-policy alpha choices, energy before/after traces, convergence flags, "
        "and divergence flags for fixed conservative, fixed aggressive, and "
        "adaptive spectral policies."
    ),
    "divergence_recovery": (
        "Telemetry showing the aggressive fixed step diverges and the adaptive "
        "spectral policy shrinks alpha before accepting stable steps."
    ),
    "random_seed": (
        "Deterministic seed used for the power-iteration probe vector and checksum "
        "provenance."
    ),
    "reproducibility_checksum": (
        "SHA-256 checksum over the deterministic fixture, lambda estimates, alpha "
        "traces, and stability decisions, excluding wall-clock time."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "spectral_control_ready",
    "lambda_max_estimates",
    "alpha_policy_results",
    "divergence_recovery",
    "random_seed",
    "reproducibility_checksum",
    "tests_run",
)
WRAPPED_FIELDS = tuple(field for field in REQUIRED_ARTIFACT_FIELDS if field != "tests_run")


@dataclass(frozen=True)
class QuadraticEnergyFixture:
    """Small continuous energy with one sharp curvature direction."""

    fixture_id: str
    hessian: Matrix
    initial_state: Vector

    @property
    def condition_number(self) -> float:
        eigenvalues = [row[index] for index, row in enumerate(self.hessian)]
        return max(eigenvalues) / min(eigenvalues)

    def energy(self, state: Sequence[float]) -> float:
        """Return 0.5 * x^T H x for the current latent state."""

        h_state = _matvec(self.hessian, state)
        return 0.5 * _dot(state, h_state)

    def gradient(self, state: Sequence[float]) -> Vector:
        """Return the exact gradient of the quadratic fixture."""

        return _matvec(self.hessian, state)

    def hessian_vector_product(self, state: Sequence[float], vector: Sequence[float]) -> Vector:
        """Estimate a local Hessian-vector product by central differences."""

        epsilon = 1e-5
        plus = _add(state, _scale(vector, epsilon))
        minus = _add(state, _scale(vector, -epsilon))
        return _scale(_sub(self.gradient(plus), self.gradient(minus)), 1.0 / (2.0 * epsilon))

    def as_serializable(self) -> JsonDict:
        return {
            "fixture_id": self.fixture_id,
            "hessian_diagonal": [row[index] for index, row in enumerate(self.hessian)],
            "initial_state": list(self.initial_state),
            "condition_number": _round_float(self.condition_number),
        }


@dataclass(frozen=True)
class StepTelemetry:
    """One accepted or rejected descent step in an alpha policy trace."""

    step_index: int
    lambda_max_estimate: float
    alpha: float
    energy_before: float
    energy_after: float
    recovery_shrinks: int
    divergence_detected: bool

    def as_serializable(self) -> JsonDict:
        return {
            "step_index": self.step_index,
            "lambda_max_estimate": _round_float(self.lambda_max_estimate),
            "alpha": _round_float(self.alpha),
            "energy_before": _round_float(self.energy_before),
            "energy_after": _round_float(self.energy_after),
            "recovery_shrinks": self.recovery_shrinks,
            "divergence_detected": self.divergence_detected,
        }


@dataclass(frozen=True)
class PolicyResult:
    """Telemetry for one fixed or adaptive energy-descent policy."""

    policy_name: str
    initial_energy: float
    final_energy: float
    steps: tuple[StepTelemetry, ...]
    converged: bool
    diverged: bool
    recovered: bool

    @property
    def total_recovery_shrinks(self) -> int:
        return sum(step.recovery_shrinks for step in self.steps)

    def as_serializable(self) -> JsonDict:
        return {
            "policy_name": self.policy_name,
            "initial_energy": _round_float(self.initial_energy),
            "final_energy": _round_float(self.final_energy),
            "converged": self.converged,
            "diverged": self.diverged,
            "recovered": self.recovered,
            "total_recovery_shrinks": self.total_recovery_shrinks,
            "steps": [step.as_serializable() for step in self.steps],
        }


@dataclass(frozen=True)
class DiagnosticResult:
    """Complete comparison of the fixed and adaptive alpha policies."""

    fixture: QuadraticEnergyFixture
    policy_results: Mapping[str, PolicyResult]

    def summary(self) -> JsonDict:
        conservative = self.policy_results["fixed_conservative"]
        aggressive = self.policy_results["fixed_aggressive"]
        adaptive = self.policy_results["adaptive_spectral"]
        spectral_ready = (
            conservative.converged
            and not conservative.diverged
            and aggressive.diverged
            and adaptive.recovered
            and adaptive.converged
            and not adaptive.diverged
        )
        return {
            "fixture": self.fixture.as_serializable(),
            "spectral_control_ready": spectral_ready,
            "lambda_max_estimates": {
                "fixture": self.fixture.fixture_id,
                "power_iterations": POWER_ITERATIONS,
                "by_policy": {
                    name: [
                        _round_float(step.lambda_max_estimate)
                        for step in result.steps
                    ]
                    for name, result in self.policy_results.items()
                },
            },
            "alpha_policy_results": {
                name: result.as_serializable()
                for name, result in self.policy_results.items()
            },
            "divergence_recovery": {
                "aggressive_diverged": aggressive.diverged,
                "aggressive_divergence_step": (
                    aggressive.steps[-1].step_index if aggressive.steps else None
                ),
                "adaptive_recovered": adaptive.recovered,
                "adaptive_diverged": adaptive.diverged,
                "adaptive_total_recovery_shrinks": adaptive.total_recovery_shrinks,
                "recovery_rule": (
                    "start from 2.4/lambda_max, then halve until the step is below "
                    "the 2/lambda_max stability limit and does not increase energy"
                ),
            },
        }


def build_sharpened_fixture() -> QuadraticEnergyFixture:
    """Build the deterministic ill-conditioned energy landscape."""

    return QuadraticEnergyFixture(
        fixture_id="ill_conditioned_sharpened_quadratic",
        hessian=(
            (1.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 100.0),
        ),
        initial_state=(0.4, -0.5, 1.0),
    )


def estimate_lambda_max(
    fixture: QuadraticEnergyFixture,
    state: Sequence[float],
    *,
    seed: int,
    iterations: int,
) -> float:
    """Estimate the largest local Hessian eigenvalue by power iteration."""

    vector = _seeded_unit_vector(len(state), seed)
    for _ in range(iterations):
        hvp = fixture.hessian_vector_product(state, vector)
        vector = _normalize(hvp)
    rayleigh = _dot(vector, fixture.hessian_vector_product(state, vector))
    return _round_float(rayleigh)


def run_policy(fixture: QuadraticEnergyFixture, policy_name: str) -> PolicyResult:
    """Run one deterministic alpha policy from the fixture initial state."""

    state = tuple(fixture.initial_state)
    initial_energy = fixture.energy(state)
    steps: list[StepTelemetry] = []
    diverged = False
    for step_index in range(MAX_STEPS):
        energy_before = fixture.energy(state)
        lambda_max = estimate_lambda_max(
            fixture,
            state,
            seed=RANDOM_SEED + step_index,
            iterations=POWER_ITERATIONS,
        )
        alpha, recovery_shrinks = _alpha_for_policy(policy_name, lambda_max)
        next_state = _descent_step(fixture, state, alpha)
        energy_after = fixture.energy(next_state)
        if policy_name == "adaptive_spectral":
            alpha, recovery_shrinks, next_state, energy_after = _recover_adaptive_step(
                fixture,
                state,
                alpha,
                lambda_max,
                energy_before,
                recovery_shrinks,
            )
        divergence_detected = (
            not math.isfinite(energy_after)
            or energy_after > energy_before * DIVERGENCE_INCREASE_FACTOR
        )
        steps.append(
            StepTelemetry(
                step_index=step_index,
                lambda_max_estimate=lambda_max,
                alpha=alpha,
                energy_before=energy_before,
                energy_after=energy_after,
                recovery_shrinks=recovery_shrinks,
                divergence_detected=divergence_detected,
            )
        )
        if divergence_detected:
            diverged = True
            break
        state = next_state
    final_energy = steps[-1].energy_after
    converged = bool(not diverged and final_energy <= initial_energy * CONVERGENCE_ENERGY_RATIO)
    recovered = any(step.recovery_shrinks > 0 for step in steps)
    return PolicyResult(
        policy_name=policy_name,
        initial_energy=initial_energy,
        final_energy=final_energy,
        steps=tuple(steps),
        converged=converged,
        diverged=diverged,
        recovered=recovered,
    )


def run_diagnostic() -> DiagnosticResult:
    """Run the three-policy stability diagnostic."""

    fixture = build_sharpened_fixture()
    return DiagnosticResult(
        fixture=fixture,
        policy_results={
            policy: run_policy(fixture, policy)
            for policy in ("fixed_conservative", "fixed_aggressive", "adaptive_spectral")
        },
    )


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the artifact principle required for Exp 5301 fields."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def reproducibility_checksum(diagnostic: DiagnosticResult) -> str:
    """Return the stable checksum for a diagnostic result."""

    return _checksum_summary(diagnostic.summary())


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and validate the Exp 5301 terminal artifact."""

    started_at = time.perf_counter()
    diagnostic = run_diagnostic()
    summary = diagnostic.summary()
    measured_duration = (
        round(time.perf_counter() - started_at, 6)
        if duration_s is None
        else duration_s
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "honest_verdict": wrap_field("honest_verdict", _honest_verdict(summary)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "spectral_control_ready": wrap_field(
            "spectral_control_ready",
            summary["spectral_control_ready"],
        ),
        "lambda_max_estimates": wrap_field(
            "lambda_max_estimates",
            summary["lambda_max_estimates"],
        ),
        "alpha_policy_results": wrap_field(
            "alpha_policy_results",
            summary["alpha_policy_results"],
        ),
        "divergence_recovery": wrap_field(
            "divergence_recovery",
            summary["divergence_recovery"],
        ),
        "random_seed": wrap_field("random_seed", RANDOM_SEED),
        "reproducibility_checksum": wrap_field(
            "reproducibility_checksum",
            _checksum_summary(summary),
        ),
        "tests_run": [dict(row) for row in tests_run or []],
        "diagnostic_summary": summary,
        "llm_quality_claimed": False,
        "hardware_speedup_claimed": False,
        "claim_limits": [
            "offline deterministic continuous-energy fixture only",
            "no LLM decoding or language-quality measurement",
            "no SOTA model, GGUF, GPU, or hardware-speedup claim",
            "adaptive control is usable only as a tiny stability diagnostic",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5301 artifact drifts from its contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    verdict = artifact["honest_verdict"]["value"]
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require("spectral step-control is usable" in verdict, "honest_verdict must state usability")
    _require(artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, "inference_substrate drift")
    _require(artifact["spectral_control_ready"]["value"] is True, "spectral control must be ready")
    _require(artifact.get("llm_quality_claimed") is False, "LLM quality claim must be false")
    _require(artifact.get("hardware_speedup_claimed") is False, "hardware speedup claim must be false")
    _require(artifact["random_seed"]["value"] == RANDOM_SEED, "random_seed drift")
    _require(_valid_tests_run(artifact["tests_run"]), "tests_run must contain command/outcome rows")

    lambda_estimates = artifact["lambda_max_estimates"]["value"]
    _require(lambda_estimates["fixture"] == "ill_conditioned_sharpened_quadratic", "lambda fixture drift")
    _require(
        all(all(value > 0.0 for value in values) for values in lambda_estimates["by_policy"].values()),
        "lambda estimates must be positive",
    )
    policy_results = artifact["alpha_policy_results"]["value"]
    _require(policy_results["fixed_conservative"]["converged"] is True, "conservative policy drift")
    _require(policy_results["fixed_aggressive"]["diverged"] is True, "aggressive policy drift")
    _require(policy_results["adaptive_spectral"]["recovered"] is True, "adaptive recovery drift")
    _require(policy_results["adaptive_spectral"]["diverged"] is False, "adaptive divergence drift")

    recovery = artifact["divergence_recovery"]["value"]
    _require(recovery["aggressive_diverged"] is True, "aggressive recovery summary drift")
    _require(recovery["adaptive_recovered"] is True, "adaptive recovery summary drift")
    _require(recovery["adaptive_diverged"] is False, "adaptive recovery divergence drift")
    expected_checksum = _checksum_summary(artifact["diagnostic_summary"])
    _require(
        artifact["reproducibility_checksum"]["value"] == expected_checksum,
        "checksum drift",
    )
    _require("REQ-INFER-5301" in artifact["spec_refs"], "spec refs must include REQ-INFER-5301")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5301 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _alpha_for_policy(policy_name: str, lambda_max: float) -> tuple[float, int]:
    if policy_name == "fixed_conservative":
        return FIXED_CONSERVATIVE_ALPHA, 0
    if policy_name == "fixed_aggressive":
        return FIXED_AGGRESSIVE_ALPHA, 0
    if policy_name == "adaptive_spectral":
        return _round_float(ADAPTIVE_INITIAL_FACTOR / lambda_max), 0
    raise ValueError(f"unknown alpha policy: {policy_name}")


def _recover_adaptive_step(
    fixture: QuadraticEnergyFixture,
    state: Vector,
    alpha: float,
    lambda_max: float,
    energy_before: float,
    recovery_shrinks: int,
) -> tuple[float, int, Vector, float]:
    stability_limit = STABILITY_LIMIT_FACTOR / lambda_max
    next_state = _descent_step(fixture, state, alpha)
    energy_after = fixture.energy(next_state)
    while alpha > stability_limit or energy_after > energy_before:
        alpha = _round_float(alpha * ADAPTIVE_BACKTRACK_FACTOR)
        recovery_shrinks += 1
        next_state = _descent_step(fixture, state, alpha)
        energy_after = fixture.energy(next_state)
    return alpha, recovery_shrinks, next_state, energy_after


def _descent_step(
    fixture: QuadraticEnergyFixture,
    state: Sequence[float],
    alpha: float,
) -> Vector:
    return _sub(state, _scale(fixture.gradient(state), alpha))


def _honest_verdict(summary: Mapping[str, Any]) -> str:
    if summary["spectral_control_ready"]:
        return (
            "complete: spectral step-control is usable as a tiny deterministic "
            "stability diagnostic before energy-guided inner-loop claims"
        )
    return "null: spectral step-control is not usable on the bounded diagnostic"


def _valid_tests_run(rows: Any) -> bool:
    return isinstance(rows, list) and all(
        isinstance(row, Mapping)
        and isinstance(row.get("command"), str)
        and isinstance(row.get("outcome"), str)
        for row in rows
    )


def _checksum_summary(summary: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "summary": summary,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _seeded_unit_vector(dim: int, seed: int) -> Vector:
    rng = random.Random(seed)
    return _normalize(tuple(rng.uniform(-1.0, 1.0) for _ in range(dim)))


def _matvec(matrix: Matrix, vector: Sequence[float]) -> Vector:
    return tuple(sum(row[index] * float(vector[index]) for index in range(len(vector))) for row in matrix)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Sequence[float]) -> Vector:
    norm = _norm(vector)
    if norm == 0.0:  # pragma: no cover - the fixture Hessian is positive definite.
        raise ValueError("cannot normalize zero vector")
    return tuple(float(value) / norm for value in vector)


def _scale(vector: Sequence[float], scalar: float) -> Vector:
    return tuple(float(value) * scalar for value in vector)


def _add(left: Sequence[float], right: Sequence[float]) -> Vector:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _sub(left: Sequence[float], right: Sequence[float]) -> Vector:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_outputs()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
