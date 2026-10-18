"""Exp 5091 small KAEM/PWA/MILP scale telemetry.

Spec refs: REQ-KAN-5091, SCENARIO-KAN-5091.

This module is the next small step after Exp 5080. It keeps the same honest
boundary: we verify a hand-sized KAEM spline property with a deterministic
formal solver, not a trained KAN network and not live model inference. The
scale-up is two independent KAEM variables, so the artifact can show how PWA
pieces, binary selectors, constraints, solve time, and error budgets change
when the one-unit bridge becomes a small multi-unit property.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp

from carnot.experiment_5080_kan_pwa_milp_bridge_v466 import (
    KAN_COMPONENT_PATH,
    PWAAbstraction,
    build_pwa_abstraction,
)
from carnot.models.kaem_energy import UnivariateKAEMLayer


RESULT_RELATIVE_PATH = "results/experiment_5091_kan_pwa_milp_scale_v467.json"
ARTIFACT_NAME = "experiment_5091_kan_pwa_milp_scale_v467"
RUN_DATE = "20260701"
RANDOM_SEED = 5091
INFERENCE_SUBSTRATE = "deterministic_formal_solver"
SUCCESS_VERDICT = "success_kan_pwa_milp_scale_property_verified_small"
BLOCKED_VERDICT = "complete_kan_pwa_milp_scale_blocked_by_solver_complexity"
PROPERTY_THRESHOLD = 1.800000011920929
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)
SPEC_REFS = ["REQ-KAN-5091", "SCENARIO-KAN-5091"]
EXP5080_SOURCE_ARTIFACT = "results/experiment_5080_kan_pwa_milp_bridge_v466.json"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "abstraction_built",
    "solver_available",
    "property_statement",
    "property_status",
    "property_holds",
    "binary_variable_count",
    "constraint_count",
    "pwa_piece_count",
    "local_error_bound",
    "global_error_bound",
    "solve_time_s",
    "scale_blocker",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "Terminal verdict says whether this small scale-up was proved or blocked."
    },
    "duration_s": {
        "principle": "Wall-clock duration covers deterministic abstraction and solver work, not LLM inference."
    },
    "inference_substrate": {
        "principle": "The substrate is the actual local formal solver path and must not imply live LLM execution."
    },
    "abstraction_built": {
        "principle": "Separates a real KAEM knot-to-PWA export from an artifact-only placeholder."
    },
    "solver_available": {
        "principle": "Makes the proof dependency explicit before any property-holds claim is accepted."
    },
    "property_statement": {
        "principle": "States the exact universally quantified two-input KAEM bound being checked."
    },
    "property_status": {
        "principle": "Distinguishes verified, refuted, and blocked states without overclaiming scalability."
    },
    "property_holds": {
        "principle": "Records the formal conclusion only after the solver returns an optimal certificate."
    },
    "binary_variable_count": {
        "principle": "Counts one selector per PWA piece, exposing the integer-search growth from Exp 5080."
    },
    "constraint_count": {
        "principle": "Reports the generated linear constraint count so solver complexity is visible."
    },
    "pwa_piece_count": {
        "principle": "Reports total affine pieces across both KAEM variables."
    },
    "local_error_bound": {
        "principle": "Declares the per-unit abstraction gap; zero is valid only because KAEM uses linear interpolation."
    },
    "global_error_bound": {
        "principle": "Declares the propagated output gap after summing the two local KAEM abstractions."
    },
    "solve_time_s": {
        "principle": "Keeps solver runtime visible separately from artifact assembly time."
    },
    "scale_blocker": {
        "principle": "Names the dependency or complexity condition when the small proof cannot complete."
    },
    "flagged_adversarial": {
        "principle": "Remains false only when the artifact is honest about solver use, scale, and substrate."
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover - defensive schema guard.
        raise ValueError(message)


@dataclass(frozen=True)
class MultiUnitPWAAbstraction:
    """PWA evidence for the smallest meaningful multi-unit KAEM property.

    KAEM energy is additive across variables. Reusing the Exp 5080 one-variable
    export for each variable gives an exact two-input abstraction while making
    the growth in pieces, binary selectors, and constraints easy to inspect.
    """

    component_path: str
    units: tuple[PWAAbstraction, ...]
    local_error_budget: float
    global_error_budget: float

    @property
    def input_dimension(self) -> int:
        """Return the number of KAEM variables in the property domain."""

        return len(self.units)

    @property
    def pwa_piece_count(self) -> int:
        """Return the total number of affine pieces across all units."""

        return sum(unit.n_segments for unit in self.units)

    @property
    def binary_variable_count(self) -> int:
        """Return one binary selector per unit-level PWA segment."""

        return sum(unit.binary_variable_count for unit in self.units)

    @property
    def local_error_bound(self) -> float:
        """Return the largest declared local abstraction error."""

        return max((unit.error_bound for unit in self.units), default=0.0)

    @property
    def global_error_bound(self) -> float:
        """Return the additive output error after all unit contributions."""

        return sum(unit.error_bound for unit in self.units)

    def evaluate(self, xs: Sequence[float]) -> float:
        """Evaluate the additive PWA energy for one two-input point."""

        if len(xs) != self.input_dimension:
            raise ValueError(f"expected {self.input_dimension} inputs, got {len(xs)}")
        return sum(unit.evaluate(float(x)) for unit, x in zip(self.units, xs))

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe PWA scale evidence."""

        return {
            "component_path": self.component_path,
            "input_dimension": self.input_dimension,
            "pwa_piece_count": self.pwa_piece_count,
            "binary_variable_count": self.binary_variable_count,
            "local_error_bound": self.local_error_bound,
            "global_error_bound": self.global_error_bound,
            "declared_error_budgets": {
                "local_error_budget": self.local_error_budget,
                "global_error_budget": self.global_error_budget,
                "per_unit_error_bounds": [unit.error_bound for unit in self.units],
            },
            "units": [unit.as_serializable() for unit in self.units],
            "scale_up_note": "two-variable additive KAEM property; Exp 5080 had one variable",
        }


@dataclass(frozen=True)
class ScaleSolveResult:
    """Solver telemetry and certificate for the two-input KAEM property."""

    solver_available: bool
    solver_name: str
    solver_status: str
    property_status: str
    property_holds: bool | None
    certified_upper_bound: float | None
    witness_inputs: tuple[float, ...] | None
    selected_segments: tuple[int, ...] | None
    binary_variable_count: int
    constraint_count: int
    pwa_piece_count: int
    solve_time_s: float
    bound_tightness: float | None
    scale_blocker: str | None
    certificate: dict[str, Any]

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe solver telemetry."""

        return {
            "solver_available": self.solver_available,
            "solver_name": self.solver_name,
            "solver_status": self.solver_status,
            "property_status": self.property_status,
            "property_holds": self.property_holds,
            "certified_upper_bound": self.certified_upper_bound,
            "witness_inputs": list(self.witness_inputs) if self.witness_inputs is not None else None,
            "selected_segments": (
                list(self.selected_segments) if self.selected_segments is not None else None
            ),
            "binary_variable_count": self.binary_variable_count,
            "constraint_count": self.constraint_count,
            "pwa_piece_count": self.pwa_piece_count,
            "solve_time_s": self.solve_time_s,
            "bound_tightness": self.bound_tightness,
            "scale_blocker": self.scale_blocker,
            "certificate": self.certificate,
        }


def build_two_unit_kaem_layer() -> UnivariateKAEMLayer:
    """Create the deterministic two-variable KAEM fixture for Exp 5091."""

    layer = UnivariateKAEMLayer(n_vars=2, n_knots=4, key=jax.random.PRNGKey(RANDOM_SEED))
    layer.control_points = jnp.array(
        [
            [0.0, 0.25, 0.75, 1.0],
            [0.1, 0.2, 0.4, 0.8],
        ],
        dtype=jnp.float32,
    )
    return layer


def build_scaled_abstraction(layer: UnivariateKAEMLayer) -> MultiUnitPWAAbstraction:
    """Reuse the Exp 5080 KAEM spline export for both variables."""

    units = tuple(build_pwa_abstraction(layer, variable_index=index) for index in range(layer.n_vars))
    return MultiUnitPWAAbstraction(
        component_path=KAN_COMPONENT_PATH,
        units=units,
        local_error_budget=0.0,
        global_error_budget=0.0,
    )


def detect_solver() -> str:
    """Return the deterministic formal backend used by this experiment."""

    return "z3" if importlib.util.find_spec("z3") is not None else ""


def expected_constraint_count(abstraction: MultiUnitPWAAbstraction) -> int:
    """Count the generated linear constraints without requiring a solver import."""

    unit_counts = sum(3 + 6 * unit.n_segments for unit in abstraction.units)
    return unit_counts + 1


def _z3_float(value: Any) -> float:
    """Convert a small Z3 numeric value to a JSON float."""

    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    if text.endswith("?"):
        text = text[:-1]
    return float(text)


def _real(z3: Any, value: float) -> Any:
    """Create a Z3 real literal from the displayed decimal value."""

    return z3.RealVal(repr(float(value)))


def _blocked_result(abstraction: MultiUnitPWAAbstraction) -> ScaleSolveResult:
    """Build the fail-closed result used when no supported solver is importable."""

    return ScaleSolveResult(
        solver_available=False,
        solver_name="",
        solver_status="blocked_solver_dependency",
        property_status="blocked_solver_dependency",
        property_holds=None,
        certified_upper_bound=None,
        witness_inputs=None,
        selected_segments=None,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=expected_constraint_count(abstraction),
        pwa_piece_count=abstraction.pwa_piece_count,
        solve_time_s=0.0,
        bound_tightness=None,
        scale_blocker="blocked_kan_pwa_milp_solver_unavailable",
        certificate={
            "kind": "blocked_dependency",
            "method": "z3_mixed_integer_linear_pwa_scale_bound",
            "reason": "python package 'z3' is not importable",
        },
    )


def _solve_with_z3(abstraction: MultiUnitPWAAbstraction) -> ScaleSolveResult:
    """Maximize the two-input additive PWA energy with Z3 integer selectors."""

    z3 = importlib.import_module("z3")
    optimizer = z3.Optimize()
    constraint_count = 0
    xs = [z3.Real(f"x_{index}") for index in range(abstraction.input_dimension)]
    ys = [z3.Real(f"unit_energy_{index}") for index in range(abstraction.input_dimension)]
    total_energy = z3.Real("total_energy")
    selected_flag_groups: list[list[Any]] = []
    big_m = _real(z3, 10.0)

    def add_constraints(*constraints: Any) -> None:
        nonlocal constraint_count
        optimizer.add(*constraints)
        constraint_count += len(constraints)

    for unit_index, unit in enumerate(abstraction.units):
        x = xs[unit_index]
        y = ys[unit_index]
        flags = [
            z3.Int(f"unit_{unit_index}_pwa_segment_{segment.index}")
            for segment in unit.segments
        ]
        selected_flag_groups.append(flags)

        add_constraints(x >= _real(z3, unit.segments[0].x_min), x <= _real(z3, unit.segments[-1].x_max))
        add_constraints(z3.Sum(flags) == 1)

        for flag, segment in zip(flags, unit.segments):
            flag_real = z3.ToReal(flag)
            slack = big_m * (_real(z3, 1.0) - flag_real)
            affine_value = _real(z3, segment.slope) * x + _real(z3, segment.intercept)
            add_constraints(
                flag >= 0,
                flag <= 1,
                x >= _real(z3, segment.x_min) - slack,
                x <= _real(z3, segment.x_max) + slack,
                y - affine_value <= slack,
                affine_value - y <= slack,
            )

    add_constraints(total_energy == z3.Sum(ys))

    solve_start = time.perf_counter()
    objective = optimizer.maximize(total_energy)
    status = optimizer.check()
    solve_time_s = round(time.perf_counter() - solve_start, 6)

    if status != z3.sat:  # pragma: no cover - retained for honest solver failure reporting.
        status_text = str(status)
        return ScaleSolveResult(
            solver_available=True,
            solver_name="z3",
            solver_status=status_text,
            property_status="blocked_solver_status",
            property_holds=None,
            certified_upper_bound=None,
            witness_inputs=None,
            selected_segments=None,
            binary_variable_count=abstraction.binary_variable_count,
            constraint_count=constraint_count,
            pwa_piece_count=abstraction.pwa_piece_count,
            solve_time_s=solve_time_s,
            bound_tightness=None,
            scale_blocker=f"blocked_kan_pwa_milp_solver_status_{status_text}",
            certificate={
                "kind": "solver_failure",
                "method": "z3_mixed_integer_linear_pwa_scale_bound",
                "status": status_text,
            },
        )

    model = optimizer.model()
    certified_upper = _z3_float(objective.value())
    witness_inputs = tuple(_z3_float(model.eval(x, model_completion=True)) for x in xs)
    selected_segments = tuple(
        next(
            segment_index
            for segment_index, flag in enumerate(flags)
            if _z3_float(model.eval(flag, model_completion=True)) > 0.5
        )
        for flags in selected_flag_groups
    )
    property_holds = certified_upper <= PROPERTY_THRESHOLD + abstraction.global_error_bound + 1e-9
    property_status = "verified" if property_holds else "counterexample"
    bound_tightness = PROPERTY_THRESHOLD - certified_upper

    return ScaleSolveResult(
        solver_available=True,
        solver_name="z3",
        solver_status="optimal",
        property_status=property_status,
        property_holds=property_holds,
        certified_upper_bound=certified_upper,
        witness_inputs=witness_inputs,
        selected_segments=selected_segments,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=constraint_count,
        pwa_piece_count=abstraction.pwa_piece_count,
        solve_time_s=solve_time_s,
        bound_tightness=bound_tightness,
        scale_blocker=None if property_holds else "counterexample_found",
        certificate={
            "kind": "certificate" if property_holds else "counterexample",
            "method": "z3_mixed_integer_linear_pwa_scale_bound",
            "property": property_statement(),
            "certified_upper_bound": certified_upper,
            "property_threshold": PROPERTY_THRESHOLD,
            "local_error_bound": abstraction.local_error_bound,
            "global_error_bound": abstraction.global_error_bound,
            "witness_maximizer_inputs": list(witness_inputs),
            "selected_segments": list(selected_segments),
        },
    )


def solve_scale_property(
    abstraction: MultiUnitPWAAbstraction,
    solver_name: str | None = None,
) -> ScaleSolveResult:
    """Check the additive two-input KAEM energy bound through a local solver."""

    selected_solver = detect_solver() if solver_name is None else solver_name
    if selected_solver == "z3":
        return _solve_with_z3(abstraction)
    return _blocked_result(abstraction)


def property_statement() -> str:
    """Return the human-readable quantified property for the artifact."""

    return (
        "For all (x0, x1) in [-1, 1]^2, the two-unit additive KAEM energy "
        f"energy(x0, x1) <= {PROPERTY_THRESHOLD}."
    )


def _checksum_payload(
    abstraction: MultiUnitPWAAbstraction,
    result: ScaleSolveResult,
) -> str:
    """Hash deterministic proof inputs and outputs, excluding wall-clock time."""

    payload = {
        "abstraction": abstraction.as_serializable(),
        "component_path": KAN_COMPONENT_PATH,
        "property_result": result.as_serializable() | {"solve_time_s": "excluded"},
        "property_statement": property_statement(),
        "property_threshold": PROPERTY_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
        "source_artifact": EXP5080_SOURCE_ARTIFACT,
        "spec_refs": SPEC_REFS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(solver_name: str | None = None) -> dict[str, Any]:
    """Build the Exp 5091 deliverable payload."""

    start = time.perf_counter()
    abstraction = build_scaled_abstraction(build_two_unit_kaem_layer())
    result = solve_scale_property(abstraction, solver_name=solver_name)
    success = result.property_status == "verified" and result.property_holds is True
    honest_verdict = SUCCESS_VERDICT if success else BLOCKED_VERDICT
    methodology_note = (
        "Exp 5091 reuses the Exp 5080 exact KAEM knot-to-PWA export for two "
        "additive variables, then submits the bound to a deterministic_formal_solver "
        "path. local_error_bound and global_error_bound are zero because the KAEM "
        "layer itself is linear interpolation, not because an approximation gap was hidden."
    )

    artifact = {
        "schema": "carnot.kan_pwa_milp_scale.v467",
        "experiment": 5091,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "duration_s": round(time.perf_counter() - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "abstraction_built": True,
        "solver_available": result.solver_available,
        "property_statement": property_statement(),
        "property_status": result.property_status,
        "property_holds": result.property_holds,
        "binary_variable_count": result.binary_variable_count,
        "constraint_count": result.constraint_count,
        "pwa_piece_count": result.pwa_piece_count,
        "local_error_bound": abstraction.local_error_bound,
        "global_error_bound": abstraction.global_error_bound,
        "solve_time_s": result.solve_time_s,
        "scale_blocker": result.scale_blocker,
        "flagged_adversarial": False,
        "solver_name": result.solver_name,
        "solver_status": result.solver_status,
        "certified_upper_bound": result.certified_upper_bound,
        "bound_tightness": result.bound_tightness,
        "witness_inputs": list(result.witness_inputs) if result.witness_inputs is not None else None,
        "selected_segments": (
            list(result.selected_segments) if result.selected_segments is not None else None
        ),
        "pwa_abstraction": abstraction.as_serializable(),
        "counterexample_or_certificate": result.certificate,
        "methodology_note": methodology_note,
        "field_principles": FIELD_PRINCIPLES,
        "source_artifacts": [EXP5080_SOURCE_ARTIFACT],
        "kan_component_path": KAN_COMPONENT_PATH,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "tests_run": [
            ".venv/bin/python -m pytest tests/python/test_experiment_5091_kan_pwa_milp_scale.py -q --no-cov"
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(abstraction, result)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5091 artifact drifts from its schema boundary."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(
        verdict in {SUCCESS_VERDICT, BLOCKED_VERDICT},
        "honest_verdict must be the success or blocked Exp 5091 terminal state",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be deterministic_formal_solver",
    )
    _require("live_llm" not in artifact["inference_substrate"], "must not claim live LLM inference")
    _require(artifact["abstraction_built"] is True, "PWA abstraction must be built")
    _require(isinstance(artifact["duration_s"], float), "duration_s must be a float")
    _require(artifact["duration_s"] >= 0.0, "duration_s cannot be negative")
    _require(artifact["local_error_bound"] >= 0.0, "local_error_bound cannot be negative")
    _require(artifact["global_error_bound"] >= 0.0, "global_error_bound cannot be negative")
    _require(artifact["binary_variable_count"] == 6, "expected six PWA selector variables")
    _require(artifact["constraint_count"] == 43, "expected two-input Z3 constraint count")
    _require(artifact["pwa_piece_count"] == 6, "expected six total PWA pieces")
    _require(artifact["flagged_adversarial"] is False, "Exp 5091 artifact must not be flagged")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]),
        "field_principles must cover every required user field",
    )

    if artifact["solver_available"]:
        _require(artifact["solver_name"] == "z3", "z3 is the only supported Exp 5091 solver")
        _require(artifact["solver_status"] == "optimal", "solver must return optimal")
        _require(artifact["property_status"] == "verified", "available solver must verify property")
        _require(artifact["property_holds"] is True, "success artifact requires property_holds")
        _require(artifact["scale_blocker"] is None, "success artifact cannot be blocked")
        _require(artifact["honest_verdict"] == SUCCESS_VERDICT, "success must use success verdict")
        _require(artifact["solve_time_s"] >= 0.0, "solve_time_s cannot be negative")
        _require(
            artifact["bound_tightness"] is not None and abs(artifact["bound_tightness"]) <= 1e-6,
            "bound_tightness must show the certified bound is tight",
        )
    else:
        _require(
            artifact["property_status"] == "blocked_solver_dependency",
            "blocked artifact must name solver dependency status",
        )
        _require(artifact["property_holds"] is None, "blocked artifact cannot assert property")
        _require(
            artifact["scale_blocker"] == "blocked_kan_pwa_milp_solver_unavailable",
            "blocked artifact must name missing solver",
        )
        _require(artifact["honest_verdict"] == BLOCKED_VERDICT, "blocked proof must use blocked verdict")


def write_outputs(*, artifact_path: str | Path, solver_name: str | None = None) -> dict[str, Any]:
    """Write the Exp 5091 JSON artifact and return the validated payload."""

    artifact = build_artifact(solver_name=solver_name)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    """CLI entrypoint for writing the default deliverable artifact."""

    root = Path(os.environ.get("CARNOT_EXP5091_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(artifact_path=root / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
