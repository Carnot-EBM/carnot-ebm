"""Exp 5098 bounded KAEM/PWA/MILP property-suite scaling.

Spec refs: REQ-KAN-5098, SCENARIO-KAN-5098.

This module scales the clean Exp 5091 bridge by checking a small suite of
bounded KAEM properties. It uses the same exact KAEM knot-to-PWA export and a
local CPU Z3 mixed-integer linear encoding. The suite includes expected-true
properties, an adversarial expected-false control, and a near-margin case whose
declared approximation budget prevents certification.
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

from carnot.experiment_5080_kan_pwa_milp_bridge_v466 import KAN_COMPONENT_PATH
from carnot.experiment_5091_kan_pwa_milp_scale_v467 import (
    EXP5080_SOURCE_ARTIFACT,
    MultiUnitPWAAbstraction,
    build_pwa_abstraction,
)
from carnot.experiment_5091_kan_pwa_milp_scale_v467 import (
    PROPERTY_THRESHOLD as EXP5091_PROPERTY_THRESHOLD,
)
from carnot.models.kaem_energy import UnivariateKAEMLayer


RESULT_RELATIVE_PATH = "results/experiment_5098_kan_pwa_milp_scale_v2.json"
ARTIFACT_NAME = "experiment_5098_kan_pwa_milp_scale_v2"
RUN_DATE = "20260701"
RANDOM_SEED = 5098
INFERENCE_SUBSTRATE = "exact_milp_solver_cpu"
SUCCESS_VERDICT = "success_kan_pwa_milp_scale_v2_property_suite_clean"
BLOCKED_VERDICT = "complete_kan_pwa_milp_scale_v2_blocked_by_solver_growth"
EXP5091_SOURCE_ARTIFACT = "results/experiment_5091_kan_pwa_milp_scale_v467.json"
SPEC_REFS = ["REQ-KAN-5098", "SCENARIO-KAN-5098"]
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

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "property_suite",
    "properties_proved",
    "false_property_controls_passed",
    "solver_statuses",
    "binary_variable_counts",
    "constraint_counts",
    "solve_times_s",
    "approximation_error_budget",
    "counterexamples",
    "max_scale_reached",
    "scale_blocker",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "Terminal verdict reports whether the bounded suite passed cleanly or blocked on solver growth/dependency."
    },
    "duration_s": {
        "principle": "Wall-clock duration covers local abstraction and exact CPU solver calls only."
    },
    "inference_substrate": {
        "principle": "Names the actual exact CPU mixed-integer solver path used for every checked property."
    },
    "property_suite": {
        "principle": "Carries one row per bounded property so true, false-control, and approximation-sensitive outcomes stay separable."
    },
    "properties_proved": {
        "principle": "Lists only properties whose solver-certified bound plus declared error budget is within threshold."
    },
    "false_property_controls_passed": {
        "principle": "True only when adversarial false properties are refuted or counterexampled instead of counted as proved."
    },
    "solver_statuses": {
        "principle": "Records the exact solver status for each property so blocked or non-optimal rows are visible."
    },
    "binary_variable_counts": {
        "principle": "Reports one selector count per property, exposing integer-search growth across the suite."
    },
    "constraint_counts": {
        "principle": "Reports generated linear-constraint counts per property, making scale telemetry explicit."
    },
    "solve_times_s": {
        "principle": "Keeps per-property solver runtime separate from total artifact assembly time."
    },
    "approximation_error_budget": {
        "principle": "Declares the certification budget applied above the PWA objective bound for each property."
    },
    "counterexamples": {
        "principle": "Stores solver witnesses for false properties so failures are auditable and not relabeled as proofs."
    },
    "max_scale_reached": {
        "principle": "Summarizes the largest bounded property solved by selector and constraint count, without extrapolating."
    },
    "scale_blocker": {
        "principle": "Names a solver dependency, solver status, or control failure when the suite cannot be called clean."
    },
    "flagged_adversarial": {
        "principle": "Remains false only when adversarial controls are handled honestly and no false property is proved."
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover - defensive schema guard.
        raise ValueError(message)


@dataclass(frozen=True)
class PropertySpec:
    """A bounded KAEM property submitted to the PWA/MILP suite."""

    property_id: str
    suite_role: str
    description: str
    control_points: tuple[tuple[float, ...], ...]
    threshold: float
    approximation_error_budget: float
    expected_outcome: str
    is_false_property_control: bool = False

    @property
    def input_dimension(self) -> int:
        """Return the number of additive KAEM variables."""

        return len(self.control_points)

    def statement(self) -> str:
        """Return the bounded property statement used in the artifact."""

        variables = ", ".join(f"x{i}" for i in range(self.input_dimension))
        return (
            f"For all ({variables}) in [-1, 1]^{self.input_dimension}, "
            f"the additive KAEM PWA energy <= {self.threshold}."
        )


@dataclass(frozen=True)
class PropertyResult:
    """Solver telemetry and verdict for one suite property."""

    property_id: str
    suite_role: str
    expected_outcome: str
    statement: str
    threshold: float
    approximation_error_budget: float
    solver_available: bool
    solver_name: str
    solver_status: str
    property_status: str
    property_holds: bool | None
    certified_upper_bound: float | None
    budgeted_upper_bound: float | None
    objective_bound: float | None
    witness_inputs: tuple[float, ...] | None
    selected_segments: tuple[int, ...] | None
    binary_variable_count: int
    constraint_count: int
    pwa_piece_count: int
    input_dimension: int
    solve_time_s: float
    is_false_property_control: bool
    counterexample: dict[str, Any] | None
    certificate: dict[str, Any]

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe property telemetry."""

        return {
            "property_id": self.property_id,
            "suite_role": self.suite_role,
            "expected_outcome": self.expected_outcome,
            "statement": self.statement,
            "threshold": self.threshold,
            "approximation_error_budget": self.approximation_error_budget,
            "solver_available": self.solver_available,
            "solver_name": self.solver_name,
            "solver_status": self.solver_status,
            "property_status": self.property_status,
            "property_holds": self.property_holds,
            "certified_upper_bound": self.certified_upper_bound,
            "budgeted_upper_bound": self.budgeted_upper_bound,
            "objective_bound": self.objective_bound,
            "witness_inputs": list(self.witness_inputs) if self.witness_inputs is not None else None,
            "selected_segments": (
                list(self.selected_segments) if self.selected_segments is not None else None
            ),
            "binary_variable_count": self.binary_variable_count,
            "constraint_count": self.constraint_count,
            "pwa_piece_count": self.pwa_piece_count,
            "input_dimension": self.input_dimension,
            "solve_time_s": self.solve_time_s,
            "is_false_property_control": self.is_false_property_control,
            "counterexample": self.counterexample,
            "certificate": self.certificate,
        }


def build_property_specs() -> tuple[PropertySpec, ...]:
    """Return the deterministic Exp 5098 bounded property suite."""

    baseline_control = (
        (0.0, 0.25, 0.75, 1.0),
        (0.1, 0.2, 0.4, 0.8),
    )
    three_unit_control = (
        (0.0, 0.2, 0.5, 0.8),
        (0.1, 0.2, 0.4, 0.6),
        (0.0, 0.1, 0.25, 0.4),
    )
    return (
        PropertySpec(
            property_id="exp5091_baseline_two_unit_true",
            suite_role="baseline_expected_true",
            description="Reproduce the clean Exp 5091 two-unit additive KAEM bound.",
            control_points=baseline_control,
            threshold=EXP5091_PROPERTY_THRESHOLD,
            approximation_error_budget=0.0,
            expected_outcome="verified",
        ),
        PropertySpec(
            property_id="three_unit_composition_true",
            suite_role="scaled_expected_true",
            description="Add one bounded KAEM unit to expose selector and constraint growth.",
            control_points=three_unit_control,
            threshold=1.81,
            approximation_error_budget=0.0,
            expected_outcome="verified",
        ),
        PropertySpec(
            property_id="adversarial_false_tight_bound",
            suite_role="adversarial_false_property_control",
            description="Use the Exp 5091 fixture with a deliberately too-tight threshold.",
            control_points=baseline_control,
            threshold=1.7,
            approximation_error_budget=0.0,
            expected_outcome="counterexample",
            is_false_property_control=True,
        ),
        PropertySpec(
            property_id="approximation_budget_sensitive_margin",
            suite_role="approximation_error_sensitive",
            description="Keep the raw PWA bound below threshold but consume the margin with declared error budget.",
            control_points=baseline_control,
            threshold=1.81,
            approximation_error_budget=0.02,
            expected_outcome="unproved_approximation_budget",
        ),
    )


def build_layer_for_property(spec: PropertySpec) -> UnivariateKAEMLayer:
    """Create the deterministic KAEM fixture for a suite property."""

    layer = UnivariateKAEMLayer(
        n_vars=spec.input_dimension,
        n_knots=4,
        key=jax.random.PRNGKey(RANDOM_SEED + spec.input_dimension),
    )
    layer.control_points = jnp.array(spec.control_points, dtype=jnp.float32)
    return layer


def build_abstraction_for_property(spec: PropertySpec) -> MultiUnitPWAAbstraction:
    """Build exact per-variable PWA abstractions for one property."""

    layer = build_layer_for_property(spec)
    units = tuple(build_pwa_abstraction(layer, variable_index=index) for index in range(layer.n_vars))
    local_budget = spec.approximation_error_budget / max(1, layer.n_vars)
    return MultiUnitPWAAbstraction(
        component_path=KAN_COMPONENT_PATH,
        units=units,
        local_error_budget=local_budget,
        global_error_budget=spec.approximation_error_budget,
    )


def detect_solver() -> str:
    """Return the exact local mixed-integer backend used by this experiment."""

    return "z3" if importlib.util.find_spec("z3") is not None else ""


def expected_constraint_count(abstraction: MultiUnitPWAAbstraction) -> int:
    """Count generated linear constraints without importing the solver."""

    return sum(3 + 6 * unit.n_segments for unit in abstraction.units) + 1


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
    """Create a Z3 real literal from a stable decimal spelling."""

    return z3.RealVal(repr(float(value)))


def _blocked_result(spec: PropertySpec, abstraction: MultiUnitPWAAbstraction) -> PropertyResult:
    """Build a fail-closed property row when no supported solver is importable."""

    return PropertyResult(
        property_id=spec.property_id,
        suite_role=spec.suite_role,
        expected_outcome=spec.expected_outcome,
        statement=spec.statement(),
        threshold=spec.threshold,
        approximation_error_budget=spec.approximation_error_budget,
        solver_available=False,
        solver_name="",
        solver_status="blocked_solver_dependency",
        property_status="blocked_solver_dependency",
        property_holds=None,
        certified_upper_bound=None,
        budgeted_upper_bound=None,
        objective_bound=None,
        witness_inputs=None,
        selected_segments=None,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=expected_constraint_count(abstraction),
        pwa_piece_count=abstraction.pwa_piece_count,
        input_dimension=abstraction.input_dimension,
        solve_time_s=0.0,
        is_false_property_control=spec.is_false_property_control,
        counterexample=None,
        certificate={
            "kind": "blocked_dependency",
            "method": "z3_mixed_integer_linear_pwa_property_suite",
            "reason": "python package 'z3' is not importable",
        },
    )


def _maximize_with_z3(abstraction: MultiUnitPWAAbstraction) -> dict[str, Any]:
    """Maximize an additive PWA abstraction with Z3 integer segment selectors."""

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
        return {
            "solver_status": str(status),
            "certified_upper_bound": None,
            "witness_inputs": None,
            "selected_segments": None,
            "constraint_count": constraint_count,
            "solve_time_s": solve_time_s,
        }

    model = optimizer.model()
    witness_inputs = tuple(_z3_float(model.eval(x, model_completion=True)) for x in xs)
    selected_segments = tuple(
        next(
            segment_index
            for segment_index, flag in enumerate(flags)
            if _z3_float(model.eval(flag, model_completion=True)) > 0.5
        )
        for flags in selected_flag_groups
    )
    return {
        "solver_status": "optimal",
        "certified_upper_bound": _z3_float(objective.value()),
        "witness_inputs": witness_inputs,
        "selected_segments": selected_segments,
        "constraint_count": constraint_count,
        "solve_time_s": solve_time_s,
    }


def solve_property(spec: PropertySpec, solver_name: str | None = None) -> PropertyResult:
    """Solve one bounded property and apply its declared error budget."""

    abstraction = build_abstraction_for_property(spec)
    selected_solver = detect_solver() if solver_name is None else solver_name
    if selected_solver != "z3":
        return _blocked_result(spec, abstraction)

    solved = _maximize_with_z3(abstraction)
    solver_status = str(solved["solver_status"])
    certified_upper = solved["certified_upper_bound"]
    if solver_status != "optimal" or certified_upper is None:  # pragma: no cover - solver failure path.
        return PropertyResult(
            property_id=spec.property_id,
            suite_role=spec.suite_role,
            expected_outcome=spec.expected_outcome,
            statement=spec.statement(),
            threshold=spec.threshold,
            approximation_error_budget=spec.approximation_error_budget,
            solver_available=True,
            solver_name="z3",
            solver_status=solver_status,
            property_status="blocked_solver_status",
            property_holds=None,
            certified_upper_bound=None,
            budgeted_upper_bound=None,
            objective_bound=None,
            witness_inputs=None,
            selected_segments=None,
            binary_variable_count=abstraction.binary_variable_count,
            constraint_count=int(solved["constraint_count"]),
            pwa_piece_count=abstraction.pwa_piece_count,
            input_dimension=abstraction.input_dimension,
            solve_time_s=float(solved["solve_time_s"]),
            is_false_property_control=spec.is_false_property_control,
            counterexample=None,
            certificate={
                "kind": "solver_failure",
                "method": "z3_mixed_integer_linear_pwa_property_suite",
                "status": solver_status,
            },
        )

    budgeted_upper = float(certified_upper) + spec.approximation_error_budget
    witness_inputs = solved["witness_inputs"]
    selected_segments = solved["selected_segments"]
    counterexample = None

    if float(certified_upper) > spec.threshold + 1e-9:
        property_status = "counterexample"
        property_holds: bool | None = False
        counterexample = {
            "property_id": spec.property_id,
            "inputs": list(witness_inputs),
            "selected_segments": list(selected_segments),
            "certified_upper_bound": float(certified_upper),
            "budgeted_upper_bound": budgeted_upper,
            "threshold": spec.threshold,
            "violation_margin": budgeted_upper - spec.threshold,
        }
        certificate = {
            "kind": "counterexample",
            "method": "z3_mixed_integer_linear_pwa_property_suite",
            "counterexample": counterexample,
        }
    elif budgeted_upper <= spec.threshold + 1e-9:
        property_status = "verified"
        property_holds = True
        certificate = {
            "kind": "certificate",
            "method": "z3_mixed_integer_linear_pwa_property_suite",
            "certified_upper_bound": float(certified_upper),
            "budgeted_upper_bound": budgeted_upper,
            "threshold": spec.threshold,
            "approximation_error_budget": spec.approximation_error_budget,
            "witness_maximizer_inputs": list(witness_inputs),
            "selected_segments": list(selected_segments),
        }
    else:
        property_status = "unproved_approximation_budget"
        property_holds = None
        certificate = {
            "kind": "budget_sensitive_noncertificate",
            "method": "z3_mixed_integer_linear_pwa_property_suite",
            "certified_upper_bound": float(certified_upper),
            "budgeted_upper_bound": budgeted_upper,
            "threshold": spec.threshold,
            "approximation_error_budget": spec.approximation_error_budget,
            "reason": "declared approximation budget consumes the solver margin",
        }

    return PropertyResult(
        property_id=spec.property_id,
        suite_role=spec.suite_role,
        expected_outcome=spec.expected_outcome,
        statement=spec.statement(),
        threshold=spec.threshold,
        approximation_error_budget=spec.approximation_error_budget,
        solver_available=True,
        solver_name="z3",
        solver_status="optimal",
        property_status=property_status,
        property_holds=property_holds,
        certified_upper_bound=float(certified_upper),
        budgeted_upper_bound=budgeted_upper,
        objective_bound=float(certified_upper),
        witness_inputs=witness_inputs,
        selected_segments=selected_segments,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=int(solved["constraint_count"]),
        pwa_piece_count=abstraction.pwa_piece_count,
        input_dimension=abstraction.input_dimension,
        solve_time_s=float(solved["solve_time_s"]),
        is_false_property_control=spec.is_false_property_control,
        counterexample=counterexample,
        certificate=certificate,
    )


def solve_property_suite(
    specs: Sequence[PropertySpec],
    solver_name: str | None = None,
) -> tuple[PropertyResult, ...]:
    """Solve every property in order and return per-property telemetry."""

    return tuple(solve_property(spec, solver_name=solver_name) for spec in specs)


def _false_property_controls_passed(results: Sequence[PropertyResult]) -> bool:
    """Return true only when every false control has a counterexample."""

    controls = [result for result in results if result.is_false_property_control]
    return bool(controls) and all(
        result.property_status == "counterexample"
        and result.property_holds is False
        and result.counterexample is not None
        for result in controls
    )


def _max_scale_reached(results: Sequence[PropertyResult]) -> dict[str, Any]:
    """Return the largest solved property by selector count, then constraints."""

    if not results:
        return {}
    largest = max(results, key=lambda row: (row.binary_variable_count, row.constraint_count))
    return {
        "property_id": largest.property_id,
        "input_dimension": largest.input_dimension,
        "pwa_piece_count": largest.pwa_piece_count,
        "binary_variable_count": largest.binary_variable_count,
        "constraint_count": largest.constraint_count,
        "solver_status": largest.solver_status,
    }


def _scale_blocker(results: Sequence[PropertyResult], false_controls_passed: bool) -> str | None:
    """Return a suite-level blocker or ``None`` for a clean bounded run."""

    if any(result.solver_status == "blocked_solver_dependency" for result in results):
        return "blocked_kan_pwa_milp_solver_unavailable"
    non_optimal = [result.solver_status for result in results if result.solver_status != "optimal"]
    if non_optimal:
        return f"blocked_solver_status_{non_optimal[0]}"
    if not false_controls_passed:
        return "false_property_control_not_counterexampled"
    expected_true_missing = [
        result.property_id
        for result in results
        if result.expected_outcome == "verified" and result.property_status != "verified"
    ]
    if expected_true_missing:
        return f"expected_true_not_proved:{expected_true_missing[0]}"
    sensitive_missing = [
        result.property_id
        for result in results
        if result.expected_outcome == "unproved_approximation_budget"
        and result.property_status != "unproved_approximation_budget"
    ]
    if sensitive_missing:
        return f"approximation_budget_control_failed:{sensitive_missing[0]}"
    return None


def _checksum_payload(results: Sequence[PropertyResult]) -> str:
    """Hash deterministic suite inputs and outputs, excluding wall-clock time."""

    property_rows: list[dict[str, Any]] = []
    for result in results:
        row = result.as_serializable()
        row["solve_time_s"] = "excluded"
        property_rows.append(row)
    payload = {
        "artifact": ARTIFACT_NAME,
        "property_specs": [spec.__dict__ for spec in build_property_specs()],
        "property_rows": property_rows,
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
        "source_artifacts": [EXP5091_SOURCE_ARTIFACT, EXP5080_SOURCE_ARTIFACT],
        "spec_refs": SPEC_REFS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(solver_name: str | None = None) -> dict[str, Any]:
    """Build the Exp 5098 deliverable payload."""

    start = time.perf_counter()
    specs = build_property_specs()
    results = solve_property_suite(specs, solver_name=solver_name)
    false_controls_passed = _false_property_controls_passed(results)
    scale_blocker = _scale_blocker(results, false_controls_passed)
    flagged_adversarial = any(
        result.is_false_property_control and result.property_holds is True for result in results
    )
    properties_proved = [
        result.property_id
        for result in results
        if result.property_status == "verified" and not result.is_false_property_control
    ]
    honest_verdict = SUCCESS_VERDICT if scale_blocker is None and not flagged_adversarial else BLOCKED_VERDICT

    artifact = {
        "schema": "carnot.kan_pwa_milp_scale.v2",
        "experiment": 5098,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "duration_s": round(time.perf_counter() - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "property_suite": [result.as_serializable() for result in results],
        "properties_proved": properties_proved,
        "false_property_controls_passed": false_controls_passed,
        "solver_statuses": {result.property_id: result.solver_status for result in results},
        "binary_variable_counts": {
            result.property_id: result.binary_variable_count for result in results
        },
        "constraint_counts": {result.property_id: result.constraint_count for result in results},
        "solve_times_s": {result.property_id: result.solve_time_s for result in results},
        "approximation_error_budget": {
            result.property_id: result.approximation_error_budget for result in results
        },
        "counterexamples": {
            result.property_id: result.counterexample
            for result in results
            if result.counterexample is not None
        },
        "max_scale_reached": _max_scale_reached(results),
        "scale_blocker": scale_blocker,
        "flagged_adversarial": flagged_adversarial,
        "baseline_reproduced": any(
            result.property_id == "exp5091_baseline_two_unit_true"
            and result.property_status == "verified"
            for result in results
        ),
        "field_principles": FIELD_PRINCIPLES,
        "source_artifacts": [EXP5091_SOURCE_ARTIFACT, EXP5080_SOURCE_ARTIFACT],
        "methodology_note": (
            "Exp 5098 submits small KAEM PWA abstractions to an exact CPU solver "
            "and reports only bounded property-suite outcomes. The near-margin "
            "row is intentionally left unproved when its declared error budget "
            "consumes the certification margin."
        ),
        "kan_component_path": KAN_COMPONENT_PATH,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "tests_run": [
            ".venv/bin/python -m pytest tests/python/test_experiment_5098_kan_pwa_milp_scale_v2.py -q --no-cov"
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(results)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5098 artifact drifts from its schema boundary."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(
        verdict in {SUCCESS_VERDICT, BLOCKED_VERDICT},
        "honest_verdict must be the success or blocked Exp 5098 terminal state",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be exact_milp_solver_cpu",
    )
    _require("live_llm" not in artifact["inference_substrate"], "must not claim live LLM inference")
    _require(isinstance(artifact["duration_s"], float), "duration_s must be a float")
    _require(artifact["duration_s"] >= 0.0, "duration_s cannot be negative")
    _require(isinstance(artifact["property_suite"], list), "property_suite must be a list")
    _require(len(artifact["property_suite"]) >= 3, "property_suite must contain at least three rows")
    property_ids = {row["property_id"] for row in artifact["property_suite"]}
    _require(
        {
            "exp5091_baseline_two_unit_true",
            "adversarial_false_tight_bound",
            "approximation_budget_sensitive_margin",
        }.issubset(property_ids),
        "property suite must include baseline, false control, and approximation-sensitive rows",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]),
        "field_principles must cover every required user field",
    )
    for field in (
        "solver_statuses",
        "binary_variable_counts",
        "constraint_counts",
        "solve_times_s",
        "approximation_error_budget",
    ):
        _require(set(artifact[field]) == property_ids, f"{field} must cover every property")
    _require(isinstance(artifact["flagged_adversarial"], bool), "flagged_adversarial must be bool")

    if artifact["scale_blocker"] is None:
        _require(artifact["honest_verdict"] == SUCCESS_VERDICT, "clean suite must use success verdict")
        _require(
            artifact["false_property_controls_passed"] is True,
            "clean suite requires false-property controls to pass",
        )
        _require(
            artifact["properties_proved"]
            == ["exp5091_baseline_two_unit_true", "three_unit_composition_true"],
            "clean suite must prove only the expected true non-control properties",
        )
        _require(
            "adversarial_false_tight_bound" in artifact["counterexamples"],
            "clean suite must include false-control counterexample",
        )
        _require(
            artifact["solver_statuses"]["approximation_budget_sensitive_margin"] == "optimal",
            "approximation-sensitive row must still be solved",
        )
        sensitive = next(
            row
            for row in artifact["property_suite"]
            if row["property_id"] == "approximation_budget_sensitive_margin"
        )
        _require(
            sensitive["property_status"] == "unproved_approximation_budget",
            "near-margin property must remain unproved due to budget",
        )
        _require(artifact["flagged_adversarial"] is False, "clean suite cannot be flagged")
    else:
        _require(artifact["honest_verdict"] == BLOCKED_VERDICT, "blocked suite must use blocked verdict")


def write_outputs(*, artifact_path: str | Path, solver_name: str | None = None) -> dict[str, Any]:
    """Write the Exp 5098 JSON artifact and return the validated payload."""

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

    root = Path(os.environ.get("CARNOT_EXP5098_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(artifact_path=root / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests.
    raise SystemExit(main())
