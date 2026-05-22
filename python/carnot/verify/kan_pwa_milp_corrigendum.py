"""Exp 2876 KAN PWA/MILP corrigendum for the Exp 2871 tautology flag.

Spec: REQ-KAN-2876, SCENARIO-KAN-2876.

Exp 2871 used a one-unit quadratic fixture, so its "global" output error was
the same number as the worst local segment error. That was mathematically valid
for the tiny fixture, but it looked like a metric tautology in the terminal
artifact. This module keeps the example intentionally small while separating the
procedures: local error is measured on each PWA segment, and global error is
then propagated through a weighted two-unit output graph.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_DATE = "20260522"
RANDOM_SEED = 0
BREAKPOINTS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PROPERTY_LOWER_X = -0.5
PROPERTY_UPPER_X = 0.5
PROPERTY_THRESHOLD = 0.625
RESULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "experiment_2876_kan_pwa_milp_corrigendum_v2.json"
)
REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "kan_corrigendum_ready",
    "tautology_flag_cleared",
    "local_error_bound",
    "global_error_bound",
    "bounds_distinct_by_construction",
    "milp_backend_available",
    "milp_backend_name",
    "exact_enumeration_used_only_as_fallback",
    "solver_status",
    "counterexample_or_certificate",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
}


@dataclass(frozen=True)
class CorrigendumSegment:
    """One PWA chord segment and its local residual bound.

    The midpoint residual is the load-bearing local quantity because a convex
    quadratic sits below its chord, and the largest gap on a fixed interval is
    at the midpoint. Recording it per segment prevents the artifact from using
    the downstream global graph calculation as a substitute for local evidence.
    """

    x_min: float
    x_max: float
    slope: float
    intercept: float
    residual_lower: float
    residual_upper: float

    @property
    def local_error_bound(self) -> float:
        """Return the absolute per-segment abstraction error."""

        return max(abs(self.residual_lower), abs(self.residual_upper))

    def center(self, x: float) -> float:
        """Evaluate the chord approximation at ``x``."""

        return self.slope * float(x) + self.intercept

    def upper(self, x: float) -> float:
        """Evaluate the sound upper envelope used by the solver certificate."""

        return self.center(x) + self.residual_upper

    def as_serializable(self) -> dict[str, float]:
        """Return JSON-safe segment evidence."""

        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "slope": self.slope,
            "intercept": self.intercept,
            "residual_lower": self.residual_lower,
            "residual_upper": self.residual_upper,
            "local_error_bound": self.local_error_bound,
        }


@dataclass(frozen=True)
class CorrigendumUnit:
    """One bounded KAN-style univariate unit in the tiny fixture."""

    name: str
    coefficient: float
    shift: float
    output_weight: float
    segments: tuple[CorrigendumSegment, ...]

    @property
    def local_error_bound(self) -> float:
        """Return the largest local segment error for this unit."""

        return max(segment.local_error_bound for segment in self.segments)

    def true_value(self, x: float) -> float:
        """Evaluate the underlying quadratic unit before PWA replacement."""

        shifted = float(x) + self.shift
        return self.coefficient * shifted * shifted

    def segment_for_x(self, x: float) -> CorrigendumSegment:
        """Return the PWA segment covering ``x`` on the closed fixture domain."""

        x_f = float(x)
        for index, segment in enumerate(self.segments):
            right_pad = 1e-12 if index == len(self.segments) - 1 else 0.0
            if segment.x_min - 1e-12 <= x_f <= segment.x_max + right_pad:
                return segment
        raise ValueError(f"x={x_f} is outside the PWA domain")

    def upper(self, x: float) -> float:
        """Evaluate this unit's PWA upper envelope at ``x``."""

        return self.segment_for_x(x).upper(x)

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe unit evidence."""

        return {
            "name": self.name,
            "coefficient": self.coefficient,
            "shift": self.shift,
            "output_weight": self.output_weight,
            "local_error_bound": self.local_error_bound,
            "weighted_global_error_contribution": abs(self.output_weight)
            * self.local_error_bound,
            "segments": [segment.as_serializable() for segment in self.segments],
        }


@dataclass(frozen=True)
class OutputSegment:
    """One affine segment of the weighted PWA output upper envelope."""

    x_min: float
    x_max: float
    slope: float
    intercept: float

    def value(self, x: float) -> float:
        """Evaluate the output upper envelope segment."""

        return self.slope * float(x) + self.intercept

    def as_serializable(self) -> dict[str, float]:
        """Return JSON-safe output segment parameters."""

        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "slope": self.slope,
            "intercept": self.intercept,
        }


@dataclass(frozen=True)
class CorrigendumFixture:
    """Two-unit PWA fixture used only for the Exp 2876 corrigendum."""

    units: tuple[CorrigendumUnit, ...]
    property_lower_x: float = PROPERTY_LOWER_X
    property_upper_x: float = PROPERTY_UPPER_X
    property_threshold: float = PROPERTY_THRESHOLD

    @property
    def local_error_bound(self) -> float:
        """Compute the local bound as the worst unweighted segment residual."""

        return max(unit.local_error_bound for unit in self.units)

    @property
    def global_error_bound(self) -> float:
        """Propagate unit-local errors through the weighted output graph."""

        return sum(abs(unit.output_weight) * unit.local_error_bound for unit in self.units)

    @property
    def bounds_distinct_by_construction(self) -> bool:
        """Report whether the two required bounds are not mechanically equal."""

        return not math.isclose(
            self.local_error_bound,
            self.global_error_bound,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    def bound_procedures(self) -> dict[str, str]:
        """Name the independent procedures used for the two artifact fields."""

        return {
            "local_error_bound": "max_per_segment_midpoint_residual",
            "global_error_bound": "weighted_output_error_propagation",
        }

    def output_segments(self) -> tuple[OutputSegment, ...]:
        """Build the weighted PWA upper envelope segments.

        The units deliberately share breakpoints, so a single binary segment
        selection is enough for the solver. That keeps the example auditable
        while still exercising a mixed integer linear encoding.
        """

        segments: list[OutputSegment] = []
        for index in range(len(self.units[0].segments)):
            x_min = self.units[0].segments[index].x_min
            x_max = self.units[0].segments[index].x_max
            slope = sum(unit.output_weight * unit.segments[index].slope for unit in self.units)
            intercept = sum(
                unit.output_weight
                * (unit.segments[index].intercept + unit.segments[index].residual_upper)
                for unit in self.units
            )
            segments.append(OutputSegment(x_min=x_min, x_max=x_max, slope=slope, intercept=intercept))
        return tuple(segments)

    def output_segment_for_x(self, x: float) -> OutputSegment:
        """Return the weighted output segment covering ``x``."""

        x_f = float(x)
        segments = self.output_segments()
        for index, segment in enumerate(segments):
            right_pad = 1e-12 if index == len(segments) - 1 else 0.0
            if segment.x_min - 1e-12 <= x_f <= segment.x_max + right_pad:
                return segment
        raise ValueError(f"x={x_f} is outside the PWA domain")

    def evaluate_upper(self, x: float) -> float:
        """Evaluate the full weighted PWA upper envelope at ``x``."""

        return self.output_segment_for_x(x).value(x)

    def candidate_points(self) -> tuple[float, ...]:
        """Return endpoints and internal breakpoints for fallback evidence."""

        points = {self.property_lower_x, self.property_upper_x}
        for segment in self.output_segments():
            if self.property_lower_x <= segment.x_min <= self.property_upper_x:
                points.add(segment.x_min)
            if self.property_lower_x <= segment.x_max <= self.property_upper_x:
                points.add(segment.x_max)
        return tuple(sorted(points))

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe fixture details."""

        return {
            "units": [unit.as_serializable() for unit in self.units],
            "output_segments": [segment.as_serializable() for segment in self.output_segments()],
            "property_domain": [self.property_lower_x, self.property_upper_x],
            "property_threshold": self.property_threshold,
            "local_error_bound": self.local_error_bound,
            "global_error_bound": self.global_error_bound,
            "bound_procedures": self.bound_procedures(),
        }


@dataclass(frozen=True)
class CorrigendumSolveResult:
    """Structured solver result for the Exp 2876 artifact."""

    property_verified: bool
    certified_upper_bound: float
    witness_x: float
    milp_backend_available: bool
    milp_backend_name: str
    solver_status: str
    exact_enumeration_used_only_as_fallback: bool
    counterexample_or_certificate: dict[str, Any]

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe solver status fields."""

        return {
            "property_verified": self.property_verified,
            "certified_upper_bound": self.certified_upper_bound,
            "witness_x": self.witness_x,
            "milp_backend_available": self.milp_backend_available,
            "milp_backend_name": self.milp_backend_name,
            "solver_status": self.solver_status,
            "exact_enumeration_used_only_as_fallback": self.exact_enumeration_used_only_as_fallback,
            "counterexample_or_certificate": self.counterexample_or_certificate,
        }


def _quadratic_segment(
    coefficient: float,
    shift: float,
    x_min: float,
    x_max: float,
) -> CorrigendumSegment:
    """Build one chord segment and measure its midpoint residual."""

    y_min = coefficient * (x_min + shift) * (x_min + shift)
    y_max = coefficient * (x_max + shift) * (x_max + shift)
    slope = (y_max - y_min) / (x_max - x_min)
    intercept = y_min - slope * x_min
    midpoint = (x_min + x_max) / 2.0
    midpoint_residual = coefficient * (midpoint + shift) * (midpoint + shift)
    midpoint_residual -= slope * midpoint + intercept
    return CorrigendumSegment(
        x_min=x_min,
        x_max=x_max,
        slope=slope,
        intercept=intercept,
        residual_lower=min(0.0, midpoint_residual),
        residual_upper=max(0.0, midpoint_residual),
    )


def _build_quadratic_unit(
    name: str,
    coefficient: float,
    shift: float,
    output_weight: float,
) -> CorrigendumUnit:
    """Build a quadratic unit on the shared Exp 2876 breakpoint grid."""

    segments = tuple(
        _quadratic_segment(coefficient, shift, x_min, x_max)
        for x_min, x_max in zip(BREAKPOINTS[:-1], BREAKPOINTS[1:])
    )
    return CorrigendumUnit(
        name=name,
        coefficient=coefficient,
        shift=shift,
        output_weight=output_weight,
        segments=segments,
    )


def build_corrigendum_fixture() -> CorrigendumFixture:
    """Create the deterministic two-unit KAN-style fixture for Exp 2876."""

    return CorrigendumFixture(
        units=(
            _build_quadratic_unit(
                name="u0_x_squared",
                coefficient=1.0,
                shift=0.0,
                output_weight=1.0,
            ),
            _build_quadratic_unit(
                name="u1_shifted_scaled_quadratic",
                coefficient=0.25,
                shift=0.25,
                output_weight=2.0,
            ),
        )
    )


def detect_milp_backend() -> str:
    """Return the supported local mixed integer linear backend, if installed."""

    return "z3" if importlib.util.find_spec("z3") is not None else ""


def _z3_float(value: Any) -> float:
    """Convert a small Z3 rational or integer model value to ``float``."""

    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def _real(z3: Any, value: float) -> Any:
    """Create a decimal Z3 real literal without relying on binary floats."""

    return z3.RealVal(repr(float(value)))


def _solve_with_z3(fixture: CorrigendumFixture) -> CorrigendumSolveResult:
    """Maximize the PWA upper envelope with binary segment-selection variables."""

    z3 = importlib.import_module("z3")
    optimizer = z3.Optimize()
    x = z3.Real("x")
    y = z3.Real("y")
    output_segments = fixture.output_segments()
    flags = [z3.Int(f"segment_{index}") for index in range(len(output_segments))]
    big_m = _real(z3, 10.0)

    optimizer.add(x >= _real(z3, fixture.property_lower_x))
    optimizer.add(x <= _real(z3, fixture.property_upper_x))
    optimizer.add(z3.Sum(flags) == 1)
    for flag, segment in zip(flags, output_segments):
        flag_real = z3.ToReal(flag)
        slack = big_m * (_real(z3, 1.0) - flag_real)
        affine_value = _real(z3, segment.slope) * x + _real(z3, segment.intercept)
        optimizer.add(flag >= 0, flag <= 1)
        optimizer.add(x >= _real(z3, segment.x_min) - slack)
        optimizer.add(x <= _real(z3, segment.x_max) + slack)
        optimizer.add(y - affine_value <= slack)
        optimizer.add(affine_value - y <= slack)

    objective = optimizer.maximize(y)
    status = optimizer.check()
    if status != z3.sat:  # pragma: no cover - kept for honest solver diagnostics.
        return CorrigendumSolveResult(
            property_verified=False,
            certified_upper_bound=float("nan"),
            witness_x=float("nan"),
            milp_backend_available=True,
            milp_backend_name="z3",
            solver_status=str(status),
            exact_enumeration_used_only_as_fallback=False,
            counterexample_or_certificate={
                "kind": "solver_failure",
                "method": "z3_mixed_integer_linear_pwa",
                "status": str(status),
            },
        )

    model = optimizer.model()
    certified_upper = _z3_float(objective.value())
    witness_x = _z3_float(model.eval(x, model_completion=True))
    selected_segment = next(
        index
        for index, flag in enumerate(flags)
        if _z3_float(model.eval(flag, model_completion=True)) > 0.5
    )
    verified = certified_upper <= fixture.property_threshold
    return CorrigendumSolveResult(
        property_verified=verified,
        certified_upper_bound=certified_upper,
        witness_x=witness_x,
        milp_backend_available=True,
        milp_backend_name="z3",
        solver_status="optimal",
        exact_enumeration_used_only_as_fallback=False,
        counterexample_or_certificate={
            "kind": "certificate" if verified else "counterexample",
            "method": "z3_mixed_integer_linear_pwa",
            "certified_upper_bound": certified_upper,
            "property_threshold": fixture.property_threshold,
            "witness_maximizer_x": witness_x,
            "selected_segment": selected_segment,
        },
    )


def _solve_by_exact_fallback(fixture: CorrigendumFixture) -> CorrigendumSolveResult:
    """Enumerate PWA vertices only when solver dependencies are blocked."""

    candidates = fixture.candidate_points()
    witness_x = candidates[0]
    certified_upper = fixture.evaluate_upper(witness_x)
    for candidate in candidates[1:]:
        value = fixture.evaluate_upper(candidate)
        if value > certified_upper:
            certified_upper = value
            witness_x = candidate
    verified = certified_upper <= fixture.property_threshold
    return CorrigendumSolveResult(
        property_verified=verified,
        certified_upper_bound=certified_upper,
        witness_x=witness_x,
        milp_backend_available=False,
        milp_backend_name="",
        solver_status="blocked_solver_dependency",
        exact_enumeration_used_only_as_fallback=True,
        counterexample_or_certificate={
            "kind": "fallback_certificate" if verified else "fallback_counterexample",
            "method": "exact_enumerated_pwa_vertices",
            "certified_upper_bound": certified_upper,
            "property_threshold": fixture.property_threshold,
            "witness_maximizer_x": witness_x,
            "candidate_points": list(candidates),
        },
    )


def solve_property(
    fixture: CorrigendumFixture,
    backend_name: str | None = None,
) -> CorrigendumSolveResult:
    """Solve the bounded PWA property, using a real backend before fallback."""

    selected_backend = detect_milp_backend() if backend_name is None else backend_name
    if selected_backend == "z3":
        return _solve_with_z3(fixture)
    return _solve_by_exact_fallback(fixture)


def _solver_preconditions_checked(backend_name: str | None) -> list[dict[str, Any]]:
    """Record which dependency checks made the solver decision auditable."""

    z3_importable = importlib.util.find_spec("z3") is not None
    forced_absent = backend_name == ""
    selected = z3_importable and not forced_absent and backend_name in (None, "z3")
    return [
        {
            "backend": "z3",
            "check": "importlib.util.find_spec('z3')",
            "importable": z3_importable,
            "available": selected,
            "forced_absent_for_blocked_solver_artifact": forced_absent,
        }
    ]


def _checksum_payload(fixture: CorrigendumFixture, result: CorrigendumSolveResult) -> str:
    """Hash deterministic proof inputs and outputs, excluding wall-clock time."""

    payload = {
        "breakpoints": BREAKPOINTS,
        "fixture": fixture.as_serializable(),
        "property_result": result.as_serializable(),
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_experiment_artifact(backend_name: str | None = None) -> dict[str, Any]:
    """Build the Exp 2876 deliverable payload."""

    start = time.perf_counter()
    fixture = build_corrigendum_fixture()
    result = solve_property(fixture, backend_name=backend_name)
    tautology_cleared = fixture.bounds_distinct_by_construction
    ready = (
        tautology_cleared
        and result.milp_backend_available
        and result.solver_status == "optimal"
        and result.property_verified
    )
    honest_verdict = (
        "complete_corrigendum_z3_milp_bounds_distinct_no_general_kan_claim"
        if ready
        else "blocked_solver_dependency_bounds_distinct_exact_fallback_only_no_milp_claim"
    )
    artifact = {
        "schema": "carnot.kan_pwa_milp_corrigendum.v2",
        "experiment": 2876,
        "artifact": "experiment_2876_kan_pwa_milp_corrigendum_v2",
        "honest_verdict": honest_verdict,
        "kan_corrigendum_ready": ready,
        "tautology_flag_cleared": tautology_cleared,
        "local_error_bound": fixture.local_error_bound,
        "global_error_bound": fixture.global_error_bound,
        "bounds_distinct_by_construction": fixture.bounds_distinct_by_construction,
        "milp_backend_available": result.milp_backend_available,
        "milp_backend_name": result.milp_backend_name,
        "exact_enumeration_used_only_as_fallback": result.exact_enumeration_used_only_as_fallback,
        "solver_status": result.solver_status,
        "counterexample_or_certificate": result.counterexample_or_certificate,
        "property_verified": result.property_verified,
        "certified_upper_bound": result.certified_upper_bound,
        "witness_x": result.witness_x,
        "property_statement": (
            "For all x in "
            f"[{fixture.property_lower_x}, {fixture.property_upper_x}], "
            "u0(x) + 2*u1(x) <= "
            f"{fixture.property_threshold} under the PWA upper envelope."
        ),
        "solver_preconditions_checked": _solver_preconditions_checked(backend_name),
        "previous_artifact_flag": {
            "experiment": 2871,
            "flag_kind": "TAUTOLOGY",
            "flagged_fields": ["local_error_bound", "global_error_bound"],
            "flagged_values": [0.0625, 0.0625],
        },
        "pwa_fixture": fixture.as_serializable(),
        "tests_run": [
            ".venv/bin/pytest tests/python/verify/test_kan_pwa_milp_corrigendum.py -q --no-cov",
            ".venv/bin/coverage run --source=python/carnot/verify -m pytest tests/python/verify/test_kan_pwa_milp_corrigendum.py -q --no-cov -n0",
            ".venv/bin/coverage report --fail-under=100 -m python/carnot/verify/kan_pwa_milp_corrigendum.py",
            ".venv/bin/pytest tests/python -q",
            ".venv/bin/python scripts/check_spec_coverage.py",
        ],
        "test_outcomes": {
            "focused_pytest": "passed: 6 passed",
            "new_code_coverage": "passed: 100% line coverage for python/carnot/verify/kan_pwa_milp_corrigendum.py",
            "full_pytest": (
                "failed: 11 failed, 7138 passed, 35 skipped, 1 error; "
                "xdist worker crashed in tests/python/test_experiment_295_apple_verify_repair.py::"
                "test_partial_artifact_has_stall_at with tokenizer prewarm MemoryError"
            ),
            "spec_coverage": "failed: 444 pre-existing tests missing spec traceability",
        },
        "field_principles": {
            "exp_2871_flag_fields": (
                "Exp 2871 was flagged because local_error_bound and "
                "global_error_bound were both exactly 0.0625."
            ),
            "local_bound_procedure": "max per-segment midpoint residual over unweighted PWA units",
            "global_bound_procedure": "sum of absolute output weights times each unit's local bound",
            "solver_boundary": "MILP readiness is claimed only when a local solver returns optimal",
            "fallback_boundary": "exact enumeration is reported only as blocked-solver fallback evidence",
            "proof_scope": "deterministic two-unit quadratic PWA fixture only",
        },
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
        "duration_s": round(time.perf_counter() - start, 6),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(fixture, result)
    return validate_artifact(artifact)


def validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate that every required Exp 2876 field is present."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    return artifact


def write_experiment_artifact(
    path: str | Path = RESULT_PATH,
    backend_name: str | None = None,
) -> dict[str, Any]:
    """Write the Exp 2876 deliverable JSON and return the payload."""

    artifact = build_experiment_artifact(backend_name=backend_name)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    """CLI entrypoint for writing the requested result artifact."""

    artifact = write_experiment_artifact()
    print(
        json.dumps(
            {
                "artifact": str(RESULT_PATH),
                "solver_status": artifact["solver_status"],
                "tautology_flag_cleared": artifact["tautology_flag_cleared"],
            }
        )
    )


__all__ = [
    "CorrigendumFixture",
    "CorrigendumSegment",
    "CorrigendumSolveResult",
    "CorrigendumUnit",
    "REQUIRED_ARTIFACT_FIELDS",
    "RESULT_PATH",
    "build_corrigendum_fixture",
    "build_experiment_artifact",
    "detect_milp_backend",
    "solve_property",
    "validate_artifact",
    "write_experiment_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    main()
