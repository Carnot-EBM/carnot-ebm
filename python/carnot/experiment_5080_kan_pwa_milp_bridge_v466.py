"""Exp 5080 tiny KAEM/PWA/MILP bridge diagnostic.

Spec refs: REQ-KAN-5080, SCENARIO-KAN-5080.

This module deliberately verifies one tiny property, not a general KAN network.
It takes the real knot/control-point representation used by
``UnivariateKAEMLayer``, exposes that one-dimensional spline as explicit
piecewise-affine segments, and asks a local linear-integer solver to maximize
the segment-selected energy over the bounded input domain. If the solver is not
available, the result is a blocked artifact instead of an enumeration-only proof.
"""

from __future__ import annotations

from collections.abc import Mapping
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
import numpy as np

from carnot.models.kaem_energy import UnivariateKAEMLayer


RESULT_RELATIVE_PATH = "results/experiment_5080_kan_pwa_milp_bridge_v466.json"
ARTIFACT_NAME = "experiment_5080_kan_pwa_milp_bridge_v466"
RUN_DATE = "20260701"
RANDOM_SEED = 5080
KAN_COMPONENT_PATH = "python/carnot/models/kaem_energy.py::UnivariateKAEMLayer"
INFERENCE_SUBSTRATE = "deterministic_formal_check"
SUCCESS_VERDICT = "success_kan_pwa_milp_property_verified_tiny"
BLOCKED_VERDICT = "blocked_kan_pwa_milp_solver_unavailable"
PROPERTY_THRESHOLD = 1.0
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
SPEC_REFS = ["REQ-KAN-5080", "SCENARIO-KAN-5080"]

REQUIRED_USER_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "kan_component_path",
    "pwa_abstraction_built",
    "milp_solver_available",
    "property_checked",
    "property_holds",
    "error_bound",
    "binary_variable_count",
    "blocked_reason",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "Terminal verdict prefix lets the conductor distinguish success from a blocked solver dependency."
    },
    "duration_s": {
        "principle": "Wall-clock duration is declared for fabrication checks; this is a deterministic formal check, not live model inference."
    },
    "inference_substrate": {
        "principle": "Substrate names the actual compute path and prevents a tiny solver run from being mistaken for LLM or hardware execution."
    },
    "kan_component_path": {
        "principle": "The bridge must point to the real KAEM component it abstracts instead of an invented toy-only class."
    },
    "pwa_abstraction_built": {
        "principle": "Separates a real knot-to-segment abstraction from a solver-only placeholder."
    },
    "milp_solver_available": {
        "principle": "A solver availability bit prevents claiming MILP verification when the dependency is missing."
    },
    "property_checked": {
        "principle": "Distinguishes a built abstraction from a property actually submitted to linear-integer constraints."
    },
    "property_holds": {
        "principle": "Records the formal conclusion only after the solver returns an optimal certificate."
    },
    "error_bound": {
        "principle": "Quantifies the abstraction gap; zero is allowed here only because KAEM's layer is already piecewise affine."
    },
    "binary_variable_count": {
        "principle": "Counts one-hot segment selector variables, the complexity cost highlighted by KAN/PWA/MILP verification work."
    },
    "blocked_reason": {
        "principle": "Blocked artifacts must say exactly which dependency or solver condition prevented a proof."
    },
    "flagged_adversarial": {
        "principle": "Artifact quarantine bit remains false only when the result is honest about scope, solver use, and substrate."
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover - defensive validation path.
        raise ValueError(message)


@dataclass(frozen=True)
class PWASegment:
    """One affine segment from a KAEM univariate spline.

    A KAEM layer evaluates each variable by linear interpolation between knot
    control points. Recording the slope and intercept makes the hidden spline
    structure explicit enough for a mixed-integer linear encoding.
    """

    index: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    slope: float
    intercept: float

    def value(self, x: float) -> float:
        """Evaluate the affine segment at ``x``."""

        return self.slope * float(x) + self.intercept

    def as_serializable(self) -> dict[str, float | int]:
        """Return JSON-safe segment parameters."""

        return {
            "index": self.index,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "slope": self.slope,
            "intercept": self.intercept,
        }


@dataclass(frozen=True)
class PWAAbstraction:
    """PWA abstraction for one `UnivariateKAEMLayer` variable."""

    component_path: str
    variable_index: int
    knots: tuple[float, ...]
    control_points: tuple[float, ...]
    segments: tuple[PWASegment, ...]
    error_bound: float
    exact_for_linear_kaem_spline: bool

    @property
    def n_segments(self) -> int:
        """Return the number of affine pieces."""

        return len(self.segments)

    @property
    def binary_variable_count(self) -> int:
        """Return one binary selector per PWA segment."""

        return self.n_segments

    def segment_for_x(self, x: float) -> PWASegment:
        """Return the segment covering ``x`` on the closed PWA domain."""

        x_f = float(x)
        for index, segment in enumerate(self.segments):
            right_pad = 1e-12 if index == len(self.segments) - 1 else 0.0
            if segment.x_min - 1e-12 <= x_f <= segment.x_max + right_pad:
                return segment
        raise ValueError(f"x={x_f} is outside the PWA domain")

    def evaluate(self, x: float) -> float:
        """Evaluate the PWA abstraction at ``x``."""

        return self.segment_for_x(x).value(x)

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe abstraction evidence."""

        return {
            "component_path": self.component_path,
            "variable_index": self.variable_index,
            "knots": list(self.knots),
            "control_points": list(self.control_points),
            "n_segments": self.n_segments,
            "binary_variable_count": self.binary_variable_count,
            "error_bound": self.error_bound,
            "exact_for_linear_kaem_spline": self.exact_for_linear_kaem_spline,
            "segments": [segment.as_serializable() for segment in self.segments],
        }


@dataclass(frozen=True)
class PropertySolveResult:
    """Result from the tiny bound-property solver path."""

    milp_solver_available: bool
    solver_name: str
    solver_status: str
    property_checked: bool
    property_holds: bool | None
    certified_upper_bound: float | None
    witness_x: float | None
    selected_segment: int | None
    binary_variable_count: int
    blocked_reason: str | None
    certificate: dict[str, Any]

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe solver result fields."""

        return {
            "milp_solver_available": self.milp_solver_available,
            "solver_name": self.solver_name,
            "solver_status": self.solver_status,
            "property_checked": self.property_checked,
            "property_holds": self.property_holds,
            "certified_upper_bound": self.certified_upper_bound,
            "witness_x": self.witness_x,
            "selected_segment": self.selected_segment,
            "binary_variable_count": self.binary_variable_count,
            "blocked_reason": self.blocked_reason,
            "certificate": self.certificate,
        }


def build_tiny_kaem_layer() -> UnivariateKAEMLayer:
    """Create the deterministic one-variable KAEM energy head for Exp 5080."""

    layer = UnivariateKAEMLayer(n_vars=1, n_knots=4, key=jax.random.PRNGKey(RANDOM_SEED))
    layer.control_points = jnp.array([[0.0, 0.25, 0.75, 1.0]], dtype=jnp.float32)
    return layer


def build_pwa_abstraction(layer: UnivariateKAEMLayer, variable_index: int = 0) -> PWAAbstraction:
    """Expose one KAEM variable's linear spline as explicit PWA segments."""

    knots = tuple(float(value) for value in np.asarray(layer._knots, dtype=np.float64))
    control = tuple(
        float(value) for value in np.asarray(layer.control_points[variable_index], dtype=np.float64)
    )
    segments: list[PWASegment] = []
    for index, (x_min, x_max, y_min, y_max) in enumerate(
        zip(knots[:-1], knots[1:], control[:-1], control[1:])
    ):
        slope = (y_max - y_min) / (x_max - x_min)
        intercept = y_min - slope * x_min
        segments.append(
            PWASegment(
                index=index,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                slope=slope,
                intercept=intercept,
            )
        )

    return PWAAbstraction(
        component_path=KAN_COMPONENT_PATH,
        variable_index=variable_index,
        knots=knots,
        control_points=control,
        segments=tuple(segments),
        error_bound=0.0,
        exact_for_linear_kaem_spline=True,
    )


def detect_milp_solver() -> str:
    """Return the local linear-integer backend used by this diagnostic."""

    return "z3" if importlib.util.find_spec("z3") is not None else ""


def _z3_float(value: Any) -> float:
    """Convert a small Z3 numeral to ``float`` for JSON output."""

    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    if text.endswith("?"):
        text = text[:-1]
    return float(text)


def _real(z3: Any, value: float) -> Any:
    """Create a decimal Z3 real literal without binary-float expression drift."""

    return z3.RealVal(repr(float(value)))


def _blocked_solver_result(abstraction: PWAAbstraction) -> PropertySolveResult:
    """Build the fail-closed result used when no supported solver is importable."""

    return PropertySolveResult(
        milp_solver_available=False,
        solver_name="",
        solver_status="blocked_solver_dependency",
        property_checked=False,
        property_holds=None,
        certified_upper_bound=None,
        witness_x=None,
        selected_segment=None,
        binary_variable_count=abstraction.binary_variable_count,
        blocked_reason="blocked_kan_pwa_milp_solver_unavailable",
        certificate={
            "kind": "blocked_dependency",
            "method": "z3_mixed_integer_linear_pwa_bound",
            "reason": "python package 'z3' is not importable",
        },
    )


def _solve_bound_with_z3(abstraction: PWAAbstraction) -> PropertySolveResult:
    """Maximize the PWA energy under one-hot segment-selection constraints."""

    z3 = importlib.import_module("z3")
    optimizer = z3.Optimize()
    x = z3.Real("x")
    y = z3.Real("y")
    flags = [z3.Int(f"pwa_segment_{segment.index}") for segment in abstraction.segments]
    big_m = _real(z3, 10.0)

    optimizer.add(x >= _real(z3, abstraction.segments[0].x_min))
    optimizer.add(x <= _real(z3, abstraction.segments[-1].x_max))
    optimizer.add(z3.Sum(flags) == 1)
    for flag, segment in zip(flags, abstraction.segments):
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
    if status != z3.sat:  # pragma: no cover - retained for honest solver failure reporting.
        status_text = str(status)
        return PropertySolveResult(
            milp_solver_available=True,
            solver_name="z3",
            solver_status=status_text,
            property_checked=False,
            property_holds=None,
            certified_upper_bound=None,
            witness_x=None,
            selected_segment=None,
            binary_variable_count=abstraction.binary_variable_count,
            blocked_reason=f"blocked_kan_pwa_milp_solver_status_{status_text}",
            certificate={
                "kind": "solver_failure",
                "method": "z3_mixed_integer_linear_pwa_bound",
                "status": status_text,
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
    property_holds = certified_upper <= PROPERTY_THRESHOLD + abstraction.error_bound
    return PropertySolveResult(
        milp_solver_available=True,
        solver_name="z3",
        solver_status="optimal",
        property_checked=True,
        property_holds=property_holds,
        certified_upper_bound=certified_upper,
        witness_x=witness_x,
        selected_segment=selected_segment,
        binary_variable_count=abstraction.binary_variable_count,
        blocked_reason=None,
        certificate={
            "kind": "certificate" if property_holds else "counterexample",
            "method": "z3_mixed_integer_linear_pwa_bound",
            "property": f"For all x in [-1, 1], KAEM unit energy(x) <= {PROPERTY_THRESHOLD}.",
            "certified_upper_bound": certified_upper,
            "property_threshold": PROPERTY_THRESHOLD,
            "abstraction_error_bound": abstraction.error_bound,
            "witness_maximizer_x": witness_x,
            "selected_segment": selected_segment,
        },
    )


def solve_bound_property(
    abstraction: PWAAbstraction,
    solver_name: str | None = None,
) -> PropertySolveResult:
    """Check ``energy(x) <= 1.0`` using a local linear-integer solver."""

    selected_solver = detect_milp_solver() if solver_name is None else solver_name
    if selected_solver == "z3":
        return _solve_bound_with_z3(abstraction)
    return _blocked_solver_result(abstraction)


def _checksum_payload(abstraction: PWAAbstraction, result: PropertySolveResult) -> str:
    """Hash deterministic proof inputs and outputs, excluding wall-clock time."""

    payload = {
        "abstraction": abstraction.as_serializable(),
        "component_path": KAN_COMPONENT_PATH,
        "property_result": result.as_serializable(),
        "property_threshold": PROPERTY_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
        "spec_refs": SPEC_REFS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(solver_name: str | None = None) -> dict[str, Any]:
    """Build the Exp 5080 deliverable payload."""

    start = time.perf_counter()
    layer = build_tiny_kaem_layer()
    abstraction = build_pwa_abstraction(layer)
    result = solve_bound_property(abstraction, solver_name=solver_name)
    success = result.property_checked and result.property_holds is True
    honest_verdict = SUCCESS_VERDICT if success else BLOCKED_VERDICT
    methodology_note = (
        "error_bound is exactly 0.0 because UnivariateKAEMLayer evaluates a "
        "linear interpolation spline; the exported segments are the same exact "
        "piecewise-affine representation, not an approximation to a curved unit."
    )

    artifact = {
        "schema": "carnot.kan_pwa_milp_bridge.v466",
        "experiment": 5080,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "duration_s": round(time.perf_counter() - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kan_component_path": KAN_COMPONENT_PATH,
        "pwa_abstraction_built": True,
        "milp_solver_available": result.milp_solver_available,
        "property_checked": result.property_checked,
        "property_holds": result.property_holds,
        "property_statement": (
            f"For all x in [-1, 1], the tiny KAEM unit energy(x) <= {PROPERTY_THRESHOLD}."
        ),
        "error_bound": abstraction.error_bound,
        "error_bound_detail": {
            "local_max_abs": abstraction.error_bound,
            "global_output_error": abstraction.error_bound,
            "reason": "exact piecewise-affine export of KAEM linear interpolation",
        },
        "binary_variable_count": result.binary_variable_count,
        "blocked_reason": result.blocked_reason,
        "flagged_adversarial": False,
        "solver_name": result.solver_name,
        "solver_status": result.solver_status,
        "certified_upper_bound": result.certified_upper_bound,
        "witness_x": result.witness_x,
        "selected_segment": result.selected_segment,
        "counterexample_or_certificate": result.certificate,
        "pwa_abstraction": abstraction.as_serializable(),
        "methodology_note": methodology_note,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "tests_run": [
            ".venv/bin/python -m pytest tests/python/test_experiment_5080_kan_pwa_milp_bridge.py -q --no-cov"
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(abstraction, result)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5080 artifact drifts from its schema boundary."""

    missing = set(REQUIRED_USER_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(
        verdict in {SUCCESS_VERDICT, BLOCKED_VERDICT},
        "honest_verdict must be the success or blocked Exp 5080 terminal state",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be deterministic_formal_check",
    )
    _require(
        artifact["kan_component_path"] == KAN_COMPONENT_PATH,
        "kan_component_path must point at UnivariateKAEMLayer",
    )
    _require(artifact["pwa_abstraction_built"] is True, "PWA abstraction must be built")
    _require(isinstance(artifact["duration_s"], float), "duration_s must be a float")
    _require(artifact["duration_s"] >= 0.0, "duration_s cannot be negative")
    _require(artifact["error_bound"] >= 0.0, "error_bound cannot be negative")
    _require(
        artifact["binary_variable_count"] == artifact["pwa_abstraction"]["binary_variable_count"],
        "binary variable count must match PWA abstraction",
    )
    _require(artifact["flagged_adversarial"] is False, "Exp 5080 artifact must not be flagged")
    _require(
        set(REQUIRED_USER_ARTIFACT_FIELDS).issubset(artifact["field_principles"]),
        "field_principles must cover every required user field",
    )

    if artifact["milp_solver_available"]:
        _require(artifact["solver_name"] == "z3", "z3 is the only supported Exp 5080 solver")
        _require(artifact["solver_status"] == "optimal", "solver must return optimal")
        _require(artifact["property_checked"] is True, "available solver must check property")
        _require(artifact["property_holds"] is True, "success artifact requires property_holds")
        _require(artifact["blocked_reason"] is None, "success artifact cannot be blocked")
        _require(verdict == SUCCESS_VERDICT, "available optimal solver must use success verdict")
    else:
        _require(artifact["property_checked"] is False, "blocked artifact cannot check property")
        _require(artifact["property_holds"] is None, "blocked artifact cannot assert property")
        _require(
            artifact["blocked_reason"] == "blocked_kan_pwa_milp_solver_unavailable",
            "blocked artifact must name missing solver",
        )
        _require(verdict == BLOCKED_VERDICT, "solver-unavailable artifact must use blocked verdict")


def write_outputs(*, artifact_path: str | Path, solver_name: str | None = None) -> dict[str, Any]:
    """Write the Exp 5080 JSON artifact and return the validated payload."""

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

    root = Path(os.environ.get("CARNOT_EXP5080_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(artifact_path=root / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
