"""Exp 5277 bounded KAN PWA/MILP certificate scale check.

Spec refs: REQ-KAN-5277, SCENARIO-KAN-5277.

V481 kept the KAN certificate path useful but intentionally small: two convex
components, explanation/refinement metadata, and a near-threshold false-property
control. This module scales only one notch. It builds a deterministic
three-component convex fixture, replaces each component with a two-piece affine
upper envelope, maximizes the envelope with a local Z3 MILP-compatible encoding,
and checks a true and nearby false upper-bound property. The dynamic spot check
is deliberately simple because its job is not to prove the property; it is a
second implementation sanity pass that catches mistakes in the envelope or
witness accounting before a terminal artifact is written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5265_kan_certificate_explanation_refinement_v481 as v481


JsonDict = dict[str, Any]

RUN_DATE = "20260705"
EXPERIMENT_ID = "exp5277-kan-milp-certificate-scale-v482"
SCHEMA = "carnot.experiment_5277.kan_milp_certificate_scale.v482"
RESULT_RELATIVE_PATH = Path("results/experiment_5277_kan_milp_certificate_scale_v482.json")
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
SPEC_REFS = ("REQ-KAN-5277", "SCENARIO-KAN-5277")
RANDOM_SEED = 5277
TERMINAL_PREFIXES = ("complete:", "blocked_")

INPUT_BOX = ((-0.4, 0.6), (-0.4, 0.6), (-0.4, 0.6))
TRUE_PROPERTY_THRESHOLD = 0.515
FALSE_PROPERTY_THRESHOLD = 0.498
PIECES_PER_COMPONENT = 2
COMPONENT_COEFFICIENTS = (
    (0.10, 0.18, 0.08),
    (0.08, 0.14, 0.05),
    (0.06, 0.10, 0.03),
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "certificate_scaled",
    "false_property_rejected",
    "approximation_slack",
    "piece_count",
    "solve_time_s",
    "dynamic_spot_check_passed",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether the scaled certificate "
        "is positive, null, blocked, or too loose."
    ),
    "inference_substrate": "Must be offline_deterministic_certificate_no_llm.",
    "certificate_scaled": (
        "True only when the source V481 certificate is ready, the V482 true property "
        "certifies, the false property rejects, and dynamic spot checks pass."
    ),
    "false_property_rejected": (
        "True only when the nearby expected-false threshold is rejected by a deterministic witness."
    ),
    "approximation_slack": (
        "Numeric threshold slack after the PWA/MILP upper-bound and error accounting."
    ),
    "piece_count": "Integer count of PWA envelope pieces submitted to the solver.",
    "solve_time_s": "Measured local deterministic solver time in seconds.",
    "dynamic_spot_check_passed": (
        "True only when sampled points stay inside the true property and expose a false-property witness."
    ),
}
WRAPPED_FIELDS = tuple(field for field in REQUIRED_ARTIFACT_FIELDS if field != "tests_run")


def _require(condition: bool, message: str) -> None:
    if condition:
        return
    raise AssertionError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact[field]
    _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
    _require("value" in wrapped, f"{field} missing value")
    return wrapped["value"]


def wrap_field(field: str, value: Any) -> JsonDict:
    """Return the principle-wrapped field shape used by terminal artifacts."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


@dataclass(frozen=True)
class SourceEvidence:
    """Validated V481 source certificate state used as the V482 anchor."""

    artifact_path: str
    ready: bool
    false_property_rejected: bool
    source_verdict: str

    def as_serializable(self) -> JsonDict:
        return {
            "artifact_path": self.artifact_path,
            "ready": self.ready,
            "false_property_rejected": self.false_property_rejected,
            "source_verdict": self.source_verdict,
        }


@dataclass(frozen=True)
class ConvexComponent:
    """One bounded convex univariate component in the small certificate fixture."""

    component_id: str
    variable_index: int
    interval: tuple[float, float]
    quadratic: float
    linear: float
    constant: float

    def evaluate(self, x_value: float) -> float:
        """Evaluate the underlying convex component, not its envelope."""

        return self.quadratic * x_value * x_value + self.linear * x_value + self.constant

    def as_serializable(self) -> JsonDict:
        return {
            "component_id": self.component_id,
            "variable_index": self.variable_index,
            "interval": list(self.interval),
            "polynomial": {
                "quadratic": self.quadratic,
                "linear": self.linear,
                "constant": self.constant,
            },
        }


@dataclass(frozen=True)
class PWAPiece:
    """One affine chord upper envelope for a convex component interval."""

    component_id: str
    variable_index: int
    piece_index: int
    interval: tuple[float, float]
    slope: float
    intercept: float
    local_error_bound: float

    def evaluate_upper(self, x_value: float) -> float:
        """Evaluate the affine upper envelope on this piece."""

        return self.slope * x_value + self.intercept

    def contains(self, x_value: float) -> bool:
        lo, hi = self.interval
        return lo - 1e-12 <= x_value <= hi + 1e-12

    def as_serializable(self) -> JsonDict:
        return {
            "component_id": self.component_id,
            "variable_index": self.variable_index,
            "piece_index": self.piece_index,
            "interval": list(self.interval),
            "slope": self.slope,
            "intercept": self.intercept,
            "local_error_bound": self.local_error_bound,
        }


@dataclass(frozen=True)
class MultiComponentAbstraction:
    """The small multi-component PWA abstraction submitted to the solver."""

    components: tuple[ConvexComponent, ...]
    pieces_by_component: tuple[tuple[PWAPiece, ...], ...]

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def input_box(self) -> tuple[tuple[float, float], ...]:
        return tuple(component.interval for component in self.components)

    @property
    def piece_count(self) -> int:
        return sum(len(pieces) for pieces in self.pieces_by_component)

    @property
    def binary_variable_count(self) -> int:
        return self.piece_count

    @property
    def local_error_bounds(self) -> tuple[float, ...]:
        return tuple(max(piece.local_error_bound for piece in pieces) for pieces in self.pieces_by_component)

    @property
    def local_error_bound_max(self) -> float:
        return max(self.local_error_bounds, default=0.0)

    @property
    def global_error_bound(self) -> float:
        return sum(self.local_error_bounds)

    def evaluate_actual(self, inputs: Sequence[float]) -> float:
        """Evaluate the underlying convex fixture at one input point."""

        _require(len(inputs) == self.component_count, "input dimension drift")
        return sum(component.evaluate(float(inputs[index])) for index, component in enumerate(self.components))

    def envelope_piece_for(self, component_index: int, x_value: float) -> PWAPiece:
        """Return the active envelope piece for a component/input pair."""

        pieces = self.pieces_by_component[component_index]
        for piece in pieces:
            if piece.contains(float(x_value)):
                return piece
        raise AssertionError(f"x={x_value} outside component {component_index} envelope")

    def evaluate_upper_envelope(self, inputs: Sequence[float]) -> float:
        """Evaluate the selected PWA upper envelope at one input point."""

        _require(len(inputs) == self.component_count, "input dimension drift")
        total = 0.0
        for index, x_value in enumerate(inputs):
            total += self.envelope_piece_for(index, float(x_value)).evaluate_upper(float(x_value))
        return total

    def as_serializable(self) -> JsonDict:
        return {
            "component_count": self.component_count,
            "input_box": [list(bounds) for bounds in self.input_box],
            "piece_count": self.piece_count,
            "binary_variable_count": self.binary_variable_count,
            "local_error_bounds": list(self.local_error_bounds),
            "local_error_bound_max": self.local_error_bound_max,
            "global_error_bound": self.global_error_bound,
            "components": [component.as_serializable() for component in self.components],
            "pieces_by_component": [
                [piece.as_serializable() for piece in pieces]
                for pieces in self.pieces_by_component
            ],
            "method": "convex_quadratic_chord_upper_envelopes_with_z3_milp_selectors",
        }


@dataclass(frozen=True)
class PropertyOutcome:
    """True or false property outcome from the scaled certificate solve."""

    property_id: str
    threshold: float
    certified_upper_bound: float | None
    certificate_slack: float | None
    certified: bool
    rejected: bool
    counterexample_inputs: tuple[float, ...] | None
    actual_witness_value: float | None

    def as_serializable(self) -> JsonDict:
        return {
            "property_id": self.property_id,
            "threshold": self.threshold,
            "certified_upper_bound": self.certified_upper_bound,
            "certificate_slack": self.certificate_slack,
            "certified": self.certified,
            "rejected": self.rejected,
            "counterexample_inputs": list(self.counterexample_inputs)
            if self.counterexample_inputs is not None
            else None,
            "actual_witness_value": self.actual_witness_value,
        }


@dataclass(frozen=True)
class SolverResult:
    """MILP-compatible solve telemetry for the shared upper-bound maximum."""

    solver_available: bool
    solver_name: str
    solver_status: str
    certified_upper_bound: float | None
    witness_inputs: tuple[float, ...] | None
    selected_pieces: tuple[int, ...] | None
    piece_count: int
    binary_variable_count: int
    constraint_count: int
    solve_time_s: float
    property_results: tuple[PropertyOutcome, ...]
    blocked_reason: str | None

    @property
    def true_property(self) -> PropertyOutcome:
        return self.property_results[0]

    @property
    def false_property(self) -> PropertyOutcome:
        return self.property_results[1]

    @property
    def approximation_slack(self) -> float:
        return float(self.true_property.certificate_slack or 0.0)

    def as_serializable(self) -> JsonDict:
        return {
            "solver_available": self.solver_available,
            "solver_name": self.solver_name,
            "solver_status": self.solver_status,
            "certified_upper_bound": self.certified_upper_bound,
            "witness_inputs": list(self.witness_inputs) if self.witness_inputs is not None else None,
            "selected_pieces": list(self.selected_pieces) if self.selected_pieces is not None else None,
            "piece_count": self.piece_count,
            "binary_variable_count": self.binary_variable_count,
            "constraint_count": self.constraint_count,
            "solve_time_s": self.solve_time_s,
            "blocked_reason": self.blocked_reason,
            "property_results": [row.as_serializable() for row in self.property_results],
        }


@dataclass(frozen=True)
class DynamicSpotCheck:
    """Deterministic sampled sanity check over the certified input region."""

    passed: bool
    sample_count: int
    max_actual_value: float
    max_upper_envelope_value: float
    max_observed_envelope_gap: float
    envelope_violation_count: int
    true_threshold_violation_count: int
    false_property_witness_seen: bool

    def as_serializable(self) -> JsonDict:
        return {
            "passed": self.passed,
            "sample_count": self.sample_count,
            "max_actual_value": self.max_actual_value,
            "max_upper_envelope_value": self.max_upper_envelope_value,
            "max_observed_envelope_gap": self.max_observed_envelope_gap,
            "envelope_violation_count": self.envelope_violation_count,
            "true_threshold_violation_count": self.true_threshold_violation_count,
            "false_property_witness_seen": self.false_property_witness_seen,
        }


def load_v481_source(path: str | Path = v481.RESULT_RELATIVE_PATH) -> SourceEvidence:
    """Load and validate the V481 certificate artifact before scaling it."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    v481.validate_artifact(artifact)
    return SourceEvidence(
        artifact_path=str(path),
        ready=artifact["certificate_refinement_ready"] is True,
        false_property_rejected=artifact["false_property_rejected"]["value"] is True,
        source_verdict=str(artifact["honest_verdict"]["value"]),
    )


def _split_interval(interval: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    lo, hi = interval
    width = (hi - lo) / PIECES_PER_COMPONENT
    return tuple((lo + width * index, lo + width * (index + 1)) for index in range(PIECES_PER_COMPONENT))


def _build_component(variable_index: int, coefficients: Sequence[float]) -> ConvexComponent:
    return ConvexComponent(
        component_id=f"v482_component_{variable_index}",
        variable_index=variable_index,
        interval=INPUT_BOX[variable_index],
        quadratic=float(coefficients[0]),
        linear=float(coefficients[1]),
        constant=float(coefficients[2]),
    )


def _build_piece(component: ConvexComponent, piece_index: int, interval: tuple[float, float]) -> PWAPiece:
    lo, hi = interval
    y_lo = component.evaluate(lo)
    y_hi = component.evaluate(hi)
    slope = (y_hi - y_lo) / (hi - lo)
    intercept = y_lo - slope * lo
    local_error_bound = component.quadratic * (hi - lo) * (hi - lo) / 4.0
    return PWAPiece(
        component_id=component.component_id,
        variable_index=component.variable_index,
        piece_index=piece_index,
        interval=interval,
        slope=slope,
        intercept=intercept,
        local_error_bound=local_error_bound,
    )


def build_multi_component_abstraction() -> MultiComponentAbstraction:
    """Build the deterministic three-component PWA upper-envelope fixture."""

    components = tuple(
        _build_component(variable_index=index, coefficients=coefficients)
        for index, coefficients in enumerate(COMPONENT_COEFFICIENTS)
    )
    pieces_by_component = tuple(
        tuple(
            _build_piece(component, piece_index=index, interval=interval)
            for index, interval in enumerate(_split_interval(component.interval))
        )
        for component in components
    )
    return MultiComponentAbstraction(
        components=components,
        pieces_by_component=pieces_by_component,
    )


def detect_solver() -> str:
    """Return the local MILP-compatible backend available for this bounded run."""

    return "z3" if importlib.util.find_spec("z3") is not None else ""


def expected_constraint_count(abstraction: MultiComponentAbstraction) -> int:
    """Count generated linear constraints without importing the solver."""

    return sum(3 + 6 * len(pieces) for pieces in abstraction.pieces_by_component) + 1


def _z3_float(value: Any) -> float:
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    if text.endswith("?"):  # pragma: no cover - Z3 decimal approximation fallback.
        text = text[:-1]
    return float(text)


def _real(z3: Any, value: float) -> Any:
    return z3.RealVal(repr(float(value)))


def _blocked_properties(abstraction: MultiComponentAbstraction) -> tuple[PropertyOutcome, ...]:
    return (
        PropertyOutcome(
            property_id="v482_true_multi_component_upper_property",
            threshold=TRUE_PROPERTY_THRESHOLD,
            certified_upper_bound=None,
            certificate_slack=None,
            certified=False,
            rejected=False,
            counterexample_inputs=None,
            actual_witness_value=None,
        ),
        PropertyOutcome(
            property_id="v482_nearby_false_multi_component_upper_property",
            threshold=FALSE_PROPERTY_THRESHOLD,
            certified_upper_bound=None,
            certificate_slack=None,
            certified=False,
            rejected=False,
            counterexample_inputs=None,
            actual_witness_value=None,
        ),
    )


def _blocked_solver_result(abstraction: MultiComponentAbstraction) -> SolverResult:
    return SolverResult(
        solver_available=False,
        solver_name="",
        solver_status="blocked_solver_dependency",
        certified_upper_bound=None,
        witness_inputs=None,
        selected_pieces=None,
        piece_count=abstraction.piece_count,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=expected_constraint_count(abstraction),
        solve_time_s=0.0,
        property_results=_blocked_properties(abstraction),
        blocked_reason="blocked_kan_pwa_milp_solver_unavailable",
    )


def _property_results(
    abstraction: MultiComponentAbstraction,
    certified_upper: float,
    witness_inputs: tuple[float, ...],
) -> tuple[PropertyOutcome, ...]:
    actual_witness = abstraction.evaluate_actual(witness_inputs)
    true_slack = TRUE_PROPERTY_THRESHOLD - certified_upper
    false_slack = FALSE_PROPERTY_THRESHOLD - certified_upper
    return (
        PropertyOutcome(
            property_id="v482_true_multi_component_upper_property",
            threshold=TRUE_PROPERTY_THRESHOLD,
            certified_upper_bound=certified_upper,
            certificate_slack=true_slack,
            certified=true_slack >= -1e-12,
            rejected=False,
            counterexample_inputs=None,
            actual_witness_value=None,
        ),
        PropertyOutcome(
            property_id="v482_nearby_false_multi_component_upper_property",
            threshold=FALSE_PROPERTY_THRESHOLD,
            certified_upper_bound=certified_upper,
            certificate_slack=false_slack,
            certified=false_slack >= -1e-12,
            rejected=false_slack < 0.0 and actual_witness > FALSE_PROPERTY_THRESHOLD,
            counterexample_inputs=witness_inputs,
            actual_witness_value=actual_witness,
        ),
    )


def solve_scaled_certificate(
    abstraction: MultiComponentAbstraction,
    *,
    solver_name: str | None = None,
) -> SolverResult:
    """Maximize the PWA upper envelope and evaluate true/false thresholds."""

    selected_solver = detect_solver() if solver_name is None else solver_name
    if selected_solver != "z3":
        return _blocked_solver_result(abstraction)

    z3 = importlib.import_module("z3")
    optimizer = z3.Optimize()
    xs = [z3.Real(f"exp5277_x_{index}") for index in range(abstraction.component_count)]
    ys = [
        z3.Real(f"exp5277_component_upper_{index}")
        for index in range(abstraction.component_count)
    ]
    total_upper = z3.Real("exp5277_total_upper")
    selected_flag_groups: list[list[Any]] = []
    constraint_count = 0
    big_m = _real(z3, 10.0)

    def add_constraints(*constraints: Any) -> None:
        nonlocal constraint_count
        optimizer.add(*constraints)
        constraint_count += len(constraints)

    for component_index, pieces in enumerate(abstraction.pieces_by_component):
        x = xs[component_index]
        y = ys[component_index]
        lo, hi = abstraction.components[component_index].interval
        flags = [
            z3.Int(f"exp5277_component_{component_index}_piece_{piece.piece_index}")
            for piece in pieces
        ]
        selected_flag_groups.append(flags)

        add_constraints(x >= _real(z3, lo), x <= _real(z3, hi), z3.Sum(flags) == 1)
        for flag, piece in zip(flags, pieces):
            flag_real = z3.ToReal(flag)
            slack = big_m * (_real(z3, 1.0) - flag_real)
            affine_value = _real(z3, piece.slope) * x + _real(z3, piece.intercept)
            add_constraints(
                flag >= 0,
                flag <= 1,
                x >= _real(z3, piece.interval[0]) - slack,
                x <= _real(z3, piece.interval[1]) + slack,
                y - affine_value <= slack,
                affine_value - y <= slack,
            )
    add_constraints(total_upper == z3.Sum(ys))

    solve_start = time.perf_counter()
    objective = optimizer.maximize(total_upper)
    status = optimizer.check()
    solve_time_s = round(time.perf_counter() - solve_start, 6)
    if status != z3.sat:  # pragma: no cover - unexpected solver status is still reported.
        status_text = str(status)
        return SolverResult(
            solver_available=True,
            solver_name="z3",
            solver_status=status_text,
            certified_upper_bound=None,
            witness_inputs=None,
            selected_pieces=None,
            piece_count=abstraction.piece_count,
            binary_variable_count=abstraction.binary_variable_count,
            constraint_count=constraint_count,
            solve_time_s=solve_time_s,
            property_results=_blocked_properties(abstraction),
            blocked_reason=f"blocked_kan_pwa_milp_solver_status_{status_text}",
        )

    model = optimizer.model()
    certified_upper = _z3_float(objective.value())
    witness_inputs = tuple(_z3_float(model.eval(x, model_completion=True)) for x in xs)
    selected_pieces = tuple(
        next(
            piece_index
            for piece_index, flag in enumerate(flags)
            if _z3_float(model.eval(flag, model_completion=True)) > 0.5
        )
        for flags in selected_flag_groups
    )
    return SolverResult(
        solver_available=True,
        solver_name="z3",
        solver_status="optimal",
        certified_upper_bound=certified_upper,
        witness_inputs=witness_inputs,
        selected_pieces=selected_pieces,
        piece_count=abstraction.piece_count,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=constraint_count,
        solve_time_s=solve_time_s,
        property_results=_property_results(abstraction, certified_upper, witness_inputs),
        blocked_reason=None,
    )


def _axis_samples(interval: tuple[float, float], count: int) -> tuple[float, ...]:
    lo, hi = interval
    if count == 1:
        return ((lo + hi) / 2.0,)
    step = (hi - lo) / (count - 1)
    return tuple(lo + step * index for index in range(count))


def run_dynamic_spot_check(
    abstraction: MultiComponentAbstraction,
    solver_result: SolverResult,
    *,
    samples_per_axis: int = 5,
) -> DynamicSpotCheck:
    """Run deterministic sampled falsification over the certified input box."""

    axes = [_axis_samples(interval, samples_per_axis) for interval in abstraction.input_box]
    sample_count = 0
    max_actual = -math.inf
    max_upper = -math.inf
    max_gap = -math.inf
    envelope_violations = 0
    true_threshold_violations = 0
    false_witness_seen = False

    for point in itertools.product(*axes):
        sample_count += 1
        actual = abstraction.evaluate_actual(point)
        upper = abstraction.evaluate_upper_envelope(point)
        max_actual = max(max_actual, actual)
        max_upper = max(max_upper, upper)
        max_gap = max(max_gap, upper - actual)
        if actual > upper + 1e-10:
            envelope_violations += 1
        if actual > TRUE_PROPERTY_THRESHOLD + 1e-10:
            true_threshold_violations += 1
        if actual > FALSE_PROPERTY_THRESHOLD:
            false_witness_seen = True

    passed = (
        solver_result.false_property.rejected
        and envelope_violations == 0
        and true_threshold_violations == 0
        and false_witness_seen
    )
    return DynamicSpotCheck(
        passed=passed,
        sample_count=sample_count,
        max_actual_value=max_actual,
        max_upper_envelope_value=max_upper,
        max_observed_envelope_gap=max_gap,
        envelope_violation_count=envelope_violations,
        true_threshold_violation_count=true_threshold_violations,
        false_property_witness_seen=false_witness_seen,
    )


def _honest_verdict(
    *,
    source: SourceEvidence,
    solver_result: SolverResult,
    spot_check: DynamicSpotCheck,
) -> str:
    if not source.ready:
        return "blocked_source_v481_certificate_not_ready"
    if not solver_result.solver_available:
        return "blocked_kan_pwa_milp_solver_unavailable"
    if not solver_result.true_property.certified:
        return "complete: scaled certificate too loose for the bounded V482 true property"
    if not solver_result.false_property.rejected:
        return "complete: scaled certificate null because the nearby false property was not rejected"
    if not spot_check.passed:
        return "complete: scaled certificate blocked by dynamic spot-check failure"
    return (
        "complete: scaled certificate positive for a bounded three-component PWA/MILP "
        "fixture with explicit approximation slack and nearby false-property rejection"
    )


def _checksum_payload(
    source: SourceEvidence,
    abstraction: MultiComponentAbstraction,
    solver_result: SolverResult,
    spot_check: DynamicSpotCheck,
) -> str:
    solver_payload = solver_result.as_serializable()
    solver_payload["solve_time_s"] = "excluded"
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "source": source.as_serializable(),
        "abstraction": abstraction.as_serializable(),
        "solver_result": solver_payload,
        "spot_check": spot_check.as_serializable(),
        "spec_refs": SPEC_REFS,
        "thresholds": {
            "true": TRUE_PROPERTY_THRESHOLD,
            "false": FALSE_PROPERTY_THRESHOLD,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    solver_name: str | None = None,
) -> JsonDict:
    """Build and validate the Exp 5277 terminal artifact."""

    start = time.perf_counter()
    source = load_v481_source()
    abstraction = build_multi_component_abstraction()
    solver_result = solve_scaled_certificate(abstraction, solver_name=solver_name)
    spot_check = run_dynamic_spot_check(abstraction, solver_result)
    certificate_scaled = (
        source.ready
        and solver_result.true_property.certified
        and solver_result.false_property.rejected
        and spot_check.passed
    )
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "honest_verdict": wrap_field(
            "honest_verdict",
            _honest_verdict(
                source=source,
                solver_result=solver_result,
                spot_check=spot_check,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "certificate_scaled": wrap_field("certificate_scaled", certificate_scaled),
        "false_property_rejected": wrap_field(
            "false_property_rejected",
            solver_result.false_property.rejected,
        ),
        "approximation_slack": wrap_field(
            "approximation_slack",
            solver_result.approximation_slack,
        ),
        "piece_count": wrap_field("piece_count", abstraction.piece_count),
        "solve_time_s": wrap_field("solve_time_s", solver_result.solve_time_s),
        "dynamic_spot_check_passed": wrap_field(
            "dynamic_spot_check_passed",
            spot_check.passed,
        ),
        "tests_run": list(tests_run or []),
        "source_v481": source.as_serializable(),
        "source_artifacts": [str(v481.RESULT_RELATIVE_PATH)],
        "pwa_abstraction": abstraction.as_serializable(),
        "solver_result": solver_result.as_serializable(),
        "property_results": [row.as_serializable() for row in solver_result.property_results],
        "slack_accounting": {
            "true_property_slack": solver_result.true_property.certificate_slack,
            "false_property_slack": solver_result.false_property.certificate_slack,
            "certified_upper_bound": solver_result.certified_upper_bound,
            "true_property_threshold": TRUE_PROPERTY_THRESHOLD,
            "false_property_threshold": FALSE_PROPERTY_THRESHOLD,
            "local_error_bounds": list(abstraction.local_error_bounds),
            "local_error_bound_max": abstraction.local_error_bound_max,
            "global_error_bound": abstraction.global_error_bound,
        },
        "spot_check": spot_check.as_serializable(),
        "claim_limits": [
            "bounded deterministic three-component convex fixture only",
            "no broad KAN verification claim",
            "no trained-network soundness claim",
            "no hardware execution or hardware speedup claim",
            "no live LLM inference claim",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "blocked_reason": None if certificate_scaled else solver_result.blocked_reason,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(
        source,
        abstraction,
        solver_result,
        spot_check,
    )
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5277 artifact drifts from the narrow contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    verdict = _wrapped_value(artifact, "honest_verdict")
    substrate = _wrapped_value(artifact, "inference_substrate")
    certificate_scaled = _wrapped_value(artifact, "certificate_scaled")
    false_rejected = _wrapped_value(artifact, "false_property_rejected")
    approximation_slack = _wrapped_value(artifact, "approximation_slack")
    piece_count = _wrapped_value(artifact, "piece_count")
    solve_time_s = _wrapped_value(artifact, "solve_time_s")
    spot_check_passed = _wrapped_value(artifact, "dynamic_spot_check_passed")

    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require(
        any(token in verdict for token in ("positive", "null", "blocked", "too loose")),
        "honest_verdict must state positive, null, blocked, or too loose",
    )
    _require(substrate == INFERENCE_SUBSTRATE, "inference_substrate must be offline deterministic")
    _require(isinstance(certificate_scaled, bool), "certificate_scaled must be bool")
    _require(isinstance(false_rejected, bool), "false_property_rejected must be bool")
    _require(_is_number(approximation_slack), "approximation_slack must be numeric")
    _require(isinstance(piece_count, int) and piece_count == 6, "piece_count must be integer 6")
    _require(_is_number(solve_time_s) and solve_time_s >= 0.0, "solve_time_s must be nonnegative numeric")
    _require(isinstance(spot_check_passed, bool), "dynamic_spot_check_passed must be bool")
    _require(isinstance(artifact["tests_run"], list), "tests_run must be list")

    claim_limits = " ".join(str(item) for item in artifact["claim_limits"])
    _require("broad KAN verification" in claim_limits, "must limit broad KAN claim")
    _require("hardware speedup" in claim_limits, "must limit hardware speedup claim")
    _require("live LLM" in claim_limits, "must limit live LLM claim")
    _require("REQ-KAN-5277" in artifact["spec_refs"], "spec_refs must include REQ-KAN-5277")

    property_by_id = {row["property_id"]: row for row in artifact["property_results"]}
    _require(
        "v482_true_multi_component_upper_property" in property_by_id,
        "missing true property result",
    )
    _require(
        "v482_nearby_false_multi_component_upper_property" in property_by_id,
        "missing false property result",
    )

    if certificate_scaled:
        _require(artifact["blocked_reason"] is None, "scaled certificate cannot have blocker")
        _require(false_rejected is True, "false property must be rejected")
        _require(spot_check_passed is True, "dynamic spot check must pass")
        _require(approximation_slack > 0.0, "approximation_slack must be positive")
        _require("positive" in verdict, "positive scaled certificate must say positive")
        _require(artifact["source_v481"]["ready"] is True, "source V481 certificate must be ready")
        _require(artifact["slack_accounting"]["true_property_slack"] == approximation_slack, "slack accounting drift")
        _require(artifact["slack_accounting"]["global_error_bound"] > 0.0, "global error bound must be explicit")
        _require(artifact["spot_check"]["passed"] is True, "spot check payload drift")
    else:
        _require(
            artifact["blocked_reason"] is not None or "null" in verdict or "too loose" in verdict,
            "unscaled certificate must explain blocked/null/too loose status",
        )


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    solver_name: str | None = None,
) -> JsonDict:
    """Write the Exp 5277 JSON artifact and return the validated payload."""

    artifact = build_artifact(
        duration_s=duration_s,
        tests_run=tests_run,
        solver_name=solver_name,
    )
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - covered through write_outputs tests.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
