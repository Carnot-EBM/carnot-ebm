"""Exp 5242 bounded KAEM certificate abstraction scale test.

Spec refs: REQ-KAN-5242, SCENARIO-KAN-5242.

The experiment starts from the committed Exp 5230 certificate, records its exact
dimensions, and then applies small bounded stresses to the same deterministic
KAEM/PWA/MILP path. The result is a certificate boundary, not a hardware or
broad KAN verification claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

RUN_DATE = "20260704"
EXPERIMENT_ID = "exp5242-kan-certificate-abstraction-scale-v479"
SCHEMA = "carnot.experiment_5242.kan_certificate_abstraction_scale.v479"
RESULT_RELATIVE_PATH = Path("results/experiment_5242_kan_certificate_abstraction_scale_v479.json")
BASELINE_RELATIVE_PATH = Path("results/experiment_5230_kan_milp_verifier_certificate_v478.json")
TARGET_MODULE = "python/carnot/models/kaem_energy.py::UnivariateKAEMLayer"
INFERENCE_SUBSTRATE = "deterministic_kan_pwa_milp_certificate"
SPEC_REFS = ("REQ-KAN-5242", "SCENARIO-KAN-5242")
RANDOM_SEED = 5242
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

STRESS_CONTROL_POINTS = (
    (0.0, 0.04, 0.14, 0.28, 0.45, 0.62),
    (0.0, 0.03, 0.09, 0.18, 0.30, 0.42),
)
MORE_SEGMENTS_BOX = ((-0.25, 0.5), (-0.25, 0.5))
WIDER_DOMAIN_BOX = ((-0.5, 0.75), (-0.5, 0.75))
MORE_SEGMENTS_THRESHOLD = 0.72
WIDER_DOMAIN_THRESHOLD = 0.88
FALSE_PROPERTY_THRESHOLD = 0.84

FIELD_PRINCIPLES = {
    "kan_certificate_baseline_reproduced": (
        "True only when the committed Exp 5230 artifact validates and its exact dimensions are recorded."
    ),
    "kan_certificate_extended": (
        "True only when at least one bounded stress certificate verifies and the deliberate false property is rejected."
    ),
    "stress_axes": "List of bounded stress dimensions applied after reproducing Exp 5230.",
    "max_pwa_segments_verified": (
        "Largest total PWA segment count among verified non-false stress certificates."
    ),
    "false_property_rejected": (
        "True only when the deliberate false property is rejected by the same deterministic solver path."
    ),
    "certificate_slack_min": (
        "Minimum positive slack across verified non-false stress certificates, or null when blocked."
    ),
    "solve_time_s": "Total deterministic solver time across stress rows.",
    "tests_run": "Commands run for this artifact, with pass/fail status.",
    "inference_substrate": "Must be deterministic_kan_pwa_milp_certificate.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state the bounded certificate scale achieved or blocked."
    ),
}
REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class BaselineReproduction:
    """Validated Exp 5230 baseline dimensions used as the stress-test anchor."""

    reproduced: bool
    dimensions: JsonDict

    def as_serializable(self) -> JsonDict:
        return {"reproduced": self.reproduced, "dimensions": self.dimensions}


@dataclass(frozen=True)
class PWASegment:
    """One exact affine span of the KAEM linear-interpolation spline."""

    index: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class PWAUnit:
    """One univariate KAEM fixture represented as exact PWA segments."""

    variable_index: int
    control_points: tuple[float, ...]
    knots: tuple[float, ...]
    segments: tuple[PWASegment, ...]

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def binary_variable_count(self) -> int:
        return self.n_segments


@dataclass(frozen=True)
class PWAAbstraction:
    """Small additive PWA abstraction for the bounded KAEM stress fixture."""

    component_path: str
    units: tuple[PWAUnit, ...]

    @property
    def input_dimension(self) -> int:
        return len(self.units)

    @property
    def pwa_piece_count(self) -> int:
        return sum(unit.n_segments for unit in self.units)

    @property
    def binary_variable_count(self) -> int:
        return sum(unit.binary_variable_count for unit in self.units)


@dataclass(frozen=True)
class StressCase:
    """One bounded KAEM stress case submitted to the deterministic solver."""

    case_id: str
    stress_axis: str
    input_box: tuple[tuple[float, float], ...]
    threshold: float
    expected_property: str
    is_false_property: bool


@dataclass(frozen=True)
class StressResult:
    """Solver result and certificate telemetry for one bounded stress case."""

    case_id: str
    stress_axis: str
    verified: bool
    false_property_rejected: bool
    input_box: tuple[tuple[float, float], ...]
    threshold: float
    certified_upper_bound: float | None
    certificate_slack: float | None
    min_slope: float | None
    solver_status: str
    solve_time_s: float
    pwa_piece_count: int
    binary_variable_count: int
    constraint_count: int
    selected_segments: tuple[int, ...] | None
    witness_inputs: tuple[float, ...] | None
    numerical_instability: str

    def as_serializable(self) -> JsonDict:
        return {
            "case_id": self.case_id,
            "stress_axis": self.stress_axis,
            "verified": self.verified,
            "false_property_rejected": self.false_property_rejected,
            "input_box": [list(bounds) for bounds in self.input_box],
            "threshold": self.threshold,
            "certified_upper_bound": self.certified_upper_bound,
            "certificate_slack": self.certificate_slack,
            "min_slope": self.min_slope,
            "solver_status": self.solver_status,
            "solve_time_s": self.solve_time_s,
            "pwa_piece_count": self.pwa_piece_count,
            "binary_variable_count": self.binary_variable_count,
            "constraint_count": self.constraint_count,
            "selected_segments": list(self.selected_segments)
            if self.selected_segments is not None
            else None,
            "witness_inputs": list(self.witness_inputs)
            if self.witness_inputs is not None
            else None,
            "numerical_instability": self.numerical_instability,
        }


def wrap_field(field: str, value: Any) -> JsonDict:
    """Return a principle-wrapped artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def detect_solver() -> str:
    """Return the deterministic solver backend available for the bounded MILP path."""

    return "z3" if importlib.util.find_spec("z3") is not None else ""


def reproduce_exp5230_baseline(
    baseline_path: str | Path = BASELINE_RELATIVE_PATH,
) -> BaselineReproduction:
    """Validate and summarize the committed Exp 5230 certificate artifact."""

    artifact = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    _validate_exp5230_baseline_artifact(artifact)
    pwa = artifact["certificate"]["pwa_abstraction"]
    unsafe = next(
        row
        for row in artifact["certificate"]["property_results"]
        if row["property_id"] == "no_unsafe_decision"
    )
    dimensions = {
        "source_artifact": str(baseline_path),
        "input_dimension": pwa["input_dimension"],
        "pwa_piece_count": pwa["pwa_piece_count"],
        "binary_variable_count": pwa["binary_variable_count"],
        "constraint_count": unsafe["details"]["constraint_count"],
        "input_box": artifact["certificate"]["input_box"],
        "baseline_slack": unsafe["bound_tightness"],
        "baseline_solve_status": unsafe["solver_status"],
    }
    return BaselineReproduction(reproduced=True, dimensions=dimensions)


def _validate_exp5230_baseline_artifact(artifact: Mapping[str, Any]) -> None:
    _require(
        artifact["experiment_id"] == "exp5230-kan-milp-verifier-certificate-v478",
        "baseline experiment_id drift",
    )
    _require(
        artifact["inference_substrate"]["value"] == "deterministic_pwa_milp_certificate",
        "baseline substrate drift",
    )
    _require(artifact["certificate"]["target_module"] == TARGET_MODULE, "baseline target drift")
    _require(
        artifact["kan_certificate_produced"]["value"] is True,
        "baseline certificate not produced",
    )


def _uniform_knots(n_knots: int) -> tuple[float, ...]:
    return tuple(-1.0 + (2.0 * index / (n_knots - 1)) for index in range(n_knots))


def _build_unit(variable_index: int, control_points: Sequence[float]) -> PWAUnit:
    knots = _uniform_knots(len(control_points))
    segments = []
    for index, (x_min, x_max, y_min, y_max) in enumerate(
        zip(knots[:-1], knots[1:], control_points[:-1], control_points[1:])
    ):
        slope = (float(y_max) - float(y_min)) / (x_max - x_min)
        intercept = float(y_min) - slope * x_min
        segments.append(
            PWASegment(
                index=index,
                x_min=x_min,
                x_max=x_max,
                y_min=float(y_min),
                y_max=float(y_max),
                slope=slope,
                intercept=intercept,
            )
        )
    return PWAUnit(
        variable_index=variable_index,
        control_points=tuple(float(value) for value in control_points),
        knots=knots,
        segments=tuple(segments),
    )


def build_stress_abstraction() -> PWAAbstraction:
    """Build the deterministic two-variable, ten-piece KAEM PWA stress fixture."""

    units = tuple(
        _build_unit(variable_index=index, control_points=points)
        for index, points in enumerate(STRESS_CONTROL_POINTS)
    )
    return PWAAbstraction(
        component_path=TARGET_MODULE,
        units=units,
    )


def stress_cases() -> tuple[StressCase, ...]:
    """Return the bounded stress axes for Exp 5242."""

    return (
        StressCase(
            case_id="more_pwa_segments",
            stress_axis="more_pwa_segments",
            input_box=MORE_SEGMENTS_BOX,
            threshold=MORE_SEGMENTS_THRESHOLD,
            expected_property="verified",
            is_false_property=False,
        ),
        StressCase(
            case_id="wider_input_bounds",
            stress_axis="wider_input_bounds",
            input_box=WIDER_DOMAIN_BOX,
            threshold=WIDER_DOMAIN_THRESHOLD,
            expected_property="verified",
            is_false_property=False,
        ),
        StressCase(
            case_id="deliberate_false_property",
            stress_axis="deliberate_false_property",
            input_box=WIDER_DOMAIN_BOX,
            threshold=FALSE_PROPERTY_THRESHOLD,
            expected_property="rejected",
            is_false_property=True,
        ),
    )


def expected_constraint_count(abstraction: PWAAbstraction) -> int:
    """Count generated linear constraints without asking Z3 to build them."""

    return sum(3 + 6 * unit.n_segments for unit in abstraction.units) + 1


def _segment_overlaps(segment: Any, lo: float, hi: float) -> bool:
    return segment.x_max >= lo - 1e-12 and segment.x_min <= hi + 1e-12


def bounded_min_slope(
    abstraction: PWAAbstraction,
    input_box: tuple[tuple[float, float], ...],
) -> float:
    """Inspect active PWA slopes over the case input box."""

    slopes = [
        segment.slope
        for unit_index, unit in enumerate(abstraction.units)
        for segment in unit.segments
        if _segment_overlaps(segment, input_box[unit_index][0], input_box[unit_index][1])
    ]
    return min(slopes)


def _z3_float(value: Any) -> float:
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    if text.endswith("?"):
        text = text[:-1]
    return float(text)


def _real(z3: Any, value: float) -> Any:
    return z3.RealVal(repr(float(value)))


def _numerical_instability(slack: float | None, upper_bound: float | None) -> str:
    if slack is None or upper_bound is None:
        return "not_evaluated_solver_blocked"
    if not (math.isfinite(slack) and math.isfinite(upper_bound)):
        return "non_finite_solver_value"
    if abs(slack) <= 1e-9:
        return "near_zero_margin"
    return "none_detected"


def _blocked_result(case: StressCase, abstraction: PWAAbstraction) -> StressResult:
    return StressResult(
        case_id=case.case_id,
        stress_axis=case.stress_axis,
        verified=False,
        false_property_rejected=False,
        input_box=case.input_box,
        threshold=case.threshold,
        certified_upper_bound=None,
        certificate_slack=None,
        min_slope=None,
        solver_status="blocked_solver_dependency",
        solve_time_s=0.0,
        pwa_piece_count=abstraction.pwa_piece_count,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=expected_constraint_count(abstraction),
        selected_segments=None,
        witness_inputs=None,
        numerical_instability="not_evaluated_solver_blocked",
    )


def solve_stress_case(
    case: StressCase,
    abstraction: PWAAbstraction,
    *,
    solver_name: str | None = None,
) -> StressResult:
    """Maximize a bounded additive PWA fixture and compare it with the case threshold."""

    selected_solver = detect_solver() if solver_name is None else solver_name
    if selected_solver != "z3":
        return _blocked_result(case, abstraction)

    z3 = importlib.import_module("z3")
    optimizer = z3.Optimize()
    xs = [z3.Real(f"exp5242_{case.case_id}_x_{index}") for index in range(abstraction.input_dimension)]
    ys = [
        z3.Real(f"exp5242_{case.case_id}_unit_energy_{index}")
        for index in range(abstraction.input_dimension)
    ]
    total_energy = z3.Real(f"exp5242_{case.case_id}_total_energy")
    selected_flag_groups: list[list[Any]] = []
    constraint_count = 0
    big_m = _real(z3, 10.0)

    def add_constraints(*constraints: Any) -> None:
        nonlocal constraint_count
        optimizer.add(*constraints)
        constraint_count += len(constraints)

    for unit_index, unit in enumerate(abstraction.units):
        x = xs[unit_index]
        y = ys[unit_index]
        lo, hi = case.input_box[unit_index]
        flags = [
            z3.Int(f"exp5242_{case.case_id}_unit_{unit_index}_segment_{segment.index}")
            for segment in unit.segments
        ]
        selected_flag_groups.append(flags)
        add_constraints(x >= _real(z3, lo), x <= _real(z3, hi), z3.Sum(flags) == 1)
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
    if status != z3.sat:  # pragma: no cover - retained for honest unexpected solver states.
        return StressResult(
            case_id=case.case_id,
            stress_axis=case.stress_axis,
            verified=False,
            false_property_rejected=False,
            input_box=case.input_box,
            threshold=case.threshold,
            certified_upper_bound=None,
            certificate_slack=None,
            min_slope=bounded_min_slope(abstraction, case.input_box),
            solver_status=str(status),
            solve_time_s=solve_time_s,
            pwa_piece_count=abstraction.pwa_piece_count,
            binary_variable_count=abstraction.binary_variable_count,
            constraint_count=constraint_count,
            selected_segments=None,
            witness_inputs=None,
            numerical_instability="not_evaluated_solver_blocked",
        )

    model = optimizer.model()
    upper_bound = _z3_float(objective.value())
    witness_inputs = tuple(_z3_float(model.eval(x, model_completion=True)) for x in xs)
    selected_segments = tuple(
        next(
            segment_index
            for segment_index, flag in enumerate(flags)
            if _z3_float(model.eval(flag, model_completion=True)) > 0.5
        )
        for flags in selected_flag_groups
    )
    slack = case.threshold - upper_bound
    verified = slack >= -1e-12
    false_property_rejected = case.is_false_property and not verified
    return StressResult(
        case_id=case.case_id,
        stress_axis=case.stress_axis,
        verified=verified,
        false_property_rejected=false_property_rejected,
        input_box=case.input_box,
        threshold=case.threshold,
        certified_upper_bound=upper_bound,
        certificate_slack=slack,
        min_slope=bounded_min_slope(abstraction, case.input_box),
        solver_status="optimal",
        solve_time_s=solve_time_s,
        pwa_piece_count=abstraction.pwa_piece_count,
        binary_variable_count=abstraction.binary_variable_count,
        constraint_count=constraint_count,
        selected_segments=selected_segments,
        witness_inputs=witness_inputs,
        numerical_instability=_numerical_instability(slack, upper_bound),
    )


def run_stress_cases(*, solver_name: str | None = None) -> tuple[StressResult, ...]:
    """Run the bounded Exp 5242 stress cases in a deterministic order."""

    abstraction = build_stress_abstraction()
    return tuple(solve_stress_case(case, abstraction, solver_name=solver_name) for case in stress_cases())


def summarize_boundary(
    baseline: BaselineReproduction,
    rows: Sequence[StressResult],
) -> JsonDict:
    """Summarize the verified scale boundary without extrapolating beyond the rows."""

    verified_true_rows = [row for row in rows if row.verified and not row.false_property_rejected]
    positive_slacks = [
        row.certificate_slack
        for row in verified_true_rows
        if row.certificate_slack is not None and row.certificate_slack > 0.0
    ]
    false_rejected = any(row.false_property_rejected for row in rows)
    extended = baseline.reproduced and bool(verified_true_rows) and false_rejected
    return {
        "kan_certificate_extended": extended,
        "stress_axes": [case.stress_axis for case in stress_cases()],
        "max_pwa_segments_verified": max(
            (row.pwa_piece_count for row in verified_true_rows),
            default=0,
        ),
        "false_property_rejected": false_rejected,
        "certificate_slack_min": min(positive_slacks) if positive_slacks else None,
        "solve_time_s": round(sum(row.solve_time_s for row in rows), 6),
    }


def _honest_verdict(summary: Mapping[str, Any], blocked_reason: str | None) -> str:
    if summary["kan_certificate_extended"]:
        return (
            "success: bounded KAEM certificate extended to 10 total PWA segments "
            "over two variables, a wider [-0.5, 0.75] input box, and a rejected "
            "false property; no hardware or broad KAN verification claim"
        )
    return f"complete: bounded KAEM certificate extension blocked; {blocked_reason}"


def _checksum_payload(
    baseline: BaselineReproduction,
    rows: Sequence[StressResult],
    summary: Mapping[str, Any],
) -> str:
    row_payload = []
    for row in rows:
        payload = row.as_serializable()
        payload["solve_time_s"] = "excluded"
        row_payload.append(payload)
    encoded = json.dumps(
        {
            "baseline": baseline.as_serializable(),
            "experiment_id": EXPERIMENT_ID,
            "rows": row_payload,
            "run_date": RUN_DATE,
            "spec_refs": SPEC_REFS,
            "summary": dict(summary),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    solver_name: str | None = None,
) -> JsonDict:
    """Build and validate the Exp 5242 bounded certificate boundary artifact."""

    start = time.perf_counter()
    baseline = reproduce_exp5230_baseline()
    rows = run_stress_cases(solver_name=solver_name)
    summary = summarize_boundary(baseline, rows)
    blocked_reason = None if summary["kan_certificate_extended"] else "blocked_kan_pwa_milp_solver_unavailable"
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "kan_certificate_baseline_reproduced": wrap_field(
            "kan_certificate_baseline_reproduced",
            baseline.reproduced,
        ),
        "kan_certificate_extended": wrap_field(
            "kan_certificate_extended",
            summary["kan_certificate_extended"],
        ),
        "stress_axes": wrap_field("stress_axes", summary["stress_axes"]),
        "max_pwa_segments_verified": wrap_field(
            "max_pwa_segments_verified",
            summary["max_pwa_segments_verified"],
        ),
        "false_property_rejected": wrap_field(
            "false_property_rejected",
            summary["false_property_rejected"],
        ),
        "certificate_slack_min": wrap_field(
            "certificate_slack_min",
            summary["certificate_slack_min"],
        ),
        "solve_time_s": wrap_field("solve_time_s", summary["solve_time_s"]),
        "tests_run": wrap_field("tests_run", list(tests_run or [])),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": wrap_field("honest_verdict", _honest_verdict(summary, blocked_reason)),
        "blocked_reason": blocked_reason,
        "baseline": baseline.as_serializable(),
        "stress_results": [row.as_serializable() for row in rows],
        "target_module": TARGET_MODULE,
        "source_artifacts": [str(BASELINE_RELATIVE_PATH)],
        "claim_limits": [
            "no hardware readiness or execution claim",
            "no analog KAN speedup claim",
            "no broad KAN verification claim",
            "no trained-network soundness claim",
            "bounded deterministic KAEM PWA/MILP fixture only",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(baseline, rows, summary)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5242 artifact drifts from the bounded certificate contract."""

    for field in REQUIRED_WRAPPED_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    baseline_reproduced = artifact["kan_certificate_baseline_reproduced"]["value"]
    extended = artifact["kan_certificate_extended"]["value"]
    max_segments = artifact["max_pwa_segments_verified"]["value"]
    false_rejected = artifact["false_property_rejected"]["value"]
    slack_min = artifact["certificate_slack_min"]["value"]
    solve_time_s = artifact["solve_time_s"]["value"]
    substrate = artifact["inference_substrate"]["value"]
    verdict = artifact["honest_verdict"]["value"]

    _require(baseline_reproduced is True, "baseline must reproduce before stress boundary")
    _require(substrate == INFERENCE_SUBSTRATE, "inference_substrate must be deterministic_kan_pwa_milp_certificate")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require(isinstance(solve_time_s, float), "solve_time_s must be float")
    _require(solve_time_s >= 0.0, "solve_time_s must be nonnegative")
    _require("broad KAN verification" in " ".join(artifact["claim_limits"]), "must limit broad KAN claim")
    _require("hardware" in " ".join(artifact["claim_limits"]), "must limit hardware claim")

    if extended:
        _require(artifact["blocked_reason"] is None, "extended artifact cannot have blocked_reason")
        _require(max_segments >= 10, "extended boundary must verify the ten-piece stress fixture")
        _require(false_rejected is True, "extended boundary requires false-property rejection")
        _require(slack_min is not None and slack_min > 0.0, "extended boundary requires positive slack")
    else:
        _require(artifact["blocked_reason"] is not None, "blocked artifact must explain blocker")
        _require(max_segments == 0, "blocked artifact cannot claim verified stress scale")
        _require(false_rejected is False, "blocked artifact cannot claim false-property rejection")
        _require(slack_min is None, "blocked artifact cannot claim certificate slack")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    solver_name: str | None = None,
) -> JsonDict:
    """Write the Exp 5242 JSON artifact and return the validated payload."""

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
