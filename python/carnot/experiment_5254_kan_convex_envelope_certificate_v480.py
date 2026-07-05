"""Exp 5254 bounded KAN convex-envelope certificate prototype.

Spec refs: REQ-KAN-5254, SCENARIO-KAN-5254.

This module is intentionally narrower than arXiv 2604.03871. It implements a
deterministic convex-envelope-style stress test for two additive univariate
quadratic components on the same bounded box used by Exp 5242. For each convex
quadratic, the chord over the bounded interval is an upper relaxation, so the
sum of the chords gives a conservative certificate for an output upper bound.
The result is a small offline certificate, not broad KAN verification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

RUN_DATE = "20260705"
EXPERIMENT_ID = "exp5254-kan-convex-envelope-certificate-v480"
SCHEMA = "carnot.experiment_5254.kan_convex_envelope_certificate.v480"
RESULT_RELATIVE_PATH = Path("results/experiment_5254_kan_convex_envelope_certificate_v480.json")
BASELINE_RELATIVE_PATH = Path("results/experiment_5242_kan_certificate_abstraction_scale_v479.json")
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
CERTIFICATE_METHOD = "convex_chord_upper_relaxation_for_bounded_univariate_quadratics"
SPEC_REFS = ("REQ-KAN-5254", "SCENARIO-KAN-5254")
RANDOM_SEED = 5254
TERMINAL_PREFIXES = ("complete:", "blocked_")

INPUT_BOX = ((-0.5, 0.75), (-0.5, 0.75))
TRUE_PROPERTY_THRESHOLD = 0.72
FALSE_PROPERTY_THRESHOLD = 0.68

QUADRATIC_COMPONENTS = (
    (0.20, 0.25, 0.15),
    (0.10, 0.15, 0.08),
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state the bounded certificate scope."
    ),
    "inference_substrate": "Must be offline_deterministic_certificate_no_llm.",
    "certificate_method": (
        "Names the narrow deterministic convex-relaxation method actually used."
    ),
    "variables_verified": "Integer count of input variables covered by the bounded certificate.",
    "max_segments_or_envelopes_verified": (
        "Integer count of convex envelopes verified by this method, not a broad KAN scale claim."
    ),
    "input_box": "Bounded input intervals used for both the baseline comparison and certificate.",
    "true_property_certified": (
        "True only when the expected-true upper-bound property is certified by the relaxation."
    ),
    "false_property_rejected": (
        "True only when the deliberate false threshold is rejected with a deterministic witness."
    ),
    "certificate_slack_min": (
        "Minimum positive slack among certified expected-true properties."
    ),
    "solve_time_s": "Measured offline deterministic certificate time in seconds.",
    "no_hardware_speedup_claim": (
        "True only when the artifact makes no hardware execution or speedup claim."
    ),
}
REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box_as_lists(input_box: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [list(bounds) for bounds in input_box]


def _box_as_tuples(input_box: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(bounds[0]), float(bounds[1])) for bounds in input_box)


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact[field]
    _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
    _require("value" in wrapped, f"{field} missing value")
    return wrapped["value"]


def wrap_field(field: str, value: Any) -> JsonDict:
    """Return a principle-wrapped artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


@dataclass(frozen=True)
class BaselineComparison:
    """Exp 5242 PWA boundary dimensions used only as a comparator."""

    reproduced: bool
    variables_verified: int
    max_pwa_segments_verified: int
    input_box: tuple[tuple[float, float], ...]
    certificate_slack_min: float
    solve_time_s: float
    false_property_rejected: bool
    source_artifact: str

    def as_serializable(self) -> JsonDict:
        return {
            "reproduced": self.reproduced,
            "variables_verified": self.variables_verified,
            "max_pwa_segments_verified": self.max_pwa_segments_verified,
            "input_box": _box_as_lists(self.input_box),
            "certificate_slack_min": self.certificate_slack_min,
            "solve_time_s": self.solve_time_s,
            "false_property_rejected": self.false_property_rejected,
            "source_artifact": self.source_artifact,
        }


@dataclass(frozen=True)
class ConvexQuadraticUnit:
    """One bounded convex polynomial component with its chord upper envelope."""

    variable_index: int
    quadratic: float
    linear: float
    constant: float
    interval: tuple[float, float]

    def __post_init__(self) -> None:
        _require(self.quadratic >= 0.0, "quadratic coefficient must be convex")
        _require(self.interval[0] < self.interval[1], "interval must be nonempty")

    def evaluate(self, x_value: float) -> float:
        return self.quadratic * x_value * x_value + self.linear * x_value + self.constant

    @property
    def chord_slope(self) -> float:
        lo, hi = self.interval
        return (self.evaluate(hi) - self.evaluate(lo)) / (hi - lo)

    @property
    def chord_intercept(self) -> float:
        lo, _ = self.interval
        return self.evaluate(lo) - self.chord_slope * lo

    def envelope_value(self, x_value: float) -> float:
        return self.chord_slope * x_value + self.chord_intercept

    @property
    def max_envelope_value(self) -> tuple[float, float]:
        lo, hi = self.interval
        endpoint_values = ((lo, self.envelope_value(lo)), (hi, self.envelope_value(hi)))
        return max(endpoint_values, key=lambda row: row[1])

    def as_serializable(self) -> JsonDict:
        witness_x, witness_y = self.max_envelope_value
        return {
            "variable_index": self.variable_index,
            "polynomial": {
                "quadratic": self.quadratic,
                "linear": self.linear,
                "constant": self.constant,
            },
            "interval": list(self.interval),
            "chord_upper_envelope": {
                "slope": self.chord_slope,
                "intercept": self.chord_intercept,
                "max_witness_x": witness_x,
                "max_envelope_value": witness_y,
            },
        }


@dataclass(frozen=True)
class ConvexRelaxation:
    """Small additive convex relaxation for the bounded Exp 5254 fixture."""

    units: tuple[ConvexQuadraticUnit, ...]

    @property
    def variable_count(self) -> int:
        return len(self.units)

    @property
    def envelope_count(self) -> int:
        return len(self.units)

    @property
    def input_box(self) -> tuple[tuple[float, float], ...]:
        return tuple(unit.interval for unit in self.units)

    @property
    def witness_inputs(self) -> tuple[float, ...]:
        return tuple(unit.max_envelope_value[0] for unit in self.units)

    def evaluate(self, inputs: Sequence[float]) -> float:
        return sum(unit.evaluate(float(x_value)) for unit, x_value in zip(self.units, inputs))

    @property
    def envelope_upper_bound(self) -> float:
        return sum(unit.max_envelope_value[1] for unit in self.units)

    def as_serializable(self) -> JsonDict:
        return {
            "method": CERTIFICATE_METHOD,
            "variable_count": self.variable_count,
            "envelope_count": self.envelope_count,
            "input_box": _box_as_lists(self.input_box),
            "units": [unit.as_serializable() for unit in self.units],
            "envelope_upper_bound": self.envelope_upper_bound,
            "witness_inputs": list(self.witness_inputs),
            "scope": (
                "bounded two-variable additive polynomial fixture; chord upper "
                "relaxations only, not full nonlinear KAN global verification"
            ),
        }


@dataclass(frozen=True)
class PropertyResult:
    """One true or false bounded property outcome for the convex relaxation."""

    property_id: str
    threshold: float
    certified_upper_bound: float
    certificate_slack: float
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
            "method": CERTIFICATE_METHOD,
        }


@dataclass(frozen=True)
class CertificateChecks:
    """Complete Exp 5254 deterministic certificate result before serialization."""

    baseline: BaselineComparison
    relaxation: ConvexRelaxation
    true_property: PropertyResult
    false_property: PropertyResult
    solve_time_s: float

    @property
    def true_property_certified(self) -> bool:
        return self.true_property.certified

    @property
    def false_property_rejected(self) -> bool:
        return self.false_property.rejected

    @property
    def certificate_slack_min(self) -> float:
        return self.true_property.certificate_slack


def load_exp5242_baseline(
    baseline_path: str | Path = BASELINE_RELATIVE_PATH,
) -> BaselineComparison:
    """Load and validate the committed Exp 5242 PWA baseline dimensions."""

    artifact = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    _require(
        artifact["experiment_id"] == "exp5242-kan-certificate-abstraction-scale-v479",
        "baseline experiment_id drift",
    )
    _require(
        _wrapped_value(artifact, "inference_substrate") == "deterministic_kan_pwa_milp_certificate",
        "baseline substrate drift",
    )
    max_segments = int(_wrapped_value(artifact, "max_pwa_segments_verified"))
    false_rejected = bool(_wrapped_value(artifact, "false_property_rejected"))
    slack = float(_wrapped_value(artifact, "certificate_slack_min"))
    solve_time = float(_wrapped_value(artifact, "solve_time_s"))
    baseline_dims = artifact["baseline"]["dimensions"]
    input_box = _box_as_tuples(
        next(
            row["input_box"]
            for row in artifact["stress_results"]
            if row["case_id"] == "wider_input_bounds"
        )
    )

    _require(max_segments >= 10, "baseline must expose the ten-segment PWA stress boundary")
    _require(false_rejected is True, "baseline must reject the false property")
    _require(input_box == INPUT_BOX, "baseline input box must match Exp 5254 comparison box")
    return BaselineComparison(
        reproduced=True,
        variables_verified=int(baseline_dims["input_dimension"]),
        max_pwa_segments_verified=max_segments,
        input_box=input_box,
        certificate_slack_min=slack,
        solve_time_s=solve_time,
        false_property_rejected=false_rejected,
        source_artifact=str(baseline_path),
    )


def build_convex_relaxation() -> ConvexRelaxation:
    """Build the deterministic two-variable convex-envelope prototype."""

    units = tuple(
        ConvexQuadraticUnit(
            variable_index=index,
            quadratic=coefficients[0],
            linear=coefficients[1],
            constant=coefficients[2],
            interval=INPUT_BOX[index],
        )
        for index, coefficients in enumerate(QUADRATIC_COMPONENTS)
    )
    return ConvexRelaxation(units=units)


def run_certificate_checks() -> CertificateChecks:
    """Evaluate the true and false bounded properties through the relaxation."""

    start = time.perf_counter()
    baseline = load_exp5242_baseline()
    relaxation = build_convex_relaxation()
    upper_bound = relaxation.envelope_upper_bound
    witness_inputs = relaxation.witness_inputs
    actual_witness = relaxation.evaluate(witness_inputs)

    true_slack = TRUE_PROPERTY_THRESHOLD - upper_bound
    true_property = PropertyResult(
        property_id="bounded_true_upper_property",
        threshold=TRUE_PROPERTY_THRESHOLD,
        certified_upper_bound=upper_bound,
        certificate_slack=true_slack,
        certified=true_slack >= -1e-12,
        rejected=False,
        counterexample_inputs=None,
        actual_witness_value=None,
    )
    false_slack = FALSE_PROPERTY_THRESHOLD - upper_bound
    false_property = PropertyResult(
        property_id="deliberate_false_upper_property",
        threshold=FALSE_PROPERTY_THRESHOLD,
        certified_upper_bound=upper_bound,
        certificate_slack=false_slack,
        certified=false_slack >= -1e-12,
        rejected=false_slack < 0.0 and actual_witness > FALSE_PROPERTY_THRESHOLD,
        counterexample_inputs=witness_inputs,
        actual_witness_value=actual_witness,
    )
    return CertificateChecks(
        baseline=baseline,
        relaxation=relaxation,
        true_property=true_property,
        false_property=false_property,
        solve_time_s=round(time.perf_counter() - start, 6),
    )


def _honest_verdict(result: CertificateChecks) -> str:
    if result.true_property_certified and result.false_property_rejected:
        return (
            "complete: bounded two-variable convex-envelope certificate prototype "
            "certified one true upper-bound property and rejected one false threshold; "
            "scope is additive quadratic univariate components on [-0.5, 0.75]^2 only"
        )
    return "blocked_kan_convex_envelope_certificate_controls_failed"


def _checksum_payload(result: CertificateChecks) -> str:
    payload = {
        "baseline": result.baseline.as_serializable(),
        "certificate_method": CERTIFICATE_METHOD,
        "experiment_id": EXPERIMENT_ID,
        "properties": [
            result.true_property.as_serializable(),
            result.false_property.as_serializable(),
        ],
        "relaxation": result.relaxation.as_serializable(),
        "run_date": RUN_DATE,
        "spec_refs": SPEC_REFS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and validate the Exp 5254 bounded convex-envelope artifact."""

    start = time.perf_counter()
    result = run_certificate_checks()
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "honest_verdict": wrap_field("honest_verdict", _honest_verdict(result)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "certificate_method": wrap_field("certificate_method", CERTIFICATE_METHOD),
        "variables_verified": wrap_field("variables_verified", result.relaxation.variable_count),
        "max_segments_or_envelopes_verified": wrap_field(
            "max_segments_or_envelopes_verified",
            result.relaxation.envelope_count,
        ),
        "input_box": wrap_field("input_box", _box_as_lists(result.relaxation.input_box)),
        "true_property_certified": wrap_field(
            "true_property_certified",
            result.true_property_certified,
        ),
        "false_property_rejected": wrap_field(
            "false_property_rejected",
            result.false_property_rejected,
        ),
        "certificate_slack_min": wrap_field(
            "certificate_slack_min",
            result.certificate_slack_min,
        ),
        "solve_time_s": wrap_field("solve_time_s", result.solve_time_s),
        "no_hardware_speedup_claim": wrap_field("no_hardware_speedup_claim", True),
        "baseline_comparison": result.baseline.as_serializable(),
        "convex_relaxation": result.relaxation.as_serializable(),
        "property_results": [
            result.true_property.as_serializable(),
            result.false_property.as_serializable(),
        ],
        "source_artifacts": [str(BASELINE_RELATIVE_PATH)],
        "claim_limits": [
            "bounded deterministic convex-relaxation prototype only",
            "no broad KAN verification claim",
            "no nonlinear global proof coverage claim",
            "no trained-network soundness claim",
            "no hardware execution or hardware speedup claim",
            "no live LLM inference claim",
        ],
        "tests_run": list(tests_run or []),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "blocked_reason": None
        if result.true_property_certified and result.false_property_rejected
        else "blocked_kan_convex_envelope_certificate_controls_failed",
    }
    artifact["reproducibility_checksum"] = _checksum_payload(result)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5254 artifact drifts from its narrow contract."""

    for field in REQUIRED_WRAPPED_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    verdict = _wrapped_value(artifact, "honest_verdict")
    substrate = _wrapped_value(artifact, "inference_substrate")
    method = _wrapped_value(artifact, "certificate_method")
    variables = _wrapped_value(artifact, "variables_verified")
    envelopes = _wrapped_value(artifact, "max_segments_or_envelopes_verified")
    input_box = _wrapped_value(artifact, "input_box")
    true_certified = _wrapped_value(artifact, "true_property_certified")
    false_rejected = _wrapped_value(artifact, "false_property_rejected")
    slack = _wrapped_value(artifact, "certificate_slack_min")
    solve_time = _wrapped_value(artifact, "solve_time_s")
    no_hardware_speedup = _wrapped_value(artifact, "no_hardware_speedup_claim")

    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require("bounded" in verdict, "honest_verdict must state bounded scope")
    _require(substrate == INFERENCE_SUBSTRATE, "inference_substrate must be offline deterministic")
    _require(method == CERTIFICATE_METHOD and "convex" in method, "certificate_method drift")
    _require(isinstance(variables, int) and variables == 2, "variables_verified must be integer 2")
    _require(isinstance(envelopes, int) and envelopes == 2, "max_segments_or_envelopes_verified must be integer 2")
    _require(input_box == _box_as_lists(INPUT_BOX), "input_box drift")
    _require(true_certified is True, "true property must be certified")
    _require(false_rejected is True, "false property must be rejected")
    _require(_is_number(slack) and slack > 0.0, "certificate_slack_min must be positive numeric")
    _require(_is_number(solve_time) and solve_time >= 0.0, "solve_time_s must be nonnegative numeric")
    _require(no_hardware_speedup is True, "no_hardware_speedup_claim must preserve no hardware speedup claim")

    baseline = artifact["baseline_comparison"]
    _require(baseline["max_pwa_segments_verified"] >= 10, "baseline comparison must record Exp 5242 segments")
    _require(baseline["variables_verified"] == 2, "baseline comparison must record two variables")
    claim_limits = " ".join(artifact["claim_limits"])
    _require("broad KAN verification" in claim_limits, "must limit broad KAN claim")
    _require("hardware speedup" in claim_limits, "must limit hardware speedup claim")
    _require("nonlinear global proof" in claim_limits, "must limit nonlinear global proof claim")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5254 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
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
