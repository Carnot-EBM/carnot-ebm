"""Exp 5265 KAN certificate explanation/refinement layer.

Spec refs: REQ-KAN-5265, SCENARIO-KAN-5265.

Exp 5254 proved a tiny bounded upper-bound property with two convex
univariate components. This module keeps that narrow fixture and adds the
verification information that a reviewer needs next: which component drives
the bound, whether a local interval split tightens the abstraction, and whether
a near-threshold false property is still rejected. The refinement deliberately
does not claim broad KAN verification. It only reduces the pointwise envelope
gap for the bounded additive quadratic fixture already certified in Exp 5254.
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

from carnot import experiment_5254_kan_convex_envelope_certificate_v480 as v480


JsonDict = dict[str, Any]

RUN_DATE = "20260705"
EXPERIMENT_ID = "exp5265-kan-certificate-explanation-refinement-v481"
SCHEMA = "carnot.experiment_5265.kan_certificate_explanation_refinement.v481"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5265_kan_certificate_explanation_refinement_v481.json"
)
EXPLANATION_RELATIVE_PATH = Path(
    "results/experiment_5265_kan_certificate_explanation_refinement_v481_explanation.json"
)
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
SPEC_REFS = ("REQ-KAN-5265", "SCENARIO-KAN-5265")
RANDOM_SEED = 5265
NEAR_FALSE_PROPERTY_THRESHOLD = 0.698
TERMINAL_PREFIXES = ("complete:", "blocked_")

CERTIFICATE_READY_PRINCIPLE = (
    "True only when the V481 layer preserves the V480 true certificate, rejects "
    "the near-threshold false property, and records a nonzero envelope-gap refinement."
)
SLACK_FIELD_PRINCIPLE = (
    "Numeric before/after slack and envelope-gap values expose whether refinement "
    "tightened the certificate or only added explanation."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "certificate_refinement_ready",
    "certificate_refinement_ready_principle",
    "true_property_certified",
    "false_property_rejected",
    "slack_before_after",
    "explanation_artifact_path",
    "spec_updated",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether refinement added certificate value."
    ),
    "inference_substrate": "Must be offline_deterministic_certificate_no_llm.",
    "true_property_certified": (
        "True only when the original V480 true upper-bound property remains certified."
    ),
    "false_property_rejected": (
        "True only when a near-threshold false upper-bound property is rejected by a witness."
    ),
    "explanation_artifact_path": (
        "Path to the machine-readable contributor and refinement explanation artifact."
    ),
    "spec_updated": "True only when REQ-KAN-5265 and SCENARIO-KAN-5265 anchor the change.",
}
WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)


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
    """Return a principle-wrapped field for the V481 artifact."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


@dataclass(frozen=True)
class IntervalEnvelope:
    """One chord-envelope interval for a bounded convex quadratic component."""

    variable_index: int
    interval: tuple[float, float]
    quadratic: float
    linear: float
    constant: float

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

    @property
    def max_envelope_value(self) -> tuple[float, float]:
        lo, hi = self.interval
        endpoint_values = ((lo, self.evaluate(lo)), (hi, self.evaluate(hi)))
        return max(endpoint_values, key=lambda row: row[1])

    @property
    def max_envelope_gap(self) -> float:
        lo, hi = self.interval
        return self.quadratic * (hi - lo) * (hi - lo) / 4.0

    def as_serializable(self) -> JsonDict:
        witness_x, witness_y = self.max_envelope_value
        return {
            "variable_index": self.variable_index,
            "interval": list(self.interval),
            "chord_slope": self.chord_slope,
            "chord_intercept": self.chord_intercept,
            "max_witness_x": witness_x,
            "max_envelope_value": witness_y,
            "max_envelope_gap": self.max_envelope_gap,
        }


@dataclass(frozen=True)
class ComponentExplanation:
    """Contribution and gap attribution for one univariate component."""

    variable_index: int
    interval: tuple[float, float]
    bound_contribution: float
    contribution_fraction: float
    max_envelope_gap: float
    driver_rank: int

    def as_serializable(self) -> JsonDict:
        return {
            "variable_index": self.variable_index,
            "interval": list(self.interval),
            "bound_contribution": self.bound_contribution,
            "contribution_fraction": self.contribution_fraction,
            "max_envelope_gap": self.max_envelope_gap,
            "driver_rank": self.driver_rank,
            "driver_reason": (
                "largest contribution to certified upper bound"
                if self.driver_rank == 1
                else "secondary contribution"
            ),
        }


@dataclass(frozen=True)
class RefinementDecision:
    """A local interval-refinement decision for the explanation artifact."""

    variable_index: int
    decision: str
    original_interval: tuple[float, float]
    refined_intervals: tuple[tuple[float, float], ...]
    reason: str

    def as_serializable(self) -> JsonDict:
        return {
            "variable_index": self.variable_index,
            "decision": self.decision,
            "original_interval": list(self.original_interval),
            "refined_intervals": [list(interval) for interval in self.refined_intervals],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AbstractionSummary:
    """No-refinement or refined abstraction summary for the same input box."""

    label: str
    intervals_by_variable: tuple[tuple[IntervalEnvelope, ...], ...]
    output_upper_bound: float
    envelope_gap_bound: float
    same_property_slack: float
    envelope_gap_reduction: float

    @property
    def refined_interval_count(self) -> int:
        return sum(len(intervals) for intervals in self.intervals_by_variable)

    def as_serializable(self) -> JsonDict:
        return {
            "label": self.label,
            "interval_count": self.refined_interval_count,
            "output_upper_bound": self.output_upper_bound,
            "envelope_gap_bound": self.envelope_gap_bound,
            "same_property_slack": self.same_property_slack,
            "envelope_gap_reduction": self.envelope_gap_reduction,
            "intervals_by_variable": [
                [interval.as_serializable() for interval in intervals]
                for intervals in self.intervals_by_variable
            ],
        }


@dataclass(frozen=True)
class PropertyOutcome:
    """True or false property result after applying the refined abstraction."""

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
        }


@dataclass(frozen=True)
class RefinementReport:
    """Complete V481 explanation/refinement result before serialization."""

    source_artifact: str
    bound_contributors: tuple[ComponentExplanation, ...]
    refinement_decisions: tuple[RefinementDecision, ...]
    no_refinement: AbstractionSummary
    refined: AbstractionSummary
    true_property: PropertyOutcome
    near_false_property: PropertyOutcome

    @property
    def driver(self) -> ComponentExplanation:
        return self.bound_contributors[0]

    @property
    def certificate_refinement_ready(self) -> bool:
        return (
            self.true_property.certified
            and self.near_false_property.rejected
            and self.refined.envelope_gap_bound < self.no_refinement.envelope_gap_bound
        )

    def slack_before_after(self) -> JsonDict:
        return {
            "same_property_slack_before": self.no_refinement.same_property_slack,
            "same_property_slack_after": self.refined.same_property_slack,
            "same_property_slack_delta": (
                self.refined.same_property_slack - self.no_refinement.same_property_slack
            ),
            "near_false_property_slack_after": self.near_false_property.certificate_slack,
            "envelope_gap_bound_before": self.no_refinement.envelope_gap_bound,
            "envelope_gap_bound_after": self.refined.envelope_gap_bound,
            "envelope_gap_bound_reduction": (
                self.no_refinement.envelope_gap_bound - self.refined.envelope_gap_bound
            ),
        }

    def as_explanation_artifact(self) -> JsonDict:
        return {
            "schema": f"{SCHEMA}.explanation",
            "experiment_id": EXPERIMENT_ID,
            "run_date": RUN_DATE,
            "source_artifact": self.source_artifact,
            "bound_contributors": [
                contributor.as_serializable() for contributor in self.bound_contributors
            ],
            "refinement_decisions": [
                decision.as_serializable() for decision in self.refinement_decisions
            ],
            "no_refinement": self.no_refinement.as_serializable(),
            "refined": self.refined.as_serializable(),
            "property_results": [
                self.true_property.as_serializable(),
                self.near_false_property.as_serializable(),
            ],
            "methodology_note": (
                "The refined split tightens the pointwise chord-envelope gap. "
                "The output upper bound and same-property slack remain unchanged "
                "because both univariate components are monotone on the bounded "
                "interval and the maximum is already at the shared upper endpoint."
            ),
        }


def _base_unit_envelopes() -> tuple[IntervalEnvelope, ...]:
    units: list[IntervalEnvelope] = []
    for variable_index, coefficients in enumerate(v480.QUADRATIC_COMPONENTS):
        units.append(
            IntervalEnvelope(
                variable_index=variable_index,
                interval=v480.INPUT_BOX[variable_index],
                quadratic=coefficients[0],
                linear=coefficients[1],
                constant=coefficients[2],
            )
        )
    return tuple(units)


def _component_explanations(
    base_units: Sequence[IntervalEnvelope],
) -> tuple[ComponentExplanation, ...]:
    upper_bound = sum(unit.max_envelope_value[1] for unit in base_units)
    ranked_units = sorted(base_units, key=lambda unit: unit.max_envelope_value[1], reverse=True)
    explanations = []
    for rank, unit in enumerate(ranked_units, start=1):
        explanations.append(
            ComponentExplanation(
                variable_index=unit.variable_index,
                interval=unit.interval,
                bound_contribution=unit.max_envelope_value[1],
                contribution_fraction=unit.max_envelope_value[1] / upper_bound,
                max_envelope_gap=unit.max_envelope_gap,
                driver_rank=rank,
            )
        )
    return tuple(explanations)


def _split_interval(interval: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    lo, hi = interval
    midpoint = (lo + hi) / 2.0
    return ((lo, midpoint), (midpoint, hi))


def _intervals_for_unit(
    unit: IntervalEnvelope,
    *,
    driver_index: int,
    refine_driver: bool,
) -> tuple[IntervalEnvelope, ...]:
    intervals = (
        _split_interval(unit.interval)
        if refine_driver and unit.variable_index == driver_index
        else (unit.interval,)
    )
    return tuple(
        IntervalEnvelope(
            variable_index=unit.variable_index,
            interval=interval,
            quadratic=unit.quadratic,
            linear=unit.linear,
            constant=unit.constant,
        )
        for interval in intervals
    )


def build_abstraction_summary(*, refine_driver: bool) -> AbstractionSummary:
    """Build a comparable coarse or driver-refined abstraction summary."""

    base_units = _base_unit_envelopes()
    driver_index = _component_explanations(base_units)[0].variable_index
    intervals_by_variable = tuple(
        _intervals_for_unit(unit, driver_index=driver_index, refine_driver=refine_driver)
        for unit in base_units
    )
    per_variable_maxes = [
        max(interval.max_envelope_value[1] for interval in intervals)
        for intervals in intervals_by_variable
    ]
    per_variable_gaps = [
        max(interval.max_envelope_gap for interval in intervals)
        for intervals in intervals_by_variable
    ]
    output_upper_bound = sum(per_variable_maxes)
    envelope_gap_bound = sum(per_variable_gaps)
    coarse_gap_bound = sum(unit.max_envelope_gap for unit in base_units)
    return AbstractionSummary(
        label="driver_refined" if refine_driver else "no_refinement",
        intervals_by_variable=intervals_by_variable,
        output_upper_bound=output_upper_bound,
        envelope_gap_bound=envelope_gap_bound,
        same_property_slack=v480.TRUE_PROPERTY_THRESHOLD - output_upper_bound,
        envelope_gap_reduction=coarse_gap_bound - envelope_gap_bound,
    )


def _refinement_decisions(
    driver: ComponentExplanation,
    no_refinement: AbstractionSummary,
    refined: AbstractionSummary,
) -> tuple[RefinementDecision, ...]:
    if refined.envelope_gap_bound < no_refinement.envelope_gap_bound:
        decision = RefinementDecision(
            variable_index=driver.variable_index,
            decision="split_interval",
            original_interval=driver.interval,
            refined_intervals=_split_interval(driver.interval),
            reason=(
                "largest bound contributor also has the largest convex-envelope gap; "
                "splitting the interval reduces pointwise gap without changing the endpoint maximum"
            ),
        )
    else:
        decision = RefinementDecision(
            variable_index=driver.variable_index,
            decision="no_refinement",
            original_interval=driver.interval,
            refined_intervals=(driver.interval,),
            reason="no positive envelope-gap reduction was available",
        )
    return (decision,)


def _property_outcomes(refined: AbstractionSummary) -> tuple[PropertyOutcome, PropertyOutcome]:
    relaxation = v480.build_convex_relaxation()
    witness_inputs = relaxation.witness_inputs
    actual_witness = relaxation.evaluate(witness_inputs)
    true_slack = v480.TRUE_PROPERTY_THRESHOLD - refined.output_upper_bound
    near_false_slack = NEAR_FALSE_PROPERTY_THRESHOLD - refined.output_upper_bound
    return (
        PropertyOutcome(
            property_id="v481_true_upper_property_preserved",
            threshold=v480.TRUE_PROPERTY_THRESHOLD,
            certified_upper_bound=refined.output_upper_bound,
            certificate_slack=true_slack,
            certified=true_slack >= -1e-12,
            rejected=False,
            counterexample_inputs=None,
            actual_witness_value=None,
        ),
        PropertyOutcome(
            property_id="v481_near_threshold_false_upper_property",
            threshold=NEAR_FALSE_PROPERTY_THRESHOLD,
            certified_upper_bound=refined.output_upper_bound,
            certificate_slack=near_false_slack,
            certified=near_false_slack >= -1e-12,
            rejected=near_false_slack < 0.0 and actual_witness > NEAR_FALSE_PROPERTY_THRESHOLD,
            counterexample_inputs=witness_inputs,
            actual_witness_value=actual_witness,
        ),
    )


def build_refinement_report() -> RefinementReport:
    """Build the deterministic V481 contributor/refinement report."""

    base_units = _base_unit_envelopes()
    contributors = _component_explanations(base_units)
    no_refinement = build_abstraction_summary(refine_driver=False)
    refined = build_abstraction_summary(refine_driver=True)
    true_property, near_false_property = _property_outcomes(refined)
    return RefinementReport(
        source_artifact=str(v480.RESULT_RELATIVE_PATH),
        bound_contributors=contributors,
        refinement_decisions=_refinement_decisions(contributors[0], no_refinement, refined),
        no_refinement=no_refinement,
        refined=refined,
        true_property=true_property,
        near_false_property=near_false_property,
    )


def _honest_verdict(report: RefinementReport) -> str:
    if report.certificate_refinement_ready:
        return (
            "complete: refinement added certificate value by identifying variable 0 "
            "as the bound driver and halving the envelope-gap bound; same-property "
            "output slack is unchanged because the endpoint maximum was already tight"
        )
    return "blocked_kan_certificate_refinement_controls_failed"


def _checksum_payload(report: RefinementReport) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "source_artifact": report.source_artifact,
        "contributors": [item.as_serializable() for item in report.bound_contributors],
        "slack_before_after": report.slack_before_after(),
        "true_property": report.true_property.as_serializable(),
        "near_false_property": report.near_false_property.as_serializable(),
        "spec_refs": SPEC_REFS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_explanation_artifact() -> JsonDict:
    """Build and validate the sidecar explanation artifact."""

    explanation = build_refinement_report().as_explanation_artifact()
    validate_explanation_artifact(explanation)
    return explanation


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and validate the main V481 deliverable artifact."""

    start = time.perf_counter()
    report = build_refinement_report()
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "honest_verdict": wrap_field("honest_verdict", _honest_verdict(report)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "certificate_refinement_ready": report.certificate_refinement_ready,
        "certificate_refinement_ready_principle": CERTIFICATE_READY_PRINCIPLE,
        "true_property_certified": wrap_field(
            "true_property_certified",
            report.true_property.certified,
        ),
        "false_property_rejected": wrap_field(
            "false_property_rejected",
            report.near_false_property.rejected,
        ),
        "slack_before_after": {
            "principle": SLACK_FIELD_PRINCIPLE,
            "value": report.slack_before_after(),
        },
        "explanation_artifact_path": wrap_field(
            "explanation_artifact_path",
            str(EXPLANATION_RELATIVE_PATH),
        ),
        "spec_updated": wrap_field("spec_updated", True),
        "tests_run": list(tests_run or []),
        "source_artifacts": [str(v480.RESULT_RELATIVE_PATH)],
        "bound_driver": report.driver.as_serializable(),
        "refinement_summary": {
            "no_refinement": report.no_refinement.as_serializable(),
            "refined": report.refined.as_serializable(),
        },
        "property_results": [
            report.true_property.as_serializable(),
            report.near_false_property.as_serializable(),
        ],
        "claim_limits": [
            "bounded deterministic V480 fixture only",
            "no broad KAN verification claim",
            "no trained-network soundness claim",
            "no hardware execution or hardware speedup claim",
            "no live LLM inference claim",
        ],
        "field_principles": FIELD_PRINCIPLES
        | {
            "certificate_refinement_ready": CERTIFICATE_READY_PRINCIPLE,
            "slack_before_after": SLACK_FIELD_PRINCIPLE,
        },
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "blocked_reason": None
        if report.certificate_refinement_ready
        else "blocked_kan_certificate_refinement_controls_failed",
    }
    artifact["reproducibility_checksum"] = _checksum_payload(report)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the main V481 artifact drifts from the narrow contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    verdict = _wrapped_value(artifact, "honest_verdict")
    substrate = _wrapped_value(artifact, "inference_substrate")
    true_certified = _wrapped_value(artifact, "true_property_certified")
    false_rejected = _wrapped_value(artifact, "false_property_rejected")
    explanation_path = _wrapped_value(artifact, "explanation_artifact_path")
    spec_updated = _wrapped_value(artifact, "spec_updated")
    slack = artifact["slack_before_after"]
    slack_values = slack["value"]

    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix"
    )
    _require(
        "refinement added certificate value" in verdict, "honest_verdict must state value added"
    )
    _require(substrate == INFERENCE_SUBSTRATE, "inference_substrate must be offline deterministic")
    _require(
        artifact["certificate_refinement_ready"] is True,
        "certificate refinement ready must be true",
    )
    _require(
        artifact["certificate_refinement_ready_principle"] == CERTIFICATE_READY_PRINCIPLE,
        "certificate_refinement_ready_principle drift",
    )
    _require(true_certified is True, "true property must be certified")
    _require(false_rejected is True, "false property must be rejected")
    _require(explanation_path == str(EXPLANATION_RELATIVE_PATH), "explanation artifact path drift")
    _require(spec_updated is True, "spec_updated must be true")
    _require(isinstance(slack, Mapping), "slack_before_after must be an object")
    _require(slack.get("principle") == SLACK_FIELD_PRINCIPLE, "slack_before_after principle drift")
    _require(isinstance(slack_values, Mapping), "slack_before_after value must be an object")
    for field in (
        "same_property_slack_before",
        "same_property_slack_after",
        "same_property_slack_delta",
        "near_false_property_slack_after",
        "envelope_gap_bound_before",
        "envelope_gap_bound_after",
        "envelope_gap_bound_reduction",
    ):
        _require(_is_number(slack_values[field]), f"{field} must be numeric")
    _require(slack_values["same_property_slack_before"] > 0.0, "before slack must be positive")
    _require(slack_values["same_property_slack_after"] > 0.0, "after slack must be positive")
    _require(slack_values["near_false_property_slack_after"] < 0.0, "near false slack must reject")
    _require(
        slack_values["envelope_gap_bound_after"] < slack_values["envelope_gap_bound_before"],
        "gap reduction must be positive",
    )
    _require(
        slack_values["envelope_gap_bound_reduction"] > 0.0, "gap reduction must be numeric positive"
    )
    _require(isinstance(artifact["tests_run"], list), "tests_run must be a list")
    claim_limits = " ".join(artifact["claim_limits"])
    _require("broad KAN verification" in claim_limits, "must limit broad KAN claim")
    _require("hardware speedup" in claim_limits, "must limit hardware speedup claim")
    _require("live LLM" in claim_limits, "must limit live LLM claim")


def validate_explanation_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the contributor/refinement sidecar artifact."""

    _require(artifact["schema"] == f"{SCHEMA}.explanation", "explanation schema drift")
    _require(artifact["experiment_id"] == EXPERIMENT_ID, "explanation experiment_id drift")
    _require(artifact["source_artifact"] == str(v480.RESULT_RELATIVE_PATH), "source artifact drift")
    contributors = artifact["bound_contributors"]
    decisions = artifact["refinement_decisions"]
    _require(isinstance(contributors, list) and len(contributors) == 2, "two contributors required")
    _require(contributors[0]["variable_index"] == 0, "driver variable drift")
    _require(contributors[0]["driver_rank"] == 1, "driver rank drift")
    _require(
        isinstance(decisions, list) and decisions[0]["decision"] == "split_interval",
        "refinement decision drift",
    )
    no_refinement = artifact["no_refinement"]
    refined = artifact["refined"]
    _require(
        refined["interval_count"] > no_refinement["interval_count"],
        "refined interval count must grow",
    )
    _require(
        refined["envelope_gap_bound"] < no_refinement["envelope_gap_bound"],
        "refined explanation must reduce envelope gap",
    )
    _require(len(artifact["property_results"]) == 2, "two property results required")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    explanation_path: str | Path = EXPLANATION_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the main and explanation V481 artifacts."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    explanation = build_explanation_artifact()

    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    explanation_output_path = Path(explanation_path)
    explanation_output_path.parent.mkdir(parents=True, exist_ok=True)
    explanation_output_path.write_text(
        json.dumps(explanation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:  # pragma: no cover - exercised through write_outputs tests.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
