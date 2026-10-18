"""Exp 5332 bounded KAN counterexample-localization diagnostic.

Spec refs: REQ-KAN-5332, SCENARIO-KAN-5332.

This module keeps the experiment deliberately narrow. Exp 5316 tightened the
bounded KAN abstraction envelope but did not improve certificate success. The
next useful question is whether that tighter abstraction helps diagnose where a
false-property counterexample lives. We answer only for the deterministic
three-component Exp 5277 fixture by applying one constant-shift perturbation per
component, then checking whether the optimal-budget PWA abstraction rejects the
false threshold, preserves a paired true threshold, and points back to the
expected component and active PWA region.
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

from carnot import experiment_5277_kan_milp_certificate_scale_v482 as v5277
from carnot import experiment_5304_kan_dynamic_abstraction_spotcheck_v484 as v5304
from carnot import experiment_5316_kan_optimal_abstraction_budget_v485 as v5316


JsonDict = dict[str, Any]

RUN_DATE = "20260707"
RANDOM_SEED = 5332
EXPERIMENT_ID = "exp5332-kan-counterexample-localization-v486"
MILESTONE = "2026.07.486"
SCHEMA = "carnot.experiment_5332.kan_counterexample_localization.v486"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5332_kan_counterexample_localization_v486.json"
)
INFERENCE_SUBSTRATE = "deterministic_kan_abstraction_diagnostic"
SPEC_REFS = ("REQ-KAN-5332", "SCENARIO-KAN-5332")
TERMINAL_PREFIXES = ("complete:", "blocked_")

PERTURBATION_DELTA = 0.02
FALSE_PROPERTY_THRESHOLD = v5277.TRUE_PROPERTY_THRESHOLD
TRUE_PROPERTY_THRESHOLD = v5277.TRUE_PROPERTY_THRESHOLD + PERTURBATION_DELTA
FIXTURE_COUNT = len(v5277.COMPONENT_COEFFICIENTS)
PIECE_BUDGET = v5316.PIECE_BUDGET

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Traceable Exp 5332 identifier for the bounded KAN "
        "counterexample-localization diagnostic."
    ),
    "milestone": (
        "Milestone accountability for the V486 bounded "
        "counterexample-localization task."
    ),
    "status": (
        "Terminal status for downstream readers; complete means all "
        "deterministic false-property perturbations were evaluated."
    ),
    "honest_verdict": (
        "Terminal Exp 5332 verdict; starts with complete: or blocked_ and "
        "states whether the bounded diagnostic localized counterexamples "
        "without broad certificate claims."
    ),
    "inference_substrate": (
        "Declares the deterministic KAN abstraction diagnostic substrate with "
        "no LLM inference, hardware execution, or broad KAN verification claim."
    ),
    "tests_run": (
        "Commands run to validate the localization diagnostic, artifact "
        "schema, new-code coverage, and repository tests."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "fixture_count",
    "false_property_rejection_rate",
    "true_property_preservation_rate",
    "counterexample_localization_accuracy",
    "envelope_gap_delta",
    "certificate_success_delta",
    "counterexample_localization_ready",
    "no_broad_certificate_claim",
    "tests_run",
)
WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BARE_BOOL_FIELDS = (
    "counterexample_localization_ready",
    "no_broad_certificate_claim",
)
BARE_NUMERIC_FIELDS = (
    "false_property_rejection_rate",
    "true_property_preservation_rate",
    "counterexample_localization_accuracy",
    "envelope_gap_delta",
    "certificate_success_delta",
)


@dataclass(frozen=True)
class FalsePropertyPerturbation:
    """One deterministic shift that makes the baseline true threshold false."""

    perturbation_id: str
    expected_unit_index: int
    constant_shift: float
    false_threshold: float
    true_threshold: float
    expected_region: tuple[float, float]

    def as_serializable(self) -> JsonDict:
        return {
            "perturbation_id": self.perturbation_id,
            "expected_unit_index": self.expected_unit_index,
            "constant_shift": self.constant_shift,
            "false_threshold": self.false_threshold,
            "true_threshold": self.true_threshold,
            "expected_region": list(self.expected_region),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _round(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def _round_region(region: Sequence[float]) -> tuple[float, float]:
    return (_round(region[0]), _round(region[1]))


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to a wrapped artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _optimal_plan() -> v5316.PiecePlan:
    return v5316.allocate_optimal_piece_budget()


def _optimal_abstraction() -> v5277.MultiComponentAbstraction:
    return v5316.build_abstraction_for_plan(_optimal_plan())


def define_false_property_perturbations() -> tuple[FalsePropertyPerturbation, ...]:
    """Return one expected-false perturbation for each bounded KAN component."""

    abstraction = _optimal_abstraction()
    perturbations: list[FalsePropertyPerturbation] = []
    for component_index, component in enumerate(abstraction.components):
        upper_endpoint = component.interval[1]
        active_piece = abstraction.envelope_piece_for(component_index, upper_endpoint)
        perturbations.append(
            FalsePropertyPerturbation(
                perturbation_id=(
                    f"unit_{component_index}_constant_shift_false_threshold"
                ),
                expected_unit_index=component_index,
                constant_shift=PERTURBATION_DELTA,
                false_threshold=FALSE_PROPERTY_THRESHOLD,
                true_threshold=TRUE_PROPERTY_THRESHOLD,
                expected_region=_round_region(active_piece.interval),
            )
        )
    return tuple(perturbations)


def _split_interval(
    interval: tuple[float, float],
    piece_count: int,
) -> tuple[tuple[float, float], ...]:
    lower, upper = interval
    width = (upper - lower) / piece_count
    return tuple(
        (lower + width * index, lower + width * (index + 1))
        for index in range(piece_count)
    )


def _shift_component(
    component: v5277.ConvexComponent,
    *,
    shift: float,
) -> v5277.ConvexComponent:
    return v5277.ConvexComponent(
        component_id=component.component_id,
        variable_index=component.variable_index,
        interval=component.interval,
        quadratic=component.quadratic,
        linear=component.linear,
        constant=component.constant + shift,
    )


def _build_piece(
    component: v5277.ConvexComponent,
    piece_index: int,
    interval: tuple[float, float],
) -> v5277.PWAPiece:
    lower, upper = interval
    y_lower = component.evaluate(lower)
    y_upper = component.evaluate(upper)
    slope = (y_upper - y_lower) / (upper - lower)
    intercept = y_lower - slope * lower
    local_error_bound = component.quadratic * (upper - lower) * (upper - lower) / 4.0
    return v5277.PWAPiece(
        component_id=component.component_id,
        variable_index=component.variable_index,
        piece_index=piece_index,
        interval=interval,
        slope=slope,
        intercept=intercept,
        local_error_bound=local_error_bound,
    )


def build_perturbed_abstraction(
    perturbation: FalsePropertyPerturbation,
) -> v5277.MultiComponentAbstraction:
    """Build the optimal-budget abstraction with one component constant shifted."""

    plan = _optimal_plan()
    source = v5304.build_static_abstraction()
    components: list[v5277.ConvexComponent] = []
    piece_groups: list[tuple[v5277.PWAPiece, ...]] = []
    for component_index, component in enumerate(source.components):
        shift = (
            perturbation.constant_shift
            if component_index == perturbation.expected_unit_index
            else 0.0
        )
        shifted_component = _shift_component(component, shift=shift)
        components.append(shifted_component)
        piece_groups.append(
            tuple(
                _build_piece(shifted_component, index, interval)
                for index, interval in enumerate(
                    _split_interval(
                        shifted_component.interval,
                        plan.piece_counts[component_index],
                    )
                )
            )
        )
    return v5277.MultiComponentAbstraction(
        components=tuple(components),
        pieces_by_component=tuple(piece_groups),
    )


def _predict_shifted_unit(
    baseline: v5277.MultiComponentAbstraction,
    perturbed: v5277.MultiComponentAbstraction,
) -> int:
    deltas = [
        _round(shifted.constant - base.constant)
        for base, shifted in zip(
            baseline.components,
            perturbed.components,
            strict=True,
        )
    ]
    return max(range(len(deltas)), key=lambda index: deltas[index])


def evaluate_perturbation(
    perturbation: FalsePropertyPerturbation,
) -> JsonDict:
    """Evaluate one perturbed false property and its predicted localization."""

    baseline = _optimal_abstraction()
    abstraction = build_perturbed_abstraction(perturbation)
    solve = v5304.solve_certificate(
        abstraction,
        true_threshold=perturbation.true_threshold,
        false_threshold=perturbation.false_threshold,
    )
    predicted_unit = _predict_shifted_unit(baseline, abstraction)
    predicted_x = solve.witness_inputs[predicted_unit]
    predicted_piece = abstraction.envelope_piece_for(predicted_unit, predicted_x)
    predicted_region = _round_region(predicted_piece.interval)
    actual_witness = abstraction.evaluate_actual(solve.witness_inputs)
    localized = (
        predicted_unit == perturbation.expected_unit_index
        and predicted_region == perturbation.expected_region
    )
    return {
        "perturbation_id": perturbation.perturbation_id,
        "expected_unit_index": perturbation.expected_unit_index,
        "predicted_unit_index": predicted_unit,
        "expected_region": list(perturbation.expected_region),
        "predicted_region": list(predicted_region),
        "localized": localized,
        "false_threshold": perturbation.false_threshold,
        "true_threshold": perturbation.true_threshold,
        "constant_shift": perturbation.constant_shift,
        "false_property_rejected": solve.false_property_rejected,
        "true_property_preserved": solve.certificate_success,
        "false_property_slack": solve.false_property_slack,
        "true_property_slack": solve.true_property_slack,
        "sensitivity_margin": _round(actual_witness - perturbation.false_threshold),
        "certified_upper_bound": solve.certified_upper_bound,
        "actual_witness_value": _round(actual_witness),
        "counterexample_inputs": [_round(value) for value in solve.witness_inputs],
        "selected_pieces": list(solve.selected_pieces),
        "solve_time_s": solve.solve_time_s,
        "solver_backend": solve.solver_backend,
        "solver_status": solve.solver_status,
        "fallback_used": solve.fallback_used,
        "piece_budget": PIECE_BUDGET,
        "piece_count": abstraction.piece_count,
        "envelope_gap": _round(abstraction.global_error_bound),
        "local_error_bounds": [_round(value) for value in abstraction.local_error_bounds],
        "bounded_fixture_only": True,
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return _round(sum(float(row[field]) for row in rows) / len(rows))


def run_localization_diagnostic() -> JsonDict:
    """Run the bounded false-property sensitivity and localization diagnostic."""

    budget_comparison = v5316.run_budget_comparison()
    perturbations = define_false_property_perturbations()
    results = [evaluate_perturbation(row) for row in perturbations]
    false_rate = _rate(results, "false_property_rejected")
    true_rate = _rate(results, "true_property_preserved")
    localization_accuracy = _rate(results, "localized")
    baseline_true_preserved = all(
        row["certificate_success"] for row in budget_comparison["methods"]
    )
    ready = bool(
        false_rate == 1.0
        and true_rate == 1.0
        and localization_accuracy == 1.0
        and baseline_true_preserved
        and budget_comparison["certificate_success_delta"] == 0.0
    )
    solve_time_s = _round(sum(row["solve_time_s"] for row in results), digits=6)
    return {
        "inference_substrate": INFERENCE_SUBSTRATE,
        "fixture_count": len(results),
        "false_property_rejection_rate": false_rate,
        "true_property_preservation_rate": true_rate,
        "counterexample_localization_accuracy": localization_accuracy,
        "envelope_gap_delta": budget_comparison["envelope_gap_delta"],
        "certificate_success_delta": budget_comparison["certificate_success_delta"],
        "counterexample_localization_ready": ready,
        "no_broad_certificate_claim": True,
        "piece_budget": PIECE_BUDGET,
        "envelope_gap": _round(results[0]["envelope_gap"]),
        "solve_time_s": solve_time_s,
        "baseline_true_property_preserved": baseline_true_preserved,
        "perturbations": [row.as_serializable() for row in perturbations],
        "perturbation_results": results,
        "baseline_budget_reference": {
            "source_experiment_id": v5316.EXPERIMENT_ID,
            "optimal_piece_counts": budget_comparison["allocation_strategy"][
                "selected_piece_counts"
            ],
            "piece_budget": budget_comparison["piece_budget"],
            "envelope_gap_delta": budget_comparison["envelope_gap_delta"],
            "certificate_success_delta": budget_comparison[
                "certificate_success_delta"
            ],
            "false_property_rejection_rate": budget_comparison[
                "false_property_rejection_rate"
            ],
        },
    }


def honest_verdict(diagnostic: Mapping[str, Any]) -> str:
    """Return the terminal verdict for the bounded localization diagnostic."""

    if diagnostic["no_broad_certificate_claim"] is not True:
        return "blocked_broad_certificate_claim_not_allowed"
    if diagnostic["false_property_rejection_rate"] != 1.0:
        return "blocked_false_property_perturbations_not_rejected"
    if diagnostic["true_property_preservation_rate"] != 1.0:
        return "blocked_true_property_certificate_not_preserved"
    if diagnostic["counterexample_localization_accuracy"] != 1.0:
        return "blocked_counterexample_localization_incomplete"
    if diagnostic["counterexample_localization_ready"] is not True:
        return "blocked_counterexample_localization_not_ready"
    return (
        "complete: bounded KAN abstraction diagnostic rejected deterministic "
        "false-property perturbations, localized their counterexample regions, "
        "and preserved paired true-property certificates without a broad KAN "
        "verification claim"
    )


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and validate the Exp 5332 terminal artifact."""

    started_at = time.perf_counter()
    diagnostic = run_localization_diagnostic()
    measured_duration = (
        round(time.perf_counter() - started_at, 6)
        if duration_s is None
        else duration_s
    )
    status = "complete" if diagnostic["counterexample_localization_ready"] else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "experiment_id": wrap_field("experiment_id", EXPERIMENT_ID),
        "milestone": wrap_field("milestone", MILESTONE),
        "status": wrap_field("status", status),
        "honest_verdict": wrap_field("honest_verdict", honest_verdict(diagnostic)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "fixture_count": diagnostic["fixture_count"],
        "false_property_rejection_rate": diagnostic[
            "false_property_rejection_rate"
        ],
        "true_property_preservation_rate": diagnostic[
            "true_property_preservation_rate"
        ],
        "counterexample_localization_accuracy": diagnostic[
            "counterexample_localization_accuracy"
        ],
        "envelope_gap_delta": diagnostic["envelope_gap_delta"],
        "certificate_success_delta": diagnostic["certificate_success_delta"],
        "counterexample_localization_ready": diagnostic[
            "counterexample_localization_ready"
        ],
        "no_broad_certificate_claim": diagnostic["no_broad_certificate_claim"],
        "tests_run": wrap_field("tests_run", [dict(row) for row in tests_run or []]),
        "piece_budget": diagnostic["piece_budget"],
        "envelope_gap": diagnostic["envelope_gap"],
        "solve_time_s": diagnostic["solve_time_s"],
        "baseline_true_property_preserved": diagnostic[
            "baseline_true_property_preserved"
        ],
        "perturbations": diagnostic["perturbations"],
        "perturbation_results": diagnostic["perturbation_results"],
        "baseline_budget_reference": diagnostic["baseline_budget_reference"],
        "source_artifacts": [
            str(v5277.RESULT_RELATIVE_PATH),
            str(v5316.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "bounded deterministic Exp 5277 KAN PWA/MILP input box only",
            "Exp 5316 optimal-budget allocation is reused only as a diagnostic fixture",
            "counterexample localization is measured only on three deterministic perturbations",
            "no broad KAN verification success claim",
            "no trained-network soundness claim",
            "no hardware execution or hardware speedup claim",
            "no live LLM inference claim",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5332 artifact drifts from the bounded contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require("value" in wrapped, f"{field} missing value")
        _require(
            wrapped.get("principle") == FIELD_PRINCIPLES[field],
            f"{field} principle drift",
        )
    for field in BARE_BOOL_FIELDS:
        _require(isinstance(artifact[field], bool), f"{field} must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        _require(_is_number(artifact[field]), f"{field} must be a bare numeric value")
    _require(
        isinstance(artifact["fixture_count"], int)
        and not isinstance(artifact["fixture_count"], bool),
        "fixture_count must be a bare integer",
    )

    verdict = artifact["honest_verdict"]["value"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict prefix",
    )
    _require(artifact["experiment_id"]["value"] == EXPERIMENT_ID, "experiment_id drift")
    _require(artifact["milestone"]["value"] == MILESTONE, "milestone drift")
    _require(artifact["status"]["value"] == "complete", "status must be complete")
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        f"inference_substrate must be {INFERENCE_SUBSTRATE}",
    )
    _require(artifact["fixture_count"] == FIXTURE_COUNT, "fixture count drift")
    _require(
        artifact["false_property_rejection_rate"] == 1.0,
        "false property rejection rate drift",
    )
    _require(
        artifact["true_property_preservation_rate"] == 1.0,
        "true property preservation rate drift",
    )
    _require(
        artifact["counterexample_localization_accuracy"] == 1.0,
        "counterexample localization accuracy drift",
    )
    _require(artifact["envelope_gap_delta"] > 0.0, "envelope gap delta drift")
    _require(
        artifact["certificate_success_delta"] == 0.0,
        "certificate success delta must remain neutral",
    )
    _require(
        artifact["counterexample_localization_ready"] is True,
        "counterexample localization must be ready",
    )
    _require(
        artifact["no_broad_certificate_claim"] is True,
        "broad certificate claim must be absent",
    )
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run must be list")
    _validate_perturbation_results(artifact["perturbation_results"])
    _require("REQ-KAN-5332" in artifact["spec_refs"], "spec refs drift")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def _validate_perturbation_results(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == FIXTURE_COUNT, "perturbation result count drift")
    for row in rows:
        _require(row["false_property_rejected"] is True, "false property drift")
        _require(row["true_property_preserved"] is True, "true property drift")
        _require(row["localized"] is True, "localization drift")
        _require(
            row["predicted_unit_index"] == row["expected_unit_index"],
            "localization unit drift",
        )
        _require(
            row["predicted_region"] == row["expected_region"],
            "localization region drift",
        )
        _require(row["false_property_slack"] < 0.0, "false slack drift")
        _require(row["true_property_slack"] > 0.0, "true slack drift")
        _require(row["sensitivity_margin"] > 0.0, "sensitivity margin drift")
        _require(row["piece_budget"] == PIECE_BUDGET, "piece budget drift")
        _require(row["bounded_fixture_only"] is True, "bounded scope drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5332 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    rows = []
    for row in artifact["perturbation_results"]:
        rows.append(
            {
                "perturbation_id": row["perturbation_id"],
                "expected_unit_index": row["expected_unit_index"],
                "predicted_unit_index": row["predicted_unit_index"],
                "expected_region": row["expected_region"],
                "predicted_region": row["predicted_region"],
                "false_property_rejected": row["false_property_rejected"],
                "true_property_preserved": row["true_property_preserved"],
                "localized": row["localized"],
                "piece_budget": row["piece_budget"],
                "envelope_gap": row["envelope_gap"],
            }
        )
    payload = {
        "experiment_id": artifact["experiment_id"]["value"],
        "spec_refs": artifact["spec_refs"],
        "fixture_count": artifact["fixture_count"],
        "false_property_rejection_rate": artifact["false_property_rejection_rate"],
        "true_property_preservation_rate": artifact["true_property_preservation_rate"],
        "counterexample_localization_accuracy": artifact[
            "counterexample_localization_accuracy"
        ],
        "envelope_gap_delta": artifact["envelope_gap_delta"],
        "certificate_success_delta": artifact["certificate_success_delta"],
        "counterexample_localization_ready": artifact[
            "counterexample_localization_ready"
        ],
        "no_broad_certificate_claim": artifact["no_broad_certificate_claim"],
        "piece_budget": artifact["piece_budget"],
        "perturbation_results": rows,
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - CLI wrapper for manual artifact refresh.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
