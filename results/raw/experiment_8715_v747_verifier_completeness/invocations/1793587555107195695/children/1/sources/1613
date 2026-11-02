"""Exp 5291 low-order factor certificate curriculum.

Spec refs: REQ-KAN-5291, SCENARIO-KAN-5291.

This module turns the V483 simplicity-bias question into a bounded certificate
measurement, not a broad EBM-training claim. It reuses the Exp 5277
three-component KAN PWA/MILP fixture and the Exp 5278 factor-graph boundary
metadata, then evaluates unary, pair, and triple certificate stages. The
low-order-first curriculum is compared with a deterministic shuffled order so
the artifact can say whether the ordering improved certificate success or only
provided cleaner measurement telemetry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any

from carnot import experiment_5277_kan_milp_certificate_scale_v482 as v5277
from carnot import experiment_5278_constraint_factor_graph_boundary_v482 as v5278


JsonDict = dict[str, Any]

RUN_DATE = "20260706"
EXPERIMENT_ID = "exp5291-low-order-factor-certificate-curriculum-v483"
SCHEMA = "carnot.experiment_5291.low_order_factor_certificate_curriculum.v483"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5291_low_order_factor_certificate_curriculum_v483.json"
)
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
SPEC_REFS = ("REQ-KAN-5291", "SCENARIO-KAN-5291")
RANDOM_SEED = 5291
TERMINAL_PREFIXES = ("complete:", "null:", "blocked_")

LOW_ORDER_READY_PRINCIPLE = (
    "Bare boolean because the task requires a direct readiness gate; true only when "
    "all bounded stages certify, all false controls reject, and factor-order telemetry is present."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "low_order_curriculum_ready",
    "low_order_curriculum_ready_principle",
    "factor_order_metrics",
    "certificate_success_by_order",
    "false_property_rejected",
    "slack_metrics",
    "solve_time_metrics",
    "piece_count_metrics",
    "tests_run",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal verdict; starts with complete:, null:, or blocked_ and states whether "
        "the low-order curriculum helped certificate success."
    ),
    "inference_substrate": (
        "Must be offline_deterministic_certificate_no_llm because this is a local "
        "deterministic certificate run with no LLM inference."
    ),
    "factor_order_metrics": (
        "Records factor order, curriculum order, and shuffled order so low-order-first "
        "claims are auditable rather than inferred from prose."
    ),
    "certificate_success_by_order": (
        "Compares certificate success under low-order-first and shuffled schedules; "
        "prevents mistaking schedule order for a causal success gain."
    ),
    "false_property_rejected": (
        "True only when every stage rejects its nearby expected-false property by witness."
    ),
    "slack_metrics": (
        "Records true-property and false-property slack by stage so certificate margins "
        "cannot be hidden behind a single success flag."
    ),
    "solve_time_metrics": (
        "Records deterministic local solver time by stage and schedule totals."
    ),
    "piece_count_metrics": (
        "Records the PWA piece count by stage so factor-order growth is visible."
    ),
}
WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class FactorStage:
    """One bounded curriculum stage with an explicit property target."""

    stage_id: str
    order_label: str
    factor_order: int
    component_indices: tuple[int, ...]
    true_property_target: float
    false_property_target: float
    property_target: str
    source_fixture_refs: tuple[str, ...]

    def as_serializable(self) -> JsonDict:
        return {
            "stage_id": self.stage_id,
            "order_label": self.order_label,
            "factor_order": self.factor_order,
            "component_indices": list(self.component_indices),
            "true_property_target": self.true_property_target,
            "false_property_target": self.false_property_target,
            "property_target": self.property_target,
            "source_fixture_refs": list(self.source_fixture_refs),
        }


@dataclass(frozen=True)
class StageCertificate:
    """Certificate result for one factor-order stage."""

    stage_id: str
    order_label: str
    factor_order: int
    component_indices: tuple[int, ...]
    true_property_target: float
    false_property_target: float
    certified_upper_bound: float
    true_property_slack: float
    false_property_slack: float
    certificate_success: bool
    false_property_rejected: bool
    failure_class: str
    solve_time_s: float
    piece_count: int
    constraint_count: int
    witness_inputs: tuple[float, ...]
    selected_pieces: tuple[int, ...]
    actual_witness_value: float

    def as_serializable(self) -> JsonDict:
        return {
            "stage_id": self.stage_id,
            "order_label": self.order_label,
            "factor_order": self.factor_order,
            "component_indices": list(self.component_indices),
            "true_property_target": self.true_property_target,
            "false_property_target": self.false_property_target,
            "certified_upper_bound": self.certified_upper_bound,
            "true_property_slack": self.true_property_slack,
            "false_property_slack": self.false_property_slack,
            "certificate_success": self.certificate_success,
            "false_property_rejected": self.false_property_rejected,
            "failure_class": self.failure_class,
            "solve_time_s": self.solve_time_s,
            "piece_count": self.piece_count,
            "constraint_count": self.constraint_count,
            "witness_inputs": list(self.witness_inputs),
            "selected_pieces": list(self.selected_pieces),
            "actual_witness_value": self.actual_witness_value,
        }


def _require(condition: bool, message: str) -> None:
    if condition:
        return
    raise AssertionError(message)


def _round_float(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-specified principle to an artifact value."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def define_factor_stages() -> tuple[FactorStage, ...]:
    """Return the low, medium, and higher-order stages used by Exp 5291."""

    exp5277_ref = str(v5277.RESULT_RELATIVE_PATH)
    exp5278_ref = str(v5278.RESULT_RELATIVE_PATH)
    return (
        FactorStage(
            stage_id="low_order_unary",
            order_label="low",
            factor_order=1,
            component_indices=(0,),
            true_property_target=0.235,
            false_property_target=0.2235,
            property_target="component_0_upper <= 0.235 and false bound 0.2235 rejects",
            source_fixture_refs=(exp5277_ref,),
        ),
        FactorStage(
            stage_id="medium_order_pair",
            order_label="medium",
            factor_order=2,
            component_indices=(0, 1),
            true_property_target=0.398,
            false_property_target=0.3865,
            property_target="components_0_1_upper <= 0.398 and false bound 0.3865 rejects",
            source_fixture_refs=(exp5277_ref, exp5278_ref),
        ),
        FactorStage(
            stage_id="higher_order_triple",
            order_label="higher",
            factor_order=3,
            component_indices=(0, 1, 2),
            true_property_target=v5277.TRUE_PROPERTY_THRESHOLD,
            false_property_target=v5277.FALSE_PROPERTY_THRESHOLD,
            property_target="components_0_1_2_upper <= 0.515 and false bound 0.498 rejects",
            source_fixture_refs=(exp5277_ref, exp5278_ref),
        ),
    )


def _stage_abstraction(stage: FactorStage) -> v5277.MultiComponentAbstraction:
    source = v5277.build_multi_component_abstraction()
    return v5277.MultiComponentAbstraction(
        components=tuple(source.components[index] for index in stage.component_indices),
        pieces_by_component=tuple(source.pieces_by_component[index] for index in stage.component_indices),
    )


def evaluate_stage(stage: FactorStage) -> StageCertificate:
    """Solve one bounded PWA/MILP certificate stage with stage-specific thresholds."""

    abstraction = _stage_abstraction(stage)
    z3 = importlib.import_module("z3")
    optimizer = z3.Optimize()
    xs = [z3.Real(f"exp5291_{stage.stage_id}_x_{index}") for index in range(abstraction.component_count)]
    ys = [
        z3.Real(f"exp5291_{stage.stage_id}_component_upper_{index}")
        for index in range(abstraction.component_count)
    ]
    total_upper = z3.Real(f"exp5291_{stage.stage_id}_total_upper")
    selected_flag_groups: list[list[Any]] = []
    constraint_count = 0
    big_m = v5277._real(z3, 10.0)

    def add_constraints(*constraints: Any) -> None:
        nonlocal constraint_count
        optimizer.add(*constraints)
        constraint_count += len(constraints)

    for component_index, pieces in enumerate(abstraction.pieces_by_component):
        x = xs[component_index]
        y = ys[component_index]
        lo, hi = abstraction.components[component_index].interval
        flags = [
            z3.Int(f"exp5291_{stage.stage_id}_component_{component_index}_piece_{piece.piece_index}")
            for piece in pieces
        ]
        selected_flag_groups.append(flags)
        add_constraints(x >= v5277._real(z3, lo), x <= v5277._real(z3, hi), z3.Sum(flags) == 1)
        for flag, piece in zip(flags, pieces, strict=True):
            flag_real = z3.ToReal(flag)
            slack = big_m * (v5277._real(z3, 1.0) - flag_real)
            affine_value = v5277._real(z3, piece.slope) * x + v5277._real(z3, piece.intercept)
            add_constraints(
                flag >= 0,
                flag <= 1,
                x >= v5277._real(z3, piece.interval[0]) - slack,
                x <= v5277._real(z3, piece.interval[1]) + slack,
                y - affine_value <= slack,
                affine_value - y <= slack,
            )
    add_constraints(total_upper == z3.Sum(ys))

    solve_start = time.perf_counter()
    objective = optimizer.maximize(total_upper)
    status = optimizer.check()
    solve_time_s = round(time.perf_counter() - solve_start, 6)
    _require(status == z3.sat, f"solver status drift: {status}")  # pragma: no cover

    model = optimizer.model()
    certified_upper = _round_float(v5277._z3_float(objective.value()))
    witness_inputs = tuple(_round_float(v5277._z3_float(model.eval(x, model_completion=True))) for x in xs)
    selected_pieces = tuple(
        next(
            piece_index
            for piece_index, flag in enumerate(flags)
            if v5277._z3_float(model.eval(flag, model_completion=True)) > 0.5
        )
        for flags in selected_flag_groups
    )
    actual_witness = _round_float(abstraction.evaluate_actual(witness_inputs))
    true_slack = _round_float(stage.true_property_target - certified_upper)
    false_slack = _round_float(stage.false_property_target - certified_upper)
    certificate_success = true_slack >= -1e-12
    false_rejected = false_slack < 0.0 and actual_witness > stage.false_property_target
    failure_class = _failure_class(certificate_success, false_rejected)
    return StageCertificate(
        stage_id=stage.stage_id,
        order_label=stage.order_label,
        factor_order=stage.factor_order,
        component_indices=stage.component_indices,
        true_property_target=stage.true_property_target,
        false_property_target=stage.false_property_target,
        certified_upper_bound=certified_upper,
        true_property_slack=true_slack,
        false_property_slack=false_slack,
        certificate_success=certificate_success,
        false_property_rejected=false_rejected,
        failure_class=failure_class,
        solve_time_s=solve_time_s,
        piece_count=abstraction.piece_count,
        constraint_count=constraint_count,
        witness_inputs=witness_inputs,
        selected_pieces=selected_pieces,
        actual_witness_value=actual_witness,
    )


def _failure_class(certificate_success: bool, false_rejected: bool) -> str:
    if not certificate_success:
        return "true_property_too_loose"
    if not false_rejected:
        return "false_property_not_rejected"
    return "none"


def factor_boundary_summary() -> JsonDict:
    """Summarize the reused Exp 5278 factor boundary for curriculum provenance."""

    source = v5278.select_tiny_fixture()
    boundary = v5278.build_boundary(source)
    false_check = v5278.reject_false_assignment(boundary, boundary.false_assignment)
    enumeration = v5278.enumerate_boundary(boundary)
    return {
        "source_fixture_id": source["fixture_id"],
        "source_artifact": str(v5278.RESULT_RELATIVE_PATH),
        "factor_count": len(boundary.factors),
        "binary_variable_count": len(boundary.bit_order),
        "pairwise_qubo_term_count": len(boundary.quadratic),
        "false_assignment_rejected": false_check["rejected"],
        "cpu_enumerator_state_count": enumeration["state_count"],
        "best_energy": enumeration["best_energy"],
    }


def run_curriculum(stages: Sequence[FactorStage] | None = None) -> JsonDict:
    """Run low-order-first and deterministic shuffled certificate schedules."""

    stage_list = tuple(stages or define_factor_stages())
    outcomes = tuple(evaluate_stage(stage) for stage in stage_list)
    by_stage = {outcome.stage_id: outcome for outcome in outcomes}
    low_order_first_sequence = [stage.stage_id for stage in stage_list]
    shuffled_sequence = ["higher_order_triple", "low_order_unary", "medium_order_pair"]
    curriculum_successes = [by_stage[stage_id].certificate_success for stage_id in low_order_first_sequence]
    shuffled_successes = [by_stage[stage_id].certificate_success for stage_id in shuffled_sequence]
    success_advantage = (
        sum(curriculum_successes) / len(curriculum_successes)
        - sum(shuffled_successes) / len(shuffled_successes)
    )
    factor_order_metrics = {
        "factor_orders_seen": [outcome.factor_order for outcome in outcomes],
        "low_order_first_sequence": low_order_first_sequence,
        "shuffled_sequence": shuffled_sequence,
        "low_before_high_in_curriculum": low_order_first_sequence.index("low_order_unary")
        < low_order_first_sequence.index("higher_order_triple"),
        "lowest_order_success_step": _success_step(low_order_first_sequence, by_stage, "low_order_unary"),
        "highest_order_success_step": _success_step(low_order_first_sequence, by_stage, "higher_order_triple"),
        "failure_class_by_stage": {outcome.stage_id: outcome.failure_class for outcome in outcomes},
    }
    certificate_success_by_order = {
        "curriculum": {stage_id: by_stage[stage_id].certificate_success for stage_id in low_order_first_sequence},
        "shuffled": {stage_id: by_stage[stage_id].certificate_success for stage_id in shuffled_sequence},
        "all_curriculum_stages_certified": all(curriculum_successes),
        "all_shuffled_stages_certified": all(shuffled_successes),
        "success_advantage_over_shuffled": _round_float(success_advantage),
        "helped_certificate_success": success_advantage > 0.0,
    }
    slack_metrics = {
        "true_property_slack_by_stage": {
            outcome.stage_id: outcome.true_property_slack for outcome in outcomes
        },
        "false_property_slack_by_stage": {
            outcome.stage_id: outcome.false_property_slack for outcome in outcomes
        },
        "minimum_true_property_slack": min(outcome.true_property_slack for outcome in outcomes),
        "all_true_property_slacks_positive": all(outcome.true_property_slack > 0.0 for outcome in outcomes),
        "all_false_property_slacks_negative": all(outcome.false_property_slack < 0.0 for outcome in outcomes),
        "true_slack_non_decreasing_with_order": _non_decreasing(
            [outcome.true_property_slack for outcome in outcomes]
        ),
    }
    solve_time_metrics = {
        "solve_time_s_by_stage": {outcome.stage_id: outcome.solve_time_s for outcome in outcomes},
        "total_low_order_first_s": _round_float(sum(by_stage[stage_id].solve_time_s for stage_id in low_order_first_sequence)),
        "total_shuffled_s": _round_float(sum(by_stage[stage_id].solve_time_s for stage_id in shuffled_sequence)),
        "all_solve_times_nonnegative": all(outcome.solve_time_s >= 0.0 for outcome in outcomes),
    }
    piece_count_metrics = {
        "piece_count_by_stage": {outcome.stage_id: outcome.piece_count for outcome in outcomes},
        "piece_counts_by_factor_order": {
            str(outcome.factor_order): outcome.piece_count for outcome in outcomes
        },
        "piece_count_increases_with_order": _strictly_increasing(
            [outcome.piece_count for outcome in outcomes]
        ),
    }
    return {
        "stages": [stage.as_serializable() for stage in stage_list],
        "stage_outcomes": [outcome.as_serializable() for outcome in outcomes],
        "factor_boundary_summary": factor_boundary_summary(),
        "factor_order_metrics": factor_order_metrics,
        "certificate_success_by_order": certificate_success_by_order,
        "false_property_rejected": all(outcome.false_property_rejected for outcome in outcomes),
        "slack_metrics": slack_metrics,
        "solve_time_metrics": solve_time_metrics,
        "piece_count_metrics": piece_count_metrics,
    }


def _success_step(sequence: Sequence[str], by_stage: Mapping[str, StageCertificate], stage_id: str) -> int | None:
    return sequence.index(stage_id) + 1 if by_stage[stage_id].certificate_success else None


def _non_decreasing(values: Sequence[float]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:], strict=False))


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:], strict=False))


def _honest_verdict(ready: bool, success_helped: bool) -> str:
    if not ready:
        return "blocked_low_order_curriculum_not_ready"
    if success_helped:
        return "complete: low-order curriculum helped certificate success in the bounded fixture"
    return (
        "complete: low-order curriculum did not improve certificate success over the shuffled "
        "ordering; all bounded stages certified, so the value is measurement and factor-order telemetry"
    )


def _curriculum_ready(curriculum: Mapping[str, Any]) -> bool:
    return bool(
        curriculum["false_property_rejected"]
        and curriculum["certificate_success_by_order"]["all_curriculum_stages_certified"]
        and curriculum["certificate_success_by_order"]["all_shuffled_stages_certified"]
        and curriculum["factor_order_metrics"]["factor_orders_seen"] == [1, 2, 3]
        and curriculum["piece_count_metrics"]["piece_count_increases_with_order"]
    )


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5291 terminal artifact."""

    start = time.perf_counter()
    curriculum = run_curriculum()
    ready = _curriculum_ready(curriculum)
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "honest_verdict": wrap_field(
            "honest_verdict",
            _honest_verdict(
                ready,
                curriculum["certificate_success_by_order"]["helped_certificate_success"],
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "low_order_curriculum_ready": ready,
        "low_order_curriculum_ready_principle": LOW_ORDER_READY_PRINCIPLE,
        "factor_order_metrics": wrap_field("factor_order_metrics", curriculum["factor_order_metrics"]),
        "certificate_success_by_order": wrap_field(
            "certificate_success_by_order",
            curriculum["certificate_success_by_order"],
        ),
        "false_property_rejected": wrap_field(
            "false_property_rejected",
            curriculum["false_property_rejected"],
        ),
        "slack_metrics": wrap_field("slack_metrics", curriculum["slack_metrics"]),
        "solve_time_metrics": wrap_field("solve_time_metrics", curriculum["solve_time_metrics"]),
        "piece_count_metrics": wrap_field("piece_count_metrics", curriculum["piece_count_metrics"]),
        "tests_run": [dict(row) for row in tests_run or []],
        "stages": curriculum["stages"],
        "stage_outcomes": curriculum["stage_outcomes"],
        "factor_boundary_summary": curriculum["factor_boundary_summary"],
        "source_artifacts": [
            str(v5277.RESULT_RELATIVE_PATH),
            str(v5278.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "bounded deterministic Exp 5277 KAN PWA/MILP fixture only",
            "Exp 5278 factor boundary used for provenance and medium-order context",
            "no broad KAN verification claim",
            "no trained-network soundness claim",
            "no hardware execution or hardware speedup claim",
            "no live LLM inference claim",
            "no causal EBM training-dynamics claim",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(curriculum)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5291 artifact drifts from the narrow contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    ready = artifact["low_order_curriculum_ready"]
    _require(isinstance(ready, bool), "low_order_curriculum_ready must be bare bool")
    _require(
        artifact["low_order_curriculum_ready_principle"] == LOW_ORDER_READY_PRINCIPLE,
        "low_order_curriculum_ready_principle drift",
    )
    verdict = artifact["honest_verdict"]["value"]
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require("low-order curriculum" in verdict, "honest_verdict must mention low-order curriculum")
    _require(artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, "inference_substrate drift")
    _require(isinstance(artifact["tests_run"], list), "tests_run must be list")

    factor_metrics = artifact["factor_order_metrics"]["value"]
    success_metrics = artifact["certificate_success_by_order"]["value"]
    false_rejected = artifact["false_property_rejected"]["value"]
    slack_metrics = artifact["slack_metrics"]["value"]
    solve_time_metrics = artifact["solve_time_metrics"]["value"]
    piece_metrics = artifact["piece_count_metrics"]["value"]

    _require(factor_metrics["factor_orders_seen"] == [1, 2, 3], "factor order metrics drift")
    _require(factor_metrics["low_before_high_in_curriculum"] is True, "low-order reporting drift")
    _require(success_metrics["all_curriculum_stages_certified"] is True, "curriculum certificates failed")
    _require(success_metrics["all_shuffled_stages_certified"] is True, "shuffled certificates failed")
    _require(success_metrics["success_advantage_over_shuffled"] == 0.0, "shuffle comparison drift")
    _require(success_metrics["helped_certificate_success"] is False, "helped metric drift")
    _require(false_rejected is True, "false property must be rejected")
    _require(slack_metrics["minimum_true_property_slack"] > 0.0, "slack must remain positive")
    _require(slack_metrics["all_true_property_slacks_positive"] is True, "slack positivity drift")
    _require(slack_metrics["all_false_property_slacks_negative"] is True, "false slack drift")
    _require(solve_time_metrics["all_solve_times_nonnegative"] is True, "solve time drift")
    _require(piece_metrics["piece_count_by_stage"] == {
        "low_order_unary": 2,
        "medium_order_pair": 4,
        "higher_order_triple": 6,
    }, "piece count drift")
    _require(piece_metrics["piece_count_increases_with_order"] is True, "piece count order drift")
    _require(ready is True, "low_order_curriculum_ready must be true for completed artifact")
    _require("REQ-KAN-5291" in artifact["spec_refs"], "spec_refs must include REQ-KAN-5291")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")
    for outcome in artifact["stage_outcomes"]:
        _require(outcome["failure_class"] == "none", "failure class drift")
        _require(outcome["false_property_rejected"] is True, "stage false property drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5291 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _checksum_payload(curriculum: Mapping[str, Any]) -> str:
    checksum_rows = []
    for outcome in curriculum["stage_outcomes"]:
        checksum_rows.append(
            {
                "stage_id": outcome["stage_id"],
                "factor_order": outcome["factor_order"],
                "certified_upper_bound": outcome["certified_upper_bound"],
                "true_property_slack": outcome["true_property_slack"],
                "false_property_slack": outcome["false_property_slack"],
                "certificate_success": outcome["certificate_success"],
                "false_property_rejected": outcome["false_property_rejected"],
                "piece_count": outcome["piece_count"],
                "failure_class": outcome["failure_class"],
            }
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "stages": curriculum["stages"],
        "stage_outcomes": checksum_rows,
        "factor_boundary_summary": curriculum["factor_boundary_summary"],
        "factor_order_metrics": curriculum["factor_order_metrics"],
        "certificate_success_by_order": curriculum["certificate_success_by_order"],
        "slack_metrics": curriculum["slack_metrics"],
        "piece_count_metrics": curriculum["piece_count_metrics"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - exercised through write_outputs tests.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
