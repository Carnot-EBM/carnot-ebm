"""Exp 5316 bounded KAN optimal abstraction budget experiment.

Spec refs: REQ-KAN-5316, SCENARIO-KAN-5316.

This module keeps the KAN claim deliberately small. It reuses the bounded
three-component convex KAN/PWA fixture from Exp 5277 and the static/dynamic
comparison style from Exp 5304, then asks one narrow question: with a fixed
piece budget and a global envelope-error budget, can a deterministic
DP/knapsack-style allocation place PWA pieces across units more efficiently
than uniform static allocation, while using fewer pieces than the `.484`
dynamic all-component refinement? The answer is only about this fixture.
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
from carnot import experiment_5291_low_order_factor_certificate_curriculum_v483 as v5291
from carnot import experiment_5304_kan_dynamic_abstraction_spotcheck_v484 as v5304


JsonDict = dict[str, Any]

RUN_DATE = "20260706"
RANDOM_SEED = 5316
EXPERIMENT_ID = "exp5316-kan-optimal-abstraction-budget-v485"
MILESTONE = "2026.07.485"
SCHEMA = "carnot.experiment_5316.kan_optimal_abstraction_budget.v485"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5316_kan_optimal_abstraction_budget_v485.json"
)
INFERENCE_SUBSTRATE = "bounded_kan_pwa_milp_certificate"
SPEC_REFS = ("REQ-KAN-5316", "SCENARIO-KAN-5316")
TERMINAL_PREFIXES = ("complete:", "null:", "blocked_")

MIN_PIECES_PER_UNIT = 2
MAX_PIECES_PER_UNIT = 6
PIECE_BUDGET = 10
GLOBAL_ERROR_BUDGET = 0.006
STATIC_PIECES_PER_UNIT = 2
DYNAMIC_PIECES_PER_UNIT = 4

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Traceable Exp 5316 identifier for the bounded KAN abstraction budget run."
    ),
    "milestone": (
        "Milestone accountability for the V485 bounded certificate allocation task."
    ),
    "status": (
        "Terminal status for downstream readers; complete means the bounded "
        "static, dynamic, and optimal-budget allocations were all compared."
    ),
    "honest_verdict": (
        "Terminal Exp 5316 verdict; starts with complete:, null:, or blocked_ "
        "and states whether optimal-budget allocation tightened only the "
        "bounded fixture."
    ),
    "inference_substrate": (
        "Declares a bounded KAN PWA/MILP certificate substrate with no LLM "
        "inference, hardware execution, or broad KAN verification claim."
    ),
    "allocation_strategy": (
        "Records the deterministic DP/knapsack-style objective, selected PWA "
        "piece counts, global error budget, and static/dynamic comparators."
    ),
    "tests_run": (
        "Commands run to validate the allocation logic, artifact schema, "
        "new-code coverage, repository tests, and applicable offline e2e checks."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "kan_optimal_abstraction_ready",
    "allocation_strategy",
    "piece_budget",
    "envelope_gap_delta",
    "certificate_success_delta",
    "false_property_rejection_rate",
    "milp_solve_time_delta_s",
    "bounded_fixture_only",
    "tests_run",
)
WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BARE_BOOL_FIELDS = ("kan_optimal_abstraction_ready", "bounded_fixture_only")
BARE_NUMERIC_FIELDS = (
    "envelope_gap_delta",
    "certificate_success_delta",
    "false_property_rejection_rate",
    "milp_solve_time_delta_s",
)


@dataclass(frozen=True)
class PiecePlan:
    """A per-unit PWA piece allocation plus its conservative envelope budget."""

    strategy_id: str
    piece_counts: tuple[int, ...]
    global_error_bound: float
    local_error_bounds: tuple[float, ...]
    piece_budget: int
    global_error_budget: float
    candidate_count: int

    @property
    def total_pieces(self) -> int:
        return sum(self.piece_counts)

    def as_serializable(self) -> JsonDict:
        return {
            "strategy_id": self.strategy_id,
            "piece_counts": list(self.piece_counts),
            "total_pieces": self.total_pieces,
            "piece_budget": self.piece_budget,
            "global_error_budget": self.global_error_budget,
            "global_error_bound": self.global_error_bound,
            "local_error_bounds": list(self.local_error_bounds),
            "candidate_count": self.candidate_count,
            "global_error_budget_met": self.global_error_bound
            <= self.global_error_budget + 1e-12,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _round(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to an artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _component_error(component: v5277.ConvexComponent, pieces: int) -> float:
    width = component.interval[1] - component.interval[0]
    segment_width = width / pieces
    return component.quadratic * segment_width * segment_width / 4.0


def _plan_for_counts(
    strategy_id: str,
    piece_counts: Sequence[int],
    *,
    piece_budget: int,
    global_error_budget: float,
    candidate_count: int,
) -> PiecePlan:
    abstraction = v5304.build_static_abstraction()
    counts = tuple(int(count) for count in piece_counts)
    local = tuple(
        _round(_component_error(component, count))
        for component, count in zip(abstraction.components, counts, strict=True)
    )
    return PiecePlan(
        strategy_id=strategy_id,
        piece_counts=counts,
        global_error_bound=_round(sum(local)),
        local_error_bounds=local,
        piece_budget=piece_budget,
        global_error_budget=global_error_budget,
        candidate_count=candidate_count,
    )


def static_piece_plan() -> PiecePlan:
    """Return the Exp 5277 two-piece-per-unit static allocation."""

    abstraction = v5304.build_static_abstraction()
    return _plan_for_counts(
        "static_uniform_two_piece",
        (STATIC_PIECES_PER_UNIT,) * abstraction.component_count,
        piece_budget=STATIC_PIECES_PER_UNIT * abstraction.component_count,
        global_error_budget=GLOBAL_ERROR_BUDGET,
        candidate_count=1,
    )


def dynamic_piece_plan() -> PiecePlan:
    """Return the Exp 5304 all-component dynamic refinement allocation."""

    abstraction = v5304.build_static_abstraction()
    return _plan_for_counts(
        "dynamic_spotcheck_refine_all_v484",
        (DYNAMIC_PIECES_PER_UNIT,) * abstraction.component_count,
        piece_budget=DYNAMIC_PIECES_PER_UNIT * abstraction.component_count,
        global_error_budget=GLOBAL_ERROR_BUDGET,
        candidate_count=1,
    )


def allocate_optimal_piece_budget(
    *,
    piece_budget: int = PIECE_BUDGET,
    global_error_budget: float = GLOBAL_ERROR_BUDGET,
    max_pieces_per_unit: int = MAX_PIECES_PER_UNIT,
) -> PiecePlan:
    """Allocate pieces with a tiny dynamic program over component budgets."""

    components = v5304.build_static_abstraction().components
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    candidate_count = 0
    for component in components:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (error_so_far, counts_so_far) in states.items():
            for pieces in range(MIN_PIECES_PER_UNIT, max_pieces_per_unit + 1):
                candidate_count += 1
                new_used = used + pieces
                if new_used > piece_budget:
                    continue
                new_error = error_so_far + _component_error(component, pieces)
                new_counts = counts_so_far + (pieces,)
                previous = next_states.get(new_used)
                if previous is None or (new_error, new_counts) < previous:
                    next_states[new_used] = (new_error, new_counts)
        states = next_states

    feasible = [
        (used, error, counts)
        for used, (error, counts) in states.items()
        if len(counts) == len(components) and error <= global_error_budget + 1e-12
    ]
    if not feasible:
        raise ValueError("blocked_no_piece_allocation_meets_global_error_budget")
    used, _error, counts = min(feasible, key=lambda row: (row[0], row[1], row[2]))
    return _plan_for_counts(
        "dp_knapsack_min_pieces_then_gap",
        counts,
        piece_budget=used,
        global_error_budget=global_error_budget,
        candidate_count=candidate_count,
    )


def _split_interval(interval: tuple[float, float], pieces: int) -> tuple[tuple[float, float], ...]:
    lower, upper = interval
    width = (upper - lower) / pieces
    return tuple((lower + width * index, lower + width * (index + 1)) for index in range(pieces))


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


def build_abstraction_for_plan(plan: PiecePlan) -> v5277.MultiComponentAbstraction:
    """Build a bounded PWA abstraction from a piece-allocation plan."""

    source = v5304.build_static_abstraction()
    piece_groups = []
    for component, piece_count in zip(source.components, plan.piece_counts, strict=True):
        piece_groups.append(
            tuple(
                _build_piece(component, index, interval)
                for index, interval in enumerate(_split_interval(component.interval, piece_count))
            )
        )
    return v5277.MultiComponentAbstraction(
        components=source.components,
        pieces_by_component=tuple(piece_groups),
    )


def _method_summary(method_id: str, plan: PiecePlan) -> JsonDict:
    abstraction = build_abstraction_for_plan(plan)
    solve = v5304.solve_certificate(abstraction)
    actual_witness = abstraction.evaluate_actual(solve.witness_inputs)
    envelope_gap = _round(abstraction.global_error_bound)
    return {
        "method_id": method_id,
        "allocation_strategy_id": plan.strategy_id,
        "allocation_piece_counts": list(plan.piece_counts),
        "piece_count": abstraction.piece_count,
        "piece_budget": plan.piece_budget,
        "global_error_budget": plan.global_error_budget,
        "global_error_budget_met": envelope_gap <= plan.global_error_budget + 1e-12,
        "local_error_bounds": [_round(value) for value in abstraction.local_error_bounds],
        "global_error_bound": envelope_gap,
        "envelope_gap": envelope_gap,
        "solver_backend": solve.solver_backend,
        "solver_status": solve.solver_status,
        "fallback_used": solve.fallback_used,
        "milp_solve_time_s": solve.solve_time_s,
        "constraint_count": solve.constraint_count,
        "binary_variable_count": abstraction.binary_variable_count,
        "certificate_success": solve.certificate_success,
        "true_property_accepted": solve.certificate_success,
        "false_property_rejected": solve.false_property_rejected,
        "true_property_slack": solve.true_property_slack,
        "false_property_slack": solve.false_property_slack,
        "certified_upper_bound": solve.certified_upper_bound,
        "actual_witness_value": _round(actual_witness),
        "witness_inputs": list(solve.witness_inputs),
        "selected_pieces": list(solve.selected_pieces),
        "bounded_fixture_only": True,
    }


def run_budget_comparison() -> JsonDict:
    """Compare static, `.484` dynamic, and DP optimal-budget allocations."""

    static_plan = static_piece_plan()
    dynamic_plan = dynamic_piece_plan()
    optimal_plan = allocate_optimal_piece_budget()
    methods = [
        _method_summary("static_allocation", static_plan),
        _method_summary("dynamic_spotcheck_allocation_v484", dynamic_plan),
        _method_summary("optimal_budget_allocation", optimal_plan),
    ]
    static_method = methods[0]
    dynamic_method = methods[1]
    optimal_method = methods[2]
    false_rejection_rate = _round(
        sum(float(method["false_property_rejected"]) for method in methods) / len(methods)
    )
    certificate_success_delta = _round(
        float(optimal_method["certificate_success"])
        - float(static_method["certificate_success"])
    )
    envelope_gap_delta = _round(
        static_method["envelope_gap"] - optimal_method["envelope_gap"]
    )
    solve_time_delta = _round(
        optimal_method["milp_solve_time_s"] - dynamic_method["milp_solve_time_s"],
        digits=6,
    )
    ready = bool(
        optimal_method["certificate_success"]
        and optimal_method["false_property_rejected"]
        and optimal_method["global_error_budget_met"]
        and false_rejection_rate == 1.0
        and optimal_method["bounded_fixture_only"]
    )
    return {
        "methods": methods,
        "allocation_plans": {
            "static": static_plan.as_serializable(),
            "dynamic_v484": dynamic_plan.as_serializable(),
            "optimal_budget": optimal_plan.as_serializable(),
        },
        "allocation_strategy": {
            "strategy_id": optimal_plan.strategy_id,
            "objective": (
                "minimize total PWA pieces that satisfy the bounded global "
                "error budget, then choose the lowest remaining envelope gap"
            ),
            "selected_piece_counts": list(optimal_plan.piece_counts),
            "selected_piece_count_total": optimal_plan.total_pieces,
            "piece_budget": PIECE_BUDGET,
            "global_error_budget": GLOBAL_ERROR_BUDGET,
            "candidate_count": optimal_plan.candidate_count,
            "static_piece_counts": list(static_plan.piece_counts),
            "dynamic_v484_piece_counts": list(dynamic_plan.piece_counts),
            "bounded_fixture_only": True,
        },
        "kan_optimal_abstraction_ready": ready,
        "piece_budget": optimal_plan.total_pieces,
        "envelope_gap_delta": envelope_gap_delta,
        "certificate_success_delta": certificate_success_delta,
        "false_property_rejection_rate": false_rejection_rate,
        "milp_solve_time_delta_s": solve_time_delta,
    }


def honest_verdict(comparison: Mapping[str, Any]) -> str:
    """Return the terminal scientific verdict for the bounded allocation run."""

    if not comparison["kan_optimal_abstraction_ready"]:
        return "blocked_kan_optimal_abstraction_budget_not_ready"
    if comparison["envelope_gap_delta"] <= 0.0:
        return (
            "null: optimal-budget allocation preserved the bounded certificate "
            "but did not tighten the static envelope gap"
        )
    return (
        "complete: optimal-budget allocation tightened the bounded fixture "
        "envelope under the piece/error budget while certificate success stayed "
        "unchanged and false-property rejection stayed intact"
    )


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5316 terminal artifact."""

    started_at = time.perf_counter()
    comparison = run_budget_comparison()
    measured_duration = (
        round(time.perf_counter() - started_at, 6)
        if duration_s is None
        else duration_s
    )
    status = "complete" if comparison["kan_optimal_abstraction_ready"] else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "experiment_id": wrap_field("experiment_id", EXPERIMENT_ID),
        "milestone": wrap_field("milestone", MILESTONE),
        "status": wrap_field("status", status),
        "honest_verdict": wrap_field("honest_verdict", honest_verdict(comparison)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "kan_optimal_abstraction_ready": comparison["kan_optimal_abstraction_ready"],
        "allocation_strategy": wrap_field(
            "allocation_strategy",
            comparison["allocation_strategy"],
        ),
        "piece_budget": comparison["piece_budget"],
        "envelope_gap_delta": comparison["envelope_gap_delta"],
        "certificate_success_delta": comparison["certificate_success_delta"],
        "false_property_rejection_rate": comparison["false_property_rejection_rate"],
        "milp_solve_time_delta_s": comparison["milp_solve_time_delta_s"],
        "bounded_fixture_only": True,
        "tests_run": wrap_field("tests_run", [dict(row) for row in tests_run or []]),
        "method_comparison": comparison["methods"],
        "allocation_plans": comparison["allocation_plans"],
        "source_artifacts": [
            str(v5277.RESULT_RELATIVE_PATH),
            str(v5291.RESULT_RELATIVE_PATH),
            str(v5304.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "bounded deterministic Exp 5277 KAN PWA/MILP input box only",
            "Exp 5304 dynamic allocation is a bounded diagnostic comparator",
            "optimal-budget allocation is not a general KAN verification improvement",
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
    """Fail closed when the Exp 5316 artifact drifts from the bounded contract."""

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
        isinstance(artifact["piece_budget"], int) and not isinstance(artifact["piece_budget"], bool),
        "piece_budget must be a bare integer",
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
    _require(
        artifact["kan_optimal_abstraction_ready"] is True,
        "kan_optimal_abstraction_ready must be a bare bool true",
    )
    _require(artifact["bounded_fixture_only"] is True, "bounded fixture only must be true")
    _require(artifact["piece_budget"] == PIECE_BUDGET, "piece budget drift")
    _require(artifact["envelope_gap_delta"] > 0.0, "envelope gap delta must be positive")
    _require(
        artifact["certificate_success_delta"] == 0.0,
        "certificate success delta must remain neutral on this bounded fixture",
    )
    _require(
        artifact["false_property_rejection_rate"] == 1.0,
        "false property rejection rate must remain complete",
    )
    strategy = artifact["allocation_strategy"]["value"]
    _require(
        strategy["strategy_id"] == "dp_knapsack_min_pieces_then_gap",
        "allocation strategy drift",
    )
    _require(strategy["selected_piece_counts"] == [4, 3, 3], "selected allocation drift")
    _require(strategy["bounded_fixture_only"] is True, "allocation strategy scope drift")
    _validate_method_comparison(artifact["method_comparison"])
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run must be list")
    _require("REQ-KAN-5316" in artifact["spec_refs"], "spec refs must include REQ-KAN-5316")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def _validate_method_comparison(methods: Sequence[Mapping[str, Any]]) -> None:
    _require(len(methods) == 3, "method comparison must contain three methods")
    by_id = {method["method_id"]: method for method in methods}
    for method_id in (
        "static_allocation",
        "dynamic_spotcheck_allocation_v484",
        "optimal_budget_allocation",
    ):
        _require(method_id in by_id, f"missing method {method_id}")
    _require(by_id["static_allocation"]["piece_count"] == 6, "static piece count drift")
    _require(
        by_id["dynamic_spotcheck_allocation_v484"]["piece_count"] == 12,
        "dynamic piece count drift",
    )
    _require(
        by_id["optimal_budget_allocation"]["piece_count"] == PIECE_BUDGET,
        "optimal piece count drift",
    )
    _require(
        by_id["static_allocation"]["envelope_gap"]
        > by_id["optimal_budget_allocation"]["envelope_gap"]
        > by_id["dynamic_spotcheck_allocation_v484"]["envelope_gap"],
        "bounded envelope gap ordering drift",
    )
    for method in methods:
        _require(method["certificate_success"] is True, "certificate success drift")
        _require(method["false_property_rejected"] is True, "false property drift")
        _require(method["bounded_fixture_only"] is True, "method scope drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5316 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    methods = []
    for row in artifact["method_comparison"]:
        methods.append(
            {
                "method_id": row["method_id"],
                "allocation_piece_counts": row["allocation_piece_counts"],
                "piece_count": row["piece_count"],
                "envelope_gap": row["envelope_gap"],
                "certificate_success": row["certificate_success"],
                "false_property_rejected": row["false_property_rejected"],
            }
        )
    payload = {
        "experiment_id": artifact["experiment_id"]["value"],
        "spec_refs": artifact["spec_refs"],
        "allocation_strategy": artifact["allocation_strategy"]["value"],
        "piece_budget": artifact["piece_budget"],
        "envelope_gap_delta": artifact["envelope_gap_delta"],
        "certificate_success_delta": artifact["certificate_success_delta"],
        "false_property_rejection_rate": artifact["false_property_rejection_rate"],
        "bounded_fixture_only": artifact["bounded_fixture_only"],
        "methods": methods,
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
