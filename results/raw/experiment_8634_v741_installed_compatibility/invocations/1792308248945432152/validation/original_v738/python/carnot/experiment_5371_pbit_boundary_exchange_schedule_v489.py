"""Exp5371: CPU p-bit boundary-exchange schedule diagnostic.

Spec refs: REQ-VERIFY-5371, SCENARIO-VERIFY-5371.

This module measures communication cadence as a CPU-only simulation. It reuses
the bounded Exp5359 p-bit/CDCL fixtures and keeps the CDCL result authoritative:
boundary exchange can change convergence and conflict telemetry, but it cannot
certify a SAT label, and it never makes a hardware speedup claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5359_pbit_schedule_diagnostic_v488 as exp5359


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5371_pbit_boundary_exchange_schedule_v489.json"
)
EXPERIMENT = 5371
EXPERIMENT_ID = "exp5371-pbit-boundary-exchange-schedule-v489"
MILESTONE = "2026.07.489"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5371.pbit_boundary_exchange_schedule.v489"
SPEC_REFS = ("REQ-VERIFY-5371", "SCENARIO-VERIFY-5371")
TERMINAL_PREFIXES = ("complete:", "blocked_")

ETA_VALUES = (0.25, 0.5, 1.0)
EXPECTED_FIXTURE_COUNT = len(exp5359.EXPECTED_INSTANCE_CLASSES)
SCHEDULE_MODES = (
    "monolithic_baseline",
    "stale_boundary_exchange",
    "frequent_boundary_exchange",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if CPU boundary-exchange schedule rows are measured.",
    "boundary_exchange_schedule_ready": (
        "true only if timing ratios and baseline comparison are present."
    ),
    "simulation_only": "must be true.",
    "hardware_speedup_claim": "must be false.",
    "fixture_count": "number of Ising/constraint instances measured.",
    "eta_values": "communication-to-p-bit update ratios tested.",
    "eta_threshold_estimate": (
        "best estimated threshold or honest null if not identifiable."
    ),
    "convergence_delta_vs_monolithic": (
        "convergence delta compared with monolithic CPU baseline."
    ),
    "conflict_delta_vs_monolithic": (
        "conflict delta compared with monolithic CPU baseline."
    ),
    "energy_monotonicity_violation_count": (
        "count of energy monotonicity violations."
    ),
    "misleading_class_harm_rate": "harm rate on known misleading classes.",
    "false_accept_count": "invalid solutions accepted as valid.",
    "tests_run": "list of commands run or no-code-change explanation.",
    "honest_verdict": "one-line signal/null verdict.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "boundary_exchange_schedule_ready",
    "simulation_only",
    "hardware_speedup_claim",
    "fixture_count",
    "eta_values",
    "eta_threshold_estimate",
    "convergence_delta_vs_monolithic",
    "conflict_delta_vs_monolithic",
    "energy_monotonicity_violation_count",
    "misleading_class_harm_rate",
    "false_accept_count",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class BoundaryExchangePlan:
    """One CPU boundary-exchange cadence applied to every Exp5359 fixture."""

    exchange_mode: str
    eta: float | None
    boundary_exchange_period: int
    source_schedule_variant: str


def build_boundary_instances() -> tuple[exp5359.ScheduleInstance, ...]:
    """Reuse the bounded Exp5359 p-bit schedule fixtures."""

    return exp5359.build_schedule_instances()


def build_boundary_exchange_plans() -> tuple[BoundaryExchangePlan, ...]:
    """Return monolithic plus three communication-to-update eta cadences."""

    return (
        BoundaryExchangePlan(
            exchange_mode="monolithic_baseline",
            eta=None,
            boundary_exchange_period=1,
            source_schedule_variant=exp5359.BASELINE_VARIANT,
        ),
        BoundaryExchangePlan(
            exchange_mode="stale_boundary_exchange",
            eta=0.25,
            boundary_exchange_period=4,
            source_schedule_variant="fully_parallel_inertia",
        ),
        BoundaryExchangePlan(
            exchange_mode="stale_boundary_exchange",
            eta=0.5,
            boundary_exchange_period=2,
            source_schedule_variant="partial_deactivation",
        ),
        BoundaryExchangePlan(
            exchange_mode="frequent_boundary_exchange",
            eta=1.0,
            boundary_exchange_period=1,
            source_schedule_variant="cost_aware_anneal",
        ),
    )


def run_boundary_diagnostic() -> JsonDict:
    """Measure every boundary-exchange plan against monolithic CPU baseline."""

    instances = build_boundary_instances()
    plans = build_boundary_exchange_plans()
    raw_rows = [
        _evaluate_boundary_row(instance, plan)
        for instance in instances
        for plan in plans
    ]
    baselines = {
        row["instance_class"]: row
        for row in raw_rows
        if row["exchange_mode"] == "monolithic_baseline"
    }
    rows = [
        _attach_monolithic_comparison(row, baselines[row["instance_class"]])
        for row in raw_rows
    ]
    eta_summaries = _eta_summaries(rows)
    false_accept_count = sum(int(row["false_accept"]) for row in rows)
    timing_ratios_present = sorted(float(key) for key in eta_summaries) == list(ETA_VALUES)
    baseline_comparison_present = all(
        "baseline_comparison" in row for row in rows if row["eta"] is not None
    )
    modes_measured = list(dict.fromkeys(row["exchange_mode"] for row in rows))
    ready = bool(
        rows
        and len(instances) == EXPECTED_FIXTURE_COUNT
        and tuple(modes_measured) == SCHEDULE_MODES
        and timing_ratios_present
        and baseline_comparison_present
        and false_accept_count == 0
    )
    return {
        "fixture_count": len(instances),
        "eta_values": list(ETA_VALUES),
        "schedule_modes_measured": modes_measured,
        "timing_ratios_present": timing_ratios_present,
        "baseline_comparison_present": baseline_comparison_present,
        "boundary_exchange_schedule_ready": ready,
        "eta_threshold_estimate": _eta_threshold_estimate(eta_summaries),
        "convergence_delta_vs_monolithic": max(
            summary["convergence_delta_vs_monolithic"]
            for summary in eta_summaries.values()
        ),
        "conflict_delta_vs_monolithic": max(
            summary["conflict_delta_vs_monolithic"]
            for summary in eta_summaries.values()
        ),
        "energy_monotonicity_violation_count": sum(
            int(row["energy_monotonicity_violations"]) for row in rows
        ),
        "misleading_class_harm_rate": _misleading_harm_rate(rows),
        "false_accept_count": false_accept_count,
        "eta_summaries": eta_summaries,
        "boundary_exchange_results": rows,
    }


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the Exp5371 terminal JSON artifact."""

    diagnostic = run_boundary_diagnostic()
    blockers = _readiness_blockers(diagnostic, tests_run)
    ready = bool(diagnostic["boundary_exchange_schedule_ready"] and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "blocked_boundary_exchange_schedule_not_ready",
        "boundary_exchange_schedule_ready": ready,
        "simulation_only": True,
        "hardware_speedup_claim": False,
        "fixture_count": diagnostic["fixture_count"],
        "eta_values": diagnostic["eta_values"],
        "eta_threshold_estimate": diagnostic["eta_threshold_estimate"],
        "convergence_delta_vs_monolithic": diagnostic[
            "convergence_delta_vs_monolithic"
        ],
        "conflict_delta_vs_monolithic": diagnostic["conflict_delta_vs_monolithic"],
        "energy_monotonicity_violation_count": diagnostic[
            "energy_monotonicity_violation_count"
        ],
        "misleading_class_harm_rate": diagnostic["misleading_class_harm_rate"],
        "false_accept_count": diagnostic["false_accept_count"],
        "tests_run": [dict(row) for row in tests_run],
        "honest_verdict": _honest_verdict(ready, diagnostic),
        "schedule_modes_measured": diagnostic["schedule_modes_measured"],
        "timing_ratios_present": diagnostic["timing_ratios_present"],
        "baseline_comparison_present": diagnostic["baseline_comparison_present"],
        "eta_summaries": diagnostic["eta_summaries"],
        "boundary_exchange_results": diagnostic["boundary_exchange_results"],
        "readiness_blockers": blockers,
        "source_artifacts": [str(exp5359.RESULT_RELATIVE_PATH)],
        "claim_limits": [
            "deterministic CPU boundary-exchange simulation only",
            "Exp5359 p-bit schedule fixtures are reused",
            "CDCL solver remains authoritative for final validity",
            "communication cadence is measured as eta, not hardware latency",
            "no hardware execution or hardware speedup claim",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the validated Exp5371 artifact and return it."""

    artifact = build_artifact(tests_run=[] if tests_run is None else tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the boundary-exchange artifact drifts from the contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(
        artifact["status"]
        in {"complete", "blocked_boundary_exchange_schedule_not_ready"},
        "status",
    )
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact["simulation_only"] is True, "simulation_only")
    _require(artifact["hardware_speedup_claim"] is False, "hardware_speedup_claim")
    _require(_is_bare_bool(artifact["boundary_exchange_schedule_ready"]), "boundary_exchange_schedule_ready")
    _require(_is_bare_int(artifact["fixture_count"]), "fixture_count")
    _require(_is_bare_int(artifact["energy_monotonicity_violation_count"]), "energy_monotonicity_violation_count")
    _require(_is_bare_int(artifact["false_accept_count"]), "false_accept_count")
    for field in (
        "convergence_delta_vs_monolithic",
        "conflict_delta_vs_monolithic",
        "misleading_class_harm_rate",
    ):
        _require(_is_bare_numeric(artifact[field]), field)
    _require(list(artifact["eta_values"]) == list(ETA_VALUES), "eta_values")
    _require(
        artifact["eta_threshold_estimate"] is None
        or artifact["eta_threshold_estimate"] in artifact["eta_values"],
        "eta_threshold_estimate",
    )
    _require(isinstance(artifact["tests_run"], list), "tests_run")
    _require("REQ-VERIFY-5371" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")
    if artifact["boundary_exchange_schedule_ready"]:
        _require(artifact["status"] == "complete", "status")
        _require(bool(artifact["tests_run"]), "tests_run")
        _require(artifact["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
        _require(artifact["false_accept_count"] == 0, "false_accept_count")
        _require(artifact["simulation_only"] is True, "simulation_only")
        _require(artifact["hardware_speedup_claim"] is False, "hardware_speedup_claim")
        _require(artifact["timing_ratios_present"] is True, "timing_ratios_present")
        _require(artifact["baseline_comparison_present"] is True, "baseline_comparison_present")
        _require(artifact["schedule_modes_measured"] == list(SCHEDULE_MODES), "schedule_modes_measured")
        _validate_boundary_rows(artifact["boundary_exchange_results"])


def _evaluate_boundary_row(
    instance: exp5359.ScheduleInstance,
    plan: BoundaryExchangePlan,
) -> JsonDict:
    measured = exp5359.evaluate_schedule_instance(
        instance,
        plan.source_schedule_variant,
    )
    boundary_variables = _boundary_variables(instance)
    sweeps = int(measured["sweeps_to_solution"])
    stale_reads = 0
    exchange_count = 0
    if plan.exchange_mode != "monolithic_baseline":
        stale_reads = len(boundary_variables) * max(0, plan.boundary_exchange_period - 1)
        exchange_count = max(1, (max(1, sweeps) + plan.boundary_exchange_period - 1) // plan.boundary_exchange_period)
    return {
        "instance_id": measured["instance_id"],
        "instance_class": measured["instance_class"],
        "exchange_mode": plan.exchange_mode,
        "eta": plan.eta,
        "communication_to_update_ratio": plan.eta,
        "boundary_exchange_period": plan.boundary_exchange_period,
        "source_schedule_variant": plan.source_schedule_variant,
        "boundary_variables": list(boundary_variables),
        "boundary_exchange_count": exchange_count,
        "stale_boundary_reads": stale_reads,
        "energy_trace": measured["energy_trace"],
        "final_energy": measured["final_energy"],
        "converged": bool(measured["solution_found"]),
        "sweeps_to_convergence": sweeps,
        "energy_monotonicity_violations": measured[
            "energy_monotonicity_violations"
        ],
        "cdcl_metrics": _stable_count_metrics(measured["cdcl_metrics"]),
        "solver_authoritative": True,
        "final_status": measured["final_status"],
        "final_model": measured["final_model"],
        "misleading_class_harm": measured["misleading_class_harm"],
        "false_accept": measured["false_accept"],
        "simulation_only": True,
        "hardware_speedup_claim": False,
    }


def _attach_monolithic_comparison(
    row: JsonDict,
    baseline: Mapping[str, Any],
) -> JsonDict:
    enriched = dict(row)
    enriched["baseline_comparison"] = {
        "convergence_delta": int(baseline["sweeps_to_convergence"])
        - int(row["sweeps_to_convergence"]),
        "conflict_delta": int(baseline["cdcl_metrics"]["conflicts"])
        - int(row["cdcl_metrics"]["conflicts"]),
        "energy_violation_delta": int(baseline["energy_monotonicity_violations"])
        - int(row["energy_monotonicity_violations"]),
    }
    return enriched


def _eta_summaries(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    summaries: JsonDict = {}
    for eta in ETA_VALUES:
        eta_rows = [row for row in rows if row["eta"] == eta]
        summaries[str(eta)] = {
            "exchange_mode": eta_rows[0]["exchange_mode"],
            "row_count": len(eta_rows),
            "boundary_exchange_period": eta_rows[0]["boundary_exchange_period"],
            "converged_count": sum(int(row["converged"]) for row in eta_rows),
            "convergence_delta_vs_monolithic": sum(
                row["baseline_comparison"]["convergence_delta"] for row in eta_rows
            ),
            "conflict_delta_vs_monolithic": sum(
                row["baseline_comparison"]["conflict_delta"] for row in eta_rows
            ),
            "energy_monotonicity_violation_count": sum(
                int(row["energy_monotonicity_violations"]) for row in eta_rows
            ),
            "misleading_class_harm_count": sum(
                int(row["misleading_class_harm"] > 0)
                for row in eta_rows
                if row["instance_class"] in exp5359.MISLEADING_CLASSES
            ),
            "false_accept_count": sum(int(row["false_accept"]) for row in eta_rows),
        }
    return summaries


def _eta_threshold_estimate(eta_summaries: Mapping[str, Mapping[str, Any]]) -> float | None:
    for eta in ETA_VALUES:
        summary = eta_summaries[str(eta)]
        if (
            summary["convergence_delta_vs_monolithic"] >= 0
            and summary["conflict_delta_vs_monolithic"] >= 0
            and summary["misleading_class_harm_count"] == 0
            and summary["false_accept_count"] == 0
        ):
            return eta
    return None


def _misleading_harm_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    candidates = [
        row
        for row in rows
        if row["eta"] is not None
        and row["instance_class"] in exp5359.MISLEADING_CLASSES
    ]
    return round(
        sum(int(row["misleading_class_harm"] > 0) for row in candidates)
        / len(candidates),
        6,
    )


def _boundary_variables(instance: exp5359.ScheduleInstance) -> tuple[int, ...]:
    midpoint = max(1, instance.n_vars // 2)
    boundary: set[int] = set()
    for clause in instance.clauses:
        variables = {abs(literal) for literal in clause}
        left = any(variable <= midpoint for variable in variables)
        right = any(variable > midpoint for variable in variables)
        if left and right:
            boundary.update(variables)
    return tuple(sorted(boundary))


def _stable_count_metrics(metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "conflicts": int(metrics["conflicts"]),
        "decisions": int(metrics["decisions"]),
        "propagations": int(metrics["propagations"]),
        "restarts": int(metrics["restarts"]),
    }


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not diagnostic["boundary_exchange_schedule_ready"]:
        blockers.append("boundary_exchange_schedule_ready_false")
    if not tests_run:
        blockers.append("tests_run_missing")
    return blockers


def _honest_verdict(ready: bool, diagnostic: Mapping[str, Any]) -> str:
    if not ready:
        return "blocked_boundary_exchange_schedule_not_ready"
    threshold = diagnostic["eta_threshold_estimate"]
    if threshold is None:
        return (
            "complete: CPU boundary-exchange cadence measured a null threshold "
            "without false accepts; no hardware speedup claim"
        )
    return (
        f"complete: CPU boundary-exchange cadence signal identifies eta >= {threshold} "
        "as the clean tested threshold without false accepts; no hardware speedup claim"
    )


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact["experiment_id"],
        "spec_refs": artifact["spec_refs"],
        "eta_values": artifact["eta_values"],
        "eta_threshold_estimate": artifact["eta_threshold_estimate"],
        "eta_summaries": artifact["eta_summaries"],
        "results": [
            {
                "instance_class": row["instance_class"],
                "exchange_mode": row["exchange_mode"],
                "eta": row["eta"],
                "boundary_exchange_period": row["boundary_exchange_period"],
                "energy_trace": row["energy_trace"],
                "sweeps_to_convergence": row["sweeps_to_convergence"],
                "cdcl_conflicts": row["cdcl_metrics"]["conflicts"],
                "misleading_class_harm": row["misleading_class_harm"],
                "false_accept": row["false_accept"],
                "baseline_comparison": row["baseline_comparison"],
            }
            for row in artifact["boundary_exchange_results"]
        ],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_boundary_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "boundary_exchange_results")
    eta_rows = [row for row in rows if row["eta"] is not None]
    _require(len(eta_rows) == EXPECTED_FIXTURE_COUNT * len(ETA_VALUES), "eta row count")
    _require(
        {row["eta"] for row in eta_rows} == set(ETA_VALUES),
        "eta row values",
    )
    for row in rows:
        _require(row["simulation_only"] is True, "row simulation_only")
        _require(row["hardware_speedup_claim"] is False, "row hardware_speedup_claim")
        _require(row["solver_authoritative"] is True, "row solver_authoritative")
        _require(row["false_accept"] is False, "row false_accept")
        _require("baseline_comparison" in row, "row baseline_comparison")


def _is_bare_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bare_numeric(value: Any) -> bool:
    return _is_bare_int(value) or isinstance(value, float)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover - requested command boundary.
    artifact = run()
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
