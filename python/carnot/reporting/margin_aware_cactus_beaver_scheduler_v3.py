"""Exp 1371 margin-aware Cactus/BEAVER scheduler replay.

Spec: REQ-VERIFY-1371,
      SCENARIO-VERIFY-1371
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


DEFAULT_RUN_DATE = "20260505"
DEFAULT_EXP1369_PATH = Path(
    "results/experiment_1369_semantic_validator_v2_nsvif_z3_constraints.json"
)
DEFAULT_EXP1370_PATH = Path("results/experiment_1370_verge_mcs_repair_localization_v2.json")
DEFAULT_OUTPUT_PATH = Path("results/experiment_1371_margin_aware_cactus_beaver_scheduler_v3.json")
ARTIFACT_NAME = "experiment_1371_margin_aware_cactus_beaver_scheduler_v3"
SCHEMA_VERSION = 1
EXP1370_REPAIR_HINT_PRECISION_GATE = 0.5
HIGH_CONFIDENCE_MARGIN = 0.8
FULL_VERIFIER_UNIT_COST = 1.0
PROXY_SCORE_UNIT_COST = 0.05
LOW_MARGIN_STATES = {"UNSAT", "UNKNOWN", "REPAIR_HINT", "UNKNOWN_COLLAPSED"}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "proxy_accept_rate",
    "low_margin_escalation_rate",
    "full_verifier_call_reduction",
    "false_acceptance_rate",
    "repair_hint_reuse_rate",
    "verifier_cost_reduction_proxy",
    "triage_claim_allowed",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_margin_aware_scheduler_artifact(
    *,
    exp1369_artifact: Mapping[str, Any],
    exp1370_artifact: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    repair_hint_precision_gate: float = EXP1370_REPAIR_HINT_PRECISION_GATE,
    high_confidence_margin: float = HIGH_CONFIDENCE_MARGIN,
) -> dict[str, Any]:
    """Build a CPU-only replay artifact for the margin-aware scheduler.

    This experiment asks a narrow triage question: how many expensive full
    verifier calls could be avoided if the scheduler accepts only rows with a
    large semantic safety margin and escalates every uncertain row?  It does not
    mint new verifier evidence.  Exp 1369 provides the semantic states and Exp
    1370 provides repair hints for escalated cases, so the replay can measure
    call reduction without hiding UNKNOWN or repair-needed cases behind a cheap
    proxy decision.
    """

    root = Path(project_root)
    repair_hint_precision = _float(exp1370_artifact.get("repair_hint_precision"))
    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact["source_context"] = _source_context(
        exp1369_artifact=exp1369_artifact,
        exp1370_artifact=exp1370_artifact,
        repair_hint_precision=repair_hint_precision,
        repair_hint_precision_gate=repair_hint_precision_gate,
        high_confidence_margin=high_confidence_margin,
    )

    if repair_hint_precision < repair_hint_precision_gate:
        artifact.update(
            {
                "status": "blocked",
                "terminal_blocker": (
                    f"exp1370_repair_hint_precision_failed:"
                    f"{repair_hint_precision:g}_lt_{repair_hint_precision_gate:g}"
                ),
                "honest_verdict": "blocked_exp1370_repair_hint_precision_below_0_5",
                "measurement_note": (
                    "Exp 1370 repair_hint_precision did not satisfy the >= 0.5 "
                    "gate, so the margin-aware scheduler replay was not allowed "
                    "to claim verifier-call reduction."
                ),
            }
        )
        return artifact

    semantic_rows = _semantic_validator_rows(exp1369_artifact)
    repair_hints = _repair_hints_by_case(exp1370_artifact)
    scheduler_rows = [
        _scheduler_row(
            row=row,
            repair_hint_row=repair_hints.get(_case_id(row)),
            high_confidence_margin=high_confidence_margin,
        )
        for row in semantic_rows
    ]
    totals = _summarize_scheduler_rows(scheduler_rows)

    full_verifier_call_reduction = totals["observed_full_verifier_call_reduction"]
    verifier_cost_reduction_proxy = totals["observed_verifier_cost_reduction_proxy"]
    if totals["false_acceptance_count"] or totals["unknown_silently_accepted_count"]:
        full_verifier_call_reduction = 0.0
        verifier_cost_reduction_proxy = 0.0

    triage_claim_allowed = bool(
        totals["case_count"]
        and totals["proxy_accept_count"]
        and full_verifier_call_reduction > 0.0
        and totals["false_acceptance_count"] == 0
        and totals["unknown_silently_accepted_count"] == 0
    )

    artifact.update(
        {
            "case_count": totals["case_count"],
            "proxy_accept_count": totals["proxy_accept_count"],
            "full_verifier_calls_baseline": totals["full_verifier_calls_baseline"],
            "full_verifier_calls_scheduler": totals["full_verifier_calls_scheduler"],
            "observed_full_verifier_call_reduction": totals[
                "observed_full_verifier_call_reduction"
            ],
            "proxy_accept_rate": totals["proxy_accept_rate"],
            "low_margin_escalation_rate": totals["low_margin_escalation_rate"],
            "full_verifier_call_reduction": full_verifier_call_reduction,
            "false_acceptance_count": totals["false_acceptance_count"],
            "false_acceptance_rate": totals["false_acceptance_rate"],
            "unknown_silently_accepted_count": totals["unknown_silently_accepted_count"],
            "repair_hint_reuse_rate": totals["repair_hint_reuse_rate"],
            "observed_verifier_cost_reduction_proxy": totals[
                "observed_verifier_cost_reduction_proxy"
            ],
            "verifier_cost_reduction_proxy": verifier_cost_reduction_proxy,
            "triage_claim_allowed": triage_claim_allowed,
            "honest_verdict": _honest_verdict(
                case_count=totals["case_count"],
                proxy_accept_count=totals["proxy_accept_count"],
                false_acceptance_count=totals["false_acceptance_count"],
                unknown_silently_accepted_count=totals["unknown_silently_accepted_count"],
                triage_claim_allowed=triage_claim_allowed,
            ),
            "scheduler_rows": scheduler_rows,
            "measurement_note": (
                "CPU-only replay over Exp 1369 semantic rows and Exp 1370 MCS "
                "repair hints.  The conservative policy accepts only high-margin "
                "SAT rows and escalates UNSAT, UNKNOWN, REPAIR_HINT, and any "
                "low-margin row to the full verifier.  Verifier-call reduction is "
                "reported as claimable only when false acceptance is zero and "
                "UNKNOWN is never silently accepted."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    exp1369_path: str | Path = DEFAULT_EXP1369_PATH,
    exp1370_path: str | Path = DEFAULT_EXP1370_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write an in-progress artifact first, then persist the terminal result."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )
    artifact = build_margin_aware_scheduler_artifact(
        exp1369_artifact=_read_json(_resolve(root, exp1369_path)),
        exp1370_artifact=_read_json(_resolve(root, exp1370_path)),
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _scheduler_row(
    *,
    row: Mapping[str, Any],
    repair_hint_row: Mapping[str, Any] | None,
    high_confidence_margin: float,
) -> dict[str, Any]:
    semantic_result = _semantic_state(row)
    semantic_margin = _semantic_margin(row=row, repair_hint_row=repair_hint_row)
    full_verifier_accepts = _full_verifier_accepts(row)
    high_confidence_sat = (
        semantic_result == "SAT"
        and bool(row.get("constraint_passed")) is True
        and semantic_margin >= high_confidence_margin
    )
    scheduler_action = "proxy_accept" if high_confidence_sat else "escalate_full_verifier"
    unknown_silently_accepted = semantic_result == "UNKNOWN" and scheduler_action == "proxy_accept"
    false_acceptance = scheduler_action == "proxy_accept" and not full_verifier_accepts
    low_margin_or_unknown_escalated = scheduler_action == "escalate_full_verifier" and (
        semantic_margin < high_confidence_margin or semantic_result == "UNKNOWN"
    )
    repair_hint_reused = (
        scheduler_action == "escalate_full_verifier"
        and repair_hint_row is not None
        and bool(repair_hint_row.get("repair_hint"))
        and bool(repair_hint_row.get("precision_match")) is True
    )

    return {
        "case_id": _case_id(row),
        "semantic_result": semantic_result,
        "expected_state": _expected_state(row),
        "certificate_state": _certificate_state(row),
        "constraint_passed": bool(row.get("constraint_passed")),
        "semantic_margin": semantic_margin,
        "margin_threshold": high_confidence_margin,
        "scheduler_action": scheduler_action,
        "full_verifier_baseline_accept": full_verifier_accepts,
        "false_acceptance": false_acceptance,
        "unknown_silently_accepted": unknown_silently_accepted,
        "low_margin_or_unknown_escalated": low_margin_or_unknown_escalated,
        "repair_hint_reused": repair_hint_reused,
        "repair_hint": str(repair_hint_row.get("repair_hint") or "")
        if repair_hint_row is not None
        else "",
    }


def _semantic_margin(
    *,
    row: Mapping[str, Any],
    repair_hint_row: Mapping[str, Any] | None,
) -> float:
    state = _semantic_state(row)
    if state == "SAT" and bool(row.get("constraint_passed")) is True:
        return 1.0
    if state == "UNSAT":
        return 0.35
    if state == "REPAIR_HINT" and repair_hint_row is not None:
        return 0.25
    if state == "UNKNOWN":
        return 0.0
    return 0.1


def _summarize_scheduler_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_count = len(rows)
    proxy_accept_count = sum(1 for row in rows if row["scheduler_action"] == "proxy_accept")
    full_verifier_calls_scheduler = case_count - proxy_accept_count
    false_acceptance_count = sum(1 for row in rows if row["false_acceptance"])
    unknown_silently_accepted_count = sum(1 for row in rows if row["unknown_silently_accepted"])
    escalated_count = full_verifier_calls_scheduler
    hint_reuse_count = sum(1 for row in rows if row["repair_hint_reused"])
    baseline_cost = case_count * FULL_VERIFIER_UNIT_COST
    scheduler_cost = (
        full_verifier_calls_scheduler * FULL_VERIFIER_UNIT_COST + case_count * PROXY_SCORE_UNIT_COST
    )

    return {
        "case_count": case_count,
        "proxy_accept_count": proxy_accept_count,
        "full_verifier_calls_baseline": case_count,
        "full_verifier_calls_scheduler": full_verifier_calls_scheduler,
        "observed_full_verifier_call_reduction": _rate(
            case_count - full_verifier_calls_scheduler,
            case_count,
        ),
        "proxy_accept_rate": _rate(proxy_accept_count, case_count),
        "low_margin_escalation_rate": _rate(
            sum(1 for row in rows if row["low_margin_or_unknown_escalated"]),
            case_count,
        ),
        "false_acceptance_count": false_acceptance_count,
        "false_acceptance_rate": _rate(false_acceptance_count, proxy_accept_count),
        "unknown_silently_accepted_count": unknown_silently_accepted_count,
        "repair_hint_reuse_rate": _rate(hint_reuse_count, escalated_count),
        "observed_verifier_cost_reduction_proxy": _cost_reduction(
            baseline_cost=baseline_cost,
            scheduler_cost=scheduler_cost,
        ),
    }


def _semantic_validator_rows(exp1369_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = exp1369_artifact.get("semantic_validator_rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _repair_hints_by_case(exp1370_artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = exp1370_artifact.get("repair_localization_rows", [])
    if not isinstance(rows, list):
        return {}
    return {
        _case_id(row): row
        for row in rows
        if isinstance(row, Mapping) and _case_id(row) and row.get("repair_hint")
    }


def _source_context(
    *,
    exp1369_artifact: Mapping[str, Any],
    exp1370_artifact: Mapping[str, Any],
    repair_hint_precision: float,
    repair_hint_precision_gate: float,
    high_confidence_margin: float,
) -> dict[str, Any]:
    return {
        "source_experiments": ["exp1369", "exp1370"],
        "exp1369_status": exp1369_artifact.get("status"),
        "exp1369_honest_verdict": exp1369_artifact.get("honest_verdict"),
        "exp1370_status": exp1370_artifact.get("status"),
        "exp1370_honest_verdict": exp1370_artifact.get("honest_verdict"),
        "exp1370_repair_hint_precision": repair_hint_precision,
        "required_repair_hint_precision_gate": repair_hint_precision_gate,
        "repair_hint_gate_passed": repair_hint_precision >= repair_hint_precision_gate,
        "high_confidence_margin": high_confidence_margin,
        "proxy_score_unit_cost": PROXY_SCORE_UNIT_COST,
        "full_verifier_unit_cost": FULL_VERIFIER_UNIT_COST,
    }


def _base_artifact(
    *,
    project_root: Path,
    run_date: str,
    status: str = "complete",
) -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": status,
        "case_count": 0,
        "proxy_accept_count": 0,
        "proxy_accept_rate": 0.0,
        "low_margin_escalation_rate": 0.0,
        "full_verifier_calls_baseline": 0,
        "full_verifier_calls_scheduler": 0,
        "observed_full_verifier_call_reduction": 0.0,
        "full_verifier_call_reduction": 0.0,
        "false_acceptance_count": 0,
        "false_acceptance_rate": 0.0,
        "unknown_silently_accepted_count": 0,
        "repair_hint_reuse_rate": 0.0,
        "observed_verifier_cost_reduction_proxy": 0.0,
        "verifier_cost_reduction_proxy": 0.0,
        "triage_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "terminal_blocker": None,
        "scheduler_rows": [],
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["exp1369", "exp1370"],
            "spec": "REQ-VERIFY-1371",
        },
    }


def _honest_verdict(
    *,
    case_count: int,
    proxy_accept_count: int,
    false_acceptance_count: int,
    unknown_silently_accepted_count: int,
    triage_claim_allowed: bool,
) -> str:
    if case_count == 0:
        return "no_exp1369_semantic_rows_available"
    if unknown_silently_accepted_count:
        return "margin_aware_scheduler_claim_blocked_unknown_silently_accepted"
    if false_acceptance_count:
        return "margin_aware_scheduler_claim_blocked_false_acceptance"
    if proxy_accept_count == 0:
        return "margin_aware_scheduler_complete_no_proxy_accepts"
    if triage_claim_allowed:
        return "margin_aware_scheduler_claim_allowed_zero_false_acceptance"
    return "margin_aware_scheduler_complete_claim_blocked"


def _full_verifier_accepts(row: Mapping[str, Any]) -> bool:
    return _semantic_state(row) == "SAT" and bool(row.get("constraint_passed")) is True


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id") or "")


def _expected_state(row: Mapping[str, Any]) -> str:
    return str(row.get("expected_state") or "").upper()


def _certificate_state(row: Mapping[str, Any]) -> str:
    return str(row.get("certificate_state") or row.get("dispatched_state") or "").upper()


def _semantic_state(row: Mapping[str, Any]) -> str:
    return str(row.get("semantic_result") or "").upper()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _cost_reduction(*, baseline_cost: float, scheduler_cost: float) -> float:
    if baseline_cost <= 0.0:
        return 0.0
    return round(max(0.0, (baseline_cost - scheduler_cost) / baseline_cost), 6)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, payload)


def main() -> None:  # pragma: no cover - thin CLI wrapper covered through run_experiment.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover - module execution hook.
    main()
