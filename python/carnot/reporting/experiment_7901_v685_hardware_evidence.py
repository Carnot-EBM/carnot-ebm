"""Reduce dated board receipts and optional current service measurements.

The board reader checks the original bytes. This code adds current custody
without issuing commands to any device. Spec ref: REQ-REPORT-7901-V685.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check
from carnot.reporting.experiment_7876_v683_hardware_evidence import (
    _load,
    read_evidence as reduce_boards,
)

PRIOR = "results/experiment_7889_v684_hardware_evidence.json"
SERVICE = "results/experiment_7900_v685_service_cost.json"
BOARDS = ("KV260", "PolarFire", "GateMate")
CRITERIA = {
    "KV260": "Authenticated SSH fabric transcript and full service timing for k<=5",
    "PolarFire": "Distinct FPGA fabric execution with device side timing",
    "GateMate": "Valid GM1Ax IDCODE after a dated physical or JTAG change",
}
NEXT_CHANGE = {
    "KV260": "Connect by SSH through kria; run a qualified k<=5 workload with transport and coupling timing",
    "PolarFire": "Deploy a distinct FPGA fabric workload and record device side timing",
    "GateMate": "Record a cable, power, port, board, or DirtyJTAG change; then read a valid GM1Ax IDCODE",
}


def _service(
    root: Path, run_date: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """An optional service receipt can add rows but cannot open a board gate."""
    value, digest = _load(root, SERVICE)
    checks = [_check(SERVICE, digest, "exists", True, digest is not None)]
    if value is not None:
        for field, expected in (
            ("experiment_id", 7900),
            ("task_id", "exp7900-service-cost"),
            ("run_date", run_date),
            ("flagged_adversarial", False),
            ("service_measurement_ready_score", 1),
        ):
            checks.append(_check(SERVICE, digest, field, expected, value.get(field)))
        checks.append(
            _check(
                SERVICE,
                digest,
                "verdict_class.eligible",
                True,
                value.get("verdict_class") in {"positive", "circular_positive", "null"},
            )
        )
        rows = value.get("rows")
        valid = (
            isinstance(rows, list)
            and bool(rows)
            and all(isinstance(row, dict) and row.get("operation") for row in rows)
        )
        checks.append(_check(SERVICE, digest, "rows.primitive_operations", True, valid))
    ready = value is not None and all(check["passed"] for check in checks)
    source = {
        "path": SERVICE,
        "sha256": digest,
        "date": value.get("run_date") if value else None,
        "role": "optional_current_service",
        "exposure_status": "qualified"
        if ready
        else "missing"
        if digest is None
        else "disqualified",
        "eligible": ready,
    }
    fit: list[dict[str, Any]] = []
    if ready and value is not None:
        for item in value["rows"]:
            row: dict[str, Any] = item
            for board in BOARDS:
                fit.append(
                    {
                        "board": board,
                        "operation": row["operation"],
                        "source_row_hash": canonical_hash(row),
                        "source_artifact_hash": digest,
                        "host_fraction": row.get("host_fraction"),
                        "transport_fraction": row.get("transport_fraction"),
                        "transfer_bytes": row.get("transfer_bytes"),
                        "connection_cost_ms": row.get("connection_cost_ms"),
                        "coupling_update_cost_ms": row.get("coupling_update_cost_ms"),
                        "fabric_boundary": "qualified k<=5 Ising only"
                        if board == "KV260"
                        else "Linux CPU only"
                        if board == "PolarFire"
                        else "blocked 0xffffffff",
                        "hardware_execution_measured": False,
                        "speedup": None,
                    }
                )
    return source, fit, [check for check in checks if not check["passed"]]


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7901-CUSTODY: retain exact historical board limits."""
    started = time.monotonic()
    base = reduce_boards(root, run_date)
    prior, digest = _load(root, PRIOR)
    checks = [_check(PRIOR, digest, "exists", True, digest is not None)]
    if digest is not None:
        checks.append(_check(PRIOR, digest, "schema_json", "dict", type(prior).__name__))
    if prior is not None:
        for field, expected in (
            ("experiment_id", 7889),
            ("task_id", "exp7889-hardware-evidence"),
            ("milestone", "2026.09.684"),
            ("verdict_class", "disqualified"),
        ):
            checks.append(_check(PRIOR, digest, field, expected, prior.get(field)))
    boards = deepcopy(base["board_rows"])
    names = [row.get("board") for row in boards if isinstance(row, dict)]
    checks.append(_check(PRIOR, digest, "board_rows.names", list(BOARDS), names))
    if names != list(BOARDS):
        boards = [{"board": name, "source_path": None, "source_hash": None} for name in BOARDS]
    for row in boards:
        board = row["board"]
        row.update(
            {
                "family": board,
                "arm": "historical_accounting",
                "seed": None,
                "claim_class": "historical",
                "current_hardware_execution": False,
                "measured_current_latency_ms": None,
                "terminal_criterion_met": False,
                "terminal_criterion": CRITERIA[board],
                "receipt_date": row.get("last_authenticated_evidence_date"),
                "receipt_hash": row.get("source_hash"),
                "next_operator_or_device_change": NEXT_CHANGE[board],
            }
        )
    failures = [*base["gate_check_summary"], *(row for row in checks if not row["passed"])]
    service, feasibility, service_failures = _service(root, run_date)
    sources = deepcopy(base["source_artifact_hashes"])
    sources[PRIOR] = {
        "path": PRIOR,
        "sha256": digest,
        "date": prior.get("run_date") if prior else None,
        "role": "disqualified_historical_runner",
        "exposure_status": "historical_read_only",
    }
    sources[SERVICE] = service
    old_failed = (
        [
            deepcopy(item)
            for item in prior.get("validation_receipts", {}).get("checks", [])
            if isinstance(item, dict)
            and item.get("classification") == "required"
            and item.get("passed") is False
        ]
        if prior
        else []
    )
    ready = not failures
    result: dict[str, Any] = {
        "experiment_id": 7901,
        "task_id": "exp7901-hardware-evidence",
        "milestone": "2026.09.685",
        "run_date": run_date,
        "honest_verdict": "complete_null_historical_board_scope"
        if ready
        else "complete_blocked_board_source_custody",
        "verdict_class": "null" if ready else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": boards,
        "rows_checksum": canonical_hash(boards),
        "board_rows": boards,
        "terminal_receipt_hashes": {str(row["source_path"]): row["source_hash"] for row in boards},
        "changed_physical_receipts": [],
        "hardware_evidence_ready_score": int(ready),
        "workload_attachment_available": bool(feasibility),
        "workload_feasibility_rows": feasibility,
        "workload_attachment_operands": service_failures,
        "current_device_execution_count": 0,
        "hardware_speedup_claimed": False,
        "source_artifact_hashes": sources,
        "preconditions_checked": [*base["preconditions_checked"], *checks],
        "historical_required_failures": old_failed,
        "historical_failures": deepcopy(base["historical_failures"]),
        "sample_size_budget": {
            "intended": 3,
            "eligible": 2 if ready else 0,
            "started": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "excluded": 1,
            "independent": 0,
        },
        "acceptance_gate_results": {
            "validity": ready,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": {"evidence_read_s": time.monotonic() - started},
        "random_seed": 0,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {"model_loads": 0, "generation_calls": 0, "tokens": 0},
        "trained_head_specs": [],
        "verifier_is_oracle": False,
        "claim_scope": {
            "board_evidence": "historical custody",
            "current_measurement": "host receipt analysis only",
            "workload": service["exposure_status"],
            "hardware_advantage": "unmeasured",
            "AMD_NPU": "unqualified",
            "Extropic_TSU_Z1T": "unqualified",
            "optical_hardware": "unqualified",
            "GateMate": "blocked_0xffffffff",
        },
        "validation_receipts": {"checks": [], "required_checks_passed": False},
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": None,
        "resolved_imports": {},
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "rows": boards,
            "date": run_date,
            "seed": 0,
            "code": sha256_file(Path(__file__)),
        }
    )
    result["field_principles"] = {
        key: f"Keep {key.replace('_', ' ')} attributable to this run and its source bytes."
        for key in result
    }
    result["field_principles"]["acceptance_gate_results"] = {
        "validity": "Exact dated board sources must pass.",
        "readiness": "Owned required checks must pass.",
        "probability_quality": "No probability quality was measured.",
        "decision_benefit": "No decision benefit was measured.",
        "retention": "No retained learning was measured.",
        "efficiency": "No board efficiency was measured.",
    }
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7901-WORKLOAD: recompute rows from original bytes."""
    fresh = read_evidence(root, candidate["run_date"])
    if fresh["gate_check_summary"] != candidate.get("gate_check_summary"):
        raise ValueError("gate_operands_changed")
    if (
        fresh["rows"] != candidate.get("rows")
        or fresh["board_rows"] != candidate.get("board_rows")
        or fresh["rows_checksum"] != candidate.get("rows_checksum")
    ):
        raise ValueError("rows_changed")
    if fresh["terminal_receipt_hashes"] != candidate.get("terminal_receipt_hashes"):
        raise ValueError("terminal_receipt_changed")
    if fresh["workload_feasibility_rows"] != candidate.get("workload_feasibility_rows") or fresh[
        "workload_attachment_operands"
    ] != candidate.get("workload_attachment_operands"):
        raise ValueError("workload_rows_changed")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}
