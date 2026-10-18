"""Reduce dated board receipts for the current hardware continuity question.

Old board measurements retain their original scope. This code reads their
bytes and records whether the next physical criterion was actually met.
Spec refs: REQ-REPORT-7889 and SCENARIO-REPORT-7889-CUSTODY/WORKLOAD.
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
    read_evidence as old_board_reduce,
)

PRIOR = "results/experiment_7876_v683_hardware_evidence.json"
SERVICE = "results/experiment_7888_v684_service_cost.json"
BOARDS = ("KV260", "PolarFire", "GateMate")


def _service(
    root: Path, run_date: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use only a current ready producer with primitive traffic rows."""
    value, digest = _load(root, SERVICE)
    operands = [_check(SERVICE, digest, "exists", True, digest is not None)]
    if value is not None:
        for field, expected in (
            ("experiment_id", 7888),
            ("task_id", "exp7888-service-cost"),
            ("run_date", run_date),
            ("flagged_adversarial", False),
            ("service_cost_ready_score", 1),
        ):
            operands.append(_check(SERVICE, digest, field, expected, value.get(field)))
        operands.append(
            _check(
                SERVICE,
                digest,
                "verdict_class.eligible",
                True,
                value.get("verdict_class") in {"positive", "circular_positive", "null"},
            )
        )
        rows = value.get("rows")
        valid_rows = (
            isinstance(rows, list)
            and bool(rows)
            and all(
                isinstance(row, dict) and row.get("operation") and "traffic_bytes" in row
                for row in rows
            )
        )
        operands.append(_check(SERVICE, digest, "rows.primitive_traffic", True, valid_rows))
    eligible = value is not None and all(item["passed"] for item in operands)
    source = {
        "path": SERVICE,
        "sha256": digest,
        "date": value.get("run_date") if value else None,
        "role": "optional_current_service",
        "exposure_status": "qualified"
        if eligible
        else "missing"
        if digest is None
        else "disqualified",
        "eligible": eligible,
    }
    attached = (
        [
            {
                **deepcopy(row),
                "source_row_hash": canonical_hash(row),
                "source_artifact_hash": digest,
                "claim_class": "current_host_measurement",
                "hardware_execution_measured": False,
                "speedup": None,
            }
            for row in value["rows"]
        ]
        if eligible and value
        else []
    )
    return source, attached, [item for item in operands if not item["passed"]]


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7889-CUSTODY: preserve board limits and old failures."""
    started = time.monotonic()
    base = old_board_reduce(root, run_date)
    prior, digest = _load(root, PRIOR)
    checks = [_check(PRIOR, digest, "exists", True, digest is not None)]
    if digest is not None:
        checks.append(_check(PRIOR, digest, "schema_json", "dict", type(prior).__name__))
    if prior is not None:
        for field, expected in (
            ("experiment_id", 7876),
            ("task_id", "exp7876-hardware-evidence"),
            ("milestone", "2026.09.683"),
            ("verdict_class", "disqualified"),
        ):
            checks.append(_check(PRIOR, digest, field, expected, prior.get(field)))
    failures = [*base["gate_check_summary"], *(row for row in checks if not row["passed"])]
    boards = deepcopy(base["board_rows"])
    if not boards:
        boards = [{"board": name, "source_path": None, "source_hash": None} for name in BOARDS]
    for row in boards:
        row["family"] = row["board"]
        row["arm"] = "historical_accounting"
        row["seed"] = None
        row["claim_class"] = "historical"
        row["current_hardware_execution"] = False
        row["measured_current_latency_ms"] = None
        row["terminal_criterion_met"] = False
        row["terminal_criterion"] = {
            "KV260": "Authenticated fabric transcript and complete service timing for k<=5 workload",
            "PolarFire": "Authenticated FPGA fabric execution and device-side timing",
            "GateMate": "Valid GM1Ax IDCODE after documented physical change",
        }[row["board"]]
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
            if item.get("classification") == "required" and item.get("passed") is False
        ]
        if prior
        else []
    )
    ready = not failures
    result: dict[str, Any] = {
        "experiment_id": 7889,
        "task_id": "exp7889-hardware-evidence",
        "milestone": "2026.09.684",
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
        "sample_size_budget": {
            "intended": 3,
            "eligible": 2 if ready else 0,
            "started": 0,
            "completed": 0,
            "censored": 0,
            "excluded": 1,
            "independent": 0,
        },
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
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7889-CLI: recompute every board and service row."""
    fresh = read_evidence(root, candidate["run_date"])
    if fresh["gate_check_summary"] != candidate.get("gate_check_summary"):
        raise ValueError("gate_operands_changed")
    if fresh["rows"] != candidate.get("rows") or fresh["rows_checksum"] != candidate.get(
        "rows_checksum"
    ):
        raise ValueError("rows_changed")
    if fresh["terminal_receipt_hashes"] != candidate.get("terminal_receipt_hashes"):
        raise ValueError("terminal_receipt_changed")
    if fresh["workload_feasibility_rows"] != candidate.get("workload_feasibility_rows"):
        raise ValueError("workload_rows_changed")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}
