"""REQ-REPORT-7951: authenticate custody before mapping measured host costs.

Missing history must remain unknown. A service estimate cannot supply a missing
board capability or establish that a device executed the workload.
"""

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from coverage import CoverageData

from carnot.reporting import experiment_7938_v688_hardware_evidence as history
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check
from carnot.reporting.experiment_7876_v683_hardware_evidence import _load

ROOT = history.ROOT
PRIOR = history.PRIOR
SERVICE = "results/experiment_7950_v689_service_cost.json"
BOARDS = ("KV260", "PolarFire", "GateMate")
OPERATIONS = (
    "source_projection",
    "feature_extraction",
    "source_head_forward",
    "energy_evaluation",
    "typed_policy",
    "storage",
    "response_serialization",
)


def _board_valid(board: dict[str, Any]) -> bool:
    """A board name alone cannot supply a missing execution scope or receipt."""
    return (
        board.get("board") in BOARDS
        and isinstance(board.get("scope"), str)
        and bool(board["scope"])
        and isinstance(board.get("source_hash"), str)
        and board["source_hash"].startswith("sha256:")
        and board.get("status") == "historical_read_only"
        and board.get("custody_valid") is not False
    )


def _row_valid(row: Any) -> bool:
    """Absent timings stay unknown; impossible measurements reject attachment."""
    return (
        isinstance(row, dict)
        and isinstance(row.get("operation"), str)
        and bool(row["operation"])
        and all(
            row.get(field) is None or history._number(row[field], upper=upper)
            for field, upper in (
                ("duration_s", float("inf")),
                ("kernel_share", 1.00000001),
                ("host_fraction", 1.00000001),
            )
        )
        and (
            row.get("transfer_bytes") is None
            or type(row["transfer_bytes"]) is int
            and row["transfer_bytes"] >= 0
        )
    )


def operation_map(
    board_rows: list[dict[str, Any]], rows: list[dict[str, Any]], digest: str | None
) -> list[dict[str, Any]]:
    """Keep failed custody unknown before applying any kernel acceleration model."""
    mapping = []
    for source in rows:
        for board in board_rows:
            name = board.get("board")
            valid = _board_valid(board)
            compatible = (
                valid
                and name == "KV260"
                and board.get("processor_class") == "fpga_fabric"
                and board.get("k_max") == 5
                and source.get("representation") == "quadratic_ising"
                and type(source.get("k_max")) is int
                and 0 < source["k_max"] <= 5
            )
            timed = (
                valid
                and name != "GateMate"
                and history._number(source.get("duration_s"))
                and history._number(source.get("kernel_share"), upper=1.00000001)
            )
            fraction = (source["kernel_share"] if compatible else 0) if timed else None
            mapping.append(
                {
                    "board": name,
                    "operation": source["operation"],
                    "source_row": deepcopy(source),
                    "source_row_hash": canonical_hash(source),
                    "source_artifact_hash": digest,
                    "board_receipt_hash": board.get("source_hash"),
                    "board_custody_valid": valid,
                    "duration_s": source.get("duration_s"),
                    "measured_host_work_s": source.get("duration_s"),
                    "kernel_share": source.get("kernel_share"),
                    "host_fraction": source.get("host_fraction"),
                    "transfer_bytes": source.get("transfer_bytes"),
                    "topology": source.get("topology"),
                    "fabric_compatible": compatible,
                    "offload_fraction": fraction,
                    "placement_constraint": board["scope"] if valid else None,
                    "disposition": "blocked_unknown_custody"
                    if not valid
                    else "blocked_physical_jtag_0xffffffff"
                    if name == "GateMate"
                    else "linux_cpu_only"
                    if name == "PolarFire"
                    else "historical_quadratic_ising_candidate"
                    if compatible
                    else "host_operation_no_qualified_fabric_mapping",
                    "comparison_class": "estimate",
                    "modeled_100x_kernel_bound": 1 / (1 - fraction + fraction / 100)
                    if fraction is not None
                    else None,
                    "amdahl_estimate_upper_bound": 1 / (1 - fraction)
                    if fraction is not None and fraction < 1
                    else None,
                    "ideal_ceiling_status": "unbounded"
                    if fraction == 1
                    else "finite"
                    if fraction is not None
                    else "unmeasured",
                    "comparison_basis": "Modeled 100x compatible kernel and ideal Amdahl ceiling assume no added transfer cost; measured bytes remain host evidence.",
                    "hardware_execution_measured": False,
                    "speedup": None,
                    "power_claim": None,
                }
            )
    return mapping


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7951-CUSTODY: missing optional work does not close custody."""
    started = time.monotonic()
    result = history.old.read_evidence(root, run_date)
    checks = history._authenticate(root, result)
    prior, digest = _load(root, PRIOR)
    boards = (prior or {}).get("board_rows")
    checks.append(
        _check(
            PRIOR,
            digest,
            "board_rows.names",
            list(BOARDS),
            [r.get("board") if isinstance(r, dict) else None for r in boards]
            if isinstance(boards, list)
            else None,
        )
    )
    for index, name in enumerate(BOARDS):
        row = (
            boards[index]
            if isinstance(boards, list) and index < len(boards) and isinstance(boards[index], dict)
            else {}
        )
        for field in ("scope", "source_hash"):
            checks.append(
                _check(
                    PRIOR,
                    digest,
                    f"board_rows.{name}.{field}",
                    result["board_rows"][index].get(field),
                    row.get(field),
                )
            )
        checks.append(_check(PRIOR, digest, f"board_rows.{name}.valid", True, _board_valid(row)))
    result["preconditions_checked"].extend(checks)
    result["gate_check_summary"].extend(r for r in checks if not r["passed"])
    ready = not result["gate_check_summary"]
    for board in result["board_rows"]:
        board["custody_valid"] = ready
        if not ready:
            board.update(
                status="blocked_custody",
                scope=None,
                eligible=0,
                excluded=True,
                blocker="required_custody_invalid",
            )
    result["typed_blocker_rows"] = (
        []
        if ready
        else [
            {
                "board": name,
                "type": "required_custody_blocked",
                "placement": None,
                "failed_operands": deepcopy(result["gate_check_summary"]),
            }
            for name in BOARDS
        ]
    )
    source, service_hash = _load(root, SERVICE)
    operands = [_check(SERVICE, service_hash, "exists", True, service_hash is not None)]
    rows = (source or {}).get("rows")
    if source is not None:
        operands.extend(
            _check(SERVICE, service_hash, field, expected, source.get(field))
            for field, expected in (
                ("experiment_id", 7950),
                ("task_id", "exp7950-service-cost"),
                ("milestone", "2026.09.689"),
                ("run_date", run_date),
                ("service_measurement_ready_score", 1),
                ("flagged_adversarial", False),
            )
        )
        operands.extend(
            [
                _check(
                    SERVICE,
                    service_hash,
                    "verdict_class.eligible",
                    True,
                    source.get("verdict_class") in {"positive", "circular_positive", "null"},
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "validation_receipts.required_checks_passed",
                    True,
                    source.get("validation_receipts", {}).get("required_checks_passed"),
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "rows.primitive_operations",
                    True,
                    isinstance(rows, list) and bool(rows) and all(_row_valid(r) for r in rows),
                ),
            ]
        )
    attached = all(r["passed"] for r in operands)
    mapping = operation_map(
        result["board_rows"],
        rows if attached else [{"operation": op} for op in OPERATIONS],
        service_hash if attached else None,
    )
    result["source_artifact_hashes"].pop(history.old.SERVICE, None)
    result["source_artifact_hashes"][SERVICE] = {
        "path": SERVICE,
        "sha256": service_hash,
        "role": "optional_current_whole_service",
        "exposure_status": "exposed_development" if attached else "unavailable",
        "eligible": attached,
    }
    failed_path = "results/experiment_7938_v688_hardware_evidence.json"
    failed, failed_hash = _load(root, failed_path)
    result["source_artifact_hashes"][failed_path] = {
        "path": failed_path,
        "sha256": failed_hash,
        "role": "historical_failed_runner",
        "exposure_status": "historical_read_only",
    }
    result["historical_required_failures"].extend(
        deepcopy(r)
        for r in (failed or {}).get("validation_receipts", {}).get("checks", [])
        if r.get("classification") == "required" and not r.get("passed")
    )
    result.update(
        {
            "experiment_id": 7951,
            "task_id": "exp7951-hardware-evidence",
            "milestone": "2026.09.689",
            "honest_verdict": "complete_null_historical_board_scope"
            if ready
            else "complete_blocked_board_source_custody",
            "verdict_class": "null" if ready else "blocked",
            "hardware_evidence_ready_score": int(ready),
            "required_custody_valid": ready,
            "operation_map_validity": "qualified_historical_custody"
            if ready
            else "blocked_unknown_placement",
            "workload_attachment_available": attached,
            "workload_attachment_operands": operands,
            "workload_feasibility_rows": mapping if attached else [],
            "operation_map": mapping,
            "measured_offload_ceiling_available": any(
                r["modeled_100x_kernel_bound"] is not None for r in mapping
            ),
            "measured_offload_ceiling": {
                "status": "estimate_from_measured_host_shares"
                if attached and ready
                else "unmeasured",
                "comparison_class": "estimate",
                "hardware_speedup": None,
            },
            "optional_workload_upstream_failures": deepcopy(
                (source or {}).get("gates_evaluated", [])
            ),
            "primary_resolution_receipt": {"path": None, "binding": "exact final bytes in sidecar"},
            "unqualified_substrates": ["NPU", "TSU"],
            "retire_if_same_verdict": {
                "prior_experiment_id": 7926,
                "identical_failure": True,
                "reason": "Retire unchanged historical board dispositions; typed missing data does not reopen device execution.",
            },
        }
    )
    result["rows"] = deepcopy(result["board_rows"])
    result["rows_checksum"] = canonical_hash(result["rows"])
    result["claim_scope"].update(
        workload="exposed_development" if attached else "unmeasured",
        mapping="host estimates; no board workload execution",
    )
    result["acceptance_gate_results"].update(validity=ready, readiness=int(ready), calibration=None)
    result["sample_size_budget"].update(eligible=2 if ready else 0, excluded=1 if ready else 3)
    result["reproducibility_checksum"] = canonical_hash(
        {
            "sources": result["source_artifact_hashes"],
            "rows": result["rows"],
            "mapping": mapping,
            "code": sha256_file(Path(__file__)),
            "date": run_date,
            "seed": 0,
        }
    )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"] = {"evidence_read_s": result["duration_s"]}
    result["field_principles"].update(
        {
            name: f"Bind {name.replace('_', ' ')} to exact evidence within historical scope."
            for name in result
            if name not in result["field_principles"]
        }
    )
    result["field_principles"].update(
        typed_blocker_rows="Failed custody retains exact operands and unknown placement.",
        required_custody_valid="All required historical inputs authenticate independently of optional service.",
        operation_map_validity="Missing scope or receipt never supplies accelerator capability.",
        operation_map="Explicit quadratic Ising and k<=5 constrain host-only acceleration estimates.",
        measured_offload_ceiling="Measured host shares bound modeled kernels; no board timing or power was measured.",
        hardware_evidence_ready_score="Authenticated historical custody and all owned checks; optional workload is independent.",
    )
    result["field_principles"]["acceptance_gate_results"]["calibration"] = (
        "No probability calibration was measured."
    )
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7951-VALIDATION: reduce claims from current primitive bytes."""
    fresh = read_evidence(root, candidate["run_date"])
    fields = (
        "experiment_id",
        "task_id",
        "milestone",
        "rows",
        "board_rows",
        "rows_checksum",
        "terminal_receipt_hashes",
        "source_artifact_hashes",
        "sample_size_budget",
        "typed_blocker_rows",
        "required_custody_valid",
        "operation_map_validity",
        "operation_map",
        "workload_feasibility_rows",
        "workload_attachment_operands",
        "workload_attachment_available",
        "measured_offload_ceiling",
        "measured_offload_ceiling_available",
        "optional_workload_upstream_failures",
        "claim_scope",
        "current_device_execution_count",
        "hardware_speedup_claimed",
        "changed_physical_receipts",
        "MODEL_SPECS",
        "model_specs",
        "model_invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "scheduled_device_operations",
        "trained_head_specs",
        "target_model",
        "unqualified_substrates",
    )
    for field in fields:
        if fresh[field] != candidate.get(field):
            raise ValueError(f"claims_changed:{field}")
    gates = candidate.get("gate_check_summary")
    if (
        not isinstance(gates, list)
        or any(not isinstance(r, dict) for r in gates)
        or fresh["gate_check_summary"]
        != [r for r in gates if r.get("upstream_id") != "owned_validation"]
    ):
        raise ValueError("claims_changed:gate_operands")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}


def check_coverage_shards(paths: list[Path]) -> None:
    """Reject empty or foreign shards before combining owned coverage."""
    from carnot.reporting.validation_7951 import MEASURED

    if not paths:
        raise ValueError("empty_coverage_shards")
    owned = {str((ROOT / name).resolve()) for name in MEASURED}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"empty_coverage_shard:{path}")
        data = CoverageData(basename=str(path))
        data.read()
        if not any(data.lines(name) for name in data.measured_files() if name in owned):
            raise ValueError(f"empty_coverage_shard:{path}")


def qualify(root: Path, run_date: str, output: Path, raw_root: Path) -> int:
    """Reuse bounded validation and checked publication without board commands."""
    from carnot.reporting.publication_7951 import qualify as publish

    return publish(root, run_date, output, raw_root)
