"""REQ-REPORT-7964: keep board custody separate from host workload estimates.

Exact historical bytes qualify custody. Optional measured service work cannot
supply missing accelerator scope or prove a device executed that work.
"""

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from coverage import CoverageData

from carnot.reporting import experiment_7951_v689_hardware_evidence as history
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check
from carnot.reporting.experiment_7876_v683_hardware_evidence import _load

ROOT = history.ROOT
PRIOR = "results/experiment_7951_v689_hardware_evidence.json"
SERVICE = "results/experiment_7963_v690_service_cost.json"
PRIOR_HASH = "sha256:aad42f02b02ae30ddfd06f1263b8985147686a36b129eb3455f6da039b27c7f3"
TERMINAL_HASH = "sha256:966cf5182c673895e944ba9d6bc177e3f15044ad3b481f88ca9f8ccdf9517f4b"
BOARDS = history.BOARDS
OPERATIONS = history.OPERATIONS
operation_map = history.operation_map


def _row_valid(row: Any) -> bool:
    """Recorded spans and integer bytes must support each imported host cost."""
    if not history._row_valid(row):
        return False
    if type(row.get("transfer_bytes")) is not int or row["transfer_bytes"] < 0:
        return False
    duration = row.get("duration_s")
    if duration is None:
        return row.get("start_s") is None and row.get("end_s") is None
    start, end = row.get("start_s"), row.get("end_s")
    return (
        history.history._number(start)
        and history.history._number(end)
        and end >= start
        and abs(end - start - duration) <= max(1e-9, duration * 1e-6)
    )


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7964-CUSTODY: authenticate once without device access."""
    started = time.monotonic()
    if run_date != "20261001":
        raise ValueError("run_date_mismatch")
    prior, digest = _load(root, PRIOR)
    # The qualified old reader supplies a complete blocked schema for missing bytes.
    result = deepcopy(prior) if digest == PRIOR_HASH else history.read_evidence(root, "20260930")
    for field in (
        "raw_rows_path",
        "raw_rows_sha256",
        "validation_evidence_files",
        "validation_command_manifest_sha256",
    ):
        result.pop(field, None)
    checks = [_check(PRIOR, digest, "sha256", PRIOR_HASH, digest)]
    sources = {}
    if prior is not None:
        label = str(prior.get("terminal_validation_sidecar_path", ""))
        terminal, terminal_hash = _load(root, label)
        checks.extend(
            [
                _check(label, terminal_hash, "sha256", TERMINAL_HASH, terminal_hash),
                _check(
                    label,
                    terminal_hash,
                    "candidate_sha256",
                    digest,
                    (terminal or {}).get("candidate_sha256"),
                ),
                _check(
                    PRIOR,
                    digest,
                    "hardware_evidence_ready_score",
                    1,
                    prior.get("hardware_evidence_ready_score"),
                ),
            ]
        )
        sources[label] = {
            "path": label,
            "sha256": terminal_hash,
            "role": "historical_terminal_validation",
        }
        receipts = {
            **{r["log_path"]: r["log_sha256"] for r in (terminal or {}).get("reports", [])},
            **prior.get("terminal_receipt_hashes", {}),
            **{
                name: row["sha256"]
                for name, row in prior.get("source_artifact_hashes", {}).items()
                if row.get("role")
                in {
                    "dated_board_receipt",
                    "historical_raw_transcript",
                    "qualified_historical_board_custody",
                }
            },
        }
        for name, expected in receipts.items():
            path = root / name
            actual = sha256_file(path) if path.is_file() else None
            checks.append(_check(name, actual, "sha256", expected, actual))
            sources[name] = {
                "path": name,
                "sha256": actual,
                "role": "authenticated_historical_custody",
            }
    ready = all(row["passed"] for row in checks)
    boards = deepcopy(result["board_rows"])
    for board in boards:
        if not ready:
            board.update(
                custody_valid=False,
                status="blocked_custody",
                scope=None,
                eligible=0,
                excluded=True,
                blocker="required_custody_invalid",
            )
    gates = [row for row in checks if not row["passed"]]
    source, service_hash = _load(root, SERVICE)
    operands = [_check(SERVICE, service_hash, "exists", True, service_hash is not None)]
    rows = (source or {}).get("rows")
    if service_hash is not None:
        operands.extend(
            _check(SERVICE, service_hash, field, expected, (source or {}).get(field))
            for field, expected in (
                ("experiment_id", 7963),
                ("task_id", "exp7963-service-cost"),
                ("milestone", "2026.09.690"),
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
                    (source or {}).get("verdict_class")
                    in {"positive", "circular_positive", "null"},
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "validation_receipts.required_checks_passed",
                    True,
                    (source or {}).get("validation_receipts", {}).get("required_checks_passed"),
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "rows.measured_spans_and_bytes",
                    True,
                    isinstance(rows, list) and bool(rows) and all(_row_valid(row) for row in rows),
                ),
            ]
        )
    attached = all(row["passed"] for row in operands)
    mapped = operation_map(
        boards,
        rows if attached else [{"operation": op} for op in OPERATIONS],
        service_hash if attached else None,
    )
    if not attached:
        for row in mapped:
            row.update(placement_constraint=None, disposition="placement_unknown_missing_service")
    sources[PRIOR] = {"path": PRIOR, "sha256": digest, "role": "qualified_historical_board_custody"}
    sources[SERVICE] = {
        "path": SERVICE,
        "sha256": service_hash,
        "role": "optional_current_whole_service",
        "eligible": attached,
    }
    result.update(
        experiment_id=7964,
        task_id="exp7964-hardware-evidence",
        milestone="2026.09.690",
        run_date=run_date,
        current_execution_date=run_date,
        honest_verdict="complete_null_historical_board_scope"
        if ready
        else "complete_blocked_board_source_custody",
        verdict_class="null" if ready else "blocked",
        flagged_adversarial=False,
        hardware_evidence_ready_score=int(ready),
        required_custody_valid=ready,
        preconditions_checked=checks,
        gate_check_summary=gates,
        source_artifact_hashes=sources,
        board_rows=boards,
        rows=deepcopy(boards),
        rows_checksum=canonical_hash(boards),
        typed_blocker_rows=[]
        if ready
        else [
            {
                "board": name,
                "type": "required_custody_blocked",
                "placement": None,
                "failed_operands": deepcopy(gates),
            }
            for name in BOARDS
        ],
        operation_map_validity="qualified_historical_custody"
        if ready
        else "blocked_unknown_placement",
        workload_attachment_available=attached,
        workload_attachment_operands=operands,
        workload_feasibility_rows=mapped if attached else [],
        operation_map=mapped,
        measured_offload_ceiling_available=any(
            row["modeled_100x_kernel_bound"] is not None for row in mapped
        ),
        measured_offload_ceiling={
            "status": "estimate_from_measured_host_shares" if attached and ready else "unmeasured",
            "comparison_class": "estimate",
            "hardware_speedup": None,
        },
        optional_workload_upstream_failures=deepcopy((source or {}).get("gates_evaluated", [])),
        validation_receipts={"checks": [], "required_checks_passed": False},
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        primary_resolution_receipt={"path": None, "binding": "exact final bytes in sidecar"},
        terminal_validation_sidecar_path=None,
        retire_if_same_verdict={
            "prior_experiment_id": 7951,
            "identical_failure": True,
            "reason": "Unchanged historical board scopes retire; no device retry is scheduled.",
        },
    )
    result["claim_scope"].update(
        workload="exposed_development" if attached else "unmeasured",
        oracle_distinct_corrigendum="September 28 correction preserved; GAP-ORACLE-DISTINCT remains open",
    )
    result["acceptance_gate_results"].update(validity=ready, readiness=int(ready))
    result["sample_size_budget"].update(eligible=2 if ready else 0, excluded=1 if ready else 3)
    result["cited_upstream_artifacts"] = [
        {
            "experiment_id": 7951,
            "fields_imported": [
                "board_rows",
                "terminal_receipt_hashes",
                "historical_required_failures",
                "repository_health",
            ],
            "path": PRIOR,
            "sha256": digest,
        },
        *(
            [
                {
                    "experiment_id": 7963,
                    "fields_imported": ["rows"],
                    "path": SERVICE,
                    "sha256": service_hash,
                }
            ]
            if attached
            else []
        ),
    ]
    result["reproducibility_checksum"] = canonical_hash(
        {
            "sources": sources,
            "rows": boards,
            "mapping": mapped,
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
        cited_upstream_artifacts="Inherited facts retain original producer identity and exact bytes.",
        workload_attachment_operands="Only current eligible service rows with measured spans and bytes attach.",
    )
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7964-VALIDATION: recompute claims in a fresh reader."""
    fresh = read_evidence(root, candidate["run_date"])
    variable = {
        "duration_s",
        "phase_spans",
        "field_principles",
        "reproducibility_checksum",
        "validation_receipts",
        "validation_command_manifest_path",
        "observed_child_commands",
        "coverage_statement_counts",
        "resolved_imports",
        "repository_health",
        "primary_resolution_receipt",
        "terminal_validation_sidecar_path",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "hardware_evidence_ready_score",
        "acceptance_gate_results",
        "gate_check_summary",
    }
    for field in fresh.keys() - variable:
        if fresh[field] != candidate.get(field):
            raise ValueError(f"claims_changed:{field}")
    gates = candidate.get("gate_check_summary")
    if (
        not isinstance(gates, list)
        or any(not isinstance(row, dict) for row in gates)
        or fresh["gate_check_summary"]
        != [row for row in gates if row.get("upstream_id") != "owned_validation"]
    ):
        raise ValueError("claims_changed:gate_operands")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}


def check_coverage_shards(paths: list[Path]) -> None:
    """Empty or unrelated measurements cannot satisfy owned coverage."""
    from carnot.reporting.validation_7964 import MEASURED

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
    """Reuse bounded child supervision and exact-byte atomic publication."""
    from carnot.reporting.publication_7964 import qualify as publish

    return publish(root, run_date, output, raw_root)
