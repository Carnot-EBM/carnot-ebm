"""Map host service operations without enlarging historical board claims.

REQ-REPORT-7938-V688. A timing fraction can bound possible savings, but it
cannot establish device execution or turn a neural head into an Ising kernel.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from coverage import CoverageData

from carnot.reporting import experiment_7926_v687_hardware_evidence as old
from carnot.reporting import qualification_7926 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check
from carnot.reporting.experiment_7876_v683_hardware_evidence import _load
from carnot.reporting.primary_publication import publish_primary, reader_receipt

ROOT = old.ROOT
PRIOR = "results/experiment_7926_v687_hardware_evidence.json"
SERVICE = "results/experiment_7937_service_cost.json"
PRIOR_HASH = "sha256:ee7844ad489757d59e4c64787099c18ca132338e4a0dafc228fbf24bff4d9a99"
TERMINAL_HASH = "sha256:793b6c5ef5fcd1df6ed81efce71854574e80566777f56219034114d7db600df5"
OPERATIONS = (
    "request_transport",
    "feature_extraction",
    "source_head_forward",
    "energy_evaluation",
    "policy_decision",
    "durable_update",
    "response_serialization",
)


def _number(value: Any, lower: float = 0, upper: float = math.inf) -> bool:
    """Reject booleans and nonfinite values before division or timing comparisons."""
    return type(value) in {int, float} and math.isfinite(value) and lower <= value < upper


def _row_valid(row: Any) -> bool:
    """Missing shares remain unknown; impossible durations or bytes reject attachment."""
    return (
        isinstance(row, dict)
        and isinstance(row.get("operation"), str)
        and bool(row["operation"])
        and _number(row.get("duration_s"))
        and (row.get("kernel_share") is None or _number(row["kernel_share"], upper=1))
        and (row.get("host_fraction") is None or _number(row["host_fraction"], upper=1.00000001))
        and (
            row.get("transfer_bytes") is None
            or (type(row["transfer_bytes"]) is int and row["transfer_bytes"] >= 0)
        )
    )


def operation_map(
    board_rows: list[dict[str, Any]], rows: list[dict[str, Any]], digest: str | None
) -> list[dict[str, Any]]:
    """Representation and chain size constrain placement before timing can matter."""
    mapping = []
    for source in rows:
        quadratic = (
            source.get("representation") == "quadratic_ising"
            and type(source.get("k_max")) is int
            and 0 < source["k_max"] <= 5
        )
        for board in board_rows:
            name = board["board"]
            compatible = name == "KV260" and quadratic
            share = source.get("kernel_share")
            fraction = share if compatible else 0 if share is not None else None
            ceiling = 1 / (1 - fraction) if fraction is not None and name != "GateMate" else None
            mapping.append(
                {
                    "board": name,
                    "operation": source["operation"],
                    "source_row": deepcopy(source),
                    "source_row_hash": canonical_hash(source),
                    "source_artifact_hash": digest,
                    "board_receipt_hash": board["source_hash"],
                    "duration_s": source.get("duration_s"),
                    "kernel_share": share,
                    "host_fraction": source.get("host_fraction"),
                    "transfer_bytes": source.get("transfer_bytes"),
                    "topology": source.get("topology"),
                    "fabric_compatible": compatible,
                    "offload_fraction": fraction,
                    "placement_constraint": board["scope"],
                    "disposition": "historical_quadratic_ising_candidate"
                    if compatible
                    else "linux_cpu_only"
                    if name == "PolarFire"
                    else "blocked_physical_jtag_0xffffffff"
                    if name == "GateMate"
                    else "host_operation_no_qualified_fabric_mapping",
                    "comparison_class": "estimate",
                    "amdahl_estimate_upper_bound": ceiling,
                    "comparison_basis": "Infinite compatible-kernel acceleration assumes zero added transport cost. Sparse topology does not remove measured bytes or host orchestration.",
                    "vendor_comparison": None,
                    "hardware_execution_measured": False,
                    "speedup": None,
                }
            )
    return mapping


def _authenticate(root: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Frozen upstream hashes prevent a rewritten primary and sidecar agreeing falsely."""
    prior, digest = _load(root, PRIOR)
    checks = [_check(PRIOR, digest, "sha256", PRIOR_HASH, digest)]
    if prior is not None:
        checks.extend(
            _check(PRIOR, digest, key, expected, prior.get(key))
            for key, expected in (
                ("experiment_id", 7926),
                ("task_id", "exp7926-hardware-evidence"),
                ("milestone", "2026.09.687"),
                ("run_date", "20260930"),
                ("verdict_class", "null"),
                ("flagged_adversarial", False),
                ("hardware_evidence_ready_score", 1),
            )
        )
        label = str(prior.get("terminal_validation_sidecar_path", ""))
        terminal, terminal_hash = _load(root, label)
        checks.append(_check(label, terminal_hash, "sha256", TERMINAL_HASH, terminal_hash))
        checks.append(
            _check(
                label,
                terminal_hash,
                "candidate_sha256",
                digest,
                (terminal or {}).get("candidate_sha256"),
            )
        )
        result["source_artifact_hashes"][label] = {
            "path": label,
            "sha256": terminal_hash,
            "role": "historical_terminal_validation",
            "exposure_status": "historical_read_only",
        }
        for receipt in (terminal or {}).get("reports", []):
            path = Path(receipt["log_path"])
            actual = sha256_file(path) if path.is_file() else None
            checks.append(_check(str(path), actual, "sha256", receipt["log_sha256"], actual))
        for name, expected in prior.get("terminal_receipt_hashes", {}).items():
            path = root / name
            actual = sha256_file(path) if path.is_file() else None
            checks.append(_check(name, actual, "sha256", expected, actual))
        for receipt in prior.get("validation_receipts", {}).get("checks", []):
            path = Path(receipt["log_path"])
            actual = sha256_file(path) if path.is_file() else None
            checks.append(_check(str(path), actual, "sha256", receipt["log_sha256"], actual))
    result["source_artifact_hashes"][PRIOR] = {
        "path": PRIOR,
        "sha256": digest,
        "role": "qualified_historical_board_custody",
        "exposure_status": "historical_read_only",
    }
    return checks


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7938-CUSTODY: optional service cannot close board custody."""
    started = time.monotonic()
    result = old.read_evidence(root, run_date)
    checks = _authenticate(root, result)
    result["preconditions_checked"].extend(checks)
    result["gate_check_summary"].extend(r for r in checks if not r["passed"])
    result["source_artifact_hashes"].pop(old.SERVICE, None)
    source, digest = _load(root, SERVICE)
    operands = [_check(SERVICE, digest, "exists", True, digest is not None)]
    if digest is not None:
        operands.append(_check(SERVICE, digest, "schema_json", "dict", type(source).__name__))
    rows = (source or {}).get("rows")
    if source is not None:
        operands.extend(
            _check(SERVICE, digest, key, expected, source.get(key))
            for key, expected in (
                ("experiment_id", 7937),
                ("task_id", "exp7937-service-cost"),
                ("milestone", "2026.09.688"),
                ("run_date", run_date),
                ("flagged_adversarial", False),
                ("service_measurement_ready_score", 1),
            )
        )
        operands.extend(
            [
                _check(
                    SERVICE,
                    digest,
                    "verdict_class.eligible",
                    True,
                    source.get("verdict_class") in {"positive", "circular_positive", "null"},
                ),
                _check(
                    SERVICE,
                    digest,
                    "validation_receipts.required_checks_passed",
                    True,
                    source.get("validation_receipts", {}).get("required_checks_passed"),
                ),
                _check(
                    SERVICE,
                    digest,
                    "rows.whole_service",
                    True,
                    isinstance(rows, list) and bool(rows) and all(_row_valid(r) for r in rows),
                ),
            ]
        )
    attached = all(r["passed"] for r in operands)
    ready = not result["gate_check_summary"]
    mapping = operation_map(
        result["board_rows"],
        rows if attached else [{"operation": name} for name in OPERATIONS],
        digest if attached else None,
    )
    result["source_artifact_hashes"][SERVICE] = {
        "path": SERVICE,
        "sha256": digest,
        "role": "optional_current_whole_service",
        "exposure_status": "exposed_development" if attached else "unavailable",
        "eligible": attached,
    }
    result.update(
        {
            "experiment_id": 7938,
            "task_id": "exp7938-hardware-evidence",
            "milestone": "2026.09.688",
            "honest_verdict": "complete_null_historical_board_scope"
            if ready
            else "complete_blocked_board_source_custody",
            "verdict_class": "null" if ready else "blocked",
            "hardware_evidence_ready_score": int(ready),
            "workload_attachment_available": attached,
            "workload_attachment_operands": operands,
            "workload_feasibility_rows": mapping if attached else [],
            "operation_map": mapping,
            "measured_offload_ceiling_available": attached
            and any(r["amdahl_estimate_upper_bound"] is not None for r in mapping),
            "measured_offload_ceiling": {
                "status": "estimate_from_measured_host_shares" if attached else "unmeasured",
                "comparison_class": "estimate",
                "hardware_speedup": None,
                "scope": "Per-operation infinite-kernel upper bounds; transfer cost remains unknown.",
            },
            "optional_workload_upstream_failures": deepcopy(
                (source or {}).get("gates_evaluated", [])
            ),
            "primary_resolution_receipt": {"path": None, "binding": "exact final bytes in sidecar"},
            "retire_if_same_verdict": {
                "prior_experiment_id": 7926,
                "identical_failure": not attached,
                "reason": "Retire unchanged board disposition; new operation map does not reopen device work.",
            },
        }
    )
    result["claim_scope"].update(
        {
            "workload": "exposed_development" if attached else "unmeasured",
            "mapping": "host estimates; no board workload execution",
        }
    )
    result["acceptance_gate_results"].update({"validity": ready, "calibration": None})
    result["sample_size_budget"]["eligible"] = (
        sum(r["eligible"] for r in result["rows"]) if ready else 0
    )
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
            name: f"Bind {name.replace('_', ' ')} to current bytes within historical scope."
            for name in result
            if name not in result["field_principles"]
        }
    )
    result["field_principles"]["operation_map"] = (
        "Explicit representation and k<=5 govern fabric placement; neural heads do not qualify."
    )
    result["field_principles"]["measured_offload_ceiling"] = (
        "Amdahl estimates use host shares, assume free transport, and never prove speedup."
    )
    result["field_principles"]["hardware_evidence_ready_score"] = (
        "Authenticated historical custody plus all owned checks; optional service is independent."
    )
    result["field_principles"]["acceptance_gate_results"]["calibration"] = (
        "No probability calibration was measured."
    )
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7938-TERMINAL: recompute task claims from upstream bytes."""
    fresh = read_evidence(root, candidate["run_date"])
    for field in (
        "experiment_id",
        "task_id",
        "milestone",
        "rows",
        "board_rows",
        "rows_checksum",
        "terminal_receipt_hashes",
        "source_artifact_hashes",
        "sample_size_budget",
        "workload_feasibility_rows",
        "workload_attachment_operands",
        "workload_attachment_available",
        "operation_map",
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
    ):
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
    """Reject empty and unrelated files before combining coverage measurements."""
    from carnot.reporting.validation_7938 import MEASURED

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
    """Reuse qualified supervision, then bind final publication to actual readers."""
    from carnot.reporting import validation_7938 as plan
    from carnot.reporting.experiment_7913_v686_hardware_evidence import progress

    started = time.monotonic()
    output = output.absolute()
    publication_raw = output.parent / "raw" / output.stem
    publication_raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carnot-7938-publish-", dir="/tmp") as folder:
        private = Path(folder)
        terminal = plan.terminal_manifest(private)
        for spec in terminal:
            spec["argv"] = [
                arg.replace(
                    str(private / "terminal-candidate.json"),
                    str(publication_raw / "terminal_candidate.json"),
                )
                for arg in spec["argv"]
            ]
        atomic_json(
            publication_raw / "publication_command_manifest.json",
            {
                "task_id": "exp7938-hardware-evidence",
                "commands": terminal,
                "dependency_hashes": {
                    name: sha256_file(root / name)
                    for name in (*plan.MEASURED, *plan.TESTS, *plan.LIBRARIES)
                },
                "coverage_includes": plan.INCLUDE,
            },
        )
        runner.qualify(
            root,
            run_date,
            private / "qualified.json",
            raw_root,
            evidence=__import__(__name__, fromlist=["read_evidence"]),
            validation=plan,
        )
        value = json.loads((private / "qualified.json").read_text())
        receipt_path = publication_raw / "primary_resolution_receipt.json"
        terminal_path = publication_raw / "terminal_validation_reports.json"
        value["primary_resolution_receipt"] = {
            "path": str(receipt_path),
            "binding": "exact final primary bytes",
        }
        value["terminal_validation_sidecar_path"] = str(terminal_path)
        value["duration_s"] = time.monotonic() - started
        value["phase_spans"] = [
            {"phase": "custody_and_owned_validation", "start_s": 0.0, "end_s": value["duration_s"]}
        ]
        value["duration_scope"] = (
            "Monotonic work until final candidate freeze; final validator durations are in its bound sidecar."
        )
        value["field_principles"]["duration_scope"] = (
            "Keep final validation time separate from the already frozen candidate."
        )

        def validator(candidate: Path) -> dict[str, Any]:
            progress(started, "final_validation", "before_cold_reduction", 0)
            reduced = cold_reduce(root, json.loads(candidate.read_text()))
            reports = [
                runner.run_child(spec, private, publication_raw / "terminal_logs", started, index)
                for index, spec in enumerate(terminal)
            ]
            flagged = json.loads(Path(reports[0]["log_path"]).read_text())["flagged_count"]
            passed = not flagged and all(r["passed"] for r in reports)
            report = {
                "passed": passed,
                "flagged_adversarial": bool(flagged),
                "candidate_sha256": sha256_file(candidate),
                "reports": reports,
                "cold_reduction": reduced,
            }
            atomic_json(terminal_path, report)
            progress(started, "final_validation", "after_validators", len(reports))
            return report

        published = publish_primary(output, value, validator)
        challenge = publication_raw / "reader_mtime_challenge.json"
        atomic_json(
            challenge,
            {
                "primary_sha256": published["primary_sha256"],
                "scope": "nested sidecar selection check",
            },
        )
        stamp = max(time.time_ns(), output.stat().st_mtime_ns + 1)
        os.utime(challenge, ns=(stamp, stamp))
        receipt = reader_receipt(
            value["task_id"],
            output.parent,
            field="hardware_evidence_ready_score",
            expected=value["hardware_evidence_ready_score"],
        )
        if (
            not receipt["passed"]
            or receipt["gate_path"] != str(output)
            or receipt["gate_sha256"] != published["primary_sha256"]
        ):
            raise ValueError("reader_identity")
        receipt["readiness_open"] = value["hardware_evidence_ready_score"] == 1
        receipt["newer_sidecar_path"] = str(challenge)
        receipt["primary_resolution"] = published
        atomic_json(receipt_path, receipt)
    progress(started, "final", "published_checked_bytes_and_reader_hashes", 3)
    return 0
