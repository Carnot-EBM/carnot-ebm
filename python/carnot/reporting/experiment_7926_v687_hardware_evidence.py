"""Preserve historical board limits without issuing a device command.

Spec ref: REQ-REPORT-7926-V687. Optional service estimates describe placement,
not measured hardware acceleration or independently held-out utility.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from coverage import CoverageData

from carnot.reporting import experiment_7913_v686_hardware_evidence as old
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check
from carnot.reporting.experiment_7876_v683_hardware_evidence import _load

ROOT = old.ROOT
PRIOR = "results/experiment_7913_v686_hardware_evidence.json"
SERVICE = "results/experiment_7925_v687_service_cost.json"


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """Authenticate original receipts before deriving any new custody claim."""
    started = time.monotonic()
    if run_date != "20260930":
        raise ValueError("run_date_mismatch")
    result = old.read_evidence(root, run_date)
    prior, digest = _load(root, PRIOR)
    checks = [_check(PRIOR, digest, "exists", True, digest is not None)]
    if digest is not None:
        checks.append(_check(PRIOR, digest, "schema_json", "dict", type(prior).__name__))
    if prior is not None:
        checks.extend(
            _check(PRIOR, digest, key, expected, prior.get(key))
            for key, expected in (
                ("experiment_id", 7913),
                ("task_id", "exp7913-hardware-evidence"),
                ("milestone", "2026.09.686"),
                ("run_date", "20260930"),
                ("verdict_class", "disqualified"),
            )
        )
        for receipt in prior.get("validation_receipts", {}).get("checks", []):
            log = Path(receipt["log_path"])
            observed = sha256_file(log) if log.is_file() else None
            checks.append(_check(str(log), observed, "log_sha256", receipt["log_sha256"], observed))
        result["historical_required_failures"].extend(
            deepcopy(r)
            for r in prior["validation_receipts"]["checks"]
            if r["classification"] == "required" and not r["passed"]
        )
    result["preconditions_checked"].extend(checks)
    result["gate_check_summary"].extend(r for r in checks if not r["passed"])
    result["source_artifact_hashes"][PRIOR] = {
        "path": PRIOR,
        "sha256": digest,
        "date": (prior or {}).get("run_date"),
        "role": "disqualified_historical_runner",
        "exposure_status": "historical_read_only",
    }
    result["source_artifact_hashes"].pop(old.SERVICE, None)
    service, service_hash = _load(root, SERVICE)
    operands = [_check(SERVICE, service_hash, "exists", True, service_hash is not None)]
    if service_hash is not None:
        operands.append(
            _check(SERVICE, service_hash, "schema_json", "dict", type(service).__name__)
        )
    rows = (service or {}).get("rows")
    if service is not None:
        operands.extend(
            _check(SERVICE, service_hash, key, expected, service.get(key))
            for key, expected in (
                ("experiment_id", 7925),
                ("task_id", "exp7925-service-cost"),
                ("run_date", run_date),
                ("flagged_adversarial", False),
                ("service_measurement_ready_score", 1),
            )
        )
        operands.extend(
            [
                _check(
                    SERVICE,
                    service_hash,
                    "verdict_class.eligible",
                    True,
                    service.get("verdict_class") in {"positive", "circular_positive", "null"},
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "validation_receipts.required_checks_passed",
                    True,
                    service.get("validation_receipts", {}).get("required_checks_passed"),
                ),
                _check(
                    SERVICE,
                    service_hash,
                    "rows.whole_service",
                    True,
                    isinstance(rows, list)
                    and bool(rows)
                    and all(
                        isinstance(r, dict)
                        and r.get("operation")
                        and isinstance(r.get("duration_s"), (int, float))
                        and r["duration_s"] >= 0
                        for r in rows
                    ),
                ),
            ]
        )
    attached = all(r["passed"] for r in operands)
    result["source_artifact_hashes"][SERVICE] = {
        "path": SERVICE,
        "sha256": service_hash,
        "date": (service or {}).get("run_date"),
        "role": "optional_current_whole_service",
        "exposure_status": "exposed_development" if attached else "unavailable",
        "eligible": attached,
    }
    fit = []
    if attached:
        for source in rows:
            share = source.get("kernel_share")
            bound = 1 / (1 - share) if isinstance(share, (float, int)) and 0 <= share < 1 else None
            for board in result["board_rows"]:
                fit.append(
                    {
                        "board": board["board"],
                        "source_row": deepcopy(source),
                        "source_row_hash": canonical_hash(source),
                        "source_artifact_hash": service_hash,
                        "transfer_bytes": source.get("transfer_bytes"),
                        "kernel_share": share,
                        "host_fraction": source.get("host_fraction"),
                        "topology": source.get("topology"),
                        "placement_constraint": board["scope"],
                        "comparison_class": "estimate",
                        "amdahl_estimate_upper_bound": bound,
                        "vendor_comparison": None,
                        "comparison_basis": "Sparse topology does not remove observed byte movement or host orchestration; kernel-only acceleration is limited by whole-service share.",
                        "hardware_execution_measured": False,
                        "speedup": None,
                    }
                )
    ready = not result["gate_check_summary"]
    result.update(
        {
            "experiment_id": 7926,
            "task_id": "exp7926-hardware-evidence",
            "milestone": "2026.09.687",
            "honest_verdict": "complete_null_historical_board_scope"
            if ready
            else "complete_blocked_board_source_custody",
            "verdict_class": "null" if ready else "blocked",
            "hardware_evidence_ready_score": int(ready),
            "workload_attachment_available": attached,
            "workload_attachment_operands": operands,
            "workload_feasibility_rows": fit,
            "coverage_statement_counts": {},
            "scheduled_device_operations": [],
            "historical_fixture_date": "20260929",
            "current_execution_date": run_date,
            "later_kv260_access_precondition": "ssh kria",
            "repository_health": (prior or {}).get("repository_health"),
            "retire_if_same_verdict": {
                "prior_experiment_id": 7913,
                "identical_failure": False,
                "reason": "Both historical fixture calls now use 20260929; the prior date rejection remains historical.",
            },
        }
    )
    result["claim_scope"]["workload"] = "exposed_development" if attached else "unmeasured"
    result["acceptance_gate_results"].update({"validity": ready, "scientific_benefit": None})
    result["sample_size_budget"]["eligible"] = (
        sum(r["eligible"] for r in result["rows"]) if ready else 0
    )
    result["reproducibility_checksum"] = canonical_hash(
        {
            "sources": result["source_artifact_hashes"],
            "rows": result["rows"],
            "workload": fit,
            "date": run_date,
            "code": sha256_file(Path(__file__)),
        }
    )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"] = {"evidence_read_s": result["duration_s"]}
    result["field_principles"].update(
        {
            key: f"Bind {key.replace('_', ' ')} to exact evidence without expanding board scope."
            for key in result
            if key not in result["field_principles"]
        }
    )
    result["field_principles"]["workload_feasibility_rows"] = (
        "Amdahl and placement comparisons are estimates; no hardware speedup was measured."
    )
    result["field_principles"]["acceptance_gate_results"]["scientific_benefit"] = (
        "Receipt custody does not measure independent scientific benefit."
    )
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """Recompute primitive claims so altered hardware scope cannot pass replay."""
    fresh = read_evidence(root, candidate["run_date"])
    for field in (
        "rows",
        "board_rows",
        "rows_checksum",
        "terminal_receipt_hashes",
        "source_artifact_hashes",
        "sample_size_budget",
        "workload_feasibility_rows",
        "workload_attachment_operands",
        "workload_attachment_available",
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
    """Empty or foreign data cannot satisfy the owned statement coverage gate."""
    from carnot.reporting.validation_7926 import MEASURED

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
