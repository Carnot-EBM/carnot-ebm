"""Read dated board evidence and optional current host-service measurements.

The board files are old measurements with narrow scopes. Rehashing them gives
continuity, but cannot turn their old speed numbers into a current service run.
Spec refs: REQ-REPORT-7847 and SCENARIO-REPORT-7847-*.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

INVENTORY = "results/experiment_7820_v679_hardware_evidence.json"
PRIOR = "results/experiment_7834_v680_hardware_evidence.json"
SERVICE = "results/experiment_7846_v681_service_cost.json"
RAW_KV = "results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"
RAW_PF = "results/raw/experiment_7231/polarfire_dispatch.json"


def _load(root: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Read only an exact repository-relative path and return its byte identity."""
    path = root / name
    if not path.is_file():
        return None, None
    return json.loads(path.read_text()), sha256_file(path)


def _gate(
    upstream: str, path: str, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep both sides of each external prerequisite for a cold reader."""
    return {
        "upstream_id": upstream,
        "artifact_path": path,
        "artifact_hash": digest,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _number(value: Any) -> bool:
    """Reject booleans and nonfinite timing values before deriving a bound."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _source(
    root: Path, name: str, expected: str | None, role: str, date: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read original bytes; a path and old digest are a single evidence unit."""
    path = root / name
    actual = sha256_file(path) if path.is_file() else None
    row = {
        "path": name,
        "sha256": actual,
        "expected_sha256": expected,
        "date": date,
        "role": role,
        "eligible": actual is not None and actual == expected,
    }
    return row, _gate("board_history", name, actual, "sha256", expected, actual)


def _service(root: Path) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
    """An exact science producer can open traffic analysis; a skip cannot."""
    data, digest = _load(root, SERVICE)
    checks = [
        _gate(
            "Exp7846",
            SERVICE,
            digest,
            "service_evidence_ready_score",
            1,
            data.get("service_evidence_ready_score") if data else None,
        )
    ]
    if data is not None:
        for field, expected in (
            ("experiment_id", 7846),
            ("task_id", "exp7846-service-cost"),
            ("milestone", "2026.09.681"),
            ("flagged_adversarial", False),
        ):
            checks.append(_gate("Exp7846", SERVICE, digest, field, expected, data.get(field)))
        checks.append(
            _gate(
                "Exp7846",
                SERVICE,
                digest,
                "verdict_class.accepted",
                True,
                data.get("verdict_class") in {"positive", "circular_positive", "null"},
            )
        )
        times = data.get("stage_times_ms")
        whole = times.get("whole_service") if isinstance(times, dict) else None
        host = times.get("host_stage") if isinstance(times, dict) else None
        valid = _number(whole) and _number(host) and whole > 0 and 0 <= host <= whole
        checks.append(_gate("Exp7846", SERVICE, digest, "stage_times_ms.valid", True, valid))
    failures = [check for check in checks if not check["passed"]]
    return (
        data,
        {
            "path": SERVICE,
            "sha256": digest,
            "qualified": not failures,
            "status": "missing"
            if data is None
            else "qualified"
            if not failures
            else "disqualified",
        },
        failures,
    )


def _board_sources(
    root: Path, inventory: dict[str, Any], prior: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify all three original artifacts and the old raw transcript bytes."""
    sources: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    old_expected = prior.get("source_artifact_hashes", {}).get(INVENTORY, {}).get("sha256")
    sources[INVENTORY], check = _source(
        root, INVENTORY, old_expected, "board_inventory", "20260928"
    )
    checks.append(check)
    rows = inventory.get("board_rows", [])
    checks.append(
        _gate(
            "Exp7820",
            INVENTORY,
            sources[INVENTORY]["sha256"],
            "board_names",
            ["KV260", "PolarFire", "GateMate"],
            [row.get("board") for row in rows],
        )
    )
    if len(rows) != 3:
        return sources, checks
    for row in rows:
        name = row["source_path"]
        sources[name], check = _source(
            root,
            name,
            row["expected_hash"],
            "dated_board_receipt",
            row["last_authenticated_evidence_date"],
        )
        checks.append(check)
    for row, field, expected in (
        (rows[0], "k_max", 5),
        (rows[1], "processor_class", "linux_cpu"),
        (rows[2], "blocker", "0xffffffff"),
    ):
        checks.append(
            _gate(
                "Exp7820", INVENTORY, sources[INVENTORY]["sha256"], field, expected, row.get(field)
            )
        )
    kv, _ = _load(root, rows[0]["source_path"])
    pf, _ = _load(root, rows[1]["source_path"])
    raw = inventory.get("source_artifact_hashes", {})
    raw_paths = [
        (RAW_KV, "sha256:" + kv["kv260_terminal_transcript_sha256"] if kv else None),
        (RAW_PF, pf["dispatch_receipt"]["raw_sha256"] if pf else None),
    ]
    for name in (
        "results/raw/experiment_7751_v674_hardware_continuity/rows.json",
        "results/raw/experiment_7779_v676_hardware_evidence/rows.json",
    ):
        raw_paths.append((name, raw.get(name, {}).get("sha256")))
    for name, expected in raw_paths:
        sources[name], check = _source(root, name, expected, "historical_raw_transcript", None)
        check["artifact_field"] = "raw_sha256"
        checks.append(check)
    checks.append(
        _gate(
            "Exp7231",
            rows[1]["source_path"],
            sources[rows[1]["source_path"]]["sha256"],
            "processor_class",
            "cpu",
            pf.get("dispatch_receipt", {}).get("processor_class") if pf else None,
        )
    )
    return sources, checks


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7847-BLOCKED: reduce history without a device operation."""
    started = time.monotonic()
    inventory, _ = _load(root, INVENTORY)
    prior, prior_hash = _load(root, PRIOR)
    if inventory is None or prior is None:
        raise ValueError("missing_board_inventory_or_prior_failure")
    sources, checks = _board_sources(root, inventory, prior)
    sources[PRIOR] = {
        "path": PRIOR,
        "sha256": prior_hash,
        "date": "20260928",
        "role": "failed_validation_history",
        "eligible": True,
    }
    service, service_check, service_failures = _service(root)
    sources[SERVICE] = {
        "path": SERVICE,
        "sha256": service_check["sha256"],
        "date": service.get("run_date") if service else None,
        "role": "current_science_producer",
        "eligible": service_check["qualified"],
    }
    board_failures = [check for check in checks if not check["passed"]]
    continuity = not board_failures
    qualified = service_check["qualified"] and continuity
    boards = deepcopy(inventory["board_rows"])
    day = datetime.strptime(run_date, "%Y%m%d")
    for row in boards:
        row["evidence_age_days"] = (
            day - datetime.strptime(row["last_authenticated_evidence_date"], "%Y%m%d")
        ).days
        row["current_hardware_execution"] = False
        row["current_run_date"] = run_date
        row["service_fraction"] = None
        row["family"] = row["board"]
        row["status"] = "historical_read_only"
        row["intended"] = 1
        row["eligible"] = int(row["board"] != "GateMate" and continuity)
        row["started_count"] = 0
        row["completed_count"] = 0
        row["censored_count"] = 0
        row["excluded_count"] = int(row["board"] == "GateMate")
        row["independent_count"] = 0
    times = service.get("stage_times_ms", {}) if qualified and service else {}
    whole = times.get("whole_service")
    host = times.get("host_stage")
    fraction = host / whole if qualified else None
    upper = 1 / (1 - fraction) if fraction is not None and fraction < 1 else None
    candidates = (
        ("CPU", "feature", "host_cpu"),
        ("Rust/PyO3", "predicate", "host_native_boundary"),
        ("GPU", "coefficient", "host_device_transfer"),
        ("KV260", "coefficient", "ssh_to_fabric"),
        ("CPU", "persistence", "host_durable_write"),
        ("future_TSU", "sampling", "host_to_tsu"),
    )
    opportunities = []
    for substrate, stage, boundary in candidates:
        opportunities.append(
            {
                "substrate": substrate,
                "stage": stage,
                "boundary": boundary,
                "measured_host_stage_ms": host if qualified else None,
                "whole_service_ms": whole if qualified else None,
                "transfer_bytes": service.get("transfer_bytes", {}).get(stage)
                if qualified and isinstance(service.get("transfer_bytes"), dict)
                else None,
                "setup_ms": service.get("setup_ms", {}).get(stage)
                if qualified and isinstance(service.get("setup_ms"), dict)
                else None,
                "update_bytes": service.get("update_bytes", {}).get(stage)
                if qualified and isinstance(service.get("update_bytes"), dict)
                else None,
                "whole_service_upper_bound": upper if substrate != "future_TSU" else None,
                "hardware_execution_measured": False,
                "applicability": "candidate_only" if qualified else "unmeasured_service",
            }
        )
    prior_coverage = next(
        (
            check
            for check in prior.get("validation_receipts", {}).get("checks", [])
            if check.get("name") == "coverage_report"
        ),
        None,
    )
    historical = {
        "source_path": PRIOR,
        "source_hash": prior_hash,
        "exp7834_verdict": prior.get("honest_verdict"),
        "exp7834_required_coverage_passed": prior_coverage.get("passed")
        if prior_coverage
        else None,
        "exp7834_required_coverage_receipt": prior_coverage,
        "exp7834_repository_health": prior.get("repository_health"),
    }
    failures = board_failures + service_failures
    verdict = (
        "complete_blocked_unqualified_board_evidence"
        if board_failures
        else (
            "complete_blocked_missing_service_evidence"
            if service is None
            else "complete_blocked_unqualified_service_evidence"
            if service_failures
            else "complete_null_no_hardware_execution"
        )
    )
    rows = deepcopy(boards)
    result: dict[str, Any] = {
        "experiment_id": 7847,
        "task_id": "exp7847-hardware-evidence",
        "milestone": "2026.09.681",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": "blocked" if failures else "null",
        "flagged_adversarial": None,
        "gate_check_summary": failures,
        "rows": rows,
        "rows_checksum": canonical_hash(rows),
        "board_rows": boards,
        "accelerator_opportunity_rows": opportunities,
        "service_check": service_check,
        "source_artifact_hashes": sources,
        "preconditions_checked": checks + (service_failures if service_failures else []),
        "hardware_inventory_ready_score": int(continuity),
        "acceptance_gate_results": {
            "validity": continuity,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "sample_size_budget": {
            "intended": 3,
            "eligible": 2 if continuity else 0,
            "started": 0,
            "completed": 0,
            "censored": 0,
            "excluded": 1,
            "independent": 0,
        },
        "historical_failures": historical,
        "duration_s": time.monotonic() - started,
        "phase_spans": {"evidence_read_s": time.monotonic() - started},
        "random_seed": 0,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "model_loads": 0,
            "generation_calls": 0,
            "tokens": 0,
            "model_file_hashes": [],
        },
        "verifier_is_oracle": False,
        "claim_scope": {
            "hardware_advantage": "unmeasured",
            "source_family_exposure": "development_only",
            "KV260": "fabric k_max<=5; commercial toolchain required for new build",
            "PolarFire": "Linux CPU dispatch only",
            "GateMate": "JTAG 0xffffffff blocked",
            "TSU": "vendor estimates only; no authenticated local access",
            "sampling": "deterministic head does not establish Ising sampling",
        },
        "field_principles": {
            "experiment_id": "An integer experiment number is distinct from the task slug.",
            "gate_check_summary": "Missing and wrong-valued operands stay distinct.",
            "board_rows": "Historical reachability is not current execution.",
            "accelerator_opportunity_rows": "Only qualified current service can yield bounds.",
            "hardware_inventory_ready_score": "Custody readiness is not performance readiness.",
        },
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "inputs": sources,
            "rows": rows,
            "date": run_date,
            "seed": 0,
            "code": sha256_file(Path(__file__)),
        }
    )
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7847-CLI: reconstruct rows from the original bytes."""
    fresh = read_evidence(root, candidate["run_date"])
    if fresh["rows_checksum"] != candidate.get("rows_checksum"):
        raise ValueError("rows_changed")
    if fresh["gate_check_summary"] != candidate.get("gate_check_summary"):
        raise ValueError("gate_operands_changed")
    for receipt in candidate.get("validation_receipts", {}).get("checks", []):
        path = Path(receipt["log_path"])
        if sha256_file(path) != receipt["log_sha256"]:
            raise ValueError("sealed_log_changed")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}
