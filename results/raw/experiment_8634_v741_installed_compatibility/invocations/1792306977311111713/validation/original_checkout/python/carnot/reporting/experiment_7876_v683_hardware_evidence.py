"""Audit board bytes and attach only qualified CPU workload shapes.

The three board measurements are historical. This module reads receipts only;
it neither opens a device nor imports a hardware transport.
Spec refs: REQ-REPORT-7876 and SCENARIO-REPORT-7876-CUSTODY/WORKLOAD.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import (
    _check,
    read_evidence as read_prior_board_sources,
)

PRIOR = "results/experiment_7862_v682_hardware_evidence.json"
WORKLOAD = "results/experiment_7875_v683_service_cost.json"
BOARDS = ("KV260", "PolarFire", "GateMate")


def _load(root: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Read one exact receipt and keep malformed bytes distinguishable."""
    path = root / name
    if not path.is_file():
        return None, None
    digest = sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        value = None
    return value if isinstance(value, dict) else None, digest


def _prior_checks(
    root: Path, prior: dict[str, Any] | None, digest: str | None, run_date: str
) -> list[dict[str, Any]]:
    """Compare source bytes, exact board claims, cutoff dates, and child logs."""
    checks = [_check(PRIOR, digest, "exists", True, digest is not None)]
    if digest is None:
        return checks
    checks.append(_check(PRIOR, digest, "schema_json", "dict", type(prior).__name__))
    if prior is None:
        return checks
    for field, expected in (
        ("experiment_id", 7862),
        ("task_id", "exp7862-hardware-evidence"),
        ("milestone", "2026.09.682"),
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
        ("hardware_evidence_ready_score", 1),
        ("new_device_execution_count", 0),
        ("current_device_execution_count", 0),
    ):
        observed = prior.get(field, 0 if field == "current_device_execution_count" else None)
        checks.append(_check(PRIOR, digest, field, expected, observed))
    cutoff = prior.get("run_date")
    checks.append(
        _check(
            PRIOR, digest, "run_date.cutoff", True, isinstance(cutoff, str) and cutoff <= run_date
        )
    )
    old_sources = prior.get("source_artifact_hashes")
    checks.append(
        _check(PRIOR, digest, "source_artifact_hashes.schema", "dict", type(old_sources).__name__)
    )
    if isinstance(old_sources, dict):
        for name, record in old_sources.items():
            if not name.startswith("results/") or name.endswith(
                "experiment_7861_v682_service_cost.json"
            ):
                continue
            expected = record.get("sha256") if isinstance(record, dict) else None
            path = root / name
            observed = sha256_file(path) if path.is_file() else None
            checks.append(_check(name, observed, "sha256", expected, observed))
    rows = prior.get("board_rows")
    names = (
        [row.get("board") if isinstance(row, dict) else None for row in rows]
        if isinstance(rows, list)
        else None
    )
    checks.append(_check(PRIOR, digest, "board_rows.names", list(BOARDS), names))
    if names == list(BOARDS):
        for row, board in zip(rows, BOARDS, strict=True):
            date = row.get("last_authenticated_evidence_date")
            try:
                valid_date = datetime.strptime(date, "%Y%m%d").strftime("%Y%m%d") <= run_date
            except (TypeError, ValueError):
                valid_date = False
            checks.append(
                _check(
                    PRIOR,
                    digest,
                    f"board_rows.{board}.last_authenticated_evidence_date",
                    True,
                    valid_date,
                )
            )
            for field, expected in (
                ("current_hardware_execution", False),
                ("status", "historical_read_only"),
            ):
                checks.append(
                    _check(PRIOR, digest, f"board_rows.{board}.{field}", expected, row.get(field))
                )
            source = row.get("source_path")
            observed_hash = (
                sha256_file(root / source)
                if isinstance(source, str) and (root / source).is_file()
                else None
            )
            checks.append(
                _check(
                    PRIOR,
                    digest,
                    f"board_rows.{board}.source_hash",
                    observed_hash,
                    row.get("source_hash"),
                )
            )
        for board, field, expected in (
            ("KV260", "k_max", 5),
            ("PolarFire", "processor_class", "linux_cpu"),
            ("GateMate", "blocker", "0xffffffff"),
        ):
            row = rows[BOARDS.index(board)]
            checks.append(
                _check(PRIOR, digest, f"board_rows.{board}.{field}", expected, row.get(field))
            )
    receipts = prior.get("validation_receipts", {}).get("checks", [])
    for receipt in receipts if isinstance(receipts, list) else []:
        if not isinstance(receipt, dict):
            continue
        log_path = Path(str(receipt.get("log_path", "")))
        actual = sha256_file(log_path) if log_path.is_file() else None
        checks.append(
            _check(
                PRIOR,
                digest,
                f"validation.{receipt.get('name')}.log_sha256",
                receipt.get("log_sha256"),
                actual,
            )
        )
    return checks


def _workload(root: Path, run_date: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep an absent or invalid science receipt outside the board gate."""
    value, digest = _load(root, WORKLOAD)
    shapes = value.get("workload_shapes") if value else None
    qualified = bool(
        value
        and value.get("experiment_id") == 7875
        and value.get("task_id") == "exp7875-service-cost"
        and value.get("run_date") == run_date
        and value.get("verdict_class") in {"positive", "null", "circular_positive"}
        and value.get("flagged_adversarial") is False
        and value.get("service_cost_ready_score") == 1
        and value.get("validation_receipts", {}).get("required_checks_passed") is True
        and isinstance(shapes, list)
        and bool(shapes)
        and all(isinstance(row, dict) and row.get("operation") for row in shapes)
    )
    status = "missing" if digest is None else "qualified" if qualified else "disqualified"
    source = {
        "path": WORKLOAD,
        "sha256": digest,
        "date": value.get("run_date") if value else None,
        "role": "optional_current_cpu_workload",
        "exposure_status": status,
        "eligible": qualified,
    }
    table = []
    if qualified:
        for substrate in ("CPU", "available_GPU", "KV260", "PolarFire", "GateMate"):
            table.append(
                {
                    "substrate": substrate,
                    "cpu_workload_shapes": deepcopy(shapes),
                    "classification": "current_measurement" if substrate == "CPU" else "estimate",
                    "hardware_execution_measured": False,
                    "speedup": None,
                    "feasibility": "measured_cpu" if substrate == "CPU" else "candidate_only",
                }
            )
    return source, table


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7876-CUSTODY: return one terminal evidence record."""
    started = time.monotonic()
    base = read_prior_board_sources(root, run_date)
    prior, digest = _load(root, PRIOR)
    checks = _prior_checks(root, prior, digest, run_date)
    failures = [*base["gate_check_summary"], *(check for check in checks if not check["passed"])]
    boards = deepcopy(base["board_rows"])
    for row in boards:
        row["family"] = row["board"]
        row["arm"] = "historical_accounting"
        row["seed"] = None
        row["claim_class"] = "historical"
        row["current_hardware_execution"] = False
        row["measured_current_latency_ms"] = None
    workload_source, feasibility = _workload(root, run_date)
    sources = deepcopy(base["source_artifact_hashes"])
    sources[PRIOR] = {
        "path": PRIOR,
        "sha256": digest,
        "date": prior.get("run_date") if prior else None,
        "role": "prior_board_custody",
        "exposure_status": "historical_read_only",
    }
    sources[WORKLOAD] = workload_source
    rows = boards
    ready = not failures
    result: dict[str, Any] = {
        "experiment_id": 7876,
        "task_id": "exp7876-hardware-evidence",
        "milestone": "2026.09.683",
        "run_date": run_date,
        "honest_verdict": "complete_null_historical_board_scope"
        if ready
        else "complete_blocked_board_source_custody",
        "verdict_class": "null" if ready else "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "rows_checksum": canonical_hash(rows),
        "hardware_rows": boards,
        "board_rows": boards,
        "terminal_receipt_hashes": {row["source_path"]: row["source_hash"] for row in boards},
        "changed_physical_receipts": [],
        "workload_attachment_available": bool(feasibility),
        "workload_feasibility_rows": feasibility,
        "source_artifact_hashes": sources,
        "preconditions_checked": [*base["preconditions_checked"], *checks],
        "historical_failures": deepcopy(base["historical_failures"]),
        "hardware_evidence_ready_score": int(ready),
        "current_device_execution_count": 0,
        "hardware_speedup_claimed": False,
        "sample_size_budget": {
            "intended": 3,
            "eligible": 2 if ready else 0,
            "started": 0,
            "completed": 0,
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
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {
            "model_loads": 0,
            "generation_calls": 0,
            "tokens": 0,
            "model_file_hashes": [],
        },
        "trained_head_specs": [],
        "verifier_is_oracle": False,
        "claim_scope": {
            "board_evidence": "historical custody",
            "current_measurement": "receipt analysis only",
            "workload": workload_source["exposure_status"],
            "hardware_advantage": "unmeasured",
            "NPU": "unqualified",
            "Extropic_TSU_Z1T": "unqualified",
            "p_computer": "unqualified",
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
            "rows": rows,
            "date": run_date,
            "seed": 0,
            "code": sha256_file(Path(__file__)),
        }
    )
    result["field_principles"] = {
        key: f"Record {key.replace('_', ' ')} to keep current work distinct from dated evidence."
        for key in result
    }
    result["field_principles"]["acceptance_gate_results"] = {
        "validity": "Exact source bytes and scope must pass.",
        "readiness": "All owned required checks must pass.",
        "probability_quality": "No sampling law was measured.",
        "decision_benefit": "No decision benefit was measured.",
        "retention": "No learned retention was measured.",
        "efficiency": "No device efficiency was measured.",
    }
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7876-CLI: rebuild primitive rows and sealed log identity."""
    fresh = read_evidence(root, candidate["run_date"])
    if fresh["rows"] != candidate.get("hardware_rows") or fresh["rows_checksum"] != candidate.get(
        "rows_checksum"
    ):
        raise ValueError("rows_changed")
    if fresh["gate_check_summary"] != candidate.get("gate_check_summary"):
        raise ValueError("gate_operands_changed")
    if fresh["terminal_receipt_hashes"] != candidate.get("terminal_receipt_hashes"):
        raise ValueError("terminal_receipt_changed")
    for receipt in candidate.get("validation_receipts", {}).get("checks", []):
        path = Path(receipt["log_path"])
        if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
            raise ValueError("sealed_log_changed")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}
