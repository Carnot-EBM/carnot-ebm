"""Read authenticated board history without operating the devices.

The old receipts describe what ran on their measurement dates. This reader
checks their bytes again so a current audit cannot silently widen that scope.
Spec refs: REQ-REPORT-7862 and SCENARIO-REPORT-7862-CUSTODY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.experiment_7847_v681_hardware_evidence import (
    INVENTORY,
    PRIOR,
    _board_sources,
)

EXP7847 = "results/experiment_7847_v681_hardware_evidence.json"
SERVICE = "results/experiment_7861_v682_service_cost.json"


def _check(
    path: str, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep a missing input distinct from an input with a wrong value."""
    return {
        "upstream_id": path,
        "artifact_path": path,
        "artifact_hash": digest,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _source(root: Path, name: str, role: str, date: str | None = None) -> dict[str, Any]:
    """Hash original bytes and retain their measurement or access date."""
    path = root / name
    return {
        "path": name,
        "sha256": sha256_file(path) if path.is_file() else None,
        "date": date,
        "role": role,
        "exposure_status": "historical_read_only",
    }


def _required_json(
    root: Path, name: str, digest: str | None, checks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Turn unreadable history into an explicit failed custody operand."""
    if digest is None:
        return None
    try:
        value = json.loads((root / name).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        value = None
    if not isinstance(value, dict):
        checks.append(_check(name, digest, "schema_json", "object", type(value).__name__))
        return None
    return value


def _service_fit(
    root: Path, boards: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare only a qualified current component with each board's limits."""
    path = root / SERVICE
    digest = sha256_file(path) if path.is_file() else None
    value = json.loads(path.read_text()) if digest else None
    valid = bool(
        value
        and value.get("experiment_id") == 7861
        and value.get("verdict_class") in {"positive", "null", "circular_positive"}
        and value.get("flagged_adversarial") is False
        and value.get("service_evidence_ready_score") == 1
    )
    component = value.get("component_requirements") if valid and value else None
    fit = []
    for board in boards:
        name = board["board"]
        qualified = (
            name == "KV260"
            and isinstance(component, dict)
            and component.get("model_class") == "quadratic_ising"
            and isinstance(component.get("k"), int)
            and component["k"] <= 5
        )
        fit.append(
            {
                "board": name,
                "component_requirements": component,
                "applicability": "unknown"
                if value is None
                else "candidate_only"
                if qualified
                else "unqualified",
                "fabric_fit": qualified,
                "hardware_execution_measured": False,
                "host_orchestration": "unmeasured",
                "transfer": "unmeasured",
                "precision": "unmeasured",
                "topology": "k<=5 quadratic Ising only"
                if name == "KV260"
                else "no qualified fabric",
            }
        )
    return fit, {
        "path": SERVICE,
        "sha256": digest,
        "status": "missing" if value is None else "qualified" if valid else "disqualified",
        "qualified": valid,
    }


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7862-CUSTODY: reduce old bytes to a dated board table."""
    began = time.monotonic()
    sources = {
        name: _source(root, name, role, date)
        for name, role, date in (
            (INVENTORY, "board_inventory", "20260928"),
            (PRIOR, "failed_validation_history", "20260928"),
            (EXP7847, "failed_validation_history", "20260929"),
            (SERVICE, "optional_current_service", None),
            ("research-references.md", "external_reference_index", "20260929"),
            ("research-hardware-wishlist.md", "hardware_note", "20260929"),
        )
    }
    missing = [
        _check(name, sources[name]["sha256"], "exists", True, None)
        for name in (INVENTORY, PRIOR, EXP7847)
        if sources[name]["sha256"] is None
    ]
    inventory = _required_json(root, INVENTORY, sources[INVENTORY]["sha256"], missing)
    prior = _required_json(root, PRIOR, sources[PRIOR]["sha256"], missing)
    old = _required_json(root, EXP7847, sources[EXP7847]["sha256"], missing)
    if inventory is not None:
        rows = inventory.get("board_rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 3
            or any(
                not isinstance(row, dict)
                or not all(
                    isinstance(row.get(field), str)
                    for field in (
                        "board",
                        "source_path",
                        "expected_hash",
                        "last_authenticated_evidence_date",
                    )
                )
                for row in rows
            )
        ):
            missing.append(
                _check(
                    INVENTORY,
                    sources[INVENTORY]["sha256"],
                    "board_rows.schema",
                    "three board receipts",
                    rows,
                )
            )
            inventory = None
    board_sources, checks = (
        _board_sources(root, inventory, prior) if inventory and prior else ({}, [])
    )
    sources.update(board_sources)
    if old:
        old_check = next(
            (
                x
                for x in old.get("validation_receipts", {}).get("checks", [])
                if x.get("name") == "affected_pytest"
            ),
            None,
        )
        checks.append(
            _check(
                EXP7847,
                sources[EXP7847]["sha256"],
                "affected_pytest.passed",
                False,
                old_check.get("passed") if old_check else None,
            )
        )
        required_failures = [
            receipt
            for receipt in old.get("validation_receipts", {}).get("checks", [])
            if receipt.get("classification") == "required" and receipt.get("passed") is False
        ]
        for receipt in required_failures:
            log_path = Path(receipt["log_path"])
            observed = sha256_file(log_path) if log_path.is_file() else None
            checks.append(
                _check(
                    EXP7847,
                    sources[EXP7847]["sha256"],
                    f"{receipt['name']}.log_sha256",
                    receipt["log_sha256"],
                    observed,
                )
            )
    failures = missing + [check for check in checks if not check["passed"]]
    boards = deepcopy(inventory.get("board_rows", [])) if inventory else []
    for board in boards:
        board["current_hardware_execution"] = False
        board["status"] = "historical_read_only"
        board["source_family"] = board["board"]
    fit, service = _service_fit(root, boards)
    sources[SERVICE]["exposure_status"] = service["status"]
    rows = deepcopy(boards)
    budget = {
        "intended": 3,
        "eligible": 2 if not failures else 0,
        "started": 0,
        "completed": 0,
        "censored": 0,
        "excluded": 1,
        "independent": 0,
    }
    historical = deepcopy(old.get("historical_failures", {})) if old else {}
    historical.update(
        {
            "exp7847_source_path": EXP7847,
            "exp7847_source_hash": sources[EXP7847]["sha256"],
            "exp7847_affected_pytest_passed": old_check.get("passed")
            if old and old_check
            else None,
            "exp7847_repository_health": old.get("repository_health") if old else None,
            "exp7847_verdict": old.get("honest_verdict") if old else None,
            "exp7847_required_failures": required_failures if old else [],
        }
    )
    references = [
        {
            "source_family": "vendor",
            "title": "Extropic Z1T",
            "url": "https://extropic.ai/writing/z1t",
            "access_date": run_date,
            "published_date": "20260904",
            "access_status": "readable",
            "claim": "Sparse probabilistic and FPGA stages; energy figures are vendor estimates, not local measurements.",
        },
        {
            "source_family": "vendor",
            "title": "From One to One Billion",
            "url": "https://extropic.ai/writing/from-one-to-one-billion",
            "access_date": run_date,
            "published_date": "20260803",
            "access_status": "readable",
            "claim": "Vendor hardware roadmap and software update do not establish local TSU access.",
        },
        {
            "source_family": "paper",
            "title": "FPGA-ASIC co-design v2",
            "url": "https://arxiv.org/html/2602.15985v2",
            "access_date": run_date,
            "published_date": "20260904",
            "access_status": "readable",
            "claim": "Orchestration and memory movement constrain whole-system benefit; this is not a local benchmark.",
        },
    ]
    triggers = [
        {
            "board": "KV260",
            "trigger": "new qualified k<=5 quadratic Ising workload plus SSH fabric and full-service timing",
            "changed_evidence_present": False,
        },
        {
            "board": "PolarFire",
            "trigger": "device-side FPGA execution and timing transcript distinct from Linux CPU",
            "changed_evidence_present": False,
        },
        {
            "board": "GateMate",
            "trigger": "dated physical or operator change and valid IDCODE after 0xffffffff",
            "changed_evidence_present": False,
        },
        {
            "board": "NPU",
            "trigger": "authenticated device execution for the exact service component",
            "changed_evidence_present": False,
        },
        {
            "board": "TSU",
            "trigger": "authenticated local access, compatible graph and measured whole-service boundary",
            "changed_evidence_present": False,
        },
    ]
    result: dict[str, Any] = {
        "experiment_id": 7862,
        "task_id": "exp7862-hardware-evidence",
        "milestone": "2026.09.682",
        "run_date": run_date,
        "honest_verdict": "complete_blocked_board_source_custody"
        if failures
        else "complete_null_historical_board_scope",
        "verdict_class": "blocked" if failures else "null",
        "flagged_adversarial": None,
        "gate_check_summary": failures,
        "rows": rows,
        "rows_checksum": canonical_hash(rows),
        "board_rows": boards,
        "service_fit_rows": fit,
        "vendor_reference_rows": references,
        "change_trigger_rows": triggers,
        "service_check": service,
        "source_artifact_hashes": sources,
        "preconditions_checked": missing + checks,
        "historical_failures": historical,
        "hardware_evidence_ready_score": int(not failures),
        "new_device_execution_count": 0,
        "sample_size_budget": budget,
        "acceptance_gate_results": {
            "validity": not failures,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - began,
        "phase_spans": {"evidence_read_s": time.monotonic() - began},
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
            "board_evidence": "dated read-only capability custody",
            "current_service": service["status"],
            "new_execution": "none",
            "hardware_advantage": "unmeasured",
            "NPU": "unqualified",
            "TSU": "unqualified",
            "host_orchestration": "unmeasured",
            "transfer": "unmeasured",
            "precision": "unmeasured",
            "topology": "KV260 quadratic k<=5 only",
        },
        "validation_receipts": {"checks": [], "required_checks_passed": False},
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "repository_health": None,
        "resolved_imports": {},
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
    result["field_principles"] = {
        key: f"Record {key.replace('_', ' ')} so a cold reader can audit evidence scope."
        for key in (*result, "field_principles")
    }
    result["field_principles"]["acceptance_gate_results"] = {
        "validity": "Exact source custody must pass.",
        "readiness": "Owned validation must pass before evidence can be reused.",
        "probability_quality": "No sampling law was measured.",
        "decision_benefit": "No decision benefit was measured.",
        "retention": "No retained learning was measured.",
        "efficiency": "No full service device efficiency was measured.",
    }
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7862-CLI: rebuild rows from original source bytes."""
    fresh = read_evidence(root, candidate["run_date"])
    if fresh["rows_checksum"] != candidate.get("rows_checksum"):
        raise ValueError("rows_changed")
    if fresh["gate_check_summary"] != candidate.get("gate_check_summary"):
        raise ValueError("gate_operands_changed")
    for receipt in candidate.get("validation_receipts", {}).get("checks", []):
        path = Path(receipt["log_path"])
        if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
            raise ValueError("sealed_log_changed")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}
