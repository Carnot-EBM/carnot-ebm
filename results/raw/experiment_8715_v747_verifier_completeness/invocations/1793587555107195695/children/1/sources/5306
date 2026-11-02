"""Reduce V671 capstone custody and registered gates (REQ-REPORT-7712)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v671_contract
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

REQUIRED_SCIENCE = (7704, 7706, 7707)
ELIGIBLE = {"positive", "circular_positive", "null"}


def authority(root: Path) -> dict[str, Any]:
    """Authenticate matching roadmap against its independent design."""
    path, roadmap, candidates = v671_contract.resolve_authority(root)
    design_path = root / v671_contract.DESIGN_PATH
    comparison = v671_contract.compare_authorities(design_path.read_text(), roadmap)
    if not comparison["passed"]:
        raise ValueError(f"V671 authority mismatch: {comparison['errors']}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "design_path": str(v671_contract.DESIGN_PATH),
        "design_sha256": sha256_file(design_path),
        "candidates": candidates,
        "comparison": comparison,
        "tasks": roadmap["tasks"],
    }


def failure(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: object,
    observed: object,
) -> dict[str, Any]:
    """Keep the exact operands of one unavailable upstream gate."""
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": False,
    }


def _pre_gate(root: Path, task: dict[str, Any]) -> Path | None:
    """Recognize a conductor gate receipt without calling it a producer."""
    number = task["id"].split("-", 1)[0][3:]
    for path in sorted((root / "results").glob(f"experiment_{number}_*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if value.get("schema") == "blocked_gate_check_v1":
            return path
    return None


def account(root: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Account for all tasks and block missing required science once."""
    if len(tasks) != 14 or any(
        not task["id"].startswith(f"exp{7699 + index}-") for index, task in enumerate(tasks)
    ):
        raise ValueError("V671 fourteen-task order required")
    payloads: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "producers": [],
        "pre_gate_receipts": [],
        "missing_evidence": [],
        "conductor_log": [],
        "planned_output_is_input": False,
    }
    failed: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_id, label = task["id"], task["deliverable"]
        path = root / label
        value: dict[str, Any] = {}
        evidence: str | None = None
        if index == 13:
            kind = "planned_output"
        elif path.is_file():
            value = json.loads(path.read_text())
            payloads[task_id] = value
            kind, evidence = "producer", label
            hashes["producers"].append(
                {
                    "task_id": task_id,
                    "path": label,
                    "sha256": sha256_file(path),
                }
            )
        else:
            receipt = _pre_gate(root, task)
            if receipt is None:
                kind = "absent"
                hashes["missing_evidence"].append({"task_id": task_id, "path": label})
            else:
                kind = "pre_gate_receipt"
                evidence = receipt.relative_to(root).as_posix()
                hashes["pre_gate_receipts"].append(
                    {
                        "task_id": task_id,
                        "path": evidence,
                        "sha256": sha256_file(receipt),
                    }
                )
        rows.append(
            {
                "order": index + 1,
                "task_id": task_id,
                "planned_path": label,
                "availability": kind,
                "evidence_path": evidence,
                "verdict_class": value.get("verdict_class"),
                "honest_verdict": value.get("honest_verdict"),
                "scientific_disposition": value.get("verdict_class"),
                "flagged_adversarial": value.get("flagged_adversarial"),
                "registered_gates": [],
            }
        )
    by_id = {task["id"]: task for task in tasks}
    for row, task in zip(rows, tasks, strict=True):
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            source = payloads.get(upstream, {})
            observed = source.get(gate["artifact_field"])
            expected, operator = gate["value"], gate["op"]
            passed = observed == expected if operator == "==" else observed in expected
            check = {
                "check": "registered_gate",
                "upstream": upstream,
                "path": by_id[upstream]["deliverable"],
                "field": gate["artifact_field"],
                "operator": operator,
                "expected": expected,
                "observed": observed,
                "passed": passed,
                "consumer": task["id"],
            }
            row["registered_gates"].append(check)
            if not passed:
                failed.append(check)
    for number in REQUIRED_SCIENCE:
        row = rows[number - 7699]
        source = payloads.get(row["task_id"])
        if source is None:
            failed.append(
                failure(
                    "required_scientific_producer",
                    row["task_id"],
                    row["planned_path"],
                    "exists",
                    "==",
                    True,
                    False,
                )
            )
        elif (
            source.get("verdict_class") not in ELIGIBLE
            or source.get("flagged_adversarial") is not False
        ):
            field = (
                "flagged_adversarial"
                if source.get("flagged_adversarial") is not False
                else "verdict_class"
            )
            failed.append(
                failure(
                    "required_scientific_producer",
                    row["task_id"],
                    row["planned_path"],
                    field,
                    "==" if field == "flagged_adversarial" else "in",
                    False if field == "flagged_adversarial" else sorted(ELIGIBLE),
                    source.get(field),
                )
            )
    log_path = root / v671_contract.LOG_PATH
    if log_path.is_file():
        hashes["conductor_log"].append(
            {
                "path": str(v671_contract.LOG_PATH),
                "sha256": sha256_file(log_path),
            }
        )
        lines = log_path.read_text().splitlines()
        for row, task in zip(rows[:-1], tasks[:-1], strict=True):
            if row["availability"] != "absent":
                continue
            matching = [
                line for line in lines if task["title"][:44] in line and "GATE_BLOCK" in line
            ]
            if matching:
                row["availability"] = "gate_skipped"
                row["evidence_path"] = str(v671_contract.LOG_PATH)
                row["log_custody"] = matching[-1]
    verdict_class = "blocked" if failed else "null"
    honest_verdict = (
        "complete_blocked_required_v671_scientific_evidence"
        if failed
        else "complete_null_v671_no_registered_benefit"
    )
    rows[-1]["verdict_class"] = verdict_class
    rows[-1]["honest_verdict"] = honest_verdict
    rows[-1]["scientific_disposition"] = verdict_class
    return {
        "prior_dispositions": rows,
        "source_artifact_hashes": hashes,
        "gate_check_summary": {
            "passed": not failed,
            "failed_count": len(failed),
            "first_failure": failed[0] if failed else None,
            "failed_checks": failed,
        },
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
    }


def cold_reduce(value: object, root: Path, tasks: list[dict[str, Any]]) -> list[str]:
    """Reopen source bytes and reject changed custody or gate interpretation."""
    if not isinstance(value, dict):
        return ["artifact_object_required"]
    try:
        expected = account(root, tasks)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"source_reduction_failed:{type(exc).__name__}"]
    errors = [
        f"{key}_mismatch"
        for key, item in expected.items()
        if key != "source_artifact_hashes"
        if canonical_hash(value.get(key)) != canonical_hash(item)
    ]
    actual_hashes = value.get("source_artifact_hashes", {})
    for key, item in expected["source_artifact_hashes"].items():
        if canonical_hash(actual_hashes.get(key)) != canonical_hash(item):
            errors.append(f"source_artifact_hashes.{key}_mismatch")
    return errors
