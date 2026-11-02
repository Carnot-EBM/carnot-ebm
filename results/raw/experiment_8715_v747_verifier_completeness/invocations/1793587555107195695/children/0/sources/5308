"""Reduce V672 capstone custody without promoting missing producers (REQ-REPORT-7725)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


REQUIRED = {7718, 7720, 7721}
ELIGIBLE = {"positive", "circular_positive", "null"}


def failed_check(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: object,
    observed: object,
) -> dict[str, Any]:
    """Keep the exact operands that explain why evidence cannot advance."""
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


def account(root: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Count every planned task once while treating gate receipts as custody."""
    if len(tasks) != 13 or any(
        not task["id"].startswith(f"exp{7713 + index}-") for index, task in enumerate(tasks)
    ):
        raise ValueError("V672 thirteen-task sequence required")
    hashes: dict[str, Any] = {
        "producers": [],
        "flagged_historical_evidence": [],
        "pre_gate_receipts": [],
        "missing_custody": [],
        "planned_output_is_input": False,
    }
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_id, label = task["id"], task["deliverable"]
        path = root / label
        value: dict[str, Any] = {}
        kind = "planned_output" if index == 12 else "absent"
        evidence: str | None = None
        if index != 12 and path.is_file():
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"producer object required: {label}")
            evidence = label
            kind = (
                "pre_gate_receipt" if value.get("schema") == "blocked_gate_check_v1" else "producer"
            )
            bucket = (
                "pre_gate_receipts"
                if kind == "pre_gate_receipt"
                else (
                    "flagged_historical_evidence"
                    if value.get("flagged_adversarial") is True
                    else "producers"
                )
            )
            hashes[bucket].append({"task_id": task_id, "path": label, "sha256": sha256_file(path)})
            if kind == "producer":
                payloads[task_id] = value
        elif index != 12:
            hashes["missing_custody"].append({"task_id": task_id, "path": label})
        rows.append(
            {
                "order": index + 1,
                "task_id": task_id,
                "planned_path": label,
                "availability": kind,
                "evidence_path": evidence,
                "verdict_class": value.get("verdict_class"),
                "honest_verdict": value.get("honest_verdict"),
                "flagged_adversarial": value.get("flagged_adversarial"),
                "registered_gates": [],
            }
        )
    by_id = {task["id"]: task for task in tasks}
    for row, task in zip(rows, tasks, strict=True):
        for gate in task.get("gated_on") or []:
            upstream = gate["upstream"]
            observed = payloads.get(upstream, {}).get(gate["artifact_field"])
            passed = observed == gate["value"] if gate["op"] == "==" else observed in gate["value"]
            check = failed_check(
                "registered_gate",
                upstream,
                by_id[upstream]["deliverable"],
                gate["artifact_field"],
                gate["op"],
                gate["value"],
                observed,
            )
            check["passed"] = passed
            check["consumer"] = task["id"]
            row["registered_gates"].append(check)
            if not passed:
                failures.append(check)
    for number in sorted(REQUIRED):
        row = rows[number - 7713]
        source = payloads.get(row["task_id"])
        if source is None:
            failures.append(
                failed_check(
                    "required_scientific_producer",
                    row["task_id"],
                    row["planned_path"],
                    "exists",
                    "==",
                    True,
                    False,
                )
            )
        else:
            field = (
                "flagged_adversarial"
                if source.get("flagged_adversarial") is not False
                else "verdict_class"
            )
            expected: object = False if field == "flagged_adversarial" else sorted(ELIGIBLE)
            observed = source.get(field)
            if observed != expected if field == "flagged_adversarial" else observed not in ELIGIBLE:
                failures.append(
                    failed_check(
                        "required_scientific_producer",
                        row["task_id"],
                        row["planned_path"],
                        field,
                        "==" if field == "flagged_adversarial" else "in",
                        expected,
                        observed,
                    )
                )
    verdict = "blocked" if failures else "null"
    honest = (
        "complete_blocked_required_v672_evidence" if failures else "complete_null_v672_evidence"
    )
    rows[-1]["verdict_class"] = verdict
    rows[-1]["honest_verdict"] = honest
    return {
        "prior_dispositions": rows,
        "source_artifact_hashes": hashes,
        "gate_check_summary": {
            "passed": not failures,
            "failed_count": len(failures),
            "first_failure": failures[0] if failures else None,
            "failed_checks": failures,
        },
        "verdict_class": verdict,
        "honest_verdict": honest,
    }


def cold_reduce(value: object, root: Path, tasks: list[dict[str, Any]]) -> list[str]:
    """Reopen source bytes so a changed result cannot keep an old conclusion."""
    if not isinstance(value, dict):
        return ["artifact_object_required"]
    expected = account(root, tasks)
    return [
        f"{key}_mismatch"
        for key, item in expected.items()
        if canonical_hash(value.get(key)) != canonical_hash(item)
    ]
