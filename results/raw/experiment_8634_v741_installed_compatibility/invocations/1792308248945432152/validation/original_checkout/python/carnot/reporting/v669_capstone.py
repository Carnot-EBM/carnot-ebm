"""Reduce V669 custody without turning missing science into measured zeros.

REQ-REPORT-7684 and REQ-CAPSTONE-7684. The reducer reads source bytes again
in a cold process, so a capstone claim cannot substitute for its inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting import v669_contract


REQUIRED_SCIENCE = (7675, 7678, 7679)


def reduce_audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count source families once even when controls share the same family."""

    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_stage.setdefault(row["evidence_stage"], []).append(row)
    cohort = by_stage.get("cohort_source_control", [])
    fixture = by_stage.get("fixture_protocol", [])
    quote = by_stage.get("qwen_quote_diagnostic", [])
    original = [row for row in cohort if row["arm"] == "original_source"]
    bound = [row for row in fixture if row["arm"] == "bound_relation"]
    return {
        "cohort_families": len({row["independent_unit"] for row in cohort}),
        "cohort_unknown_families": len(
            {row["independent_unit"] for row in original if row["censored"]}
        ),
        "fixture_families": len({row["independent_unit"] for row in fixture}),
        "fixture_correct_families": sum(
            row["observed"] == row["truth"] for row in bound if row["truth"] is not None
        ),
        "quote_families": len({row["independent_unit"] for row in quote}),
        "quote_supported_relations": sum(
            row["raw_metrics"]["full_proposition_supported"] for row in quote
        ),
    }


def authority(root: Path) -> dict[str, Any]:
    """Compare the activated or staged roadmap with two independent designs."""

    path, roadmap, candidates = v669_contract.resolve_authority(root)
    design = (root / v669_contract.DESIGN_PATH).read_text(encoding="utf-8")
    comparison = v669_contract.compare_authorities(design, roadmap)
    if not comparison["passed"]:
        raise ValueError(f"V669 authority mismatch: {comparison['errors']}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "design_path": str(v669_contract.DESIGN_PATH),
        "design_sha256": sha256_file(root / v669_contract.DESIGN_PATH),
        "candidates": candidates,
        "comparison": comparison,
        "tasks": roadmap["tasks"],
    }


def _failure(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: object,
    observed: object,
) -> dict[str, Any]:
    """Keep every operand needed to locate an unavailable branch."""

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
    """Find only conductor gate custody, never mistake it for a producer."""

    number = task["id"].split("-", 1)[0][3:]
    for path in sorted((root / "results").glob(f"experiment_{number}_*.json")):
        try:
            if json.loads(path.read_text()).get("schema") == "blocked_gate_check_v1":
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


def account(root: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the exact roster and evaluate registered upstream gates."""

    if len(tasks) != 14 or any(
        not task["id"].startswith(f"exp{7671 + index}-") for index, task in enumerate(tasks)
    ):
        raise ValueError("V669 fourteen-task order required")
    payloads: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    producers: list[dict[str, str]] = []
    pre_gates: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    failed: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_id = task["id"]
        label = task["deliverable"]
        path = root / label
        if index == 13:
            kind, value, evidence = "current_capstone", None, None
        elif path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            payloads[task_id] = value
            kind, evidence = "producer", label
            producers.append({"task_id": task_id, "path": label, "sha256": sha256_file(path)})
        else:
            value = None
            receipt = _pre_gate(root, task)
            kind = "pre_gate_receipt" if receipt else "absent"
            evidence = receipt.relative_to(root).as_posix() if receipt else None
            if receipt:
                pre_gates.append(
                    {"task_id": task_id, "path": evidence, "sha256": sha256_file(receipt)}
                )
            else:
                missing.append({"task_id": task_id, "path": label})
        rows.append(
            {
                "order": index + 1,
                "task_id": task_id,
                "planned_path": label,
                "availability": kind,
                "evidence_path": evidence,
                "verdict_class": value.get("verdict_class") if value else None,
                "honest_verdict": value.get("honest_verdict") if value else None,
                "scientific_disposition": value.get("verdict_class") if value else None,
                "flagged_adversarial": value.get("flagged_adversarial") if value else None,
                "registered_gates": [],
            }
        )
    task_by_id = {task["id"]: task for task in tasks}
    for row, task in zip(rows, tasks, strict=True):
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            field = gate["artifact_field"]
            upstream_path = task_by_id[upstream]["deliverable"]
            source = payloads.get(upstream)
            observed = source.get(field) if source else None
            operator = gate["op"]
            expected = gate["value"]
            passed = observed == expected if operator == "==" else observed in expected
            check = {
                "check": "registered_gate",
                "upstream": upstream,
                "path": upstream_path,
                "field": field,
                "operator": operator,
                "expected": expected,
                "observed": observed,
                "passed": passed,
                "consumer": task["id"],
            }
            row["registered_gates"].append(check)
            if not passed:
                failed.append(check)
    log_path = root / v669_contract.LOG_PATH
    log_lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
    for row, task in zip(rows[:-1], tasks[:-1], strict=True):
        if row["availability"] != "absent" or not row["registered_gates"]:
            continue
        gate_log = [
            line for line in log_lines if task["title"][:44] in line and "GATE_BLOCK" in line
        ]
        if gate_log:
            row["availability"] = "gate_skipped"
            row["evidence_path"] = str(v669_contract.LOG_PATH)
            row["log_custody"] = gate_log[-1]
    for number in REQUIRED_SCIENCE:
        row = rows[number - 7671]
        source = payloads.get(row["task_id"])
        if source is None:
            failed.append(
                _failure(
                    "required_scientific_producer",
                    row["task_id"],
                    row["planned_path"],
                    "exists",
                    "==",
                    True,
                    False,
                )
            )
        elif source.get("verdict_class") not in {"positive", "null", "circular_positive"}:
            failed.append(
                _failure(
                    "required_scientific_producer",
                    row["task_id"],
                    row["planned_path"],
                    "verdict_class",
                    "in",
                    ["positive", "null", "circular_positive"],
                    source.get("verdict_class"),
                )
            )
    for row in rows[:-1]:
        if (
            row["availability"] == "absent"
            and not row["registered_gates"]
            and not any(
                item["upstream"] == row["task_id"] and item["field"] == "exists" for item in failed
            )
        ):
            failed.append(
                _failure(
                    "producer_absent",
                    row["task_id"],
                    row["planned_path"],
                    "exists",
                    "==",
                    True,
                    False,
                )
            )
    verdict_class = "blocked" if failed else "null"
    honest_verdict = (
        "complete_blocked_required_v669_scientific_evidence"
        if failed
        else "complete_null_v669_no_registered_benefit"
    )
    rows[-1]["verdict_class"] = verdict_class
    rows[-1]["honest_verdict"] = honest_verdict
    rows[-1]["scientific_disposition"] = verdict_class
    return {
        "prior_dispositions": rows,
        "source_artifact_hashes": {
            "producers": producers,
            "pre_gate_receipts": pre_gates,
            "missing_evidence": missing,
            "conductor_log": (
                [{"path": str(v669_contract.LOG_PATH), "sha256": sha256_file(log_path)}]
                if log_path.is_file()
                else []
            ),
        },
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
    """Reopen producer bytes and reject changed custody or gate interpretation."""

    if not isinstance(value, dict):
        return ["artifact_object_required"]
    try:
        expected = account(root, tasks)
    except (OSError, ValueError, KeyError) as exc:
        return [f"source_reduction_failed:{type(exc).__name__}"]
    errors = [
        f"{key}_mismatch"
        for key in expected
        if key != "source_artifact_hashes"
        if canonical_hash(value.get(key)) != canonical_hash(expected[key])
    ]
    actual_hashes = value.get("source_artifact_hashes", {})
    for key, items in expected["source_artifact_hashes"].items():
        if canonical_hash(actual_hashes.get(key)) != canonical_hash(items):
            errors.append(f"source_artifact_hashes.{key}_mismatch")
    return errors
