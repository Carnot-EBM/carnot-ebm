"""Reduce V671 authority and literal V670 custody. REQ-REPORT-7699."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import sha256_file

MILESTONE = "2026.09.671"
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PRIOR_DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-v670-preserved-20260926.md")
LOG_PATH = Path("ops/conductor-log.md")
RESULT_PATH = Path("results/experiment_7699_v671_contract_methods.json")
FIELDS = (
    "id",
    "title",
    "phase",
    "deliverable",
    "inference_substrate_class",
    "MODEL_SPECS",
    "gated_on",
)


def resolve_authority(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Choose matching staged bytes first, then matching activated bytes."""

    candidates = []
    selected = None
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        observed = value.get("milestone") if isinstance(value, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "observed": observed})
        if selected is None and observed == MILESTONE:
            selected = (path, value)
    if selected is None:
        raise ValueError("V671 matching roadmap unavailable")
    return selected[0], selected[1], candidates


def design_contract(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Read the independent table and machine block in the exact-task section."""

    section = text.split("## Exact Task Contract", 1)[1]
    block = re.search(r"```json\s*(.*?)\s*```", section, re.S)
    if block is None:
        raise ValueError("machine contract missing")
    machine = json.loads(block.group(1))
    table = []
    for line in section.splitlines():
        match = re.match(r"\| (\d+) \| `([^`]+)` \| (\d+) \| ([^|]+) \| `([^`]+)` \|", line)
        if match:
            order, task_id, phase, title, deliverable = match.groups()
            table.append(
                {
                    "order": int(order),
                    "id": task_id,
                    "phase": int(phase),
                    "title": title.strip(),
                    "deliverable": deliverable,
                }
            )
    return machine["tasks"], table, machine["milestone"]


def compare_authorities(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Compare all fields, ordered table cells and complete structured gates."""

    machine, table, milestone = design_contract(text)
    tasks = roadmap.get("tasks", [])
    errors = []
    if milestone != MILESTONE or roadmap.get("milestone") != MILESTONE:
        errors.append("milestone")
    if len(machine) != 14 or len(table) != 14 or len(tasks) != 14:
        errors.append("task_count")
    rows = []
    for index in range(14):
        expected = machine[index] if index < len(machine) else {}
        actual = tasks[index] if index < len(tasks) else {}
        shown = table[index] if index < len(table) else {}
        checks = {
            key: actual.get(key, [] if key == "gated_on" else None) == expected.get(key)
            for key in FIELDS
        }
        checks["sequence"] = str(expected.get("id", "")).startswith(f"exp{7699 + index}-")
        checks["table"] = (
            all(
                shown.get(key) == expected.get(key)
                for key in ("id", "title", "phase", "deliverable")
            )
            and shown.get("order") == index + 1
        )
        checks["self_input"] = all(
            gate.get("upstream") != actual.get("id") for gate in actual.get("gated_on", [])
        )
        matched = all(checks.values())
        rows.append(
            {
                "unit_id": expected.get("id"),
                "arm": "design_vs_authority",
                "order": index + 1,
                "expected": {key: expected.get(key) for key in FIELDS},
                "observed": {
                    key: actual.get(key, [] if key == "gated_on" else None) for key in FIELDS
                },
                "checks": checks,
                "matched": matched,
                "absolute_metric": int(matched),
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "counts": {"independent_groups": 1},
                "exclusions": [],
                "censored": False,
                "provenance": [str(DESIGN_PATH), "selected_roadmap"],
                "seed": None,
            }
        )
    if not all(row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {"passed": not errors, "errors": errors, "rows": rows}


def mutate_authority(roadmap: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Change one private contract field to demonstrate rejection."""

    changed = deepcopy(roadmap)
    tasks = changed["tasks"]
    if mutation == "delete":
        tasks.pop()
    elif mutation == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "producer_field":
        next(task for task in tasks if task.get("gated_on"))["gated_on"][0]["artifact_field"] = (
            "wrong"
        )
    elif mutation == "model":
        next(task for task in tasks if task["MODEL_SPECS"])["MODEL_SPECS"] = []
    elif mutation == "stale":
        changed["milestone"] = "2026.09.670"
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return changed


def collect_prior_dispositions(root: Path) -> list[dict[str, Any]]:
    """Keep missing V670 producers tied to their literal conductor entries."""

    design = (root / PRIOR_DESIGN_PATH).read_text(encoding="utf-8")
    tasks, _, milestone = design_contract(design)
    if milestone != "2026.09.670":
        raise ValueError("V670 preserved contract changed")
    log_path = root / LOG_PATH
    lines = log_path.read_text(encoding="utf-8").splitlines()
    log_hash = sha256_file(log_path)
    rows = []
    for index, task in enumerate(tasks):
        path = root / task["deliverable"]
        marker = task["title"][:44]
        matches = [line for line in lines if marker in line and "2026-09-26" in line]
        failures = [
            line
            for line in matches
            if "| FAIL | Codex CLI error:" in line and "usage limit" in line
        ]
        skips = [line for line in matches if "| GATE_BLOCK | Pre-emptive skip:" in line]
        disposition = (
            "usage_limit_three_attempts"
            if len(failures) == 3
            else "gate_skipped"
            if skips
            else "unresolved"
        )
        custody = failures if failures else skips
        rows.append(
            {
                "order": index + 1,
                "task_id": task["id"],
                "planned_path": task["deliverable"],
                "custody_kind": "not_emitted" if not path.is_file() else "producer_file",
                "disposition": disposition,
                "authentication_path": str(LOG_PATH),
                "authentication_sha256": log_hash,
                "authenticated": not path.exists() and disposition != "unresolved",
                "log_lines": custody,
                "honest_verdict": None,
                "verdict_class": None,
                "prior_failure_marker": "not_emitted_usage_limit_three_attempts"
                if failures
                else "not_emitted_gate_skip",
            }
        )
    return rows


def input_check(
    check: str, upstream: str, field: str, operator: str, expected: object, observed: object
) -> dict[str, Any]:
    """Retain every operand needed to diagnose a missing prerequisite."""

    if operator != "==":
        raise ValueError("only equality checks are supported")
    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def classify(valid: bool, inputs_present: bool) -> tuple[str, str]:
    """Keep invalid current work and absent external evidence distinct."""

    if not valid:
        return "complete_disqualified_v671_validation", "disqualified"
    if not inputs_present:
        return "complete_blocked_v671_external_evidence", "blocked"
    return "complete_null_v671_contract_methods", "null"
