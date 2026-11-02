"""Compare V672 planning sources without granting scientific readiness.

The design must contain a separately written table and JSON contract. A
missing source stays missing; copying YAML into it would erase the check.
REQ-REPORT-7713.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml


MILESTONE = "2026.09.672"
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESULT_PATH = Path("results/experiment_7713_v672_contract_methods.json")
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
    """Prefer matching staged bytes, then the active roadmap."""

    candidates: list[dict[str, Any]] = []
    selected: tuple[Path, dict[str, Any]] | None = None
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        data = yaml.safe_load(path.read_text()) if path.is_file() else None
        observed = data.get("milestone") if isinstance(data, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "observed": observed})
        if selected is None and observed == MILESTONE and isinstance(data, dict):
            selected = (path, data)
    if selected is None:
        raise ValueError("V672 matching roadmap unavailable")
    return selected[0], selected[1], candidates


def parse_design(
    text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, list[str]]:
    """Parse the table and machine block independently from the prose."""

    errors: list[str] = []
    section = text.split("## Exact Task Contract", 1)
    body = section[1] if len(section) == 2 else ""
    table: list[dict[str, Any]] = []
    for line in body.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        try:
            table.append(
                {
                    "order": int(cells[0]),
                    "id": cells[1].strip("`"),
                    "phase": int(cells[2]),
                    "title": cells[3],
                    "deliverable": cells[4].strip("`"),
                    "inference_substrate_class": cells[5].strip("`"),
                    "MODEL_SPECS": json.loads(cells[6].strip("`")),
                    "gated_on": json.loads(cells[7].strip("`")),
                }
            )
        except (ValueError, json.JSONDecodeError):
            errors.append("design_table_invalid")
    if not table:
        errors.append("design_table_missing")
    block = re.search(r"```json\s*(.*?)\s*```", body, re.S)
    if block is None:
        errors.append("design_json_missing")
        return [], table, None, errors
    try:
        machine = json.loads(block.group(1))
        if not isinstance(machine, dict) or not isinstance(machine.get("tasks"), list):
            raise ValueError("machine tasks absent")
        return machine["tasks"], table, machine.get("milestone"), errors
    except (ValueError, json.JSONDecodeError):
        errors.append("design_json_invalid")
        return [], table, None, errors


def compare_authorities(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Retain one explicit comparison row for every planned task slot."""

    machine, table, milestone, errors = parse_design(text)
    tasks = roadmap.get("tasks", [])
    if milestone != MILESTONE or roadmap.get("milestone") != MILESTONE:
        errors.append("milestone")
    if len(machine) != 13 or len(table) != 13 or len(tasks) != 13:
        errors.append("task_count")
    rows = []
    for index in range(13):
        expected = machine[index] if index < len(machine) else {}
        shown = table[index] if index < len(table) else {}
        actual = tasks[index] if index < len(tasks) else {}
        checks = {
            field: field in expected
            and actual.get(field, [] if field == "gated_on" else None)
            == expected.get(field)
            == shown.get(field)
            for field in FIELDS
        }
        checks["order"] = shown.get("order") == index + 1
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{7713 + index}-")
        checks["task_milestone"] = actual.get("milestone") == MILESTONE
        checks["gate_producer"] = all(
            gate.get("upstream") in {task.get("id") for task in tasks[:index]}
            and gate.get("artifact_field")
            in next(
                (
                    task.get("prompt", "")
                    for task in tasks[:index]
                    if task.get("id") == gate.get("upstream")
                ),
                "",
            )
            for gate in actual.get("gated_on", [])
        )
        matched = all(checks.values())
        rows.append(
            {
                "unit_id": actual.get("id", f"missing-{index + 1}"),
                "arm": "design_vs_authority",
                "order": index + 1,
                "expected": expected,
                "observed": {key: actual.get(key) for key in FIELDS},
                "checks": checks,
                "matched": matched,
                "absolute_metric": int(matched),
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "counts": {"independent_groups": 1},
                "exclusions": [] if matched else ["contract_mismatch_or_missing_design"],
                "censored": False,
                "provenance": [str(DESIGN_PATH), "selected_roadmap"],
            }
        )
    if not all(row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {"passed": not errors, "errors": sorted(set(errors)), "rows": rows}


def mutate_authority(roadmap: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Make each private defect without changing the source authority."""

    changed = deepcopy(roadmap)
    tasks = changed["tasks"]
    if mutation == "delete":
        tasks.pop()
    elif mutation == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "stale":
        changed["milestone"] = "2026.09.671"
    elif mutation == "producer_field":
        next(task for task in tasks if task.get("gated_on"))["gated_on"][0]["artifact_field"] = (
            "wrong"
        )
    elif mutation == "model":
        next(task for task in tasks if task["MODEL_SPECS"])["MODEL_SPECS"] = []
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return changed


def classify(
    comparison: dict[str, Any], inputs_present: bool, validation_passed: bool
) -> tuple[str, str]:
    """Keep missing independent input separate from failed owned checks."""

    if not inputs_present or any(error.startswith("design_") for error in comparison["errors"]):
        return "complete_blocked_v672_independent_contract", "blocked"
    if not validation_passed or not comparison["passed"]:
        return "complete_disqualified_v672_contract_validation", "disqualified"
    return "complete_null_v672_contract_methods", "null"
