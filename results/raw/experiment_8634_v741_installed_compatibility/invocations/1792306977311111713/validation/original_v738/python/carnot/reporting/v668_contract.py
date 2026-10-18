"""V668 contract and V667 custody reduction. REQ-REPORT-7657."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import parse_markdown_contract
from carnot.reporting.current_work_receipt import sha256_file

MILESTONE = "2026.09.668"
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PRIOR_PATH = Path("results/experiment_7656_v667_capstone.json")
RESULT_PATH = Path("results/experiment_7657_v668_contract_methods.json")
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
    """Prefer matching staged authority, accepting consumed staging as absent."""

    candidates = []
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        milestone = value.get("milestone") if isinstance(value, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "observed": milestone})
        if milestone == MILESTONE:
            return path, value, candidates
    raise ValueError("V668 roadmap authority unavailable")


def machine_contract(text: str) -> list[dict[str, Any]]:
    """Read the independent JSON machine block from the preserved design."""

    match = re.search(r"<!-- V668-TASK-CONTRACT-BEGIN -->\s*```json\s*(.*?)\s*```", text, re.S)
    if match is None:
        raise ValueError("V668 machine contract missing")
    value = json.loads(match.group(1))
    if not isinstance(value, list):
        raise ValueError("V668 machine contract must be a list")
    return value


def compare_authorities(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Compare each ordered table row, machine row, and executable task."""

    machine = machine_contract(text)
    table = parse_markdown_contract(text)
    tasks = roadmap.get("tasks", [])
    rows = []
    errors = []
    if table["milestone"] != MILESTONE or roadmap.get("milestone") != MILESTONE:
        errors.append("milestone")
    if any(len(group) != 14 for group in (machine, table["tasks"], tasks)):
        errors.append("task_count")
    for index in range(14):
        expected = machine[index] if index < len(machine) else {}
        shown = table["tasks"][index] if index < len(table["tasks"]) else {}
        actual = tasks[index] if index < len(tasks) else {}
        checks = {
            key: actual.get(key, [] if key == "gated_on" else None) == expected.get(key)
            for key in FIELDS
        }
        checks["sequence"] = str(expected.get("id", "")).startswith(f"exp{7657 + index}-")
        checks["table"] = all(
            shown.get(key) == value
            for key, value in (
                ("order", index + 1),
                ("id", expected.get("id")),
                ("title", expected.get("title")),
                ("phase", expected.get("phase")),
                ("deliverable", expected.get("deliverable")),
                ("substrate", expected.get("inference_substrate_class")),
            )
        )
        checks["self_input"] = not any(
            gate.get("upstream") == actual.get("id") for gate in actual.get("gated_on", [])
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
                "provenance": [str(DESIGN_PATH), "selected_roadmap"],
                "exclusions": [],
                "censored": False,
                "seed": None,
            }
        )
    if not all(row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {"passed": not errors, "errors": errors, "rows": rows}


def mutate_authority(roadmap: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Change one private authority copy to test fail-closed comparison."""

    changed = deepcopy(roadmap)
    tasks = changed["tasks"]
    if mutation == "delete":
        tasks.pop()
    elif mutation == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "gate":
        next(task for task in tasks if task.get("gated_on"))["gated_on"][0]["artifact_field"] = (
            "wrong"
        )
    elif mutation == "self_input":
        tasks[2]["gated_on"][0]["upstream"] = tasks[2]["id"]
    elif mutation == "model":
        next(task for task in tasks if task["MODEL_SPECS"])["MODEL_SPECS"] = []
    elif mutation == "stale":
        changed["milestone"] = "2026.09.667"
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return changed


def collect_prior_dispositions(root: Path) -> list[dict[str, Any]]:
    """Authenticate literal V667 producer, pre-gate, absence and capstone bytes."""

    capstone = json.loads((root / PRIOR_PATH).read_text(encoding="utf-8"))
    source = capstone["milestone_dispositions"]
    if len(source) != 14:
        raise ValueError("V667 disposition count changed")
    rows = []
    for index, original in enumerate(source):
        row = deepcopy(original)
        if row["order"] != index + 1 or not row["task_id"].startswith(f"exp{7643 + index}-"):
            raise ValueError("V667 disposition order changed")
        kind = row["custody_kind"]
        label = (
            PRIOR_PATH.as_posix()
            if kind == "current_self"
            else (row.get("actual_path") or row["planned_path"])
        )
        path = root / label
        digest = sha256_file(path) if path.is_file() else None
        expected = digest if kind == "current_self" else row.get("sha256")
        row["authentication_path"] = label
        row["authentication_sha256"] = digest
        row["authenticated"] = (kind == "missing_work" and digest is None and expected is None) or (
            digest is not None and digest == expected
        )
        rows.append(row)
    return rows


def input_check(
    check: str, upstream: str, field: str, operator: str, expected: object, observed: object
) -> dict[str, Any]:
    """Record exact operands for a failed upstream or resource check."""

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
    """Keep validation failure separate from absent external evidence."""

    if not valid:
        return "complete_disqualified_v668_validation", "disqualified"
    if not inputs_present:
        return "complete_blocked_v668_external_evidence", "blocked"
    return "complete_null_v668_contract_methods", "null"
