"""Reduce V669 authority and V668 custody. REQ-REPORT-7671."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import parse_markdown_contract
from carnot.reporting.current_work_receipt import sha256_file

MILESTONE = "2026.09.669"
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PRIOR_DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-v668-preserved-20260925.md")
LOG_PATH = Path("ops/conductor-log.md")
RESULT_PATH = Path("results/experiment_7671_v669_contract_methods.json")
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
    """Use a staged copy only when it actually declares this milestone."""

    candidates = []
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        milestone = value.get("milestone") if isinstance(value, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "observed": milestone})
        if milestone == MILESTONE:
            return path, value, candidates
    raise ValueError("V669 roadmap authority unavailable")


def machine_contract(text: str) -> list[dict[str, Any]]:
    """Read the independent machine block; prose alone cannot prove parity."""

    match = re.search(r"<!-- V669-TASK-CONTRACT-BEGIN -->\s*```json\s*(.*?)\s*```", text, re.S)
    if match is None:
        raise ValueError("V669 machine contract missing")
    value = json.loads(match.group(1))
    if not isinstance(value, list):
        raise ValueError("V669 machine contract must be a list")
    return value


def compare_authorities(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Compare independent ordered rows, including models and structured gates."""

    machine = machine_contract(text)
    table = parse_markdown_contract(text)
    tasks = roadmap.get("tasks", [])
    errors = []
    if table["milestone"] != MILESTONE or roadmap.get("milestone") != MILESTONE:
        errors.append("milestone")
    if any(len(group) != 14 for group in (machine, table["tasks"], tasks)):
        errors.append("task_count")
    rows = []
    for index in range(14):
        expected = machine[index] if index < len(machine) else {}
        shown = table["tasks"][index] if index < len(table["tasks"]) else {}
        actual = tasks[index] if index < len(tasks) else {}
        checks = {
            key: actual.get(key, [] if key == "gated_on" else None) == expected.get(key)
            for key in FIELDS
        }
        checks["sequence"] = str(expected.get("id", "")).startswith(f"exp{7671 + index}-")
        checks["table"] = all(
            shown.get(key) == value
            for key, value in (
                ("order", index + 1),
                ("id", expected.get("id")),
                ("title", expected.get("title")),
                ("phase", expected.get("phase")),
                ("deliverable", expected.get("deliverable")),
                ("substrate", expected.get("inference_substrate_class")),
                ("gates", expected.get("gated_on")),
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
    """Attack a private copy so a successful comparison has a falsification check."""

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
        next(task for task in tasks if task.get("gated_on"))["gated_on"][0]["upstream"] = tasks[2][
            "id"
        ]
    elif mutation == "model":
        next(task for task in tasks if task["MODEL_SPECS"])["MODEL_SPECS"] = []
    elif mutation == "stale":
        changed["milestone"] = "2026.09.668"
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return changed


def prior_machine_contract(root: Path) -> list[dict[str, Any]]:
    """Read the preserved V668 task identities without touching its bytes."""

    text = (root / PRIOR_DESIGN_PATH).read_text(encoding="utf-8")
    match = re.search(r"<!-- V668-TASK-CONTRACT-BEGIN -->\s*```json\s*(.*?)\s*```", text, re.S)
    if match is None:
        raise ValueError("V668 preserved contract missing")
    tasks = json.loads(match.group(1))
    if len(tasks) != 14:
        raise ValueError("V668 prior task count changed")
    return tasks


def collect_prior_dispositions(root: Path) -> list[dict[str, Any]]:
    """Authenticate producer files; keep missing work tied to literal log lines."""

    tasks = prior_machine_contract(root)
    log_lines = (root / LOG_PATH).read_text(encoding="utf-8").splitlines()
    log_hash = sha256_file(root / LOG_PATH)
    rows = []
    for index, task in enumerate(tasks):
        path = root / task["deliverable"]
        exists = path.is_file()
        if exists:
            producer = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "order": index + 1,
                    "task_id": task["id"],
                    "planned_path": task["deliverable"],
                    "custody_kind": "producer_file",
                    "authentication_path": task["deliverable"],
                    "authentication_sha256": sha256_file(path),
                    "authenticated": producer.get("milestone") == "2026.09.668"
                    and str(7657 + index)
                    in str(producer.get("experiment_id", producer.get("experiment", ""))),
                    "honest_verdict": producer.get("honest_verdict"),
                    "verdict_class": producer.get("verdict_class"),
                    "exact_supported_proposals": producer.get("acceptance_gate_results", {})
                    .get("coverage", {})
                    .get("measured_operands", {})
                    .get("exact_supported")
                    if index == 8
                    else None,
                    "log_custody": None,
                }
            )
        else:
            marker = task["title"][:44]
            evidence = [line for line in log_lines if marker in line]
            rows.append(
                {
                    "order": index + 1,
                    "task_id": task["id"],
                    "planned_path": task["deliverable"],
                    "custody_kind": "missing_producer_log",
                    "authentication_path": str(LOG_PATH),
                    "authentication_sha256": log_hash,
                    "authenticated": bool(evidence) and 7657 + index >= 7667,
                    "honest_verdict": None,
                    "verdict_class": None,
                    "exact_supported_proposals": None,
                    "log_custody": evidence[-3:],
                }
            )
    return rows


def input_check(
    check: str, upstream: str, field: str, operator: str, expected: object, observed: object
) -> dict[str, Any]:
    """Carry exact failed operands into a blocked result for later diagnosis."""

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
    """Keep invalid current work, absent evidence and valid null separate."""

    if not valid:
        return "complete_disqualified_v669_validation", "disqualified"
    if not inputs_present:
        return "complete_blocked_v669_external_evidence", "blocked"
    return "complete_null_v669_contract_methods", "null"
