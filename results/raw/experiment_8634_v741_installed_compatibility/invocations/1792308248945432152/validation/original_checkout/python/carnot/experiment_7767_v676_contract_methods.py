"""Bind V676 planning sources without promoting unmeasured science.

REQ-REPORT-7767; SCENARIO-REPORT-7767-CONTRACT/CUSTODY/TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.676"
DESIGN = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PRESERVED = Path("openspec/change-proposals/research-roadmap-v675-preserved-20260927.md")
METHOD = Path("docs/research-notes/v676-method-map.md")
RAW = Path("results/raw/experiment_7767_v676_contract_methods")
RESULT = Path("results/experiment_7767_v676_contract_methods.json")
MODULE = Path("python/carnot/experiment_7767_v676_contract_methods.py")
TEST = Path("tests/python/test_experiment_7767_v676_contract_methods.py")
CLI = Path("scripts/experiments/experiment_7767_v676_contract_methods.py")
FIELDS = (
    "id",
    "title",
    "phase",
    "deliverable",
    "inference_substrate_class",
    "MODEL_SPECS",
    "gated_on",
)
PRIOR = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")


def resolve_authority(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Use only a staged or active roadmap for this exact milestone."""
    candidates: list[dict[str, Any]] = []
    for name in ("research-roadmap-next.yaml", "research-roadmap.yaml"):
        path = root / name
        value = yaml.safe_load(path.read_text()) if path.is_file() else None
        milestone = value.get("milestone") if isinstance(value, dict) else None
        candidates.append({"path": name, "exists": path.is_file(), "milestone": milestone})
        if milestone == MILESTONE:
            from scripts.roadmap_schema import Roadmap

            Roadmap.model_validate(value)
            return path, value, candidates
    raise ValueError("matching V676 authority missing")


def parse_design(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the visible table and JSON block as independent authorities."""
    section = text.split("## Exact task contract", 1)[1]
    table: list[dict[str, Any]] = []
    for line in section.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0].isdigit():
            table.append(
                dict(
                    zip(
                        (
                            "order",
                            "id",
                            "title",
                            "phase",
                            "deliverable",
                            "inference_substrate_class",
                            "MODEL_SPECS",
                        ),
                        (
                            int(cells[0]),
                            cells[1],
                            cells[2],
                            int(cells[3]),
                            cells[4],
                            cells[5],
                            json.loads(cells[6]),
                        ),
                    )
                )
            )
    block = re.search(r"<!-- V676_TASK_CONTRACT_BEGIN -->\s*```json\s*(.*?)\s*```", section, re.S)
    if block is None:
        raise ValueError("V676 JSON contract missing")
    machine = json.loads(block.group(1))
    if machine.get("milestone") != MILESTONE:
        raise ValueError("V676 JSON milestone mismatch")
    return table, machine["tasks"]


def required_fields(prompt: str) -> set[str]:
    """Limit gate evidence to the producer's declared artifact fields."""
    section = prompt.split("REQUIRED ARTIFACT FIELDS:", 1)[1].split("Run command:", 1)[0]
    fields: set[str] = set()
    for line in section.splitlines():
        match = re.match(r"\s*-\s+([A-Za-z_][\w]*(?:,\s*[A-Za-z_][\w]*)*)\s*:", line)
        if match:
            fields.update(name.strip() for name in match.group(1).split(","))
    return fields


def compare_contract(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Keep every task row even when one authority is missing or wrong."""
    table, machine = parse_design(text)
    tasks = roadmap["tasks"]
    errors = []
    if roadmap.get("milestone") != MILESTONE:
        errors.append("roadmap_milestone")
    if (len(table), len(machine), len(tasks)) != (14, 14, 14):
        errors.append("task_count")
    rows = []
    for index in range(14):
        shown = table[index] if index < len(table) else {}
        expected = machine[index] if index < len(machine) else {}
        actual = tasks[index] if index < len(tasks) else {}
        checks = {
            field: (field in shown or field == "gated_on")
            and field in expected
            and actual.get(field, [] if field == "gated_on" else None)
            == expected[field]
            == shown.get(field, expected[field])
            for field in FIELDS
        }
        checks["order"] = shown.get("order") == index + 1
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{7767 + index}-")
        checks["milestone"] = actual.get("milestone") == MILESTONE
        prior = actual.get("prior_failures") or []
        checks["prior_fields"] = bool(prior) and all(
            all(key in item and item[key] not in (None, "") for key in PRIOR) for item in prior
        )
        checks["retirement"] = bool(prior) and all(
            item.get("retire_if_same_verdict") is True for item in prior
        )
        gates = actual.get("gated_on") or []
        checks["gate_keys"] = all(
            set(gate) == {"upstream", "artifact_field", "op", "value"} for gate in gates
        )
        checks["producer_precedes"] = all(
            any(task.get("id") == gate.get("upstream") for task in tasks[:index]) for gate in gates
        )
        checks["producer_field"] = all(
            any(
                task.get("id") == gate.get("upstream")
                and gate.get("artifact_field") in required_fields(task["prompt"])
                for task in tasks[:index]
            )
            for gate in gates
        )
        matched = all(checks.values())
        rows.append(
            {
                "unit_id": actual.get("id", f"missing-{index + 1}"),
                "order": index + 1,
                "arm": "three_source_contract",
                "checks": checks,
                "matched": matched,
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "absolute_metric": int(matched),
                "censored": False,
                "exclusions": [] if matched else ["contract_mismatch"],
                "raw_paths": [str(DESIGN), "selected_roadmap"],
                "input_hashes": {
                    "design": canonical_hash(text),
                    "roadmap": canonical_hash(roadmap),
                },
                "effective_independent_groups": 1,
            }
        )
    if not all(row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "rows": rows,
        "table_count": len(table),
        "machine_count": len(machine),
        "roadmap_count": len(tasks),
    }


def mutate(roadmap: dict[str, Any], name: str) -> dict[str, Any]:
    """Change only a private copy to prove every contract guard can reject drift."""
    value = deepcopy(roadmap)
    tasks = value["tasks"]
    if name == "drop":
        tasks.pop()
    elif name == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif name == "title":
        tasks[0]["title"] = "Unregistered title"
    elif name in {"unknown_producer", "gate_field"}:
        gate = next(task for task in tasks if task.get("gated_on"))["gated_on"][0]
        gate["upstream" if name == "unknown_producer" else "artifact_field"] = "unknown"
    elif name == "substrate":
        tasks[6]["inference_substrate_class"] = "no_model_load"
    elif name.startswith("prior_"):
        key = {
            "prior_experiment_id": "experiment_id",
            "prior_verdict": "verdict",
            "prior_addressed_by": "addressed_by",
            "prior_retirement": "retire_if_same_verdict",
        }[name]
        del tasks[0]["prior_failures"][0][key]
    else:
        raise ValueError(f"unknown mutation: {name}")
    return value


def prior_inventory(root: Path) -> list[dict[str, Any]]:
    """Read declared V675 producer paths; discover skips by receipt content."""
    text = (root / PRESERVED).read_text()
    block = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    if block is None:
        raise ValueError("preserved V675 machine contract missing")
    tasks = json.loads(block.group(1))["tasks"]
    if [task["id"].split("-")[0] for task in tasks] != [f"exp{i}" for i in range(7753, 7767)]:
        raise ValueError("preserved V675 task order changed")
    rows = []
    for number, task in enumerate(tasks, 7753):
        declared = Path(task["deliverable"])
        producer = root / declared
        value = json.loads(producer.read_text()) if producer.is_file() else {}
        verdict_class = value.get("verdict_class")
        state = "missing" if not producer.is_file() else verdict_class or "unclassified"
        receipts = []
        if state == "missing":
            for candidate in sorted((root / "results").glob(f"experiment_{number}_*.json")):
                receipt_value = json.loads(candidate.read_text())
                if (
                    receipt_value.get("schema") == "blocked_gate_check_v1"
                    and receipt_value.get("experiment") == number
                ):
                    receipts.append(candidate)
        receipt = receipts[0] if receipts else None
        rows.append(
            {
                "experiment": number,
                "task_id": task["id"],
                "producer_path": str(declared),
                "producer_sha256": sha256_file(producer) if producer.is_file() else None,
                "producer_state": state,
                "verdict_class": verdict_class,
                "honest_verdict": value.get("honest_verdict"),
                "pre_gate_path": str(receipt.relative_to(root)) if receipt else None,
                "pre_gate_sha256": sha256_file(receipt) if receipt else None,
                "pre_gate_verdict": json.loads(receipt.read_text()).get("honest_verdict")
                if receipt
                else None,
            }
        )
    return rows


def source_hashes(root: Path, authority: Path, paths: list[Path]) -> list[dict[str, Any]]:
    """Bind exact current bytes and record absent declared inputs explicitly."""
    rows = []
    for relative in [authority, *paths]:
        actual = root / relative
        exists = actual.is_file()
        rows.append(
            {
                "path": str(relative),
                "role": "current_input",
                "exists": exists,
                "sha256": sha256_file(actual) if exists else None,
                "date": "2026-09-27",
                "imported_fields": ["bytes"] if exists else [],
                "eligible": exists,
            }
        )
    return rows


def cold_validate(root: Path, raw: Path, sources: list[dict[str, Any]]) -> bool:
    """Reopen source bytes and reproduce raw rows in a fresh process."""
    if any(
        not row["exists"]
        or not (root / row["path"]).is_file()
        or sha256_file(root / row["path"]) != row["sha256"]
        for row in sources
    ):
        return False
    authority = root / sources[0]["path"]
    design = root / DESIGN
    comparison = compare_contract(design.read_text(), yaml.safe_load(authority.read_text()))
    return comparison["passed"] and comparison["rows"] == json.loads(raw.read_text())


def failed_checks(
    comparison: dict[str, Any], sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Name each failed operand so a block is distinguishable from a typo."""
    failures = []
    design_hash = next((row["sha256"] for row in sources if row["path"] == str(DESIGN)), None)
    for row in sources:
        if not row["exists"]:
            failures.append(
                {
                    "upstream_id": row["path"],
                    "artifact_path": row["path"],
                    "artifact_sha256": None,
                    "field": "exists",
                    "op": "==",
                    "expected": True,
                    "observed": False,
                }
            )
    for row in comparison["rows"]:
        for field, passed in row["checks"].items():
            if not passed:
                failures.append(
                    {
                        "upstream_id": row["unit_id"],
                        "artifact_path": str(DESIGN),
                        "artifact_sha256": design_hash,
                        "field": field,
                        "op": "==",
                        "expected": True,
                        "observed": False,
                    }
                )
    return failures
