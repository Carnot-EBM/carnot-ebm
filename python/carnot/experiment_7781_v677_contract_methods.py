"""Bind V677's exact task contract without inferring a scientific effect.

REQ-REPORT-7781; SCENARIO-REPORT-7781-CONTRACT/CUSTODY/TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.experiment_7767_v676_contract_methods import required_fields

ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.677"
DESIGN = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PRESERVED = Path("openspec/change-proposals/research-roadmap-v676-preserved-20260927.md")
DESIGN_SNAPSHOT = Path("docs/research-notes/v677-authority-snapshots/design.md")
YAML_SNAPSHOT = Path("docs/research-notes/v677-authority-snapshots/roadmap.yaml")
METHOD = Path("docs/research-notes/v677-method-map.md")
RAW = Path("results/raw/experiment_7781_v677_contract_methods")
RESULT = Path("results/experiment_7781_v677_contract_methods.json")
MODULE = Path("python/carnot/experiment_7781_v677_contract_methods.py")
TEST = Path("tests/python/test_experiment_7781_v677_contract_methods.py")
CLI = Path("scripts/experiments/experiment_7781_v677_contract_methods.py")
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
    """Prefer a matching staged V677 authority; never accept a later milestone."""
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
    raise ValueError("matching V677 authority missing")


def parse_design(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the literal five-column table and embedded JSON independently."""
    section = text.split("## Exact task contract", 1)[1]
    table = []
    for line in section.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            table.append(
                dict(
                    zip(
                        ("order", "id", "title", "phase", "deliverable"),
                        (int(cells[0]), cells[1], cells[2], int(cells[3]), cells[4]),
                    )
                )
            )
    block = re.search(r"<!-- V677_TASK_CONTRACT_START -->\s*```json\s*(.*?)\s*```", section, re.S)
    if block is None:
        raise ValueError("V677 JSON contract missing")
    machine = json.loads(block.group(1))
    if machine.get("milestone") != MILESTONE:
        raise ValueError("V677 JSON milestone mismatch")
    return table, machine["tasks"]


def compare_contract(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Keep all fourteen checks visible even when a private mutation fails."""
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
            field: field in expected
            and actual.get(field, [] if field == "gated_on" else None) == expected[field]
            for field in FIELDS
        }
        checks.update(
            {
                f"table_{field}": shown.get(field) == expected.get(field)
                for field in ("id", "title", "phase", "deliverable")
            }
        )
        checks["order"] = shown.get("order") == index + 1
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{7781 + index}-")
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
                "raw_paths": [str(DESIGN_SNAPSHOT), str(YAML_SNAPSHOT)],
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
    """Change a private authority copy so each registered refusal is executable."""
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
    elif name == "model":
        tasks[6]["MODEL_SPECS"] = ["wrong/model"]
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
    """Use V676 declared deliverables; queue receipts never become producers."""
    text = (root / PRESERVED).read_text()
    block = re.search(r"<!-- V676_TASK_CONTRACT_BEGIN -->\s*```json\s*(.*?)\s*```", text, re.S)
    if block is None:
        raise ValueError("preserved V676 machine contract missing")
    tasks = json.loads(block.group(1))["tasks"]
    if [task["id"].split("-")[0] for task in tasks] != [f"exp{i}" for i in range(7767, 7781)]:
        raise ValueError("preserved V676 task order changed")
    rows = []
    for number, task in enumerate(tasks, 7767):
        declared = Path(task["deliverable"])
        producer = root / declared
        evidence = json.loads(producer.read_text()) if producer.is_file() else {}
        state = evidence.get("verdict_class", "unclassified") if producer.is_file() else "missing"
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
                "verdict_class": evidence.get("verdict_class"),
                "honest_verdict": evidence.get("honest_verdict"),
                "pre_gate_path": str(receipt.relative_to(root)) if receipt else None,
                "pre_gate_sha256": sha256_file(receipt) if receipt else None,
                "pre_gate_verdict": json.loads(receipt.read_text()).get("honest_verdict")
                if receipt
                else None,
            }
        )
    return rows


def source_hashes(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    """Record each exact byte source, including an absent declared input."""
    rows = []
    for relative in paths:
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
    """Replay from immutable snapshots even after the active roadmap moves on."""
    if any(
        not row["exists"]
        or not (root / row["path"]).is_file()
        or sha256_file(root / row["path"]) != row["sha256"]
        for row in sources
    ):
        return False
    comparison = compare_contract(
        (root / DESIGN_SNAPSHOT).read_text(), yaml.safe_load((root / YAML_SNAPSHOT).read_text())
    )
    return comparison["passed"] and comparison["rows"] == json.loads(raw.read_text())


def failed_checks(
    comparison: dict[str, Any], sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Name the failed field and observed value instead of hiding a block."""
    failures = []
    for source in sources:
        if not source["exists"]:
            failures.append(
                {
                    "upstream_id": source["path"],
                    "artifact_path": source["path"],
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
                        "artifact_path": str(DESIGN_SNAPSHOT),
                        "artifact_sha256": next(
                            (s["sha256"] for s in sources if s["path"] == str(DESIGN_SNAPSHOT)),
                            None,
                        ),
                        "field": field,
                        "op": "==",
                        "expected": True,
                        "observed": False,
                    }
                )
    return failures
