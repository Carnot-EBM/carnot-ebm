"""Bind V679's administrative contract to exact historical bytes.

REQ-REPORT-7809. A passing contract only shows agreement between published
planning sources; it cannot establish a scientific outcome.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.experiment_7767_v676_contract_methods import required_fields
from carnot.reporting.current_work_receipt import sha256_file

ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.679"
DESIGN = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
DESIGN_SNAPSHOT = Path("docs/research-notes/v679-authority-snapshots/design.md")
YAML_SNAPSHOT = Path("docs/research-notes/v679-authority-snapshots/roadmap.yaml")
METHOD = Path("docs/research-notes/v679-method-map.md")
RAW = Path("results/raw/experiment_7809_v679_contract_methods")
MANIFEST = RAW / "validation_command_manifest.json"
RESULT = Path("results/experiment_7809_v679_contract_methods.json")
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
    """Select matching staged or active YAML, never a later milestone."""
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
    raise ValueError("matching V679 authority missing")


def parse_design(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read human-visible cells separately from embedded machine values."""
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
    block = re.search(r"<!-- V679_TASK_CONTRACT_START -->\s*```json\s*(.*?)\s*```", section, re.S)
    if block is None:
        raise ValueError("V679 JSON contract missing")
    machine = json.loads(block.group(1))
    if machine.get("milestone") != MILESTONE:
        raise ValueError("V679 JSON milestone mismatch")
    return table, machine["tasks"]


def compare_contract(text: str, roadmap: dict[str, Any]) -> dict[str, Any]:
    """Keep each operand visible so malformed contracts fail specifically."""
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
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{7809 + index}-")
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
                    "design": sha256_file(ROOT / DESIGN_SNAPSHOT),
                    "roadmap": sha256_file(ROOT / YAML_SNAPSHOT),
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
    """Change a private copy; live authority bytes remain untouched."""
    value = deepcopy(roadmap)
    tasks = value["tasks"]
    if name == "drop":
        tasks.pop()
    elif name == "reorder":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif name in {"title", "phase", "deliverable", "model", "substrate"}:
        index = 6 if name in {"model", "substrate"} else 0
        field = {
            "title": "title",
            "phase": "phase",
            "deliverable": "deliverable",
            "model": "MODEL_SPECS",
            "substrate": "inference_substrate_class",
        }[name]
        tasks[index][field] = {
            "title": "Wrong title",
            "phase": 4,
            "deliverable": "results/wrong.json",
            "model": ["wrong/model"],
            "substrate": "no_model_load",
        }[name]
    elif name in {"unknown_producer", "gate_field"}:
        gate = next(task for task in tasks if task.get("gated_on"))["gated_on"][0]
        gate["upstream" if name == "unknown_producer" else "artifact_field"] = "unknown"
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
    """Read V678's declared outputs, not queue-level pre-gate receipts."""
    archived = yaml.safe_load(
        (root / "docs/research-notes/v678-authority-snapshots/roadmap.yaml").read_text()
    )
    tasks = archived["tasks"]
    if [task["id"].split("-")[0] for task in tasks] != [f"exp{i}" for i in range(7795, 7809)]:
        raise ValueError("preserved V678 task order changed")
    rows = []
    for number, task in enumerate(tasks, 7795):
        declared = Path(task["deliverable"])
        producer = root / declared
        evidence = json.loads(producer.read_text()) if producer.is_file() else {}
        state = evidence.get("verdict_class", "unclassified") if producer.is_file() else "missing"
        receipts = []
        if state == "missing":
            for candidate in sorted((root / "results").glob(f"experiment_{number}_*.json")):
                receipt = json.loads(candidate.read_text())
                if (
                    receipt.get("schema") == "blocked_gate_check_v1"
                    and receipt.get("experiment") == number
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
            }
        )
    return rows


def cold_validate(candidate: Path, raw: Path, root: Path) -> bool:
    """Replay exact frozen task rows after the active roadmap has moved."""
    value = json.loads(candidate.read_text())
    expected = json.loads(raw.read_text())
    design = root / DESIGN_SNAPSHOT
    authority = root / YAML_SNAPSHOT
    if not design.is_file() or not authority.is_file():
        return False
    replay = compare_contract(design.read_text(), yaml.safe_load(authority.read_text()))
    return replay["passed"] and replay["rows"] == expected == value.get("rows")
