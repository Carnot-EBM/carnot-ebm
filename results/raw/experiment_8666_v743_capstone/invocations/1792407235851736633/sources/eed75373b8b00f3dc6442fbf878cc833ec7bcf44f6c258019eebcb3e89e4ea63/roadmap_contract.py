"""Bind a visible roadmap table to its JSON and YAML authorities.

REQ-REPORT-7837. The result is an administrative receipt, not a decision
quality measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file

MILESTONE = "2026.09.681"
FIELDS = (
    "id",
    "title",
    "phase",
    "deliverable",
    "MODEL_SPECS",
    "inference_substrate_class",
    "gated_on",
)
TABLE_FIELDS = ("id", "title", "phase", "deliverable")
PRIOR_FIELDS = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")


def parse_design(
    text: str, *, milestone: str = MILESTONE
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the visible table separately from the embedded machine contract."""
    section = text.split("## Exact task contract", 1)[1]
    table: list[dict[str, Any]] = []
    for line in section.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            table.append(
                dict(
                    zip(
                        ("order", *TABLE_FIELDS),
                        (int(cells[0]), cells[1], cells[2], int(cells[3]), cells[4]),
                    )
                )
            )
    version = milestone.rsplit(".", 1)[-1]
    block = re.search(
        rf"<!-- V{re.escape(version)}_TASK_CONTRACT_START -->\s*```json\s*(.*?)\s*```",
        section,
        re.S,
    )
    if block is None:
        raise ValueError(f"V{version} JSON contract missing")
    machine = json.loads(block.group(1))
    if machine.get("milestone") != milestone:
        raise ValueError(f"V{version} JSON milestone mismatch")
    return table, machine["tasks"]


def compare_contract(
    text: str,
    staged: dict[str, Any],
    active: dict[str, Any],
    *,
    milestone: str = MILESTONE,
    first_id: int = 7837,
    count: int = 14,
) -> dict[str, Any]:
    """Return one falsifiable row per task for an explicit milestone."""
    table, machine = parse_design(text, milestone=milestone)
    staged_tasks, active_tasks = staged["tasks"], active["tasks"]
    errors = []
    if any(value.get("milestone") != milestone for value in (staged, active)):
        errors.append("roadmap_milestone")
    if tuple(map(len, (table, machine, staged_tasks, active_tasks))) != (count,) * 4:
        errors.append("task_count")
    rows = []
    for index in range(count):
        shown = table[index] if index < len(table) else {}
        expected = machine[index] if index < len(machine) else {}
        planned = staged_tasks[index] if index < len(staged_tasks) else {}
        actual = active_tasks[index] if index < len(active_tasks) else {}
        checks = {
            field: field in expected
            and planned.get(field, [] if field == "gated_on" else None) == expected[field]
            and actual.get(field, [] if field == "gated_on" else None) == expected[field]
            for field in FIELDS
        }
        checks.update(
            {f"table_{field}": shown.get(field) == expected.get(field) for field in TABLE_FIELDS}
        )
        checks["order"] = shown.get("order") == index + 1
        checks["sequence"] = str(planned.get("id", "")).startswith(f"exp{first_id + index}-")
        checks["milestone"] = all(task.get("milestone") == milestone for task in (planned, actual))
        checks["prior"] = all(
            bool(task.get("prior_failures"))
            and all(
                all(item.get(field) not in (None, "") for field in PRIOR_FIELDS)
                and item["retire_if_same_verdict"] is True
                for item in task["prior_failures"]
            )
            for task in (planned, actual)
        )
        checks["gate_keys"] = all(
            set(gate) == {"upstream", "artifact_field", "op", "value"}
            for gate in planned.get("gated_on", [])
        )
        checks["gate_order"] = all(
            gate.get("upstream") in {task.get("id") for task in staged_tasks[:index]}
            for gate in planned.get("gated_on", [])
        )
        matched = all(checks.values())
        rows.append(
            {
                "unit_id": planned.get("id", f"missing-{index + 1}"),
                "order": index + 1,
                "arm": "four_source_contract",
                "family": planned.get("id"),
                "seed": None,
                "status": "completed" if planned else "unstarted",
                "matched": matched,
                "checks": checks,
                "absolute_metric": int(matched),
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "censored": False,
                "excluded": not matched,
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
        "staged_count": len(staged_tasks),
        "active_count": len(active_tasks),
    }


def snapshot_authorities(sources: tuple[Path, ...], directory: Path) -> list[dict[str, str]]:
    """Write immutable source bytes and reject a later attempt to change them."""
    directory.mkdir(parents=True, exist_ok=True)
    labels = ("design.md", "staged.yaml", "active.yaml")
    if len(sources) != len(labels):
        raise ValueError("three authorities required")
    result = []
    for source, label in zip(sources, labels):
        target = directory / label
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"immutable snapshot differs: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
        result.append(
            {
                "path": str(target),
                "sha256": sha256_file(target),
                "source_path": str(source),
                "source_sha256": sha256_file(source),
            }
        )
    return result


def verify_snapshots(snapshots: list[dict[str, str]]) -> bool:
    """Cold-check both the saved byte copies and their named sources."""
    return bool(snapshots) and all(
        Path(item["path"]).is_file()
        and Path(item["source_path"]).is_file()
        and sha256_file(Path(item["path"])) == item["sha256"]
        and sha256_file(Path(item["source_path"])) == item["source_sha256"]
        for item in snapshots
    )


def cold_replay(candidate: Path, raw_rows: Path, *, experiment_id: int = 7837) -> bool:
    """Reconstruct row identity without trusting a reported aggregate."""
    value = json.loads(candidate.read_text())
    raw = json.loads(raw_rows.read_text())
    rows = raw["rows"] if isinstance(raw, dict) else raw
    return (
        type(value.get("experiment_id")) is int
        and value["experiment_id"] == experiment_id
        and value.get("task_id") == f"exp{experiment_id}-contract-methods"
        and value.get("rows") == rows
        and len(rows) == 14
        and all(row["absolute_metric"] == int(all(row["checks"].values())) for row in rows)
    )
