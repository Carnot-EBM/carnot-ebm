"""Check staged and activated V685 authority from separate source bytes.

REQ-REPORT-7891-V685. The digest includes prompts and failure history because
those fields can change what a task does even when its short title stays fixed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.roadmap_contract import FIELDS, parse_design

MILESTONE = "2026.09.685"
COUNT = 12


def tasks_digest(tasks: Any) -> str:
    """Hash the full executable task list with the design's JSON encoding."""
    data = json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _read(path: Path) -> tuple[bytes | None, dict[str, Any] | None]:
    """Read each actual path once so one role never borrows another role's bytes."""
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    return raw, value if isinstance(value, dict) else None


def _snapshot(path: Path, raw: bytes | None, directory: Path, role: str) -> dict[str, Any]:
    """Store each observed version once; later authority changes get new names."""
    if raw is None:
        return {"role": role, "source_path": str(path), "exists": False, "sha256": None}
    digest = hashlib.sha256(raw).hexdigest()
    target = directory / f"{role}-{digest}.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != raw:
        raise ValueError("immutable authority snapshot differs")
    if not target.exists():
        target.write_bytes(raw)
    return {
        "role": role,
        "source_path": str(path),
        "exists": True,
        "sha256": f"sha256:{digest}",
        "snapshot_path": str(target),
        "snapshot_sha256": sha256_file(target),
    }


def _failure(
    path: Path, digest: str | None, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Name the exact failed operand so absence differs from a wrong value."""
    return {
        "upstream_id": "V685_authority",
        "artifact_path": str(path),
        "artifact_hash": digest,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def assess_authorities(
    design: Path,
    staged: Path,
    active: Path,
    snapshot_dir: Path,
    *,
    milestone: str = MILESTONE,
    first_id: int = 7891,
    count: int = COUNT,
) -> dict[str, Any]:
    """Accept activation only when active tasks match the complete design digest."""
    design_raw = design.read_bytes()
    design_text = design_raw.decode("utf-8")
    table, machine = parse_design(design_text, milestone=milestone)
    match = re.search(
        r"Canonical (?:full-task|complete-task|task) SHA-?256: `([0-9a-f]{64})`", design_text
    )
    if match is None:
        raise ValueError(f"V{milestone.rsplit('.', 1)[-1]} design digest missing")
    expected_digest = match.group(1)
    staged_raw, staged_value = _read(staged)
    active_raw, active_value = _read(active)
    snapshots = {
        "design": _snapshot(design, design_raw, snapshot_dir, "design"),
        "staged": _snapshot(staged, staged_raw, snapshot_dir, "staged"),
        "active": _snapshot(active, active_raw, snapshot_dir, "active"),
    }
    failures: list[dict[str, Any]] = []
    active_tasks = active_value.get("tasks", []) if active_value else []
    staged_tasks = staged_value.get("tasks", []) if staged_value else []
    design_ok = len(table) == len(machine) == count
    if not design_ok:
        failures.append(
            _failure(
                design,
                snapshots["design"]["sha256"],
                "task_count",
                count,
                [len(table), len(machine)],
            )
        )
    active_milestone = active_value.get("milestone") if active_value else None
    active_digest = tasks_digest(active_tasks) if isinstance(active_tasks, list) else None
    if active_milestone != milestone:
        failures.append(
            _failure(
                active, snapshots["active"]["sha256"], "milestone", milestone, active_milestone
            )
        )
    if active_digest != expected_digest:
        failures.append(
            _failure(
                active,
                snapshots["active"]["sha256"],
                "canonical_tasks_sha256",
                expected_digest,
                active_digest,
            )
        )
    if len(active_tasks) != count:
        failures.append(
            _failure(active, snapshots["active"]["sha256"], "task_count", count, len(active_tasks))
        )
    staged_current = staged_value is not None and staged_value.get("milestone") == milestone
    staged_digest = tasks_digest(staged_tasks) if staged_current else None
    if staged_current and staged_digest != expected_digest:
        failures.append(
            _failure(
                staged,
                snapshots["staged"]["sha256"],
                "canonical_tasks_sha256",
                expected_digest,
                staged_digest,
            )
        )
    rows: list[dict[str, Any]] = []
    for index in range(count):
        expected = machine[index] if index < len(machine) else {}
        shown = table[index] if index < len(table) else {}
        actual = (
            active_tasks[index]
            if index < len(active_tasks) and isinstance(active_tasks[index], dict)
            else {}
        )
        checks = {field: expected.get(field) == actual.get(field) for field in FIELDS}
        checks.update(
            {
                f"table_{field}": shown.get(field) == expected.get(field)
                for field in ("id", "title", "phase", "deliverable")
            }
        )
        checks["order"] = shown.get("order") == index + 1
        checks["sequence"] = str(actual.get("id", "")).startswith(f"exp{first_id + index}-")
        checks["milestone"] = actual.get("milestone") == milestone
        checks["prior"] = bool(actual.get("prior_failures")) and all(
            isinstance(item, dict)
            and all(
                item.get(key) not in (None, "")
                for key in ("experiment_id", "verdict", "addressed_by")
            )
            and item.get("retire_if_same_verdict") is True
            for item in actual.get("prior_failures", [])
        )
        checks["gate_keys"] = isinstance(actual.get("gated_on"), list) and all(
            isinstance(gate, dict) and set(gate) == {"upstream", "artifact_field", "op", "value"}
            for gate in actual.get("gated_on", [])
        )
        checks["gate_order"] = isinstance(actual.get("gated_on"), list) and all(
            gate.get("upstream")
            in {task.get("id") for task in active_tasks[:index] if isinstance(task, dict)}
            for gate in actual.get("gated_on", [])
            if isinstance(gate, dict)
        )
        matched = all(checks.values())
        rows.append(
            {
                "family": expected.get("id"),
                "unit_id": expected.get("id"),
                "arm": "active_contract",
                "seed": None,
                "order": index + 1,
                "status": "completed" if actual else "unstarted",
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
        failures.append(
            _failure(
                active,
                snapshots["active"]["sha256"],
                "contract_rows.matched",
                True,
                [row["order"] for row in rows if not row["matched"]],
            )
        )
    return {
        "activated": not failures,
        "planning_matched": bool(staged_current and staged_digest == expected_digest and design_ok),
        "canonical_tasks_sha256": expected_digest,
        "active_tasks_sha256": active_digest,
        "staged_tasks_sha256": staged_digest,
        "contract_rows": rows,
        "authority_snapshots": snapshots,
        "gate_check_summary": failures,
    }


def cold_replay(candidate: Path, raw_rows: Path) -> bool:
    """Reject a reported row whose primitive checks or raw copy disagree."""
    value = json.loads(candidate.read_text())
    raw = json.loads(raw_rows.read_text())
    rows = raw["rows"] if isinstance(raw, dict) else raw
    return (
        value.get("experiment_id") == 7891
        and value.get("task_id") == "exp7891-authority-lifecycle"
        and value.get("rows") == rows
        and len(rows) == COUNT
        and all(row["absolute_metric"] == int(all(row["checks"].values())) for row in rows)
    )
