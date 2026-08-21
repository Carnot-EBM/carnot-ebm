"""Reusable monotonic phase and concurrency receipt helpers.

Spec refs: REQ-INFRA-6481, SCENARIO-INFRA-6481-MONOTONIC-PHASES,
SCENARIO-INFRA-6481-DEPENDENCY-BINDING,
SCENARIO-INFRA-6481-RESOURCE-OWNERSHIP,
SCENARIO-INFRA-6481-CONCURRENCY-OVERLAP,
SCENARIO-INFRA-6481-FAIL-CLOSED-VALIDATION.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

SCHEMA_VERSION = "carnot.phase_concurrency_receipt.v1"
REQUIRED_PHASES = (
    "queue_wait",
    "dependency_resolution",
    "resource_acquisition",
    "model_or_fixture_load",
    "execution",
    "exact_verification",
    "artifact_write",
    "resource_release",
)
PHASE_ORDER = {phase: index for index, phase in enumerate(REQUIRED_PHASES)}
ATTACK_IDS = (
    "borrowed_nvidia_smi_activity",
    "stale_dependency_artifact",
    "duplicated_attempt_id",
    "forged_clocks",
    "cross_task_output_path",
    "parent_child_pid_confusion",
    "pid_reuse",
    "output_before_execution",
    "copied_receipt",
)


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _sha_prefixed(value: Any) -> bool:
    text = str(value)
    return len(text) == 71 and text.startswith("sha256:")


def _int_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_wall_clock(value: Any) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _row_hash_payload(row: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in row.items() if key != "row_hash"}


def refresh_row_hash(row: JsonDict) -> JsonDict:
    """Refresh the self-hash after a deliberate fixture mutation."""

    row.pop("row_hash", None)
    row["row_hash"] = receipts.sha256_json(row)
    return row


def receipt_schema_and_hash() -> JsonDict:
    """Return the versioned row schema and its stable hash."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "required_phases": list(REQUIRED_PHASES),
        "row_types": [
            "process_identity",
            "phase",
            "dependency",
            "resource_interval",
            "output",
            "concurrency_decision",
            "attack",
        ],
        "process_identity_fields": [
            "pid",
            "process_start_identity",
            "parent_pid",
            "parent_process_start_identity",
        ],
        "clock_sources": ["time.monotonic_ns", "datetime.now(UTC)"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": receipts.sha256_json(payload),
        "payload": payload,
    }


def _proc_start_identity(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        parts = stat.read_text(encoding="utf-8", errors="replace").split()
        if len(parts) > 21:
            return f"linux_proc_stat_starttime:{parts[21]}"
    return f"pid_start_unavailable:{pid}"


def current_process_identity() -> JsonDict:
    """Return the current process identity used by live receipts."""

    pid = os.getpid()
    parent_pid = os.getppid()
    command = [sys.executable, "-m", "carnot.phase_concurrency_receipts"]
    return {
        "pid": pid,
        "process_start_identity": _proc_start_identity(pid),
        "parent_pid": parent_pid,
        "parent_process_start_identity": _proc_start_identity(parent_pid),
        "executable_path": sys.executable,
        "executable_sha256": receipts.sha256_file(sys.executable)
        or receipts.sha256_text(sys.executable),
        "command": command,
        "command_hash": receipts.sha256_json(command),
    }


def _base_row(
    *,
    row_type: str,
    task_id: str,
    attempt_id: str,
    process: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION,
        "row_type": row_type,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "pid": int(process["pid"]),
        "process_start_identity": str(process["process_start_identity"]),
        "parent_pid": int(process["parent_pid"]),
        "parent_process_start_identity": str(process["parent_process_start_identity"]),
    }


def build_process_identity_row(
    *,
    task_id: str,
    attempt_id: str,
    process: Mapping[str, Any],
) -> JsonDict:
    """Build the attempt-owned process identity row."""

    row = _base_row(
        row_type="process_identity",
        task_id=task_id,
        attempt_id=attempt_id,
        process=process,
    )
    row.update(
        {
            "executable_path": process.get("executable_path", ""),
            "executable_sha256": process.get("executable_sha256", ""),
            "command": list(process.get("command", [])),
            "command_hash": process.get("command_hash", ""),
        }
    )
    return refresh_row_hash(row)


def build_phase_row(
    *,
    task_id: str,
    attempt_id: str,
    phase: str,
    process: Mapping[str, Any],
    monotonic_start_ns: int,
    monotonic_end_ns: int,
    wall_clock_start: str,
    wall_clock_end: str,
    exit_state: Mapping[str, Any],
) -> JsonDict:
    """Build one ordered phase interval row."""

    row = _base_row(row_type="phase", task_id=task_id, attempt_id=attempt_id, process=process)
    row.update(
        {
            "phase": phase,
            "phase_index": PHASE_ORDER.get(phase, -1),
            "monotonic_start_ns": int(monotonic_start_ns),
            "monotonic_end_ns": int(monotonic_end_ns),
            "wall_clock_start": wall_clock_start,
            "wall_clock_end": wall_clock_end,
            "exit_state": dict(exit_state),
        }
    )
    return refresh_row_hash(row)


def build_dependency_row(
    *,
    task_id: str,
    attempt_id: str,
    process: Mapping[str, Any],
    path: str | Path,
    phase: str = "dependency_resolution",
) -> JsonDict:
    """Build one dependency path and SHA-256 binding row."""

    digest = receipts.sha256_file(path)
    row = _base_row(
        row_type="dependency",
        task_id=task_id,
        attempt_id=attempt_id,
        process=process,
    )
    row.update(
        {
            "phase": phase,
            "path": str(path),
            "sha256": digest,
            "hash_algorithm": "sha256",
        }
    )
    return refresh_row_hash(row)


def build_resource_interval_row(
    *,
    task_id: str,
    attempt_id: str,
    process: Mapping[str, Any],
    resource_key: str,
    resource_type: str,
    exclusive: bool,
    monotonic_start_ns: int,
    monotonic_end_ns: int,
    acquired_in_phase: str,
    released_in_phase: str,
    release_present: bool = True,
    activity_sample: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one resource ownership interval."""

    row = _base_row(
        row_type="resource_interval",
        task_id=task_id,
        attempt_id=attempt_id,
        process=process,
    )
    row.update(
        {
            "resource_key": resource_key,
            "resource_type": resource_type,
            "exclusive": bool(exclusive),
            "owner_pid": int(process["pid"]),
            "owner_process_start_identity": str(process["process_start_identity"]),
            "monotonic_start_ns": int(monotonic_start_ns),
            "monotonic_end_ns": int(monotonic_end_ns),
            "acquired_in_phase": acquired_in_phase,
            "released_in_phase": released_in_phase,
            "release_present": bool(release_present),
            "activity_sample": dict(activity_sample or {}),
        }
    )
    return refresh_row_hash(row)


def build_output_row(
    *,
    task_id: str,
    attempt_id: str,
    process: Mapping[str, Any],
    path: str | Path,
    output_bytes: bytes,
    write_monotonic_ns: int,
    phase: str = "artifact_write",
) -> JsonDict:
    """Build one output path and hash binding row."""

    row = _base_row(row_type="output", task_id=task_id, attempt_id=attempt_id, process=process)
    row.update(
        {
            "phase": phase,
            "path": str(path),
            "sha256": receipts.sha256_bytes(output_bytes),
            "output_byte_length": len(output_bytes),
            "write_monotonic_ns": int(write_monotonic_ns),
        }
    )
    return refresh_row_hash(row)


def _interval(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
    return _int_value(row.get("monotonic_start_ns")), _int_value(row.get("monotonic_end_ns"))


def _attempt_key(row: Mapping[str, Any]) -> str:
    return str(row.get("attempt_id"))


def _same_resource(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return str(left.get("resource_key")) == str(right.get("resource_key"))


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_start, left_end = _interval(left)
    right_start, right_end = _interval(right)
    if None in (left_start, left_end, right_start, right_end):
        return False
    return bool(left_start < right_end and right_start < left_end)


def build_concurrency_decision_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Classify safe shared overlap and serialized exclusive ownership."""

    resources = [dict(row) for row in rows if row.get("row_type") == "resource_interval"]
    decisions: list[JsonDict] = []
    for index, left in enumerate(resources):
        for right in resources[index + 1 :]:
            if not _same_resource(left, right) or left.get("attempt_id") == right.get("attempt_id"):
                continue
            overlaps = _overlap(left, right)
            exclusive = bool(left.get("exclusive")) or bool(right.get("exclusive"))
            if overlaps and not exclusive:
                decision = "safe_overlap"
            elif not overlaps and exclusive:
                decision = "serialized_exclusive"
            elif overlaps and exclusive:
                decision = "conflict"
            else:
                decision = "independent_serial"
            row = {
                "schema_version": SCHEMA_VERSION,
                "row_type": "concurrency_decision",
                "resource_key": left["resource_key"],
                "left_task_id": left["task_id"],
                "left_attempt_id": left["attempt_id"],
                "right_task_id": right["task_id"],
                "right_attempt_id": right["attempt_id"],
                "overlap": overlaps,
                "exclusive": exclusive,
                "decision": decision,
            }
            decisions.append(refresh_row_hash(row))
    return decisions


def _validate_row_hash(row: Mapping[str, Any], reasons: list[str]) -> None:
    expected = receipts.sha256_json(_row_hash_payload(row))
    if row.get("row_hash") != expected:
        _append_once(reasons, "row_hash_mismatch")


def _validate_attempts(
    rows: Sequence[Mapping[str, Any]],
    expected_attempts: Mapping[str, str] | None,
    reasons: list[str],
) -> None:
    tasks_by_attempt: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("row_type") == "concurrency_decision":
            continue
        attempt_id = _attempt_key(row)
        task_id = str(row.get("task_id"))
        tasks_by_attempt[attempt_id].add(task_id)
        if expected_attempts is not None:
            expected_task = expected_attempts.get(attempt_id)
            if expected_task is None or expected_task != task_id:
                _append_once(reasons, "copied_receipt_task_mismatch")
    for task_ids in tasks_by_attempt.values():
        if len(task_ids) > 1:
            _append_once(reasons, "duplicated_attempt_id")


def _process_rows_by_attempt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        _attempt_key(row): row
        for row in rows
        if row.get("row_type") == "process_identity"
    }


def _validate_process_identity(
    rows: Sequence[Mapping[str, Any]],
    processes: Mapping[str, Mapping[str, Any]],
    reasons: list[str],
) -> None:
    identities_by_pid: dict[int, set[str]] = defaultdict(set)
    for process in processes.values():
        pid = _int_value(process.get("pid"))
        start_id = str(process.get("process_start_identity"))
        if pid is not None:
            identities_by_pid[pid].add(start_id)
        if process.get("pid") == process.get("parent_pid"):
            _append_once(reasons, "parent_child_pid_confusion")
    for identities in identities_by_pid.values():
        if len(identities) > 1:
            _append_once(reasons, "pid_reuse")

    for row in rows:
        if row.get("row_type") in {"concurrency_decision", "attack"}:
            continue
        process = processes.get(_attempt_key(row))
        if process is None:
            _append_once(reasons, "process_identity_missing")
            continue
        if row.get("pid") == process.get("parent_pid"):
            _append_once(reasons, "parent_child_pid_confusion")
        if (
            row.get("pid") != process.get("pid")
            or row.get("process_start_identity") != process.get("process_start_identity")
        ):
            _append_once(reasons, "process_identity_mismatch")


def _validate_phases(
    phase_rows: Sequence[Mapping[str, Any]],
    expected_attempts: Mapping[str, str] | None,
    reasons: list[str],
) -> dict[str, Mapping[str, Any]]:
    by_attempt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    execution_by_attempt: dict[str, Mapping[str, Any]] = {}
    for row in phase_rows:
        start, end = _interval(row)
        if start is None or end is None:
            _append_once(reasons, "missing_monotonic_interval")
            continue
        if end < start:
            _append_once(reasons, "negative_interval")
        wall_start = _parse_wall_clock(row.get("wall_clock_start"))
        wall_end = _parse_wall_clock(row.get("wall_clock_end"))
        if wall_start is None or wall_end is None:
            _append_once(reasons, "wall_clock_interval_missing")
        elif wall_end < wall_start:
            _append_once(reasons, "wall_clock_inversion")
        by_attempt[_attempt_key(row)].append(row)
        if row.get("phase") == "execution":
            execution_by_attempt[_attempt_key(row)] = row

    expected_ids = expected_attempts.keys() if expected_attempts is not None else by_attempt.keys()
    for attempt_id in expected_ids:
        observed = {str(row.get("phase")) for row in by_attempt.get(attempt_id, [])}
        for phase in REQUIRED_PHASES:
            if phase not in observed:
                _append_once(reasons, f"missing_phase:{attempt_id}:{phase}")
        ordered = sorted(
            by_attempt.get(attempt_id, []),
            key=lambda row: PHASE_ORDER.get(str(row.get("phase")), 99),
        )
        previous_end: int | None = None
        for row in ordered:
            start, end = _interval(row)
            assert start is not None and end is not None
            if previous_end is not None and start < previous_end:
                _append_once(reasons, "phase_inversion")
            previous_end = max(previous_end or end, end)
    return execution_by_attempt


def _validate_dependencies(
    dependency_rows: Sequence[Mapping[str, Any]],
    verify_dependency_files: bool,
    reasons: list[str],
) -> None:
    hashes_by_path: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in dependency_rows:
        path = str(row.get("path"))
        digest = str(row.get("sha256"))
        if not _sha_prefixed(digest):
            _append_once(reasons, "dependency_hash_missing")
        hashes_by_path[(_attempt_key(row), path)].add(digest)
        if verify_dependency_files:
            actual = receipts.sha256_file(path)
            if actual != digest:
                _append_once(reasons, "dependency_hash_changed")
    for hashes in hashes_by_path.values():
        if len(hashes) > 1:
            _append_once(reasons, "dependency_hash_changed")


def _validate_resources(
    resource_rows: Sequence[Mapping[str, Any]],
    processes: Mapping[str, Mapping[str, Any]],
    reasons: list[str],
) -> None:
    by_resource: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in resource_rows:
        start, end = _interval(row)
        if start is None or end is None:
            _append_once(reasons, "missing_resource_interval")
        elif end < start:
            _append_once(reasons, "negative_interval")
        if row.get("release_present") is not True or row.get("released_in_phase") != "resource_release":
            _append_once(reasons, "missing_release")
        process = processes.get(_attempt_key(row), {})
        if (
            row.get("owner_pid") != process.get("pid")
            or row.get("owner_process_start_identity") != process.get("process_start_identity")
        ):
            _append_once(reasons, "process_identity_mismatch")
        if row.get("owner_pid") == process.get("parent_pid"):
            _append_once(reasons, "parent_child_pid_confusion")
        sample = row.get("activity_sample")
        if isinstance(sample, Mapping) and sample:
            if (
                sample.get("pid") != process.get("pid")
                or sample.get("process_start_identity") != process.get("process_start_identity")
            ):
                _append_once(reasons, "borrowed_global_activity")
        by_resource[str(row.get("resource_key"))].append(row)

    for grouped in by_resource.values():
        ordered = sorted(grouped, key=lambda row: (_interval(row)[0] or 0, _interval(row)[1] or 0))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if left.get("attempt_id") == right.get("attempt_id"):
                    continue
                if (left.get("exclusive") is True or right.get("exclusive") is True) and _overlap(
                    left, right
                ):
                    _append_once(reasons, "overlapping_exclusive_resource_claim")


def _validate_outputs(
    output_rows: Sequence[Mapping[str, Any]],
    execution_by_attempt: Mapping[str, Mapping[str, Any]],
    reasons: list[str],
) -> None:
    tasks_by_path: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        path = str(row.get("path"))
        tasks_by_path[path].add(str(row.get("task_id")))
        if not _sha_prefixed(row.get("sha256")):
            _append_once(reasons, "output_hash_missing")
        execution = execution_by_attempt.get(_attempt_key(row))
        execution_start = _int_value(execution.get("monotonic_start_ns")) if execution else None
        write_ns = _int_value(row.get("write_monotonic_ns"))
        if execution_start is None or write_ns is None:
            _append_once(reasons, "output_write_time_missing")
        elif write_ns < execution_start:
            _append_once(reasons, "output_write_before_execution")
    for task_ids in tasks_by_path.values():
        if len(task_ids) > 1:
            _append_once(reasons, "cross_task_output_path")


def validate_receipt_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_attempts: Mapping[str, str] | None = None,
    verify_dependency_files: bool = True,
) -> JsonDict:
    """Validate task-owned phase, dependency, resource, and output rows."""

    reasons: list[str] = []
    for row in rows:
        _validate_row_hash(row, reasons)

    _validate_attempts(rows, expected_attempts, reasons)
    processes = _process_rows_by_attempt(rows)
    _validate_process_identity(rows, processes, reasons)
    phase_rows = [row for row in rows if row.get("row_type") == "phase"]
    dependency_rows = [row for row in rows if row.get("row_type") == "dependency"]
    resource_rows = [row for row in rows if row.get("row_type") == "resource_interval"]
    output_rows = [row for row in rows if row.get("row_type") == "output"]
    execution_by_attempt = _validate_phases(phase_rows, expected_attempts, reasons)
    _validate_dependencies(dependency_rows, verify_dependency_files, reasons)
    _validate_resources(resource_rows, processes, reasons)
    _validate_outputs(output_rows, execution_by_attempt, reasons)
    decisions = build_concurrency_decision_rows(rows)
    phase_count_by_attempt = Counter(_attempt_key(row) for row in phase_rows)
    duration_ns = 0
    for row in phase_rows:
        start, end = _interval(row)
        if start is not None and end is not None:
            duration_ns += max(0, end - start)
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "phase_count_by_attempt": dict(sorted(phase_count_by_attempt.items())),
        "dependency_row_count": len(dependency_rows),
        "resource_interval_count": len(resource_rows),
        "output_row_count": len(output_rows),
        "concurrency_decision_count": len(decisions),
        "safe_overlap_count": sum(1 for row in decisions if row["decision"] == "safe_overlap"),
        "serialized_exclusive_count": sum(
            1 for row in decisions if row["decision"] == "serialized_exclusive"
        ),
        "recomputed_duration_s": round(duration_ns / 1_000_000_000, 9),
    }


def mutate_rows_for_attack(attack_id: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return rows mutated for one critical attribution attack."""

    mutated: list[JsonDict] = receipts.json.loads(receipts.canonical_json(list(rows)))
    if attack_id == "borrowed_nvidia_smi_activity":
        row = next(item for item in mutated if item.get("activity_sample"))
        row["activity_sample"]["pid"] = 999_999
        row["activity_sample"]["process_start_identity"] = "foreign"
        refresh_row_hash(row)
    elif attack_id == "stale_dependency_artifact":
        row = next(item for item in mutated if item.get("row_type") == "dependency")
        row["sha256"] = "sha256:" + "0" * 64
        refresh_row_hash(row)
    elif attack_id == "duplicated_attempt_id":
        row = next(item for item in mutated if item.get("task_id") != mutated[0].get("task_id"))
        row["attempt_id"] = str(mutated[0]["attempt_id"])
        refresh_row_hash(row)
    elif attack_id == "forged_clocks":
        row = next(item for item in mutated if item.get("row_type") == "phase")
        row["monotonic_end_ns"] = row["monotonic_start_ns"] - 1
        refresh_row_hash(row)
    elif attack_id == "cross_task_output_path":
        outputs = [item for item in mutated if item.get("row_type") == "output"]
        outputs[1]["path"] = outputs[0]["path"]
        refresh_row_hash(outputs[1])
    elif attack_id == "parent_child_pid_confusion":
        row = next(item for item in mutated if item.get("row_type") == "resource_interval")
        row["pid"] = row["parent_pid"]
        row["owner_pid"] = row["parent_pid"]
        refresh_row_hash(row)
    elif attack_id == "pid_reuse":
        processes = [item for item in mutated if item.get("row_type") == "process_identity"]
        processes[1]["pid"] = processes[0]["pid"]
        processes[1]["process_start_identity"] = "reused-start-identity"
        refresh_row_hash(processes[1])
    elif attack_id == "output_before_execution":
        output = next(item for item in mutated if item.get("row_type") == "output")
        output["write_monotonic_ns"] = 1
        refresh_row_hash(output)
    elif attack_id == "copied_receipt":
        row = next(item for item in mutated if item.get("row_type") == "dependency")
        row["task_id"] = "copied-from-other-task"
    else:
        raise ValueError(f"unknown attack_id: {attack_id}")
    return mutated


def mutation_attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_attempts: Mapping[str, str],
    verify_dependency_files: bool = True,
) -> JsonDict:
    """Run critical mutations and confirm each one fails closed."""

    matrix_rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        mutated = mutate_rows_for_attack(attack_id, rows)
        report = validate_receipt_rows(
            mutated,
            expected_attempts=expected_attempts,
            verify_dependency_files=verify_dependency_files,
        )
        matrix_rows.append(
            {
                "row_type": "attack",
                "attack_id": attack_id,
                "accepted": report["accepted"],
                "fail_closed": not report["accepted"],
                "reasons": report["reasons"],
            }
        )
    for row in matrix_rows:
        refresh_row_hash(row)
    false_accept_count = sum(1 for row in matrix_rows if row["accepted"])
    return {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": matrix_rows,
        "attack_count": len(matrix_rows),
        "false_accept_count": false_accept_count,
        "all_critical_fail_closed": false_accept_count == 0,
    }
