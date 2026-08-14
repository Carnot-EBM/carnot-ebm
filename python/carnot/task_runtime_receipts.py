"""Reusable task-scoped runtime receipt helpers.

Spec refs: REQ-INFRA-6426, SCENARIO-INFRA-6426-1,
SCENARIO-INFRA-6426-4, SCENARIO-INFRA-6426-5.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


JsonDict = dict[str, Any]

SCHEMA_VERSION = "carnot.task_scoped_runtime_receipt.v1"
REQUIRED_PHASES = (
    "queue_wait",
    "model_load",
    "generation",
    "exact_verification",
    "artifact_write",
)
REQUIRED_ROW_FIELDS = (
    "schema_version",
    "task_id",
    "control_id",
    "phase",
    "monotonic_start_ns",
    "monotonic_end_ns",
    "wall_clock_start",
    "wall_clock_end",
    "parent_pid",
    "child_pids",
    "command_hash",
    "config_hash",
    "model_hash",
    "runner_selection",
    "device_ids",
    "concurrency_group",
    "raw_output_hash",
    "exit_status",
    "attribution_confidence",
)
ATTACK_IDS = (
    "forged_pid",
    "stale_nvidia_sample",
    "model_name_only_substitution",
    "raw_output_reuse",
    "runner_swap",
    "clock_rollback",
    "truncated_receipt",
    "concurrency_collision",
    "cpu_fallback",
    "child_exit_omission",
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for receipt hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest with the project prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Stream a file hash, or return None when the file is absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=target.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(target)
    return target


def _runner_selection_hash(selection: Mapping[str, Any]) -> str:
    """Hash runner selection while excluding its stored self-hash."""

    payload = {key: value for key, value in selection.items() if key != "selection_hash"}
    return sha256_json(payload)


def build_phase_row(
    *,
    task_id: str,
    control_id: str,
    phase: str,
    monotonic_start_ns: int,
    monotonic_end_ns: int,
    wall_clock_start: str,
    wall_clock_end: str,
    parent_pid: int,
    child_pids: Sequence[int],
    command: Sequence[str],
    config: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    runner_selection: Mapping[str, Any],
    device_ids: Sequence[str],
    concurrency_group: str,
    raw_output_bytes: bytes,
    exit_status: Mapping[str, Any],
    attribution_confidence: float,
    gpu_samples: Sequence[Mapping[str, Any]] | None = None,
    synthesized_runtime_fields: int = 0,
    cpu_fallback: bool = False,
    blocked_reason: str = "",
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one phase row with all hashes computed from structured inputs."""

    selection = dict(runner_selection)
    selection.setdefault("selection_hash", _runner_selection_hash(selection))
    model_hash = str(model_identity.get("model_sha256") or sha256_json(model_identity))
    row: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "control_id": control_id,
        "phase": phase,
        "monotonic_start_ns": int(monotonic_start_ns),
        "monotonic_end_ns": int(monotonic_end_ns),
        "wall_clock_start": wall_clock_start,
        "wall_clock_end": wall_clock_end,
        "parent_pid": int(parent_pid),
        "child_pids": [int(pid) for pid in child_pids],
        "command": list(command),
        "command_hash": sha256_json(list(command)),
        "config": dict(config),
        "config_hash": sha256_json(dict(config)),
        "model_identity": dict(model_identity),
        "model_hash": model_hash,
        "runner_selection": selection,
        "device_ids": list(device_ids),
        "concurrency_group": concurrency_group,
        "raw_output_hash": sha256_bytes(raw_output_bytes),
        "raw_output_byte_length": len(raw_output_bytes),
        "exit_status": dict(exit_status),
        "attribution_confidence": float(attribution_confidence),
        "gpu_samples": [dict(sample) for sample in (gpu_samples or [])],
        "synthesized_runtime_fields": int(synthesized_runtime_fields),
        "cpu_fallback": bool(cpu_fallback),
        "blocked_reason": blocked_reason,
    }
    if extra:
        row.update(dict(extra))
    return row


class TaskScopedReceiptWriter:
    """Persist rows as they complete so interruption keeps useful evidence."""

    def __init__(self, path: str | Path, *, task_id: str) -> None:
        self.path = Path(path)
        self.task_id = task_id
        self.rows: list[JsonDict] = []

    def record_phase(self, row: Mapping[str, Any]) -> None:
        """Append one row and atomically publish the partial receipt."""

        self.rows.append(dict(row))
        self._write({"status": "partial"})

    def finalize(self, payload: Mapping[str, Any]) -> None:
        """Write the final sidecar payload while preserving recorded rows."""

        self._write(dict(payload))

    def _write(self, payload: Mapping[str, Any]) -> None:
        base = {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "rows": self.rows,
        }
        base.update(dict(payload))
        write_json_atomic(self.path, base)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and other values as an empty map."""

    return value if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int | None:
    """Return an int when conversion is exact enough for receipt checks."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha_prefixed(value: Any) -> bool:
    """Return true for the digest spelling used by Carnot artifacts."""

    text = str(value)
    return len(text) == 71 and text.startswith("sha256:")


def _append_once(reasons: list[str], reason: str) -> None:
    """Keep reason lists stable and readable."""

    if reason not in reasons:
        reasons.append(reason)


def validate_contract_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_controls: Sequence[str],
) -> JsonDict:
    """Validate phase rows and recompute task duration from monotonic clocks."""

    reasons: list[str] = []
    intervals: list[tuple[int, int, Mapping[str, Any]]] = []
    phases_by_control: dict[str, set[str]] = defaultdict(set)
    raw_hash_controls: dict[str, set[str]] = defaultdict(set)
    synthesized_count = 0
    cpu_fallback_count = 0
    attribution_failure_count = 0

    for row in rows:
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            _append_once(reasons, "truncated_receipt")
            if "monotonic_start_ns" in missing or "monotonic_end_ns" in missing:
                _append_once(reasons, "missing_monotonic_interval")
            continue

        control_id = str(row.get("control_id"))
        phase = str(row.get("phase"))
        phases_by_control[control_id].add(phase)
        start = _int_value(row.get("monotonic_start_ns"))
        end = _int_value(row.get("monotonic_end_ns"))
        if start is None or end is None:
            _append_once(reasons, "missing_monotonic_interval")
        else:
            intervals.append((start, end, row))
            if end < start:
                _append_once(reasons, "negative_interval")

        if not str(row.get("wall_clock_start", "")) or not str(row.get("wall_clock_end", "")):
            _append_once(reasons, "wall_clock_interval_missing")
        if _int_value(row.get("parent_pid")) is None or int(row.get("parent_pid", 0)) <= 1:
            _append_once(reasons, "parent_pid_invalid")
        child_pids = [_int_value(pid) for pid in row.get("child_pids", [])]
        if any(pid is None or pid <= 1 for pid in child_pids):
            _append_once(reasons, "forged_pid")
        exit_status = _as_mapping(row.get("exit_status"))
        if child_pids and "returncode" not in exit_status:
            _append_once(reasons, "child_exit_omitted")
        if not _sha_prefixed(row.get("command_hash")):
            _append_once(reasons, "command_hash_missing")
        if not _sha_prefixed(row.get("config_hash")):
            _append_once(reasons, "config_hash_missing")
        if not _sha_prefixed(row.get("raw_output_hash")):
            _append_once(reasons, "raw_output_hash_missing")
        raw_hash_controls[str(row.get("raw_output_hash"))].add(control_id)

        model_identity = _as_mapping(row.get("model_identity"))
        if (
            not _sha_prefixed(row.get("model_hash"))
            or model_identity.get("model_identity_bound") is not True
        ):
            _append_once(reasons, "model_name_only_substitution")
        runner = _as_mapping(row.get("runner_selection"))
        if runner.get("selection_hash") != _runner_selection_hash(runner):
            _append_once(reasons, "runner_selection_hash_mismatch")
        if runner.get("selected") is not True:
            _append_once(reasons, "runner_not_selected")
        if control_id == "powered" and (
            row.get("cpu_fallback") is True or runner.get("substrate") != "cuda_gguf"
        ):
            cpu_fallback_count += 1
            _append_once(reasons, "cpu_fallback")
        synthesized = int(row.get("synthesized_runtime_fields", 0) or 0)
        synthesized_count += synthesized
        if synthesized:
            _append_once(reasons, "synthesized_runtime_field")
        if float(row.get("attribution_confidence", 0.0) or 0.0) < 0.99:
            attribution_failure_count += 1
            _append_once(reasons, "low_attribution_confidence")

        _validate_gpu_sample_row(row, reasons)

    for control_id in expected_controls:
        for phase in REQUIRED_PHASES:
            if phase not in phases_by_control.get(control_id, set()):
                _append_once(reasons, f"missing_control_phase:{control_id}:{phase}")

    for controls in raw_hash_controls.values():
        if len(controls) > 1:
            _append_once(reasons, "raw_output_reuse")

    _validate_interval_order(intervals, reasons)
    _validate_concurrency(intervals, reasons)

    duration_ns = sum(max(0, end - start) for start, end, _row in intervals)
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "recomputed_duration_s": round(duration_ns / 1_000_000_000, 9),
        "synthesized_runtime_field_count": synthesized_count,
        "cpu_fallback_count": cpu_fallback_count,
        "attribution_failure_count": attribution_failure_count,
        "control_phase_counts": {
            control_id: len(phases_by_control.get(control_id, set()))
            for control_id in expected_controls
        },
    }


def _validate_gpu_sample_row(row: Mapping[str, Any], reasons: list[str]) -> None:
    """Check powered generation has fresh PID-linked GPU telemetry."""

    if row.get("control_id") != "powered" or row.get("phase") != "generation":
        return
    samples = row.get("gpu_samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
        _append_once(reasons, "pid_linked_gpu_sample_missing")
        return
    child_pids = {_int_value(pid) for pid in row.get("child_pids", [])}
    start = _int_value(row.get("monotonic_start_ns"))
    end = _int_value(row.get("monotonic_end_ns"))
    found = False
    stale = False
    for sample in samples:
        sample_map = _as_mapping(sample)
        sample_pid = _int_value(sample_map.get("pid"))
        sample_clock = _int_value(sample_map.get("monotonic_ns"))
        if (
            sample_clock is None
            or start is None
            or end is None
            or sample_clock < start
            or sample_clock > end
        ):
            stale = True
        if float(sample_map.get("sample_age_s", 0.0) or 0.0) > 5.0:
            stale = True
        if sample_pid in child_pids and int(sample_map.get("pid_memory_mb", 0) or 0) > 0:
            found = True
    if stale:
        _append_once(reasons, "stale_nvidia_sample")
    if not found:
        _append_once(reasons, "pid_linked_gpu_sample_missing")


def _validate_interval_order(
    intervals: Sequence[tuple[int, int, Mapping[str, Any]]],
    reasons: list[str],
) -> None:
    """Reject interval overlaps unless a row explicitly explains them."""

    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    previous_end: int | None = None
    for start, end, row in ordered:
        if (
            previous_end is not None
            and start < previous_end
            and row.get("overlap_explained") is not True
        ):
            _append_once(reasons, "overlap_unexplained")
        previous_end = max(previous_end or end, end)


def _validate_concurrency(
    intervals: Sequence[tuple[int, int, Mapping[str, Any]]],
    reasons: list[str],
) -> None:
    """Reject cross-control overlap inside one concurrency group."""

    by_group: dict[str, list[tuple[int, int, Mapping[str, Any]]]] = defaultdict(list)
    for start, end, row in intervals:
        by_group[str(row.get("concurrency_group"))].append((start, end, row))
    for grouped in by_group.values():
        ordered = sorted(grouped, key=lambda item: (item[0], item[1]))
        for left, right in zip(ordered, ordered[1:], strict=False):
            left_start, left_end, left_row = left
            right_start, _right_end, right_row = right
            if (
                left_row.get("control_id") != right_row.get("control_id")
                and right_start < left_end
                and right_row.get("overlap_explained") is not True
            ):
                _append_once(reasons, "concurrency_collision")
                break


def mutate_rows_for_attack(attack_id: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return a mutated row set for one critical attribution attack."""

    mutated: list[JsonDict] = json.loads(canonical_json(list(rows)))
    powered = next(
        row
        for row in mutated
        if row.get("control_id") == "powered" and row.get("phase") == "generation"
    )
    if attack_id == "forged_pid":
        powered["child_pids"] = [1]
    elif attack_id == "stale_nvidia_sample":
        powered["gpu_samples"][0]["monotonic_ns"] = powered["monotonic_start_ns"] - 10
        powered["gpu_samples"][0]["sample_age_s"] = 99.0
    elif attack_id == "model_name_only_substitution":
        powered["model_hash"] = "Gemma4-26B-A4B-it"
        powered["model_identity"]["model_identity_bound"] = False
    elif attack_id == "raw_output_reuse":
        cpu = next(row for row in mutated if row.get("control_id") == "cpu")
        powered["raw_output_hash"] = cpu["raw_output_hash"]
    elif attack_id == "runner_swap":
        powered["runner_selection"]["binary_sha256"] = "sha256:" + "0" * 64
    elif attack_id == "clock_rollback":
        powered["monotonic_end_ns"] = powered["monotonic_start_ns"] - 1
    elif attack_id == "truncated_receipt":
        del powered["exit_status"]
    elif attack_id == "concurrency_collision":
        cpu = next(
            row
            for row in mutated
            if row.get("control_id") == "cpu" and row.get("phase") == "generation"
        )
        powered["concurrency_group"] = cpu["concurrency_group"]
        powered["monotonic_start_ns"] = cpu["monotonic_start_ns"]
        powered["monotonic_end_ns"] = cpu["monotonic_end_ns"]
        powered["gpu_samples"][0]["monotonic_ns"] = powered["monotonic_start_ns"]
    elif attack_id == "cpu_fallback":
        powered["cpu_fallback"] = True
        powered["runner_selection"]["substrate"] = "cpu"
        powered["runner_selection"]["selection_hash"] = _runner_selection_hash(
            powered["runner_selection"]
        )
    elif attack_id == "child_exit_omission":
        powered["exit_status"].pop("returncode", None)
    else:
        raise ValueError(f"unknown attack_id: {attack_id}")
    return mutated


def mutation_attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_controls: Sequence[str],
) -> JsonDict:
    """Run attribution mutations and confirm each one fails closed."""

    matrix_rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        mutated = mutate_rows_for_attack(attack_id, rows)
        report = validate_contract_rows(mutated, expected_controls=expected_controls)
        matrix_rows.append(
            {
                "attack_id": attack_id,
                "accepted": report["accepted"],
                "fail_closed": not report["accepted"],
                "reasons": report["reasons"],
            }
        )
    false_accept_count = sum(1 for row in matrix_rows if row["accepted"])
    return {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": matrix_rows,
        "all_critical_fail_closed": false_accept_count == 0,
        "false_accept_count": false_accept_count,
        "attack_count": len(matrix_rows),
    }


def raw_hash_duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count raw hashes reused across controls."""

    controls_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        controls_by_hash[str(row.get("raw_output_hash"))].add(str(row.get("control_id")))
    return sum(1 for controls in controls_by_hash.values() if len(controls) > 1)


def parent_child_exit_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Summarize parent, child, and exit evidence by control."""

    summary: dict[str, JsonDict] = {}
    for row in rows:
        control_id = str(row.get("control_id"))
        item = summary.setdefault(
            control_id,
            {"parent_pids": set(), "child_pids": set(), "exit_statuses": []},
        )
        item["parent_pids"].add(row.get("parent_pid"))
        item["child_pids"].update(row.get("child_pids", []))
        item["exit_statuses"].append(row.get("exit_status", {}))
    return {
        control_id: {
            "parent_pids": sorted(value["parent_pids"]),
            "child_pids": sorted(value["child_pids"]),
            "exit_statuses": value["exit_statuses"],
        }
        for control_id, value in summary.items()
    }


def concurrency_group_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Summarize controls and phases assigned to each concurrency group."""

    groups: dict[str, JsonDict] = {}
    for row in rows:
        group = str(row.get("concurrency_group"))
        item = groups.setdefault(group, {"controls": set(), "phases": [], "row_count": 0})
        item["controls"].add(str(row.get("control_id")))
        item["phases"].append(str(row.get("phase")))
        item["row_count"] += 1
    return {
        group: {
            "controls": sorted(value["controls"]),
            "phases": value["phases"],
            "row_count": value["row_count"],
        }
        for group, value in groups.items()
    }


def command_config_model_raw_hashes(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Collect the hashes that bind a control to command, config, model, and output."""

    by_control: dict[str, JsonDict] = {}
    for row in rows:
        control_id = str(row.get("control_id"))
        item = by_control.setdefault(
            control_id,
            {
                "command_hashes": set(),
                "config_hashes": set(),
                "model_hashes": set(),
                "raw_output_hashes": set(),
            },
        )
        item["command_hashes"].add(str(row.get("command_hash")))
        item["config_hashes"].add(str(row.get("config_hash")))
        item["model_hashes"].add(str(row.get("model_hash")))
        item["raw_output_hashes"].add(str(row.get("raw_output_hash")))
    return {
        control_id: {key: sorted(values) for key, values in item.items()}
        for control_id, item in by_control.items()
    }


def pid_linked_gpu_samples(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return GPU samples that claim a concrete child PID link."""

    samples: list[JsonDict] = []
    for row in rows:
        child_pids = set(row.get("child_pids", []))
        for sample in row.get("gpu_samples", []):
            sample_map = dict(sample)
            sample_map["control_id"] = row.get("control_id")
            sample_map["phase"] = row.get("phase")
            sample_map["pid_linked"] = sample.get("pid") in child_pids
            samples.append(sample_map)
    return samples


def control_phase_counter(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count phase rows per control."""

    return dict(Counter(str(row.get("control_id")) for row in rows))
