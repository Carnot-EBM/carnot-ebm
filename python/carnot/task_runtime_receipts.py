"""Reusable task-scoped runtime receipt helpers.

Spec refs: REQ-INFRA-6426, SCENARIO-INFRA-6426-1,
SCENARIO-INFRA-6426-4, SCENARIO-INFRA-6426-5.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import contextlib
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

SCHEMA_VERSION = "carnot.task_scoped_runtime_receipt.v1"
ADOPTION_SCHEMA_VERSION = "carnot.task_runtime_receipt_adoption.v1"
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


def seal_adoption_row(row: Mapping[str, Any]) -> JsonDict:
    """Bind a phase row to its complete structured contents.

    The stored digest makes accidental edits and partial copies visible during
    a fresh-process check. Process ownership still comes from the recorded
    Linux process identities, not from this digest alone.
    """

    sealed = dict(row)
    sealed.pop("receipt_hash", None)
    sealed["receipt_hash"] = sha256_json(sealed)
    return sealed


def _adoption_row_hash_valid(row: Mapping[str, Any]) -> bool:
    """Return true when a row still matches the digest made by its task."""

    payload = dict(row)
    stored = payload.pop("receipt_hash", None)
    return stored == sha256_json(payload)


def read_process_identity(pid: int) -> JsonDict | None:
    """Read stable Linux identity fields for one live process.

    A PID can be reused after a process exits. The kernel start-time field and
    boot ID distinguish that reuse, while the command hash binds the identity
    to the process that the task actually observed.
    """

    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        suffix = stat[stat.rfind(")") + 2 :].split()
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").rstrip()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        return {
            "pid": int(pid),
            "parent_pid": int(suffix[1]),
            "start_time_ticks": int(suffix[19]),
            "boot_id": boot_id,
            "cmdline_hash": sha256_bytes(cmdline),
        }
    except (IndexError, OSError, ValueError):
        return None


def capture_process_lineage(child_pid: int, task_identity: Mapping[str, Any]) -> JsonDict:
    """Capture a child-to-task chain while every process identity is observable."""

    chain: list[JsonDict] = []
    current_pid = int(child_pid)
    task_pid = int(task_identity["pid"])
    for _ in range(64):
        identity = read_process_identity(current_pid)
        if identity is None:
            break
        chain.append(identity)
        if current_pid == task_pid:
            break
        parent_pid = int(identity["parent_pid"])
        if parent_pid <= 1 or parent_pid == current_pid:
            break
        current_pid = parent_pid
    owned = bool(chain) and chain[-1] == dict(task_identity)
    return {"child_pid": int(child_pid), "owned": owned, "chain": chain}


def _lineage_is_task_owned(
    lineage: Mapping[str, Any], task_identity: Mapping[str, Any]
) -> bool:
    """Check that a recorded child chain terminates at the exact task identity."""

    chain = lineage.get("chain")
    if not isinstance(chain, Sequence) or isinstance(chain, (str, bytes)) or not chain:
        return False
    first = _as_mapping(chain[0])
    last = _as_mapping(chain[-1])
    return (
        lineage.get("owned") is True
        and _int_value(lineage.get("child_pid")) == _int_value(first.get("pid"))
        and dict(last) == dict(task_identity)
    )


def _validate_adoption_gpu_samples(row: Mapping[str, Any], reasons: list[str]) -> None:
    """Validate GPU samples only for rows that declare a GPU device."""

    devices = {str(device) for device in row.get("device_ids", []) if str(device) != "CPU"}
    if not devices or row.get("phase") != "generation":
        return
    samples = row.get("gpu_samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
        _append_once(reasons, "gpu_sample_missing")
        return
    start = _int_value(row.get("monotonic_start_ns"))
    end = _int_value(row.get("monotonic_end_ns"))
    child_pids = {_int_value(pid) for pid in row.get("child_pids", [])}
    clocks: list[int] = []
    required = (
        "pid",
        "device_uuid",
        "pid_memory_mb",
        "device_memory_used_mb",
        "utilization_pct",
        "offload_layers",
        "monotonic_ns",
    )
    for sample_value in samples:
        sample = _as_mapping(sample_value)
        if any(field not in sample for field in required):
            _append_once(reasons, "gpu_sample_field_missing")
        if _int_value(sample.get("pid")) not in child_pids:
            _append_once(reasons, "gpu_sample_pid_mismatch")
        if str(sample.get("device_uuid")) not in devices:
            _append_once(reasons, "gpu_uuid_mismatch")
        clock = _int_value(sample.get("monotonic_ns"))
        if clock is None or start is None or end is None or not start <= clock <= end:
            _append_once(reasons, "gpu_sample_outside_phase")
        else:
            clocks.append(clock)
    if clocks and start is not None and end is not None:
        points = [start, *sorted(clocks), end]
        largest_gap_s = max(right - left for left, right in zip(points, points[1:], strict=False))
        largest_gap_s /= 1_000_000_000
        limit = float(row.get("telemetry_sample_gap_limit_s", 5.0) or 0.0)
        if largest_gap_s > limit:
            _append_once(reasons, "gpu_telemetry_gap")


def _peak_model_concurrency(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count the largest number of distinct models active at one instant."""

    events: list[tuple[int, int, str]] = []
    for row in rows:
        lifecycle = _as_mapping(row.get("model_lifecycle"))
        model_id = str(lifecycle.get("model_id", ""))
        start = _int_value(row.get("monotonic_start_ns"))
        end = _int_value(row.get("monotonic_end_ns"))
        if model_id and start is not None and end is not None:
            events.extend(((start, 1, model_id), (end, -1, model_id)))
    active: Counter[str] = Counter()
    peak = 0
    for _clock, delta, model_id in sorted(events, key=lambda item: (item[0], item[1])):
        active[model_id] += delta
        if active[model_id] <= 0:
            del active[model_id]
        peak = max(peak, len(active))
    return peak


def validate_adoption_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_task_id: str,
    expected_task_pid: int,
) -> JsonDict:
    """Recompute ownership, timing, concurrency, GPU, and teardown evidence."""

    reasons: list[str] = []
    intervals: list[tuple[int, int, Mapping[str, Any]]] = []
    model_intervals: list[tuple[int, int, Mapping[str, Any]]] = []
    started_servers: dict[str, int] = {}
    stopped_servers: dict[str, int] = {}
    canonical_task_identity: Mapping[str, Any] | None = None

    for row in rows:
        if not _adoption_row_hash_valid(row):
            _append_once(reasons, "receipt_hash_mismatch")
        if row.get("task_id") != expected_task_id:
            _append_once(reasons, "task_id_mismatch")
        if _int_value(row.get("parent_pid")) != expected_task_pid:
            _append_once(reasons, "task_pid_mismatch")
        task_identity = _as_mapping(row.get("task_process_identity"))
        if _int_value(task_identity.get("pid")) != expected_task_pid:
            _append_once(reasons, "task_identity_mismatch")
        if canonical_task_identity is None:
            canonical_task_identity = task_identity
        elif dict(task_identity) != dict(canonical_task_identity):
            _append_once(reasons, "cross_process_receipt")

        lineages = row.get("process_lineage", [])
        lineage_by_pid = {
            _int_value(_as_mapping(item).get("child_pid")): _as_mapping(item)
            for item in lineages
        }
        for child_pid_value in row.get("child_pids", []):
            child_pid = _int_value(child_pid_value)
            lineage = lineage_by_pid.get(child_pid, {})
            if not _lineage_is_task_owned(lineage, task_identity):
                _append_once(reasons, "cross_process_child")

        start = _int_value(row.get("monotonic_start_ns"))
        end = _int_value(row.get("monotonic_end_ns"))
        if start is None or end is None or end < start:
            _append_once(reasons, "invalid_monotonic_interval")
        else:
            intervals.append((start, end, row))
            if _as_mapping(row.get("model_lifecycle")).get("model_id"):
                model_intervals.append((start, end, row))

        runner = _as_mapping(row.get("runner_selection"))
        if runner.get("selection_hash") != _runner_selection_hash(runner):
            _append_once(reasons, "runner_selection_hash_mismatch")
        if runner.get("selected") is not True:
            _append_once(reasons, "runner_not_selected")
        _validate_adoption_gpu_samples(row, reasons)

        lifecycle = _as_mapping(row.get("server_lifecycle"))
        server_id = str(lifecycle.get("server_id", ""))
        server_pid = _int_value(lifecycle.get("pid"))
        if lifecycle.get("event") == "started" and server_id and server_pid is not None:
            started_servers[server_id] = server_pid
        if lifecycle.get("event") == "teardown" and server_id and server_pid is not None:
            if (
                lifecycle.get("process_exit_confirmed") is True
                and lifecycle.get("process_reaped") is True
            ):
                stopped_servers[server_id] = server_pid

    _validate_interval_order(intervals, reasons)
    ordered_models = sorted(model_intervals, key=lambda item: (item[0], item[1]))
    for left_index, (left_start, left_end, left_row) in enumerate(ordered_models):
        left_lifecycle = _as_mapping(left_row.get("model_lifecycle"))
        for right_start, right_end, right_row in ordered_models[left_index + 1 :]:
            if right_start >= left_end:
                break
            right_lifecycle = _as_mapping(right_row.get("model_lifecycle"))
            different_models = left_lifecycle.get("model_id") != right_lifecycle.get("model_id")
            declared = (
                different_models
                and left_lifecycle.get("concurrency_mode") == "concurrent"
                and right_lifecycle.get("concurrency_mode") == "concurrent"
                and left_row.get("concurrency_group") == right_row.get("concurrency_group")
                and left_row.get("overlap_explained") is True
                and right_row.get("overlap_explained") is True
            )
            if different_models and not declared and right_start < min(left_end, right_end):
                _append_once(reasons, "sequential_model_overlap")

    for server_id, server_pid in started_servers.items():
        if stopped_servers.get(server_id) != server_pid:
            _append_once(reasons, "missing_server_teardown")

    duration_ns = sum(max(0, end - start) for start, end, _row in intervals)
    ownership_reasons = {
        "task_id_mismatch",
        "task_pid_mismatch",
        "task_identity_mismatch",
        "cross_process_receipt",
        "cross_process_child",
        "receipt_hash_mismatch",
    }
    ordering_reasons = {"invalid_monotonic_interval", "overlap_unexplained"}
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "phase_order_valid": not bool(ordering_reasons.intersection(reasons)),
        "ownership_valid": not bool(ownership_reasons.intersection(reasons)),
        "teardown_complete": "missing_server_teardown" not in reasons,
        "peak_model_concurrency": _peak_model_concurrency(rows),
        "recomputed_duration_s": round(duration_ns / 1_000_000_000, 9),
        "row_count": len(rows),
    }


def build_adoption_receipt(
    *,
    task_id: str,
    task_process_identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Build the stable top-level receipt around existing phase rows."""

    copied_rows = [dict(row) for row in rows]
    return {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "task_id": task_id,
        "task_process_identity": dict(task_process_identity),
        "rows": copied_rows,
        "receipt_sha256": sha256_json(copied_rows),
        "validation": dict(validation),
    }


def write_adoption_receipt(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write an adoption receipt with deterministic key ordering."""

    return write_json_atomic(path, payload)


def load_and_validate_adoption_receipt(
    path: str | Path,
    *,
    expected_task_id: str,
    expected_task_pid: int,
) -> JsonDict:
    """Load serialized rows and independently recompute their evidence."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    report = validate_adoption_rows(
        rows,
        expected_task_id=expected_task_id,
        expected_task_pid=expected_task_pid,
    )
    if payload.get("receipt_sha256") != sha256_json(rows):
        report["accepted"] = False
        _append_once(report["reasons"], "receipt_payload_hash_mismatch")
    return report


class TaskRuntimeReceiptAdoption:
    """Record task-owned phase rows through one reusable context manager.

    The class uses the existing phase-row schema and partial writer. A caller
    adds command output, exit status, or samples to the yielded state before
    the phase closes, so the final row hashes the evidence actually observed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        task_id: str,
        control_id: str,
        runner_selection: Mapping[str, Any],
        model_identity: Mapping[str, Any],
        device_ids: Sequence[str],
        model_count: int,
        concurrency_group: str | None = None,
        config: Mapping[str, Any] | None = None,
        phase_timings: list[JsonDict] | None = None,
    ) -> None:
        self.path = Path(path)
        self.task_id = task_id
        self.control_id = control_id
        self.runner_selection = dict(runner_selection)
        self.model_identity = dict(model_identity)
        self.device_ids = list(device_ids)
        self.model_count = int(model_count)
        self.concurrency_group = concurrency_group or f"{task_id}:{control_id}"
        self.config = dict(config or {})
        self.task_pid = os.getpid()
        identity = read_process_identity(self.task_pid)
        if identity is None:
            raise RuntimeError("task process identity is unavailable")
        self.task_process_identity = identity
        self.rows: list[JsonDict] = []
        self._writer = TaskScopedReceiptWriter(self.path, task_id=task_id)
        self._phase_timings = phase_timings

    def __enter__(self) -> TaskRuntimeReceiptAdoption:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        validation = validate_adoption_rows(
            self.rows,
            expected_task_id=self.task_id,
            expected_task_pid=self.task_pid,
        )
        payload = build_adoption_receipt(
            task_id=self.task_id,
            task_process_identity=self.task_process_identity,
            rows=self.rows,
            validation=validation,
        )
        if exc is not None:
            payload["status"] = "partial"
            payload["exception"] = f"{type(exc).__name__}: {exc}"
        else:
            payload["status"] = "complete" if validation["accepted"] else "rejected"
        write_adoption_receipt(self.path, payload)
        return False

    @contextlib.contextmanager
    def phase(
        self,
        name: str,
        *,
        model_id: str | None = None,
        child_pids: Sequence[int] = (),
        server_lifecycle: Mapping[str, Any] | None = None,
        concurrency_mode: str = "sequential",
        overlap_explained: bool = False,
        telemetry_sample_gap_limit_s: float = 5.0,
        metadata: Mapping[str, Any] | None = None,
    ):
        """Capture one phase and yield mutable evidence fields to the caller."""

        start_ns = time.monotonic_ns()
        wall_start = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        lineage = [
            capture_process_lineage(int(child_pid), self.task_process_identity)
            for child_pid in child_pids
        ]
        state: JsonDict = {
            "raw_output_bytes": b"",
            "exit_status": {"returncode": 0, "timed_out": False, "signal": None},
            "gpu_samples": [],
        }
        try:
            yield state
        finally:
            end_ns = time.monotonic_ns()
            wall_end = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            raw_output = state.get("raw_output_bytes", b"")
            if isinstance(raw_output, str):
                raw_output = raw_output.encode("utf-8")
            selected_model_id = model_id or str(
                self.model_identity.get("model_id")
                or self.model_identity.get("hf_id")
                or self.model_identity.get("name")
                or ""
            )
            row = build_phase_row(
                task_id=self.task_id,
                control_id=self.control_id,
                phase=name,
                monotonic_start_ns=start_ns,
                monotonic_end_ns=end_ns,
                wall_clock_start=wall_start,
                wall_clock_end=wall_end,
                parent_pid=self.task_pid,
                child_pids=child_pids,
                command=sys.argv,
                config=self.config,
                model_identity=self.model_identity,
                runner_selection=self.runner_selection,
                device_ids=self.device_ids,
                concurrency_group=self.concurrency_group,
                raw_output_bytes=raw_output,
                exit_status=_as_mapping(state.get("exit_status")),
                attribution_confidence=1.0,
                gpu_samples=state.get("gpu_samples", []),
                extra={
                    "task_process_identity": self.task_process_identity,
                    "process_lineage": lineage,
                    "model_lifecycle": {
                        "model_id": selected_model_id,
                        "model_count": self.model_count,
                        "concurrency_mode": concurrency_mode,
                    },
                    "server_lifecycle": dict(server_lifecycle or {}),
                    "overlap_explained": overlap_explained,
                    "telemetry_sample_gap_limit_s": telemetry_sample_gap_limit_s,
                    "phase_metadata": dict(metadata or {}),
                },
            )
            sealed = seal_adoption_row(row)
            self.rows.append(sealed)
            self._writer.record_phase(sealed)
            if self._phase_timings is not None:
                self._phase_timings.append(
                    {
                        "name": name,
                        "elapsed_s": round((end_ns - start_ns) / 1_000_000_000, 9),
                        "monotonic_start_ns": start_ns,
                        "monotonic_end_ns": end_ns,
                        **dict(metadata or {}),
                    }
                )


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
