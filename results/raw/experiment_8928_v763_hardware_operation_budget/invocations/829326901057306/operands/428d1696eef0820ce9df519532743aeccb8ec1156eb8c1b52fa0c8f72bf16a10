"""Build current-work provenance without absorbing cited model activity.

The current process owns one explicit event ledger. Scripted transport and old
model receipts stay in immutable sidecars, so their positive counters cannot
silently become claims about the current run.

Spec refs: REQ-REPORT-7395 and SCENARIO-REPORT-7395-CURRENT/MUTATIONS.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
OPERATIONS = {"model_load": "model_loads", "generation": "generation_calls"}
STATES = {"attempted", "completed", "failed", "cancelled"}
TERMINAL_STATES = {"completed", "failed", "cancelled"}
SIDECAR_SCOPES = {"historical_model_receipts", "simulated_transport_events"}


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so a removed event changes the receipt identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a referenced receipt cannot change unnoticed."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON only after its bytes reach local storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _path_label(path: Path, root: Path) -> str:
    """Use a stable relative label when the sidecar is below its declared root."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sidecar_reference(path: Path, *, root: Path, scope: str) -> JsonDict:
    """Return only path, byte hash, and scope for non-current evidence."""

    if scope not in SIDECAR_SCOPES:
        raise ValueError(f"sidecar_scope_invalid:{scope}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": _path_label(resolved, root),
        "sha256": sha256_file(resolved),
        "scope": scope,
    }


def write_immutable_sidecar(
    path: Path,
    *,
    scope: str,
    payload: Mapping[str, Any],
    root: Path,
) -> JsonDict:
    """Write one sidecar once and reject later bytes that differ.

    A rerun may reuse exact bytes. It must not replace preserved historical or
    simulated evidence with a new payload.
    """

    if scope not in SIDECAR_SCOPES:
        raise ValueError(f"sidecar_scope_invalid:{scope}")
    value = {
        "schema": "carnot.current_work_receipt.sidecar.v1",
        "scope": scope,
        "payload": deepcopy(dict(payload)),
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable_sidecar_conflict:{path}")
    else:
        atomic_json(path, value)
    return sidecar_reference(path, root=root, scope=scope)


def _reduce_events(
    events: Sequence[Mapping[str, Any]], *, run_id: str, owner_pid: int
) -> tuple[JsonDict, list[str]]:
    """Reduce one owned ledger and name each malformed or unfinished call."""

    counts = deepcopy(ZERO_INVOCATION_COUNTS)
    errors: list[str] = []
    by_call: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for index, event in enumerate(events):
        call_id = str(event.get("call_id") or f"index-{index}")
        operation = str(event.get("operation") or "")
        state = str(event.get("state") or "")
        timestamp = event.get("monotonic_ns")
        if (
            event.get("scope") != "current"
            or event.get("transport") != "owned_runtime"
            or event.get("run_id") != run_id
            or event.get("owner_pid") != owner_pid
        ):
            errors.append(f"event_not_owned:{call_id}")
        if operation not in OPERATIONS:
            errors.append(f"operation_invalid:{call_id}")
        if state not in STATES:
            errors.append(f"state_invalid:{call_id}")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            errors.append(f"event_time_invalid:{call_id}")
        if operation in OPERATIONS and state in STATES and isinstance(timestamp, int):
            by_call[call_id].append((operation, state, timestamp))

    for call_id, transitions in by_call.items():
        operations = {operation for operation, _state, _time in transitions}
        if len(operations) != 1:
            errors.append(f"operation_changed:{call_id}")
            continue
        operation = next(iter(operations))
        prefix = OPERATIONS[operation]
        state_counts = Counter(state for _operation, state, _time in transitions)
        attempts = state_counts["attempted"]
        terminals = sum(state_counts[state] for state in TERMINAL_STATES)
        if attempts == 0 and terminals:
            errors.append(f"terminal_without_attempt:{call_id}")
        elif attempts > 1:
            errors.append(f"duplicate_attempt:{call_id}")
        if attempts == 1 and terminals == 0:
            errors.append(f"unfinished_call:{call_id}")
        elif terminals > 1:
            errors.append(f"duplicate_terminal:{call_id}")
        attempt_times = [time_ns for _op, state, time_ns in transitions if state == "attempted"]
        terminal_times = [
            time_ns for _op, state, time_ns in transitions if state in TERMINAL_STATES
        ]
        if attempt_times and terminal_times and min(terminal_times) < min(attempt_times):
            errors.append(f"event_time_regression:{call_id}")
        counts[f"{prefix}_attempted"] += int(attempts == 1)
        for state in TERMINAL_STATES:
            counts[f"{prefix}_{state}"] += state_counts[state]
        counts[f"{prefix}_in_flight"] += int(attempts == 1 and terminals == 0)
    return counts, list(dict.fromkeys(errors))


def build_current_work_receipt(
    *,
    run_id: str,
    owner_pid: int,
    events: Sequence[Mapping[str, Any]],
    inference_substrate: str,
    inference_substrate_details: Mapping[str, Any],
    inference_substrate_class: str,
    execution_venue: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    sidecar_references: Sequence[Mapping[str, Any]] = (),
    phase_spans: Sequence[Mapping[str, Any]] = (),
    small_ebm_training: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build current provenance from owned events and opaque sidecar references."""

    if not isinstance(inference_substrate, str) or not inference_substrate.strip():
        raise ValueError("inference_substrate_must_be_string")
    if ended_monotonic_ns < started_monotonic_ns:
        raise ValueError("monotonic_boundary_order_invalid")
    counts, errors = _reduce_events(events, run_id=run_id, owner_pid=owner_pid)
    if errors:
        raise ValueError("invalid_current_event_ledger:" + ",".join(errors))
    attempted = counts["model_loads_attempted"] + counts["generation_calls_attempted"]
    return {
        "MODEL_SPECS": [],
        "model_invoked": attempted > 0,
        "invocation_counts": counts,
        "current_invocation_events": [deepcopy(dict(row)) for row in events],
        "current_run_id": run_id,
        "current_owner_pid": owner_pid,
        "event_count": len(events),
        "event_sha256": canonical_hash(events),
        "inference_substrate": inference_substrate,
        "inference_substrate_details": deepcopy(dict(inference_substrate_details)),
        "inference_substrate_class": inference_substrate_class,
        "execution_venue": execution_venue,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "receipt_sidecars": [deepcopy(dict(row)) for row in sidecar_references],
        "small_ebm_training": deepcopy(dict(small_ebm_training or {"performed": False})),
    }


def validate_current_work_receipt(
    value: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]] | None = None,
    *,
    root: Path | None = None,
) -> list[str]:
    """Recompute current counters, timing, venue, and each sidecar byte hash."""

    errors: list[str] = []
    if not isinstance(value.get("inference_substrate"), str):
        errors.append("inference_substrate_not_string")
    if not isinstance(value.get("inference_substrate_details"), Mapping):
        errors.append("inference_substrate_details_invalid")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    embedded = value.get("current_invocation_events")
    event_rows = (
        list(events) if events is not None else (embedded if isinstance(embedded, list) else [])
    )
    run_id = value.get("current_run_id")
    owner_pid = value.get("current_owner_pid")
    if not isinstance(run_id, str) or not isinstance(owner_pid, int):
        errors.append("current_owner_identity_invalid")
        counts = deepcopy(ZERO_INVOCATION_COUNTS)
    else:
        counts, event_errors = _reduce_events(event_rows, run_id=run_id, owner_pid=owner_pid)
        errors.extend(event_errors)
    if value.get("event_count") != len(event_rows):
        errors.append("event_count_mismatch")
    if value.get("event_sha256") != canonical_hash(event_rows):
        errors.append("event_hash_mismatch")
    if value.get("invocation_counts") != counts:
        errors.append("invocation_counts_mismatch")
    expected_invoked = counts["model_loads_attempted"] + counts["generation_calls_attempted"] > 0
    if value.get("model_invoked") is not expected_invoked:
        errors.append("model_invoked_mismatch")

    started = value.get("started_monotonic_ns")
    ended = value.get("ended_monotonic_ns")
    duration = value.get("duration_s")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in (started, ended)):
        errors.append("monotonic_boundaries_invalid")
    elif ended < started:
        errors.append("monotonic_boundary_order_invalid")
    else:
        expected_duration = (ended - started) / 1_000_000_000
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            errors.append("duration_invalid")
        elif abs(float(duration) - expected_duration) > 1e-9:
            errors.append("duration_mismatch")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        for span in value.get("phase_spans") or []:
            if not isinstance(span, Mapping):
                errors.append("phase_span_invalid")
                continue
            end_s = span.get("end_s")
            if isinstance(end_s, (int, float)) and end_s > duration:
                errors.append(f"phase_span_exceeds_duration:{span.get('phase')}")

    base = (root or Path.cwd()).resolve()
    for index, row in enumerate(value.get("receipt_sidecars") or []):
        if not isinstance(row, Mapping):
            errors.append(f"sidecar_reference_invalid:{index}")
            continue
        path_label = str(row.get("path") or "")
        if row.get("scope") not in SIDECAR_SCOPES:
            errors.append(f"sidecar_scope_invalid:{index}")
        hash_value = row.get("sha256")
        if not isinstance(hash_value, str) or not hash_value.startswith("sha256:"):
            errors.append(f"sidecar_hash_invalid:{index}")
        path = Path(path_label)
        resolved = path if path.is_absolute() else base / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        if observed != hash_value:
            errors.append(f"sidecar_hash_mismatch:{path_label}")
    return list(dict.fromkeys(errors))
