"""Persist ARC model activity before a blocking native or HTTP call starts.

A child can be killed while a model loads or generates. In-memory counters then
die with the child. This module writes one small JSON line before the call so the
owning parent can distinguish in-flight work from work that never started.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot.agentic.arc_request_budget import RequestReservation, reserve_for_proposer

BOUNDARY_LEDGER_ENV = "CARNOT_ARC_BOUNDARY_LEDGER_PATH"
BOUNDARY_EVENT_SCHEMA = "carnot.arc_inference_boundary_event.v1"
OPERATIONS = frozenset({"model_load", "generation"})
STATES = frozenset({"attempted", "child_started", "completed", "failed"})


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _event_id(call_id: str, state: str) -> str:
    payload = f"{call_id}\x00{state}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _error_text(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"[:500]
    return str(error)[:500]


def model_identity_from_proposer(proposer: Any) -> dict[str, Any]:
    """Copy identity observations already held by the live proposer.

    The boundary must not hash a multi-gigabyte model while a request waits.
    Strict file identity remains the job of the existing typed identity receipt.
    """
    path = getattr(proposer, "requested_model_path", None) or getattr(proposer, "model_path", None)
    return {
        "model_repository": str(getattr(proposer, "model_repository", "") or ""),
        "model_filename": str(
            getattr(proposer, "requested_model_filename", None)
            or getattr(proposer, "model_filename", "")
            or ""
        ),
        "model_revision": str(getattr(proposer, "model_revision", "") or ""),
        "model_path": str(path or ""),
    }


def _identity_errors(identity: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("model_repository", "model_filename", "model_revision"):
        if not isinstance(identity.get(field), str) or not str(identity.get(field)).strip():
            errors.append(f"{field}_missing")
    path = identity.get("model_path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        errors.append("model_path_not_absolute")
    digest = identity.get("model_hash")
    if digest is not None and (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest.removeprefix("sha256:")) != 64
        or any(char not in "0123456789abcdef" for char in digest.removeprefix("sha256:").lower())
    ):
        errors.append("model_hash_invalid")
    return errors


@dataclass
class BoundaryCall:
    """Append later states for one already-persisted attempted call."""

    ledger: InvocationBoundaryLedger | None
    operation: str
    call_id: str
    model_identity: dict[str, Any]
    owner_pid: int
    started_monotonic_ns: int
    child_pid: int | None
    request_reservation: RequestReservation | None = None
    terminal: bool = False

    @classmethod
    def disabled(
        cls, operation: str, request_reservation: RequestReservation | None = None
    ) -> BoundaryCall:
        return cls(None, operation, "", {}, os.getpid(), 0, None, request_reservation)

    def _append(self, state: str, **values: Any) -> None:
        if self.ledger is None:
            return
        event = {
            "schema": BOUNDARY_EVENT_SCHEMA,
            "event_id": _event_id(self.call_id, state),
            "call_id": self.call_id,
            "operation": self.operation,
            "state": state,
            "recorded_monotonic_ns": time.monotonic_ns(),
            "started_monotonic_ns": self.started_monotonic_ns,
            "owner_pid": self.owner_pid,
            "child_pid": self.child_pid,
            "model_identity": dict(self.model_identity),
            **values,
        }
        self.ledger.append_event(event)

    def child_started(self, child_pid: int) -> None:
        """Persist the native child PID as soon as ``Popen`` returns."""
        if self.terminal:
            raise RuntimeError("boundary call is already terminal")
        self.child_pid = int(child_pid)
        self._append("child_started")

    def complete(self, *, usable: bool | None = None) -> None:
        """Append a completed state without treating content quality as transport."""
        if self.terminal:
            raise RuntimeError("boundary call is already terminal")
        if self.request_reservation is not None:
            self.request_reservation.complete()
        self.terminal = True
        self._append(
            "completed",
            ended_monotonic_ns=time.monotonic_ns(),
            usable=usable,
        )

    def fail(self, error: BaseException | str) -> None:
        """Append a failed state while retaining the earlier attempt."""
        if self.terminal:
            raise RuntimeError("boundary call is already terminal")
        if self.request_reservation is not None:
            self.request_reservation.fail(error)
        self.terminal = True
        self._append(
            "failed",
            ended_monotonic_ns=time.monotonic_ns(),
            error=_error_text(error),
        )


class InvocationBoundaryLedger:
    """Write JSONL with append, lock, flush, and fsync semantics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append_event(self, event: Mapping[str, Any]) -> None:
        """Append one complete line so a reader never observes an overwrite."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_bytes(dict(event)) + b"\n"
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            written = os.write(fd, payload)
            if written != len(payload):  # pragma: no cover - defensive OS short-write guard
                raise OSError(f"short boundary-ledger write: {written}/{len(payload)}")
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def begin(
        self,
        operation: str,
        model_identity: Mapping[str, Any],
        *,
        child_pid: int | None = None,
        call_id: str | None = None,
    ) -> BoundaryCall:
        """Persist the attempted event before returning control to the caller."""
        if operation not in OPERATIONS:
            raise ValueError(f"unknown boundary operation: {operation}")
        identity = dict(model_identity)
        identifier = call_id or f"{os.getpid()}-{time.monotonic_ns()}-{uuid.uuid4().hex}"
        started = time.monotonic_ns()
        call = BoundaryCall(
            self,
            operation,
            identifier,
            identity,
            os.getpid(),
            started,
            int(child_pid) if child_pid is not None else None,
        )
        call._append("attempted")
        return call

    def read_events(self) -> list[dict[str, Any]]:
        """Read every full line and preserve malformed activity as a rejection row."""
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.path.read_bytes().splitlines(), start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "_malformed": True,
                        "line_number": line_number,
                        "error": f"JSONDecodeError: {exc.msg}",
                    }
                )
                continue
            if not isinstance(value, dict):
                rows.append({"_malformed": True, "line_number": line_number, "error": "not_object"})
            else:
                rows.append(value)
        return rows


def boundary_call_for_proposer(proposer: Any, operation: str) -> BoundaryCall:
    """Start a receipt when the optional live-ledger environment variable is set."""
    reservation = reserve_for_proposer(proposer, operation)
    path = os.environ.get(BOUNDARY_LEDGER_ENV)
    if not path:
        return BoundaryCall.disabled(operation, reservation)
    process = getattr(proposer, "_proc", None)
    child_pid = getattr(process, "pid", None) if operation == "generation" else None
    call = InvocationBoundaryLedger(path).begin(
        operation,
        model_identity_from_proposer(proposer),
        child_pid=child_pid,
    )
    call.request_reservation = reservation
    return call


def _empty_counts() -> dict[str, int]:
    return {
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "model_loads_failed": 0,
        "model_loads_in_flight": 0,
        "generation_calls_attempted": 0,
        "generation_calls_completed": 0,
        "generation_calls_failed": 0,
        "generation_calls_in_flight": 0,
        "usable_answers": 0,
    }


def reduce_boundary_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce append-only events without inventing activity after corrupt input."""
    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    errors: list[str] = []
    for raw in events:
        event = dict(raw)
        if event.get("_malformed"):
            errors.append(f"malformed_event_line:{event.get('line_number')}")
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            errors.append("event_id_missing")
            continue
        prior = unique.get(event_id)
        if prior is not None:
            if prior == event:
                duplicate_count += 1
            else:
                errors.append("conflicting_duplicate_event")
            continue
        unique[event_id] = event

    calls: dict[str, list[dict[str, Any]]] = {}
    for event in unique.values():
        if event.get("schema") != BOUNDARY_EVENT_SCHEMA:
            errors.append("event_schema_invalid")
        if event.get("operation") not in OPERATIONS:
            errors.append("unknown_operation")
        if event.get("state") not in STATES:
            errors.append("unknown_state")
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            errors.append("call_id_missing")
            continue
        calls.setdefault(call_id, []).append(event)

    call_rows: list[dict[str, Any]] = []
    counts = _empty_counts()
    for call_id, call_events in calls.items():
        attempted = [row for row in call_events if row.get("state") == "attempted"]
        completed = [row for row in call_events if row.get("state") == "completed"]
        failed = [row for row in call_events if row.get("state") == "failed"]
        child_events = [row for row in call_events if row.get("state") == "child_started"]
        if len(attempted) != 1:
            errors.append(f"attempted_event_count_invalid:{call_id}")
            continue
        first = attempted[0]
        identity = first.get("model_identity")
        if not isinstance(identity, Mapping):
            errors.append(f"model_identity_missing:{call_id}")
            identity = {}
        errors.extend(f"{item}:{call_id}" for item in _identity_errors(identity))
        operation = first.get("operation")
        if any(row.get("operation") != operation for row in call_events):
            errors.append(f"operation_changed:{call_id}")
        if len(completed) + len(failed) > 1:
            errors.append(f"multiple_terminal_events:{call_id}")
        terminal = (completed or failed or [None])[0]
        start_ns = first.get("started_monotonic_ns")
        if not isinstance(start_ns, int) or start_ns < 0:
            errors.append(f"start_monotonic_invalid:{call_id}")
        if terminal is not None:
            end_ns = terminal.get("ended_monotonic_ns")
            if not isinstance(end_ns, int) or not isinstance(start_ns, int) or end_ns < start_ns:
                errors.append(f"end_monotonic_invalid:{call_id}")
        child_pid = first.get("child_pid")
        if child_events:
            child_pid = child_events[-1].get("child_pid")
        if terminal is not None and terminal.get("child_pid") is not None:
            child_pid = terminal.get("child_pid")
        prefix = "model_loads" if operation == "model_load" else "generation_calls"
        counts[f"{prefix}_attempted"] += 1
        if completed:
            counts[f"{prefix}_completed"] += 1
            if operation == "generation" and completed[0].get("usable") is True:
                counts["usable_answers"] += 1
        elif failed:
            counts[f"{prefix}_failed"] += 1
        else:
            counts[f"{prefix}_in_flight"] += 1
        call_rows.append(
            {
                "call_id": call_id,
                "operation": operation,
                "terminal_state": (
                    "completed" if completed else "failed" if failed else "in_flight"
                ),
                "started_monotonic_ns": start_ns,
                "ended_monotonic_ns": terminal.get("ended_monotonic_ns") if terminal else None,
                "owner_pid": first.get("owner_pid"),
                "child_pid": child_pid,
                "model_identity": dict(identity),
                "usable": completed[0].get("usable") if completed else None,
                "error": failed[0].get("error") if failed else None,
            }
        )

    if errors:
        return {
            "activity_known": False,
            "disqualified": True,
            "errors": sorted(set(errors)),
            "duplicate_event_count": duplicate_count,
            "unique_event_count": len(unique),
            "model_invoked": None,
            "invocation_counts": None,
            "inference_substrate": "disqualified_unknown_activity",
            "call_rows": sorted(call_rows, key=lambda row: str(row["call_id"])),
        }
    invoked = bool(counts["model_loads_attempted"] or counts["generation_calls_attempted"])
    if counts["generation_calls_attempted"]:
        substrate = "model_full_generation"
    elif counts["model_loads_completed"]:
        substrate = "model_load_no_generation"
    elif counts["model_loads_failed"]:
        substrate = "model_load_failed"
    elif counts["model_loads_in_flight"]:
        substrate = "model_load_in_flight"
    else:
        substrate = "no_model_activity"
    return {
        "activity_known": True,
        "disqualified": False,
        "errors": [],
        "duplicate_event_count": duplicate_count,
        "unique_event_count": len(unique),
        "model_invoked": invoked,
        "invocation_counts": counts,
        "inference_substrate": substrate,
        "call_rows": sorted(call_rows, key=lambda row: str(row["call_id"])),
    }
