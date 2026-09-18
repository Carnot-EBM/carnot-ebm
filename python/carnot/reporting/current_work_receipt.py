"""Build current-work provenance without absorbing cited model activity.

A report often needs old model receipts or scripted HTTP events as evidence.
Those records do not describe what the current process did. This helper keeps
them behind hash-bound references and derives current counters from one owned
event ledger.

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
SIDECAR_SCOPES = {"historical", "simulated_transport"}


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a referenced receipt cannot change unnoticed."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete sidecar only after its bytes reach local storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sidecar_reference(path: Path, *, root: Path, scope: str) -> JsonDict:
    """Return only path, byte hash, and scope for non-current evidence."""

    resolved = path.resolve()
    base = root.resolve()
    try:
        label = resolved.relative_to(base).as_posix()
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved), "scope": scope}


def _reduce_events(events: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, list[str]]:
    """Reduce one event ledger and name every malformed or unfinished call."""

    counts = deepcopy(ZERO_INVOCATION_COUNTS)
    errors: list[str] = []
    by_id: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, event in enumerate(events):
        event_id = str(event.get("event_id") or f"index-{index}")
        operation = str(event.get("operation") or "")
        state = str(event.get("state") or "")
        if operation not in OPERATIONS:
            errors.append(f"event_operation_invalid:{event_id}")
        if state not in STATES:
            errors.append(f"event_state_invalid:{event_id}")
        if operation in OPERATIONS and state in STATES:
            by_id[event_id].append((operation, state))

    for event_id, transitions in by_id.items():
        operations = {operation for operation, _state in transitions}
        states = [state for _operation, state in transitions]
        if len(operations) != 1:
            errors.append(f"event_operation_changed:{event_id}")
            continue
        operation = next(iter(operations))
        prefix = OPERATIONS[operation]
        state_counts = Counter(states)
        if state_counts["attempted"] != 1:
            errors.append(f"event_attempt_count_invalid:{event_id}")
        terminal_count = sum(state_counts[state] for state in TERMINAL_STATES)
        if terminal_count == 0:
            errors.append(f"unfinished_current_invocation:{event_id}")
        elif terminal_count != 1:
            errors.append(f"event_terminal_count_invalid:{event_id}")
        counts[f"{prefix}_attempted"] += int(state_counts["attempted"] == 1)
        for state in TERMINAL_STATES:
            counts[f"{prefix}_{state}"] += state_counts[state]
        counts[f"{prefix}_in_flight"] += int(state_counts["attempted"] == 1 and terminal_count == 0)
    return counts, list(dict.fromkeys(errors))


def build_current_work_receipt(
    *,
    inference_substrate: str,
    inference_substrate_details: Mapping[str, Any],
    inference_substrate_class: str,
    execution_venue: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    owned_run_events: Sequence[Mapping[str, Any]],
    sidecar_references: Sequence[Mapping[str, Any]],
    small_ebm_training: Mapping[str, Any],
) -> JsonDict:
    """Build current provenance from owned events and opaque sidecar references."""

    counts, _errors = _reduce_events(owned_run_events)
    attempted = counts["model_loads_attempted"] + counts["generation_calls_attempted"]
    return {
        "MODEL_SPECS": [],
        "model_invoked": attempted > 0,
        "invocation_counts": counts,
        "current_invocation_events": [deepcopy(dict(row)) for row in owned_run_events],
        "inference_substrate": inference_substrate,
        "inference_substrate_details": deepcopy(dict(inference_substrate_details)),
        "inference_substrate_class": inference_substrate_class,
        "execution_venue": execution_venue,
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "receipt_sidecars": [deepcopy(dict(row)) for row in sidecar_references],
        "small_ebm_training": deepcopy(dict(small_ebm_training)),
    }


def validate_current_work_receipt(
    value: Mapping[str, Any], *, root: Path | None = None
) -> list[str]:
    """Recompute current counters, timing, venue, and every sidecar byte hash."""

    errors: list[str] = []
    if not isinstance(value.get("inference_substrate"), str):
        errors.append("inference_substrate_not_string")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    events = value.get("current_invocation_events")
    event_rows = events if isinstance(events, list) else []
    counts, event_errors = _reduce_events([row for row in event_rows if isinstance(row, Mapping)])
    errors.extend(event_errors)
    if value.get("invocation_counts") != counts:
        errors.append("invocation_counts_mismatch")
    expected_invoked = counts["model_loads_attempted"] + counts["generation_calls_attempted"] > 0
    if value.get("model_invoked") is not expected_invoked:
        errors.append("model_invoked_mismatch")

    duration = value.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        errors.append("duration_invalid")
    else:
        for span in value.get("phase_spans") or []:
            if not isinstance(span, Mapping):
                errors.append("phase_span_invalid")
                continue
            end = span.get("end_s")
            if isinstance(end, (int, float)) and end > duration:
                errors.append(f"phase_span_exceeds_duration:{span.get('phase')}")

    base = (root or Path.cwd()).resolve()
    for row in value.get("receipt_sidecars") or []:
        if not isinstance(row, Mapping):
            errors.append("sidecar_reference_invalid")
            continue
        path_label = str(row.get("path") or "")
        if row.get("scope") not in SIDECAR_SCOPES:
            errors.append(f"sidecar_scope_invalid:{path_label}")
        path = Path(path_label)
        resolved = path if path.is_absolute() else base / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        if observed != row.get("sha256"):
            errors.append(f"sidecar_hash_mismatch:{path_label}")
    return list(dict.fromkeys(errors))
