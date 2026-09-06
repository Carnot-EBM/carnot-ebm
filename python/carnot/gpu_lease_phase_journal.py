"""Task-scoped GPU ownership and phase journals.

The lease uses a kernel lock for exclusion and a checksummed JSON journal for
evidence. The kernel releases the lock after a crash. The journal stays on
disk, so a later owner can prove why recovery was safe without signaling the
old PID.

Spec refs: REQ-INFRA-6633, SCENARIO-INFRA-6633-ATOMIC-RACE,
SCENARIO-INFRA-6633-INDEPENDENT-DEVICES,
SCENARIO-INFRA-6633-OWNER-AND-PHASES,
SCENARIO-INFRA-6633-FAIL-CLOSED, and
SCENARIO-INFRA-6633-CRASH-RECOVERY, REQ-INFRA-7078, and
SCENARIO-INFRA-7078-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from typing import Any
from datetime import UTC, datetime

from carnot.inference.llama_server_supervisor import parse_proc_stat


JsonDict = dict[str, Any]
SCHEMA = "carnot.gpu_lease_phase_journal.v1"
PHASES = (
    "preflight",
    "admitted",
    "loading",
    "resident",
    "inferencing",
    "unloading",
    "validating",
    "terminal_complete",
    "terminal_blocked",
)
TERMINAL_PHASES = frozenset({"terminal_complete", "terminal_blocked"})
COMPLETE_PHASE_SEQUENCE = (
    "preflight",
    "admitted",
    "loading",
    "resident",
    "inferencing",
    "unloading",
    "validating",
    "terminal_complete",
)
LEGACY_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "owner",
        "device_uuid",
        "expected_model",
        "acquired_monotonic_ns",
        "heartbeat_monotonic_ns",
        "expires_monotonic_ns",
        "ttl_ns",
        "phase",
        "phase_history",
        "vram_mb",
        "exit_evidence",
        "unload_evidence",
        "recovery",
        "released",
        "released_monotonic_ns",
        "lease_generation",
        "checksum",
    }
)
LEGACY_OWNER_FIELDS = frozenset(
    {"pid", "pid_start_ticks", "executable", "argv_digest", "token_digest"}
)
LEGACY_EVENT_FIELDS = frozenset(
    {
        "phase",
        "previous_phase",
        "previous_event_checksum",
        "monotonic_ns",
        "owner_token_digest",
        "details",
        "event_checksum",
    }
)
LEGACY_VRAM_FIELDS = frozenset({"before", "resident", "after"})
LEGACY_EXIT_FIELDS = frozenset({"exit_code", "observed_monotonic_ns"})
LEGACY_UNLOAD_FIELDS = frozenset({"required", "observed", "observed_monotonic_ns"})
LEGACY_RECOVERY_FIELDS = frozenset({"performed", "signals_sent"})
LEGACY_PERFORMED_RECOVERY_FIELDS = frozenset(
    {
        "performed",
        "reason",
        "previous_checksum",
        "previous_task_id",
        "previous_pid",
        "previous_pid_start_ticks",
        "signals_sent",
    }
)
LEGACY_MIGRATION_REASON = "legacy_same_schema_missing_lease_id"
ALLOWED_TRANSITIONS = {
    "preflight": frozenset({"admitted", "terminal_blocked"}),
    "admitted": frozenset({"loading", "terminal_blocked"}),
    "loading": frozenset({"resident", "terminal_blocked"}),
    "resident": frozenset({"inferencing", "unloading"}),
    "inferencing": frozenset({"unloading"}),
    "unloading": frozenset({"validating"}),
    "validating": frozenset(TERMINAL_PHASES),
    "terminal_complete": frozenset(),
    "terminal_blocked": frozenset(),
}


class LeaseError(RuntimeError):
    """Base error for a lease operation that must fail closed."""


class LeaseBusy(LeaseError):
    """The requested device already has a kernel-locked owner."""


class OwnershipError(LeaseError):
    """The caller does not match the journal owner."""


class TransitionError(LeaseError):
    """The requested phase change violates the ordered state machine."""


class JournalError(LeaseError):
    """The durable journal is missing, malformed, or changed."""


class LeaseExpired(LeaseError):
    """The owner missed its heartbeat deadline."""


class RecoveryError(LeaseError):
    """Recovery cannot prove that the recorded owner is gone."""


class MigrationBlocked(LeaseError):
    """Legacy evidence did not satisfy every non-destructive migration gate."""


def canonical_json(value: Any) -> str:
    """Return stable JSON text for content hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value with the project prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes so preservation checks do not depend on JSON parsing."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _without_checksum(value: Mapping[str, Any], field: str) -> JsonDict:
    return {key: item for key, item in value.items() if key != field}


def event_checksum(event: Mapping[str, Any]) -> str:
    """Hash one history event without its self-referential field."""

    return sha256_json(_without_checksum(event, "event_checksum"))


def journal_checksum(document: Mapping[str, Any]) -> str:
    """Hash a journal without its final self-referential field."""

    return sha256_json(_without_checksum(document, "checksum"))


def _device_key(device_uuid: str) -> str:
    return hashlib.sha256(device_uuid.encode("utf-8")).hexdigest()


def lock_path_for(runtime_dir: str | Path, device_uuid: str) -> Path:
    """Map a device UUID to one stable lock path without trusting path text."""

    return Path(runtime_dir) / f"device-{_device_key(device_uuid)}.lock"


def journal_path_for(runtime_dir: str | Path, device_uuid: str) -> Path:
    """Map a device UUID to its durable evidence path."""

    return Path(runtime_dir) / f"device-{_device_key(device_uuid)}.journal.json"


def preserved_source_path_for(runtime_dir: str | Path, source_sha256: str) -> Path:
    """Place one immutable legacy source at a path derived from its byte hash."""

    digest = source_sha256.removeprefix("sha256:")
    return Path(runtime_dir) / "recovery" / f"sha256-{digest}.legacy-journal.json"


def migration_receipt_path_for(runtime_dir: str | Path, source_sha256: str) -> Path:
    """Keep the migration receipt beside its content-addressed legacy source."""

    digest = source_sha256.removeprefix("sha256:")
    return Path(runtime_dir) / "recovery" / f"sha256-{digest}.migration-receipt.json"


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    replace: Callable[
        [
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
        ],
        None,
    ] = os.replace,
) -> None:
    """Publish complete JSON with file sync, atomic replace, and directory sync."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_bytes_atomic(path: str | Path, payload: bytes) -> None:
    """Preserve exact bytes with the same durable publication steps as JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists():
        if target.read_bytes() != payload:
            raise MigrationBlocked("preserved_source_hash_collision")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def proc_start_ticks(pid: int) -> int | None:
    """Read Linux PID start ticks, which distinguish PID reuse."""

    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(parse_proc_stat(text)["start_time_ticks"])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def process_start_matches(pid: int, start_ticks: int) -> bool:
    """Return true only while the same Linux process identity is live."""

    return proc_start_ticks(pid) == int(start_ticks)


def current_process_identity() -> JsonDict:
    """Bind the current PID, start time, executable, and arguments."""

    pid = os.getpid()
    start_ticks = proc_start_ticks(pid)
    if start_ticks is None:
        raise JournalError("pid_start_unavailable")
    try:
        executable = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        executable = sys.executable
    return {
        "pid": pid,
        "pid_start_ticks": start_ticks,
        "executable": executable,
        "argv_digest": sha256_json(list(sys.argv)),
    }


def _history_errors(history: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(history, list) or not history:
        return ["phase_history_missing"]
    previous_phase: str | None = None
    previous_checksum: str | None = None
    terminal_count = 0
    for index, event in enumerate(history):
        if not isinstance(event, Mapping):
            errors.append("phase_event_invalid")
            continue
        phase = str(event.get("phase", ""))
        if event.get("event_checksum") != event_checksum(event):
            errors.append("event_checksum_mismatch")
        if event.get("previous_event_checksum") != previous_checksum:
            errors.append("event_chain_mismatch")
        if index == 0:
            if phase != "preflight" or event.get("previous_phase") is not None:
                errors.append("initial_phase_invalid")
        elif event.get("previous_phase") != previous_phase or phase not in ALLOWED_TRANSITIONS.get(
            str(previous_phase), frozenset()
        ):
            errors.append("phase_history_transition_invalid")
        if phase in TERMINAL_PHASES:
            terminal_count += 1
        previous_phase = phase
        previous_checksum = str(event.get("event_checksum"))
    if terminal_count > 1:
        errors.append("second_terminal")
    return list(dict.fromkeys(errors))


def validate_journal_document(
    document: Mapping[str, Any],
    *,
    expected_pid: int | None = None,
    expected_pid_start_ticks: int | None = None,
    expected_device_uuid: str | None = None,
    expected_model: str | None = None,
    now_ns: int | None = None,
    check_freshness: bool = True,
) -> list[str]:
    """Return all structural, ownership, time, and unload failures."""

    errors: list[str] = []
    required = {
        "schema",
        "lease_id",
        "task_id",
        "owner",
        "device_uuid",
        "expected_model",
        "acquired_monotonic_ns",
        "heartbeat_monotonic_ns",
        "expires_monotonic_ns",
        "ttl_ns",
        "phase",
        "phase_history",
        "vram_mb",
        "exit_evidence",
        "unload_evidence",
        "recovery",
        "released",
        "checksum",
    }
    if missing := sorted(required - set(document)):
        errors.extend(f"missing_field:{field}" for field in missing)
        return errors
    if document.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if not str(document.get("lease_id", "")).startswith("lease:"):
        errors.append("lease_id_invalid")
    if document.get("checksum") != journal_checksum(document):
        errors.append("checksum_mismatch")
    phase = str(document.get("phase"))
    if phase not in PHASES:
        errors.append("phase_invalid")
    history = document.get("phase_history")
    errors.extend(_history_errors(history))
    if isinstance(history, list) and history and history[-1].get("phase") != phase:
        errors.append("current_phase_history_mismatch")

    owner = document.get("owner")
    owner = owner if isinstance(owner, Mapping) else {}
    pid = owner.get("pid")
    start_ticks = owner.get("pid_start_ticks")
    if not isinstance(pid, int) or pid <= 1:
        errors.append("pid_invalid")
    if not isinstance(start_ticks, int) or start_ticks < 0:
        errors.append("pid_start_invalid")
    if not str(owner.get("token_digest", "")).startswith("sha256:"):
        errors.append("token_digest_invalid")
    if expected_pid is not None and pid != expected_pid:
        errors.append("pid_mismatch")
    if expected_pid_start_ticks is not None and start_ticks != expected_pid_start_ticks:
        errors.append("pid_start_mismatch")
    if expected_device_uuid is not None and document.get("device_uuid") != expected_device_uuid:
        errors.append("device_mismatch")
    if expected_model is not None and document.get("expected_model") != expected_model:
        errors.append("model_mismatch")

    acquired = document.get("acquired_monotonic_ns")
    heartbeat = document.get("heartbeat_monotonic_ns")
    expires = document.get("expires_monotonic_ns")
    ttl_ns = document.get("ttl_ns")
    if not all(isinstance(value, int) for value in (acquired, heartbeat, expires, ttl_ns)):
        errors.append("monotonic_time_invalid")
    elif not (acquired <= heartbeat < expires and expires == heartbeat + ttl_ns):
        errors.append("monotonic_time_order_invalid")
    elif check_freshness:
        observed_now = time.monotonic_ns() if now_ns is None else int(now_ns)
        if observed_now > expires:
            errors.extend(["lease_expired", "stale_heartbeat"])

    phases = [event.get("phase") for event in history] if isinstance(history, list) else []
    resident_seen = "resident" in phases
    unload = document.get("unload_evidence")
    unload = unload if isinstance(unload, Mapping) else {}
    exit_evidence = document.get("exit_evidence")
    exit_evidence = exit_evidence if isinstance(exit_evidence, Mapping) else {}
    if resident_seen and unload.get("required") is not True:
        errors.append("unload_requirement_missing")
    if phase in TERMINAL_PHASES and resident_seen:
        vram = document.get("vram_mb")
        vram = vram if isinstance(vram, Mapping) else {}
        if unload.get("observed") is not True:
            errors.append("missing_unload_evidence")
        if exit_evidence.get("exit_code") is None:
            errors.append("exit_evidence_missing")
        if vram.get("resident") is None or vram.get("after") is None:
            errors.append("vram_evidence_missing")
    if document.get("released") is True and phase not in TERMINAL_PHASES:
        errors.append("nonterminal_release")
    return list(dict.fromkeys(errors))


def read_journal(path: str | Path) -> JsonDict:
    """Read one complete journal and reject malformed or changed content."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JournalError(f"journal_unreadable:{type(exc).__name__}") from exc
    if not isinstance(payload, Mapping):
        raise JournalError("journal_not_object")
    document = dict(payload)
    errors = validate_journal_document(document, check_freshness=False)
    if errors:
        raise JournalError(",".join(errors))
    return document


def _mapping_keys_match(value: Any, expected: frozenset[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def legacy_journal_errors(
    document: Mapping[str, Any], *, expected_device_uuid: str | None = None
) -> list[str]:
    """Validate only the exact pre-lease-ID journal shape found on disk.

    The normal reader must not accept these bytes. This separate validator is
    narrow so an unrelated malformed v1 document cannot gain ownership status.
    """

    errors: list[str] = []
    if set(document) != LEGACY_TOP_LEVEL_FIELDS:
        return ["legacy_top_level_fingerprint_mismatch"]
    if document.get("schema") != SCHEMA:
        errors.append("legacy_schema_mismatch")
    if document.get("checksum") != journal_checksum(document):
        errors.append("legacy_checksum_mismatch")
    if expected_device_uuid is not None and document.get("device_uuid") != expected_device_uuid:
        errors.append("legacy_device_mismatch")
    if not all(
        isinstance(document.get(field), str) and bool(document.get(field))
        for field in ("task_id", "device_uuid", "expected_model")
    ):
        errors.append("legacy_identity_field_invalid")

    owner = document.get("owner")
    if not _mapping_keys_match(owner, LEGACY_OWNER_FIELDS):
        errors.append("legacy_owner_fingerprint_mismatch")
        owner = {}
    pid = owner.get("pid")
    start_ticks = owner.get("pid_start_ticks")
    if not isinstance(pid, int) or pid <= 1:
        errors.append("legacy_pid_invalid")
    if not isinstance(start_ticks, int) or start_ticks < 0:
        errors.append("legacy_pid_start_invalid")
    if not str(owner.get("executable", "")):
        errors.append("legacy_executable_invalid")
    for digest_field in ("argv_digest", "token_digest"):
        if not str(owner.get(digest_field, "")).startswith("sha256:"):
            errors.append(f"legacy_{digest_field}_invalid")

    if not _mapping_keys_match(document.get("vram_mb"), LEGACY_VRAM_FIELDS):
        errors.append("legacy_vram_fingerprint_mismatch")
    if not _mapping_keys_match(document.get("exit_evidence"), LEGACY_EXIT_FIELDS):
        errors.append("legacy_exit_fingerprint_mismatch")
    if not _mapping_keys_match(document.get("unload_evidence"), LEGACY_UNLOAD_FIELDS):
        errors.append("legacy_unload_fingerprint_mismatch")
    recovery = document.get("recovery")
    recovery_fields = (
        LEGACY_PERFORMED_RECOVERY_FIELDS
        if isinstance(recovery, Mapping) and recovery.get("performed") is True
        else LEGACY_RECOVERY_FIELDS
    )
    if not _mapping_keys_match(recovery, recovery_fields):
        errors.append("legacy_recovery_fingerprint_mismatch")
    elif recovery.get("signals_sent") != []:
        errors.append("legacy_recovery_signal_evidence_invalid")

    history = document.get("phase_history")
    errors.extend(f"legacy_{error}" for error in _history_errors(history))
    event_times: list[int] = []
    if isinstance(history, list):
        for event in history:
            if not _mapping_keys_match(event, LEGACY_EVENT_FIELDS):
                errors.append("legacy_event_fingerprint_mismatch")
                continue
            if not isinstance(event.get("details"), Mapping):
                errors.append("legacy_event_details_invalid")
            event_time = event.get("monotonic_ns")
            if not isinstance(event_time, int):
                errors.append("legacy_event_time_invalid")
            else:
                event_times.append(event_time)
            if event.get("owner_token_digest") != owner.get("token_digest"):
                errors.append("legacy_event_owner_mismatch")
    if event_times != sorted(event_times) or len(set(event_times)) != len(event_times):
        errors.append("legacy_event_time_order_invalid")

    acquired = document.get("acquired_monotonic_ns")
    heartbeat = document.get("heartbeat_monotonic_ns")
    expires = document.get("expires_monotonic_ns")
    ttl_ns = document.get("ttl_ns")
    if not all(isinstance(value, int) for value in (acquired, heartbeat, expires, ttl_ns)):
        errors.append("legacy_monotonic_time_invalid")
    elif not (acquired <= heartbeat < expires and expires == heartbeat + ttl_ns):
        errors.append("legacy_monotonic_time_order_invalid")
    if event_times and event_times[0] != acquired:
        errors.append("legacy_acquisition_event_mismatch")

    if document.get("released") is not True:
        errors.append("legacy_not_released")
    phase = document.get("phase")
    if phase not in TERMINAL_PHASES:
        errors.append("legacy_nonterminal_phase")
    if isinstance(history, list) and history and history[-1].get("phase") != phase:
        errors.append("legacy_current_phase_history_mismatch")
    released_ns = document.get("released_monotonic_ns")
    if not isinstance(released_ns, int) or (event_times and released_ns < event_times[-1]):
        errors.append("legacy_release_time_invalid")
    if not isinstance(document.get("lease_generation"), int) or document["lease_generation"] < 1:
        errors.append("legacy_generation_invalid")

    surrogate = deepcopy(dict(document))
    surrogate["lease_id"] = "lease:legacy-validation-surrogate"
    surrogate["checksum"] = journal_checksum(surrogate)
    errors.extend(
        f"legacy_{error}"
        for error in validate_journal_document(surrogate, check_freshness=False)
        if error not in {"checksum_mismatch", "lease_id_invalid"}
    )
    return list(dict.fromkeys(errors))


def _load_object_bytes(payload: bytes) -> JsonDict:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationBlocked(f"journal_unreadable:{type(exc).__name__}") from exc
    if not isinstance(value, Mapping):
        raise MigrationBlocked("journal_not_object")
    return dict(value)


def migration_receipt_checksum(receipt: Mapping[str, Any]) -> str:
    """Bind receipt fields without making the checksum self-referential."""

    return sha256_json(_without_checksum(receipt, "receipt_checksum"))


def _json_file_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validate_migrated_current(
    *, runtime_dir: Path, journal_path: Path, document: JsonDict, current_bytes: bytes
) -> JsonDict:
    recovery = document.get("recovery")
    migration = recovery.get("legacy_migration") if isinstance(recovery, Mapping) else None
    if not isinstance(migration, Mapping):
        return {
            "action": "current_noop",
            "idempotent": True,
            "migrated": False,
            "device_uuid": document["device_uuid"],
            "lease_id": document["lease_id"],
            "journal_path": str(journal_path),
            "target_sha256": sha256_bytes(current_bytes),
            "signals_sent": [],
            "files_removed": [],
        }

    source_sha256 = str(migration.get("source_sha256", ""))
    preserved_path = preserved_source_path_for(runtime_dir, source_sha256)
    receipt_path = migration_receipt_path_for(runtime_dir, source_sha256)
    try:
        preserved_bytes = preserved_path.read_bytes()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationBlocked(f"migration_receipt_unreadable:{type(exc).__name__}") from exc
    if not isinstance(receipt, Mapping):
        raise MigrationBlocked("migration_receipt_not_object")
    checks = (
        sha256_bytes(preserved_bytes) == source_sha256,
        receipt.get("receipt_checksum") == migration_receipt_checksum(receipt),
        receipt.get("source_sha256") == source_sha256,
        receipt.get("target_sha256") == sha256_bytes(current_bytes),
        receipt.get("device_uuid") == document.get("device_uuid"),
        receipt.get("generated_lease_id") == document.get("lease_id"),
        receipt.get("old_task_id") == migration.get("old_task_id"),
        receipt.get("reason") == LEGACY_MIGRATION_REASON,
        receipt.get("preserved_path") == str(preserved_path),
        receipt.get("journal_path") == str(journal_path),
    )
    if not all(checks):
        raise MigrationBlocked("migration_receipt_invalid")
    return {
        **dict(receipt),
        "action": "idempotent_noop",
        "idempotent": True,
        "migrated": False,
        "receipt_path": str(receipt_path),
        "signals_sent": [],
        "files_removed": [],
    }


def migrate_legacy_journal(
    *,
    runtime_dir: str | Path,
    device_uuid: str,
    process_match: Callable[[int, int], bool] = process_start_matches,
    lease_id_factory: Callable[[], str] | None = None,
    migration_monotonic_ns: int | None = None,
    migration_utc: str | None = None,
    journal_replace: Callable[
        [
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
            str | bytes | os.PathLike[str] | os.PathLike[bytes],
        ],
        None,
    ] = os.replace,
) -> JsonDict:
    """Migrate one exact legacy journal after identity and kernel-lock checks."""

    runtime = Path(runtime_dir)
    journal_path = journal_path_for(runtime, device_uuid)
    try:
        source_bytes = journal_path.read_bytes()
    except OSError as exc:
        raise MigrationBlocked(f"journal_unreadable:{type(exc).__name__}") from exc
    source_document = _load_object_bytes(source_bytes)
    if "lease_id" in source_document:
        current_errors = validate_journal_document(source_document, check_freshness=False)
        if current_errors:
            raise MigrationBlocked("current_journal_invalid:" + ",".join(current_errors))
        return _validate_migrated_current(
            runtime_dir=runtime,
            journal_path=journal_path,
            document=source_document,
            current_bytes=source_bytes,
        )

    errors = legacy_journal_errors(source_document, expected_device_uuid=device_uuid)
    if errors:
        raise MigrationBlocked(",".join(errors))
    owner = source_document["owner"]
    if process_match(int(owner["pid"]), int(owner["pid_start_ticks"])):
        raise MigrationBlocked("recorded_owner_still_live")

    lock_path = lock_path_for(runtime, device_uuid)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MigrationBlocked("device_lock_held") from exc

        try:
            current_bytes = journal_path.read_bytes()
        except OSError as exc:
            raise MigrationBlocked(f"journal_reread_failed:{type(exc).__name__}") from exc
        if current_bytes != source_bytes:
            raise MigrationBlocked("journal_changed_after_precheck")
        current_document = _load_object_bytes(current_bytes)
        errors = legacy_journal_errors(current_document, expected_device_uuid=device_uuid)
        if errors:
            raise MigrationBlocked(",".join(errors))
        current_owner = current_document["owner"]
        if process_match(int(current_owner["pid"]), int(current_owner["pid_start_ticks"])):
            raise MigrationBlocked("recorded_owner_became_live")

        source_sha256 = sha256_bytes(source_bytes)
        preserved_path = preserved_source_path_for(runtime, source_sha256)
        receipt_path = migration_receipt_path_for(runtime, source_sha256)
        write_bytes_atomic(preserved_path, source_bytes)

        observed_ns = (
            time.monotonic_ns() if migration_monotonic_ns is None else int(migration_monotonic_ns)
        )
        observed_utc = migration_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        generated_lease_id = (
            lease_id_factory()
            if lease_id_factory is not None
            else "lease:migrated:" + secrets.token_hex(32)
        )
        if not str(generated_lease_id).startswith("lease:"):
            raise MigrationBlocked("generated_lease_id_invalid")

        target = deepcopy(current_document)
        target["lease_id"] = generated_lease_id
        target["recovery"] = {
            "performed": True,
            "reason": LEGACY_MIGRATION_REASON,
            "signals_sent": [],
            "legacy_recovery": deepcopy(current_document["recovery"]),
            "legacy_migration": {
                "source_sha256": source_sha256,
                "old_task_id": current_document["task_id"],
                "old_checksum": current_document["checksum"],
                "preserved_path": str(preserved_path),
                "receipt_path": str(receipt_path),
                "migration_monotonic_ns": observed_ns,
                "migration_utc": observed_utc,
                "reason": LEGACY_MIGRATION_REASON,
            },
        }
        target["checksum"] = journal_checksum(target)
        target_errors = validate_journal_document(target, check_freshness=False)
        if target_errors:
            raise MigrationBlocked("target_journal_invalid:" + ",".join(target_errors))
        target_sha256 = sha256_bytes(_json_file_bytes(target))
        receipt: JsonDict = {
            "schema": "carnot.gpu_lease_legacy_migration_receipt.v1",
            "source_sha256": source_sha256,
            "target_sha256": target_sha256,
            "device_uuid": device_uuid,
            "old_task_id": current_document["task_id"],
            "generated_lease_id": generated_lease_id,
            "reason": LEGACY_MIGRATION_REASON,
            "journal_path": str(journal_path),
            "preserved_path": str(preserved_path),
            "migration_monotonic_ns": observed_ns,
            "migration_utc": observed_utc,
            "legacy_acquired_monotonic_ns": current_document["acquired_monotonic_ns"],
            "legacy_released_monotonic_ns": current_document["released_monotonic_ns"],
            "signals_sent": [],
            "files_removed": [],
        }
        receipt["receipt_checksum"] = migration_receipt_checksum(receipt)
        write_json_atomic(receipt_path, receipt)
        try:
            write_json_atomic(journal_path, target, replace=journal_replace)
        except OSError as exc:
            raise MigrationBlocked(f"atomic_publish_failed:{type(exc).__name__}:{exc}") from exc
        if sha256_bytes(journal_path.read_bytes()) != target_sha256:
            raise MigrationBlocked("atomic_publish_hash_mismatch")
        return {
            **receipt,
            "action": "migrated",
            "idempotent": False,
            "migrated": True,
            "receipt_path": str(receipt_path),
            "lock_path": str(lock_path),
            "lock_acquired": True,
        }
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _phase_event(
    *,
    phase: str,
    previous_phase: str | None,
    previous_event_checksum: str | None,
    monotonic_ns: int,
    token_digest: str,
    details: Mapping[str, Any] | None = None,
) -> JsonDict:
    event: JsonDict = {
        "phase": phase,
        "previous_phase": previous_phase,
        "previous_event_checksum": previous_event_checksum,
        "monotonic_ns": int(monotonic_ns),
        "owner_token_digest": token_digest,
        "details": dict(details or {}),
    }
    event["event_checksum"] = event_checksum(event)
    return event


class GpuLease:
    """Hold one device lock and publish owner-bound phase evidence."""

    def __init__(
        self,
        *,
        lock_fd: int,
        lock_path: Path,
        journal_path: Path,
        token: str,
        document: JsonDict,
    ) -> None:
        self._lock_fd = lock_fd
        self.lock_path = lock_path
        self.journal_path = journal_path
        self._token = token
        self.document = document
        self.lease_id = str(document["lease_id"])
        owner = document["owner"]
        self.pid = int(owner["pid"])
        self.pid_start_ticks = int(owner["pid_start_ticks"])
        self.device_uuid = str(document["device_uuid"])
        self.expected_model = str(document["expected_model"])

    @classmethod
    def acquire(
        cls,
        *,
        runtime_dir: str | Path,
        task_id: str,
        device_uuid: str,
        expected_model: str,
        vram_before_mb: int,
        ttl_s: float = 30.0,
    ) -> GpuLease:
        """Atomically acquire one device and recover only after owner death."""

        if not task_id or not device_uuid or not expected_model:
            raise ValueError("task_device_and_model_required")
        ttl_ns = int(float(ttl_s) * 1_000_000_000)
        if ttl_ns <= 0:
            raise ValueError("ttl_must_be_positive")
        runtime = Path(runtime_dir)
        runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = lock_path_for(runtime, device_uuid)
        journal_path = journal_path_for(runtime, device_uuid)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(lock_fd)
            raise LeaseBusy(f"device_busy:{device_uuid}") from exc

        try:
            previous: JsonDict | None = None
            recovery: JsonDict = {"performed": False, "signals_sent": []}
            if journal_path.exists():
                previous = read_journal(journal_path)
                if previous.get("released") is not True:
                    old_owner = previous["owner"]
                    if process_start_matches(
                        int(old_owner["pid"]), int(old_owner["pid_start_ticks"])
                    ):
                        raise RecoveryError("recorded_owner_still_live")
                    recovery = {
                        "performed": True,
                        "reason": "recorded_owner_absent_or_pid_reused",
                        "previous_checksum": previous["checksum"],
                        "previous_task_id": previous["task_id"],
                        "previous_pid": old_owner["pid"],
                        "previous_pid_start_ticks": old_owner["pid_start_ticks"],
                        "signals_sent": [],
                    }

            identity = current_process_identity()
            token = secrets.token_urlsafe(32)
            token_digest = sha256_json(token)
            now_ns = time.monotonic_ns()
            lease_id = (
                "lease:"
                + hashlib.sha256(
                    canonical_json(
                        {
                            "task_id": task_id,
                            "device_uuid": device_uuid,
                            "pid": identity["pid"],
                            "pid_start_ticks": identity["pid_start_ticks"],
                            "token_digest": token_digest,
                        }
                    ).encode("utf-8")
                ).hexdigest()
            )
            first_event = _phase_event(
                phase="preflight",
                previous_phase=None,
                previous_event_checksum=None,
                monotonic_ns=now_ns,
                token_digest=token_digest,
                details={"recovery_performed": recovery["performed"]},
            )
            document: JsonDict = {
                "schema": SCHEMA,
                "lease_id": lease_id,
                "task_id": task_id,
                "owner": {**identity, "token_digest": token_digest},
                "device_uuid": device_uuid,
                "expected_model": expected_model,
                "acquired_monotonic_ns": now_ns,
                "heartbeat_monotonic_ns": now_ns,
                "expires_monotonic_ns": now_ns + ttl_ns,
                "ttl_ns": ttl_ns,
                "phase": "preflight",
                "phase_history": [first_event],
                "vram_mb": {
                    "before": int(vram_before_mb),
                    "resident": None,
                    "after": None,
                },
                "exit_evidence": {"exit_code": None, "observed_monotonic_ns": None},
                "unload_evidence": {
                    "required": False,
                    "observed": False,
                    "observed_monotonic_ns": None,
                },
                "recovery": recovery,
                "released": False,
                "released_monotonic_ns": None,
                "lease_generation": 1
                if previous is None
                else int(previous.get("lease_generation", 0)) + 1,
            }
            document["checksum"] = journal_checksum(document)
            write_json_atomic(journal_path, document)
            os.ftruncate(lock_fd, 0)
            os.write(
                lock_fd,
                canonical_json(
                    {
                        "task_id": task_id,
                        "pid": identity["pid"],
                        "pid_start_ticks": identity["pid_start_ticks"],
                        "device_uuid": device_uuid,
                    }
                ).encode("utf-8"),
            )
            os.fsync(lock_fd)
            return cls(
                lock_fd=lock_fd,
                lock_path=lock_path,
                journal_path=journal_path,
                token=token,
                document=document,
            )
        except Exception:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            raise

    def owner_receipt(self) -> JsonDict:
        """Return redacted ownership evidence without exposing the token."""

        owner = self.document["owner"]
        return {
            "lease_id": self.lease_id,
            "task_id": self.document["task_id"],
            "device_uuid": self.device_uuid,
            "pid": self.pid,
            "pid_start_ticks": self.pid_start_ticks,
            "executable": owner["executable"],
            "argv_digest": owner["argv_digest"],
            "expected_model": self.expected_model,
            "token_digest": owner["token_digest"],
            "token_opaque": True,
            "token_length": len(self._token),
            "acquired_monotonic_ns": self.document["acquired_monotonic_ns"],
            "expires_monotonic_ns": self.document["expires_monotonic_ns"],
            "recovery": deepcopy(self.document["recovery"]),
            "signals_sent": [],
        }

    def _refresh(self) -> None:
        disk = read_journal(self.journal_path)
        if disk["checksum"] != self.document["checksum"]:
            raise JournalError("journal_changed_by_other_writer")
        self.document = disk

    def _verify_owner(
        self,
        *,
        token: str | None,
        device_uuid: str | None,
        expected_model: str | None,
        pid_start_ticks: int | None,
    ) -> None:
        if (
            sha256_json(self._token if token is None else token)
            != self.document["owner"]["token_digest"]
        ):
            raise OwnershipError("wrong_token")
        if (self.device_uuid if device_uuid is None else device_uuid) != self.device_uuid:
            raise OwnershipError("wrong_device")
        if (
            self.expected_model if expected_model is None else expected_model
        ) != self.expected_model:
            raise OwnershipError("wrong_model")
        supplied_start = self.pid_start_ticks if pid_start_ticks is None else int(pid_start_ticks)
        if supplied_start != self.pid_start_ticks or not process_start_matches(
            self.pid, self.pid_start_ticks
        ):
            raise OwnershipError("pid_start_mismatch")
        if os.getpid() != self.pid:
            raise OwnershipError("wrong_pid")

    def _ensure_fresh(self, now_ns: int) -> None:
        if int(now_ns) > int(self.document["expires_monotonic_ns"]):
            raise LeaseExpired("lease_expired_stale_heartbeat")

    def _commit(self) -> None:
        self.document["checksum"] = journal_checksum(self.document)
        write_json_atomic(self.journal_path, self.document)

    def heartbeat(
        self,
        *,
        token: str | None = None,
        device_uuid: str | None = None,
        expected_model: str | None = None,
        pid_start_ticks: int | None = None,
        now_ns: int | None = None,
    ) -> JsonDict:
        """Extend expiry after complete owner verification."""

        self._refresh()
        self._verify_owner(
            token=token,
            device_uuid=device_uuid,
            expected_model=expected_model,
            pid_start_ticks=pid_start_ticks,
        )
        observed_now = time.monotonic_ns() if now_ns is None else int(now_ns)
        self._ensure_fresh(observed_now)
        if self.document["phase"] in TERMINAL_PHASES:
            raise TransitionError("terminal_already_set")
        self.document["heartbeat_monotonic_ns"] = observed_now
        self.document["expires_monotonic_ns"] = observed_now + self.document["ttl_ns"]
        self._commit()
        return {
            "owner_verified": True,
            "heartbeat_monotonic_ns": observed_now,
            "expires_monotonic_ns": self.document["expires_monotonic_ns"],
            "checksum": self.document["checksum"],
        }

    def transition(
        self,
        phase: str,
        *,
        token: str | None = None,
        device_uuid: str | None = None,
        expected_model: str | None = None,
        pid_start_ticks: int | None = None,
        now_ns: int | None = None,
        vram_mb: int | None = None,
        exit_code: int | None = None,
        unload_observed: bool | None = None,
    ) -> JsonDict:
        """Advance one allowed phase and bind phase-specific evidence."""

        self._refresh()
        self._verify_owner(
            token=token,
            device_uuid=device_uuid,
            expected_model=expected_model,
            pid_start_ticks=pid_start_ticks,
        )
        observed_now = time.monotonic_ns() if now_ns is None else int(now_ns)
        self._ensure_fresh(observed_now)
        current = str(self.document["phase"])
        if current in TERMINAL_PHASES:
            raise TransitionError("terminal_already_set")
        if phase not in ALLOWED_TRANSITIONS[current]:
            raise TransitionError(f"transition_not_allowed:{current}->{phase}")
        details: JsonDict = {}
        if phase == "resident":
            if vram_mb is None:
                raise TransitionError("resident_vram_missing")
            self.document["vram_mb"]["resident"] = int(vram_mb)
            self.document["unload_evidence"]["required"] = True
            details["vram_resident_mb"] = int(vram_mb)
        if phase == "validating":
            if self.document["unload_evidence"]["required"] is True and unload_observed is not True:
                raise TransitionError("missing_unload_evidence")
            if vram_mb is None or exit_code is None:
                raise TransitionError("validation_exit_or_vram_missing")
            self.document["vram_mb"]["after"] = int(vram_mb)
            self.document["exit_evidence"] = {
                "exit_code": int(exit_code),
                "observed_monotonic_ns": observed_now,
            }
            self.document["unload_evidence"]["observed"] = bool(unload_observed)
            self.document["unload_evidence"]["observed_monotonic_ns"] = observed_now
            details.update(
                {
                    "vram_after_mb": int(vram_mb),
                    "exit_code": int(exit_code),
                    "unload_observed": bool(unload_observed),
                }
            )
        previous_event_checksum = self.document["phase_history"][-1]["event_checksum"]
        event = _phase_event(
            phase=phase,
            previous_phase=current,
            previous_event_checksum=previous_event_checksum,
            monotonic_ns=observed_now,
            token_digest=self.document["owner"]["token_digest"],
            details=details,
        )
        self.document["phase"] = phase
        self.document["phase_history"].append(event)
        self._commit()
        return {
            "from_phase": current,
            "to_phase": phase,
            "accepted": True,
            "event_checksum": event["event_checksum"],
            "journal_checksum": self.document["checksum"],
        }

    def release(self, *, token: str | None = None) -> JsonDict:
        """Release only after an owner-bound terminal journal is durable."""

        self._refresh()
        self._verify_owner(
            token=token,
            device_uuid=None,
            expected_model=None,
            pid_start_ticks=None,
        )
        if self.document["phase"] not in TERMINAL_PHASES:
            raise TransitionError("release_requires_terminal_phase")
        self.document["released"] = True
        self.document["released_monotonic_ns"] = time.monotonic_ns()
        self._commit()
        receipt = {
            "lease_id": self.lease_id,
            "released": True,
            "phase": self.document["phase"],
            "device_uuid": self.device_uuid,
            "pid": self.pid,
            "pid_start_ticks": self.pid_start_ticks,
            "checksum": self.document["checksum"],
            "signals_sent": [],
        }
        self.close()
        return receipt

    def close(self) -> None:
        """Drop this process's kernel lock without changing the journal."""

        if self._lock_fd < 0:
            return
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_fd)
            self._lock_fd = -1

    def __del__(self) -> None:  # pragma: no cover - interpreter cleanup only.
        self.close()


def _complete_fixture(lease: GpuLease) -> None:
    lease.transition("admitted")
    lease.transition("loading")
    lease.transition("resident", vram_mb=1028)
    lease.transition("inferencing")
    lease.transition("unloading")
    lease.transition("validating", vram_mb=4, exit_code=0, unload_observed=True)
    lease.transition("terminal_complete")


def fixture_worker_main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run one bounded worker used by subprocess ownership fixtures."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--expected-model", default="fixture/model.gguf")
    parser.add_argument(
        "--behavior",
        choices=("complete", "hold_complete", "crash", "stale", "recover_complete"),
        required=True,
    )
    parser.add_argument("--hold-s", type=float, default=0.0)
    parser.add_argument("--ttl-s", type=float, default=5.0)
    parser.add_argument("--exit-code", type=int, default=23)
    args = parser.parse_args(argv)
    try:
        lease = GpuLease.acquire(
            runtime_dir=args.runtime_dir,
            task_id=args.task_id,
            device_uuid=args.device_uuid,
            expected_model=args.expected_model,
            vram_before_mb=4,
            ttl_s=args.ttl_s,
        )
    except LeaseBusy:
        print(json.dumps({"outcome": "busy", "signals_sent": []}), flush=True)
        return 3
    except (JournalError, RecoveryError) as exc:
        print(
            json.dumps({"outcome": "fail_closed", "reason": str(exc), "signals_sent": []}),
            flush=True,
        )
        return 4
    print(
        json.dumps(
            {
                "outcome": "acquired",
                "owner": lease.owner_receipt(),
                "journal_path": str(lease.journal_path),
            }
        ),
        flush=True,
    )
    if args.hold_s:
        time.sleep(args.hold_s)
    if args.behavior in {"crash", "stale"}:
        os._exit(args.exit_code)
    _complete_fixture(lease)
    released = lease.release()
    print(json.dumps({"outcome": "released", "release": released}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fixture_worker_main())
