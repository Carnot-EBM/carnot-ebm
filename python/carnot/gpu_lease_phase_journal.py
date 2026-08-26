"""Task-scoped GPU ownership and phase journals.

The lease uses a kernel lock for exclusion and a checksummed JSON journal for
evidence. The kernel releases the lock after a crash. The journal stays on
disk, so a later owner can prove why recovery was safe without signaling the
old PID.

Spec refs: REQ-INFRA-6633, SCENARIO-INFRA-6633-ATOMIC-RACE,
SCENARIO-INFRA-6633-INDEPENDENT-DEVICES,
SCENARIO-INFRA-6633-OWNER-AND-PHASES,
SCENARIO-INFRA-6633-FAIL-CLOSED, and
SCENARIO-INFRA-6633-CRASH-RECOVERY.
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


def canonical_json(value: Any) -> str:
    """Return stable JSON text for content hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value with the project prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
