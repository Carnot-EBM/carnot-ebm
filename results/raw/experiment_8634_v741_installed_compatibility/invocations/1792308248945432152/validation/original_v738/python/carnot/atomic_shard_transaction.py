"""Crash-safe shard journal for row-producing experiments.

The helper keeps resumable shard data away from the conductor-visible result
JSON. A producer can restart from verified shards, then expose one terminal
artifact only after every planned unit has a closed disposition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time
from typing import Any


TRANSACTION_SCHEMA = "carnot.atomic_shard_transaction.v1"
TERMINAL_PREFIXES = (
    "complete",
    "success",
    "blocked",
    "disqualified",
    "skipped",
    "retired",
    "flagged",
)
NONTERMINAL_PREFIXES = (
    "running",
    "in_progress",
    "bootstrap",
    "running_bootstrap",
    "bootstrap_only",
    "partial",
)


class TransactionError(RuntimeError):
    """Base class for transaction failures that callers must close explicitly."""


class CrashInjected(TransactionError):
    """Raised by tests and experiments at a named crash point."""


class ConcurrentWriterError(TransactionError):
    """Raised when a live lock proves another writer owns the transaction."""


class DuplicateUnitError(TransactionError):
    """Raised when one unit ID maps to two different terminal contents."""


class MissingTerminalUnitError(TransactionError):
    """Raised when finalization sees planned units without terminal rows."""


class CorruptShardError(TransactionError):
    """Raised when a journal record or shard hash does not verify."""

    def __init__(self, message: str, row: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.row = dict(row or {})


class InsufficientDiskError(TransactionError):
    """Raised before writes when available disk space is below the guard."""


@dataclass
class CrashPlan:
    """One-shot crash injector for deterministic recovery tests."""

    stages: set[str] = field(default_factory=set)
    fired: list[str] = field(default_factory=list)

    @classmethod
    def once(cls, stage: str) -> "CrashPlan":
        return cls(stages={stage})

    def maybe_crash(self, stage: str) -> None:
        if stage in self.stages and stage not in self.fired:
            self.fired.append(stage)
            raise CrashInjected(stage)


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes used for shard and journal hashes."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return a project-style SHA-256 digest string."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON bytes."""

    return sha256_bytes(canonical_json_bytes(value))


def nonterminal_status_reason(payload: Mapping[str, Any]) -> str | None:
    """Return why a final payload is nonterminal, or ``None`` when closed."""

    for key in ("status", "honest_verdict"):
        raw = str(payload.get(key) or "").strip().lower().replace("-", "_")
        if not raw:
            return f"{key}=missing"
        if raw.startswith(NONTERMINAL_PREFIXES):
            return f"{key}={payload.get(key)}"
    status = str(payload.get("status") or "").strip().lower().replace("-", "_")
    verdict = str(payload.get("honest_verdict") or "").strip().lower().replace("-", "_")
    if not status.startswith(TERMINAL_PREFIXES):
        return f"status={payload.get('status')}"
    if not verdict.startswith(TERMINAL_PREFIXES):
        return f"honest_verdict={payload.get('honest_verdict')}"
    return None


def _record_hash(record: Mapping[str, Any]) -> str:
    base = {key: value for key, value in record.items() if key != "record_hash"}
    return sha256_json(base)


def _read_json_line(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):  # pragma: no cover - defensive malformed file guard.
        raise CorruptShardError("journal record is not an object")
    return value


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - depends on another user's process.
        return True
    return True


class AtomicShardTransaction:
    """A lock-protected transaction with content shards and a hashed journal."""

    def __init__(
        self,
        *,
        work_dir: Path | str,
        final_path: Path | str,
        transaction_id: str,
        crash_plan: CrashPlan | None = None,
        stale_lock_s: float = 3600.0,
        min_free_bytes: int = 0,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.final_path = Path(final_path)
        self.transaction_id = transaction_id
        self.crash_plan = crash_plan or CrashPlan()
        self.stale_lock_s = stale_lock_s
        self.min_free_bytes = min_free_bytes
        self.shard_dir = self.work_dir / "shards"
        self.quarantine_dir = self.work_dir / "quarantine"
        self.journal_path = self.work_dir / "journal.jsonl"
        self.lock_path = self.work_dir / "LOCK"
        self.lock_receipt: dict[str, Any] = {}
        self._lock_fd: int | None = None

    def __enter__(self) -> "AtomicShardTransaction":
        self.begin()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def begin(self) -> "AtomicShardTransaction":
        """Create directories and acquire the transaction lock."""

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        return self

    def close(self) -> None:
        """Release the lock owned by this instance."""

        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        if self.lock_path.exists() and self.lock_receipt.get("lock_owner_pid") == os.getpid():
            try:
                self.lock_path.unlink()
            except OSError:  # pragma: no cover - best effort cleanup only.
                pass

    def _acquire_lock(self) -> None:
        stale_recovered = False
        prior: dict[str, Any] | None = None
        if self.lock_path.exists():
            prior = self._read_lock()
            if self._lock_is_stale(prior):
                self.lock_path.unlink()
                stale_recovered = True
            else:
                raise ConcurrentWriterError(f"active lock: {self.lock_path}")
        try:
            fd = os.open(str(self.lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:  # pragma: no cover - race window guard.
            raise ConcurrentWriterError(f"active lock: {self.lock_path}") from exc
        row = {
            "schema": TRANSACTION_SCHEMA,
            "transaction_id": self.transaction_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_unix_s": time.time(),
        }
        os.write(fd, canonical_json_bytes(row))
        os.fsync(fd)
        self._lock_fd = fd
        self.lock_receipt = {
            "lock_path": str(self.lock_path),
            "lock_owner_pid": os.getpid(),
            "stale_lock_recovered": stale_recovered,
            "prior_lock": prior,
        }

    def _read_lock(self) -> dict[str, Any]:
        try:
            value = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - malformed stale guard.
            return {"pid": None, "malformed": True}
        return value if isinstance(value, dict) else {"pid": None, "malformed": True}

    def _lock_is_stale(self, row: Mapping[str, Any] | None) -> bool:
        age = time.time() - self.lock_path.stat().st_mtime
        pid = row.get("pid") if isinstance(row, Mapping) else None
        pid_dead = not isinstance(pid, int) or not _pid_alive(pid)
        return age >= self.stale_lock_s and pid_dead

    def plan_units(self, unit_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Append planned-unit records for units not already planned."""

        existing = {
            row["unit_id"]
            for row in self.read_journal()
            if row.get("record_type") == "planned_unit"
        }
        receipts: list[dict[str, Any]] = []
        for unit_id in unit_ids:
            if not unit_id:  # pragma: no cover - caller contract guard.
                raise ValueError("unit_id must be non-empty")
            if unit_id in existing:
                continue
            record = {
                "schema": TRANSACTION_SCHEMA,
                "transaction_id": self.transaction_id,
                "record_type": "planned_unit",
                "unit_id": unit_id,
                "recorded_unix_s": time.time(),
            }
            receipts.append(self._append_journal_record(record))
            existing.add(unit_id)
        return receipts

    def write_terminal_unit(
        self,
        unit_id: str,
        payload: Mapping[str, Any],
        *,
        disposition: str = "success",
    ) -> dict[str, Any]:
        """Write one terminal unit shard and its journal record."""

        data = canonical_json_bytes(payload)
        shard_hash = sha256_bytes(data)
        state = self.resume_state()
        existing = state["terminal_units"].get(unit_id)
        if existing is not None:
            if existing["shard_hash"] == shard_hash and existing["disposition"] == disposition:
                return {
                    "unit_id": unit_id,
                    "disposition": disposition,
                    "shard_hash": shard_hash,
                    "shard_path": str(self._shard_path(shard_hash)),
                    "idempotent": True,
                    "reused_existing_shard": True,
                }
            raise DuplicateUnitError(f"unit_id {unit_id} already has different content")
        self._check_disk(len(data))
        self.crash_plan.maybe_crash("before_shard_write")
        shard_path = self._shard_path(shard_hash)
        reused = False
        if shard_path.exists():
            self._verify_shard(shard_hash)
            reused = True
        else:
            self._write_file_atomically(shard_path, data)
        self.crash_plan.maybe_crash("after_shard_write")
        record = {
            "schema": TRANSACTION_SCHEMA,
            "transaction_id": self.transaction_id,
            "record_type": "terminal_unit",
            "unit_id": unit_id,
            "disposition": disposition,
            "shard_hash": shard_hash,
            "shard_path": str(shard_path),
            "recorded_unix_s": time.time(),
        }
        self.crash_plan.maybe_crash("during_journal_update")
        written = self._append_journal_record(record)
        return {
            "unit_id": unit_id,
            "disposition": disposition,
            "shard_hash": shard_hash,
            "shard_path": str(shard_path),
            "record_hash": written["record_hash"],
            "idempotent": False,
            "reused_existing_shard": reused,
        }

    def read_journal(self) -> list[dict[str, Any]]:
        """Read and verify every journal record hash."""

        if not self.journal_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.journal_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                continue
            record = _read_json_line(line)
            if record.get("record_hash") != _record_hash(record):
                raise CorruptShardError(
                    f"journal record hash mismatch at line {line_number}",
                    {"line_number": line_number, "record": record},
                )
            records.append(record)
        return records

    def resume_state(self) -> dict[str, Any]:
        """Return verified planned, terminal, missing, corrupt, and orphan state."""

        planned: set[str] = set()
        terminal_units: dict[str, dict[str, Any]] = {}
        terminal_hashes: set[str] = set()
        corrupt_rows: list[dict[str, Any]] = []
        for record in self.read_journal():
            if record.get("record_type") == "planned_unit":
                planned.add(str(record["unit_id"]))
            elif record.get("record_type") == "terminal_unit":
                unit_id = str(record["unit_id"])
                if unit_id in terminal_units:
                    old = terminal_units[unit_id]
                    if old["shard_hash"] != record["shard_hash"]:
                        raise DuplicateUnitError(
                            f"unit_id {unit_id} has conflicting journal hashes"
                        )
                    continue
                try:
                    self._verify_shard(str(record["shard_hash"]))
                except CorruptShardError as exc:
                    corrupt_rows.append(exc.row)
                    continue
                terminal_units[unit_id] = dict(record)
                terminal_hashes.add(str(record["shard_hash"]))
        orphan_hashes = []
        for shard in sorted(self.shard_dir.glob("*.json")):
            shard_hash = "sha256:" + shard.stem
            if shard_hash not in terminal_hashes:
                orphan_hashes.append(shard_hash)
        missing = sorted(planned - set(terminal_units))
        return {
            "planned_unit_ids": sorted(planned),
            "terminal_units": terminal_units,
            "terminal_unit_ids": sorted(terminal_units),
            "missing_unit_ids": missing,
            "corrupt_shard_rows": corrupt_rows,
            "orphan_shard_hashes": orphan_hashes,
            "all_planned_terminal": not missing,
        }

    def finalize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically replace the final artifact after all planned units close."""

        state = self.resume_state()
        if state["missing_unit_ids"]:
            raise MissingTerminalUnitError(
                "missing terminal units: " + ",".join(state["missing_unit_ids"])
            )
        reason = nonterminal_status_reason(payload)
        if reason is not None:
            raise ValueError(f"nonterminal final payload: {reason}")
        data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        data += b"\n"
        receipt = self._atomic_replace_final(data)
        receipt.update(
            {
                "final_path": str(self.final_path),
                "final_sha256": sha256_bytes(self.final_path.read_bytes()),
                "final_path_status": "terminal_complete",
                "success_path_nonterminal_artifact": False,
            }
        )
        return receipt

    def write_failure_artifact(
        self,
        payload: Mapping[str, Any],
        *,
        preserve_existing_terminal: bool = True,
    ) -> dict[str, Any]:
        """Write or preserve a closed failure artifact with diagnostics."""

        if preserve_existing_terminal and self.final_path.exists():
            try:
                existing = json.loads(self.final_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:  # pragma: no cover - malformed old file guard.
                existing = {}
            if isinstance(existing, Mapping) and nonterminal_status_reason(existing) is None:
                return {
                    "failure_artifact_written": False,
                    "existing_terminal_preserved": True,
                    "final_path": str(self.final_path),
                }
        reason = nonterminal_status_reason(payload)
        if reason is not None:
            raise ValueError(f"nonterminal failure payload: {reason}")
        data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        data += b"\n"
        receipt = self._atomic_replace_final(data)
        receipt.update(
            {
                "failure_artifact_written": True,
                "existing_terminal_preserved": False,
                "final_path": str(self.final_path),
                "final_sha256": sha256_bytes(self.final_path.read_bytes()),
            }
        )
        return receipt

    def _append_journal_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        full = dict(record)
        full["record_hash"] = _record_hash(full)
        line = json.dumps(full, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("ab") as fh:
            fh.write(line.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        self._fsync_dir(self.journal_path.parent)
        return full

    def _shard_path(self, shard_hash: str) -> Path:
        return self.shard_dir / f"{shard_hash.removeprefix('sha256:')}.json"

    def _verify_shard(self, shard_hash: str) -> dict[str, Any]:
        path = self._shard_path(shard_hash)
        if not path.exists():
            raise CorruptShardError(
                "missing shard",
                {"expected_hash": shard_hash, "shard_path": str(path), "reason": "missing"},
            )
        actual = sha256_bytes(path.read_bytes())
        if actual != shard_hash:
            quarantine = self._quarantine(path, shard_hash, actual)
            raise CorruptShardError("corrupt shard", quarantine)
        return {"shard_hash": shard_hash, "shard_path": str(path), "verified": True}

    def _quarantine(self, path: Path, expected_hash: str, actual_hash: str) -> dict[str, Any]:
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = self.quarantine_dir / f"{path.name}.{time.time_ns()}.corrupt"
        os.replace(path, target)
        self._fsync_dir(self.quarantine_dir)
        self._fsync_dir(path.parent)
        return {
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "shard_path": str(path),
            "quarantine_path": str(target),
            "quarantined": True,
        }

    def _check_disk(self, bytes_to_write: int) -> None:
        usage = shutil.disk_usage(self.work_dir if self.work_dir.exists() else self.work_dir.parent)
        required = max(self.min_free_bytes, bytes_to_write)
        if usage.free < required:
            raise InsufficientDiskError(f"insufficient disk: free={usage.free} required={required}")

    def _write_file_atomically(self, target: Path, data: bytes) -> dict[str, Any]:
        target.parent.mkdir(parents=True, exist_ok=True)
        self._check_disk(len(data))
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            dir_receipt = self._fsync_dir(target.parent)
            return {"file_fsync": True, "atomic_replace": True, **dir_receipt}
        finally:
            if tmp.exists():
                tmp.unlink()

    def _atomic_replace_final(self, data: bytes) -> dict[str, Any]:
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_disk(len(data))
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.final_path.name}.complete.",
            suffix=".tmp",
            dir=self.final_path.parent,
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            self.crash_plan.maybe_crash("before_replace")
            os.replace(tmp, self.final_path)
            dir_receipt = self._fsync_dir(self.final_path.parent)
            receipt = {"file_fsync": True, "atomic_replace": True, **dir_receipt}
            self.crash_plan.maybe_crash("after_replace")
            return receipt
        finally:
            if tmp.exists():
                tmp.unlink()

    def _fsync_dir(self, path: Path) -> dict[str, Any]:
        attempted = True
        supported = False
        try:
            fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
                supported = True
            finally:
                os.close(fd)
        except OSError:  # pragma: no cover - depends on filesystem support.
            supported = False
        return {
            "directory_fsync_attempted": attempted,
            "directory_fsync_supported": supported,
        }
