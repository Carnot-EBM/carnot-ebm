"""Durable bounded policy state for chronological continual learning.

The store writes an append-only prepare record before it replaces active state.
This order lets a fresh process distinguish an unapplied proposal from a
committed update after interruption. Model weights never enter this store.

Spec refs: REQ-LEARN-6978 and SCENARIO-LEARN-6978-JOURNAL/ROLLBACK/RESTART.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]
STATE_SCHEMA = "carnot.constraint_policy_store.v1"
JOURNAL_SCHEMA = "carnot.constraint_policy_journal.v1"


class ForcedInterruption(RuntimeError):
    """Mark the test fixture that stops after durable prepare."""


class JournalIntegrityError(RuntimeError):
    """Reject journal bytes that do not form the recorded hash chain."""


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON with stable bytes so equality and hashes are exact."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest with the repository's explicit prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry after rename or journal creation."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace one file only after its complete bytes reach stable storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class PolicyStore:
    """Keep one arm's small policy memory in an atomic state and hash journal."""

    def __init__(
        self,
        root: Path | str,
        *,
        initial_records: Sequence[Mapping[str, Any]],
        max_state_bytes: int,
        read_only: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.journal_path = self.root / "journal.jsonl"
        self.max_state_bytes = int(max_state_bytes)
        self.read_only = bool(read_only)
        self.recovery_rows: list[JsonDict] = []
        initial = {
            "schema": STATE_SCHEMA,
            "version": 0,
            "records": [deepcopy(dict(row)) for row in initial_records],
        }
        if not self.state_path.exists():
            payload = canonical_bytes(initial)
            if len(payload) > self.max_state_bytes:
                raise ValueError("initial policy state exceeds byte limit")
            _atomic_write(self.state_path, payload)
        if not self.journal_path.exists():
            self.journal_path.touch()
            _fsync_directory(self.root)
        self._state = self._read_state()
        self._recover_incomplete_prepare()

    def _read_state(self) -> JsonDict:
        """Load one complete state and reject an incompatible schema."""

        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid policy state schema")
        if not isinstance(value.get("records"), list):
            raise ValueError("invalid policy records")
        return value

    @property
    def state_bytes(self) -> bytes:
        """Return the exact published state bytes."""

        return self.state_path.read_bytes()

    @property
    def state_hash(self) -> str:
        """Hash the published bytes instead of an in-memory approximation."""

        return sha256_bytes(self.state_bytes)

    @property
    def state_size(self) -> int:
        """Measure the bounded state in bytes."""

        return len(self.state_bytes)

    def records(self) -> list[JsonDict]:
        """Return a copy so callers cannot mutate active state in memory."""

        return deepcopy(list(self._state["records"]))

    def _journal_row(self, phase: str, transaction_id: str, **fields: Any) -> JsonDict:
        """Append and fsync one hash-linked transaction phase."""

        rows = self._read_journal(validate=True)
        row: JsonDict = {
            "schema": JOURNAL_SCHEMA,
            "sequence": len(rows),
            "phase": phase,
            "transaction_id": transaction_id,
            "previous_row_hash": rows[-1]["row_hash"] if rows else None,
            "file_fsync": True,
            **deepcopy(fields),
        }
        row["row_hash"] = sha256_bytes(canonical_bytes(row))
        with self.journal_path.open("ab") as handle:
            handle.write(canonical_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.root)
        return row

    def _read_journal(self, *, validate: bool) -> list[JsonDict]:
        """Read all journal rows and optionally verify their complete chain."""

        rows: list[JsonDict] = []
        try:
            lines = self.journal_path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            raise JournalIntegrityError(f"journal parse failed: {exc}") from exc
        if validate:
            previous: str | None = None
            for sequence, row in enumerate(rows):
                claimed = row.get("row_hash")
                payload = dict(row)
                payload.pop("row_hash", None)
                if claimed != sha256_bytes(canonical_bytes(payload)):
                    raise JournalIntegrityError(f"journal row hash mismatch:{sequence}")
                if row.get("sequence") != sequence or row.get("previous_row_hash") != previous:
                    raise JournalIntegrityError(f"journal chain mismatch:{sequence}")
                previous = str(claimed)
        return rows

    def journal_rows(self) -> list[JsonDict]:
        """Return validated durable journal rows."""

        return deepcopy(self._read_journal(validate=True))

    def _candidate_state(self, proposal: Mapping[str, Any]) -> tuple[JsonDict, bytes]:
        """Replace one policy key and evict old learned rows until bytes fit."""

        candidate = deepcopy(self._state)
        candidate["version"] = int(candidate["version"]) + 1
        key = str(proposal["policy_key"])
        candidate["records"] = [
            row for row in candidate["records"] if str(row.get("policy_key")) != key
        ]
        candidate["records"].append(deepcopy(dict(proposal)))
        payload = canonical_bytes(candidate)
        while len(payload) > self.max_state_bytes:
            removable = next(
                (
                    index
                    for index, row in enumerate(candidate["records"])
                    if not row.get("protected") and str(row.get("policy_key")) != key
                ),
                None,
            )
            if removable is None:
                raise ValueError("policy proposal exceeds byte limit")
            candidate["records"].pop(removable)
            payload = canonical_bytes(candidate)
        return candidate, payload

    def commit(
        self,
        proposal: Mapping[str, Any],
        *,
        interrupt_after_prepare: bool = False,
    ) -> JsonDict:
        """Prepare, publish, and commit one post-outcome policy update."""

        if self.read_only:
            raise PermissionError("policy store is read-only")
        parent = self.state_bytes
        candidate, candidate_bytes = self._candidate_state(proposal)
        transaction_id = sha256_bytes(
            canonical_bytes(
                {
                    "parent_state_hash": sha256_bytes(parent),
                    "proposal": proposal,
                    "next_version": candidate["version"],
                }
            )
        )
        prepare = self._journal_row(
            "prepare",
            transaction_id,
            parent_state_hash=sha256_bytes(parent),
            new_state_hash=sha256_bytes(candidate_bytes),
            parent_state_b64=b64encode(parent).decode("ascii"),
            new_state_b64=b64encode(candidate_bytes).decode("ascii"),
            proposal=deepcopy(dict(proposal)),
        )
        if interrupt_after_prepare:
            raise ForcedInterruption(transaction_id)
        _atomic_write(self.state_path, candidate_bytes)
        self._state = candidate
        committed = self._journal_row(
            "commit",
            transaction_id,
            parent_state_hash=prepare["parent_state_hash"],
            new_state_hash=self.state_hash,
        )
        return {
            "transaction_id": transaction_id,
            "committed": True,
            "parent_state_hash": prepare["parent_state_hash"],
            "new_state_hash": committed["new_state_hash"],
            "prepare_sequence": prepare["sequence"],
            "commit_sequence": committed["sequence"],
            "state_bytes": self.state_size,
        }

    def rollback(self, parent_bytes: bytes, *, transaction_id: str, reason: str) -> JsonDict:
        """Restore caller-held parent bytes and record the exact restored hash."""

        if self.read_only:
            raise PermissionError("policy store is read-only")
        state = json.loads(parent_bytes)
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("rollback parent has invalid schema")
        _atomic_write(self.state_path, parent_bytes)
        self._state = self._read_state()
        row = self._journal_row(
            "rollback",
            transaction_id,
            reason=reason,
            restored_state_hash=self.state_hash,
            restored_state_b64=b64encode(parent_bytes).decode("ascii"),
        )
        return {
            "transaction_id": transaction_id,
            "rolled_back": self.state_bytes == parent_bytes,
            "restored_state_hash": self.state_hash,
            "journal_sequence": row["sequence"],
        }

    def _recover_incomplete_prepare(self) -> None:
        """Resolve a durable prepare by comparing it with the published state."""

        rows = self._read_journal(validate=True)
        terminal = {
            str(row["transaction_id"])
            for row in rows
            if row.get("phase") in {"commit", "commit_recovered", "abort_recovered"}
        }
        pending = [
            row
            for row in rows
            if row.get("phase") == "prepare" and str(row["transaction_id"]) not in terminal
        ]
        for prepare in pending:
            transaction_id = str(prepare["transaction_id"])
            current = self.state_hash
            if current == prepare["parent_state_hash"]:
                phase = "abort_recovered"
                applied = False
            elif current == prepare["new_state_hash"]:
                phase = "commit_recovered"
                applied = True
            else:
                raise JournalIntegrityError(
                    f"incomplete transaction state mismatch:{transaction_id}"
                )
            row = self._journal_row(
                phase,
                transaction_id,
                parent_state_hash=prepare["parent_state_hash"],
                new_state_hash=prepare["new_state_hash"],
            )
            self.recovery_rows.append(
                {
                    "transaction_id": transaction_id,
                    "incomplete_prepare_recovered": True,
                    "applied": applied,
                    "phase": phase,
                    "journal_sequence": row["sequence"],
                }
            )

    def replay_journal(self) -> JsonDict:
        """Replay state hashes without trusting stored aggregate claims."""

        rows = self._read_journal(validate=True)
        prepares: dict[str, JsonDict] = {}
        logical_hash: str | None = None
        errors: list[str] = []
        for row in rows:
            transaction_id = str(row["transaction_id"])
            phase = str(row["phase"])
            if phase == "prepare":
                prepares[transaction_id] = row
                if logical_hash is None:
                    logical_hash = str(row["parent_state_hash"])
                elif logical_hash != row["parent_state_hash"]:
                    errors.append(f"prepare_parent_mismatch:{transaction_id}")
            elif phase in {"commit", "commit_recovered"}:
                prepare = prepares.get(transaction_id)
                if prepare is None or row.get("new_state_hash") != prepare.get("new_state_hash"):
                    errors.append(f"commit_without_matching_prepare:{transaction_id}")
                else:
                    logical_hash = str(row["new_state_hash"])
            elif phase == "abort_recovered":
                if transaction_id not in prepares:
                    errors.append(f"abort_without_prepare:{transaction_id}")
            elif phase == "rollback":
                restored = b64decode(str(row["restored_state_b64"]))
                if sha256_bytes(restored) != row.get("restored_state_hash"):
                    errors.append(f"rollback_bytes_mismatch:{transaction_id}")
                logical_hash = str(row["restored_state_hash"])
            else:
                errors.append(f"unknown_phase:{phase}")
        final_hash = logical_hash or self.state_hash
        if final_hash != self.state_hash:
            errors.append("replay_final_state_mismatch")
        return {
            "passed": not errors,
            "errors": errors,
            "row_count": len(rows),
            "final_state_hash": final_hash,
        }
