"""Parent-owned row checkpoints with exact manifest and payload binding.

Spec refs: REQ-INFRA-6785, SCENARIO-INFRA-6785-DURABLE-PUBLISH, and
SCENARIO-INFRA-6785-CONFLICTS-REFUSE.

Workers can lose temporary files when they stop. This store keeps complete row
envelopes at a path chosen by the parent. Each update syncs bytes before the
rename and syncs the directory after it. This order makes a successful receipt
mean that restart can find either the old complete file or the new complete
file, never a partly written file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


JsonDict = dict[str, Any]

CHECKPOINT_SCHEMA = "carnot.durable_row_checkpoint.v1"
ENVELOPE_FIELDS = frozenset(
    {
        "row_id",
        "manifest_hash",
        "payload",
        "payload_hash",
        "attempt",
        "start_receipt",
        "end_receipt",
        "status",
    }
)
CHECKPOINT_FIELDS = frozenset({"schema", "manifest", "manifest_hash", "revision", "rows"})


class CheckpointError(ValueError):
    """Base error for data that cannot safely enter or resume a checkpoint."""


class InvalidEnvelopeError(CheckpointError):
    """A worker row is incomplete or does not match its declared hashes."""


class RowConflictError(CheckpointError):
    """One row ID was reused for a different payload."""


class ManifestMismatchError(CheckpointError):
    """A caller tried to resume state that belongs to another frozen manifest."""


class CorruptCheckpointError(CheckpointError):
    """Stored bytes do not satisfy the durable checkpoint schema."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one stable JSON form so hashes do not depend on key order."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the project hash form for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash the stable JSON representation of a value."""

    return sha256_bytes(canonical_json_bytes(value))


def atomic_write_json(path: Path | str, value: Any) -> JsonDict:
    """Publish complete JSON bytes and return the durability operations used."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(value)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    existed = target.exists()
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if (
            temporary.exists()
        ):  # pragma: no cover - only an operating-system write failure leaves it.
            temporary.unlink()
    return {
        "target": str(target),
        "replaced_existing": existed,
        "file_fsync": True,
        "atomic_replace": True,
        "directory_fsync": True,
        "published_sha256": sha256_bytes(data),
    }


def complete_row_envelope(
    *,
    row_id: str,
    manifest_hash: str,
    payload: Any,
    attempt: int,
    start_receipt: Mapping[str, Any],
    end_receipt: Mapping[str, Any],
) -> JsonDict:
    """Build the complete message that a worker can send to its parent."""

    copied_payload = deepcopy(payload)
    return {
        "row_id": row_id,
        "manifest_hash": manifest_hash,
        "payload": copied_payload,
        "payload_hash": sha256_json(copied_payload),
        "attempt": attempt,
        "start_receipt": deepcopy(dict(start_receipt)),
        "end_receipt": deepcopy(dict(end_receipt)),
        "status": "complete",
    }


class DurableRowCheckpoint:
    """Keep manifest-bound rows at the exact path selected by the parent."""

    def __init__(self, path: Path | str, frozen_manifest: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.manifest = deepcopy(dict(frozen_manifest))
        self.manifest_hash = sha256_json(self.manifest)
        self.initialization_receipt: JsonDict | None = None
        if self.path.exists():
            self._state = self._read_state()
        else:
            self._state = {
                "schema": CHECKPOINT_SCHEMA,
                "manifest": deepcopy(self.manifest),
                "manifest_hash": self.manifest_hash,
                "revision": 0,
                "rows": [],
            }
            self.initialization_receipt = atomic_write_json(self.path, self._state)

    @property
    def rows(self) -> list[JsonDict]:
        """Return copies so callers cannot mutate accepted state in memory."""

        return deepcopy(self._state["rows"])

    def pending(self, ordered_row_ids: Sequence[str]) -> list[str]:
        """Return manifest row IDs that have no complete checkpoint row."""

        complete = {str(row["row_id"]) for row in self._state["rows"]}
        return [str(row_id) for row_id in ordered_row_ids if str(row_id) not in complete]

    def append(self, envelope: Mapping[str, Any]) -> JsonDict:
        """Accept one complete row, suppress a repeat, or refuse a conflict."""

        row = deepcopy(dict(envelope))
        self._validate_envelope(row)
        existing = {str(item["row_id"]): item for item in self._state["rows"]}.get(
            str(row["row_id"])
        )
        if existing is not None:
            if existing["payload_hash"] != row["payload_hash"]:
                raise RowConflictError(f"conflicting payload for row ID {row['row_id']}")
            return {
                "row_id": row["row_id"],
                "accepted": False,
                "duplicate_suppressed": True,
                "atomic_replace": False,
                "file_fsync": False,
                "directory_fsync": False,
                "checkpoint_sha256": sha256_bytes(self.path.read_bytes()),
            }

        candidate = deepcopy(self._state)
        candidate["rows"].append(row)
        candidate["revision"] = len(candidate["rows"])
        publish = atomic_write_json(self.path, candidate)
        self._state = candidate
        return {
            "row_id": row["row_id"],
            "accepted": True,
            "duplicate_suppressed": False,
            **publish,
            "checkpoint_sha256": sha256_bytes(self.path.read_bytes()),
        }

    def _validate_envelope(self, row: Mapping[str, Any]) -> None:
        if set(row) != ENVELOPE_FIELDS:
            raise InvalidEnvelopeError("row envelope field set mismatch")
        if not isinstance(row["row_id"], str) or not row["row_id"]:
            raise InvalidEnvelopeError("row_id must be a non-empty string")
        if row["manifest_hash"] != self.manifest_hash:
            raise InvalidEnvelopeError("row manifest hash mismatch")
        if row["payload_hash"] != sha256_json(row["payload"]):
            raise InvalidEnvelopeError("row payload hash mismatch")
        if type(row["attempt"]) is not int or row["attempt"] < 1:
            raise InvalidEnvelopeError("attempt must be a positive integer")
        if not isinstance(row["start_receipt"], Mapping):
            raise InvalidEnvelopeError("start_receipt must be an object")
        if not isinstance(row["end_receipt"], Mapping):
            raise InvalidEnvelopeError("end_receipt must be an object")
        if row["status"] != "complete":
            raise InvalidEnvelopeError("only complete rows can enter durable state")

    def _read_state(self) -> JsonDict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or set(value) != CHECKPOINT_FIELDS:
                raise CorruptCheckpointError("checkpoint field set mismatch")
            if value["schema"] != CHECKPOINT_SCHEMA:
                raise CorruptCheckpointError("checkpoint schema mismatch")
            if not isinstance(value["manifest"], dict):
                raise CorruptCheckpointError("checkpoint manifest must be an object")
            if value["manifest_hash"] != sha256_json(value["manifest"]):
                raise CorruptCheckpointError("stored manifest hash mismatch")
            if value["manifest_hash"] != self.manifest_hash:
                raise ManifestMismatchError("checkpoint belongs to another frozen manifest")
            if not isinstance(value["rows"], list):
                raise CorruptCheckpointError("checkpoint rows must be a list")
            if value["revision"] != len(value["rows"]):
                raise CorruptCheckpointError("checkpoint revision does not match row count")
            seen: set[str] = set()
            for row in value["rows"]:
                if not isinstance(row, dict):
                    raise CorruptCheckpointError("checkpoint row must be an object")
                try:
                    self._validate_envelope(row)
                except InvalidEnvelopeError as exc:
                    raise CorruptCheckpointError(str(exc)) from exc
                row_id = str(row["row_id"])
                if row_id in seen:
                    raise CorruptCheckpointError(f"duplicate stored row ID {row_id}")
                seen.add(row_id)
            return value
        except ManifestMismatchError:
            raise
        except CorruptCheckpointError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise CorruptCheckpointError(f"checkpoint could not be read: {exc}") from exc
