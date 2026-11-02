"""REQ-VERIFY-8493: experimental durable append with a stable journal name.

A writer locks the directory, checks the sealed transition inputs, writes a
complete frame and syncs it before returning. Readers own immutable state copies.
Checkpoint publication is durable before any old checkpoint is removed.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import journal_equivalence_8465 as base  # explicit public adapter
from carnot.verify import publication_cost_model_8492 as framing  # explicit public adapter

__all__ = ["Store", "BARRIERS", "ROOT", "base", "framing", "write_all"]

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
BARRIERS = (
    "before_append",
    "after_append",
    "before_fsync",
    "after_fsync",
    "before_acknowledgment",
    "after_acknowledgment",
    "before_checkpoint_fsync",
    "after_checkpoint_fsync",
    "before_checkpoint_replace",
    "after_checkpoint_replace",
    "before_directory_sync",
    "after_directory_sync",
    "before_reclaim",
    "after_reclaim",
)


def write_all(fd: int, payload: bytes) -> None:
    """Retry short writes; a zero-length write is failure rather than a silent commit."""
    offset = 0
    while offset < len(payload):
        wrote = os.write(fd, payload[offset:])
        if wrote <= 0:
            raise OSError("short_write")
        offset += wrote


class Store:
    """Separate experimental store; copied reader states remain valid after reclamation."""

    def __init__(self, path: Path, frozen: Json):
        self.path, self.frozen = path, deepcopy(frozen)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.journal = path / "events.frames"
        self.operations: list[Json] = []
        self.namespace_pending = True

    def _sync(self, fd: int, target: Path, kind: str = "file_fsync") -> None:
        """Record calls only after the operating system reports successful synchronization."""
        os.fsync(fd)
        self.operations.append(dict(operation=kind, path=str(target)))

    def _directory(self, path: Path) -> None:
        """Namespace creation, replacement and unlink require their parent to be synced."""
        fd = os.open(path, os.O_DIRECTORY)
        try:
            self._sync(fd, path, "directory_fsync")
            if path == self.path:
                self.namespace_pending = False
        finally:
            os.close(fd)

    def _budget(self, extra: int = 0) -> None:
        """Retained evidence and temporary publication bytes count toward the sealed ceiling."""
        if (
            sum(p.stat().st_size for p in self.path.iterdir() if p.is_file()) + extra
            > base.JOURNAL_BYTES
        ):
            raise ValueError("journal_ceiling")

    def _publish(self, name: str, value: Json, hook: Callable[[str], None]) -> None:
        """Sync temporary bytes, replace the name, then sync the directory before continuing."""
        target = self.path / name
        temporary = self.path / ("." + name + ".pending")
        payload = base.encode(value)
        self._budget(len(payload))
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self.namespace_pending = True
        self.operations.append(dict(operation="create", path=str(temporary)))
        try:
            write_all(fd, payload)
            hook("before_checkpoint_fsync")
            self._sync(fd, temporary)
            hook("after_checkpoint_fsync")
        finally:
            os.close(fd)
        hook("before_checkpoint_replace")
        os.replace(temporary, target)
        self.namespace_pending = True
        self.operations.append(dict(operation="replace", path=str(target)))
        hook("after_checkpoint_replace")
        hook("before_directory_sync")
        self._directory(self.path)
        hook("after_directory_sync")

    def initialize(self) -> None:
        """Create and sync both the journal and namespace before accepting any transaction."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            seal = self.path / "initial-seal.json"
            if not seal.exists():
                if self.journal.exists() or (self.path / "checkpoint-current.json").exists():
                    raise ValueError("missing_initial_seal")
                self._publish(
                    "initial-seal.json", dict(sha256=canonical_hash(self.frozen)), lambda p: None
                )
                fd = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                self.operations.append(dict(operation="create", path=str(self.journal)))
                try:
                    self._sync(fd, self.journal)
                finally:
                    os.close(fd)
                self._directory(self.path)
                self._directory(self.path.parent)
                self._checkpoint(base.oracle(self.frozen, 0), framing.GENESIS, lambda p: None)
            self._read()

    def _read(self) -> tuple[Json, str]:
        """Verify independent operands; complete unacknowledged frames become durable exactly once."""
        self._budget()
        if json.loads((self.path / "initial-seal.json").read_bytes()) != dict(
            sha256=canonical_hash(self.frozen)
        ):
            raise ValueError("independent_initial_seal")
        recovered = framing.recover(self.journal.read_bytes(), self.frozen)
        with self.journal.open("r+b", buffering=0) as stream:
            if recovered["discarded_tail_bytes"]:
                stream.truncate(recovered["durable_prefix_bytes"])
            self._sync(stream.fileno(), self.journal)
        # Recovery also finishes namespace durability if a writer exited after replacement.
        if self.namespace_pending:
            self._directory(self.path)
            self.namespace_pending = False
        view = base.oracle(self.frozen, recovered["sequence"])
        base.check_state(self.frozen, view)
        pointer = json.loads((self.path / "checkpoint-current.json").read_bytes())
        seq = pointer["sequence"]
        if not 0 <= seq <= view["sequence"] or pointer["file"] != f"checkpoint-{seq:04d}.json":
            raise ValueError("checkpoint_prefix")
        checkpoint = json.loads((self.path / pointer["file"]).read_bytes())
        previous = framing.GENESIS
        for n in range(1, seq + 1):
            previous = json.loads(framing.frame(self.frozen, n, previous)[4:])["digest"]
        expected = dict(view=base.oracle(self.frozen, seq), prefix_digest=previous)
        if base.encode(checkpoint) != base.encode(expected) or pointer["sha256"] != canonical_hash(
            expected
        ):
            raise ValueError("sealed_checkpoint")
        return view, str(recovered["prefix_digest"])

    def read(self, acknowledged: list[str] | None = None) -> Json:
        """Return a verified copied prefix; an external acknowledgment ledger detects lost suffixes."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            view, _ = self._read()
            if acknowledged is not None and view["ledger"][: len(acknowledged)] != acknowledged:
                raise ValueError("acknowledged_ledger_loss")
            return view

    def _checkpoint(self, view: Json, digest: str, hook: Callable[[str], None]) -> None:
        """Publish a durable checkpoint and prefix before reclaiming old immutable files."""
        checkpoint = dict(view=view, prefix_digest=digest)
        name = f"checkpoint-{view['sequence']:04d}.json"
        self._publish(name, checkpoint, hook)
        self._publish(
            "checkpoint-current.json",
            dict(sequence=view["sequence"], file=name, sha256=canonical_hash(checkpoint)),
            hook,
        )
        hook("before_reclaim")
        for path in self.path.glob("checkpoint-*.json"):
            if path.name not in {name, "checkpoint-current.json"}:
                path.unlink()
                self.namespace_pending = True
                self.operations.append(dict(operation="unlink", path=str(path)))
        hook("after_reclaim")
        self._directory(self.path)

    def apply(self, tx: Json, hook: Callable[[str], None] = lambda p: None) -> list[Json]:
        """A retry returns the saved response; append sync precedes every new acknowledgment."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before, previous = self._read()
            seq = base.next_sequence(self.frozen, before, tx)
            if seq <= before["sequence"]:
                return list(before["replies"][seq - 1])
            after = deepcopy(before)
            replies = [
                base.source.step(self.frozen["bundle"], self.frozen["cell"], after["state"], e, q)
                for e, q in zip(tx["events"], tx["request_ids"], strict=True)
            ]
            after["sequence"] = seq
            after["ledger"].append(tx["id"])
            after["replies"].append(replies)
            base.check_state(self.frozen, after)
            if base.encode(after) != base.encode(base.oracle(self.frozen, seq)):
                raise ValueError("independent_transition")
            payload = framing.frame(self.frozen, seq, previous)
            self._budget(len(payload))
            hook("before_append")
            fd = os.open(self.journal, os.O_WRONLY | os.O_APPEND)
            try:
                write_all(fd, payload)
                self.operations.append(
                    dict(
                        operation="append", path=str(self.journal), sequence=seq, bytes=len(payload)
                    )
                )
                hook("after_append")
                hook("before_fsync")
                self._sync(fd, self.journal)
                hook("after_fsync")
            finally:
                os.close(fd)
            if seq % 32 == 0:
                self._checkpoint(after, json.loads(payload[4:])["digest"], hook)
            hook("before_acknowledgment")
            return replies

    def drain(self) -> None:
        """Final drain checkpoints the complete verified prefix without changing the journal inode."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            view, digest = self._read()
            self._checkpoint(view, digest, lambda p: None)
