"""REQ-VERIFY-8465: an experimental journal retains the original service oracle.

Commit certificates identify the fsynced prefix before a response can escape.
Independent operands reconstruct each response and coefficient, so repaired
hashes cannot turn a different transaction into valid service history.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import python_transaction_cost_8407 as source  # explicit public adapter

Json = dict[str, Any]
JOURNAL_BYTES = 16 * 1024 * 1024
GENESIS = "sha256:" + "0" * 64
BARRIERS = (
    "before_payload",
    "after_payload",
    "before_fsync",
    "after_fsync",
    "before_checkpoint",
    "after_checkpoint",
    "before_acknowledgment",
    "after_acknowledgment",
)


def encode(value: Json) -> bytes:
    """Canonical bytes give payload lengths a single reproducible meaning."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def freeze(bundle: Json, cell: Json, repeat: int) -> Json:
    """Seal the same boundaries and response identities as Exp8464 before any write."""
    groups = source.batches(source.cost_tape(8), cell["batch"])
    prefix = f"natural-b{cell['batch']}-r8-rep{repeat}"
    count = 0
    transactions = []
    for number, group in enumerate(groups):
        ids = [f"{prefix}-q{i + 1}" for i in range(count, count + len(group))]
        transactions.append(dict(id=f"{prefix}-tx{number}", events=group, request_ids=ids))
        count += len(group)
    return dict(
        bundle=deepcopy(bundle),
        cell=cell,
        transactions=transactions,
        code_hash=canonical_hash(Path(__file__).read_text()),
        oracle_hash=canonical_hash(Path(source.__file__).read_text()),
        config_hash=canonical_hash(dict(cell=cell, delay8=8, checkpoint_every=32)),
    )


def oracle(frozen: Json, stop: int) -> Json:
    """Start with independent initial coefficients and call unchanged arithmetic in order."""
    if not 0 <= stop <= len(frozen["transactions"]):
        raise ValueError("oracle_prefix")
    label_hash = sha256_file(Path(frozen["bundle"]["labels"]["path"]))
    if label_hash != frozen["bundle"]["labels"]["sha256"]:
        raise ValueError("independent_label_seal")
    return deepcopy(_prefix(encode(frozen), label_hash, stop))


@lru_cache(maxsize=256)
def _prefix(encoded: bytes, label_hash: str, stop: int) -> Json:
    """Cache immutable causal prefixes by all operands and actual label bytes, within one process."""
    frozen = json.loads(encoded)
    if stop:
        view = deepcopy(_prefix(encoded, label_hash, stop - 1))
        tx = frozen["transactions"][stop - 1]
        rows = [
            source.step(frozen["bundle"], frozen["cell"], view["state"], event, identity)
            for event, identity in zip(tx["events"], tx["request_ids"], strict=True)
        ]
        view["replies"].append(rows)
        view["ledger"].append(tx["id"])
        view["sequence"] = stop
        return view
    state = source.s.initial(frozen["bundle"]["head"])
    return dict(state=state, replies=[], sequence=0, ledger=[])


def record(payload: Json, sequence: int, previous: str, frozen: Json) -> Json:
    """Keep original operands in compact records rather than serializing a whole state."""
    value = dict(
        sequence=sequence,
        previous_digest=previous,
        payload=payload,
        payload_length=len(encode(payload)),
        payload_hash=canonical_hash(payload),
        code_hash=frozen["code_hash"],
        oracle_hash=frozen["oracle_hash"],
        config_hash=frozen["config_hash"],
        acknowledgment_id=payload["id"],
    )
    return dict(value, digest=canonical_hash(value))


def certificate(row: Json) -> Json:
    """A retained immutable commit certificate prevents an acknowledged suffix vanishing."""
    return {key: row[key] for key in ("sequence", "digest", "acknowledgment_id")}


class Journal:
    """One writer commits; readers copy a complete immutable prefix under the same lock.

    This bounded qualification retains all segments and checkpoints. Reclamation
    is unnecessary inside the ceiling, so every pinned reader keeps its operands.
    """

    def __init__(self, path: Path, frozen: Json):
        self.path, self.frozen = path, deepcopy(frozen)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def initialize(self) -> None:
        """Initialization is explicit and cannot erase a previously committed directory."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (self.path / "initial-seal.json").exists():
                if list(self.path.glob("commit-*.json")) or (self.path / "events.jsonl").exists():
                    raise ValueError("missing_initial_seal")
                atomic_json(
                    self.path / "initial-seal.json", dict(sha256=canonical_hash(self.frozen))
                )
                with (self.path / "events.jsonl").open("xb") as stream:
                    os.fsync(stream.fileno())
                source.s.sync_directory(self.path)
                self._checkpoint(oracle(self.frozen, 0), GENESIS, lambda phase: None)
            self._read()

    def _read(self) -> tuple[Json, str]:
        """Validate certificates before trimming only bytes outside the committed prefix."""
        self._budget(0)
        seal = json.loads((self.path / "initial-seal.json").read_bytes())
        if seal != dict(sha256=canonical_hash(self.frozen)):
            raise ValueError("independent_initial_seal")
        certs = sorted(self.path.glob("commit-*.json"))
        data = (self.path / "events.jsonl").read_bytes()
        lines = data.splitlines(keepends=True)
        previous, offset = GENESIS, 0
        for seq, cert_path in enumerate(certs, 1):
            if cert_path.name != f"commit-{seq:04d}.json" or seq > len(lines):
                raise ValueError("acknowledged_prefix_missing")
            try:
                row = json.loads(lines[seq - 1])
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("acknowledged_record_corrupt") from error
            if not lines[seq - 1].endswith(b"\n"):
                raise ValueError("acknowledged_record_truncated")
            if seq > len(self.frozen["transactions"]):
                raise ValueError("sealed_sequence")
            expected = record(self.frozen["transactions"][seq - 1], seq, previous, self.frozen)
            if encode(row) != encode(expected) or json.loads(cert_path.read_bytes()) != certificate(
                expected
            ):
                raise ValueError("sealed_transition_record")
            previous = row["digest"]
            offset += len(lines[seq - 1])
        # A newline in an uncommitted suffix is a complete abandoned transaction,
        # not an interior gap: certificates prove precisely which prefix survived.
        if len(data) != offset:
            with (self.path / "events.jsonl").open("r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
        view = oracle(self.frozen, len(certs))
        pointer = json.loads((self.path / "checkpoint-current.json").read_bytes())
        published = sorted(self.path.glob("checkpoint-*.published"))
        if published and pointer["sequence"] < int(published[-1].stem.split("-")[1]):
            raise ValueError("stale_checkpoint_pointer")
        seq = pointer["sequence"]
        if not 0 <= seq <= len(certs) or Path(pointer["file"]).name != pointer["file"]:
            raise ValueError("checkpoint_prefix")
        checkpoint = json.loads((self.path / pointer["file"]).read_bytes())
        digest = GENESIS if seq == 0 else json.loads(certs[seq - 1].read_bytes())["digest"]
        expected_cp = dict(view=oracle(self.frozen, seq), prefix_digest=digest)
        if encode(checkpoint) != encode(expected_cp) or pointer["sha256"] != canonical_hash(
            expected_cp
        ):
            raise ValueError("sealed_checkpoint")
        return view, previous

    def read(self, acknowledged: list[str] | None = None) -> Json:
        """An independent caller ledger makes acknowledged suffix deletion detectable."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            view, _ = self._read()
            if acknowledged is not None and view["ledger"][: len(acknowledged)] != acknowledged:
                raise ValueError("acknowledged_ledger_loss")
            return view

    def _budget(self, extra: int) -> None:
        """The ceiling counts all retained journal evidence and publication temporaries."""
        if (
            sum(p.stat().st_size for p in self.path.iterdir() if p.is_file()) + extra
            > JOURNAL_BYTES
        ):
            raise ValueError("journal_ceiling")

    def _checkpoint(self, view: Json, digest: str, hook: Callable[[str], None]) -> None:
        """Bind the full causal state to an exact prefix before any reclamation is possible."""
        hook("before_checkpoint")
        seq = view["sequence"]
        checkpoint = dict(view=view, prefix_digest=digest)
        name = f"checkpoint-{seq:04d}.json"
        self._budget(len(json.dumps(checkpoint, indent=2).encode()) * 2 + 2048)
        atomic_json(self.path / name, checkpoint)
        source.s.sync_directory(self.path)
        atomic_json(
            self.path / "checkpoint-current.json",
            dict(sequence=seq, file=name, sha256=canonical_hash(checkpoint)),
        )
        source.s.sync_directory(self.path)
        atomic_json(self.path / f"checkpoint-{seq:04d}.published", dict(sequence=seq))
        source.s.sync_directory(self.path)
        hook("after_checkpoint")

    def apply(self, tx: Json, hook: Callable[[str], None] = lambda phase: None) -> list[Json]:
        """Publish a fsynced event and its durable certificate before returning any reply."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before, previous = self._read()
            seq = next_sequence(self.frozen, before, tx)
            if seq <= before["sequence"]:
                return list(before["replies"][seq - 1])
            after = oracle(self.frozen, seq)
            check_state(self.frozen, after)
            row = record(tx, seq, previous, self.frozen)
            payload = encode(row) + b"\n"
            self._budget(len(payload) + 2048)
            hook("before_payload")
            with (self.path / "events.jsonl").open("ab", buffering=0) as stream:
                stream.write(payload)
                hook("after_payload")
                hook("before_fsync")
                os.fsync(stream.fileno())
                hook("after_fsync")
            atomic_json(self.path / f"commit-{seq:04d}.json", certificate(row))
            source.s.sync_directory(self.path)
            if seq % 32 == 0:
                self._checkpoint(after, row["digest"], hook)
            hook("before_acknowledgment")
            hook("after_acknowledgment")
            return list(after["replies"][-1])

    def drain(self) -> None:
        """Final drain publishes all pending IDs and coefficients without dropping segments."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            view, digest = self._read()
            self._checkpoint(view, digest, lambda phase: None)


def next_sequence(frozen: Json, before: Json, tx: Json) -> int:
    """A duplicate returns its saved reply; a conflicting ID never reaches arithmetic."""
    if tx["id"] in before["ledger"]:
        index = before["ledger"].index(tx["id"])
        if encode(tx) != encode(frozen["transactions"][index]):
            raise ValueError("duplicate_conflict")
        return int(index + 1)
    seq = before["sequence"] + 1
    if seq > len(frozen["transactions"]) or encode(tx) != encode(frozen["transactions"][seq - 1]):
        raise ValueError("sealed_transaction_order")
    return int(seq)


def check_state(frozen: Json, view: Json) -> None:
    """Keep the original limits on logical requests and full durable state bytes."""
    if (
        sum(len(tx["events"]) for tx in frozen["transactions"][: view["sequence"]])
        > source.s.MAX_EVENTS
    ):
        raise ValueError("event_budget")
    if (
        len(json.dumps(dict(state=view["state"], sha256=canonical_hash(view["state"]))).encode())
        > source.s.MAX_BYTES
    ):
        raise ValueError("state_byte_budget")


class Snapshot:
    """Use the qualified production snapshot writer unchanged as the control arm."""

    def __init__(self, path: Path, frozen: Json):
        self.path, self.frozen = path, frozen
        self.store = source.QualifiedStore(path, dict(head=frozen["bundle"]["head"], events=[]))

    def initialize(self) -> None:
        """Explicit startup writes the same independent initial state as journal replay."""
        self.store.commit(oracle(self.frozen, 0)["state"])

    def read(self) -> Json:
        """Read actual pinned snapshot bytes and reconstruct their meaning independently."""
        state = source.read_snapshot(self.path)
        versions = [0]
        for tx in self.frozen["transactions"]:
            versions.append(versions[-1] + len(tx["events"]))
        if state["version"] not in versions:
            raise ValueError("snapshot_sequence")
        view = oracle(self.frozen, versions.index(state["version"]))
        if canonical_hash(view["state"]) != canonical_hash(state):
            raise ValueError("sealed_snapshot")
        return view

    def apply(self, tx: Json, hook: Callable[[str], None] = lambda phase: None) -> list[Json]:
        """Only hook names adapt; snapshot write, fsync and pointer operations stay shipped."""
        before = self.read()
        seq = next_sequence(self.frozen, before, tx)
        if seq <= before["sequence"]:
            return list(before["replies"][seq - 1])
        after = oracle(self.frozen, seq)
        check_state(self.frozen, after)
        hook("before_payload")

        def mapped(phase: str) -> None:
            for name in dict(
                temporary_write=("after_payload", "before_fsync"),
                file_fsync=("after_fsync", "before_checkpoint"),
                pointer_replace=("after_checkpoint",),
                acknowledgment=("before_acknowledgment",),
            ).get(phase, ()):
                hook(name)

        self.store.commit(after["state"], mapped)
        hook("after_acknowledgment")
        return list(after["replies"][-1])
