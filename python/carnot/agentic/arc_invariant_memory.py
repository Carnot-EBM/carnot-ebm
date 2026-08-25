"""Typed, bounded side-state for live invariant projection.

The store keeps exact verifier evidence with each invariant for its complete
lifecycle. It never interprets stored strings as code, prompts, or queries.
Admission and retrieval use the same deterministic validation contract.

Spec refs: REQ-STORE-6613, SCENARIO-STORE-6613-ADMISSION-RETRIEVAL,
SCENARIO-STORE-6613-INJECTION, SCENARIO-STORE-6613-LIFECYCLE,
SCENARIO-STORE-6613-RECOVERY, SCENARIO-STORE-6613-IMMUTABILITY-HARDWARE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
from types import MappingProxyType
from typing import Any


RECORD_SCHEMA_VERSION = "carnot.arc.invariant_record.v1"
VERIFIER_SCHEMA_VERSION = "carnot.arc.invariant_verifier_descriptor.v1"
STORE_SCHEMA_VERSION = "carnot.arc.invariant_memory_store.v1"
JOURNAL_SCHEMA_VERSION = "carnot.arc.invariant_memory_journal.v1"
FEATURE_SCHEMA_VERSION = "carnot.arc.grid_features.mean_rms.v1"
COMPACT_LAYOUT_VERSION = 1
COMPACT_STRUCT = struct.Struct("<4sHBBQ32s32s32s32s32s5d6d")
_EMPTY_CHECKSUM = "sha256:" + "0" * 64
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_QUERY_RE = re.compile(r"^\s*(?:select|insert|update|delete|match|query\b)", re.IGNORECASE)
_COMMAND_MARKERS = (
    "ignore previous",
    "system:",
    "assistant:",
    "rm -",
    "curl ",
    "wget ",
    "exec(",
)


JsonDict = dict[str, Any]
MetricItems = tuple[tuple[str, float], ...]


class InvariantMemoryError(RuntimeError):
    """Base error for lifecycle operations that must stop without partial use."""


class JournalCorruptionError(InvariantMemoryError):
    """The append-only journal failed its checksum or predecessor chain."""


class InterruptedWriteError(InvariantMemoryError):
    """A test interrupted an atomic commit before its final replacement."""


class LifecycleState(StrEnum):
    """The closed state set for a verifier-governed invariant record."""

    PROVISIONAL = "provisional"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    ARCHIVED = "archived"


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes for state, record, and journal checksums."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return one project-style SHA-256 value."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value through the canonical byte representation."""

    return sha256_bytes(canonical_json_bytes(value))


def _require_hash(value: str, name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a sha256 digest")


def _metric_items(values: Mapping[str, int | float]) -> MetricItems:
    rows = tuple(sorted((str(key), float(value)) for key, value in values.items()))
    if not rows or any(not key or not math.isfinite(value) for key, value in rows):
        raise ValueError("exact metrics must be non-empty and finite")
    return rows


@dataclass(frozen=True)
class VerifierDescriptor:
    """Exact evidence and advisory uncertainty kept for the full lifecycle."""

    schema_version: str
    source_transition_hashes: tuple[str, ...]
    world_model_hash: str
    feature_schema: str
    exact_pre_metrics: MetricItems
    exact_post_metrics: MetricItems
    confidence: float
    uncertainty: float
    exact_evidence: bool
    observed_sequence_index: int
    max_staleness_steps: int
    descriptor_checksum: str

    def __post_init__(self) -> None:
        if self.schema_version != VERIFIER_SCHEMA_VERSION:
            raise ValueError("verifier schema version mismatch")
        if not self.source_transition_hashes:
            raise ValueError("source transition hashes must be non-empty")
        for value in self.source_transition_hashes:
            _require_hash(value, "source transition hash")
        _require_hash(self.world_model_hash, "world model hash")
        _require_hash(self.descriptor_checksum, "descriptor checksum")
        if not self.feature_schema:
            raise ValueError("feature schema must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in the inclusive range 0..1")
        if not math.isfinite(self.uncertainty) or self.uncertainty < 0.0:
            raise ValueError("uncertainty must be finite and non-negative")
        if self.observed_sequence_index < 0 or self.max_staleness_steps < 0:
            raise ValueError("sequence and staleness values must be non-negative")
        if self.descriptor_checksum != sha256_json(self._checksum_payload()):
            raise ValueError("descriptor checksum mismatch")

    @classmethod
    def create(
        cls,
        *,
        source_transition_hashes: Sequence[str],
        world_model_hash: str,
        feature_schema: str,
        exact_pre_metrics: Mapping[str, int | float],
        exact_post_metrics: Mapping[str, int | float],
        confidence: float,
        uncertainty: float,
        exact_evidence: bool,
        observed_sequence_index: int,
        max_staleness_steps: int,
    ) -> "VerifierDescriptor":
        payload = {
            "schema_version": VERIFIER_SCHEMA_VERSION,
            "source_transition_hashes": tuple(str(value) for value in source_transition_hashes),
            "world_model_hash": str(world_model_hash),
            "feature_schema": str(feature_schema),
            "exact_pre_metrics": _metric_items(exact_pre_metrics),
            "exact_post_metrics": _metric_items(exact_post_metrics),
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "exact_evidence": bool(exact_evidence),
            "observed_sequence_index": int(observed_sequence_index),
            "max_staleness_steps": int(max_staleness_steps),
        }
        return cls(**payload, descriptor_checksum=sha256_json(payload))

    def _checksum_payload(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "source_transition_hashes": self.source_transition_hashes,
            "world_model_hash": self.world_model_hash,
            "feature_schema": self.feature_schema,
            "exact_pre_metrics": self.exact_pre_metrics,
            "exact_post_metrics": self.exact_post_metrics,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "exact_evidence": self.exact_evidence,
            "observed_sequence_index": self.observed_sequence_index,
            "max_staleness_steps": self.max_staleness_steps,
        }

    def to_dict(self) -> JsonDict:
        return {
            **self._checksum_payload(),
            "source_transition_hashes": list(self.source_transition_hashes),
            "exact_pre_metrics": dict(self.exact_pre_metrics),
            "exact_post_metrics": dict(self.exact_post_metrics),
            "descriptor_checksum": self.descriptor_checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerifierDescriptor":
        return cls(
            schema_version=str(payload["schema_version"]),
            source_transition_hashes=tuple(
                str(value) for value in payload["source_transition_hashes"]
            ),
            world_model_hash=str(payload["world_model_hash"]),
            feature_schema=str(payload["feature_schema"]),
            exact_pre_metrics=_metric_items(payload["exact_pre_metrics"]),
            exact_post_metrics=_metric_items(payload["exact_post_metrics"]),
            confidence=float(payload["confidence"]),
            uncertainty=float(payload["uncertainty"]),
            exact_evidence=bool(payload["exact_evidence"]),
            observed_sequence_index=int(payload["observed_sequence_index"]),
            max_staleness_steps=int(payload["max_staleness_steps"]),
            descriptor_checksum=str(payload["descriptor_checksum"]),
        )


@dataclass(frozen=True)
class InvariantRecord:
    """One immutable invariant proposal with mutable lifecycle copies."""

    schema_version: str
    record_id: str
    source_id: str
    descriptor: VerifierDescriptor
    invariant_basis: tuple[float, float, float, float]
    invariant_threshold: float
    lifecycle_state: LifecycleState
    admission_reason: str
    created_sequence_index: int
    updated_sequence_index: int
    journal_checksum: str

    def __post_init__(self) -> None:
        if self.schema_version != RECORD_SCHEMA_VERSION:
            raise ValueError("record schema version mismatch")
        _require_hash(self.record_id, "record id")
        _require_hash(self.journal_checksum, "journal checksum")
        if not self.source_id:
            raise ValueError("source id must be non-empty")
        if len(self.invariant_basis) != 4 or any(
            not math.isfinite(value) for value in self.invariant_basis
        ):
            raise ValueError("invariant basis must contain four finite values")
        if not math.isfinite(self.invariant_threshold):
            raise ValueError("invariant threshold must be finite")
        if self.created_sequence_index < 0 or self.updated_sequence_index < 0:
            raise ValueError("record sequence values must be non-negative")

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "source_id": self.source_id,
            "descriptor": self.descriptor.to_dict(),
            "invariant_basis": list(self.invariant_basis),
            "invariant_threshold": self.invariant_threshold,
            "lifecycle_state": self.lifecycle_state.value,
            "admission_reason": self.admission_reason,
            "created_sequence_index": self.created_sequence_index,
            "updated_sequence_index": self.updated_sequence_index,
            "journal_checksum": self.journal_checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvariantRecord":
        basis = tuple(float(value) for value in payload["invariant_basis"])
        return cls(
            schema_version=str(payload["schema_version"]),
            record_id=str(payload["record_id"]),
            source_id=str(payload["source_id"]),
            descriptor=VerifierDescriptor.from_dict(payload["descriptor"]),
            invariant_basis=basis,  # type: ignore[arg-type]
            invariant_threshold=float(payload["invariant_threshold"]),
            lifecycle_state=LifecycleState(str(payload["lifecycle_state"])),
            admission_reason=str(payload["admission_reason"]),
            created_sequence_index=int(payload["created_sequence_index"]),
            updated_sequence_index=int(payload["updated_sequence_index"]),
            journal_checksum=str(payload["journal_checksum"]),
        )


def make_invariant_record(
    *,
    source_id: str,
    descriptor: VerifierDescriptor,
    invariant_basis: Sequence[float],
    invariant_threshold: float,
    admission_reason: str,
    sequence_index: int,
) -> InvariantRecord:
    """Create a content-addressed provisional record from typed values."""

    basis = tuple(float(value) for value in invariant_basis)
    if len(basis) != 4:
        raise ValueError("invariant basis must contain four values")
    identity = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "source_id": str(source_id),
        "descriptor_checksum": descriptor.descriptor_checksum,
        "invariant_basis": basis,
        "invariant_threshold": float(invariant_threshold),
    }
    record_id = sha256_json(identity)
    journal_checksum = sha256_json({**identity, "record_id": record_id})
    return InvariantRecord(
        schema_version=RECORD_SCHEMA_VERSION,
        record_id=record_id,
        source_id=str(source_id),
        descriptor=descriptor,
        invariant_basis=basis,  # type: ignore[arg-type]
        invariant_threshold=float(invariant_threshold),
        lifecycle_state=LifecycleState.PROVISIONAL,
        admission_reason=str(admission_reason),
        created_sequence_index=int(sequence_index),
        updated_sequence_index=int(sequence_index),
        journal_checksum=journal_checksum,
    )


@dataclass(frozen=True)
class RetrievalContext:
    """Current trusted identities used by admission and retrieval."""

    source_hashes: Mapping[str, tuple[str, ...]]
    world_model_hash: str
    feature_schema: str
    sequence_index: int

    def __post_init__(self) -> None:
        frozen = {
            str(key): tuple(str(value) for value in values)
            for key, values in self.source_hashes.items()
        }
        for values in frozen.values():
            for value in values:
                _require_hash(value, "source transition hash")
        _require_hash(self.world_model_hash, "world model hash")
        if self.sequence_index < 0:
            raise ValueError("context sequence index must be non-negative")
        object.__setattr__(self, "source_hashes", MappingProxyType(frozen))


@dataclass(frozen=True)
class TransitionReceipt:
    """One replayable lifecycle decision returned to the caller."""

    action: str
    reason: str
    record_id: str
    pre_state: LifecycleState | None
    post_state: LifecycleState | None
    snapshot_index: int
    journal_checksum: str


def _injection_reason(record: InvariantRecord) -> str | None:
    values = (record.source_id, record.admission_reason, record.descriptor.feature_schema)
    lowered = "\n".join(values).lower()
    if any(marker in lowered for marker in _COMMAND_MARKERS):
        return "command_bearing_value"
    if any(_QUERY_RE.search(value) for value in values):
        return "query_shaped_value"
    return None


def _validation_reason(record: InvariantRecord, context: RetrievalContext) -> str | None:
    injection = _injection_reason(record)
    if injection:
        return injection
    expected_sources = context.source_hashes.get(record.source_id)
    if expected_sources != record.descriptor.source_transition_hashes:
        return "source_hash_mismatch"
    if record.descriptor.world_model_hash != context.world_model_hash:
        return "world_model_mismatch"
    if record.descriptor.feature_schema != context.feature_schema:
        return "feature_schema_mismatch"
    if not record.descriptor.exact_evidence:
        return "exact_evidence_missing"
    if record.descriptor.uncertainty != 0.0 or record.descriptor.confidence != 1.0:
        return "uncertain_metadata_not_authority"
    age = context.sequence_index - record.descriptor.observed_sequence_index
    if age < 0 or age > record.descriptor.max_staleness_steps:
        return "stale_evidence"
    return None


class InvariantMemoryStore:
    """A bounded record set with snapshots and a checksummed atomic journal."""

    def __init__(self, root: Path | str, *, total_capacity: int, per_source_capacity: int) -> None:
        if total_capacity <= 0 or per_source_capacity <= 0:
            raise ValueError("occupancy limits must be positive")
        if per_source_capacity > total_capacity:
            raise ValueError("per-source capacity cannot exceed total capacity")
        self.root = Path(root)
        self.total_capacity = int(total_capacity)
        self.per_source_capacity = int(per_source_capacity)
        self.state_path = self.root / "state.json"
        self.journal_path = self.root / "journal.jsonl"
        self.snapshot_dir = self.root / "snapshots"
        self.quarantine_dir = self.root / "quarantine"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, InvariantRecord] = {}
        if not self.state_path.exists():
            self._atomic_write(self.state_path, self.canonical_state_bytes())

    @classmethod
    def open(cls, root: Path | str) -> "InvariantMemoryStore":
        """Open a store, verify the journal, and finish a journaled commit."""

        root_path = Path(root)
        state_path = root_path / "state.json"
        journal_path = root_path / "journal.jsonl"
        try:
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvariantMemoryError("state file is missing or corrupt") from exc
        store = cls.__new__(cls)
        store.root = root_path
        store.state_path = state_path
        store.journal_path = journal_path
        store.snapshot_dir = root_path / "snapshots"
        store.quarantine_dir = root_path / "quarantine"
        store.snapshot_dir.mkdir(parents=True, exist_ok=True)
        store.quarantine_dir.mkdir(parents=True, exist_ok=True)
        store.total_capacity = int(state_payload["total_capacity"])
        store.per_source_capacity = int(state_payload["per_source_capacity"])
        store._records = {}
        try:
            rows = store._read_journal_verified()
        except JournalCorruptionError:
            store._quarantine_journal()
            raise
        source = rows[-1]["after_state"] if rows else state_payload
        store._load_state_payload(source)
        expected = store.canonical_state_bytes()
        if not state_path.exists() or state_path.read_bytes() != expected:
            store._atomic_write(state_path, expected)
        return store

    def records(self) -> list[InvariantRecord]:
        return [self._records[key] for key in sorted(self._records)]

    def active_records(self) -> list[InvariantRecord]:
        return [row for row in self.records() if row.lifecycle_state == LifecycleState.ACTIVE]

    def canonical_state_bytes(self) -> bytes:
        return canonical_json_bytes(self._state_payload())

    def journal_rows(self) -> list[JsonDict]:
        rows = self._read_journal_verified()
        return [{**row, "checksum_valid": True} for row in rows]

    def admit(
        self,
        record: InvariantRecord,
        context: RetrievalContext,
        *,
        interrupt_at: str | None = None,
    ) -> TransitionReceipt:
        """Persist a provisional proposal, then activate or quarantine it."""

        existing = self._records.get(record.record_id)
        if existing is not None:
            return self._commit(
                records=dict(self._records),
                action="duplicate",
                reason="duplicate_record",
                record_id=record.record_id,
                pre_state=existing.lifecycle_state,
                post_state=existing.lifecycle_state,
            )
        self._make_room(record.source_id)
        provisional = replace(record, lifecycle_state=LifecycleState.PROVISIONAL)
        proposed = dict(self._records)
        proposed[record.record_id] = provisional
        self._commit(
            records=proposed,
            action="provisional",
            reason="proposal_recorded_as_data",
            record_id=record.record_id,
            pre_state=None,
            post_state=LifecycleState.PROVISIONAL,
        )
        reason = _validation_reason(provisional, context)
        related = [row for row in self.records() if row.source_id == record.source_id]
        prior = [row for row in related if row.record_id != record.record_id]
        active_prior = [row for row in prior if row.lifecycle_state == LifecycleState.ACTIVE]
        if reason is None and active_prior:
            if any(
                row.descriptor.source_transition_hashes
                == provisional.descriptor.source_transition_hashes
                and (
                    row.invariant_basis != provisional.invariant_basis
                    or row.invariant_threshold != provisional.invariant_threshold
                )
                for row in active_prior
            ):
                reason = "contradictory_invariant"
            elif any(
                row.descriptor.source_transition_hashes
                == provisional.descriptor.source_transition_hashes
                for row in active_prior
            ):
                return self._transition_record(
                    provisional.record_id,
                    LifecycleState.ARCHIVED,
                    action="duplicate",
                    reason="duplicate_source_evidence",
                    interrupt_at=interrupt_at,
                )
        if reason is not None:
            return self._transition_record(
                provisional.record_id,
                LifecycleState.QUARANTINED,
                action="quarantine",
                reason=reason,
                interrupt_at=interrupt_at,
            )
        supersedes = any(
            row.descriptor.source_transition_hashes
            != provisional.descriptor.source_transition_hashes
            for row in prior
        )
        return self._transition_record(
            provisional.record_id,
            LifecycleState.ACTIVE,
            action="supersede" if supersedes else "activate",
            reason="exact_revalidation_passed",
            interrupt_at=interrupt_at,
        )

    def retrieve(self, source_id: str, context: RetrievalContext) -> InvariantRecord | None:
        """Return one active record only after full current-context revalidation."""

        candidates = sorted(
            (row for row in self.active_records() if row.source_id == source_id),
            key=lambda row: (-row.updated_sequence_index, row.record_id),
        )
        for record in candidates:
            reason = _validation_reason(record, context)
            if reason is None:
                return record
            self._transition_record(
                record.record_id,
                LifecycleState.QUARANTINED,
                action="quarantine",
                reason=f"retrieval_{reason}",
            )
        return None

    def archive(self, record_id: str, *, reason: str) -> TransitionReceipt:
        return self._transition_record(
            record_id,
            LifecycleState.ARCHIVED,
            action="archive",
            reason=reason,
        )

    def restore(self, record_id: str, context: RetrievalContext) -> TransitionReceipt:
        record = self._require_record(record_id)
        reason = _validation_reason(record, context)
        if reason is not None:
            return self._transition_record(
                record_id,
                LifecycleState.QUARANTINED,
                action="quarantine",
                reason=f"restore_{reason}",
            )
        return self._transition_record(
            record_id,
            LifecycleState.ACTIVE,
            action="restore",
            reason="archive_exact_revalidation_passed",
        )

    def rollback(self, snapshot_index: int) -> TransitionReceipt:
        rows = self._read_journal_verified()
        if snapshot_index <= 0 or snapshot_index > len(rows):
            raise InvariantMemoryError("rollback target is not a committed event")
        target = rows[snapshot_index - 1]["after_state"]
        records = {
            str(row["record_id"]): InvariantRecord.from_dict(row) for row in target["records"]
        }
        return self._commit(
            records=records,
            action="rollback",
            reason=f"restore_committed_event_{snapshot_index}",
            record_id="store",
            pre_state=None,
            post_state=None,
        )

    def _transition_record(
        self,
        record_id: str,
        state: LifecycleState,
        *,
        action: str,
        reason: str,
        interrupt_at: str | None = None,
    ) -> TransitionReceipt:
        current = self._require_record(record_id)
        rows = dict(self._records)
        rows[record_id] = replace(
            current,
            lifecycle_state=state,
            updated_sequence_index=max(
                current.updated_sequence_index,
                current.descriptor.observed_sequence_index,
            ),
        )
        return self._commit(
            records=rows,
            action=action,
            reason=reason,
            record_id=record_id,
            pre_state=current.lifecycle_state,
            post_state=state,
            interrupt_at=interrupt_at,
        )

    def _make_room(self, source_id: str) -> None:
        while len(self._records) >= self.total_capacity:
            self._evict_one(list(self._records.values()), "total_capacity")
        while (
            sum(row.source_id == source_id for row in self._records.values())
            >= self.per_source_capacity
        ):
            candidates = [row for row in self._records.values() if row.source_id == source_id]
            self._evict_one(candidates, "per_source_capacity")

    def _evict_one(self, candidates: Sequence[InvariantRecord], reason: str) -> None:
        priority = {
            LifecycleState.ARCHIVED: 0,
            LifecycleState.QUARANTINED: 1,
            LifecycleState.PROVISIONAL: 2,
            LifecycleState.ACTIVE: 3,
        }
        victim = min(
            candidates,
            key=lambda row: (
                priority[row.lifecycle_state],
                row.updated_sequence_index,
                row.record_id,
            ),
        )
        records = dict(self._records)
        del records[victim.record_id]
        self._commit(
            records=records,
            action="evict",
            reason=reason,
            record_id=victim.record_id,
            pre_state=victim.lifecycle_state,
            post_state=None,
        )

    def _require_record(self, record_id: str) -> InvariantRecord:
        try:
            return self._records[record_id]
        except KeyError as exc:
            raise InvariantMemoryError("record is not present") from exc

    def _state_payload(self) -> JsonDict:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "total_capacity": self.total_capacity,
            "per_source_capacity": self.per_source_capacity,
            "records": [row.to_dict() for row in self.records()],
        }

    def _load_state_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != STORE_SCHEMA_VERSION:
            raise InvariantMemoryError("store schema version mismatch")
        self.total_capacity = int(payload["total_capacity"])
        self.per_source_capacity = int(payload["per_source_capacity"])
        records = [InvariantRecord.from_dict(row) for row in payload["records"]]
        if len(records) > self.total_capacity:
            raise InvariantMemoryError("stored occupancy exceeds total capacity")
        counts: dict[str, int] = {}
        for record in records:
            counts[record.source_id] = counts.get(record.source_id, 0) + 1
        if any(value > self.per_source_capacity for value in counts.values()):
            raise InvariantMemoryError("stored occupancy exceeds per-source capacity")
        self._records = {row.record_id: row for row in records}

    def _commit(
        self,
        *,
        records: Mapping[str, InvariantRecord],
        action: str,
        reason: str,
        record_id: str,
        pre_state: LifecycleState | None,
        post_state: LifecycleState | None,
        interrupt_at: str | None = None,
    ) -> TransitionReceipt:
        journal = self._read_journal_verified()
        event_index = len(journal) + 1
        before_payload = self._state_payload()
        after_payload = {
            "schema_version": STORE_SCHEMA_VERSION,
            "total_capacity": self.total_capacity,
            "per_source_capacity": self.per_source_capacity,
            "records": [records[key].to_dict() for key in sorted(records)],
        }
        snapshot_path = self.snapshot_dir / f"{event_index:08d}.json"
        self._atomic_write(snapshot_path, canonical_json_bytes(before_payload))
        base = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "event_index": event_index,
            "previous_checksum": journal[-1]["journal_checksum"] if journal else _EMPTY_CHECKSUM,
            "action": action,
            "reason": reason,
            "record_id": record_id,
            "pre_state": pre_state.value if pre_state else None,
            "post_state": post_state.value if post_state else None,
            "snapshot_index": event_index,
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": sha256_bytes(canonical_json_bytes(before_payload)),
            "before_state_sha256": sha256_json(before_payload),
            "after_state_sha256": sha256_json(after_payload),
            "after_state": after_payload,
        }
        event = {**base, "journal_checksum": sha256_json(base)}
        journal_bytes = b"".join(canonical_json_bytes(row) for row in (*journal, event))
        self._atomic_write(
            self.journal_path,
            journal_bytes,
            interrupt=interrupt_at == "before_journal_replace",
        )
        self._atomic_write(
            self.state_path,
            canonical_json_bytes(after_payload),
            interrupt=interrupt_at == "before_state_replace",
        )
        self._records = dict(records)
        return TransitionReceipt(
            action=action,
            reason=reason,
            record_id=record_id,
            pre_state=pre_state,
            post_state=post_state,
            snapshot_index=event_index,
            journal_checksum=event["journal_checksum"],
        )

    def _read_journal_verified(self) -> list[JsonDict]:
        if not self.journal_path.exists():
            return []
        try:
            lines = self.journal_path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines if line]
        except (OSError, json.JSONDecodeError) as exc:
            raise JournalCorruptionError("journal is not valid JSON lines") from exc
        previous = _EMPTY_CHECKSUM
        for index, row in enumerate(rows, start=1):
            base = {key: value for key, value in row.items() if key != "journal_checksum"}
            if row.get("event_index") != index:
                raise JournalCorruptionError("journal sequence is not contiguous")
            if row.get("previous_checksum") != previous:
                raise JournalCorruptionError("journal predecessor checksum mismatch")
            if row.get("journal_checksum") != sha256_json(base):
                raise JournalCorruptionError("journal event checksum mismatch")
            if row.get("after_state_sha256") != sha256_json(row.get("after_state")):
                raise JournalCorruptionError("journal state checksum mismatch")
            previous = str(row["journal_checksum"])
        return rows

    def _quarantine_journal(self) -> None:
        if not self.journal_path.exists():
            return
        digest = sha256_bytes(self.journal_path.read_bytes()).split(":", 1)[1][:16]
        target = self.quarantine_dir / f"journal-{digest}.jsonl"
        os.replace(self.journal_path, target)

    @staticmethod
    def _atomic_write(path: Path, data: bytes, *, interrupt: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if interrupt:
            raise InterruptedWriteError(f"interrupted before replacing {path.name}")
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _digest_bytes(value: str) -> bytes:
    if _SHA256_RE.fullmatch(value):
        return bytes.fromhex(value.split(":", 1)[1])
    return hashlib.sha256(value.encode("utf-8")).digest()


def _metric_value(items: MetricItems, key: str) -> float:
    return float(dict(items).get(key, 0.0))


def serialize_compact_record(record: InvariantRecord) -> bytes:
    """Serialize one active record with a fixed Rust-compatible layout."""

    state_codes = {
        LifecycleState.PROVISIONAL: 0,
        LifecycleState.ACTIVE: 1,
        LifecycleState.QUARANTINED: 2,
        LifecycleState.ARCHIVED: 3,
    }
    descriptor = record.descriptor
    metrics = (
        _metric_value(descriptor.exact_pre_metrics, "prediction_error"),
        _metric_value(descriptor.exact_pre_metrics, "invariant_residual"),
        _metric_value(descriptor.exact_pre_metrics, "evidence_count"),
        _metric_value(descriptor.exact_post_metrics, "prediction_error"),
        _metric_value(descriptor.exact_post_metrics, "invariant_residual"),
        _metric_value(descriptor.exact_post_metrics, "evidence_count"),
    )
    return COMPACT_STRUCT.pack(
        b"IMR1",
        COMPACT_LAYOUT_VERSION,
        state_codes[record.lifecycle_state],
        0,
        record.updated_sequence_index,
        _digest_bytes(record.record_id),
        _digest_bytes(record.source_id),
        _digest_bytes(descriptor.source_transition_hashes[0]),
        _digest_bytes(descriptor.world_model_hash),
        _digest_bytes(descriptor.feature_schema),
        *record.invariant_basis,
        record.invariant_threshold,
        *metrics,
    )


def deserialize_compact_record(data: bytes) -> JsonDict:
    """Decode the fixed layout without creating an executable record value."""

    if len(data) != COMPACT_STRUCT.size:
        raise ValueError("compact record has the wrong byte length")
    values = COMPACT_STRUCT.unpack(data)
    if values[0] != b"IMR1" or values[1] != COMPACT_LAYOUT_VERSION:
        raise ValueError("compact record header mismatch")
    states = ("provisional", "active", "quarantined", "archived")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "layout_version": values[1],
        "lifecycle_state": states[values[2]],
        "updated_sequence_index": values[4],
        "record_id_sha256": "sha256:" + values[5].hex(),
        "source_id_sha256": "sha256:" + values[6].hex(),
        "source_transition_sha256": "sha256:" + values[7].hex(),
        "world_model_sha256": "sha256:" + values[8].hex(),
        "feature_schema_sha256": "sha256:" + values[9].hex(),
        "invariant_basis": tuple(values[10:14]),
        "invariant_threshold": values[14],
        "exact_metrics": tuple(values[15:21]),
    }


def compact_projection(
    features: tuple[float, float],
    basis: tuple[float, float, float, float],
    threshold: float,
) -> JsonDict:
    """Run one bounded quadratic projection step with explicit arithmetic cost."""

    x, y = (float(features[0]), float(features[1]))
    a, b, c, d = (float(value) for value in basis)
    value = x * (a * x + b * y) + y * (c * x + d * y)
    residual = value - float(threshold)
    gradient_x = (2.0 * a) * x + (b + c) * y
    gradient_y = (b + c) * x + (2.0 * d) * y
    norm_squared = gradient_x * gradient_x + gradient_y * gradient_y
    if norm_squared <= 1e-18:
        projected = (x, y)
    else:
        scale = residual / norm_squared
        projected = (x - scale * gradient_x, y - scale * gradient_y)
    return {
        "projected_features": projected,
        "residual_before": residual,
        "operation_count": 20,
        "operation_bound": 24,
        "bounded": True,
    }


def compact_hardware_receipt(
    records: Sequence[InvariantRecord],
    *,
    lookup_count: int,
    capacity: int,
) -> JsonDict:
    """Describe the bounded CPU reference and portable Rust data layout."""

    encoded = [serialize_compact_record(record) for record in records]
    return {
        "layout": "little_endian_IMR1_fixed_width_v1",
        "bytes_per_record": COMPACT_STRUCT.size,
        "resident_record_count": len(records),
        "resident_bytes": sum(len(row) for row in encoded),
        "capacity": int(capacity),
        "memory_bound_bytes": int(capacity) * COMPACT_STRUCT.size,
        "lookup_count": int(lookup_count),
        "lookup_comparison_bound": int(lookup_count) * int(capacity),
        "projection_operation_bound_per_lookup": 24,
        "execution_substrate": "python_cpu_reference",
        "rust_compatible_layout": True,
        "rust_execution_claimed": False,
        "fpga_execution_claimed": False,
        "bounded": len(records) <= int(capacity) and lookup_count >= 0,
    }
