"""Transactional revocable repair memory for exact-evidence replays.

Spec refs: REQ-LEARN-6290, SCENARIO-LEARN-6290-KEYS,
SCENARIO-LEARN-6290-TRANSACTION, SCENARIO-LEARN-6290-REVOCATION,
SCENARIO-LEARN-6290-RESTART.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def canonical_json(value: Any) -> str:
    """Return stable JSON bytes for hashes and restart checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stable_precedent_key(parts: Mapping[str, str]) -> str:
    """Hash only stable lookup fields, not mutable evidence or versions."""

    required = ("namespace", "model_family", "task_family", "repair_atom", "scope")
    missing = [field for field in required if not str(parts.get(field, "")).strip()]
    if missing:
        raise ValueError(f"precedent key missing fields: {missing}")
    material = {field: str(parts[field]) for field in required}
    return sha256_json({"schema": "carnot.revocable_precedent_key.v1", "parts": material})


@dataclass(frozen=True)
class AtomicRepairItem:
    """One repair item that can stand alone under current exact evidence."""

    namespace: str
    model_family: str
    task_family: str
    repair_atom: str
    scope: str
    exact_evidence_key: str
    exact_evidence_hash: str
    correction_id: str
    source_event_id: str
    atomic: bool = True
    poisoned: bool = False
    version: int = 0
    state: str = "candidate"

    @property
    def key_parts(self) -> JsonDict:
        return {
            "namespace": self.namespace,
            "model_family": self.model_family,
            "task_family": self.task_family,
            "repair_atom": self.repair_atom,
            "scope": self.scope,
        }

    @property
    def precedent_key(self) -> str:
        return stable_precedent_key(self.key_parts)

    @property
    def item_hash(self) -> str:
        return sha256_json(
            {
                "key_parts": self.key_parts,
                "exact_evidence_key": self.exact_evidence_key,
                "exact_evidence_hash": self.exact_evidence_hash,
                "correction_id": self.correction_id,
                "source_event_id": self.source_event_id,
                "atomic": self.atomic,
                "poisoned": self.poisoned,
            }
        )

    def active_version(self, version: int) -> AtomicRepairItem:
        return replace(self, version=version, state="active")

    def to_json(self) -> JsonDict:
        return {
            "namespace": self.namespace,
            "model_family": self.model_family,
            "task_family": self.task_family,
            "repair_atom": self.repair_atom,
            "scope": self.scope,
            "exact_evidence_key": self.exact_evidence_key,
            "exact_evidence_hash": self.exact_evidence_hash,
            "correction_id": self.correction_id,
            "source_event_id": self.source_event_id,
            "atomic": self.atomic,
            "poisoned": self.poisoned,
            "version": self.version,
            "state": self.state,
            "precedent_key": self.precedent_key,
            "item_hash": self.item_hash,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> AtomicRepairItem:
        return cls(
            namespace=str(payload["namespace"]),
            model_family=str(payload["model_family"]),
            task_family=str(payload["task_family"]),
            repair_atom=str(payload["repair_atom"]),
            scope=str(payload["scope"]),
            exact_evidence_key=str(payload["exact_evidence_key"]),
            exact_evidence_hash=str(payload["exact_evidence_hash"]),
            correction_id=str(payload["correction_id"]),
            source_event_id=str(payload["source_event_id"]),
            atomic=bool(payload.get("atomic", True)),
            poisoned=bool(payload.get("poisoned", False)),
            version=int(payload.get("version", 0)),
            state=str(payload.get("state", "candidate")),
        )


@dataclass(frozen=True)
class TransactionReceipt:
    committed: bool
    accepted_count: int
    rejected_count: int
    rejection_reasons: list[str]
    active_view_hash: str
    audit_hash: str
    transaction_id: str


@dataclass(frozen=True)
class RetrievalReceipt:
    items: list[AtomicRepairItem]
    active_retrieval_count: int
    revoked_retrieval_count: int
    stale_retrieval_count: int
    exact_evidence_rejection_count: int


@dataclass(frozen=True)
class MemoryCheckpoint:
    active_view: list[JsonDict]
    audit_entries: list[JsonDict]
    last_event_index: int
    snapshot_hash: str
    audit_hash: str


class TransactionalRevocableRepairMemory:
    """Append-only audit plus active view for revocable repair items."""

    def __init__(self, audit_log_path: Path | None = None) -> None:
        self.audit_log_path = audit_log_path
        self._active: dict[str, AtomicRepairItem] = {}
        self._history_by_key: dict[str, list[JsonDict]] = defaultdict(list)
        self._revoked_support_hashes: dict[str, set[str]] = defaultdict(set)
        self.audit_entries: list[JsonDict] = []
        self.last_event_index = -1
        if self.audit_log_path is not None:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_audit_log(cls, audit_log_path: Path) -> TransactionalRevocableRepairMemory:
        store = cls(audit_log_path=audit_log_path)
        if not audit_log_path.exists():
            return store
        for line in audit_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, Mapping):
                raise ValueError("audit log entry must be a JSON object")
            store._apply_audit_entry(dict(entry))
        return store

    def clone(self) -> TransactionalRevocableRepairMemory:
        other = TransactionalRevocableRepairMemory()
        for entry in self.audit_entries:
            other._apply_audit_entry(json.loads(canonical_json(entry)))
        return other

    def checkpoint(self) -> MemoryCheckpoint:
        return MemoryCheckpoint(
            active_view=self._active_json(),
            audit_entries=list(self.audit_entries),
            last_event_index=self.last_event_index,
            snapshot_hash=self.snapshot_hash(),
            audit_hash=self.audit_hash(),
        )

    def rollback(self, checkpoint: MemoryCheckpoint, *, persist: bool = True) -> None:
        self._active = {
            str(row["precedent_key"]): AtomicRepairItem.from_json(row)
            for row in checkpoint.active_view
        }
        self.audit_entries = list(checkpoint.audit_entries)
        self.last_event_index = checkpoint.last_event_index
        self._rebuild_indexes()
        if persist:
            self._rewrite_audit_log()

    def commit_transaction(
        self,
        items: Sequence[AtomicRepairItem],
        *,
        exact_evidence: Mapping[str, str],
        event_index: int,
        stream_id: str,
    ) -> TransactionReceipt:
        reasons = self._transaction_rejection_reasons(items, exact_evidence, event_index)
        transaction_id = sha256_json(
            {
                "stream_id": stream_id,
                "event_index": event_index,
                "item_hashes": [item.item_hash for item in items],
            }
        )
        if reasons:
            return TransactionReceipt(
                committed=False,
                accepted_count=0,
                rejected_count=len(items),
                rejection_reasons=reasons,
                active_view_hash=self.active_view_hash(),
                audit_hash=self.audit_hash(),
                transaction_id=transaction_id,
            )
        for item in items:
            version = len(self._history_by_key[item.precedent_key]) + 1
            state = "repromoted" if self._revoked_support_hashes[item.precedent_key] else "active"
            active_item = item.active_version(version)
            self._active[item.precedent_key] = active_item
            self._append_audit_entry(
                action="activate",
                state=state,
                item=active_item,
                event_index=event_index,
                stream_id=stream_id,
                exact_evidence_hash=item.exact_evidence_hash,
                transaction_id=transaction_id,
            )
        self.last_event_index = event_index
        return TransactionReceipt(
            committed=True,
            accepted_count=len(items),
            rejected_count=0,
            rejection_reasons=[],
            active_view_hash=self.active_view_hash(),
            audit_hash=self.audit_hash(),
            transaction_id=transaction_id,
        )

    def revoke(
        self,
        precedent_key: str,
        *,
        exact_evidence_hash: str,
        event_index: int,
        stream_id: str,
    ) -> TransactionReceipt:
        transaction_id = sha256_json(
            {
                "stream_id": stream_id,
                "event_index": event_index,
                "precedent_key": precedent_key,
                "action": "revoke",
            }
        )
        reasons = []
        if event_index <= self.last_event_index:
            reasons.append("time_reversal")
        if precedent_key not in self._active:
            reasons.append("missing_active_precedent")
        if reasons:
            return TransactionReceipt(
                committed=False,
                accepted_count=0,
                rejected_count=1,
                rejection_reasons=reasons,
                active_view_hash=self.active_view_hash(),
                audit_hash=self.audit_hash(),
                transaction_id=transaction_id,
            )
        item = self._active.pop(precedent_key)
        self._revoked_support_hashes[precedent_key].add(item.exact_evidence_hash)
        version = len(self._history_by_key[precedent_key]) + 1
        revoked_item = replace(item, version=version, state="revoked")
        self._append_audit_entry(
            action="revoke",
            state="revoked",
            item=revoked_item,
            event_index=event_index,
            stream_id=stream_id,
            exact_evidence_hash=exact_evidence_hash,
            transaction_id=transaction_id,
        )
        self.last_event_index = event_index
        return TransactionReceipt(
            committed=True,
            accepted_count=1,
            rejected_count=0,
            rejection_reasons=[],
            active_view_hash=self.active_view_hash(),
            audit_hash=self.audit_hash(),
            transaction_id=transaction_id,
        )

    def retrieve(
        self,
        precedent_key: str,
        *,
        exact_evidence: Mapping[str, str],
    ) -> RetrievalReceipt:
        if precedent_key in self._active:
            item = self._active[precedent_key]
            if exact_evidence.get(item.exact_evidence_key) == item.exact_evidence_hash:
                return RetrievalReceipt([item], 1, 0, 0, 0)
            return RetrievalReceipt([], 0, 0, 0, 1)
        if self._revoked_support_hashes.get(precedent_key):
            return RetrievalReceipt([], 0, 1, 0, 0)
        return RetrievalReceipt([], 0, 0, 0, 0)

    def state_counts(self) -> dict[str, int]:
        counts = Counter(str(entry["state"]) for entry in self.audit_entries)
        return {
            "active": counts.get("active", 0),
            "revoked": counts.get("revoked", 0),
            "repromoted": counts.get("repromoted", 0),
        }

    def active_view_hash(self) -> str:
        return sha256_json(self._active_json())

    def audit_hash(self) -> str:
        return sha256_json(self.audit_entries)

    def snapshot_hash(self) -> str:
        return sha256_json(
            {
                "active_view": self._active_json(),
                "audit_entries": self.audit_entries,
                "last_event_index": self.last_event_index,
            }
        )

    def _transaction_rejection_reasons(
        self,
        items: Sequence[AtomicRepairItem],
        exact_evidence: Mapping[str, str],
        event_index: int,
    ) -> list[str]:
        reasons: list[str] = []
        if event_index <= self.last_event_index:
            reasons.append("time_reversal")
        by_key: dict[str, str] = {}
        for item in items:
            prior_hash = by_key.setdefault(item.precedent_key, item.item_hash)
            if prior_hash != item.item_hash and "key_collision" not in reasons:
                reasons.append("key_collision")
            if not item.atomic and "bundled_repair" not in reasons:
                reasons.append("bundled_repair")
            if item.poisoned and "poison" not in reasons:
                reasons.append("poison")
            if exact_evidence.get(item.exact_evidence_key) != item.exact_evidence_hash:
                if "unsupported_exact_evidence" not in reasons:
                    reasons.append("unsupported_exact_evidence")
            if item.exact_evidence_hash in self._revoked_support_hashes[item.precedent_key]:
                if "stale_resurrection" not in reasons:
                    reasons.append("stale_resurrection")
            active = self._active.get(item.precedent_key)
            if active is not None and active.item_hash != item.item_hash:
                if "active_conflict" not in reasons:
                    reasons.append("active_conflict")
        return reasons

    def _append_audit_entry(
        self,
        *,
        action: str,
        state: str,
        item: AtomicRepairItem,
        event_index: int,
        stream_id: str,
        exact_evidence_hash: str,
        transaction_id: str,
    ) -> None:
        entry = {
            "seq": len(self.audit_entries) + 1,
            "action": action,
            "state": state,
            "event_index": event_index,
            "stream_id": stream_id,
            "precedent_key": item.precedent_key,
            "version": item.version,
            "exact_evidence_hash": exact_evidence_hash,
            "transaction_id": transaction_id,
            "item": item.to_json(),
        }
        self.audit_entries.append(entry)
        self._history_by_key[item.precedent_key].append(entry)
        if self.audit_log_path is not None:
            with self.audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(entry) + "\n")

    def _apply_audit_entry(self, entry: JsonDict) -> None:
        item = AtomicRepairItem.from_json(dict(entry["item"]))
        key = item.precedent_key
        if entry["state"] in ("active", "repromoted"):
            self._active[key] = replace(item, state="active")
        elif entry["state"] == "revoked":
            self._active.pop(key, None)
            self._revoked_support_hashes[key].add(item.exact_evidence_hash)
        else:
            raise ValueError(f"unknown audit state: {entry['state']}")
        self.audit_entries.append(entry)
        self._history_by_key[key].append(entry)
        self.last_event_index = max(self.last_event_index, int(entry["event_index"]))

    def _active_json(self) -> list[JsonDict]:
        return [self._active[key].to_json() for key in sorted(self._active)]

    def _rebuild_indexes(self) -> None:
        active_rows = self._active_json()
        entries = list(self.audit_entries)
        self._active = {}
        self._history_by_key = defaultdict(list)
        self._revoked_support_hashes = defaultdict(set)
        self.audit_entries = []
        self.last_event_index = -1
        for entry in entries:
            self._apply_audit_entry(dict(entry))
        for row in active_rows:
            item = AtomicRepairItem.from_json(row)
            self._active[item.precedent_key] = item

    def _rewrite_audit_log(self) -> None:
        if self.audit_log_path is None:
            return
        text = "".join(canonical_json(entry) + "\n" for entry in self.audit_entries)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.write_text(text, encoding="utf-8")
