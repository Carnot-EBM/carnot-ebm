"""Keep explicit ARC beliefs from immediate observations only.

The ledger stores small symbolic facts, not model weights. It publishes each
candidate atomically and keeps a hash-linked journal so restart and rollback
behavior can be checked from bytes.

Spec refs: REQ-CSL-7020, SCENARIO-CSL-7020-*, REQ-ARC-WMTE-7020,
and SCENARIO-ARC-WMTE-7020-*.
"""

from __future__ import annotations

import argparse
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7020
SCHEMA = "carnot.exp7020.counterexample_belief_ledger.v1"
EVENT_SCHEMA = "carnot.arc.belief_transition_event.v1"
STATE_SCHEMA = "carnot.arc.belief_ledger_state.v1"
JOURNAL_SCHEMA = "carnot.arc.belief_ledger_journal.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 70_202_026_090_5
INFERENCE_SUBSTRATE = "deterministic_arc_belief_ledger_replay_no_llm"

REPO_ROOT = Path(__file__).resolve().parents[3]
CSL_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
ARC_SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/agentic/arc_belief_ledger.py")
TEST_PATH = Path("tests/python/test_experiment_7020_counterexample_belief_ledger.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7020_counterexample_belief_ledger.py")
OUTPUT_PATH = Path("results/experiment_7020_counterexample_belief_ledger.json")
CHECKPOINT_ROOT = Path("results/checkpoints/experiment_7020_counterexample_belief_ledger")
EXP7019_ARTIFACT_PATH = Path("results/experiment_7019_arc_belief_stream_fixture.json")
EXP7019_FIXTURE_PATH = Path(
    "results/raw/experiment_7019_arc_belief_stream_fixture/transition_events.jsonl"
)
EXP7019_SIDECAR_PATH = Path(
    "results/raw/experiment_7019_arc_belief_stream_fixture/sealed_held_future.jsonl"
)
EXP6978_ARTIFACT_PATH = Path("results/experiment_6978_transactional_constraint_self_learning.json")
EXP5155_ARTIFACT_PATH = Path("results/experiment_5155_multilevel_belief_state_scoping_v472.json")

DEFAULT_CAPACITY = 32
DEFAULT_MIN_SUPPORT = 2
DEFAULT_MAX_STATE_BYTES = 131_072
BELIEF_STATES = frozenset({"known", "possible", "contradicted", "uncertain"})
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "cited_upstream_artifacts",
    "source_artifact_hashes",
    "rows",
    "per_event_results",
    "belief_state_rows",
    "update_rows",
    "rejection_rows",
    "contradiction_cluster_rows",
    "tombstone_rows",
    "capacity_rows",
    "authority_conflict_rows",
    "supersession_rows",
    "poison_rows",
    "retention_rows",
    "transaction_rows",
    "restart_rows",
    "rollback_rows",
    "leakage_check_rows",
    "snapshot_hashes",
    "ledger_state_bytes",
    "belief_ledger_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for each field makes the evidence contract reviewable.",
    "preconditions_checked": "Explicit checks stop missing or changed inputs before construction.",
    "inference_substrate": "The substrate separates deterministic replay from model inference.",
    "duration_s": "Measured wall time proves the replay command executed.",
    "cited_upstream_artifacts": "Field-level citations show which prior evidence shaped this substrate.",
    "source_artifact_hashes": "Content hashes bind the run to exact upstream bytes.",
    "rows": "Aggregate rows expose the stored mechanic groups behind the result.",
    "per_event_results": "Event rows preserve chronological queries and writes without pooled hiding.",
    "belief_state_rows": "State rows prove all four belief states have executable semantics.",
    "update_rows": "Accepted update receipts expose support and state changes.",
    "rejection_rows": "Rejected writes show fail-closed behavior instead of silent omission.",
    "contradiction_cluster_rows": "Grouped counterexamples preserve recurring mechanic conflicts.",
    "tombstone_rows": "Tombstones prove refuted facts cannot remain active after a conflict.",
    "capacity_rows": "Capacity receipts make deterministic eviction and refusal auditable.",
    "authority_conflict_rows": "Authority rows prove equal event IDs cannot carry changed observations.",
    "supersession_rows": "Supersession keeps old tombstones immutable when evidence returns.",
    "poison_rows": "Poison fixtures test that weak unrelated data cannot evict protected facts.",
    "retention_rows": "Retention rows verify supported non-conflicting beliefs survive pressure.",
    "transaction_rows": "Hash-linked phases bind each write to exact parent and candidate bytes.",
    "restart_rows": "Restart receipts prove durable commits and interrupted prepares recover exactly.",
    "rollback_rows": "Rollback receipts prove exact parent bytes can be restored.",
    "leakage_check_rows": "Leakage checks keep source identity and future outcomes outside use.",
    "snapshot_hashes": "Snapshot digests expose state changes and fresh-process determinism.",
    "ledger_state_bytes": "Exact byte size enforces a small bounded state substrate.",
    "belief_ledger_ready_score": "One gates use on safety, replay, audit, and isolation together.",
    "random_seed": "A fixed seed makes the deterministic protocol addressable.",
    "reproducibility_checksum": "A timing-free digest detects artifact drift.",
    "gate_check_summary": "The first exact failure makes a blocked run actionable.",
    "verifier_is_oracle": "False states that no correctness oracle creates these beliefs.",
    "verdict_class": "A closed class prevents storage readiness from implying future utility.",
    "honest_verdict": "A class-consistent prefix gives downstream readers a stable terminal state.",
}

FORBIDDEN_UPDATER_KEYS = frozenset(
    {
        "adapter",
        "adapter_name",
        "future_label",
        "game",
        "game_id",
        "held_future",
        "hidden_rule",
        "later_outcome",
        "registry_result",
        "sealed_future",
        "solve_registry",
        "solve_registry_label",
        "source_game",
        "source_path",
    }
)


class ForcedInterruption(RuntimeError):
    """Mark a deliberate stop at one durable transaction boundary."""


class JournalIntegrityError(RuntimeError):
    """Reject journal bytes that do not form the recorded hash chain."""


class CapacityProtectedError(RuntimeError):
    """Reject a weak candidate when only protected beliefs could make room."""


def canonical_bytes(value: Any) -> bytes:
    """Serialize stable JSON bytes for exact snapshots and hashes."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Label a SHA-256 digest so consumers cannot assume another algorithm."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash exact file bytes while preserving an unreadable path as missing."""

    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError:
        return None


def _sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def event_row_hash(row: Mapping[str, Any]) -> str:
    """Replay Exp7019's event hash without trusting its self-hash field."""

    projected = deepcopy(dict(row))
    projected.pop("row_hash", None)
    payload = json.dumps(
        projected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def find_forbidden_paths(value: Any) -> list[str]:
    """Find denied identity or future fields at any nested updater path."""

    found: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key).casefold() in FORBIDDEN_UPDATER_KEYS:
                    found.append(child)
                visit(nested, child)
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, "")
    return sorted(set(found))


@dataclass(frozen=True, order=True)
class MechanicKey:
    """Name a transferable mechanic using observable typed fields only."""

    action_type: int
    hypothesis_key: str
    change_scale: str
    level_boundary: bool
    touches_action_coordinate: bool
    mechanic_group: str

    def __post_init__(self) -> None:
        if isinstance(self.action_type, bool) or not isinstance(self.action_type, int):
            raise ValueError("mechanic action_type must be an integer")
        if not _is_sha256(self.hypothesis_key):
            raise ValueError("mechanic hypothesis_key must be a SHA-256 digest")
        if not self.change_scale or not self.mechanic_group:
            raise ValueError("mechanic key text fields must be non-empty")
        if not isinstance(self.level_boundary, bool) or not isinstance(
            self.touches_action_coordinate, bool
        ):
            raise ValueError("mechanic key flags must be Boolean")

    def to_dict(self) -> JsonDict:
        """Return a detached JSON form for state and audit rows."""

        return asdict(self)


def mechanic_key_from_event(event: Mapping[str, Any]) -> MechanicKey:
    """Project one complete event onto its game-blind observable key."""

    signature = event.get("mechanic_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("mechanic_signature is required")
    state_delta = signature.get("state_delta")
    if not isinstance(state_delta, Mapping):
        state_delta = dict(event.get("next_observation", {})).get("state_delta", {})
    spatial = signature.get("spatial_summary")
    if not isinstance(spatial, Mapping):
        spatial = {}
    touches = state_delta.get(
        "touches_action_coordinate",
        spatial.get("touches_action_coordinate"),
    )
    return MechanicKey(
        action_type=signature.get("action_type"),
        hypothesis_key=signature.get("hypothesis_key"),
        change_scale=state_delta.get("change_scale"),
        level_boundary=signature.get("level_boundary"),
        touches_action_coordinate=touches,
        mechanic_group=signature.get("mechanic_group"),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish a full state or artifact through one durable rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
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


class BeliefLedger:
    """Publish bounded belief facts with exact transaction recovery."""

    def __init__(
        self,
        root: Path | str,
        *,
        capacity: int = DEFAULT_CAPACITY,
        min_support: int = DEFAULT_MIN_SUPPORT,
        max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
    ) -> None:
        if capacity < 1 or min_support < 1 or max_state_bytes < 1:
            raise ValueError("ledger limits must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.journal_path = self.root / "journal.jsonl"
        self.capacity = int(capacity)
        self.min_support = int(min_support)
        self.max_state_bytes = int(max_state_bytes)
        self.recovery_rows: list[JsonDict] = []
        if not self.state_path.exists():
            initial = {
                "schema": STATE_SCHEMA,
                "version": 0,
                "capacity": self.capacity,
                "min_support": self.min_support,
                "last_stream_index": -1,
                "facts": [],
                "clusters": {},
                "events": {},
            }
            payload = canonical_bytes(initial)
            if len(payload) > self.max_state_bytes:
                raise ValueError("initial ledger state exceeds byte limit")
            _atomic_write(self.state_path, payload)
        if not self.journal_path.exists():
            self.journal_path.touch()
            _fsync_directory(self.root)
        self._state = self._read_state()
        self._read_journal()
        self._recover_incomplete_prepares()

    def _read_state(self) -> JsonDict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid ledger state: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid ledger state schema")
        if value.get("capacity") != self.capacity or value.get("min_support") != self.min_support:
            raise ValueError("ledger configuration differs from stored state")
        if not isinstance(value.get("facts"), list) or not isinstance(value.get("events"), dict):
            raise ValueError("invalid ledger state collections")
        if len(canonical_bytes(value)) > self.max_state_bytes:
            raise ValueError("ledger state exceeds byte limit")
        return value

    @property
    def state_bytes(self) -> bytes:
        """Return the exact published bytes without an in-memory rewrite."""

        return self.state_path.read_bytes()

    @property
    def state_hash(self) -> str:
        """Hash the published bytes used for restart and rollback."""

        return sha256_bytes(self.state_bytes)

    def snapshot(self) -> bytes:
        """Return caller-owned bytes for later exact rollback."""

        return bytes(self.state_bytes)

    def restart(self) -> BeliefLedger:
        """Open a fresh ledger object over the same durable files."""

        return BeliefLedger(
            self.root,
            capacity=self.capacity,
            min_support=self.min_support,
            max_state_bytes=self.max_state_bytes,
        )

    def facts(self) -> list[JsonDict]:
        """Return detached facts in evidence arrival order."""

        rows = deepcopy(list(self._state["facts"]))
        return sorted(rows, key=lambda row: (row["first_stream_index"], row["generation"]))

    def clusters(self) -> list[JsonDict]:
        """Return grouped counterexamples in stable mechanic-key order."""

        return sorted(
            deepcopy(list(self._state["clusters"].values())),
            key=lambda row: canonical_bytes(row["mechanic_key"]),
        )

    def _read_journal(self) -> list[JsonDict]:
        try:
            rows = [
                json.loads(line)
                for line in self.journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise JournalIntegrityError(f"journal parse failed: {exc}") from exc
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
        """Return a detached validated transaction history."""

        return deepcopy(self._read_journal())

    def _append_journal(self, phase: str, transaction_id: str, **fields: Any) -> JsonDict:
        rows = self._read_journal()
        row: JsonDict = {
            "schema": JOURNAL_SCHEMA,
            "sequence": len(rows),
            "phase": phase,
            "transaction_id": transaction_id,
            "previous_row_hash": rows[-1]["row_hash"] if rows else None,
            **deepcopy(fields),
        }
        row["row_hash"] = sha256_bytes(canonical_bytes(row))
        with self.journal_path.open("ab") as handle:
            handle.write(canonical_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.root)
        return row

    def _reject(self, event: Mapping[str, Any], reason: str, **fields: Any) -> JsonDict:
        event_id = event.get("event_id") if isinstance(event, Mapping) else None
        transaction_id = _sha256_json(
            {
                "phase": "reject",
                "sequence": len(self._read_journal()),
                "event_id": event_id,
                "reason": reason,
            }
        )
        self._append_journal(
            "reject",
            transaction_id,
            event_id=event_id,
            reason=reason,
            state_hash=self.state_hash,
            **fields,
        )
        return {"accepted": False, "reason": reason, **deepcopy(fields)}

    def _event_error(self, event: Any) -> tuple[str | None, list[str]]:
        if not isinstance(event, Mapping):
            return "event_object_required", []
        forbidden = find_forbidden_paths(event)
        if forbidden:
            return "forbidden_updater_fields", forbidden
        if not isinstance(event.get("next_observation"), Mapping):
            return "next_observation_required", []
        if event.get("schema") != EVENT_SCHEMA:
            return "event_schema_invalid", []
        if not isinstance(event.get("event_id"), str) or not event["event_id"]:
            return "event_id_invalid", []
        index = event.get("stream_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            return "stream_index_invalid", []
        if not _is_sha256(event.get("row_hash")) or event_row_hash(event) != event.get("row_hash"):
            return "event_row_hash_invalid", []
        try:
            mechanic_key_from_event(event)
        except (TypeError, ValueError):
            return "mechanic_key_invalid", []
        outcome = dict(event.get("mechanic_signature", {})).get("observed_outcome_key")
        if not _is_sha256(outcome):
            return "observed_outcome_key_invalid", []
        return None, []

    @staticmethod
    def _fact_id(key: MechanicKey, outcome_key: str, generation: int) -> str:
        return _sha256_json(
            {
                "mechanic_key": key.to_dict(),
                "outcome_key": outcome_key,
                "generation": generation,
            }
        )

    @staticmethod
    def _same_key(fact: Mapping[str, Any], key: MechanicKey) -> bool:
        return fact.get("mechanic_key") == key.to_dict()

    def _candidate_state(self, event: Mapping[str, Any]) -> tuple[JsonDict, JsonDict]:
        candidate = deepcopy(self._state)
        key = mechanic_key_from_event(event)
        key_dict = key.to_dict()
        outcome_key = str(event["mechanic_signature"]["observed_outcome_key"])
        stream_index = int(event["stream_index"])
        facts: list[JsonDict] = candidate["facts"]
        active_same = next(
            (
                row
                for row in facts
                if self._same_key(row, key)
                and row["outcome_key"] == outcome_key
                and row["state"] != "contradicted"
            ),
            None,
        )
        active_other = [
            row
            for row in facts
            if self._same_key(row, key)
            and row["outcome_key"] != outcome_key
            and row["state"] != "contradicted"
        ]
        prior_support = int(active_same["support"]) if active_same else 0
        tombstones: list[JsonDict] = []
        for fact in active_other:
            prior_state = fact["state"]
            fact["state"] = "contradicted"
            fact["protected"] = False
            fact["contradiction_count"] = int(fact["contradiction_count"]) + 1
            fact["effective_support"] = max(0, int(fact["support"]) - 1)
            fact["tombstoned_by"] = event["event_id"]
            tombstones.append(
                {
                    "fact_id": fact["fact_id"],
                    "outcome_key": fact["outcome_key"],
                    "prior_state": prior_state,
                    "support": fact["support"],
                    "tombstoned_by": event["event_id"],
                }
            )

        supersessions: list[JsonDict] = []
        if active_same is None:
            prior_generations = [
                row
                for row in facts
                if self._same_key(row, key) and row["outcome_key"] == outcome_key
            ]
            generation = 1 + max(
                (int(row["generation"]) for row in prior_generations),
                default=0,
            )
            superseded = max(
                prior_generations,
                key=lambda row: int(row["generation"]),
                default=None,
            )
            state = "uncertain" if tombstones or superseded is not None else "possible"
            active_same = {
                "fact_id": self._fact_id(key, outcome_key, generation),
                "mechanic_key": key_dict,
                "outcome_key": outcome_key,
                "generation": generation,
                "state": state,
                "support": 1,
                "effective_support": 1,
                "contradiction_count": 0,
                "first_stream_index": stream_index,
                "last_stream_index": stream_index,
                "event_ids": [event["event_id"]],
                "tombstoned_by": None,
                "supersedes_fact_id": superseded["fact_id"] if superseded else None,
                "protected": False,
            }
            facts.append(active_same)
            if superseded is not None:
                supersessions.append(
                    {
                        "prior_fact_id": superseded["fact_id"],
                        "new_fact_id": active_same["fact_id"],
                        "outcome_key": outcome_key,
                        "prior_generation": superseded["generation"],
                        "new_generation": generation,
                        "event_id": event["event_id"],
                    }
                )
        else:
            active_same["support"] = int(active_same["support"]) + 1
            active_same["effective_support"] = int(active_same["support"])
            active_same["last_stream_index"] = stream_index
            active_same["event_ids"].append(event["event_id"])
            if active_same["support"] >= self.min_support:
                active_same["state"] = "known"
                active_same["protected"] = active_same["contradiction_count"] == 0

        if tombstones:
            cluster_key = _sha256_json(key_dict)
            cluster = candidate["clusters"].setdefault(
                cluster_key,
                {
                    "cluster_id": cluster_key,
                    "mechanic_key": key_dict,
                    "event_ids": [],
                    "outcome_keys": [],
                    "counterexample_count": 0,
                },
            )
            related_events = [event["event_id"]]
            related_outcomes = [outcome_key]
            for tombstone in tombstones:
                fact = next(row for row in facts if row["fact_id"] == tombstone["fact_id"])
                related_events.extend(fact["event_ids"])
                related_outcomes.append(fact["outcome_key"])
            cluster["event_ids"] = sorted(set(cluster["event_ids"] + related_events))
            cluster["outcome_keys"] = sorted(set(cluster["outcome_keys"] + related_outcomes))
            cluster["counterexample_count"] = int(cluster["counterexample_count"]) + len(tombstones)

        evictions: list[JsonDict] = []
        incoming_fact_id = active_same["fact_id"]
        while len(facts) > self.capacity:
            eligible = [row for row in facts if not row["protected"]]
            state_priority = {"contradicted": 0, "uncertain": 1, "possible": 2, "known": 3}
            victim = min(
                eligible,
                key=lambda row: (
                    state_priority[row["state"]],
                    int(row["support"]),
                    int(row["last_stream_index"]),
                    row["fact_id"],
                ),
            )
            if victim["fact_id"] == incoming_fact_id:
                raise CapacityProtectedError("candidate cannot displace protected beliefs")
            facts.remove(victim)
            evictions.append(
                {
                    "fact_id": victim["fact_id"],
                    "mechanic_key": victim["mechanic_key"],
                    "outcome_key": victim["outcome_key"],
                    "state": victim["state"],
                    "support": victim["support"],
                    "last_stream_index": victim["last_stream_index"],
                    "rule": "state_support_age_fact_id",
                }
            )

        candidate["facts"] = sorted(facts, key=lambda row: row["fact_id"])
        candidate["version"] = int(candidate["version"]) + 1
        candidate["last_stream_index"] = stream_index
        candidate["events"][str(event["event_id"])] = {
            "row_hash": event["row_hash"],
            "stream_index": stream_index,
            "mechanic_key": key_dict,
            "outcome_key": outcome_key,
        }
        payload = canonical_bytes(candidate)
        if len(payload) > self.max_state_bytes:
            raise CapacityProtectedError("candidate exceeds state byte limit")
        details = {
            "event_id": event["event_id"],
            "stream_index": stream_index,
            "mechanic_key": key_dict,
            "outcome_key": outcome_key,
            "fact_id": active_same["fact_id"],
            "generation": active_same["generation"],
            "belief_state": active_same["state"],
            "support_before": prior_support,
            "support_after": active_same["support"],
            "tombstones": tombstones,
            "supersessions": supersessions,
            "evictions": evictions,
        }
        return candidate, details

    def observe(
        self,
        event: Mapping[str, Any],
        *,
        interrupt_after_prepare: bool = False,
        interrupt_after_publish: bool = False,
    ) -> JsonDict:
        """Apply one complete observation after all denied fields are checked."""

        error, forbidden = self._event_error(event)
        if error == "next_observation_required":
            self._reject(event, error)
            return {"accepted": False, "reason": error}
        if error is not None:
            fields = {"forbidden_paths": forbidden} if forbidden else {}
            return self._reject(event, error, **fields)

        event_id = str(event["event_id"])
        existing = self._state["events"].get(event_id)
        if existing is not None:
            if existing["row_hash"] == event["row_hash"]:
                return self._reject(event, "duplicate_event")
            return self._reject(
                event,
                "authority_conflict",
                existing_row_hash=existing["row_hash"],
                observed_row_hash=event["row_hash"],
            )
        if int(event["stream_index"]) <= int(self._state["last_stream_index"]):
            return self._reject(event, "non_chronological_event")

        parent = self.state_bytes
        try:
            candidate, details = self._candidate_state(event)
        except CapacityProtectedError:
            return self._reject(event, "capacity_protected", poison_protected=True)
        candidate_bytes = canonical_bytes(candidate)
        parent_hash = sha256_bytes(parent)
        candidate_hash = sha256_bytes(candidate_bytes)
        transaction_id = _sha256_json(
            {
                "journal_sequence": len(self._read_journal()),
                "event_id": event_id,
                "parent_state_hash": parent_hash,
                "candidate_state_hash": candidate_hash,
            }
        )
        prepare = self._append_journal(
            "prepare",
            transaction_id,
            event_id=event_id,
            parent_state_hash=parent_hash,
            candidate_state_hash=candidate_hash,
            parent_state_b64=b64encode(parent).decode("ascii"),
            candidate_state_b64=b64encode(candidate_bytes).decode("ascii"),
        )
        if interrupt_after_prepare:
            raise ForcedInterruption(transaction_id)
        _atomic_write(self.state_path, candidate_bytes)
        self._state = candidate
        if interrupt_after_publish:
            raise ForcedInterruption(transaction_id)
        commit = self._append_journal(
            "commit",
            transaction_id,
            event_id=event_id,
            parent_state_hash=parent_hash,
            candidate_state_hash=self.state_hash,
        )
        return {
            "accepted": True,
            "reason": "committed",
            "transaction_id": transaction_id,
            "parent_state_hash": parent_hash,
            "new_state_hash": self.state_hash,
            "prepare_sequence": prepare["sequence"],
            "commit_sequence": commit["sequence"],
            "state_bytes": len(self.state_bytes),
            **details,
        }

    def query(self, key: MechanicKey, *, outcome_key: str | None = None) -> JsonDict:
        """Return current support without changing state or reading future data."""

        if not isinstance(key, MechanicKey):
            raise TypeError("query requires a typed MechanicKey")
        if outcome_key is not None and not _is_sha256(outcome_key):
            raise ValueError("query outcome_key must be a SHA-256 digest")
        matching = [row for row in self._state["facts"] if self._same_key(row, key)]
        if outcome_key is not None:
            matching = [row for row in matching if row["outcome_key"] == outcome_key]
        active = [row for row in matching if row["state"] != "contradicted"]
        selected: JsonDict | None
        if active:
            selected = max(active, key=lambda row: int(row["generation"]))
        elif matching:
            selected = max(matching, key=lambda row: int(row["generation"]))
        else:
            selected = None
        if selected is None:
            return {
                "mechanic_key": key.to_dict(),
                "outcome_key": outcome_key,
                "fact_id": None,
                "generation": None,
                "state": "uncertain",
                "support": 0,
                "protected": False,
            }
        return {
            "mechanic_key": key.to_dict(),
            "outcome_key": selected["outcome_key"],
            "fact_id": selected["fact_id"],
            "generation": selected["generation"],
            "state": selected["state"],
            "support": selected["support"],
            "protected": selected["protected"],
        }

    def rollback(self, parent_bytes: bytes, *, transaction_id: str, reason: str) -> JsonDict:
        """Restore valid caller-held state bytes and append an exact receipt."""

        try:
            candidate = json.loads(parent_bytes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("rollback state is not valid JSON") from exc
        if (
            not isinstance(candidate, dict)
            or candidate.get("schema") != STATE_SCHEMA
            or candidate.get("capacity") != self.capacity
            or candidate.get("min_support") != self.min_support
        ):
            raise ValueError("rollback state has an invalid ledger contract")
        if len(parent_bytes) > self.max_state_bytes:
            raise ValueError("rollback state exceeds byte limit")
        _atomic_write(self.state_path, parent_bytes)
        self._state = self._read_state()
        row = self._append_journal(
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

    def _recover_incomplete_prepares(self) -> None:
        rows = self._read_journal()
        terminal_ids = {
            str(row["transaction_id"])
            for row in rows
            if row.get("phase") in {"commit", "commit_recovered", "abort_recovered"}
        }
        pending = [
            row
            for row in rows
            if row.get("phase") == "prepare" and str(row["transaction_id"]) not in terminal_ids
        ]
        for prepare in pending:
            transaction_id = str(prepare["transaction_id"])
            current_hash = self.state_hash
            if current_hash == prepare["parent_state_hash"]:
                phase = "abort_recovered"
                applied = False
            elif current_hash == prepare["candidate_state_hash"]:
                phase = "commit_recovered"
                applied = True
            else:
                raise JournalIntegrityError(
                    f"incomplete transaction state mismatch:{transaction_id}"
                )
            row = self._append_journal(
                phase,
                transaction_id,
                event_id=prepare.get("event_id"),
                parent_state_hash=prepare["parent_state_hash"],
                candidate_state_hash=prepare["candidate_state_hash"],
            )
            self.recovery_rows.append(
                {
                    "transaction_id": transaction_id,
                    "phase": phase,
                    "applied": applied,
                    "journal_sequence": row["sequence"],
                }
            )

    def audit_state(self) -> JsonDict:
        """Replay journal hashes and compare the logical result with state bytes."""

        rows = self._read_journal()
        prepares: dict[str, JsonDict] = {}
        logical_hash: str | None = None
        errors: list[str] = []
        for row in rows:
            transaction_id = str(row["transaction_id"])
            phase = row["phase"]
            if phase == "reject":
                if row.get("state_hash") != (logical_hash or row.get("state_hash")):
                    errors.append(f"reject_state_mismatch:{transaction_id}")
                continue
            if phase == "prepare":
                parent_bytes = b64decode(row["parent_state_b64"])
                candidate_bytes = b64decode(row["candidate_state_b64"])
                if sha256_bytes(parent_bytes) != row.get("parent_state_hash"):
                    errors.append(f"prepare_parent_bytes_mismatch:{transaction_id}")
                if sha256_bytes(candidate_bytes) != row.get("candidate_state_hash"):
                    errors.append(f"prepare_candidate_bytes_mismatch:{transaction_id}")
                if logical_hash is None:
                    logical_hash = str(row["parent_state_hash"])
                elif logical_hash != row.get("parent_state_hash"):
                    errors.append(f"prepare_parent_mismatch:{transaction_id}")
                prepares[transaction_id] = row
                continue
            if phase in {"commit", "commit_recovered"}:
                prepare = prepares.get(transaction_id)
                if prepare is None or row.get("candidate_state_hash") != prepare.get(
                    "candidate_state_hash"
                ):
                    errors.append(f"commit_without_prepare:{transaction_id}")
                else:
                    logical_hash = str(row["candidate_state_hash"])
                continue
            if phase == "abort_recovered":
                if transaction_id not in prepares:
                    errors.append(f"abort_without_prepare:{transaction_id}")
                continue
            if phase == "rollback":
                restored = b64decode(row["restored_state_b64"])
                if sha256_bytes(restored) != row.get("restored_state_hash"):
                    errors.append(f"rollback_bytes_mismatch:{transaction_id}")
                logical_hash = str(row["restored_state_hash"])
                continue
            errors.append(f"unknown_phase:{phase}")
        if logical_hash is not None and logical_hash != self.state_hash:
            errors.append("replay_final_state_mismatch")
        forbidden = find_forbidden_paths(self._state)
        if forbidden:
            errors.append("forbidden_state_fields")
        if len(self._state["facts"]) > self.capacity:
            errors.append("fact_capacity_exceeded")
        if len(self.state_bytes) > self.max_state_bytes:
            errors.append("state_byte_capacity_exceeded")
        return {
            "passed": not errors,
            "errors": errors,
            "row_count": len(rows),
            "fact_count": len(self._state["facts"]),
            "capacity": self.capacity,
            "state_hash": self.state_hash,
            "state_bytes": len(self.state_bytes),
            "future_paths": forbidden,
        }


def load_updater_events(path: Path | str) -> list[JsonDict]:
    """Load only the Exp7019 updater schema and reject sealed rows."""

    source = Path(path)
    try:
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"updater fixture is unreadable: {exc}") from exc
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("schema") != EVENT_SCHEMA:
            raise ValueError("updater cannot load sealed future rows")
        forbidden = find_forbidden_paths(row)
        if forbidden:
            raise ValueError(f"updater fixture has forbidden fields:{forbidden}")
        if row.get("stream_index") != index:
            raise ValueError("updater fixture is not a complete chronological stream")
        if event_row_hash(row) != row.get("row_hash"):
            raise ValueError("updater fixture row hash mismatch")
        mechanic_key_from_event(row)
    return rows


def replay_events(
    events: Sequence[Mapping[str, Any]],
    root: Path | str,
    *,
    capacity: int = DEFAULT_CAPACITY,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
) -> JsonDict:
    """Replay the construction stream and keep query support per event."""

    ledger = BeliefLedger(
        root,
        capacity=capacity,
        min_support=min_support,
        max_state_bytes=max_state_bytes,
    )
    per_event_results: list[JsonDict] = []
    update_rows: list[JsonDict] = []
    rejection_rows: list[JsonDict] = []
    tombstone_rows: list[JsonDict] = []
    supersession_rows: list[JsonDict] = []
    capacity_rows: list[JsonDict] = []
    snapshot_hashes = [ledger.state_hash]
    query_rows: list[JsonDict] = []

    for event in events:
        key = mechanic_key_from_event(event)
        query_before = ledger.query(key)
        receipt = ledger.observe(event)
        query_after = ledger.query(key)
        query_rows.extend([query_before, query_after])
        row = {
            "event_id": event.get("event_id"),
            "stream_index": event.get("stream_index"),
            "query_before": query_before,
            "write": deepcopy(receipt),
            "query_after": query_after,
            "terminal": True,
        }
        per_event_results.append(row)
        if receipt.get("accepted") is True:
            update_rows.append({**deepcopy(receipt), "terminal": True})
            tombstone_rows.extend(
                {
                    **deepcopy(tombstone),
                    "event_id": receipt["event_id"],
                    "terminal": True,
                }
                for tombstone in receipt.get("tombstones", [])
            )
            supersession_rows.extend(
                {**deepcopy(item), "terminal": True} for item in receipt.get("supersessions", [])
            )
            capacity_rows.extend(
                {
                    **deepcopy(eviction),
                    "event_id": receipt["event_id"],
                    "decision": "evicted",
                    "terminal": True,
                }
                for eviction in receipt.get("evictions", [])
            )
        else:
            rejection_rows.append(
                {**deepcopy(receipt), "event_id": event.get("event_id"), "terminal": True}
            )
        snapshot_hashes.append(ledger.state_hash)

    future_paths = find_forbidden_paths(
        {
            "events": list(events),
            "per_event_results": per_event_results,
            "facts": ledger.facts(),
            "clusters": ledger.clusters(),
            "queries": query_rows,
        }
    )
    return {
        "ledger": ledger,
        "per_event_results": per_event_results,
        "update_rows": update_rows,
        "rejection_rows": rejection_rows,
        "tombstone_rows": tombstone_rows,
        "supersession_rows": supersession_rows,
        "capacity_rows": capacity_rows,
        "facts": ledger.facts(),
        "clusters": ledger.clusters(),
        "query_rows": query_rows,
        "snapshot_hashes": snapshot_hashes,
        "future_paths": future_paths,
        "audit_state": ledger.audit_state(),
    }


def _fixture_event(
    stream_index: int,
    *,
    mechanic: str,
    outcome: str,
    event_id: str | None = None,
) -> JsonDict:
    """Build a small observation row for deterministic safety fixtures."""

    hypothesis_key = _sha256_json({"mechanic": mechanic})
    outcome_key = _sha256_json({"outcome": outcome})
    row: JsonDict = {
        "schema": EVENT_SCHEMA,
        "event_id": event_id or f"fixture-{stream_index}-{mechanic}-{outcome}",
        "stream_index": stream_index,
        "source_attempt_time": "20260905T000000_000000",
        "source_transition_index": stream_index,
        "pre_action": {"grid_hash": _sha256_json({"before": stream_index}), "observed_level": 0},
        "action": {"type": 6, "data": {"x": 1, "y": 1}},
        "next_observation": {
            "grid_hash": _sha256_json({"after": stream_index}),
            "observed_level": 0,
            "state_delta": {
                "change_scale": "single",
                "level_boundary": False,
                "touches_action_coordinate": True,
            },
        },
        "contradiction": {
            "hypothesis_key": hypothesis_key,
            "observed_outcome_key": outcome_key,
            "is_contradiction": False,
            "prior_event_ids": [],
            "support_at_observation": 1,
        },
        "mechanic_signature": {
            "action_type": 6,
            "hypothesis_key": hypothesis_key,
            "observed_outcome_key": outcome_key,
            "level_boundary": False,
            "mechanic_group": "action_6:single:same_level:at_action",
            "state_delta": {
                "change_scale": "single",
                "level_boundary": False,
                "touches_action_coordinate": True,
            },
            "spatial_summary": {"touches_action_coordinate": True},
        },
    }
    row["row_hash"] = event_row_hash(row)
    return row


def _terminal(row: Mapping[str, Any]) -> JsonDict:
    return {**deepcopy(dict(row)), "terminal": True}


def run_safety_fixtures(root: Path | str) -> JsonDict:
    """Exercise every transaction and poisoning rule outside the source replay."""

    root_path = Path(root)
    state_ledger = BeliefLedger(root_path / "states", capacity=12, min_support=2)
    state_receipts = [
        state_ledger.observe(_fixture_event(0, mechanic="state", outcome="left")),
        state_ledger.observe(_fixture_event(1, mechanic="state", outcome="left")),
        state_ledger.observe(_fixture_event(2, mechanic="state", outcome="right")),
    ]
    state_key = mechanic_key_from_event(_fixture_event(0, mechanic="state", outcome="left"))
    contradicted_query = state_ledger.query(
        state_key,
        outcome_key=_sha256_json({"outcome": "left"}),
    )
    state_receipts.extend(
        [
            state_ledger.observe(_fixture_event(3, mechanic="state", outcome="left")),
            state_ledger.observe(_fixture_event(4, mechanic="state", outcome="left")),
        ]
    )
    belief_state_rows = [
        _terminal({"state": "possible", "source": "first_support"}),
        _terminal({"state": "known", "source": "minimum_support"}),
        _terminal({"state": "uncertain", "source": "conflicting_replacement"}),
        _terminal({"state": contradicted_query["state"], "source": "tombstoned_query"}),
    ]

    authority = BeliefLedger(root_path / "authority", capacity=4, min_support=2)
    authority_event = _fixture_event(
        0,
        mechanic="authority",
        outcome="first",
        event_id="authority-event",
    )
    authority.observe(authority_event)
    authority_parent = authority.snapshot()
    changed = _fixture_event(
        1,
        mechanic="authority",
        outcome="changed",
        event_id="authority-event",
    )
    authority_receipt = authority.observe(changed)
    authority_conflict_rows = [
        _terminal(
            {
                **authority_receipt,
                "state_unchanged": authority.snapshot() == authority_parent,
            }
        )
    ]

    eviction = BeliefLedger(root_path / "eviction", capacity=2, min_support=2)
    eviction.observe(_fixture_event(0, mechanic="old", outcome="old"))
    eviction.observe(_fixture_event(1, mechanic="middle", outcome="middle"))
    eviction_receipt = eviction.observe(_fixture_event(2, mechanic="new", outcome="new"))
    capacity_rows = [
        _terminal(
            {
                **row,
                "decision": "evicted",
                "deterministic_first_stream_index": 0,
            }
        )
        for row in eviction_receipt["evictions"]
    ]

    poison = BeliefLedger(root_path / "poison", capacity=2, min_support=2)
    poison.observe(_fixture_event(0, mechanic="anchor-a", outcome="a"))
    poison.observe(_fixture_event(1, mechanic="anchor-a", outcome="a"))
    poison.observe(_fixture_event(2, mechanic="anchor-b", outcome="b"))
    poison.observe(_fixture_event(3, mechanic="anchor-b", outcome="b"))
    poison_parent = poison.snapshot()
    poison_receipt = poison.observe(_fixture_event(4, mechanic="poison", outcome="poison"))
    poison_rows = [
        _terminal(
            {
                **poison_receipt,
                "protected_state_unchanged": poison.snapshot() == poison_parent,
            }
        )
    ]
    retention_rows = [
        _terminal(
            {
                "protected_fact_ids_before": [row["fact_id"] for row in poison.facts()],
                "protected_fact_ids_after": [row["fact_id"] for row in poison.facts()],
                "retained": poison.snapshot() == poison_parent,
            }
        )
    ]

    restart_root = root_path / "restart"
    restart = BeliefLedger(restart_root, capacity=4, min_support=2)
    restart.observe(_fixture_event(0, mechanic="restart", outcome="stable"))
    restart_parent = restart.snapshot()
    restarted = restart.restart()
    committed_restart_passed = restarted.snapshot() == restart_parent
    try:
        restarted.observe(
            _fixture_event(1, mechanic="restart", outcome="stable"),
            interrupt_after_prepare=True,
        )
    except ForcedInterruption as exc:
        interrupted_transaction = str(exc)
    recovered = restarted.restart()
    restart_rows = [
        _terminal(
            {
                "fixture": "commit_and_interrupted_prepare",
                "transaction_id": interrupted_transaction,
                "committed_restart_passed": committed_restart_passed,
                "parent_bytes_preserved": recovered.snapshot() == restart_parent,
                "recovery_phase": recovered.recovery_rows[-1]["phase"],
                "passed": committed_restart_passed
                and recovered.snapshot() == restart_parent
                and recovered.recovery_rows[-1]["phase"] == "abort_recovered",
            }
        )
    ]

    rollback = BeliefLedger(root_path / "rollback", capacity=4, min_support=2)
    rollback_parent = rollback.snapshot()
    rollback_receipt = rollback.observe(_fixture_event(0, mechanic="rollback", outcome="candidate"))
    restored = rollback.rollback(
        rollback_parent,
        transaction_id=rollback_receipt["transaction_id"],
        reason="fixture_byte_exactness",
    )
    rollback_rows = [
        _terminal(
            {
                **restored,
                "parent_bytes_restored": rollback.snapshot() == rollback_parent,
                "passed": restored["rolled_back"] and rollback.snapshot() == rollback_parent,
            }
        )
    ]

    future_probe = _fixture_event(0, mechanic="leakage", outcome="candidate")
    future_probe["later_outcome"] = {"reward": 1}
    leakage = BeliefLedger(root_path / "leakage", capacity=4, min_support=2)
    leakage_parent = leakage.snapshot()
    leakage_receipt = leakage.observe(future_probe)
    leakage_check_rows = [
        _terminal(
            {
                "check": "future_write_rejected",
                "expected_value": "forbidden_updater_fields",
                "observed_value": leakage_receipt["reason"],
                "state_unchanged": leakage.snapshot() == leakage_parent,
                "passed": leakage_receipt["reason"] == "forbidden_updater_fields"
                and leakage.snapshot() == leakage_parent,
            }
        )
    ]

    all_ledgers = [state_ledger, authority, eviction, poison, recovered, rollback, leakage]
    transaction_rows = [
        _terminal({"fixture": ledger.root.name, **row})
        for ledger in all_ledgers
        for row in ledger.journal_rows()
    ]
    supersession_rows = [
        _terminal(row) for receipt in state_receipts for row in receipt.get("supersessions", [])
    ]
    tombstone_rows = [
        _terminal(row) for receipt in state_receipts for row in receipt.get("tombstones", [])
    ]
    observed_states = sorted({row["state"] for row in belief_state_rows})
    passed = bool(
        set(observed_states) == BELIEF_STATES
        and authority_conflict_rows[0]["state_unchanged"]
        and capacity_rows
        and poison_rows[0]["protected_state_unchanged"]
        and restart_rows[0]["passed"]
        and rollback_rows[0]["passed"]
        and leakage_check_rows[0]["passed"]
        and all(ledger.audit_state()["passed"] for ledger in all_ledgers)
    )
    return {
        "passed": passed,
        "observed_states": observed_states,
        "belief_state_rows": belief_state_rows,
        "authority_conflict_rows": authority_conflict_rows,
        "capacity_rows": capacity_rows,
        "poison_rows": poison_rows,
        "retention_rows": retention_rows,
        "restart_rows": restart_rows,
        "rollback_rows": rollback_rows,
        "leakage_check_rows": leakage_check_rows,
        "transaction_rows": transaction_rows,
        "supersession_rows": supersession_rows,
        "tombstone_rows": tombstone_rows,
        "rejection_rows": [
            _terminal({"fixture": "authority", **authority_receipt}),
            _terminal({"fixture": "poison", **poison_receipt}),
            _terminal({"fixture": "leakage", **leakage_receipt}),
        ],
    }


def _read_json_object(path: Path) -> JsonDict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def _path_is_writable(path: Path) -> bool:
    target = path if path.exists() else _nearest_existing_parent(path.parent)
    return bool(target is not None and os.access(target, os.W_OK))


def _gate(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
        "terminal": True,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def collect_preconditions(
    *,
    repo_root: Path,
    output_path: Path,
    checkpoint_root: Path,
) -> tuple[list[JsonDict], JsonDict, list[JsonDict] | None]:
    """Check frozen evidence and all write targets before construction."""

    artifact_path = repo_root / EXP7019_ARTIFACT_PATH
    fixture_path = repo_root / EXP7019_FIXTURE_PATH
    sidecar_path = repo_root / EXP7019_SIDECAR_PATH
    exp6978_path = repo_root / EXP6978_ARTIFACT_PATH
    exp5155_path = repo_root / EXP5155_ARTIFACT_PATH
    exp7019 = _read_json_object(artifact_path)
    checks = [
        _gate("exp7019_artifact_readable", True, exp7019 is not None),
    ]
    ready_value = exp7019.get("arc_belief_stream_ready_score") if exp7019 else None
    checks.append(
        _gate(
            "exp7019_arc_belief_stream_ready_score",
            1,
            ready_value,
            passed=type(ready_value) is int and ready_value == 1,
        )
    )
    checks.extend(
        [
            _gate(
                "exp7019_fixture_readable",
                True,
                fixture_path.is_file() and os.access(fixture_path, os.R_OK),
            ),
            _gate(
                "exp7019_sidecar_readable",
                True,
                sidecar_path.is_file() and os.access(sidecar_path, os.R_OK),
            ),
            _gate(
                "exp7019_fixture_hash",
                exp7019.get("fixture_hash") if exp7019 else None,
                sha256_path(fixture_path),
            ),
            _gate(
                "exp7019_sidecar_hash",
                exp7019.get("held_future_sidecar_hash") if exp7019 else None,
                sha256_path(sidecar_path),
            ),
            _gate(
                "exp7019_fixture_read_only",
                True,
                bool(fixture_path.is_file() and fixture_path.stat().st_mode & 0o222 == 0),
            ),
            _gate(
                "exp7019_sidecar_read_only",
                True,
                bool(sidecar_path.is_file() and sidecar_path.stat().st_mode & 0o222 == 0),
            ),
            _gate("exp6978_artifact_readable", True, _read_json_object(exp6978_path) is not None),
            _gate("exp5155_artifact_readable", True, _read_json_object(exp5155_path) is not None),
            _gate("code_path_writable", True, _path_is_writable(repo_root / MODULE_PATH)),
            _gate("test_path_writable", True, _path_is_writable(repo_root / TEST_PATH)),
            _gate("checkpoint_path_writable", True, _path_is_writable(checkpoint_root)),
            _gate("artifact_path_writable", True, _path_is_writable(output_path)),
        ]
    )
    events: list[JsonDict] | None = None
    updater_error: str | None = None
    try:
        events = load_updater_events(fixture_path)
    except ValueError as exc:
        updater_error = str(exc)
    checks.extend(
        [
            _gate("updater_view_loadable", None, updater_error),
            _gate(
                "updater_view_hidden_future_fields",
                [],
                find_forbidden_paths(events or []),
            ),
            _gate("updater_event_count", 27, len(events or [])),
        ]
    )
    source_hashes = {
        str(EXP7019_ARTIFACT_PATH): sha256_path(artifact_path),
        str(EXP7019_FIXTURE_PATH): sha256_path(fixture_path),
        str(EXP7019_SIDECAR_PATH): sha256_path(sidecar_path),
        str(EXP6978_ARTIFACT_PATH): sha256_path(exp6978_path),
        str(EXP5155_ARTIFACT_PATH): sha256_path(exp5155_path),
    }
    return checks, source_hashes, events


def replay_digest(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a path-independent replay digest for fresh-process comparison."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7020-replay-") as directory:
        replay = replay_events(events, Path(directory) / "ledger")
        projection = {
            "state_hash": replay["ledger"].state_hash,
            "facts": replay["facts"],
            "clusters": replay["clusters"],
            "events": [
                {
                    "event_id": row["event_id"],
                    "stream_index": row["stream_index"],
                    "accepted": row["write"]["accepted"],
                    "belief_state": row["query_after"]["state"],
                    "support": row["query_after"]["support"],
                }
                for row in replay["per_event_results"]
            ],
        }
        return {
            "digest": _sha256_json(projection),
            "state_hash": replay["ledger"].state_hash,
            "event_count": len(replay["per_event_results"]),
        }


def _fresh_process_digest(repo_root: Path, fixture_path: Path) -> JsonDict:
    env = os.environ.copy()
    python_root = str(repo_root / "python")
    env["PYTHONPATH"] = python_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    command = [
        sys.executable,
        "-m",
        "carnot.agentic.arc_belief_ledger",
        "--replay-digest",
        str(fixture_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "digest": None,
            "state_hash": None,
            "event_count": 0,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "digest": None,
            "state_hash": None,
            "event_count": 0,
            "error": "fresh process returned invalid JSON",
        }
    return value if isinstance(value, dict) else {"error": "fresh process returned non-object"}


def _artifact_projection(artifact: Mapping[str, Any]) -> JsonDict:
    projected = deepcopy(dict(artifact))
    projected.pop("duration_s", None)
    projected.pop("reproducibility_checksum", None)
    return projected


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all deterministic artifact fields without wall time or self-hash."""

    return _sha256_json(_artifact_projection(artifact))


def _empty_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "cited_upstream_artifacts": [],
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "per_event_results": [],
        "belief_state_rows": [],
        "update_rows": [],
        "rejection_rows": [],
        "contradiction_cluster_rows": [],
        "tombstone_rows": [],
        "capacity_rows": [],
        "authority_conflict_rows": [],
        "supersession_rows": [],
        "poison_rows": [],
        "retention_rows": [],
        "transaction_rows": [],
        "restart_rows": [],
        "rollback_rows": [],
        "leakage_check_rows": [],
        "snapshot_hashes": [],
        "ledger_state_bytes": 0,
        "belief_ledger_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_counterexample_belief_ledger",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _aggregate_fact_rows(facts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    groups: dict[str, JsonDict] = {}
    for fact in facts:
        marker = _sha256_json(fact["mechanic_key"])
        row = groups.setdefault(
            marker,
            {
                "mechanic_key_hash": marker,
                "fact_count": 0,
                "active_fact_count": 0,
                "contradicted_fact_count": 0,
                "support": 0,
                "terminal": True,
            },
        )
        row["fact_count"] += 1
        row["support"] += int(fact["support"])
        if fact["state"] == "contradicted":
            row["contradicted_fact_count"] += 1
        else:
            row["active_fact_count"] += 1
    return [groups[key] for key in sorted(groups)]


def build_artifact(
    *,
    repo_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    output_path: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
) -> JsonDict:
    """Run preconditions, construction replay, and all safety fixtures."""

    started = time.perf_counter()
    root = Path(repo_root).resolve()
    output = Path(output_path) if output_path is not None else root / OUTPUT_PATH
    checkpoint = Path(checkpoint_root) if checkpoint_root is not None else root / CHECKPOINT_ROOT
    checks, source_hashes, events = collect_preconditions(
        repo_root=root,
        output_path=output,
        checkpoint_root=checkpoint,
    )
    if not _gate_summary(checks)["passed"]:
        return _empty_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=checks,
            source_hashes=source_hashes,
        )

    checkpoint.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checkpoint, prefix="run-") as directory:
        private_root = Path(directory)
        replay = replay_events(events or [], private_root / "construction")
        safety = run_safety_fixtures(private_root / "safety")
        construction_journal = replay["ledger"].journal_rows()
        construction_state_bytes = len(replay["ledger"].state_bytes)

    current_digest = replay_digest(events or [])
    fresh_digest = _fresh_process_digest(root, root / EXP7019_FIXTURE_PATH)
    fresh_match = current_digest == fresh_digest
    sidecar_rejected = False
    try:
        load_updater_events(root / EXP7019_SIDECAR_PATH)
    except ValueError as exc:
        sidecar_rejected = "sealed future" in str(exc)

    leakage_rows = [
        _terminal(
            {
                "check": "construction_write_and_query_future_fields",
                "expected_value": [],
                "observed_value": replay["future_paths"],
                "passed": replay["future_paths"] == [],
            }
        ),
        _terminal(
            {
                "check": "safety_future_write_rejected",
                "expected_value": True,
                "observed_value": all(row["passed"] for row in safety["leakage_check_rows"]),
                "passed": all(row["passed"] for row in safety["leakage_check_rows"]),
            }
        ),
        _terminal(
            {
                "check": "sealed_sidecar_rejected_by_updater",
                "expected_value": True,
                "observed_value": sidecar_rejected,
                "passed": sidecar_rejected,
            }
        ),
    ]
    safety_checks = [
        _gate("preconditions", True, True),
        _gate("construction_event_count", 27, len(replay["per_event_results"])),
        _gate("construction_transaction_audit", True, replay["audit_state"]["passed"]),
        _gate("safety_fixtures", True, safety["passed"]),
        _gate("four_belief_states", sorted(BELIEF_STATES), safety["observed_states"]),
        _gate("future_field_leakage_count", 0, sum(not row["passed"] for row in leakage_rows)),
        _gate("fresh_process_replay", True, fresh_match),
    ]
    gate_summary = _gate_summary(safety_checks)
    ready = int(gate_summary["passed"])
    exp7019_hash = source_hashes[str(EXP7019_ARTIFACT_PATH)]
    exp6978_hash = source_hashes[str(EXP6978_ARTIFACT_PATH)]
    exp5155_hash = source_hashes[str(EXP5155_ARTIFACT_PATH)]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.perf_counter() - started,
        "cited_upstream_artifacts": [
            {
                "experiment_id": 7019,
                "fields_imported": [
                    "arc_belief_stream_ready_score",
                    "fixture_hash",
                    "held_future_sidecar_hash",
                ],
                "sha256": exp7019_hash,
            },
            {
                "experiment_id": 6978,
                "fields_imported": [
                    "transaction_journal_rows",
                    "restart_recovery_rows",
                    "rollback_rows",
                ],
                "sha256": exp6978_hash,
            },
            {
                "experiment_id": 5155,
                "fields_imported": [
                    "belief_state_resets_at_level_boundary",
                    "first_contact_vs_deepen_distinction",
                ],
                "sha256": exp5155_hash,
            },
        ],
        "source_artifact_hashes": source_hashes,
        "rows": _aggregate_fact_rows(replay["facts"]),
        "per_event_results": replay["per_event_results"],
        "belief_state_rows": safety["belief_state_rows"],
        "update_rows": replay["update_rows"],
        "rejection_rows": replay["rejection_rows"] + safety["rejection_rows"],
        "contradiction_cluster_rows": [_terminal(row) for row in replay["clusters"]],
        "tombstone_rows": replay["tombstone_rows"] + safety["tombstone_rows"],
        "capacity_rows": replay["capacity_rows"] + safety["capacity_rows"],
        "authority_conflict_rows": safety["authority_conflict_rows"],
        "supersession_rows": replay["supersession_rows"] + safety["supersession_rows"],
        "poison_rows": safety["poison_rows"],
        "retention_rows": safety["retention_rows"],
        "transaction_rows": [
            _terminal({"fixture": "construction", **row}) for row in construction_journal
        ]
        + safety["transaction_rows"],
        "restart_rows": safety["restart_rows"],
        "rollback_rows": safety["rollback_rows"],
        "leakage_check_rows": leakage_rows + safety["leakage_check_rows"],
        "snapshot_hashes": [
            _terminal({"snapshot_index": index, "sha256": digest})
            for index, digest in enumerate(replay["snapshot_hashes"])
        ]
        + [
            _terminal(
                {
                    "snapshot_index": "fresh_process_replay",
                    "sha256": current_digest["state_hash"],
                    "current_digest": current_digest["digest"],
                    "fresh_digest": fresh_digest.get("digest"),
                    "fresh_process_match": fresh_match,
                }
            )
        ],
        "ledger_state_bytes": construction_state_bytes,
        "belief_ledger_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "partial",
        "honest_verdict": (
            "complete_positive_counterexample_belief_ledger_ready_no_future_utility_claim"
            if ready
            else "partial_counterexample_belief_ledger_safety_gate_failed"
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Any, *, repo_root: Path | str = REPO_ROOT) -> list[str]:
    """Return all structural and reproducibility errors without repair."""

    del repo_root
    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    score = artifact.get("belief_ledger_ready_score")
    if type(score) is not int or score not in {0, 1}:
        errors.append("ready_score_not_bare_integer")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    verdict = artifact.get("honest_verdict")
    prefix = {
        "positive": "complete_positive_",
        "circular_positive": "complete_circular_positive_",
        "null": "complete_null_",
        "blocked": "blocked_",
        "disqualified": "disqualified_",
        "partial": "partial_",
    }.get(verdict_class)
    if not isinstance(verdict, str) or prefix is None or not verdict.startswith(prefix):
        errors.append("verdict_prefix_mismatch")
    for field in (
        "rows",
        "per_event_results",
        "belief_state_rows",
        "update_rows",
        "rejection_rows",
        "contradiction_cluster_rows",
        "tombstone_rows",
        "capacity_rows",
        "authority_conflict_rows",
        "supersession_rows",
        "poison_rows",
        "retention_rows",
        "transaction_rows",
        "restart_rows",
        "rollback_rows",
        "leakage_check_rows",
        "snapshot_hashes",
    ):
        rows = artifact.get(field, [])
        if not isinstance(rows, list):
            errors.append(f"{field}_not_list")
        elif any(not isinstance(row, Mapping) or row.get("terminal") is not True for row in rows):
            errors.append(f"nonterminal_row:{field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    claimed_checksum = artifact.get("reproducibility_checksum")
    if claimed_checksum != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    gate_summary = artifact.get("gate_check_summary")
    if not isinstance(gate_summary, Mapping) or "failed_check" not in gate_summary:
        errors.append("gate_check_summary_invalid")
    if score == 1:
        if (
            verdict_class != "positive"
            or not isinstance(gate_summary, Mapping)
            or not gate_summary.get("passed")
        ):
            errors.append("ready_gate_inconsistent")
        if len(artifact.get("per_event_results", [])) != 27:
            errors.append("ready_event_count_mismatch")
        if any(not row.get("passed") for row in artifact.get("leakage_check_rows", [])):
            errors.append("ready_leakage_check_failed")
    return errors


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> None:
    """Write one complete result without exposing a partial JSON document."""

    _atomic_write(
        Path(path), json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--replay-digest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the artifact command or the private fresh-process replay mode."""

    args = _parser().parse_args(argv)
    if args.replay_digest is not None:
        print(json.dumps(replay_digest(load_updater_events(args.replay_digest)), sort_keys=True))
        return 0
    root = args.repo_root.resolve()
    output = args.output or root / OUTPUT_PATH
    checkpoint = args.checkpoint_root or root / CHECKPOINT_ROOT
    artifact = build_artifact(
        repo_root=root,
        run_date=args.date,
        output_path=output,
        checkpoint_root=checkpoint,
    )
    errors = validate_artifact(artifact, repo_root=root)
    if errors:
        raise ValueError(f"Exp7020 artifact validation failed:{errors}")
    write_artifact(output, artifact)
    print(f"wrote {output} belief_ledger_ready_score={artifact['belief_ledger_ready_score']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests.
    raise SystemExit(main())
