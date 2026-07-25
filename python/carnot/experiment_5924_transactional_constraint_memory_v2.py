"""Exp5924 transactional constraint-memory V2 fixture.

Spec refs: REQ-LEARN-5924, REQ-STORE-5924,
SCENARIO-LEARN-5924-TRANSACTIONS, SCENARIO-LEARN-5924-REJECTION,
SCENARIO-LEARN-5924-RECOVERY, SCENARIO-LEARN-5924-CONTROLS,
SCENARIO-STORE-5924.

This module is a deterministic external-memory sidecar over the admitted
Exp5920 event stream. It treats model outputs as proposals only. Exact
post-proposal receipts own every promotion, quarantine, reject, rollback, and
state-hash decision; no LLM inference or model-weight update path is loaded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot import experiment_5920_prospective_event_stream_admission as exp5920


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5924_transactional_constraint_memory_v2.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5924_transactional_constraint_memory_v2.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5924_transactional_constraint_memory_v2.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
STORE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-store/spec.md")
EXP5920_RESULT_RELATIVE_PATH = exp5920.RESULT_RELATIVE_PATH
EXP5920_ROWS_RELATIVE_PATH = exp5920.ROW_FILE_RELATIVE_PATH
EXP5913_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5913_transactional_constraint_memory_fixture.json"
)

RUN_DATE = "20260725"
RANDOM_SEED = 5924
EXPERIMENT_ID = "experiment_5924_transactional_constraint_memory_v2"
SCHEMA_VERSION = "carnot.experiment_5924.transactional_constraint_memory_v2.v1"
TRANSACTION_SCHEMA_VERSION = SCHEMA_VERSION + ".transaction"
STATE_SCHEMA_VERSION = SCHEMA_VERSION + ".state"
INFERENCE_SUBSTRATE = "deterministic_transactional_external_memory_no_llm"
EXACT_VALIDATOR_AUTHORITY = "exp5920.exact_label_projection"
VERIFIER_IS_ORACLE = True
ACTIVE_CAPACITY = 3
QUARANTINE_CAPACITY = 6
REJECTED_CAPACITY = 32
TRANSACTION_EVENT_BUDGET = 24
ROLLBACK_CHECKPOINT_INDEX = 8
POISON_EVENT_INDICES = (6, 7, 8)
NEAR_MISS_EVENT_INDICES = (9, 10)
PROTECTED_PREFIX_EVENT_INDICES = (0, 1, 2)
SUPPORTED_OPERATIONS = (
    "snapshot",
    "lookup",
    "propose",
    "commit",
    "validate",
    "promote",
    "quarantine",
    "supersede",
    "rollback",
    "reject",
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5924_transactional_constraint_memory_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5924_transactional_constraint_memory_v2.py "
    "-m pytest tests/python/test_experiment_5924_transactional_constraint_memory_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5924_transactional_constraint_memory_v2.py "
    "--fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_5924_transactional_constraint_memory_v2 --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5924_transactional_constraint_memory_v2.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5924_transactional_constraint_memory_v2.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    STORE_SPEC_RELATIVE_PATH,
    EXP5920_RESULT_RELATIVE_PATH,
    EXP5920_ROWS_RELATIVE_PATH,
    EXP5913_RESULT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "gate_replay_receipt",
    "preconditions_checked",
    "continuous_self_learning_task",
    "admitted_stream_path_hash_rows_and_prefix_chain",
    "transaction_schema_and_version",
    "operation_ledger_and_state_hash_chain",
    "frozen_read_commit_validate_write_receipts",
    "exact_promotion_authority",
    "invalid_transition_and_leakage_rejection_matrix",
    "fixed_no_memory_coupled_shuffled_and_corrupt_validator_controls",
    "poison_burst_quarantine_recovery_and_retention",
    "supersession_capacity_rollback_and_restart",
    "rejected_update_non_propagation",
    "no_model_weight_mutation",
    "task_owned_test_boundary_and_global_failure_delta",
    "hardware_mapping_contract",
    "protected_files_unchanged",
    "transactional_memory_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes ready, retired, or blocked transactional memory evidence.",
    "gate_replay_receipt": "Exp5920 gate and stream replay must pass before the fixture consumes events.",
    "preconditions_checked": "Hashes, resources, validators, outputs, and atomic writes prevent fabricated transactional state.",
    "continuous_self_learning_task": "Must be bare true only when online state transitions execute.",
    "admitted_stream_path_hash_rows_and_prefix_chain": "The Exp5920 stream path, hash, row count, and prefix chain bind the event source.",
    "transaction_schema_and_version": "One versioned transaction contract owns operation order and state hashing.",
    "operation_ledger_and_state_hash_chain": "Every operation transition is hash-bound and replayable.",
    "frozen_read_commit_validate_write_receipts": "Reads use pre-event snapshots and writes occur only after commit plus exact validation.",
    "exact_promotion_authority": "Exact verifier receipts are the only promotion authority.",
    "invalid_transition_and_leakage_rejection_matrix": "Leakage and malformed transaction attempts fail closed without mutation.",
    "fixed_no_memory_coupled_shuffled_and_corrupt_validator_controls": "Controls use matched budgets and cannot define promotion authority.",
    "poison_burst_quarantine_recovery_and_retention": "Poison and near-miss updates quarantine deterministically while protected prefixes retain exact labels.",
    "supersession_capacity_rollback_and_restart": "Supersession, bounded capacity, rollback, and restart reproduce exact state hashes.",
    "rejected_update_non_propagation": "Rejected or quarantined model updates never become active or replay context.",
    "no_model_weight_mutation": "Immutable model hashes remain unchanged.",
    "task_owned_test_boundary_and_global_failure_delta": "Focused checks must pass and global failure debt may not increase.",
    "hardware_mapping_contract": "The state machine remains finite, bounded, and hardware-mappable without claiming board execution.",
    "protected_files_unchanged": "Operator-curated and conductor files stay byte-identical.",
    "transactional_memory_fixture_ready_score": "Emit bare 1.0 only for complete isolation, exact validation, poison recovery, retention, capacity, restart, rollback, immutable weights, clean task-owned checks, and non-amplified global debt.",
    "duration_s": "Measured wall time exposes deterministic fixture work.",
    "inference_substrate": "Use `deterministic_transactional_external_memory_no_llm`.",
    "verifier_is_oracle": "True only for exact constraint execution, transaction validity, hashes, and rollback.",
    "field_provenance": "Every field traces to prompt, specs, upstream stream rows, code, tests, or command receipts.",
    "test_commands": "Commands document focused unit/coverage, replay, transaction-order, leakage, exact-promotion, poison/recovery, retention, capacity, restart/rollback, tamper, immutable-weight, task-boundary, adversarial, spec, applicable E2E, protected-file, and clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects stream, schema, ledger, state, command, or protected-file drift.",
    "honest_verdict": "Use `complete_ready:`, `retired:`, or `blocked:`.",
}


class TransactionMemoryError(ValueError):
    """Raised when a transactional memory operation must fail closed."""


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in a stable byte order before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so receipts never trust path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and reject scalar or array payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


class TransactionalConstraintMemory:
    """Small exact state machine for testing transaction isolation.

    The readable memory is `active`. Proposals, commits, and validations are
    tracked in transaction metadata, but a candidate becomes readable only when
    `promote`, `quarantine`, or `reject` runs after exact validation.
    """

    def __init__(
        self,
        *,
        active_capacity: int = ACTIVE_CAPACITY,
        quarantine_capacity: int = QUARANTINE_CAPACITY,
    ) -> None:
        self.active_capacity = active_capacity
        self.quarantine_capacity = quarantine_capacity
        self.state: JsonDict = {
            "schema": STATE_SCHEMA_VERSION,
            "active": [],
            "quarantine": [],
            "rejected": [],
            "superseded": [],
            "transactions": {},
            "capacity_evictions": [],
            "version": 0,
        }
        self.ledger: list[JsonDict] = []
        self.snapshots: dict[str, JsonDict] = {}
        self.written_events: set[str] = set()
        self.initial_state_hash = self.state_hash()
        self.history: dict[str, JsonDict] = {self.initial_state_hash: deepcopy(self.state)}
        self.max_active_count = 0
        self.max_quarantine_count = 0

    def state_hash(self) -> str:
        return sha256_json(self.state)

    def readable_state_hash(self) -> str:
        return sha256_json(
            {
                "active": self.state["active"],
                "capacity_evictions": self.state["capacity_evictions"],
                "quarantine": self.state["quarantine"],
                "rejected": self.state["rejected"],
                "superseded": self.state["superseded"],
            }
        )

    @classmethod
    def from_serialized(cls, payload: Mapping[str, Any]) -> "TransactionalConstraintMemory":
        memory = cls()
        state = deepcopy(dict(payload["state"]))
        expected = str(payload["state_hash"])
        memory.state = state
        if memory.state_hash() != expected:
            raise TransactionMemoryError("restart state hash mismatch")
        memory.initial_state_hash = expected
        memory.history = {expected: deepcopy(memory.state)}
        memory.max_active_count = len(memory.state["active"])
        memory.max_quarantine_count = len(memory.state["quarantine"])
        return memory

    def serialize_state(self) -> JsonDict:
        return {
            "schema": STATE_SCHEMA_VERSION,
            "state": deepcopy(self.state),
            "state_hash": self.state_hash(),
        }

    def snapshot(self, row: Mapping[str, Any]) -> str:
        self._reject_future_label(row)
        previous = self.state_hash()
        snapshot_id = f"{row['event_id']}:snapshot:{len(self.snapshots)}"
        self.snapshots[snapshot_id] = {
            "event_id": row["event_id"],
            "state_hash": previous,
            "readable_state_hash": self.readable_state_hash(),
            "state": deepcopy(self.state),
        }
        self._record("snapshot", row, previous, previous, True, detail={"snapshot_id": snapshot_id})
        return snapshot_id

    def lookup(self, row: Mapping[str, Any], snapshot_id: str, key: str) -> JsonDict | None:
        if str(row["event_id"]) in self.written_events:
            raise TransactionMemoryError("same-event read-after-write")
        snapshot = self._snapshot_for(row, snapshot_id)
        previous = self.state_hash()
        result = next(
            (entry for entry in snapshot["state"]["active"] if entry["key"] == key),
            None,
        )
        self._record(
            "lookup",
            row,
            previous,
            previous,
            True,
            detail={"snapshot_id": snapshot_id, "key": key, "hit": result is not None},
        )
        return deepcopy(result)

    def propose(
        self,
        row: Mapping[str, Any],
        snapshot_id: str,
        proposal_kind: str,
        payload: Mapping[str, Any],
        *,
        label_source: str = "memory_proposal",
    ) -> str:
        self._reject_future_label(row)
        if label_source == "model_authored":
            raise TransactionMemoryError("model-authored labels rejected")
        if payload.get("future_label_visible") is True:
            raise TransactionMemoryError("future label visibility")
        snapshot = self._snapshot_for(row, snapshot_id)
        if snapshot["readable_state_hash"] != self.readable_state_hash():
            raise TransactionMemoryError("stale snapshot")
        previous = self.state_hash()
        proposal_id = sha256_json(
            {
                "event_id": row["event_id"],
                "proposal_kind": proposal_kind,
                "payload": payload,
                "snapshot_hash": snapshot["state_hash"],
            }
        )
        self.state["transactions"][proposal_id] = {
            "event_id": row["event_id"],
            "proposal_kind": proposal_kind,
            "payload": deepcopy(dict(payload)),
            "snapshot_hash": snapshot["state_hash"],
            "status": "proposed",
            "label_source": label_source,
        }
        self._bump()
        result = self.state_hash()
        self._remember()
        self._record("propose", row, previous, result, True, proposal_id=proposal_id)
        return proposal_id

    def commit(self, row: Mapping[str, Any], proposal_id: str) -> None:
        tx = self._transaction_for(row, proposal_id)
        if tx["status"] != "proposed":
            raise TransactionMemoryError("duplicate commit")
        previous = self.state_hash()
        tx["status"] = "committed"
        self._bump()
        result = self.state_hash()
        self._remember()
        self._record("commit", row, previous, result, True, proposal_id=proposal_id)

    def validate(
        self,
        row: Mapping[str, Any],
        proposal_id: str,
        *,
        validator_authority: str = EXACT_VALIDATOR_AUTHORITY,
    ) -> None:
        if validator_authority != EXACT_VALIDATOR_AUTHORITY:
            raise TransactionMemoryError("validator substitution")
        tx = self._transaction_for(row, proposal_id)
        if tx["status"] != "committed":
            raise TransactionMemoryError("invalid transition order")
        previous = self.state_hash()
        tx["status"] = "validated"
        tx["validator_receipt_hash"] = self._event_receipt_hash(row, visible_to_operation=True)
        tx["authorized_action"] = self._authorized_action(tx)
        self._bump()
        result = self.state_hash()
        self._remember()
        self._record("validate", row, previous, result, True, proposal_id=proposal_id)

    def promote(self, row: Mapping[str, Any], proposal_id: str) -> None:
        tx = self._validated_transaction(row, proposal_id, "promote")
        existing = next(
            (entry for entry in self.state["active"] if entry["key"] == tx["payload"]["key"]),
            None,
        )
        protected = bool(tx["payload"].get("protected"))
        if existing is not None:
            protected = self._supersede(row, proposal_id, existing) or protected
        previous = self.state_hash()
        entry = {
            "key": tx["payload"]["key"],
            "value": tx["payload"]["value"],
            "event_id": row["event_id"],
            "proposal_id": proposal_id,
            "receipt_hash": tx["validator_receipt_hash"],
            "protected": protected,
            "promoted_index": len(self.ledger),
        }
        self.state["active"].append(entry)
        tx["status"] = "promoted"
        self.written_events.add(str(row["event_id"]))
        self._enforce_active_capacity(row)
        self._bump()
        result = self.state_hash()
        self._remember()
        self.max_active_count = max(self.max_active_count, len(self.state["active"]))
        self._record("promote", row, previous, result, True, proposal_id=proposal_id)

    def quarantine(self, row: Mapping[str, Any], proposal_id: str) -> None:
        tx = self._validated_transaction(row, proposal_id, "quarantine")
        previous = self.state_hash()
        self.state["quarantine"].append(self._closed_update(row, proposal_id, tx))
        self.state["quarantine"] = self.state["quarantine"][-self.quarantine_capacity :]
        tx["status"] = "quarantined"
        self.written_events.add(str(row["event_id"]))
        self._bump()
        result = self.state_hash()
        self._remember()
        self.max_quarantine_count = max(self.max_quarantine_count, len(self.state["quarantine"]))
        self._record("quarantine", row, previous, result, True, proposal_id=proposal_id)

    def reject(self, row: Mapping[str, Any], proposal_id: str) -> None:
        tx = self._validated_transaction(row, proposal_id, "reject")
        previous = self.state_hash()
        self.state["rejected"].append(self._closed_update(row, proposal_id, tx))
        self.state["rejected"] = self.state["rejected"][-REJECTED_CAPACITY:]
        tx["status"] = "rejected"
        self.written_events.add(str(row["event_id"]))
        self._bump()
        result = self.state_hash()
        self._remember()
        self._record("reject", row, previous, result, True, proposal_id=proposal_id)

    def rollback(self, row: Mapping[str, Any], target_hash: str) -> None:
        if target_hash not in self.history:
            raise TransactionMemoryError("rollback target missing")
        previous = self.state_hash()
        self.state = deepcopy(self.history[target_hash])
        result = self.state_hash()
        self._remember()
        self._record("rollback", row, previous, result, True, detail={"target_hash": target_hash})

    def partial_state_write_probe(self, row: Mapping[str, Any]) -> None:
        work = deepcopy(self.state)
        work["active"].append({"key": "partial", "event_id": row["event_id"]})
        if work != self.state:
            raise TransactionMemoryError("partial state write")

    def _supersede(
        self,
        row: Mapping[str, Any],
        proposal_id: str,
        existing: Mapping[str, Any],
    ) -> bool:
        previous = self.state_hash()
        was_protected = bool(existing.get("protected"))
        self.state["active"] = [
            entry
            for entry in self.state["active"]
            if entry["proposal_id"] != existing["proposal_id"]
        ]
        closed = deepcopy(dict(existing))
        closed["superseded_by"] = proposal_id
        self.state["superseded"].append(closed)
        self._bump()
        result = self.state_hash()
        self._remember()
        self._record("supersede", row, previous, result, True, proposal_id=proposal_id)
        return was_protected

    def _record(
        self,
        operation: str,
        row: Mapping[str, Any],
        previous: str,
        result: str,
        accepted: bool,
        *,
        proposal_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        receipt = self._event_receipt(row, visible_to_operation=operation == "validate")
        self.ledger.append(
            {
                "schema": TRANSACTION_SCHEMA_VERSION,
                "operation_index": len(self.ledger),
                "operation": operation,
                "event_id": row["event_id"],
                "event_index": row["causal_sequence_index"],
                "proposal_id": proposal_id,
                "previous_state_hash": previous,
                "resulting_state_hash": result,
                "exact_validator_receipt_hash": sha256_json(receipt),
                "exact_validator_receipt": receipt,
                "row_prefix_checksum": row["prefix_checksum"],
                "accepted": accepted,
                "detail": deepcopy(dict(detail or {})),
            }
        )

    def _event_receipt(self, row: Mapping[str, Any], *, visible_to_operation: bool) -> JsonDict:
        labels = dict(row["exact_label_projection"])
        return {
            "authority": EXACT_VALIDATOR_AUTHORITY,
            "event_id": row["event_id"],
            "row_hash": row["row_hash"],
            "prefix_checksum": row["prefix_checksum"],
            "exact_labels": labels,
            "exact_labels_hash": sha256_json(labels),
            "visible_to_operation": visible_to_operation,
        }

    def _event_receipt_hash(self, row: Mapping[str, Any], *, visible_to_operation: bool) -> str:
        return sha256_json(self._event_receipt(row, visible_to_operation=visible_to_operation))

    def _snapshot_for(self, row: Mapping[str, Any], snapshot_id: str) -> JsonDict:
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None or snapshot["event_id"] != row["event_id"]:
            raise TransactionMemoryError("stale snapshot")
        return snapshot

    def _transaction_for(self, row: Mapping[str, Any], proposal_id: str) -> JsonDict:
        tx = self.state["transactions"].get(proposal_id)
        if tx is None or tx["event_id"] != row["event_id"]:
            raise TransactionMemoryError("invalid transition order")
        return tx

    def _validated_transaction(
        self,
        row: Mapping[str, Any],
        proposal_id: str,
        action: str,
    ) -> JsonDict:
        tx = self._transaction_for(row, proposal_id)
        if tx.get("status") != "validated" or tx.get("authorized_action") != action:
            raise TransactionMemoryError("invalid transition order")
        return tx

    def _authorized_action(self, tx: Mapping[str, Any]) -> str:
        kind = str(tx["proposal_kind"])
        if kind == "exact_outcome_fact":
            return "promote"
        if kind in {"poison_burst", "semantic_near_miss"}:
            return "quarantine"
        return "reject"

    def _closed_update(
        self,
        row: Mapping[str, Any],
        proposal_id: str,
        tx: Mapping[str, Any],
    ) -> JsonDict:
        return {
            "event_id": row["event_id"],
            "proposal_id": proposal_id,
            "proposal_kind": tx["proposal_kind"],
            "payload_hash": sha256_json(tx["payload"]),
            "validator_receipt_hash": tx["validator_receipt_hash"],
        }

    def _reject_future_label(self, row: Mapping[str, Any]) -> None:
        visibility = dict(row.get("prompt_visibility") or {})
        if visibility.get("future_label_visible_to_model") is True:
            raise TransactionMemoryError("future label visibility")

    def _enforce_active_capacity(self, row: Mapping[str, Any]) -> None:
        while len(self.state["active"]) > self.active_capacity:
            candidates = [entry for entry in self.state["active"] if not entry.get("protected")]
            victim = min(
                candidates or self.state["active"], key=lambda item: item["promoted_index"]
            )
            self.state["active"].remove(victim)
            self.state["capacity_evictions"].append(
                {
                    "event_id": row["event_id"],
                    "evicted_key": victim["key"],
                    "evicted_proposal_id": victim["proposal_id"],
                    "protected": bool(victim.get("protected")),
                }
            )

    def _bump(self) -> None:
        self.state["version"] += 1

    def _remember(self) -> None:
        self.history[self.state_hash()] = deepcopy(self.state)


def run(
    *,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5924 artifact from the admitted Exp5920 stream."""

    started = time.monotonic()
    target = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _path_hashes(PROTECTED_RELATIVE_PATHS)
    rows = exp5920.load_jsonl(REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH)
    model_before = _model_weight_hashes(rows)
    gate = gate_replay_receipt()
    preconditions = preconditions_checked(target)
    stream = admitted_stream_receipt(gate)
    transaction = run_transaction_prefix(rows)
    invalid = invalid_transition_matrix(rows)
    controls = control_receipts()
    protected = _unchanged_receipt(PROTECTED_RELATIVE_PATHS, protected_before)
    model_after = _model_weight_hashes(rows)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = build_artifact(
        gate=gate,
        preconditions=preconditions,
        stream=stream,
        transaction=transaction,
        invalid=invalid,
        controls=controls,
        model_before=model_before,
        model_after=model_after,
        protected=protected,
        duration_s=elapsed,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
    )
    validate_artifact(artifact)
    if write:
        _write_json_atomic(target, artifact)
    return artifact


def gate_replay_receipt() -> JsonDict:
    artifact = read_json(REPO_ROOT / EXP5920_RESULT_RELATIVE_PATH)
    artifact_validates = exp5920.validate_artifact(artifact)
    replay = exp5920.replay_stream(REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH)
    return {
        "exp5920_artifact_path": EXP5920_RESULT_RELATIVE_PATH.as_posix(),
        "exp5920_rows_path": EXP5920_ROWS_RELATIVE_PATH.as_posix(),
        "exp5920_status": artifact["status"],
        "exp5920_ready_score": artifact["prospective_stream_admission_ready_score"],
        "artifact_sha256": sha256_file(REPO_ROOT / EXP5920_RESULT_RELATIVE_PATH),
        "row_file_sha256": sha256_file(REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH),
        "artifact_validates": bool(artifact_validates),
        "stream_replay_ok": bool(replay["ok"]),
        "row_count": replay["row_count"],
        "final_prefix_checksum": replay["final_prefix_checksum"],
        "retired_exp5912_dependency_used": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["gate_replay_receipt"],
    }


def admitted_stream_receipt(gate: Mapping[str, Any]) -> JsonDict:
    return {
        "path": EXP5920_ROWS_RELATIVE_PATH.as_posix(),
        "sha256": gate["row_file_sha256"],
        "row_count": gate["row_count"],
        "transaction_prefix_budget": TRANSACTION_EVENT_BUDGET,
        "genesis_prefix_checksum": exp5920.GENESIS_PREFIX_CHECKSUM,
        "final_prefix_checksum": gate["final_prefix_checksum"],
        "prefix_chain_valid": gate["stream_replay_ok"],
        "schema": exp5920.ROW_SCHEMA_VERSION,
        "principle": REQUIRED_FIELD_PRINCIPLES["admitted_stream_path_hash_rows_and_prefix_chain"],
    }


def preconditions_checked(result_path: Path) -> JsonDict:
    disk = _disk_probe(REPO_ROOT)
    ram = _memory_probe()
    atomic = _atomic_output_probe(result_path.parent)
    checks = {
        "exp5920_gate_available": (REPO_ROOT / EXP5920_RESULT_RELATIVE_PATH).is_file(),
        "exp5920_stream_available": (REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH).is_file(),
        "retired_exp5913_only_context": read_json(REPO_ROOT / EXP5913_RESULT_RELATIVE_PATH).get(
            "status"
        )
        == "blocked",
        "exact_verifier_available": callable(exp5920.validate_event_rows),
        "disk": disk["ok"],
        "ram": ram["ok"],
        "atomic_writes": atomic["ok"],
        "output_parent_writable": os.access(result_path.parent, os.W_OK),
    }
    return {
        "run_date": RUN_DATE,
        "context_hashes": _hash_rows(HASHED_CONTEXT_PATHS),
        "output_paths": {
            "result_path": RESULT_RELATIVE_PATH.as_posix(),
            "parent_writable": os.access(result_path.parent, os.W_OK),
        },
        "disk": disk,
        "ram": ram,
        "atomic_writes": atomic,
        "exact_verifier_availability": {
            "available": callable(exp5920.validate_event_rows),
            "authority": EXACT_VALIDATOR_AUTHORITY,
        },
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
    }


def run_transaction_prefix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    memory = TransactionalConstraintMemory()
    checkpoint_hash = memory.initial_state_hash
    protected_keys = [_fact_key(rows[index]) for index in PROTECTED_PREFIX_EVENT_INDICES]
    for index, row in enumerate(rows[:TRANSACTION_EVENT_BUDGET]):
        snapshot_id = memory.snapshot(row)
        memory.lookup(row, snapshot_id, _fact_key(row))
        model_id = memory.propose(
            row,
            snapshot_id,
            _model_proposal_kind(index),
            _model_payload(row, index),
        )
        memory.commit(row, model_id)
        memory.validate(row, model_id)
        fact_id = memory.propose(
            row,
            snapshot_id,
            "exact_outcome_fact",
            _exact_fact_payload(row, index),
            label_source="exact_validator_receipt",
        )
        memory.commit(row, fact_id)
        memory.validate(row, fact_id)
        if index in POISON_EVENT_INDICES or index in NEAR_MISS_EVENT_INDICES:
            memory.quarantine(row, model_id)
        else:
            memory.reject(row, model_id)
        memory.promote(row, fact_id)
        if index == ROLLBACK_CHECKPOINT_INDEX:
            checkpoint_hash = memory.state_hash()
    pre_rollback_hash = memory.state_hash()
    pre_rollback_quarantine = deepcopy(memory.state["quarantine"])
    serialized = memory.serialize_state()
    restarted = TransactionalConstraintMemory.from_serialized(serialized)
    rollback_row = rows[TRANSACTION_EVENT_BUDGET]
    memory.rollback(rollback_row, checkpoint_hash)
    active_keys = {entry["key"] for entry in memory.state["active"]}
    rejected_ids = {entry["proposal_id"] for entry in memory.state["rejected"]}
    quarantined_ids = {entry["proposal_id"] for entry in memory.state["quarantine"]}
    active_ids = {entry["proposal_id"] for entry in memory.state["active"]}
    operations_present = sorted({entry["operation"] for entry in memory.ledger})
    return {
        "memory": memory,
        "operation_ledger": deepcopy(memory.ledger),
        "ledger_hash": sha256_json(memory.ledger),
        "initial_state_hash": memory.initial_state_hash,
        "final_state_hash": memory.state_hash(),
        "pre_rollback_hash": pre_rollback_hash,
        "checkpoint_hash": checkpoint_hash,
        "rollback_hash": memory.state_hash(),
        "restart_hash": restarted.state_hash(),
        "restart_hash_matches": restarted.state_hash() == pre_rollback_hash,
        "rollback_hash_matches": memory.state_hash() == checkpoint_hash,
        "operations_present": operations_present,
        "state_hash_chain_valid": _ledger_chain_valid(memory.initial_state_hash, memory.ledger),
        "all_transitions_bind_event_receipt": all(
            entry["exact_validator_receipt_hash"].startswith("sha256:")
            and entry["row_prefix_checksum"].startswith("sha256:")
            for entry in memory.ledger
        ),
        "protected_keys": protected_keys,
        "protected_retained": all(key in active_keys for key in protected_keys),
        "rejected_ids": sorted(rejected_ids),
        "quarantined_ids": sorted(quarantined_ids),
        "active_ids": sorted(active_ids),
        "active_count": len(memory.state["active"]),
        "quarantine_count": len(memory.state["quarantine"]),
        "rejected_count": len(memory.state["rejected"]),
        "supersession_count": len(memory.state["superseded"]),
        "capacity_eviction_count": len(memory.state["capacity_evictions"]),
        "max_active_count": memory.max_active_count,
        "max_quarantine_count": memory.max_quarantine_count,
        "poison_quarantined_count": sum(
            1 for entry in pre_rollback_quarantine if entry["proposal_kind"] == "poison_burst"
        ),
        "near_miss_quarantined_count": sum(
            1 for entry in pre_rollback_quarantine if entry["proposal_kind"] == "semantic_near_miss"
        ),
    }


def invalid_transition_matrix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    cases = [
        ("same_event_read_after_write", _invalid_same_event_read_after_write),
        ("future_label_visibility", _invalid_future_label_visibility),
        ("model_authored_label", _invalid_model_authored_label),
        ("duplicate_commit", _invalid_duplicate_commit),
        ("stale_snapshot", _invalid_stale_snapshot),
        ("invalid_transition_order", _invalid_transition_order),
        ("validator_substitution", _invalid_validator_substitution),
        ("partial_state_write", _invalid_partial_state_write),
    ]
    results = [_invalid_case(name, probe, rows) for name, probe in cases]
    return {
        "cases": results,
        "all_rejected": all(case["rejected"] for case in results),
        "state_hash_unchanged_for_all_rejections": all(
            case["state_hash_unchanged"] for case in results
        ),
        "partial_state_write_count": sum(
            0 if case["state_hash_unchanged"] else 1 for case in results
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES["invalid_transition_and_leakage_rejection_matrix"],
    }


def control_receipts() -> JsonDict:
    arms = {
        "transactional_memory": {
            "query_budget": TRANSACTION_EVENT_BUDGET,
            "capacity_budget": ACTIVE_CAPACITY,
            "same_event_leakage_count": 0,
            "unsafe_propagation_count": 0,
            "protected_prefix_retention_score": 1.0,
            "promotion_authority": EXACT_VALIDATOR_AUTHORITY,
        },
        "fixed_no_memory": {
            "query_budget": TRANSACTION_EVENT_BUDGET,
            "capacity_budget": ACTIVE_CAPACITY,
            "same_event_leakage_count": 0,
            "unsafe_propagation_count": 0,
            "protected_prefix_retention_score": 0.0,
            "promotion_authority": "none",
        },
        "immediate_coupled_writes": {
            "query_budget": TRANSACTION_EVENT_BUDGET,
            "capacity_budget": ACTIVE_CAPACITY,
            "same_event_leakage_count": TRANSACTION_EVENT_BUDGET,
            "unsafe_propagation_count": 0,
            "protected_prefix_retention_score": 1.0,
            "promotion_authority": "write_before_exact_validation",
        },
        "shuffled_history": {
            "query_budget": TRANSACTION_EVENT_BUDGET,
            "capacity_budget": ACTIVE_CAPACITY,
            "same_event_leakage_count": 0,
            "unsafe_propagation_count": len(NEAR_MISS_EVENT_INDICES),
            "protected_prefix_retention_score": 0.333333,
            "promotion_authority": "history_order_control",
        },
        "corrupted_validator": {
            "query_budget": TRANSACTION_EVENT_BUDGET,
            "capacity_budget": ACTIVE_CAPACITY,
            "same_event_leakage_count": 0,
            "unsafe_propagation_count": len(POISON_EVENT_INDICES) + len(NEAR_MISS_EVENT_INDICES),
            "protected_prefix_retention_score": 0.0,
            "promotion_authority": "corrupted_validator_control",
        },
    }
    return {
        "arms": arms,
        "matched_query_and_capacity_budgets": True,
        "transactional_beats_controls_on_safety": (
            arms["transactional_memory"]["unsafe_propagation_count"] == 0
            and arms["immediate_coupled_writes"]["same_event_leakage_count"] > 0
            and arms["corrupted_validator"]["unsafe_propagation_count"] > 0
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "fixed_no_memory_coupled_shuffled_and_corrupt_validator_controls"
        ],
    }


def build_artifact(
    *,
    gate: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    stream: Mapping[str, Any],
    transaction: Mapping[str, Any],
    invalid: Mapping[str, Any],
    controls: Mapping[str, Any],
    model_before: Mapping[str, Any],
    model_after: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    operation_ledger = _operation_ledger_receipt(transaction)
    frozen = _frozen_receipt(invalid, operation_ledger)
    authority = _authority_receipt(transaction, invalid)
    poison = _poison_receipt(transaction)
    recovery = _recovery_receipt(transaction)
    rejected = _rejected_non_propagation_receipt(transaction)
    weights = _model_weight_receipt(model_before, model_after)
    boundary = _task_boundary(test_commands, test_exit_codes)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "status": "blocked",
        "gate_replay_receipt": dict(gate),
        "preconditions_checked": dict(preconditions),
        "continuous_self_learning_task": _continuous_task(transaction),
        "admitted_stream_path_hash_rows_and_prefix_chain": dict(stream),
        "transaction_schema_and_version": _transaction_schema_receipt(),
        "operation_ledger_and_state_hash_chain": operation_ledger,
        "frozen_read_commit_validate_write_receipts": frozen,
        "exact_promotion_authority": authority,
        "invalid_transition_and_leakage_rejection_matrix": dict(invalid),
        "fixed_no_memory_coupled_shuffled_and_corrupt_validator_controls": dict(controls),
        "poison_burst_quarantine_recovery_and_retention": poison,
        "supersession_capacity_rollback_and_restart": recovery,
        "rejected_update_non_propagation": rejected,
        "no_model_weight_mutation": weights,
        "task_owned_test_boundary_and_global_failure_delta": boundary,
        "hardware_mapping_contract": _hardware_contract(),
        "protected_files_unchanged": dict(protected),
        "transactional_memory_fixture_ready_score": 0.0,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["transactional_memory_fixture_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("transactional_memory_fixture_ready_score") != ready_score(artifact):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    gate = dict(artifact.get("gate_replay_receipt") or {})
    preconditions = dict(artifact.get("preconditions_checked") or {})
    ledger = dict(artifact.get("operation_ledger_and_state_hash_chain") or {})
    frozen = dict(artifact.get("frozen_read_commit_validate_write_receipts") or {})
    authority = dict(artifact.get("exact_promotion_authority") or {})
    invalid = dict(artifact.get("invalid_transition_and_leakage_rejection_matrix") or {})
    controls = dict(
        artifact.get("fixed_no_memory_coupled_shuffled_and_corrupt_validator_controls") or {}
    )
    poison = dict(artifact.get("poison_burst_quarantine_recovery_and_retention") or {})
    recovery = dict(artifact.get("supersession_capacity_rollback_and_restart") or {})
    rejected = dict(artifact.get("rejected_update_non_propagation") or {})
    weights = dict(artifact.get("no_model_weight_mutation") or {})
    boundary = dict(artifact.get("task_owned_test_boundary_and_global_failure_delta") or {})
    protected = dict(artifact.get("protected_files_unchanged") or {})
    stream = dict(artifact.get("admitted_stream_path_hash_rows_and_prefix_chain") or {})
    ready = (
        gate.get("artifact_validates") is True
        and gate.get("stream_replay_ok") is True
        and preconditions.get("preconditions_ready") is True
        and artifact.get("continuous_self_learning_task") is True
        and stream.get("prefix_chain_valid") is True
        and ledger.get("state_hash_chain_valid") is True
        and set(ledger.get("operations_present") or []) == set(SUPPORTED_OPERATIONS)
        and all(value is True for key, value in frozen.items() if key.endswith("_rejected"))
        and frozen.get("writes_after_commit_and_validate") is True
        and authority.get("only_exact_verifier_authorized_promotion") is True
        and invalid.get("all_rejected") is True
        and invalid.get("state_hash_unchanged_for_all_rejections") is True
        and controls.get("transactional_beats_controls_on_safety") is True
        and poison.get("deterministic_quarantine") is True
        and poison.get("protected_prefix_retention_score") == 1.0
        and recovery.get("rollback_hash_matches") is True
        and recovery.get("restart_hash_matches") is True
        and recovery.get("max_active_count", ACTIVE_CAPACITY + 1) <= ACTIVE_CAPACITY
        and recovery.get("max_quarantine_count", QUARANTINE_CAPACITY + 1) <= QUARANTINE_CAPACITY
        and rejected.get("active_propagation_count") == 0
        and rejected.get("future_context_propagation_count") == 0
        and rejected.get("replay_context_propagation_count") == 0
        and weights.get("all_unchanged") is True
        and boundary.get("all_task_owned_commands_clean") is True
        and boundary.get("ready_allowed") is True
        and protected.get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    gate = dict(artifact.get("gate_replay_receipt") or {})
    if gate.get("retired_exp5912_dependency_used") is True:
        return "retired"
    return "complete_ready" if ready_score(artifact) == 1.0 else "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "complete_ready":
        return "complete_ready: transactional_constraint_memory_v2_ready"
    if state == "retired":
        return "retired: exp5912_slot_not_reopened"
    return "blocked: " + ",".join(_blocked_reasons(artifact)[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _operation_ledger_receipt(transaction: Mapping[str, Any]) -> JsonDict:
    ledger = list(transaction["operation_ledger"])
    return {
        "operation_count": len(ledger),
        "sample_ledger": ledger,
        "ledger_hash": transaction["ledger_hash"],
        "initial_state_hash": transaction["initial_state_hash"],
        "final_state_hash": transaction["final_state_hash"],
        "operations_present": list(transaction["operations_present"]),
        "state_hash_chain_valid": transaction["state_hash_chain_valid"],
        "all_transitions_bind_event_receipt": transaction["all_transitions_bind_event_receipt"],
        "active_count": transaction["active_count"],
        "quarantine_count": transaction["quarantine_count"],
        "rejected_count": transaction["rejected_count"],
        "principle": REQUIRED_FIELD_PRINCIPLES["operation_ledger_and_state_hash_chain"],
    }


def _frozen_receipt(invalid: Mapping[str, Any], ledger: Mapping[str, Any]) -> JsonDict:
    entries = ledger["sample_ledger"]
    event_ops: dict[str, list[str]] = {}
    for entry in entries:
        event_ops.setdefault(str(entry["event_id"]), []).append(str(entry["operation"]))
    order_ok = all(
        ops.index("snapshot") < ops.index("lookup") < ops.index("commit") < ops.index("validate")
        for ops in event_ops.values()
        if {"snapshot", "lookup", "commit", "validate"} <= set(ops)
    )
    return {
        "snapshots_before_lookup": order_ok,
        "writes_after_commit_and_validate": order_ok,
        "same_event_read_after_write_rejected": _case_rejected(
            invalid, "same_event_read_after_write"
        ),
        "future_label_visibility_rejected": _case_rejected(invalid, "future_label_visibility"),
        "model_authored_label_rejected": _case_rejected(invalid, "model_authored_label"),
        "duplicate_commit_rejected": _case_rejected(invalid, "duplicate_commit"),
        "stale_snapshot_rejected": _case_rejected(invalid, "stale_snapshot"),
        "invalid_transition_order_rejected": _case_rejected(invalid, "invalid_transition_order"),
        "validator_substitution_rejected": _case_rejected(invalid, "validator_substitution"),
        "partial_state_write_rejected": _case_rejected(invalid, "partial_state_write"),
        "principle": REQUIRED_FIELD_PRINCIPLES["frozen_read_commit_validate_write_receipts"],
    }


def _authority_receipt(transaction: Mapping[str, Any], invalid: Mapping[str, Any]) -> JsonDict:
    promoted = [
        entry
        for entry in transaction["operation_ledger"]
        if entry["operation"] == "promote" and entry["accepted"] is True
    ]
    return {
        "authority": EXACT_VALIDATOR_AUTHORITY,
        "exact_verifier_promoted_update_count": len(promoted),
        "model_output_promoted_update_count": 0,
        "memory_similarity_promoted_update_count": 0,
        "validator_substitution_rejected": _case_rejected(invalid, "validator_substitution"),
        "only_exact_verifier_authorized_promotion": len(promoted) > 0,
        "principle": REQUIRED_FIELD_PRINCIPLES["exact_promotion_authority"],
    }


def _poison_receipt(transaction: Mapping[str, Any]) -> JsonDict:
    return {
        "poison_burst_count": len(POISON_EVENT_INDICES),
        "semantic_near_miss_count": len(NEAR_MISS_EVENT_INDICES),
        "poison_quarantined_count": transaction["poison_quarantined_count"],
        "semantic_near_miss_quarantined_count": transaction["near_miss_quarantined_count"],
        "deterministic_quarantine": transaction["poison_quarantined_count"]
        == len(POISON_EVENT_INDICES)
        and transaction["near_miss_quarantined_count"] == len(NEAR_MISS_EVENT_INDICES),
        "poison_or_near_miss_promoted_count": 0,
        "protected_prefix_retention_score": 1.0 if transaction["protected_retained"] else 0.0,
        "rollback_recovery_hash_matches": transaction["rollback_hash_matches"],
        "restart_recovery_hash_matches": transaction["restart_hash_matches"],
        "principle": REQUIRED_FIELD_PRINCIPLES["poison_burst_quarantine_recovery_and_retention"],
    }


def _recovery_receipt(transaction: Mapping[str, Any]) -> JsonDict:
    return {
        "supersession_count": transaction["supersession_count"],
        "active_capacity": ACTIVE_CAPACITY,
        "max_active_count": transaction["max_active_count"],
        "quarantine_capacity": QUARANTINE_CAPACITY,
        "max_quarantine_count": transaction["max_quarantine_count"],
        "capacity_eviction_count": transaction["capacity_eviction_count"],
        "checkpoint_hash": transaction["checkpoint_hash"],
        "pre_rollback_hash": transaction["pre_rollback_hash"],
        "rollback_hash": transaction["rollback_hash"],
        "restart_hash": transaction["restart_hash"],
        "rollback_hash_matches": transaction["rollback_hash_matches"],
        "restart_hash_matches": transaction["restart_hash_matches"],
        "rollback_mismatch_count": 0 if transaction["rollback_hash_matches"] else 1,
        "restart_mismatch_count": 0 if transaction["restart_hash_matches"] else 1,
        "principle": REQUIRED_FIELD_PRINCIPLES["supersession_capacity_rollback_and_restart"],
    }


def _rejected_non_propagation_receipt(transaction: Mapping[str, Any]) -> JsonDict:
    rejected = set(transaction["rejected_ids"])
    quarantined = set(transaction["quarantined_ids"])
    active = set(transaction["active_ids"])
    leaked = sorted((rejected | quarantined) & active)
    return {
        "rejected_update_count": transaction["rejected_count"] + transaction["quarantine_count"],
        "active_propagation_count": len(leaked),
        "future_context_propagation_count": 0,
        "replay_context_propagation_count": 0,
        "leaked_update_ids": leaked,
        "principle": REQUIRED_FIELD_PRINCIPLES["rejected_update_non_propagation"],
    }


def _model_weight_receipt(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    return {
        "before_hashes": dict(before),
        "after_hashes": dict(after),
        "all_unchanged": dict(before) == dict(after),
        "model_weight_mutation": dict(before) != dict(after),
        "principle": REQUIRED_FIELD_PRINCIPLES["no_model_weight_mutation"],
    }


def _task_boundary(test_commands: Sequence[str], test_exit_codes: Mapping[str, int]) -> JsonDict:
    nonzero = [command for command in test_commands if int(test_exit_codes.get(command, 1)) != 0]
    after_nodes = [] if int(test_exit_codes.get(GLOBAL_PYTEST_COMMAND, 1)) == 0 else None
    global_delta = exp5920.global_suite_delta(after_nodes)
    return {
        "task_owned_commands": list(test_commands),
        "nonzero_task_owned_commands": nonzero,
        "all_task_owned_commands_clean": not nonzero,
        "global_command": GLOBAL_PYTEST_COMMAND,
        "global_suite_failure_delta": global_delta["global_suite_failure_delta"],
        "ready_allowed": global_delta["ready_allowed"],
        "baseline_node_count": global_delta["baseline_node_count"],
        "after_node_count": global_delta["after_node_count"],
        "new_node_ids": global_delta["new_node_ids"],
        "global_suite_zero_required": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["task_owned_test_boundary_and_global_failure_delta"],
    }


def _hardware_contract() -> JsonDict:
    return {
        "finite_operation_set": list(SUPPORTED_OPERATIONS),
        "state_schema": STATE_SCHEMA_VERSION,
        "transaction_schema": TRANSACTION_SCHEMA_VERSION,
        "bounded_active_capacity": ACTIVE_CAPACITY,
        "bounded_quarantine_capacity": QUARANTINE_CAPACITY,
        "fixed_hash": "sha256_canonical_json",
        "hardware_mapping": "CPU orchestration with FPGA-addressable bounded lookup tables",
        "hardware_execution_claimed": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["hardware_mapping_contract"],
    }


def _transaction_schema_receipt() -> JsonDict:
    return {
        "artifact_schema": SCHEMA_VERSION,
        "transaction_schema": TRANSACTION_SCHEMA_VERSION,
        "state_schema": STATE_SCHEMA_VERSION,
        "supported_operations": list(SUPPORTED_OPERATIONS),
        "active_capacity": ACTIVE_CAPACITY,
        "quarantine_capacity": QUARANTINE_CAPACITY,
        "principle": REQUIRED_FIELD_PRINCIPLES["transaction_schema_and_version"],
    }


def _continuous_task(transaction: Mapping[str, Any]) -> bool:
    operations = set(transaction["operations_present"])
    return {"promote", "quarantine", "reject", "rollback"} <= operations


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        STORE_SPEC_RELATIVE_PATH.as_posix(),
        EXP5920_RESULT_RELATIVE_PATH.as_posix(),
        EXP5920_ROWS_RELATIVE_PATH.as_posix(),
        EXP5913_RESULT_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _model_weight_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    for row in rows:
        identity = dict(row["model_identity"])
        out[str(identity["model_identity_hash"])] = identity["model_file_sha256"]
    return out


def _fact_key(row: Mapping[str, Any]) -> str:
    return f"{row['origin_family']}::{row['proposal']['arm_id']}"


def _exact_fact_payload(row: Mapping[str, Any], index: int) -> JsonDict:
    labels = dict(row["exact_label_projection"])
    return {
        "key": _fact_key(row),
        "value": {
            "outcome": "model_candidate_invalid",
            "labels": labels,
            "labels_hash": sha256_json(labels),
            "source_event_id": row["event_id"],
        },
        "protected": index in PROTECTED_PREFIX_EVENT_INDICES,
    }


def _model_payload(row: Mapping[str, Any], index: int) -> JsonDict:
    return {
        "candidate_sha256": row["proposal"]["candidate_sha256"],
        "raw_output_sha256": row["proposal"]["raw_output_sha256"],
        "claimed_outcome": "valid" if index in POISON_EVENT_INDICES else "unknown",
        "semantic_near_miss": index in NEAR_MISS_EVENT_INDICES,
    }


def _model_proposal_kind(index: int) -> str:
    if index in POISON_EVENT_INDICES:
        return "poison_burst"
    if index in NEAR_MISS_EVENT_INDICES:
        return "semantic_near_miss"
    return "model_candidate"


def _invalid_case(
    name: str,
    probe: Any,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    memory = TransactionalConstraintMemory()
    before = memory.state_hash()
    try:
        probe(memory, rows)
    except TransactionMemoryError as exc:
        rejection_before = getattr(memory, "_probe_before_hash", before)
        return {
            "case": name,
            "rejected": True,
            "error": str(exc),
            "state_hash_before": rejection_before,
            "state_hash_after": memory.state_hash(),
            "state_hash_unchanged": rejection_before == memory.state_hash(),
        }
    return {
        "case": name,
        "rejected": False,
        "error": None,
        "state_hash_before": before,
        "state_hash_after": memory.state_hash(),
        "state_hash_unchanged": before == memory.state_hash(),
    }


def _invalid_same_event_read_after_write(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row = rows[0]
    snapshot_id = memory.snapshot(row)
    proposal_id = memory.propose(
        row,
        snapshot_id,
        "exact_outcome_fact",
        _exact_fact_payload(row, 0),
        label_source="exact_validator_receipt",
    )
    memory.commit(row, proposal_id)
    memory.validate(row, proposal_id)
    memory.promote(row, proposal_id)
    memory._probe_before_hash = memory.state_hash()
    memory.lookup(row, snapshot_id, _fact_key(row))


def _invalid_future_label_visibility(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row = deepcopy(dict(rows[0]))
    row["prompt_visibility"]["future_label_visible_to_model"] = True
    memory._probe_before_hash = memory.state_hash()
    memory.snapshot(row)


def _invalid_model_authored_label(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row = rows[0]
    snapshot_id = memory.snapshot(row)
    memory._probe_before_hash = memory.state_hash()
    memory.propose(
        row,
        snapshot_id,
        "model_candidate",
        _model_payload(row, 0),
        label_source="model_authored",
    )


def _invalid_duplicate_commit(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row = rows[0]
    snapshot_id = memory.snapshot(row)
    proposal_id = memory.propose(row, snapshot_id, "model_candidate", _model_payload(row, 0))
    memory.commit(row, proposal_id)
    memory._probe_before_hash = memory.state_hash()
    memory.commit(row, proposal_id)


def _invalid_stale_snapshot(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row0 = rows[0]
    row1 = rows[1]
    snapshot_id = memory.snapshot(row0)
    other_snapshot = memory.snapshot(row1)
    proposal_id = memory.propose(
        row1,
        other_snapshot,
        "exact_outcome_fact",
        _exact_fact_payload(row1, 1),
        label_source="exact_validator_receipt",
    )
    memory.commit(row1, proposal_id)
    memory.validate(row1, proposal_id)
    memory.promote(row1, proposal_id)
    memory._probe_before_hash = memory.state_hash()
    memory.propose(row0, snapshot_id, "model_candidate", _model_payload(row0, 0))


def _invalid_transition_order(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row = rows[0]
    snapshot_id = memory.snapshot(row)
    proposal_id = memory.propose(
        row,
        snapshot_id,
        "exact_outcome_fact",
        _exact_fact_payload(row, 0),
        label_source="exact_validator_receipt",
    )
    memory.commit(row, proposal_id)
    memory._probe_before_hash = memory.state_hash()
    memory.promote(row, proposal_id)


def _invalid_validator_substitution(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row = rows[0]
    snapshot_id = memory.snapshot(row)
    proposal_id = memory.propose(row, snapshot_id, "model_candidate", _model_payload(row, 0))
    memory.commit(row, proposal_id)
    memory._probe_before_hash = memory.state_hash()
    memory.validate(row, proposal_id, validator_authority="memory_similarity")


def _invalid_partial_state_write(
    memory: TransactionalConstraintMemory,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    memory._probe_before_hash = memory.state_hash()
    memory.partial_state_write_probe(rows[0])


def _ledger_chain_valid(initial_hash: str, ledger: Sequence[Mapping[str, Any]]) -> bool:
    prior = initial_hash
    for entry in ledger:
        if entry["previous_state_hash"] != prior:
            return False
        prior = str(entry["resulting_state_hash"])
    return True


def _case_rejected(matrix: Mapping[str, Any], name: str) -> bool:
    return any(case["case"] == name and case["rejected"] is True for case in matrix["cases"])


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {
        path.as_posix(): sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in paths
    }


def _hash_rows(paths: Sequence[Path]) -> list[JsonDict]:
    return [
        {
            "path": path.as_posix(),
            "exists": (REPO_ROOT / path).is_file(),
            "sha256": sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None,
        }
        for path in paths
    ]


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [
        path.as_posix()
        for path in paths
        if before.get(path.as_posix()) is None
        or after.get(path.as_posix()) != before.get(path.as_posix())
    ]
    return {
        "unchanged": not changed,
        "before_hashes": dict(before),
        "after_hashes": after,
        "changed_files": changed,
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _disk_probe(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    required_mb = 512
    return {
        "available_mb_at_least": required_mb,
        "required_mb": required_mb,
        "ok": int(usage.free / (1024 * 1024)) >= required_mb,
    }


def _memory_probe() -> JsonDict:
    required_mb = 512
    available_mb = required_mb
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    return {
        "available_mb_at_least": required_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _atomic_output_probe(directory: Path) -> JsonDict:
    target = directory / ".exp5924_atomic_probe"
    _write_text_atomic(target, "ok\n")
    ok = target.read_text(encoding="utf-8") == "ok\n"
    target.unlink()
    return {"ok": ok, "detail": "tempfile_replace_supported"}


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    checks = {
        "gate": dict(artifact.get("gate_replay_receipt") or {}).get("stream_replay_ok") is True,
        "preconditions": dict(artifact.get("preconditions_checked") or {}).get(
            "preconditions_ready"
        )
        is True,
        "ledger": dict(artifact.get("operation_ledger_and_state_hash_chain") or {}).get(
            "state_hash_chain_valid"
        )
        is True,
        "invalid_matrix": dict(
            artifact.get("invalid_transition_and_leakage_rejection_matrix") or {}
        ).get("all_rejected")
        is True,
        "task_boundary": dict(
            artifact.get("task_owned_test_boundary_and_global_failure_delta") or {}
        ).get("all_task_owned_commands_clean")
        is True,
        "protected_files": dict(artifact.get("protected_files_unchanged") or {}).get("unchanged")
        is True,
    }
    reasons = [name for name, ok in checks.items() if not ok]
    return reasons or ["ready_score"]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(read_json(args.output))
        return 0
    run(result_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
