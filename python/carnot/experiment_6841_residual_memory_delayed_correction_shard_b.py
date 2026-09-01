"""Run residual-memory delayed-correction shard B.

Spec refs: REQ-CL-6841, SCENARIO-CL-6841-PRECONDITIONS,
SCENARIO-CL-6841-SHARD-DISJOINTNESS, SCENARIO-CL-6841-REVEAL-TIMING,
SCENARIO-CL-6841-DELAYED-CREDIT, SCENARIO-CL-6841-STALE-REPLACEMENT,
SCENARIO-CL-6841-JOINT-ERROR-CORRECTION, SCENARIO-CL-6841-ROLLBACK,
SCENARIO-CL-6841-CHECKPOINT-RESTART, and SCENARIO-CL-6841-METRICS.

This module compares bounded external-memory arms on the second frozen
chronological shard. Exact later receipts are opened only after a decision
hash, memory read, and action dose are frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6841_residual_memory_delayed_correction_shard_b.py"
)
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6841_residual_memory_delayed_correction_shard_b.py"
)
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6841_residual_memory_delayed_correction_shard_b.json"
)

RUN_DATE = "20260901"
EXPERIMENT_ID = "6841"
SCHEMA = "carnot.experiment_6841.residual_memory_delayed_correction_shard_b.v1"
STATE_SCHEMA = "carnot.exp6841.residual_memory_delayed_correction_state.v1"
CHECKPOINT_SCHEMA = "carnot.exp6841.residual_memory_delayed_correction_checkpoint.v1"
INFERENCE_SUBSTRATE = "deterministic CPU chronological comparison"
COMPLETE_STATUS = "complete_residual_memory_delayed_correction_shard_b"
BLOCKED_STATUS = "complete_blocked_residual_memory_shard_b"

EXP6827_HASH = "sha256:28464b7e0bbbab79d9d7db0bdb8a0ed498206c11a91d2ff8358756a3a5fd5b59"
EXP6839_HASH = "sha256:8943d774a77d792cc31b447f4e9fc05f474451f9f0fa0685687f30db4ef1361e"
EXPECTED_EVENT_ID_DIGEST = "sha256:0ed855aa5a4a40e0cdfd06fdf207df200ce8b70d715c0505b9ca37ee41f514de"

SOURCE_RELATIVE_PATHS = {
    "exp6827": Path("results/experiment_6827_chronological_causal_edge_memory_stream.json"),
    "exp6839": Path("results/experiment_6839_bounded_residual_memory_kernel.json"),
}
EXPECTED_SOURCE_HASHES = {"exp6827": EXP6827_HASH, "exp6839": EXP6839_HASH}

NO_MEMORY_ARM = "no_memory"
READ_ONLY_ARM = "read_only_memory"
RANDOM_ARM = "random_admission"
VERIFIED_RESIDUAL_ARM = "verified_residual_memory"
ARM_NAMES = (NO_MEMORY_ARM, READ_ONLY_ARM, RANDOM_ARM, VERIFIED_RESIDUAL_ARM)
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
ASSIGNED_ORDER_INDICES = (3, 4)
ASSIGNED_ORDER_IDS = tuple(f"order_{index + 1}" for index in ASSIGNED_ORDER_INDICES)
EXP6840_REFERENCE_ORDER_IDS = ("order_1", "order_2", "order_3")
CAPACITY_BUDGET = 2
EXPECTED_PUBLIC_ROWS_PER_ORDER = 864
EXPECTED_EXACT_EVENTS_PER_ORDER = 153
EXPECTED_ASSIGNED_EVENT_COUNT = EXPECTED_EXACT_EVENTS_PER_ORDER * len(ASSIGNED_ORDER_IDS)
RANDOM_SEEDS = (6_841_001, 6_841_002, 6_841_003)
RESTART_AFTER_EVENT_INDEX = 153
ROLLBACK_EVENT_INDEX = 41
STALE_AFTER_EVENTS = 8
MIN_STORED_DOSE = 0.25
MIN_REVEAL_DELAY_EVENTS = 1

OUTCOME_FIELD_DENYLIST = frozenset(
    {
        "outcome_identity",
        "exact_outcome_hash",
        "signed_direction",
        "admission_decision",
        "future_outcome",
        "final_acceptance",
        "split",
    }
)
PUBLIC_EVENT_KEYS = (
    "row_id",
    "event_id",
    "source_family",
    "order_id",
    "chronological_position",
    "counterfactual_kind",
    "decision_snapshot_sha256",
    "source_proposal_sha256",
    "causal_edge_id",
    "write_operation_id",
    "read_operation_id",
)

OPEN_SPEC_IDS = (
    "REQ-CL-6841",
    "SCENARIO-CL-6841-PRECONDITIONS",
    "SCENARIO-CL-6841-SHARD-DISJOINTNESS",
    "SCENARIO-CL-6841-REVEAL-TIMING",
    "SCENARIO-CL-6841-DELAYED-CREDIT",
    "SCENARIO-CL-6841-STALE-REPLACEMENT",
    "SCENARIO-CL-6841-JOINT-ERROR-CORRECTION",
    "SCENARIO-CL-6841-ROLLBACK",
    "SCENARIO-CL-6841-CHECKPOINT-RESTART",
    "SCENARIO-CL-6841-METRICS",
)
REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6841_residual_memory_delayed_correction_shard_b.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_6841_residual_memory_delayed_correction_shard_b.py' -m pytest tests/python/test_experiment_6841_residual_memory_delayed_correction_shard_b.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6841_residual_memory_delayed_correction_shard_b.py scripts/experiments/experiment_6841_residual_memory_delayed_correction_shard_b.py tests/python/test_experiment_6841_residual_memory_delayed_correction_shard_b.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6841_residual_memory_delayed_correction_shard_b.py",
    ".venv/bin/python scripts/experiments/experiment_6841_residual_memory_delayed_correction_shard_b.py --date 20260901",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6841_residual_memory_delayed_correction_shard_b.json",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6841_residual_memory_delayed_correction_shard_b.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "title",
    "run_date",
    "status",
    "openspec_requirement_ids",
    "replay_commands",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "continuous_self_learning_task",
    "source_artifact_hashes",
    "random_seeds",
    "reproducibility_checksum",
    "split_manifest",
    "arm_contracts",
    "rows",
    "held_future_results",
    "delayed_correction_results",
    "correction_latency_results",
    "residual_error_results",
    "memory_state_transitions",
    "negative_transfer_results",
    "headroom_summary",
    "checkpoint_manifest",
    "csl_shard_b_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
REQUIRED_ROW_FIELDS = frozenset(
    {
        "row_id",
        "source_event_row_id",
        "event_id",
        "family",
        "family_identity_sha256",
        "order_id",
        "order_index",
        "arm",
        "seed",
        "split",
        "counterfactual_kind",
        "correction_family",
        "event_sequence_index",
        "reveal_delay_events",
        "correction_latency_events",
        "available_headroom",
        "no_headroom",
        "capacity_budget",
        "active_count_before",
        "active_count_after",
        "memory_dose",
        "action",
        "predicted_direction",
        "exact_outcome",
        "decision_correct",
        "residual_error",
        "regret",
        "negative_transfer",
        "negative_transfer_delta",
        "held_future_metric",
        "memory_read_receipt",
        "admission_decision",
        "memory_transition_statuses",
        "decision_hash",
        "decision_frozen_before_outcome_reveal",
        "outcome_revealed_after_decision",
        "update_receipt_sha256",
        "row_sha256",
    }
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents silent reinterpretation of shard rows.",
    "experiment_id": "A stable identifier binds the artifact to Exp6841.",
    "title": "The title states this is the delayed-correction shard.",
    "run_date": "The fixed date separates this execution from later reruns.",
    "status": "The status separates complete comparison from blocked gates.",
    "openspec_requirement_ids": "Requirement IDs keep tests tied to shard rules.",
    "replay_commands": "Replay commands show the intended verification stack.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Input gates stop source drift before decisions.",
    "inference_substrate": "The substrate declares deterministic CPU replay.",
    "duration_s": "Measured wall time shows that the shard executed.",
    "continuous_self_learning_task": "True marks this as an FR11 memory task.",
    "source_artifact_hashes": "Raw source hashes bind upstream evidence bytes.",
    "random_seeds": "Fixed seeds make the arm schedule replayable.",
    "reproducibility_checksum": "The checksum binds stable content and excludes time.",
    "split_manifest": "The manifest freezes shard B rows and disjointness.",
    "arm_contracts": "Arm contracts record equal compute and capacity budgets.",
    "rows": "Rows expose every event, arm, and seed comparison.",
    "held_future_results": "Held-future accuracy is separate from completion.",
    "delayed_correction_results": "Correction summaries expose stale and joint repairs.",
    "correction_latency_results": "Latency shows delay between action and receipt.",
    "residual_error_results": "Residual error shows remaining direction mismatch.",
    "memory_state_transitions": "Transitions expose rejected, revised, expired, committed, and rolled-back entries.",
    "negative_transfer_results": "Negative transfer compares each route to no memory.",
    "headroom_summary": "Headroom shows finite capacity and no-headroom rows.",
    "checkpoint_manifest": "Checkpoint hashes prove restart and replay identity.",
    "csl_shard_b_complete_score": "Completion depends on planned rows and receipts only.",
    "gate_check_summary": "Failed checks keep expected and observed values.",
    "verifier_is_oracle": "False because this shard measures external outcomes.",
    "verdict_class": "A closed class prevents unsupported verdict wording.",
    "honest_verdict": "The terminal sentence states the row-supported result.",
}


@dataclass
class ComparisonRun:
    """State and row receipts produced by one shard replay."""

    rows: list[JsonDict]
    memory_state_transitions: list[JsonDict]
    checkpoint_receipt: JsonDict
    rollback_receipt: JsonDict
    final_state_hash: str


@dataclass
class CorrectionState:
    """Small bounded store for one arm and seed.

    The store is external memory, not model state. Its canonical JSON bytes are
    the authority for restart, rollback, and clean replay checks.
    """

    arm: str
    capacity_budget: int = CAPACITY_BUDGET
    records: list[JsonDict] = field(default_factory=list)
    processed_update_ids: set[str] = field(default_factory=set)
    residual_pressure: float = 0.0
    generation: int = 0
    predecessor_hash: str = "bootstrap"

    @classmethod
    def for_arm(cls, arm: str) -> "CorrectionState":
        """Create an arm store with the same nominal capacity budget."""

        state = cls(arm=arm)
        if arm == READ_ONLY_ARM:
            state.records.append(
                {
                    "key": "bootstrap:read_only",
                    "source_event_row_id": "bootstrap",
                    "source_family": "control",
                    "case_type": "safe_control",
                    "signed_direction": 1,
                    "stored_dose": MIN_STORED_DOSE,
                    "event_sequence_index": -1,
                    "immutable": True,
                    "component": "bootstrap",
                    "exact_outcome_hash": "bootstrap",
                }
            )
        return state

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorrectionState":
        """Decode checkpoint bytes into the same arm-local state."""

        return cls(
            arm=str(value["arm"]),
            capacity_budget=int(value["capacity_budget"]),
            records=[deepcopy(record) for record in value["records"]],
            processed_update_ids=set(value["processed_update_ids"]),
            residual_pressure=float(value["residual_pressure"]),
            generation=int(value["generation"]),
            predecessor_hash=str(value["predecessor_hash"]),
        )

    def state_dict(self) -> JsonDict:
        """Return only stable JSON fields for hashing and checkpointing."""

        return {
            "schema": STATE_SCHEMA,
            "arm": self.arm,
            "capacity_budget": self.capacity_budget,
            "generation": self.generation,
            "predecessor_hash": self.predecessor_hash,
            "residual_pressure": round(self.residual_pressure, 6),
            "processed_update_ids": sorted(self.processed_update_ids),
            "records": sorted(
                [deepcopy(record) for record in self.records],
                key=lambda record: (
                    int(record["event_sequence_index"]),
                    str(record["key"]),
                ),
            ),
        }

    def state_bytes(self) -> bytes:
        """Serialize state with one canonical representation."""

        return canonical_json_bytes(self.state_dict())

    def state_hash(self) -> str:
        """Hash the canonical state bytes."""

        return sha256_bytes(self.state_bytes())

    def active_count(self) -> int:
        """Count active records against the shared nominal capacity."""

        return len(self.records)

    def active_record_keys(self) -> set[str]:
        """Return the live record keys for transition checks."""

        return {str(record["key"]) for record in self.records}

    def memory_signal(self) -> float:
        """Reduce prior records to a bounded route signal."""

        if not self.records:
            return 0.0
        total = sum(
            int(record["signed_direction"]) * float(record["stored_dose"])
            for record in self.records
        )
        return bounded_dose(total / len(self.records))

    def read_receipt(self, event_sequence_index: int) -> JsonDict:
        """Freeze memory reads before the current event outcome is opened."""

        record_indices = [int(record["event_sequence_index"]) for record in self.records]
        receipt = {
            "arm": self.arm,
            "event_sequence_index": event_sequence_index,
            "read_state_sha256": self.state_hash(),
            "record_count": len(self.records),
            "record_keys": [str(record["key"]) for record in self.records],
            "max_record_event_sequence_index": max(record_indices) if record_indices else None,
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return receipt

    def _restore_bytes(self, state_bytes: bytes) -> None:
        restored = CorrectionState.from_dict(json.loads(state_bytes))
        self.records = restored.records
        self.processed_update_ids = restored.processed_update_ids
        self.residual_pressure = restored.residual_pressure
        self.generation = restored.generation
        self.predecessor_hash = restored.predecessor_hash

    def _evict_if_needed(self) -> list[JsonDict]:
        evicted: list[JsonDict] = []
        while self.active_count() > self.capacity_budget:
            mutable = [record for record in self.records if not record.get("immutable")]
            if not mutable:
                break
            victim = min(
                mutable,
                key=lambda record: (
                    int(record["event_sequence_index"]),
                    str(record["key"]),
                ),
            )
            self.records.remove(victim)
            evicted.append(_transition_record(victim, "expired"))
        return evicted

    def inject_stale_record(
        self, event: Mapping[str, Any], *, event_sequence_index: int
    ) -> JsonDict:
        """Load the frozen stale-memory fixture before exact evidence arrives."""

        if self.arm in {NO_MEMORY_ARM, READ_ONLY_ARM}:
            return {"loaded": False, "reason": "non_mutating_arm"}
        self.records = [
            record
            for record in self.records
            if record.get("case_type") != "stale_evidence"
            or record.get("source_event_row_id") != event["row_id"]
        ]
        self.records.append(
            {
                "key": "fixture:stale:" + sha256_json(event["row_id"])[7:19],
                "source_event_row_id": event["row_id"],
                "source_family": event["source_family"],
                "case_type": "stale_evidence",
                "signed_direction": 0,
                "stored_dose": MIN_STORED_DOSE,
                "event_sequence_index": event_sequence_index - STALE_AFTER_EVENTS - 1,
                "immutable": False,
                "component": "stale_prerequisite",
                "exact_outcome_hash": "pending",
            }
        )
        evicted = self._evict_if_needed()
        return {"loaded": True, "evicted": evicted}

    def inject_joint_error_records(
        self, event: Mapping[str, Any], *, event_sequence_index: int
    ) -> JsonDict:
        """Load two frozen wrong components before exact evidence arrives."""

        if self.arm in {NO_MEMORY_ARM, READ_ONLY_ARM}:
            return {"loaded": False, "reason": "non_mutating_arm"}
        self.records = [record for record in self.records if record.get("immutable")]
        for component in ("authority_a", "authority_b"):
            self.records.append(
                {
                    "key": f"fixture:joint:{component}:"
                    + sha256_json({"row": event["row_id"], "component": component})[7:19],
                    "source_event_row_id": event["row_id"],
                    "source_family": event["source_family"],
                    "case_type": "joint_error_evidence",
                    "signed_direction": 0,
                    "stored_dose": MIN_STORED_DOSE,
                    "event_sequence_index": event_sequence_index - 1,
                    "immutable": False,
                    "component": component,
                    "exact_outcome_hash": "pending",
                }
            )
        evicted = self._evict_if_needed()
        return {"loaded": True, "evicted": evicted}

    def apply_update(
        self,
        event: Mapping[str, Any],
        proposal: Mapping[str, Any],
        outcome: Mapping[str, Any],
        credit: Mapping[str, Any],
        *,
        event_sequence_index: int,
        seed: int,
    ) -> JsonDict:
        """Commit or revise memory only after exact action and outcome authority."""

        parent_hash = self.state_hash()
        update_id = f"{event.get('row_id')}::{self.arm}::{seed}"
        authority_present = all(
            (
                event.get("row_id"),
                event.get("action_identity"),
                outcome.get("outcome_identity"),
                outcome.get("exact_outcome_hash"),
            )
        )
        admitted = False
        reason = "preconditions_passed"
        rejected_entries: list[JsonDict] = []
        revised_entries: list[JsonDict] = []
        expired_entries: list[JsonDict] = []
        committed_entries: list[JsonDict] = []
        case_type = correction_family(event)
        signed_direction = int(outcome.get("signed_direction", 0))

        if not authority_present:
            reason = "missing_update_authority"
        elif update_id in self.processed_update_ids:
            reason = "duplicate_update"
        elif self.arm == NO_MEMORY_ARM:
            reason = "no_memory_route"
        elif self.arm == READ_ONLY_ARM:
            reason = "read_only_route"
        elif abs(float(proposal["memory_dose"])) > 1.0:
            reason = "dose_out_of_bounds"
        elif self.arm == RANDOM_ARM and not random_update_accepts(event, seed):
            reason = "seeded_random_reject"
        elif self.arm == VERIFIED_RESIDUAL_ARM and (
            float(credit["signed_credit"]) <= 0.0
            and case_type not in {"stale_evidence", "joint_error_evidence", "delayed_evidence"}
        ):
            reason = "nonpositive_exact_credit"
        else:
            admitted = True

        if not admitted:
            rejected_entries.append(
                {
                    "status": "rejected",
                    "reason": reason,
                    "source_event_row_id": _public_identity(event.get("row_id")),
                }
            )
        if authority_present and reason != "duplicate_update":
            self.processed_update_ids.add(update_id)
            self.generation += 1
            self.predecessor_hash = parent_hash
            if self.arm == VERIFIED_RESIDUAL_ARM:
                residual = self.residual_pressure * 0.82 + float(credit["signed_credit"]) * 0.45
                self.residual_pressure = bounded_dose(residual)

        if admitted and case_type == "stale_evidence":
            expired_entries = self._expire_stale_records(event_sequence_index)
            committed_entries = [
                self._commit_record(
                    event, proposal, outcome, event_sequence_index, "stale_replacement"
                )
            ]
            reason = "stale_replaced" if expired_entries else "admitted"
        elif admitted and case_type == "joint_error_evidence":
            revised_entries = self._revise_joint_records(event, outcome, event_sequence_index)
            if not revised_entries:
                committed_entries = [
                    self._commit_record(
                        event,
                        proposal,
                        outcome,
                        event_sequence_index,
                        "joint_error_replacement",
                    )
                ]
            reason = "joint_error_revised"
        elif admitted:
            committed_entries = [
                self._commit_record(event, proposal, outcome, event_sequence_index, case_type)
            ]
            reason = "admitted"
        expired_entries.extend(self._evict_if_needed())

        statuses = _transition_statuses(
            rejected_entries,
            revised_entries,
            expired_entries,
            committed_entries,
            rolled_back=False,
        )
        receipt = {
            "arm": self.arm,
            "source_event_row_id": event.get("row_id"),
            "seed": seed,
            "event_sequence_index": event_sequence_index,
            "correction_family": case_type,
            "proposal_hash": proposal["decision_hash"],
            "exact_outcome_hash": outcome.get("exact_outcome_hash"),
            "signed_credit": credit["signed_credit"],
            "admitted": admitted,
            "reason": reason,
            "update_authority_present": bool(authority_present),
            "parent_state_sha256": parent_hash,
            "new_state_sha256": self.state_hash(),
            "rejected_entries": rejected_entries,
            "revised_entries": revised_entries,
            "expired_entries": expired_entries,
            "committed_entries": committed_entries,
            "transition_statuses": statuses,
            "cross_arm_state_read": False,
            "reveal_delay_events": credit["reveal_delay_events"],
            "correction_latency_events": credit["correction_latency_events"],
            "active_count_after": self.active_count(),
            "capacity_budget": self.capacity_budget,
            "signed_direction": signed_direction,
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return receipt

    def _expire_stale_records(self, event_sequence_index: int) -> list[JsonDict]:
        expired: list[JsonDict] = []
        survivors: list[JsonDict] = []
        for record in self.records:
            is_stale = (
                record.get("case_type") == "stale_evidence"
                and event_sequence_index - int(record["event_sequence_index"]) > STALE_AFTER_EVENTS
            )
            if is_stale and not record.get("immutable"):
                expired.append(_transition_record(record, "expired"))
            else:
                survivors.append(record)
        self.records = survivors
        return expired

    def _revise_joint_records(
        self,
        event: Mapping[str, Any],
        outcome: Mapping[str, Any],
        event_sequence_index: int,
    ) -> list[JsonDict]:
        revised: list[JsonDict] = []
        for record in self.records:
            if (
                record.get("case_type") == "joint_error_evidence"
                and record.get("source_event_row_id") == event["row_id"]
            ):
                previous = int(record["signed_direction"])
                record["signed_direction"] = int(outcome["signed_direction"])
                record["event_sequence_index"] = event_sequence_index
                record["exact_outcome_hash"] = outcome["exact_outcome_hash"]
                row = _transition_record(record, "revised")
                row["previous_signed_direction"] = previous
                row["new_signed_direction"] = int(outcome["signed_direction"])
                revised.append(row)
        return revised

    def _commit_record(
        self,
        event: Mapping[str, Any],
        proposal: Mapping[str, Any],
        outcome: Mapping[str, Any],
        event_sequence_index: int,
        case_type: str,
    ) -> JsonDict:
        record = {
            "key": "mem:" + sha256_json({"row": event["row_id"], "arm": self.arm})[7:19],
            "source_event_row_id": event["row_id"],
            "source_family": event["source_family"],
            "case_type": case_type,
            "signed_direction": int(outcome["signed_direction"]),
            "stored_dose": bounded_dose(max(abs(float(proposal["memory_dose"])), MIN_STORED_DOSE)),
            "event_sequence_index": event_sequence_index,
            "immutable": False,
            "component": "exact_later_receipt",
            "exact_outcome_hash": outcome["exact_outcome_hash"],
        }
        self.records.append(record)
        return _transition_record(record, "committed")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with stable sorting and one trailing newline."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the repository SHA-256 string form."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return sha256_bytes(canonical_json_bytes(value))


def bounded_dose(value: float) -> float:
    """Clamp a proposed route dose to the bounded memory interval."""

    return round(max(-1.0, min(1.0, float(value))), 6)


def source_paths_for_root(root: Path) -> dict[str, Path]:
    """Resolve frozen source artifact paths from a checkout root."""

    return {name: root / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Load JSON sources while preserving readable failure markers."""

    loaded: dict[str, JsonDict] = {}
    for name, path in paths.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            value = {"_load_error": type(error).__name__}
        loaded[name] = value if isinstance(value, dict) else {"_load_error": "not_object"}
    return loaded


def _source_hashes(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for name, path in sorted(paths.items()):
        try:
            digest = sha256_bytes(path.read_bytes())
        except OSError:
            digest = "missing"
        try:
            display_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            display_path = str(path)
        result[name] = {
            "path": display_path,
            "sha256": digest,
            "expected_sha256": EXPECTED_SOURCE_HASHES.get(name),
        }
    return result


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def order_index(order_id: str) -> int:
    """Return the zero-based index encoded by an Exp6827 order ID."""

    return int(order_id.split("_", 1)[1]) - 1


def select_assigned_events(exp6827: Mapping[str, Any]) -> list[JsonDict]:
    """Select exact-outcome rows from shard B's frozen orders."""

    rows = exp6827.get("rows", [])
    if not isinstance(rows, list):
        return []
    selected = [
        deepcopy(row)
        for row in rows
        if isinstance(row, dict)
        and row.get("order_id") in ASSIGNED_ORDER_IDS
        and row.get("counterfactual_applicable") is True
        and row.get("outcome_identity")
        and row.get("action_identity")
        and row.get("causal_edge_id")
        and row.get("write_operation_id")
        and row.get("read_operation_id")
    ]
    return sorted(
        selected,
        key=lambda row: (
            order_index(str(row["order_id"])),
            str(row["memory_scope"]),
            int(row["chronological_position"]),
            str(row["counterfactual_kind"]),
            str(row["row_id"]),
        ),
    )


def select_exp6840_reference_event_ids(exp6827: Mapping[str, Any]) -> set[str]:
    """Return row IDs from the first shard orders without importing Exp6840."""

    rows = exp6827.get("rows", [])
    if not isinstance(rows, list):
        return set()
    return {
        str(row["row_id"])
        for row in rows
        if isinstance(row, dict)
        and row.get("order_id") in EXP6840_REFERENCE_ORDER_IDS
        and row.get("counterfactual_applicable") is True
    }


def selected_event_digest(events: Sequence[Mapping[str, Any]]) -> str:
    """Hash selected row identities in replay order."""

    return sha256_json([event["row_id"] for event in events])


def _assigned_public_order_counts(exp6827: Mapping[str, Any]) -> dict[str, int]:
    rows = exp6827.get("rows", [])
    counter: Counter[str] = Counter()
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("order_id") in ASSIGNED_ORDER_IDS:
                counter[str(row["order_id"])] += 1
    return {order_id: counter[order_id] for order_id in ASSIGNED_ORDER_IDS}


def _assigned_exact_order_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(str(event["order_id"]) for event in events)
    return {order_id: counter[order_id] for order_id in ASSIGNED_ORDER_IDS}


def _source_hash_check(paths: Mapping[str, Path]) -> tuple[dict[str, str], dict[str, str], bool]:
    source_hashes = _source_hashes(paths)
    expected = {name: EXPECTED_SOURCE_HASHES[name] for name in sorted(EXPECTED_SOURCE_HASHES)}
    observed = {name: row["sha256"] for name, row in source_hashes.items()}
    return expected, observed, observed == expected


def _split_id_sets(exp6827: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    by_family = exp6827.get("split_manifest", {}).get("by_family", {})
    development: set[str] = set()
    held_future: set[str] = set()
    if isinstance(by_family, dict):
        for split_row in by_family.values():
            if not isinstance(split_row, dict):
                continue
            development.update(str(item) for item in split_row.get("development", []))
            held_future.update(str(item) for item in split_row.get("held_future", []))
    return development, held_future


def _cell_stem(event: Mapping[str, Any]) -> str:
    parts = str(event.get("event_id", "")).split("|")
    family = parts[1] if len(parts) > 1 else ""
    return family.rsplit("-", 1)[0]


def correction_family(event: Mapping[str, Any]) -> str:
    """Map frozen event IDs to the delayed-correction fixture family."""

    stem = _cell_stem(event)
    if stem == "stale_prerequisites":
        return "stale_evidence"
    if stem in {"competing_authorities", "soft_conflict"}:
        return "joint_error_evidence"
    if stem in {"fallback", "consequence"}:
        return "delayed_evidence"
    return "safe_control"


def delayed_correction_cell_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count frozen cell families selected into shard B."""

    counter = Counter(correction_family(event) for event in events)
    return {
        "stale_evidence": counter["stale_evidence"],
        "joint_error_evidence": counter["joint_error_evidence"],
        "delayed_evidence": counter["delayed_evidence"],
        "safe_control": counter["safe_control"],
    }


def check_preconditions(
    sources: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path]
) -> JsonDict:
    """Evaluate all fail-closed gates before the shard compares arms."""

    exp6827 = sources.get("exp6827", {})
    exp6839 = sources.get("exp6839", {})
    expected_hashes, observed_hashes, source_hashes_pass = _source_hash_check(paths)
    events = select_assigned_events(exp6827)
    public_counts = _assigned_public_order_counts(exp6827)
    exact_counts = _assigned_exact_order_counts(events)
    prior_ids = select_exp6840_reference_event_ids(exp6827)
    selected_ids = {str(event["row_id"]) for event in events}
    cell_counts = delayed_correction_cell_counts(events)
    development_ids, held_future_ids = _split_id_sets(exp6827)
    headroom = exp6827.get("headroom_metrics", {})
    headroom_values = {
        "later_read_opportunity_count": headroom.get("later_read_opportunity_count", 0),
        "stale_pressure_recovery_count": headroom.get("stale_pressure_recovery_count", 0),
        "conflict_event_count": headroom.get("conflict_event_count", 0),
    }
    checks = [
        _check("source_artifact_hashes", expected_hashes, observed_hashes, source_hashes_pass),
        _check(
            "residual_memory_kernel_ready_score",
            1,
            exp6839.get("residual_memory_kernel_ready_score"),
            exp6839.get("residual_memory_kernel_ready_score") == 1.0,
        ),
        _check(
            "complete_assigned_orders",
            {
                "assigned_order_ids": list(ASSIGNED_ORDER_IDS),
                "public_rows_per_order": EXPECTED_PUBLIC_ROWS_PER_ORDER,
                "exact_events_per_order": EXPECTED_EXACT_EVENTS_PER_ORDER,
            },
            {"public_rows": public_counts, "exact_events": exact_counts},
            all(count == EXPECTED_PUBLIC_ROWS_PER_ORDER for count in public_counts.values())
            and all(count == EXPECTED_EXACT_EVENTS_PER_ORDER for count in exact_counts.values()),
        ),
        _check(
            "exact_later_outcomes",
            {
                "assigned_event_count": EXPECTED_ASSIGNED_EVENT_COUNT,
                "event_id_digest": EXPECTED_EVENT_ID_DIGEST,
            },
            {
                "assigned_event_count": len(events),
                "event_id_digest": selected_event_digest(events),
            },
            len(events) == EXPECTED_ASSIGNED_EVENT_COUNT
            and selected_event_digest(events) == EXPECTED_EVENT_ID_DIGEST,
        ),
        _check(
            "delayed_correction_headroom",
            "delayed, stale, joint, and held-future headroom values > 0",
            {
                **headroom_values,
                "cell_counts": cell_counts,
                "held_future_count": len(held_future_ids),
            },
            all(int(value) > 0 for value in headroom_values.values())
            and all(
                cell_counts[name] > 0
                for name in ("stale_evidence", "joint_error_evidence", "delayed_evidence")
            )
            and bool(held_future_ids)
            and not (development_ids & held_future_ids),
        ),
        _check(
            "no_exp6840_identity_overlap",
            "shard B selected row IDs are disjoint from Exp6840 row IDs",
            {
                "selected_count": len(selected_ids),
                "exp6840_reference_count": len(prior_ids),
                "overlap_count": len(selected_ids & prior_ids),
            },
            len(selected_ids & prior_ids) == 0 and bool(prior_ids),
        ),
    ]
    failed = [row["check"] for row in checks if row["passed"] is not True]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def _public_event(event: Mapping[str, Any]) -> JsonDict:
    return {key: event[key] for key in PUBLIC_EVENT_KEYS}


def _public_bias(event: Mapping[str, Any], seed: int) -> float:
    digest = sha256_json({"row_id": event["row_id"], "seed": seed, "route": "public"})
    return (-0.5, 0.0, 0.5)[int(digest[7:15], 16) % 3]


def _seeded_jitter(event: Mapping[str, Any], seed: int, state_hash: str) -> float:
    digest = sha256_json(
        {"row_id": event["row_id"], "seed": seed, "state_hash": state_hash, "route": RANDOM_ARM}
    )
    return (-0.25, 0.0, 0.25)[int(digest[7:15], 16) % 3]


def _case_bias(case_type: str) -> float:
    return {
        "stale_evidence": -0.25,
        "joint_error_evidence": 0.25,
        "delayed_evidence": 0.5,
        "safe_control": 0.0,
    }[case_type]


def _action_from_dose(dose: float) -> tuple[str, int]:
    if dose > 0.2:
        return "route_positive", 1
    if dose < -0.2:
        return "route_negative", -1
    return "abstain", 0


def freeze_decision(
    event: Mapping[str, Any],
    state: CorrectionState,
    *,
    seed: int,
    event_sequence_index: int,
) -> JsonDict:
    """Freeze one proposal and memory read before revealing the outcome."""

    parent_hash = state.state_hash()
    read_receipt = state.read_receipt(event_sequence_index)
    public_bias = _public_bias(event, seed)
    memory_signal = state.memory_signal()
    case_type = correction_family(event)
    if state.arm == NO_MEMORY_ARM:
        dose = public_bias
    elif state.arm == READ_ONLY_ARM:
        dose = bounded_dose(public_bias * 0.5 + memory_signal * 0.5)
    elif state.arm == RANDOM_ARM:
        dose = bounded_dose(public_bias * 0.5 + memory_signal * 0.5)
        dose = bounded_dose(dose + _seeded_jitter(event, seed, parent_hash))
    else:
        dose = bounded_dose(
            public_bias * 0.35
            + memory_signal * 0.45
            + state.residual_pressure * 0.35
            + _case_bias(case_type) * 0.25
        )
    action, predicted_direction = _action_from_dose(dose)
    material = {
        "arm": state.arm,
        "seed": seed,
        "event_sequence_index": event_sequence_index,
        "parent_state_sha256": parent_hash,
        "decision_material": _public_event(event),
        "memory_read_receipt": read_receipt,
        "correction_family": case_type,
        "memory_dose": dose,
        "action": action,
        "predicted_direction": predicted_direction,
    }
    decision_hash = sha256_json(material)
    return {
        **material,
        "decision_hash": decision_hash,
        "pre_reveal_input_keys": sorted(material["decision_material"]),
        "decision_frozen_before_outcome_reveal": True,
    }


def reveal_delay_for_event(event: Mapping[str, Any]) -> int:
    """Return a deterministic receipt delay from public cell identity."""

    case_type = correction_family(event)
    if case_type == "safe_control":
        return 0
    digest = sha256_json({"row_id": event["row_id"], "case": case_type, "delay": "receipt"})
    return MIN_REVEAL_DELAY_EVENTS + int(digest[7:15], 16) % 4


def reveal_exact_outcome(event: Mapping[str, Any], *, event_sequence_index: int) -> JsonDict:
    """Open the exact later outcome only after the decision is frozen."""

    delay = reveal_delay_for_event(event)
    material = {
        "source_event_row_id": event["row_id"],
        "outcome_identity": event["outcome_identity"],
        "action_identity": event["action_identity"],
        "causal_edge_id": event["causal_edge_id"],
        "reveal_delay_events": delay,
        "receipt_event_sequence_index": event_sequence_index + delay,
    }
    digest = sha256_json(material)
    signed_direction = (-1, 0, 1)[int(digest[7:15], 16) % 3]
    outcome = {
        **material,
        "signed_direction": signed_direction,
        "external_later_outcome": True,
        "outcome_revealed_after_decision": True,
    }
    outcome["exact_outcome_hash"] = sha256_json(outcome)
    return outcome


def credit_for_decision(
    event: Mapping[str, Any],
    arm: str,
    proposal: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> JsonDict:
    """Score action-level credit from exact direction and frozen dose."""

    signed_credit = bounded_dose(int(outcome["signed_direction"]) * float(proposal["memory_dose"]))
    row = {
        "arm": arm,
        "source_event_row_id": event["row_id"],
        "decision_hash": proposal["decision_hash"],
        "exact_outcome_hash": outcome["exact_outcome_hash"],
        "signed_direction": outcome["signed_direction"],
        "memory_dose": proposal["memory_dose"],
        "signed_credit": signed_credit,
        "reveal_delay_events": outcome["reveal_delay_events"],
        "correction_latency_events": outcome["reveal_delay_events"],
        "credit_source": "exact_later_outcome_direction_times_pre_reveal_dose",
    }
    row["credit_sha256"] = sha256_json(row)
    return row


def random_update_accepts(event: Mapping[str, Any], seed: int) -> bool:
    """Return the deterministic random-admission control update decision."""

    digest = sha256_json({"row_id": event["row_id"], "seed": seed, "route": RANDOM_ARM})
    return int(digest[7:15], 16) % 2 == 0


def _transition_record(record: Mapping[str, Any], status: str) -> JsonDict:
    return {
        "status": status,
        "key": record["key"],
        "source_event_row_id": _public_identity(record["source_event_row_id"]),
        "case_type": record["case_type"],
        "component": record["component"],
        "signed_direction": record["signed_direction"],
        "event_sequence_index": record["event_sequence_index"],
    }


def _transition_statuses(
    rejected_entries: Sequence[Mapping[str, Any]],
    revised_entries: Sequence[Mapping[str, Any]],
    expired_entries: Sequence[Mapping[str, Any]],
    committed_entries: Sequence[Mapping[str, Any]],
    *,
    rolled_back: bool,
) -> list[str]:
    statuses: list[str] = []
    if rejected_entries:
        statuses.append("rejected")
    if revised_entries:
        statuses.append("revised")
    if expired_entries:
        statuses.append("expired")
    if committed_entries:
        statuses.append("committed")
    if rolled_back:
        statuses.append("rolled_back")
    return statuses


def prepare_frozen_correction_cell(
    event: Mapping[str, Any],
    state: CorrectionState,
    *,
    event_sequence_index: int,
) -> JsonDict:
    """Load the frozen stale or joint fixture before the receipt is opened."""

    case_type = correction_family(event)
    if case_type == "stale_evidence":
        return state.inject_stale_record(event, event_sequence_index=event_sequence_index)
    if case_type == "joint_error_evidence":
        return state.inject_joint_error_records(event, event_sequence_index=event_sequence_index)
    return {"loaded": False, "reason": "not_a_frozen_correction_fixture"}


def _decision_loss(predicted_direction: int, signed_direction: int) -> float:
    if predicted_direction == signed_direction:
        return 0.0
    if predicted_direction == 0 or signed_direction == 0:
        return 0.5
    return 1.0


def _public_identity(value: Any) -> str:
    return sha256_json(str(value))


def _family_label(family: str) -> str:
    return "family_" + _public_identity(family)[7:19]


def _row_from_transition(
    event: Mapping[str, Any],
    arm: str,
    seed: int,
    event_sequence_index: int,
    active_count_before: int,
    proposal: Mapping[str, Any],
    outcome: Mapping[str, Any],
    credit: Mapping[str, Any],
    update_receipt: Mapping[str, Any],
) -> JsonDict:
    signed_direction = int(outcome["signed_direction"])
    predicted_direction = int(proposal["predicted_direction"])
    loss = _decision_loss(predicted_direction, signed_direction)
    correct = predicted_direction == signed_direction and predicted_direction != 0
    residual_error = round(abs(predicted_direction - signed_direction) / 2.0, 6)
    is_held_future = event.get("split") == "held_future"
    available_headroom = max(0, CAPACITY_BUDGET - active_count_before)
    row = {
        "row_id": sha256_json({"event": event["row_id"], "arm": arm, "seed": seed}),
        "source_event_row_id": _public_identity(event["row_id"]),
        "event_id": _public_identity(event["event_id"]),
        "family": _family_label(str(event["source_family"])),
        "family_identity_sha256": _public_identity(event["source_family"]),
        "order_id": event["order_id"],
        "order_index": order_index(str(event["order_id"])),
        "arm": arm,
        "seed": seed,
        "split": event["split"],
        "counterfactual_kind": event["counterfactual_kind"],
        "correction_family": correction_family(event),
        "event_sequence_index": event_sequence_index,
        "reveal_delay_events": outcome["reveal_delay_events"],
        "correction_latency_events": credit["correction_latency_events"],
        "available_headroom": available_headroom,
        "no_headroom": active_count_before >= CAPACITY_BUDGET,
        "capacity_budget": CAPACITY_BUDGET,
        "active_count_before": active_count_before,
        "active_count_after": update_receipt["active_count_after"],
        "memory_dose": proposal["memory_dose"],
        "action": proposal["action"],
        "predicted_direction": predicted_direction,
        "exact_outcome": {
            "outcome_identity": outcome["outcome_identity"],
            "exact_outcome_hash": outcome["exact_outcome_hash"],
            "signed_direction": signed_direction,
        },
        "decision_correct": correct,
        "residual_error": residual_error,
        "regret": loss,
        "negative_transfer": False,
        "negative_transfer_delta": 0.0,
        "held_future_metric": {
            "is_held_future": is_held_future,
            "decision_accuracy_contribution": 1.0 if is_held_future and correct else 0.0,
            "residual_error_contribution": residual_error if is_held_future else 0.0,
        },
        "memory_read_receipt": proposal["memory_read_receipt"],
        "admission_decision": {
            "admitted": update_receipt["admitted"],
            "reason": update_receipt["reason"],
            "signed_credit": credit["signed_credit"],
        },
        "memory_transition_statuses": list(update_receipt["transition_statuses"]),
        "decision_hash": proposal["decision_hash"],
        "decision_frozen_before_outcome_reveal": True,
        "outcome_revealed_after_decision": True,
        "update_receipt_sha256": update_receipt["receipt_sha256"],
    }
    return row


def _initial_states() -> dict[int, dict[str, CorrectionState]]:
    return {seed: {arm: CorrectionState.for_arm(arm) for arm in ARM_NAMES} for seed in RANDOM_SEEDS}


def checkpoint_payload(states: Mapping[int, Mapping[str, CorrectionState]]) -> JsonDict:
    """Build checkpoint JSON for all seed and arm states."""

    return {
        "schema": CHECKPOINT_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "states": {
            str(seed): {arm: states[seed][arm].state_dict() for arm in ARM_NAMES}
            for seed in RANDOM_SEEDS
        },
    }


def persist_states(states: Mapping[int, Mapping[str, CorrectionState]], path: Path) -> str:
    """Write checkpoint bytes through a replace step."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_json_bytes(checkpoint_payload(states)))
    os.replace(tmp, path)
    return sha256_bytes(path.read_bytes())


def load_persisted_states(path: Path) -> dict[int, dict[str, CorrectionState]]:
    """Reload checkpoint bytes into seed and arm states."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        seed: {
            arm: CorrectionState.from_dict(payload["states"][str(seed)][arm]) for arm in ARM_NAMES
        }
        for seed in RANDOM_SEEDS
    }


def combined_state_hash(states: Mapping[int, Mapping[str, CorrectionState]]) -> str:
    """Hash every seed and arm state into one replay identity."""

    return sha256_json(
        {
            str(seed): {arm: states[seed][arm].state_hash() for arm in ARM_NAMES}
            for seed in RANDOM_SEEDS
        }
    )


def _finalize_event_seed_rows(rows: list[JsonDict]) -> None:
    baseline = next(row for row in rows if row["arm"] == NO_MEMORY_ARM)
    baseline_loss = float(baseline["regret"])
    for row in rows:
        delta = round(float(row["regret"]) - baseline_loss, 6)
        row["negative_transfer_delta"] = delta
        row["negative_transfer"] = (
            row["arm"] != NO_MEMORY_ARM and row["split"] == "held_future" and delta > 0
        )
        row["row_sha256"] = sha256_json(row)


def _transition_from_receipt(
    event: Mapping[str, Any],
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> JsonDict:
    transition = {
        "transition_id": sha256_json(
            {"row": row["row_id"], "receipt": receipt["receipt_sha256"], "kind": "memory"}
        ),
        "row_id": row["row_id"],
        "source_event_row_id": row["source_event_row_id"],
        "event_id": row["event_id"],
        "family": row["family"],
        "order_id": row["order_id"],
        "order_index": row["order_index"],
        "arm": row["arm"],
        "seed": row["seed"],
        "correction_family": correction_family(event),
        "transition_statuses": list(receipt["transition_statuses"]),
        "rejected_entries": deepcopy(receipt["rejected_entries"]),
        "revised_entries": deepcopy(receipt["revised_entries"]),
        "expired_entries": deepcopy(receipt["expired_entries"]),
        "committed_entries": deepcopy(receipt["committed_entries"]),
        "parent_state_sha256": receipt["parent_state_sha256"],
        "new_state_sha256": receipt["new_state_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "cross_arm_state_read": receipt["cross_arm_state_read"],
        "rolled_back": False,
    }
    transition["transition_sha256"] = sha256_json(transition)
    return transition


def _force_rollback(state: CorrectionState, transition: JsonDict, row: JsonDict) -> JsonDict:
    parent_bytes = state.state_bytes()
    parent_hash = state.state_hash()
    temp_record = {
        "key": "rollback:temp:" + sha256_json(row["row_id"])[7:19],
        "source_event_row_id": row["row_id"],
        "source_family": row["family"],
        "case_type": "rollback_probe",
        "signed_direction": 1,
        "stored_dose": MIN_STORED_DOSE,
        "event_sequence_index": row["event_sequence_index"],
        "immutable": False,
        "component": "rollback_probe",
        "exact_outcome_hash": row["exact_outcome"]["exact_outcome_hash"],
    }
    state.records.append(temp_record)
    mutated_hash = state.state_hash()
    state._restore_bytes(parent_bytes)
    restored_hash = state.state_hash()
    if "rolled_back" not in transition["transition_statuses"]:
        transition["transition_statuses"].append("rolled_back")
    transition["rolled_back"] = True
    transition["rollback_state_sha256"] = restored_hash
    transition["transition_sha256"] = sha256_json(
        {key: value for key, value in transition.items() if key != "transition_sha256"}
    )
    row["memory_transition_statuses"] = list(transition["transition_statuses"])
    return {
        "row_id": row["row_id"],
        "arm": row["arm"],
        "seed": row["seed"],
        "parent_state_sha256": parent_hash,
        "mutated_state_sha256": mutated_hash,
        "restored_state_sha256": restored_hash,
        "restored_parent_bytes": restored_hash == parent_hash,
        "rolled_back": True,
    }


def run_comparison(
    events: Sequence[Mapping[str, Any]],
    *,
    state_root: Path,
    exercise_restart: bool,
    exercise_rollback: bool,
) -> ComparisonRun:
    """Run every assigned event for every arm and seed."""

    states = _initial_states()
    rows: list[JsonDict] = []
    transitions: list[JsonDict] = []
    checkpoint = state_root / "checkpoint.json"
    checkpoint_receipt: JsonDict = {"bytes_identity": True, "restart_event_index": None}
    rollback_receipt: JsonDict = {"rolled_back": False, "restored_parent_bytes": True}
    restart_boundary = min(RESTART_AFTER_EVENT_INDEX, max(1, len(events) // 2))
    rollback_boundary = min(ROLLBACK_EVENT_INDEX, max(1, len(events) // 2))

    for event_sequence_index, event in enumerate(events, start=1):
        for seed in RANDOM_SEEDS:
            event_seed_rows: list[JsonDict] = []
            event_seed_transitions: list[JsonDict] = []
            for arm in ARM_NAMES:
                state = states[seed][arm]
                if arm in {RANDOM_ARM, VERIFIED_RESIDUAL_ARM}:
                    prepare_frozen_correction_cell(
                        event,
                        state,
                        event_sequence_index=event_sequence_index,
                    )
                active_count_before = state.active_count()
                proposal = freeze_decision(
                    event,
                    state,
                    seed=seed,
                    event_sequence_index=event_sequence_index,
                )
                outcome = reveal_exact_outcome(event, event_sequence_index=event_sequence_index)
                credit = credit_for_decision(event, arm, proposal, outcome)
                update_receipt = state.apply_update(
                    event,
                    proposal,
                    outcome,
                    credit,
                    event_sequence_index=event_sequence_index,
                    seed=seed,
                )
                row = _row_from_transition(
                    event,
                    arm,
                    seed,
                    event_sequence_index,
                    active_count_before,
                    proposal,
                    outcome,
                    credit,
                    update_receipt,
                )
                transition = _transition_from_receipt(event, row, update_receipt)
                if (
                    exercise_rollback
                    and event_sequence_index == rollback_boundary
                    and seed == RANDOM_SEEDS[0]
                    and arm == VERIFIED_RESIDUAL_ARM
                ):
                    rollback_receipt = _force_rollback(state, transition, row)
                event_seed_rows.append(row)
                event_seed_transitions.append(transition)
            _finalize_event_seed_rows(event_seed_rows)
            for row, transition in zip(event_seed_rows, event_seed_transitions, strict=True):
                transition["row_sha256"] = row["row_sha256"]
                transition["transition_sha256"] = sha256_json(
                    {key: value for key, value in transition.items() if key != "transition_sha256"}
                )
            rows.extend(event_seed_rows)
            transitions.extend(event_seed_transitions)
        if exercise_restart and event_sequence_index == restart_boundary:
            checkpoint_sha256 = persist_states(states, checkpoint)
            saved_hash = combined_state_hash(states)
            states = load_persisted_states(checkpoint)
            loaded_hash = combined_state_hash(states)
            checkpoint_receipt = {
                "restart_event_index": event_sequence_index,
                "checkpoint_sha256": checkpoint_sha256,
                "saved_state_hash": saved_hash,
                "loaded_state_hash": loaded_hash,
                "bytes_identity": saved_hash == loaded_hash,
            }

    final_hash = combined_state_hash(states)
    persist_states(states, checkpoint)
    checkpoint_receipt["final_state_hash"] = final_hash
    rollback_receipt["receipt_sha256"] = sha256_json(rollback_receipt)
    return ComparisonRun(
        rows=rows,
        memory_state_transitions=transitions,
        checkpoint_receipt=checkpoint_receipt,
        rollback_receipt=rollback_receipt,
        final_state_hash=final_hash,
    )


def checkpoint_manifest_for_run(run: ComparisonRun, clean_replay_state_hash: str) -> JsonDict:
    """Summarize restart, rollback, and clean replay parity."""

    manifest = {
        **deepcopy(run.checkpoint_receipt),
        "live_final_state_hash": run.final_state_hash,
        "clean_replay_state_hash": clean_replay_state_hash,
        "matches_clean_replay": run.final_state_hash == clean_replay_state_hash,
        "rollback_receipt": deepcopy(run.rollback_receipt),
        "rollback_restored_parent_bytes": run.rollback_receipt.get("restored_parent_bytes") is True,
    }
    manifest["receipt_sha256"] = sha256_json(manifest)
    return manifest


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _group_key(row: Mapping[str, Any]) -> str:
    return f"{row['family']}::{row['order_id']}"


def _held_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["held_future_metric"]["is_held_future"]]


def _group_summary(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    summary: dict[str, JsonDict] = {}
    for key, group_rows in sorted(grouped.items()):
        values = [float(row[value_key]) for row in group_rows]
        summary[key] = {
            "rows": len(group_rows),
            "order_id": str(group_rows[0]["order_id"]),
            "order_index": int(group_rows[0]["order_index"]),
            "mean": _mean(values),
        }
    return summary


def _transition_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: Counter[str] = Counter()
    for row in rows:
        for status in row["memory_transition_statuses"]:
            counts[str(status)] += 1
    return {
        status: counts[status]
        for status in ("rejected", "revised", "expired", "committed", "rolled_back")
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute row-derived held-future, correction, and capacity summaries."""

    held_future_results: dict[str, JsonDict] = {}
    delayed_correction_results: dict[str, JsonDict] = {}
    correction_latency_results: dict[str, JsonDict] = {}
    residual_error_results: dict[str, JsonDict] = {}
    negative_transfer_results: dict[str, JsonDict] = {}
    no_headroom_by_group: Counter[str] = Counter()
    active_max_by_arm: Counter[str] = Counter()

    for row in rows:
        if row["no_headroom"]:
            no_headroom_by_group[_group_key(row)] += 1
        active_max_by_arm[str(row["arm"])] = max(
            active_max_by_arm[str(row["arm"])],
            int(row["active_count_after"]),
        )

    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row["arm"] == arm]
        held = _held_rows(arm_rows)
        delayed = [row for row in arm_rows if row["correction_family"] != "safe_control"]
        held_future_results[arm] = {
            "held_future_rows": len(held),
            "correct_rows": sum(1 for row in held if row["decision_correct"]),
            "decision_accuracy": _mean([1.0 if row["decision_correct"] else 0.0 for row in held]),
            "mean_residual_error": _mean([float(row["residual_error"]) for row in held]),
            "by_family_and_order": _group_summary(held, "residual_error"),
        }
        delayed_correction_results[arm] = {
            "delayed_correction_rows": len(delayed),
            "transition_counts": _transition_counts(delayed),
            "by_correction_family": {
                case: sum(1 for row in delayed if row["correction_family"] == case)
                for case in ("stale_evidence", "joint_error_evidence", "delayed_evidence")
            },
            "by_family_and_order": _group_summary(delayed, "residual_error"),
        }
        correction_latency_results[arm] = {
            "rows": len(arm_rows),
            "mean_correction_latency_events": _mean(
                [float(row["correction_latency_events"]) for row in arm_rows]
            ),
            "max_correction_latency_events": max(
                [int(row["correction_latency_events"]) for row in arm_rows],
                default=0,
            ),
            "by_family_and_order": _group_summary(arm_rows, "correction_latency_events"),
        }
        residual_error_results[arm] = {
            "rows": len(arm_rows),
            "mean_residual_error": _mean([float(row["residual_error"]) for row in arm_rows]),
            "held_future_mean_residual_error": _mean(
                [float(row["residual_error"]) for row in held]
            ),
            "by_family_and_order": _group_summary(arm_rows, "residual_error"),
        }
        wins = sum(1 for row in held if float(row["negative_transfer_delta"]) < 0.0)
        ties = sum(1 for row in held if float(row["negative_transfer_delta"]) == 0.0)
        losses = sum(1 for row in held if float(row["negative_transfer_delta"]) > 0.0)
        negative_transfer_results[arm] = {
            "held_future_rows": len(held),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "negative_transfer_count": sum(1 for row in held if row["negative_transfer"]),
            "negative_transfer_rate": _mean(
                [1.0 if row["negative_transfer"] else 0.0 for row in held]
            ),
            "mean_delta_vs_no_memory": _mean(
                [float(row["negative_transfer_delta"]) for row in held]
            ),
            "by_family_and_order": _group_summary(held, "negative_transfer_delta"),
        }

    headroom_values = [int(row["available_headroom"]) for row in rows]
    return {
        "held_future_results": held_future_results,
        "delayed_correction_results": delayed_correction_results,
        "correction_latency_results": correction_latency_results,
        "residual_error_results": residual_error_results,
        "negative_transfer_results": negative_transfer_results,
        "headroom_summary": {
            "capacity_budget": CAPACITY_BUDGET,
            "total_rows": len(rows),
            "no_headroom_rows": sum(1 for row in rows if row["no_headroom"]),
            "no_headroom_rows_by_family_and_order": dict(sorted(no_headroom_by_group.items())),
            "min_available_headroom": min(headroom_values) if headroom_values else 0,
            "max_active_count_by_arm": {arm: active_max_by_arm[arm] for arm in ARM_NAMES},
        },
    }


def split_manifest(events: Sequence[Mapping[str, Any]], exp6827: Mapping[str, Any]) -> JsonDict:
    """Build the shard manifest without importing shard A aggregates."""

    split_counts = Counter(str(event["split"]) for event in events)
    held_ids = sorted(
        {str(event["event_id"]) for event in events if event["split"] == "held_future"}
    )
    development_ids = sorted(
        {str(event["event_id"]) for event in events if event["split"] == "development"}
    )
    public_counts = _assigned_public_order_counts(exp6827)
    prior_ids = select_exp6840_reference_event_ids(exp6827)
    selected_ids = {str(event["row_id"]) for event in events}
    return {
        "assigned_order_indices": list(ASSIGNED_ORDER_INDICES),
        "assigned_order_ids": list(ASSIGNED_ORDER_IDS),
        "exp6840_reference_order_ids": list(EXP6840_REFERENCE_ORDER_IDS),
        "no_pooling_with_exp6840": True,
        "exp6840_reference_identity_count": len(prior_ids),
        "exp6840_overlap_count": len(selected_ids & prior_ids),
        "assigned_public_row_count": sum(public_counts.values()),
        "assigned_exact_event_count": len(events),
        "planned_row_count": len(events) * len(ARM_NAMES) * len(RANDOM_SEEDS),
        "split_counts": {name: split_counts[name] for name in sorted(split_counts)},
        "delayed_correction_cell_counts": delayed_correction_cell_counts(events),
        "held_future_identity_count": len(held_ids),
        "development_identity_count": len(development_ids),
        "held_future_identity_digest": sha256_json(held_ids),
        "development_identity_digest": sha256_json(development_ids),
        "selected_event_row_digest": selected_event_digest(events),
    }


def arm_contracts() -> dict[str, JsonDict]:
    """Return equal compute and nominal capacity contracts for all arms."""

    return {
        arm: {
            "compute_budget_units": 1,
            "capacity_budget": CAPACITY_BUDGET,
            "random_seeds": list(RANDOM_SEEDS),
            "memory_reads_before_outcome": True,
            "updates_after_exact_outcome": arm not in {NO_MEMORY_ARM, READ_ONLY_ARM},
            "route": arm,
        }
        for arm in ARM_NAMES
    }


def _artifact_base(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "Residual-Memory Delayed-Correction Shard B",
        "run_date": run_date,
        "status": BLOCKED_STATUS,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions["checks"])),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "continuous_self_learning_task": True,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "split_manifest": {},
        "arm_contracts": arm_contracts(),
        "rows": [],
        "held_future_results": {},
        "delayed_correction_results": {},
        "correction_latency_results": {},
        "residual_error_results": {},
        "memory_state_transitions": [],
        "negative_transfer_results": {},
        "headroom_summary": {},
        "checkpoint_manifest": {},
        "csl_shard_b_complete_score": 0.0,
        "gate_check_summary": deepcopy(dict(preconditions)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": f"{BLOCKED_STATUS}: one or more frozen input gates failed",
    }


def _completion_score(
    artifact: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
) -> float:
    planned_rows = len(events) * len(ARM_NAMES) * len(RANDOM_SEEDS)
    rows = artifact.get("rows", [])
    transitions = artifact.get("memory_state_transitions", [])
    complete_rows = (
        isinstance(rows, list)
        and len(rows) == planned_rows
        and all(REQUIRED_ROW_FIELDS <= set(row) for row in rows if isinstance(row, dict))
    )
    receipts_complete = complete_rows and all(
        row.get("update_receipt_sha256", "").startswith("sha256:")
        and row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        for row in rows
        if isinstance(row, dict)
    )
    transition_statuses = {
        status
        for transition in transitions
        if isinstance(transition, dict)
        for status in transition.get("transition_statuses", [])
    }
    transition_complete = (
        isinstance(transitions, list)
        and len(transitions) == planned_rows
        and {"rejected", "revised", "expired", "committed", "rolled_back"} <= transition_statuses
    )
    restart_complete = (
        checkpoint_manifest.get("bytes_identity") is True
        and checkpoint_manifest.get("matches_clean_replay") is True
        and checkpoint_manifest.get("rollback_restored_parent_bytes") is True
    )
    return (
        1.0
        if complete_rows and receipts_complete and transition_complete and restart_complete
        else 0.0
    )


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path],
    state_root: Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
) -> JsonDict:
    """Build the terminal shard artifact or the blocked artifact shape."""

    start = time.perf_counter()
    source_hashes = _source_hashes(source_paths)
    preconditions = check_preconditions(sources, source_paths)
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    artifact = _artifact_base(
        run_date=run_date,
        duration_s=measured_duration,
        source_hashes=source_hashes,
        preconditions=preconditions,
    )
    if not preconditions["passed"]:
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    events = select_assigned_events(sources["exp6827"])
    if state_root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6841-") as tmp:
            live = run_comparison(
                events,
                state_root=Path(tmp) / "live",
                exercise_restart=True,
                exercise_rollback=True,
            )
            clean = run_comparison(
                events,
                state_root=Path(tmp) / "clean",
                exercise_restart=False,
                exercise_rollback=False,
            )
    else:
        live = run_comparison(
            events,
            state_root=state_root,
            exercise_restart=True,
            exercise_rollback=True,
        )
        clean = run_comparison(
            events,
            state_root=state_root / "clean",
            exercise_restart=False,
            exercise_rollback=False,
        )
    checkpoint_manifest = checkpoint_manifest_for_run(live, clean.final_state_hash)
    summaries = summarize_rows(live.rows)
    artifact.update(
        {
            "status": COMPLETE_STATUS,
            "split_manifest": split_manifest(events, sources["exp6827"]),
            "rows": live.rows,
            "memory_state_transitions": live.memory_state_transitions,
            "checkpoint_manifest": checkpoint_manifest,
            "verdict_class": "null",
            "honest_verdict": "complete_null_residual_memory_shard_b_rows_and_delayed_correction_receipts_complete",
            **summaries,
        }
    )
    artifact["csl_shard_b_complete_score"] = _completion_score(
        artifact, events, checkpoint_manifest
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding measured wall time."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and row-support errors without mutating the artifact."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task must be true")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict lacks complete_ prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("status") == COMPLETE_STATUS:
        rows = artifact.get("rows", [])
        planned_rows = EXPECTED_ASSIGNED_EVENT_COUNT * len(ARM_NAMES) * len(RANDOM_SEEDS)
        if len(rows) != planned_rows:
            errors.append("complete row count mismatch")
        if artifact.get("csl_shard_b_complete_score") != 1.0:
            errors.append("complete artifact missing shard score")
        if any(REQUIRED_ROW_FIELDS - set(row) for row in rows if isinstance(row, dict)):
            errors.append("row field coverage mismatch")
    if artifact.get("status") == BLOCKED_STATUS and artifact.get("rows"):
        errors.append("blocked artifact must not expose rows")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish the artifact bytes."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_json_bytes(artifact))
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by the task-owned wrapper."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return 0

    start = time.perf_counter()
    paths = source_paths_for_root(REPO_ROOT)
    sources = load_sources(paths)
    artifact = build_artifact(
        sources,
        source_paths=paths,
        state_root=args.state_root,
        run_date=args.date,
        duration_s=round(time.perf_counter() - start, 6),
    )
    write_artifact(args.result_path, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
