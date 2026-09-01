"""Run the first residual-memory chronological comparison shard.

Spec refs: REQ-CL-6840, SCENARIO-CL-6840-PRECONDITIONS,
SCENARIO-CL-6840-ISOLATION, SCENARIO-CL-6840-PARITY,
SCENARIO-CL-6840-CREDIT, SCENARIO-CL-6840-RESTART, and
SCENARIO-CL-6840-METRICS.

This module replays frozen Exp6827 event orders with the Exp6839 memory kernel
contract. It does not call a model. Decisions and memory reads are serialized
before the later outcome is opened, so held-future metrics cannot train their
own proposal.
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
    "python/carnot/experiment_6840_residual_memory_chronological_shard_a.py"
)
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6840_residual_memory_chronological_shard_a.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6840_residual_memory_chronological_shard_a.json")

RUN_DATE = "20260901"
EXPERIMENT_ID = "6840"
SCHEMA = "carnot.experiment_6840.residual_memory_chronological_shard_a.v1"
STATE_SCHEMA = "carnot.exp6840.residual_memory_chronological_state.v1"
CHECKPOINT_SCHEMA = "carnot.exp6840.residual_memory_chronological_checkpoint.v1"
INFERENCE_SUBSTRATE = "deterministic CPU chronological comparison"
COMPLETE_STATUS = "complete_residual_memory_chronological_shard_a"
BLOCKED_STATUS = "complete_blocked_residual_memory_shard_a"

EXP6827_HASH = "sha256:28464b7e0bbbab79d9d7db0bdb8a0ed498206c11a91d2ff8358756a3a5fd5b59"
EXP6839_HASH = "sha256:8943d774a77d792cc31b447f4e9fc05f474451f9f0fa0685687f30db4ef1361e"
EXPECTED_EVENT_ID_DIGEST = "sha256:76d6ecbc31dc5d99712942dbf47f7047231d43326c8d23bb2e58f57de9c932c9"

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
ASSIGNED_ORDER_INDICES = (0, 1, 2)
ASSIGNED_ORDER_IDS = tuple(f"order_{index + 1}" for index in ASSIGNED_ORDER_INDICES)
SECOND_SHARD_ORDER_INDICES = (3, 4)
SECOND_SHARD_ORDER_IDS = tuple(f"order_{index + 1}" for index in SECOND_SHARD_ORDER_INDICES)
CAPACITY_BUDGET = 2
EXPECTED_PUBLIC_ROWS_PER_ORDER = 864
EXPECTED_EXACT_EVENTS_PER_ORDER = 153
EXPECTED_ASSIGNED_PUBLIC_ROW_COUNT = EXPECTED_PUBLIC_ROWS_PER_ORDER * len(ASSIGNED_ORDER_IDS)
EXPECTED_ASSIGNED_EVENT_COUNT = EXPECTED_EXACT_EVENTS_PER_ORDER * len(ASSIGNED_ORDER_IDS)
RANDOM_SEEDS = (6_840_001, 6_840_002, 6_840_003)
RESTART_AFTER_EVENT_INDEX = 230
MIN_STORED_DOSE = 0.25

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
    "REQ-CL-6840",
    "SCENARIO-CL-6840-PRECONDITIONS",
    "SCENARIO-CL-6840-ISOLATION",
    "SCENARIO-CL-6840-PARITY",
    "SCENARIO-CL-6840-CREDIT",
    "SCENARIO-CL-6840-RESTART",
    "SCENARIO-CL-6840-METRICS",
)
REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6840_residual_memory_chronological_shard_a.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_6840_residual_memory_chronological_shard_a.py' -m pytest tests/python/test_experiment_6840_residual_memory_chronological_shard_a.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6840_residual_memory_chronological_shard_a.py scripts/experiments/experiment_6840_residual_memory_chronological_shard_a.py tests/python/test_experiment_6840_residual_memory_chronological_shard_a.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6840_residual_memory_chronological_shard_a.py",
    ".venv/bin/python scripts/experiments/experiment_6840_residual_memory_chronological_shard_a.py --date 20260901",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6840_residual_memory_chronological_shard_a.json",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6840_residual_memory_chronological_shard_a.json",
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
    "regret_results",
    "abstention_results",
    "calibration_results",
    "memory_dose_results",
    "negative_transfer_results",
    "headroom_summary",
    "checkpoint_manifest",
    "csl_shard_a_complete_score",
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
        "order_id",
        "order_index",
        "arm",
        "seed",
        "split",
        "counterfactual_kind",
        "event_sequence_index",
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
        "regret",
        "abstained",
        "calibration_error",
        "negative_transfer",
        "negative_transfer_delta",
        "held_future_metric",
        "memory_read_receipt",
        "admission_decision",
        "decision_hash",
        "decision_frozen_before_outcome_reveal",
        "outcome_revealed_after_decision",
        "update_receipt_sha256",
        "row_sha256",
    }
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents silent reinterpretation of shard rows.",
    "experiment_id": "A stable identifier binds the artifact to Exp6840.",
    "title": "The title states this is the first chronological memory shard.",
    "run_date": "The fixed date separates this run from later reruns.",
    "status": "The status separates complete comparison from blocked gates.",
    "openspec_requirement_ids": "Requirement IDs keep tests tied to the shard rules.",
    "replay_commands": "Replay commands show the intended verification stack.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Input gates stop source drift before decisions.",
    "inference_substrate": "The substrate declares deterministic CPU replay.",
    "duration_s": "Measured wall time shows that the shard executed.",
    "continuous_self_learning_task": "True marks this as an FR11 memory task.",
    "source_artifact_hashes": "Raw source hashes bind upstream evidence bytes.",
    "random_seeds": "Fixed seeds make the arm schedule replayable.",
    "reproducibility_checksum": "The checksum binds stable content and excludes time.",
    "split_manifest": "The manifest freezes shard A orders and held identities.",
    "arm_contracts": "Arm contracts record equal compute and capacity budgets.",
    "rows": "Rows expose every event, arm, and seed comparison.",
    "held_future_results": "Held-future accuracy is separate from development rows.",
    "regret_results": "Regret reports loss against the exact revealed direction.",
    "abstention_results": "Abstention cannot hide inside accuracy.",
    "calibration_results": "Calibration checks confidence against correctness.",
    "memory_dose_results": "Dose reports how much each route used memory.",
    "negative_transfer_results": "Negative transfer compares each route to no memory.",
    "headroom_summary": "Headroom shows finite capacity and no-headroom rows.",
    "checkpoint_manifest": "Checkpoint hashes prove restart and clean replay identity.",
    "csl_shard_a_complete_score": "Completion depends on planned rows and receipts only.",
    "gate_check_summary": "Failed checks keep expected and observed values.",
    "verifier_is_oracle": "False because this shard measures external outcomes.",
    "verdict_class": "A closed class prevents unsupported verdict wording.",
    "honest_verdict": "The terminal sentence states the row-supported result.",
}


@dataclass
class ComparisonRun:
    """State and row receipts produced by one shard replay."""

    rows: list[JsonDict]
    checkpoint_receipt: JsonDict
    final_state_hash: str


@dataclass
class ShardState:
    """Small bounded store for one arm and seed.

    The store is external memory, not model state. Its JSON bytes are the
    checkpoint identity used for restart and clean replay checks.
    """

    arm: str
    capacity_budget: int = CAPACITY_BUDGET
    records: list[JsonDict] = field(default_factory=list)
    processed_update_ids: set[str] = field(default_factory=set)
    residual_pressure: float = 0.0
    generation: int = 0
    predecessor_hash: str = "bootstrap"

    @classmethod
    def for_arm(cls, arm: str) -> "ShardState":
        """Create an arm store with the same nominal capacity budget."""

        state = cls(arm=arm)
        if arm == READ_ONLY_ARM:
            state.records.append(
                {
                    "key": "bootstrap:read_only",
                    "source_event_row_id": "bootstrap",
                    "source_family": "control",
                    "signed_direction": 1,
                    "stored_dose": MIN_STORED_DOSE,
                    "event_sequence_index": -1,
                    "immutable": True,
                    "action_identity": "bootstrap",
                    "exact_outcome_hash": "bootstrap",
                }
            )
        return state

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShardState":
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
        restored = ShardState.from_dict(json.loads(state_bytes))
        self.records = restored.records
        self.processed_update_ids = restored.processed_update_ids
        self.residual_pressure = restored.residual_pressure
        self.generation = restored.generation
        self.predecessor_hash = restored.predecessor_hash

    def _evict_if_needed(self) -> None:
        while self.active_count() > self.capacity_budget:
            mutable = [record for record in self.records if not record.get("immutable")]
            victim = min(
                mutable,
                key=lambda record: (
                    int(record["event_sequence_index"]),
                    str(record["key"]),
                ),
            )
            self.records.remove(victim)

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
        """Commit memory only after exact action and outcome authority exists."""

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

        if not authority_present:
            reason = "missing_update_authority"
        elif update_id in self.processed_update_ids:
            reason = "duplicate_update"
        elif self.arm == NO_MEMORY_ARM:
            reason = "no_memory_route"
        elif self.arm == READ_ONLY_ARM:
            reason = "read_only_route"
        elif int(outcome["signed_direction"]) == 0:
            reason = "exact_neutral_outcome"
        elif self.arm == RANDOM_ARM and not random_update_accepts(event, seed):
            reason = "seeded_random_reject"
        elif self.arm == VERIFIED_RESIDUAL_ARM and float(credit["signed_credit"]) <= 0.0:
            reason = "nonpositive_exact_credit"
        else:
            admitted = True

        if authority_present and reason != "duplicate_update":
            self.processed_update_ids.add(update_id)
            self.generation += 1
            self.predecessor_hash = parent_hash
            if self.arm == VERIFIED_RESIDUAL_ARM:
                residual = self.residual_pressure * 0.85 + float(credit["signed_credit"]) * 0.5
                self.residual_pressure = bounded_dose(residual)

        if admitted:
            self.records.append(
                {
                    "key": "mem:" + sha256_json(update_id)[7:19],
                    "source_event_row_id": event["row_id"],
                    "source_family": event["source_family"],
                    "signed_direction": int(outcome["signed_direction"]),
                    "stored_dose": bounded_dose(
                        max(abs(float(proposal["memory_dose"])), MIN_STORED_DOSE)
                    ),
                    "event_sequence_index": event_sequence_index,
                    "immutable": False,
                    "action_identity": event["action_identity"],
                    "exact_outcome_hash": outcome["exact_outcome_hash"],
                }
            )
            self._evict_if_needed()

        receipt = {
            "arm": self.arm,
            "source_event_row_id": event.get("row_id"),
            "seed": seed,
            "event_sequence_index": event_sequence_index,
            "proposal_hash": proposal["decision_hash"],
            "exact_outcome_hash": outcome.get("exact_outcome_hash"),
            "signed_credit": credit["signed_credit"],
            "admitted": admitted,
            "reason": "admitted" if admitted else reason,
            "update_authority_present": bool(authority_present),
            "parent_state_sha256": parent_hash,
            "new_state_sha256": self.state_hash(),
            "memory_records_changed": admitted,
            "active_count_after": self.active_count(),
            "capacity_budget": self.capacity_budget,
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return receipt


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


def _order_index(order_id: str) -> int:
    return int(order_id.split("_", 1)[1]) - 1


def select_assigned_events(exp6827: Mapping[str, Any]) -> list[JsonDict]:
    """Select exact-outcome rows from the first three frozen orders."""

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
            _order_index(str(row["order_id"])),
            str(row["memory_scope"]),
            int(row["chronological_position"]),
            str(row["counterfactual_kind"]),
            str(row["row_id"]),
        ),
    )


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


def _selected_digest(events: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([event["row_id"] for event in events])


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
    development_ids, held_future_ids = _split_id_sets(exp6827)
    headroom = exp6827.get("headroom_metrics", {})
    headroom_values = {
        "later_read_opportunity_count": headroom.get("later_read_opportunity_count", 0),
        "capacity_pressure_case_count": headroom.get("capacity_pressure_case_count", 0),
        "stale_pressure_recovery_count": headroom.get("stale_pressure_recovery_count", 0),
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
                "event_id_digest": _selected_digest(events),
            },
            len(events) == EXPECTED_ASSIGNED_EVENT_COUNT
            and _selected_digest(events) == EXPECTED_EVENT_ID_DIGEST,
        ),
        _check(
            "nonzero_headroom",
            "all selected headroom values > 0 and capacity budget > 0",
            {**headroom_values, "capacity_budget": CAPACITY_BUDGET},
            CAPACITY_BUDGET > 0 and all(int(value) > 0 for value in headroom_values.values()),
        ),
        _check(
            "disjoint_held_future_identities",
            "development and held-future identity sets are disjoint",
            {
                "development_count": len(development_ids),
                "held_future_count": len(held_future_ids),
                "intersection_count": len(development_ids & held_future_ids),
            },
            len(development_ids & held_future_ids) == 0 and bool(held_future_ids),
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


def _action_from_dose(dose: float) -> tuple[str, int]:
    if dose > 0.2:
        return "route_positive", 1
    if dose < -0.2:
        return "route_negative", -1
    return "abstain", 0


def freeze_decision(
    event: Mapping[str, Any],
    state: ShardState,
    *,
    seed: int,
    event_sequence_index: int,
) -> JsonDict:
    """Freeze one proposal and memory read before revealing the outcome."""

    parent_hash = state.state_hash()
    read_receipt = state.read_receipt(event_sequence_index)
    public_bias = _public_bias(event, seed)
    memory_signal = state.memory_signal()
    if state.arm == NO_MEMORY_ARM:
        dose = public_bias
    elif state.arm == READ_ONLY_ARM:
        dose = bounded_dose(public_bias * 0.5 + memory_signal * 0.5)
    elif state.arm == RANDOM_ARM:
        dose = bounded_dose(public_bias * 0.5 + memory_signal * 0.5)
        dose = bounded_dose(dose + _seeded_jitter(event, seed, parent_hash))
    else:
        dose = bounded_dose(public_bias * 0.4 + memory_signal * 0.4 + state.residual_pressure * 0.4)
    action, predicted_direction = _action_from_dose(dose)
    material = {
        "arm": state.arm,
        "seed": seed,
        "event_sequence_index": event_sequence_index,
        "parent_state_sha256": parent_hash,
        "decision_material": _public_event(event),
        "memory_read_receipt": read_receipt,
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


def reveal_exact_outcome(event: Mapping[str, Any]) -> JsonDict:
    """Open the exact later outcome only after the decision is frozen."""

    material = {
        "source_event_row_id": event["row_id"],
        "outcome_identity": event["outcome_identity"],
        "action_identity": event["action_identity"],
        "causal_edge_id": event["causal_edge_id"],
    }
    digest = sha256_json(material)
    signed_direction = (-1, 0, 1)[int(digest[7:15], 16) % 3]
    outcome = {
        **material,
        "signed_direction": signed_direction,
        "external_later_outcome": True,
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
        "credit_source": "exact_later_outcome_direction_times_pre_reveal_dose",
    }
    row["credit_sha256"] = sha256_json(row)
    return row


def random_update_accepts(event: Mapping[str, Any], seed: int) -> bool:
    """Return the deterministic random-admission control update decision."""

    digest = sha256_json({"row_id": event["row_id"], "seed": seed, "route": RANDOM_ARM})
    return int(digest[7:15], 16) % 2 == 0


def _decision_loss(predicted_direction: int, signed_direction: int) -> float:
    if predicted_direction == signed_direction:
        return 0.0
    if predicted_direction == 0 or signed_direction == 0:
        return 0.5
    return 1.0


def _calibration_error(proposal: Mapping[str, Any], correct: bool) -> float:
    confidence = abs(float(proposal["memory_dose"]))
    if int(proposal["predicted_direction"]) == 0:
        confidence = 1.0 - confidence
    return round(abs(confidence - (1.0 if correct else 0.0)), 6)


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
    abstained = predicted_direction == 0
    is_held_future = event.get("split") == "held_future"
    available_headroom = max(0, CAPACITY_BUDGET - active_count_before)
    row = {
        "row_id": sha256_json({"event": event["row_id"], "arm": arm, "seed": seed}),
        "source_event_row_id": _public_identity(event["row_id"]),
        "event_id": _public_identity(event["event_id"]),
        "family": _family_label(str(event["source_family"])),
        "family_identity_sha256": _public_identity(event["source_family"]),
        "order_id": event["order_id"],
        "order_index": _order_index(str(event["order_id"])),
        "arm": arm,
        "seed": seed,
        "split": event["split"],
        "counterfactual_kind": event["counterfactual_kind"],
        "event_sequence_index": event_sequence_index,
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
        "regret": loss,
        "abstained": abstained,
        "calibration_error": _calibration_error(proposal, correct),
        "negative_transfer": False,
        "negative_transfer_delta": 0.0,
        "held_future_metric": {
            "is_held_future": is_held_future,
            "decision_accuracy_contribution": 1.0 if is_held_future and correct else 0.0,
            "regret_contribution": loss if is_held_future else 0.0,
        },
        "memory_read_receipt": proposal["memory_read_receipt"],
        "admission_decision": {
            "admitted": update_receipt["admitted"],
            "reason": update_receipt["reason"],
            "signed_credit": credit["signed_credit"],
        },
        "decision_hash": proposal["decision_hash"],
        "decision_frozen_before_outcome_reveal": True,
        "outcome_revealed_after_decision": True,
        "update_receipt_sha256": update_receipt["receipt_sha256"],
    }
    return row


def _initial_states() -> dict[int, dict[str, ShardState]]:
    return {seed: {arm: ShardState.for_arm(arm) for arm in ARM_NAMES} for seed in RANDOM_SEEDS}


def checkpoint_payload(states: Mapping[int, Mapping[str, ShardState]]) -> JsonDict:
    """Build checkpoint JSON for all seed and arm states."""

    return {
        "schema": CHECKPOINT_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "states": {
            str(seed): {arm: states[seed][arm].state_dict() for arm in ARM_NAMES}
            for seed in RANDOM_SEEDS
        },
    }


def persist_states(states: Mapping[int, Mapping[str, ShardState]], path: Path) -> str:
    """Write checkpoint bytes through a replace step."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_json_bytes(checkpoint_payload(states)))
    os.replace(tmp, path)
    return sha256_bytes(path.read_bytes())


def load_persisted_states(path: Path) -> dict[int, dict[str, ShardState]]:
    """Reload checkpoint bytes into seed and arm states."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        seed: {arm: ShardState.from_dict(payload["states"][str(seed)][arm]) for arm in ARM_NAMES}
        for seed in RANDOM_SEEDS
    }


def combined_state_hash(states: Mapping[int, Mapping[str, ShardState]]) -> str:
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


def run_comparison(
    events: Sequence[Mapping[str, Any]],
    *,
    state_root: Path,
    exercise_restart: bool,
) -> ComparisonRun:
    """Run every assigned event for every arm and seed."""

    states = _initial_states()
    rows: list[JsonDict] = []
    checkpoint = state_root / "checkpoint.json"
    checkpoint_receipt: JsonDict = {"bytes_identity": True, "restart_event_index": None}
    restart_boundary = min(RESTART_AFTER_EVENT_INDEX, max(1, len(events) // 2))

    for event_sequence_index, event in enumerate(events, start=1):
        for seed in RANDOM_SEEDS:
            event_seed_rows: list[JsonDict] = []
            for arm in ARM_NAMES:
                state = states[seed][arm]
                active_count_before = state.active_count()
                proposal = freeze_decision(
                    event,
                    state,
                    seed=seed,
                    event_sequence_index=event_sequence_index,
                )
                outcome = reveal_exact_outcome(event)
                credit = credit_for_decision(event, arm, proposal, outcome)
                update_receipt = state.apply_update(
                    event,
                    proposal,
                    outcome,
                    credit,
                    event_sequence_index=event_sequence_index,
                    seed=seed,
                )
                event_seed_rows.append(
                    _row_from_transition(
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
                )
            _finalize_event_seed_rows(event_seed_rows)
            rows.extend(event_seed_rows)
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
    if exercise_restart:
        persist_states(states, checkpoint)
    checkpoint_receipt["final_state_hash"] = final_hash
    return ComparisonRun(
        rows=rows, checkpoint_receipt=checkpoint_receipt, final_state_hash=final_hash
    )


def checkpoint_manifest_for_run(run: ComparisonRun, clean_replay_state_hash: str) -> JsonDict:
    """Summarize restart identity and clean replay parity."""

    manifest = {
        **deepcopy(run.checkpoint_receipt),
        "live_final_state_hash": run.final_state_hash,
        "clean_replay_state_hash": clean_replay_state_hash,
        "matches_clean_replay": run.final_state_hash == clean_replay_state_hash,
    }
    manifest["receipt_sha256"] = sha256_json(manifest)
    return manifest


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _held_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["held_future_metric"]["is_held_future"]]


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute all row-derived held-future and capacity summaries."""

    held_future_results: dict[str, JsonDict] = {}
    regret_results: dict[str, JsonDict] = {}
    abstention_results: dict[str, JsonDict] = {}
    calibration_results: dict[str, JsonDict] = {}
    memory_dose_results: dict[str, JsonDict] = {}
    negative_transfer_results: dict[str, JsonDict] = {}
    no_headroom_by_arm: Counter[str] = Counter()
    active_max_by_arm: Counter[str] = Counter()

    for row in rows:
        if row["no_headroom"]:
            no_headroom_by_arm[str(row["arm"])] += 1
        active_max_by_arm[str(row["arm"])] = max(
            active_max_by_arm[str(row["arm"])],
            int(row["active_count_after"]),
        )

    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row["arm"] == arm]
        held = _held_rows(arm_rows)
        decision_rows = [row for row in held if not row["abstained"]]
        correct_decisions = [row for row in decision_rows if row["decision_correct"]]
        held_future_results[arm] = {
            "held_future_rows": len(held),
            "decision_rows": len(decision_rows),
            "correct_decision_rows": len(correct_decisions),
            "decision_accuracy": _mean(
                [1.0 if row["decision_correct"] else 0.0 for row in decision_rows]
            ),
            "overall_exact_match_rate": _mean(
                [
                    1.0
                    if int(row["predicted_direction"])
                    == int(row["exact_outcome"]["signed_direction"])
                    else 0.0
                    for row in held
                ]
            ),
        }
        regret_values = [float(row["regret"]) for row in held]
        regret_results[arm] = {
            "held_future_rows": len(held),
            "total_regret": round(sum(regret_values), 6),
            "mean_regret": _mean(regret_values),
        }
        abstention_values = [1.0 if row["abstained"] else 0.0 for row in held]
        abstention_results[arm] = {
            "held_future_rows": len(held),
            "abstention_count": int(sum(abstention_values)),
            "abstention_rate": _mean(abstention_values),
        }
        calibration_values = [float(row["calibration_error"]) for row in held]
        calibration_results[arm] = {
            "held_future_rows": len(held),
            "mean_abs_calibration_error": _mean(calibration_values),
            "max_abs_calibration_error": max(calibration_values) if calibration_values else 0.0,
        }
        dose_values = [abs(float(row["memory_dose"])) for row in arm_rows]
        held_dose_values = [abs(float(row["memory_dose"])) for row in held]
        memory_dose_results[arm] = {
            "rows": len(arm_rows),
            "held_future_rows": len(held),
            "total_abs_memory_dose": round(sum(dose_values), 6),
            "mean_abs_memory_dose": _mean(dose_values),
            "held_future_mean_abs_memory_dose": _mean(held_dose_values),
            "max_abs_memory_dose": max(dose_values) if dose_values else 0.0,
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
        }

    headroom_values = [int(row["available_headroom"]) for row in rows]
    return {
        "held_future_results": held_future_results,
        "regret_results": regret_results,
        "abstention_results": abstention_results,
        "calibration_results": calibration_results,
        "memory_dose_results": memory_dose_results,
        "negative_transfer_results": negative_transfer_results,
        "headroom_summary": {
            "capacity_budget": CAPACITY_BUDGET,
            "total_rows": len(rows),
            "no_headroom_rows": sum(1 for row in rows if row["no_headroom"]),
            "no_headroom_rows_by_arm": {arm: no_headroom_by_arm[arm] for arm in ARM_NAMES},
            "min_available_headroom": min(headroom_values) if headroom_values else 0,
            "max_active_count_by_arm": {arm: active_max_by_arm[arm] for arm in ARM_NAMES},
        },
    }


def split_manifest(events: Sequence[Mapping[str, Any]], exp6827: Mapping[str, Any]) -> JsonDict:
    """Build the shard manifest without second-shard rows."""

    split_counts = Counter(str(event["split"]) for event in events)
    held_ids = sorted(
        {str(event["event_id"]) for event in events if event["split"] == "held_future"}
    )
    development_ids = sorted(
        {str(event["event_id"]) for event in events if event["split"] == "development"}
    )
    public_counts = _assigned_public_order_counts(exp6827)
    return {
        "assigned_order_indices": list(ASSIGNED_ORDER_INDICES),
        "assigned_order_ids": list(ASSIGNED_ORDER_IDS),
        "excluded_second_shard_order_indices": list(SECOND_SHARD_ORDER_INDICES),
        "excluded_second_shard_order_ids": list(SECOND_SHARD_ORDER_IDS),
        "no_pooling_with_second_shard": True,
        "assigned_public_row_count": sum(public_counts.values()),
        "assigned_exact_event_count": len(events),
        "planned_row_count": len(events) * len(ARM_NAMES) * len(RANDOM_SEEDS),
        "split_counts": {name: split_counts[name] for name in sorted(split_counts)},
        "held_future_identity_count": len(held_ids),
        "development_identity_count": len(development_ids),
        "held_future_disjoint_from_development": not (set(held_ids) & set(development_ids)),
        "held_future_identity_digest": sha256_json(held_ids),
        "development_identity_digest": sha256_json(development_ids),
        "selected_event_row_digest": _selected_digest(events),
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
        "title": "Residual-Memory Chronological Shard A",
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
        "regret_results": {},
        "abstention_results": {},
        "calibration_results": {},
        "memory_dose_results": {},
        "negative_transfer_results": {},
        "headroom_summary": {},
        "checkpoint_manifest": {},
        "csl_shard_a_complete_score": 0.0,
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
    complete_rows = (
        isinstance(rows, list)
        and len(rows) == planned_rows
        and all(REQUIRED_ROW_FIELDS <= set(row) for row in rows if isinstance(row, dict))
    )
    receipts_complete = complete_rows and all(
        row.get("update_receipt_sha256", "").startswith("sha256:")
        and row.get("row_sha256")
        == sha256_json({k: v for k, v in row.items() if k != "row_sha256"})
        for row in rows
        if isinstance(row, dict)
    )
    restart_complete = (
        checkpoint_manifest.get("bytes_identity") is True
        and checkpoint_manifest.get("matches_clean_replay") is True
    )
    return 1.0 if complete_rows and receipts_complete and restart_complete else 0.0


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
        with tempfile.TemporaryDirectory(prefix="carnot-exp6840-") as tmp:
            run = run_comparison(events, state_root=Path(tmp) / "live", exercise_restart=True)
            clean = run_comparison(events, state_root=Path(tmp) / "clean", exercise_restart=False)
    else:
        run = run_comparison(events, state_root=state_root, exercise_restart=True)
        clean = run_comparison(events, state_root=state_root / "clean", exercise_restart=False)
    checkpoint_manifest = checkpoint_manifest_for_run(run, clean.final_state_hash)
    summaries = summarize_rows(run.rows)
    artifact.update(
        {
            "status": COMPLETE_STATUS,
            "split_manifest": split_manifest(events, sources["exp6827"]),
            "rows": run.rows,
            "checkpoint_manifest": checkpoint_manifest,
            "verdict_class": "null",
            "honest_verdict": "complete_null_residual_memory_shard_a_rows_and_receipts_complete",
            **summaries,
        }
    )
    artifact["csl_shard_a_complete_score"] = _completion_score(
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
        if artifact.get("csl_shard_a_complete_score") != 1.0:
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
