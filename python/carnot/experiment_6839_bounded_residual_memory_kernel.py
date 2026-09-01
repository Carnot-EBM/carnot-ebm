"""Run a bounded deterministic residual-memory kernel canary.

Spec refs: REQ-CL-6839, SCENARIO-CL-6839-PRECONDITIONS,
SCENARIO-CL-6839-CHRONOLOGY, SCENARIO-CL-6839-PROPOSAL-IDENTITY,
SCENARIO-CL-6839-CREDIT, SCENARIO-CL-6839-BOUNDS,
SCENARIO-CL-6839-STALE, SCENARIO-CL-6839-RECOVERY,
SCENARIO-CL-6839-DUPLICATES, and SCENARIO-CL-6839-VERDICT.

The canary proves a state machine, not a learning result. It freezes each
decision before the later outcome is read, then uses that external outcome only
to score credit and decide whether a bounded memory update may commit.
"""

from __future__ import annotations

import argparse
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
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6839_bounded_residual_memory_kernel.py")
SCRIPT_RELATIVE_PATH = Path("scripts/experiments/experiment_6839_bounded_residual_memory_kernel.py")
RESULT_RELATIVE_PATH = Path("results/experiment_6839_bounded_residual_memory_kernel.json")

RUN_DATE = "20260901"
EXPERIMENT_ID = "6839"
SCHEMA = "carnot.experiment_6839.bounded_residual_memory_kernel.v1"
KERNEL_STATE_SCHEMA = "carnot.exp6839.residual_memory_kernel_state.v1"
CHECKPOINT_SCHEMA = "carnot.exp6839.residual_memory_kernel_checkpoint.v1"
INFERENCE_SUBSTRATE = "CPU prospective Tier-2 constraint-memory controller, no LLM"
COMPLETE_STATUS = "complete_bounded_residual_memory_kernel"
BLOCKED_STATUS = "complete_blocked_residual_memory_kernel"

EXP6827_HASH = "sha256:28464b7e0bbbab79d9d7db0bdb8a0ed498206c11a91d2ff8358756a3a5fd5b59"
EXP6496_HASH = "sha256:42da816586684c927811376ae2d68e5b3a8968c3d5bb9f9c17ff97fbc7e412be"
EXP6835_HASH = "sha256:d2a6cf80b19f37aa14ab7151a8e836ad8340f8cb74c2bddca38df6fed27611a2"
EXPECTED_ORDER_HASH_DIGEST = (
    "sha256:95211466e2ab1c6616afc6d5a19252bef8b0f8649ada4e6b29978887f932c7ca"
)
EXPECTED_SPLIT_HASH_DIGEST = (
    "sha256:bd5f9a4e6030464366209f68d4c1fa02bcfb84014b45c959c6c36bff36eb773c"
)
EXPECTED_SLICE_ID_DIGEST = "sha256:1b1b87b3ab6dc68c6cc0d67254bf0db945c6f1a9b4e8b4ba244252011b60f19a"

SOURCE_RELATIVE_PATHS = {
    "exp6827": Path("results/experiment_6827_chronological_causal_edge_memory_stream.json"),
    "exp6496": Path("results/experiment_6496_continuous_factor_learning.json"),
    "exp6835": Path("results/experiment_6835_v598_terminal_evidence_freeze.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6827": EXP6827_HASH,
    "exp6496": EXP6496_HASH,
    "exp6835": EXP6835_HASH,
}

NO_MEMORY_ARM = "no_memory"
READ_ONLY_ARM = "read_only_memory"
RANDOM_ARM = "random_admission"
VERIFIED_RESIDUAL_ARM = "verified_residual_admission"
ARM_NAMES = (NO_MEMORY_ARM, READ_ONLY_ARM, RANDOM_ARM, VERIFIED_RESIDUAL_ARM)
MEMORY_CAPACITY = 2
CANARY_EVENT_COUNT = 96
STALE_AFTER_EVENTS = 8
RESTART_AFTER_INDEX = 48
ROLLBACK_EVENT_INDEX = 32
INVALID_UPDATE_EVENT_INDEX = 19
DUPLICATE_EVENT_INDEX = 13
RANDOM_SEED = 6_839_001
RESIDUAL_GAIN = 0.35
RESIDUAL_DECAY = 0.2
MIN_ENTRY_DOSE = 0.25

OPEN_SPEC_IDS = (
    "REQ-CL-6839",
    "SCENARIO-CL-6839-PRECONDITIONS",
    "SCENARIO-CL-6839-CHRONOLOGY",
    "SCENARIO-CL-6839-PROPOSAL-IDENTITY",
    "SCENARIO-CL-6839-CREDIT",
    "SCENARIO-CL-6839-BOUNDS",
    "SCENARIO-CL-6839-STALE",
    "SCENARIO-CL-6839-RECOVERY",
    "SCENARIO-CL-6839-DUPLICATES",
    "SCENARIO-CL-6839-VERDICT",
)
REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6839_bounded_residual_memory_kernel.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_6839_bounded_residual_memory_kernel.py' -m pytest tests/python/test_experiment_6839_bounded_residual_memory_kernel.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing",
    ".venv/bin/ruff check python/carnot/experiment_6839_bounded_residual_memory_kernel.py scripts/experiments/experiment_6839_bounded_residual_memory_kernel.py tests/python/test_experiment_6839_bounded_residual_memory_kernel.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6839_bounded_residual_memory_kernel.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6839_bounded_residual_memory_kernel.json",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6839_bounded_residual_memory_kernel.json",
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
    "random_seed",
    "reproducibility_checksum",
    "kernel_state_schema",
    "rows",
    "exact_outcome_credit_rows",
    "admission_rows",
    "capacity_and_eviction_receipts",
    "stale_decay_receipts",
    "restart_receipt",
    "rollback_receipt",
    "clean_replay_state_hash",
    "residual_memory_kernel_ready_score",
    "csl_kernel_execution_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
OUTCOME_FIELD_DENYLIST = (
    "outcome_identity",
    "exact_outcome_hash",
    "signed_direction",
    "admitted",
    "admission_decision",
    "future_outcome",
    "final_acceptance",
)
EVENT_FEATURE_KEYS = (
    "row_id",
    "event_id",
    "source_family",
    "order_id",
    "chronological_position",
    "decision_snapshot_sha256",
    "source_proposal_sha256",
    "causal_edge_id",
    "write_operation_id",
    "read_operation_id",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents silent reinterpretation of canary rows.",
    "experiment_id": "A stable identifier binds the artifact to Exp6839.",
    "title": "The title states that this is a bounded kernel canary.",
    "run_date": "The fixed date separates this run from later full comparisons.",
    "status": "The status separates complete canary proof from blocked input gates.",
    "openspec_requirement_ids": "Requirement IDs keep tests and artifact fields traceable.",
    "replay_commands": "Replay commands show how the deterministic checks were run.",
    "field_principles": "Each field states why a reader needs it.",
    "preconditions_checked": "Input gates stop source drift before state mutation.",
    "inference_substrate": "The substrate declares deterministic CPU execution and no LLM.",
    "duration_s": "Measured wall time shows that the canary executed.",
    "continuous_self_learning_task": "True marks this as an FR11 external-memory task.",
    "source_artifact_hashes": "Raw source hashes bind the frozen evidence bytes.",
    "random_seed": "The seed fixes the random-admission control deterministically.",
    "reproducibility_checksum": "The checksum binds stable content and excludes wall time.",
    "kernel_state_schema": "The schema names the exact state bytes being replayed.",
    "rows": "One row records one event and one arm transition.",
    "exact_outcome_credit_rows": "Credit rows expose signed action-level credit.",
    "admission_rows": "Admission rows show every post-outcome write decision.",
    "capacity_and_eviction_receipts": "Receipts prove memory stays within capacity.",
    "stale_decay_receipts": "Receipts prove stale entries leave active state.",
    "restart_receipt": "Restart evidence proves persisted bytes recover exactly.",
    "rollback_receipt": "Rollback evidence proves parent bytes restore exactly.",
    "clean_replay_state_hash": "A clean replay hash detects lifecycle drift.",
    "residual_memory_kernel_ready_score": "Readiness scores the state machine only.",
    "csl_kernel_execution_complete_score": "Completion scores row and recovery coverage.",
    "gate_check_summary": "Failed checks keep expected and observed values.",
    "verifier_is_oracle": "False because later outcomes are external receipts.",
    "verdict_class": "A closed class prevents unsupported positive wording.",
    "honest_verdict": "The terminal prefix states no held-future benefit is claimed.",
}


@dataclass
class KernelRun:
    """Collected state-machine receipts for artifact assembly."""

    rows: list[JsonDict]
    exact_outcome_credit_rows: list[JsonDict]
    admission_rows: list[JsonDict]
    capacity_and_eviction_receipts: list[JsonDict]
    stale_decay_receipts: list[JsonDict]
    restart_receipt: JsonDict
    rollback_receipt: JsonDict
    duplicate_receipt: JsonDict
    invalid_update_receipt: JsonDict
    final_state_hash: str


@dataclass
class KernelState:
    """Small arm-local memory whose bytes define restart and rollback identity."""

    arm: str
    capacity: int
    records: list[JsonDict] = field(default_factory=list)
    processed_event_ids: set[str] = field(default_factory=set)
    residual_pressure: float = 0.0
    generation: int = 0
    predecessor_hash: str = "bootstrap"

    @classmethod
    def for_arm(cls, arm: str) -> "KernelState":
        """Create one isolated arm with only fixed public bootstrap state."""

        capacity = 0 if arm == NO_MEMORY_ARM else MEMORY_CAPACITY
        state = cls(arm=arm, capacity=capacity)
        if arm == READ_ONLY_ARM:
            state.records.append(
                {
                    "key": "bootstrap:read_only",
                    "event_row_id": "bootstrap",
                    "source_family": "control",
                    "signed_direction": 1,
                    "stored_dose": MIN_ENTRY_DOSE,
                    "created_at": -10_000,
                    "immutable": True,
                    "source_outcome_identity": "bootstrap",
                }
            )
        return state

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelState":
        """Decode persisted canonical state back into a live store."""

        return cls(
            arm=str(value["arm"]),
            capacity=int(value["capacity"]),
            records=[deepcopy(record) for record in value["records"]],
            processed_event_ids=set(value["processed_event_ids"]),
            residual_pressure=float(value["residual_pressure"]),
            generation=int(value["generation"]),
            predecessor_hash=str(value["predecessor_hash"]),
        )

    def state_dict(self) -> JsonDict:
        """Sort state fields so restart bytes do not depend on Python object identity."""

        return {
            "schema": KERNEL_STATE_SCHEMA,
            "arm": self.arm,
            "capacity": self.capacity,
            "generation": self.generation,
            "predecessor_hash": self.predecessor_hash,
            "residual_pressure": round(self.residual_pressure, 6),
            "processed_event_ids": sorted(self.processed_event_ids),
            "records": sorted(
                [deepcopy(record) for record in self.records],
                key=lambda record: (int(record["created_at"]), str(record["key"])),
            ),
        }

    def state_bytes(self) -> bytes:
        """Return the canonical state representation used for hashes."""

        return canonical_json_bytes(self.state_dict())

    def state_hash(self) -> str:
        """Hash the exact state bytes."""

        return sha256_bytes(self.state_bytes())

    def active_count(self) -> int:
        """Count active records while including read-only bootstrap records."""

        return len(self.records)

    def _restore_bytes(self, state_bytes: bytes) -> None:
        restored = KernelState.from_dict(json.loads(state_bytes))
        self.records = restored.records
        self.processed_event_ids = restored.processed_event_ids
        self.residual_pressure = restored.residual_pressure
        self.generation = restored.generation
        self.predecessor_hash = restored.predecessor_hash

    def expire_stale(self, chronology_index: int) -> list[JsonDict]:
        """Remove old mutable entries before the next decision snapshot."""

        parent_hash = self.state_hash()
        kept = [
            record
            for record in self.records
            if record.get("immutable")
            or chronology_index - int(record["created_at"]) < STALE_AFTER_EVENTS
        ]
        expired = [record for record in self.records if record not in kept]
        if not expired:
            return []
        self.records = kept
        self.generation += 1
        self.predecessor_hash = parent_hash
        receipt = {
            "arm": self.arm,
            "chronology_index": chronology_index,
            "expired_keys": [record["key"] for record in expired],
            "expired_count": len(expired),
            "parent_state_sha256": parent_hash,
            "new_state_sha256": self.state_hash(),
            "active_count_after": self.active_count(),
            "capacity": self.capacity,
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return [receipt]

    def _evict_if_needed(self, event_row_id: str) -> list[JsonDict]:
        receipts: list[JsonDict] = []
        while self.active_count() > self.capacity:
            parent_hash = self.state_hash()
            candidates = [record for record in self.records if not record.get("immutable")]
            victim = min(
                candidates, key=lambda record: (int(record["created_at"]), str(record["key"]))
            )
            self.records.remove(victim)
            self.generation += 1
            self.predecessor_hash = parent_hash
            receipt = {
                "arm": self.arm,
                "event_row_id": event_row_id,
                "evicted_key": victim["key"],
                "evicted_event_row_id": victim["event_row_id"],
                "parent_state_sha256": parent_hash,
                "new_state_sha256": self.state_hash(),
                "active_count_after": self.active_count(),
                "capacity": self.capacity,
                "eviction_rule": "oldest_mutable_entry",
            }
            receipt["receipt_sha256"] = sha256_json(receipt)
            receipts.append(receipt)
        return receipts

    def _update_residual(self, signed_direction: int) -> None:
        self.residual_pressure = bounded_dose(
            self.residual_pressure * (1.0 - RESIDUAL_DECAY) + signed_direction * RESIDUAL_GAIN
        )

    def admit(
        self,
        event: Mapping[str, Any],
        proposal: Mapping[str, Any],
        outcome: Mapping[str, Any],
        chronology_index: int,
        *,
        force_invalid: bool = False,
        force_rollback: bool = False,
    ) -> tuple[JsonDict, list[JsonDict], JsonDict]:
        """Evaluate one post-outcome admission and mutate only on accepted writes."""

        parent_bytes = self.state_bytes()
        parent_hash = sha256_bytes(parent_bytes)
        event_row_id = str(event["row_id"])
        signed_direction = int(outcome["signed_direction"])
        admitted = False
        reason = "preconditions_passed"
        rollback_receipt: JsonDict = {}

        if event_row_id in self.processed_event_ids:
            reason = "duplicate_event"
        elif force_invalid:
            reason = "invalid_parent_hash"
        elif abs(float(proposal["action_dose"])) > 1.0:
            reason = "dose_out_of_bounds"
        elif self.arm == NO_MEMORY_ARM:
            reason = "no_memory_arm"
        elif self.arm == READ_ONLY_ARM:
            reason = "read_only_arm"
        elif signed_direction <= 0:
            reason = "exact_nonpositive_direction"
        elif self.arm == RANDOM_ARM and not random_admission_accepts(event, RANDOM_SEED):
            reason = "seeded_random_reject"
        else:
            admitted = True

        if reason != "duplicate_event" and not force_invalid:
            self.processed_event_ids.add(event_row_id)
            self.generation += 1
            self.predecessor_hash = parent_hash
            if self.arm == VERIFIED_RESIDUAL_ARM:
                self._update_residual(signed_direction)

        evictions: list[JsonDict] = []
        if admitted:
            stored_dose = bounded_dose(max(abs(float(proposal["action_dose"])), MIN_ENTRY_DOSE))
            self.records.append(
                {
                    "key": "mem:" + sha256_json(event_row_id)[7:19],
                    "event_row_id": event_row_id,
                    "source_family": event["source_family"],
                    "signed_direction": signed_direction,
                    "stored_dose": stored_dose,
                    "created_at": chronology_index,
                    "immutable": False,
                    "source_outcome_identity": outcome["outcome_identity"],
                }
            )
            evictions = self._evict_if_needed(event_row_id)

        if admitted and force_rollback:
            committed_hash = self.state_hash()
            self._restore_bytes(parent_bytes)
            rollback_receipt = {
                "arm": self.arm,
                "event_row_id": event_row_id,
                "rolled_back": True,
                "committed_state_sha256": committed_hash,
                "rollback_target_sha256": parent_hash,
                "restored_state_sha256": self.state_hash(),
                "restored_parent_bytes": self.state_bytes() == parent_bytes,
            }
            rollback_receipt["receipt_sha256"] = sha256_json(rollback_receipt)

        receipt = {
            "arm": self.arm,
            "event_row_id": event_row_id,
            "chronology_index": chronology_index,
            "proposal_hash": proposal["proposal_hash"],
            "exact_outcome_hash": outcome["exact_outcome_hash"],
            "signed_direction": signed_direction,
            "action_dose": proposal["action_dose"],
            "admitted": admitted,
            "reason": "admitted" if admitted else reason,
            "parent_state_sha256": parent_hash,
            "new_state_sha256": self.state_hash(),
            "memory_records_changed": admitted and not force_rollback,
            "active_count_after": self.active_count(),
            "capacity": self.capacity,
            "rolled_back": bool(rollback_receipt),
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return receipt, evictions, rollback_receipt


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with one canonical representation and one trailing newline."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 form."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash only after canonical JSON serialization."""

    return sha256_bytes(canonical_json_bytes(value))


def bounded_dose(value: float) -> float:
    """Clamp a route dose so one event cannot overrun bounded memory pressure."""

    return round(max(-1.0, min(1.0, float(value))), 6)


def source_paths_for_root(root: Path) -> dict[str, Path]:
    """Resolve all frozen source paths relative to a checkout."""

    return {name: root / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Load source JSON while preserving readable failure markers."""

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


def select_canary_events(source: Mapping[str, Any]) -> list[JsonDict]:
    """Choose the fixed 96 applicable causal-edge rows used by the canary."""

    rows = source.get("rows", [])
    if not isinstance(rows, list):
        return []
    applicable = [
        deepcopy(row)
        for row in rows
        if isinstance(row, dict)
        and row.get("counterfactual_kind") == "remove"
        and row.get("counterfactual_applicable") is True
    ]
    return sorted(
        applicable,
        key=lambda row: (
            str(row.get("order_id", "")),
            str(row.get("memory_scope", "")),
            int(row.get("chronological_position", 0)),
            str(row.get("row_id", "")),
        ),
    )[:CANARY_EVENT_COUNT]


def _stable_order_digest(source: Mapping[str, Any]) -> str:
    return sha256_json(source.get("order_hashes", {}))


def _stable_split_digest(source: Mapping[str, Any]) -> str:
    return sha256_json(source.get("split_manifest", {}))


def _slice_digest(events: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([event.get("row_id") for event in events])


def check_preconditions(
    sources: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path]
) -> JsonDict:
    """Check every upstream gate before constructing state-machine rows."""

    exp6827 = sources.get("exp6827", {})
    exp6835 = sources.get("exp6835", {})
    source_hashes = _source_hashes(paths)
    expected_hashes = {
        name: EXPECTED_SOURCE_HASHES[name] for name in sorted(EXPECTED_SOURCE_HASHES)
    }
    observed_hashes = {name: row["sha256"] for name, row in source_hashes.items()}
    source_hashes_pass = observed_hashes == expected_hashes
    rows = exp6827.get("rows", [])
    row_count = len(rows) if isinstance(rows, list) else 0
    events = select_canary_events(exp6827)
    exact_outcomes = [
        event
        for event in events
        if event.get("outcome_identity")
        and event.get("action_identity")
        and event.get("causal_edge_id")
        and event.get("write_operation_id")
        and event.get("read_operation_id")
    ]
    headroom = exp6827.get("headroom_metrics", {})
    headroom_values = {
        "later_read_opportunity_count": headroom.get("later_read_opportunity_count", 0),
        "capacity_pressure_case_count": headroom.get("capacity_pressure_case_count", 0),
        "stale_pressure_recovery_count": headroom.get("stale_pressure_recovery_count", 0),
    }
    checks = [
        _check("source_artifact_hashes", expected_hashes, observed_hashes, source_hashes_pass),
        _check(
            "v598_evidence_root_ready_score",
            1,
            exp6835.get("v598_evidence_root_ready_score"),
            exp6835.get("v598_evidence_root_ready_score") == 1,
        ),
        _check("complete_exp6827_stream", 4320, row_count, row_count == 4320),
        _check(
            "stable_order_hashes",
            EXPECTED_ORDER_HASH_DIGEST,
            _stable_order_digest(exp6827),
            _stable_order_digest(exp6827) == EXPECTED_ORDER_HASH_DIGEST,
        ),
        _check(
            "stable_split_hash",
            EXPECTED_SPLIT_HASH_DIGEST,
            _stable_split_digest(exp6827),
            _stable_split_digest(exp6827) == EXPECTED_SPLIT_HASH_DIGEST,
        ),
        _check(
            "exact_later_outcomes",
            {
                "canary_event_count": CANARY_EVENT_COUNT,
                "slice_id_digest": EXPECTED_SLICE_ID_DIGEST,
            },
            {
                "canary_event_count": len(events),
                "exact_outcome_count": len(exact_outcomes),
                "slice_id_digest": _slice_digest(events),
            },
            len(exact_outcomes) == CANARY_EVENT_COUNT
            and _slice_digest(events) == EXPECTED_SLICE_ID_DIGEST,
        ),
        _check(
            "nonzero_decision_headroom",
            "all selected headroom values > 0",
            headroom_values,
            all(int(value) > 0 for value in headroom_values.values()),
        ),
    ]
    failed = [row["check"] for row in checks if row["passed"] is not True]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def _public_event(event: Mapping[str, Any]) -> JsonDict:
    return {key: event[key] for key in EVENT_FEATURE_KEYS}


def _memory_signal(state: KernelState) -> float:
    if not state.records:
        return 0.0
    total = sum(
        int(record["signed_direction"]) * float(record["stored_dose"]) for record in state.records
    )
    return bounded_dose(total / max(1, len(state.records)))


def _seeded_dose(event: Mapping[str, Any], state_hash: str, random_seed: int) -> float:
    digest = sha256_json(
        {
            "seed": random_seed,
            "row_id": event["row_id"],
            "state_hash": state_hash,
            "arm": RANDOM_ARM,
        }
    )
    bucket = int(digest[7:15], 16) % 5
    return (-0.5, -0.25, 0.0, 0.25, 0.5)[bucket]


def _action_from_dose(dose: float) -> str:
    if dose > 0.25:
        return "route_positive"
    if dose < -0.25:
        return "route_negative"
    return "route_neutral"


def propose_action(
    event: Mapping[str, Any],
    state: KernelState,
    *,
    random_seed: int,
) -> JsonDict:
    """Freeze the arm action without reading the later outcome fields."""

    parent_hash = state.state_hash()
    if state.arm == NO_MEMORY_ARM:
        dose = 0.0
    elif state.arm == RANDOM_ARM:
        dose = _seeded_dose(event, parent_hash, random_seed)
    elif state.arm == VERIFIED_RESIDUAL_ARM:
        dose = bounded_dose(state.residual_pressure + _memory_signal(state) * 0.5)
    else:
        dose = _memory_signal(state)
    material = {
        "arm": state.arm,
        "random_seed": random_seed,
        "parent_state_sha256": parent_hash,
        "decision_material": _public_event(event),
        "action_dose": dose,
        "proposed_action": _action_from_dose(dose),
    }
    proposal_hash = sha256_json(material)
    return {
        **material,
        "proposal_hash": proposal_hash,
        "decision_input_keys": sorted(material["decision_material"]),
        "decision_frozen_before_outcome_reveal": True,
    }


def exact_outcome_for_event(event: Mapping[str, Any]) -> JsonDict:
    """Reveal the external later outcome identity after proposal freeze."""

    material = {
        "event_row_id": event["row_id"],
        "outcome_identity": event["outcome_identity"],
        "action_identity": event["action_identity"],
        "causal_edge_id": event["causal_edge_id"],
    }
    direction_digest = sha256_json(material)
    signed_direction = (-1, 0, 1)[int(direction_digest[7:15], 16) % 3]
    outcome = {**material, "signed_direction": signed_direction, "external_later_outcome": True}
    outcome["exact_outcome_hash"] = sha256_json(outcome)
    return outcome


def credit_for_transition(
    event: Mapping[str, Any],
    arm: str,
    proposal: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> JsonDict:
    """Assign signed credit from the frozen dose and exact later direction."""

    dose = float(proposal["action_dose"])
    direction = int(outcome["signed_direction"])
    signed_credit = bounded_dose(direction * dose)
    row = {
        "credit_id": sha256_json(
            {"arm": arm, "event_row_id": event["row_id"], "proposal": proposal["proposal_hash"]}
        ),
        "arm": arm,
        "event_row_id": event["row_id"],
        "proposal_hash": proposal["proposal_hash"],
        "exact_outcome_hash": outcome["exact_outcome_hash"],
        "signed_direction": direction,
        "action_dose": dose,
        "signed_credit": signed_credit,
        "credited": signed_credit != 0.0,
        "credit_source": "external_later_outcome_direction_times_pre_reveal_dose",
    }
    row["row_sha256"] = sha256_json(row)
    return row


def random_admission_accepts(event: Mapping[str, Any], random_seed: int) -> bool:
    """Return the deterministic placebo-admission decision for one event."""

    digest = sha256_json({"seed": random_seed, "row_id": event["row_id"], "control": RANDOM_ARM})
    return int(digest[7:15], 16) % 2 == 0


def _transition_row(
    event: Mapping[str, Any],
    arm: str,
    proposal: Mapping[str, Any],
    outcome: Mapping[str, Any],
    admission: Mapping[str, Any],
    state_hash_before_decision: str,
) -> JsonDict:
    row = {
        "row_id": f"{event['row_id']}::{arm}",
        "event": _public_event(event),
        "arm": arm,
        "state_hash_before_decision": state_hash_before_decision,
        "decision_hash": proposal["proposal_hash"],
        "decision_frozen_before_outcome_reveal": True,
        "outcome_revealed_after_decision": True,
        "proposed_action": proposal["proposed_action"],
        "exact_outcome": {
            "outcome_identity": outcome["outcome_identity"],
            "exact_outcome_hash": outcome["exact_outcome_hash"],
        },
        "signed_direction": outcome["signed_direction"],
        "action_dose": proposal["action_dose"],
        "admission_decision": {
            "admitted": admission["admitted"],
            "reason": admission["reason"],
            "receipt_sha256": admission["receipt_sha256"],
        },
        "state_hash": admission["new_state_sha256"],
    }
    row["row_sha256"] = sha256_json(row)
    return row


def _initial_states() -> dict[str, KernelState]:
    return {arm: KernelState.for_arm(arm) for arm in ARM_NAMES}


def checkpoint_payload(states: Mapping[str, KernelState]) -> JsonDict:
    """Build the checkpoint JSON written atomically during persistence."""

    return {
        "schema": CHECKPOINT_SCHEMA,
        "kernel_state_schema": KERNEL_STATE_SCHEMA,
        "states": {arm: states[arm].state_dict() for arm in ARM_NAMES},
    }


def persist_states(states: Mapping[str, KernelState], path: Path) -> str:
    """Write canonical checkpoint bytes through a replace step."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(states)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(canonical_json_bytes(payload))
    os.replace(tmp, path)
    return sha256_bytes(path.read_bytes())


def load_persisted_states(path: Path) -> dict[str, KernelState]:
    """Load the last complete checkpoint and ignore any partial temp file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {arm: KernelState.from_dict(payload["states"][arm]) for arm in ARM_NAMES}


def combined_state_hash(states: Mapping[str, KernelState]) -> str:
    """Hash all arm states into one replay-comparable identity."""

    return sha256_json({arm: states[arm].state_hash() for arm in ARM_NAMES})


def _process_one(
    event: Mapping[str, Any],
    state: KernelState,
    chronology_index: int,
    *,
    force_invalid: bool = False,
    force_rollback: bool = False,
) -> tuple[JsonDict, JsonDict, JsonDict, list[JsonDict], JsonDict]:
    expired = state.expire_stale(chronology_index)
    state_hash_before_decision = state.state_hash()
    proposal = propose_action(event, state, random_seed=RANDOM_SEED)
    outcome = exact_outcome_for_event(event)
    credit = credit_for_transition(event, state.arm, proposal, outcome)
    admission, evictions, rollback = state.admit(
        event,
        proposal,
        outcome,
        chronology_index,
        force_invalid=force_invalid,
        force_rollback=force_rollback,
    )
    row = _transition_row(
        event, state.arm, proposal, outcome, admission, state_hash_before_decision
    )
    return row, credit, admission, [*expired, *evictions], rollback


def run_state_machine(
    events: Sequence[Mapping[str, Any]],
    state_root: Path,
    *,
    exercise_recovery: bool,
) -> KernelRun:
    """Run every arm over the fixed canary slice and collect lifecycle receipts."""

    states = _initial_states()
    rows: list[JsonDict] = []
    credit_rows: list[JsonDict] = []
    admissions: list[JsonDict] = []
    evictions: list[JsonDict] = []
    stale_receipts: list[JsonDict] = []
    rollback_receipt: JsonDict = {"rolled_back": False, "restored_parent_bytes": False}
    duplicate_receipt: JsonDict = {}
    invalid_receipt: JsonDict = {}
    checkpoint = state_root / "checkpoint.json"

    for index, event in enumerate(events, start=1):
        for arm in ARM_NAMES:
            force_invalid = arm == VERIFIED_RESIDUAL_ARM and index == INVALID_UPDATE_EVENT_INDEX
            force_rollback = arm == VERIFIED_RESIDUAL_ARM and index == ROLLBACK_EVENT_INDEX
            row, credit, admission, lifecycle, rollback = _process_one(
                event,
                states[arm],
                index,
                force_invalid=force_invalid,
                force_rollback=force_rollback,
            )
            rows.append(row)
            credit_rows.append(credit)
            admissions.append(admission)
            stale_receipts.extend(item for item in lifecycle if "expired_keys" in item)
            evictions.extend(item for item in lifecycle if "evicted_key" in item)
            if rollback:
                rollback_receipt = rollback
            if force_invalid:
                invalid_receipt = admission
            if exercise_recovery and arm == RANDOM_ARM and index == DUPLICATE_EVENT_INDEX:
                duplicate_receipt, _, _ = states[arm].admit(
                    event,
                    propose_action(event, states[arm], random_seed=RANDOM_SEED),
                    exact_outcome_for_event(event),
                    index,
                )
        if exercise_recovery:
            persist_states(states, checkpoint)
            if index == RESTART_AFTER_INDEX:
                checkpoint.with_name(checkpoint.name + ".tmp").write_text(
                    "{partial", encoding="utf-8"
                )
                before_restart = combined_state_hash(states)
                loaded = load_persisted_states(checkpoint)
                restart_receipt = {
                    "boundary_index": index,
                    "persisted_state_hash": before_restart,
                    "restarted_state_hash": combined_state_hash(loaded),
                    "bytes_identity": before_restart == combined_state_hash(loaded),
                    "crash_recovery_ignored_partial_checkpoint": checkpoint.with_name(
                        checkpoint.name + ".tmp"
                    ).exists(),
                    "checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()),
                }
                states = loaded

    final_hash = combined_state_hash(states)
    if exercise_recovery:
        persist_states(states, checkpoint)
        restart_receipt["final_state_hash"] = final_hash
    else:
        restart_receipt = {
            "boundary_index": None,
            "bytes_identity": True,
            "crash_recovery_ignored_partial_checkpoint": False,
            "final_state_hash": final_hash,
        }

    return KernelRun(
        rows=rows,
        exact_outcome_credit_rows=credit_rows,
        admission_rows=admissions,
        capacity_and_eviction_receipts=evictions,
        stale_decay_receipts=stale_receipts,
        restart_receipt=restart_receipt,
        rollback_receipt=rollback_receipt,
        duplicate_receipt=duplicate_receipt,
        invalid_update_receipt=invalid_receipt,
        final_state_hash=final_hash,
    )


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
        "title": "Bounded Residual-Memory Kernel Canary",
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
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "kernel_state_schema": {
            "schema": KERNEL_STATE_SCHEMA,
            "arms": list(ARM_NAMES),
            "memory_capacity": MEMORY_CAPACITY,
            "canary_event_count": CANARY_EVENT_COUNT,
            "stale_after_events": STALE_AFTER_EVENTS,
            "decision_dose_bounds": [-1.0, 1.0],
            "substrate_detail": "deterministic CPU state-machine canary",
        },
        "rows": [],
        "exact_outcome_credit_rows": [],
        "admission_rows": [],
        "capacity_and_eviction_receipts": [],
        "stale_decay_receipts": [],
        "restart_receipt": {},
        "rollback_receipt": {},
        "clean_replay_state_hash": "",
        "residual_memory_kernel_ready_score": 0.0,
        "csl_kernel_execution_complete_score": 0.0,
        "gate_check_summary": deepcopy(dict(preconditions)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": f"{BLOCKED_STATUS}: one or more frozen input gates failed",
    }


def _completion_ready(run: KernelRun, clean_hash: str) -> bool:
    return (
        len(run.rows) == CANARY_EVENT_COUNT * len(ARM_NAMES)
        and len(run.exact_outcome_credit_rows) == len(run.rows)
        and bool(run.capacity_and_eviction_receipts)
        and bool(run.stale_decay_receipts)
        and run.restart_receipt.get("bytes_identity") is True
        and run.restart_receipt.get("crash_recovery_ignored_partial_checkpoint") is True
        and run.rollback_receipt.get("restored_parent_bytes") is True
        and run.duplicate_receipt.get("reason") == "duplicate_event"
        and run.invalid_update_receipt.get("reason") == "invalid_parent_hash"
        and run.final_state_hash == clean_hash
        and all(-1.0 <= float(row["action_dose"]) <= 1.0 for row in run.rows)
    )


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path],
    state_root: Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
) -> JsonDict:
    """Build the terminal artifact or a complete blocked shape."""

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

    events = select_canary_events(sources["exp6827"])
    if state_root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6839-") as tmp:
            run = run_state_machine(events, Path(tmp) / "live", exercise_recovery=True)
            clean = run_state_machine(events, Path(tmp) / "clean", exercise_recovery=False)
    else:
        run = run_state_machine(events, state_root, exercise_recovery=True)
        clean = run_state_machine(events, state_root / "clean", exercise_recovery=False)
    ready = _completion_ready(run, clean.final_state_hash)
    run.restart_receipt["clean_replay_state_hash"] = clean.final_state_hash
    run.restart_receipt["matches_clean_replay"] = run.final_state_hash == clean.final_state_hash
    artifact.update(
        {
            "status": COMPLETE_STATUS,
            "rows": run.rows,
            "exact_outcome_credit_rows": run.exact_outcome_credit_rows,
            "admission_rows": run.admission_rows,
            "capacity_and_eviction_receipts": run.capacity_and_eviction_receipts,
            "stale_decay_receipts": run.stale_decay_receipts,
            "restart_receipt": run.restart_receipt,
            "rollback_receipt": run.rollback_receipt,
            "clean_replay_state_hash": clean.final_state_hash,
            "residual_memory_kernel_ready_score": 1.0 if ready else 0.0,
            "csl_kernel_execution_complete_score": 1.0 if ready else 0.0,
            "verdict_class": "null",
            "honest_verdict": (
                "complete_null_bounded_residual_memory_kernel_state_machine_ready_"
                "no_future_benefit_claim"
            ),
        }
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
    """Return schema and row-support problems without mutating the artifact."""

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
    if artifact.get("verdict_class") == "positive":
        errors.append("canary must not claim held-future benefit")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("status") == COMPLETE_STATUS:
        if len(artifact.get("rows", [])) != CANARY_EVENT_COUNT * len(ARM_NAMES):
            errors.append("complete row count mismatch")
        if len(artifact.get("exact_outcome_credit_rows", [])) != len(artifact.get("rows", [])):
            errors.append("credit row count mismatch")
        if artifact.get("residual_memory_kernel_ready_score") != 1.0:
            errors.append("complete artifact missing ready score")
        if artifact.get("csl_kernel_execution_complete_score") != 1.0:
            errors.append("complete artifact missing execution score")
    if artifact.get("status") == BLOCKED_STATUS and artifact.get("rows"):
        errors.append("blocked artifact must not expose transition rows")
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
