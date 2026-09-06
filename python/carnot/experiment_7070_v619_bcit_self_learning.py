"""Compare context-bound learning with three chronological controls.

The runner separates public event rows from exact outcomes. It seals each
decision before it reads the matching outcome. This module can exercise the
full protocol with test fixtures, but the repository run fails closed when the
frozen upstream stream does not meet its declared sample-size gates.

Spec refs: REQ-SELFLEARN-7070 and SCENARIO-SELFLEARN-7070-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot.learning.constraint_policy_store import ForcedInterruption, PolicyStore
from carnot.learning.context_authorization import (
    Authorization,
    AuthorizationDecision,
    ContextAuthorizationMachine,
    ContextBoundExperience,
    DecisionView,
    NumericInterval,
    RetentionResult,
    SupportInterval,
    canonical_bytes,
    sha256_json,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260906"
RANDOM_SEED = 707020260906
EXPERIMENT_ID = "experiment_7070_v619_bcit_self_learning"
SCHEMA = "carnot.experiment_7070.v619_bcit_self_learning.v1"
INFERENCE_SUBSTRATE = "deterministic_cpu_chronological_comparison"
ARMS = ("context_bound", "flat_reuse", "validate_all", "no_reuse")
MIN_EVENT_COUNT = 120
MIN_SOURCE_GROUP_COUNT = 12
NO_PREFERENCE_BOUND = 0.05
VALIDATION_COST = 1.0
DECISION_BUDGET = 1
VALIDATION_BUDGET = 1
DEFAULT_CAPACITY_BYTES = 4_096
PROTECTED_RETENTION_BOUND = 1.0
POLICY_HASH = "sha256:" + "a" * 64
CONSTRAINT_SCHEMA_HASH = "sha256:" + "b" * 64

SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_7070_v619_bcit_self_learning.json")
WORK_RELATIVE_PATH = Path("results/raw/experiment_7070_v619_bcit_self_learning")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_7069_v619_context_authorization_contract.json")
EXP6978_RELATIVE_PATH = Path("results/experiment_6978_transactional_constraint_self_learning.json")
EXP7021_RELATIVE_PATH = Path("results/experiment_7021_prospective_belief_utility.json")
AUTHORIZATION_MODULE_RELATIVE_PATH = Path("python/carnot/learning/context_authorization.py")
UPSTREAM_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_7069_v619_context_authorization_contract.py"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7070_v619_bcit_self_learning.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_7070_v619_bcit_self_learning.py")
WRAPPER_RELATIVE_PATH = Path("scripts/experiments/experiment_7070_v619_bcit_self_learning.py")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "upstream_gate_rows",
    "rows",
    "per_game_results",
    "arm_definitions",
    "chronological_event_rows",
    "decision_snapshot_rows",
    "authorization_rows",
    "validation_rows",
    "exact_outcome_rows",
    "journal_rows",
    "commit_rows",
    "rollback_rows",
    "capacity_rows",
    "retention_rows",
    "per_source_group_rows",
    "harmful_update_rate_by_arm",
    "useful_update_rate_by_arm",
    "validation_cost_by_arm",
    "final_exact_quality_by_arm",
    "equal_budget_quality_by_arm",
    "abstention_rate_by_arm",
    "paired_interval_rows",
    "bcit_comparison_complete_score",
    "bcit_self_learning_value_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A principle for each required field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact checks stop comparison work when frozen evidence or storage is insufficient.",
    "inference_substrate": "The named CPU substrate separates deterministic comparison from model judgment.",
    "duration_s": "Measured wall time shows that the command reached a terminal result.",
    "source_artifact_hashes": "Content hashes bind imported evidence and code to exact bytes.",
    "cited_upstream_artifacts": "Citations identify each earlier result and the fields reused here.",
    "upstream_gate_rows": "Copied gate rows expose which upstream claims authorize this run.",
    "rows": "Arm summaries provide a compact view without replacing event evidence.",
    "per_game_results": "Game-level reductions show whether one game drives the result.",
    "arm_definitions": "Frozen arm rules and budgets prevent changes after outcomes open.",
    "chronological_event_rows": "Matched event rows own all quality and update aggregates.",
    "decision_snapshot_rows": "Read receipts prove that each decision used only prior committed evidence.",
    "authorization_rows": "Authorization receipts expose each use, validate, or reject choice.",
    "validation_rows": "Validation rows account for every bounded validation call and cost.",
    "exact_outcome_rows": "Outcome receipts prove that exact evidence opened after decision sealing.",
    "journal_rows": "Per-event journal deltas expose commit, rollback, and no-op behavior.",
    "commit_rows": "Commit receipts bind admitted updates to parent and new state hashes.",
    "rollback_rows": "Rollback receipts prove that failed or harmful writes restore parent bytes.",
    "capacity_rows": "Capacity rows expose deterministic eviction and protected record retention.",
    "retention_rows": "Protected checks measure whether learning preserves the initial policy.",
    "per_source_group_rows": "Source-group rows expose transfer, drift, and pooled-result imbalance.",
    "harmful_update_rate_by_arm": "Harmful reuse rate tests whether context binding prevents bad transfer.",
    "useful_update_rate_by_arm": "Useful reuse rate measures whether earlier evidence helps later events.",
    "validation_cost_by_arm": "Exact consumed cost prevents fewer checks from posing as quality.",
    "final_exact_quality_by_arm": "Final quality measures the chronological decisions that each arm sealed.",
    "equal_budget_quality_by_arm": "Equal-budget quality compares matched decisions under assigned budgets.",
    "abstention_rate_by_arm": "Abstention exposes safety achieved by declining unsupported reuse.",
    "paired_interval_rows": "Matched-event intervals quantify uncertainty without inflating sample size.",
    "bcit_comparison_complete_score": "One requires four complete, isolated, chronological, recomputable arms.",
    "bcit_self_learning_value_score": "One requires safer transfer, positive paired quality, and full retention.",
    "random_seed": "A fixed seed makes matched interval resampling reproducible.",
    "reproducibility_checksum": "A timing-free digest detects changes to deterministic terminal evidence.",
    "gate_check_summary": "Exact expected and observed values make a terminal failure actionable.",
    "verifier_is_oracle": "False states that the policy cannot read the exact outcome before sealing.",
    "verdict_class": "A closed class separates complete nulls from positive and blocked results.",
    "honest_verdict": "A class-consistent prefix gives downstream automation an exact terminal state.",
}

AGGREGATE_FIELDS = (
    "harmful_update_rate_by_arm",
    "useful_update_rate_by_arm",
    "validation_cost_by_arm",
    "final_exact_quality_by_arm",
    "equal_budget_quality_by_arm",
    "abstention_rate_by_arm",
)
ROW_FIELDS = (
    "rows",
    "per_game_results",
    "arm_definitions",
    "chronological_event_rows",
    "decision_snapshot_rows",
    "authorization_rows",
    "validation_rows",
    "exact_outcome_rows",
    "journal_rows",
    "commit_rows",
    "rollback_rows",
    "capacity_rows",
    "retention_rows",
    "per_source_group_rows",
    "paired_interval_rows",
)
FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "current_outcome",
        "exact_outcome",
        "future_event",
        "future_events",
        "future_outcome",
        "held_group_label",
        "model_text",
        "post_event_aggregate",
        "self_score",
    }
)


def _forbidden_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    """Find forbidden keys at any depth so wrappers cannot hide outcome data."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in FORBIDDEN_DECISION_FIELDS:
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return tuple(found)


@dataclass(frozen=True)
class DecisionSnapshot:
    """Freeze the complete outcome-free read view used for one decision."""

    event_id: str
    event_time: int
    source_group: str
    related_source_groups: tuple[str, ...]
    active_constraints: tuple[str, ...]
    read_state_hash: str
    evidence_event_times: tuple[int, ...]
    max_evidence_event_time: int

    @classmethod
    def from_event(
        cls,
        event: Mapping[str, Any],
        *,
        evidence_event_times: Sequence[int],
        read_state_hash: str = POLICY_HASH,
    ) -> DecisionSnapshot:
        """Build a snapshot only after rejecting hidden current or future data."""

        forbidden = _forbidden_paths(event)
        if forbidden:
            raise ValueError(f"forbidden decision fields: {list(forbidden)}")
        times = tuple(sorted(int(value) for value in evidence_event_times))
        event_time = int(event["event_time"])
        if any(value >= event_time for value in times):
            raise ValueError("decision evidence must strictly precede the event")
        return cls(
            event_id=str(event["event_id"]),
            event_time=event_time,
            source_group=str(event["source_group"]),
            related_source_groups=tuple(str(value) for value in event["related_source_groups"]),
            active_constraints=tuple(str(value) for value in event["active_constraints"]),
            read_state_hash=str(read_state_hash),
            evidence_event_times=times,
            max_evidence_event_time=max(times, default=-1),
        )

    def to_dict(self) -> JsonDict:
        """Serialize only the fields that existed before the outcome opened."""

        return {
            "event_id": self.event_id,
            "event_time": self.event_time,
            "source_group": self.source_group,
            "related_source_groups": list(self.related_source_groups),
            "active_constraints": list(self.active_constraints),
            "read_state_hash": self.read_state_hash,
            "evidence_event_times": list(self.evidence_event_times),
            "max_evidence_event_time": self.max_evidence_event_time,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DecisionSnapshot:
        """Parse the closed snapshot schema and reject added outcome channels."""

        forbidden = _forbidden_paths(payload)
        if forbidden:
            raise ValueError(f"forbidden decision fields: {list(forbidden)}")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError("decision snapshot fields do not match the closed schema")
        event = {
            "event_id": payload["event_id"],
            "event_time": payload["event_time"],
            "source_group": payload["source_group"],
            "related_source_groups": payload["related_source_groups"],
            "active_constraints": payload["active_constraints"],
        }
        return cls.from_event(
            event,
            evidence_event_times=payload["evidence_event_times"],
            read_state_hash=str(payload["read_state_hash"]),
        )


def _sha256_path(path: Path) -> str | None:
    """Hash a file without hiding a missing source behind an empty digest."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path) -> JsonDict:
    """Read one JSON object and preserve unreadable input as an empty value."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _gate(
    check: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Record an exact gate value and its independent pass decision."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every check and identify the first failure without hiding later ones."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = [row for row in rows if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "checks": rows,
        "failed_check": first.get("check") if first else None,
        "expected_value": first.get("expected_value") if first else "all checks pass",
        "observed_value": first.get("observed_value") if first else "all checks pass",
        "failed_checks": [
            {
                "failed_check": row.get("check"),
                "expected_value": deepcopy(row.get("expected_value")),
                "observed_value": deepcopy(row.get("observed_value")),
            }
            for row in failed
        ],
        "passed": not failed,
    }


def _outcome_receipt(outcome: Mapping[str, Any]) -> str:
    """Bind all exact outcome fields to the public event's sealed receipt."""

    return sha256_json(dict(outcome))


def build_synthetic_fixture(
    *, event_count: int, source_group_count: int
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Build deterministic fixtures for protocol tests without claiming live data."""

    if event_count < 1 or source_group_count < 1:
        raise ValueError("fixture counts must be positive")
    kinds = (
        "anchor",
        "direct_use",
        "positive_transfer",
        "conflict",
        "drift",
        "unknown",
        "no_op",
        "harmful_transfer",
    )
    effects = {
        "anchor": 0.2,
        "direct_use": 0.2,
        "positive_transfer": 0.2,
        "conflict": -0.2,
        "drift": -0.2,
        "unknown": 0.0,
        "no_op": 0.0,
        "harmful_transfer": -0.2,
    }
    events: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    for index in range(event_count):
        kind = kinds[index % len(kinds)]
        group = f"source-{index % source_group_count:02d}"
        prior_group = f"source-{(index - 1) % source_group_count:02d}"
        outcome = {
            "event_id": f"event-{index:04d}",
            "reuse_effect": effects[kind],
            "retention_passed": kind not in {"drift", "harmful_transfer"},
            "baseline_exact_quality": 0.6,
        }
        related = (
            (prior_group,) if kind in {"positive_transfer", "no_op", "harmful_transfer"} else ()
        )
        event = {
            "event_id": outcome["event_id"],
            "event_time": index,
            "game_id": f"game-{index % 6:02d}",
            "source_group": group,
            "case_kind": kind,
            "related_source_groups": list(related),
            "active_constraints": ["constraint-x"] if kind == "conflict" else [],
            "sealed_outcome_receipt_hash": _outcome_receipt(outcome),
        }
        events.append(event)
        outcomes.append(outcome)
    return events, outcomes


def validate_stream(
    events: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]
) -> bool:
    """Reject changed order, identity, leakage, or sealed outcome bytes."""

    if not events or len(events) != len(outcomes):
        raise ValueError("events and outcomes must have equal nonzero length")
    times = [int(event.get("event_time", -1)) for event in events]
    if times != sorted(times) or len(set(times)) != len(times):
        raise ValueError("event order is not strictly chronological")
    event_ids = [str(event.get("event_id", "")) for event in events]
    if not all(event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError("event identities must be unique and nonempty")
    for event, outcome in zip(events, outcomes, strict=True):
        forbidden = _forbidden_paths(event)
        if forbidden:
            raise ValueError(f"outcome data before sealing: {list(forbidden)}")
        if str(outcome.get("event_id", "")) != str(event["event_id"]):
            raise ValueError("outcome identity does not match the event")
        if event.get("sealed_outcome_receipt_hash") != _outcome_receipt(outcome):
            raise ValueError("exact outcome receipt changed")
        if not str(event.get("source_group", "")):
            raise ValueError("source group is required")
    return True


def _initial_records() -> list[JsonDict]:
    """Return the same small protected policy for all four isolated arms."""

    return [
        {
            "policy_key": "initial:temporal_firewall",
            "protected": True,
            "policy_text": "Use only committed evidence from earlier events.",
        },
        {
            "policy_key": "initial:no_preference_no_op",
            "protected": True,
            "policy_text": "Keep exact state bytes when evidence supports no preference.",
        },
    ]


def _experiences(store: PolicyStore) -> tuple[ContextBoundExperience, ...]:
    """Parse only typed learned records from the arm's current committed state."""

    return tuple(
        ContextBoundExperience.from_policy_record(row)
        for row in store.records()
        if str(row.get("policy_key", "")).startswith("context_experience:")
    )


def _authorize(
    arm: str,
    snapshot: DecisionSnapshot,
    records: Sequence[ContextBoundExperience],
) -> tuple[AuthorizationDecision, str]:
    """Apply the frozen arm rule without reading the current exact outcome."""

    if arm == "no_reuse" or not records:
        return (
            AuthorizationDecision(Authorization.REJECT, "no_prior_reuse", (), snapshot.event_time),
            "no_reuse",
        )
    if arm == "flat_reuse":
        latest = max(records, key=lambda row: (row.event_time, row.experience_id))
        return (
            AuthorizationDecision(
                Authorization.USE,
                "flat_predecessor_reuse",
                (latest.experience_id,),
                snapshot.event_time,
            ),
            "reuse",
        )
    if arm == "validate_all":
        latest = max(records, key=lambda row: (row.event_time, row.experience_id))
        supported = (
            latest.observed_effect_interval.lower > NO_PREFERENCE_BOUND
            and latest.retention_result is RetentionResult.PASSED
            and not set(latest.conflicts).intersection(snapshot.active_constraints)
        )
        return (
            AuthorizationDecision(
                Authorization.VALIDATE,
                "validate_all_predecessor_evidence",
                (latest.experience_id,),
                snapshot.event_time,
            ),
            "reuse" if supported else "no_op",
        )
    if arm != "context_bound":
        raise ValueError(f"unknown arm: {arm}")
    view = DecisionView(
        event_id=snapshot.event_id,
        event_time=snapshot.event_time,
        parent_policy_hash=POLICY_HASH,
        source_group=snapshot.source_group,
        related_source_groups=snapshot.related_source_groups,
        constraint_schema_hash=CONSTRAINT_SCHEMA_HASH,
        active_constraints=snapshot.active_constraints,
    )
    decision = ContextAuthorizationMachine().authorize(records, view)
    return decision, "reuse" if decision.authorization is not Authorization.REJECT else "no_reuse"


def _experience_from_outcome(
    event: Mapping[str, Any], outcome: Mapping[str, Any]
) -> ContextBoundExperience:
    """Create one typed update only after the exact outcome receipt opens."""

    effect = float(outcome["reuse_effect"])
    retention = (
        RetentionResult.PASSED
        if outcome.get("retention_passed") is True
        else RetentionResult.FAILED
    )
    event_time = int(event["event_time"])
    return ContextBoundExperience(
        experience_id=f"{event['event_id']}:{event['source_group']}",
        parent_policy_hash=POLICY_HASH,
        source_group=str(event["source_group"]),
        constraint_schema_hash=CONSTRAINT_SCHEMA_HASH,
        support_interval=SupportInterval(event_time + 1, event_time + 24),
        observed_effect_interval=NumericInterval(effect, effect),
        retention_result=retention,
        conflicts=tuple(str(value) for value in event["active_constraints"]),
        event_time=event_time,
        source_receipt_hash=_outcome_receipt(outcome),
    )


def _exact_quality(action: str, outcome: Mapping[str, Any]) -> float:
    """Score the sealed action against the exact later reuse effect."""

    baseline = float(outcome["baseline_exact_quality"])
    effect = float(outcome["reuse_effect"])
    return round(max(0.0, min(1.0, baseline + (effect if action == "reuse" else 0.0))), 12)


def _retention_ok(store: PolicyStore) -> bool:
    """Require every protected initial policy key to remain in active state."""

    keys = {str(row.get("policy_key")) for row in store.records() if row.get("protected")}
    return keys == {str(row["policy_key"]) for row in _initial_records()}


def run_comparison(
    events: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    work_root: Path,
    capacity_bytes: int = DEFAULT_CAPACITY_BYTES,
    random_seed: int = RANDOM_SEED,
    failed_commits: set[tuple[str, str]] | None = None,
) -> JsonDict:
    """Run four isolated arms and expose every causal row for recomputation."""

    validate_stream(events, outcomes)
    failures = failed_commits or set()
    work_root.mkdir(parents=True, exist_ok=True)
    stores = {
        arm: PolicyStore(
            work_root / arm,
            initial_records=_initial_records(),
            max_state_bytes=capacity_bytes,
        )
        for arm in ARMS
    }
    initial_hashes = {arm: store.state_hash for arm, store in stores.items()}
    arm_definitions = [
        {
            "arm": arm,
            "store_path": str(stores[arm].root),
            "initial_state_hash": initial_hashes[arm],
            "decision_budget": DECISION_BUDGET,
            "validation_budget": VALIDATION_BUDGET,
            "capacity_bytes": capacity_bytes,
            "random_seed": random_seed,
            "isolated": True,
        }
        for arm in ARMS
    ]
    isolation_rows = [
        {
            "arm": arm,
            "other_arm": other,
            "arm_store_path": str(stores[arm].root),
            "other_store_path": str(stores[other].root),
        }
        for arm in ARMS
        for other in ARMS
        if arm != other
    ]
    result: JsonDict = {
        "arm_definitions": arm_definitions,
        "isolation_rows": isolation_rows,
        "chronological_event_rows": [],
        "decision_snapshot_rows": [],
        "authorization_rows": [],
        "validation_rows": [],
        "exact_outcome_rows": [],
        "journal_rows": [],
        "commit_rows": [],
        "rollback_rows": [],
        "capacity_rows": [],
        "retention_rows": [],
    }
    for event, outcome in zip(events, outcomes, strict=True):
        event_id = str(event["event_id"])
        event_time = int(event["event_time"])
        for offset, arm in enumerate(ARMS):
            store = stores[arm]
            prior_records = _experiences(store)
            snapshot = DecisionSnapshot.from_event(
                event,
                evidence_event_times=[record.event_time for record in prior_records],
                read_state_hash=store.state_hash,
            )
            decision, action = _authorize(arm, snapshot, prior_records)
            snapshot_row = {
                "arm": arm,
                **snapshot.to_dict(),
                "decision_sealed": True,
                "random_seed": random_seed + event_time,
            }
            result["decision_snapshot_rows"].append(snapshot_row)
            result["authorization_rows"].append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "event_time": event_time,
                    "authorization": decision.authorization.value,
                    "reason": decision.reason,
                    "evidence_ids": list(decision.evidence_ids),
                    "action": action,
                    "sealed": True,
                }
            )
            if decision.authorization is Authorization.VALIDATE:
                result["validation_rows"].append(
                    {
                        "arm": arm,
                        "event_id": event_id,
                        "event_time": event_time,
                        "cost": VALIDATION_COST,
                        "budget": VALIDATION_BUDGET,
                        "rule": "reuse_only_positive_retention_safe_predecessor",
                        "planned_before_outcome": True,
                    }
                )

            result["exact_outcome_rows"].append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "event_time": event_time,
                    "outcome_receipt_hash": _outcome_receipt(outcome),
                    "decision_sealed_before_outcome": True,
                    "opened_at_sequence": len(result["exact_outcome_rows"]),
                }
            )
            parent_bytes = store.state_bytes
            parent_hash = store.state_hash
            before_keys = {str(row.get("policy_key")) for row in store.records()}
            journal_start = len(store.journal_rows())
            effect = float(outcome["reuse_effect"])
            disposition = "reject"
            committed = False
            rolled_back = False
            rollback_reason: str | None = None
            if arm != "no_reuse" and abs(effect) <= NO_PREFERENCE_BOUND:
                disposition = "no_op"
            elif arm != "no_reuse":
                proposal = _experience_from_outcome(event, outcome).to_policy_record()
                if (arm, event_id) in failures:
                    try:
                        store.commit(proposal, interrupt_after_prepare=True)
                    except ForcedInterruption:
                        store = PolicyStore(
                            store.root,
                            initial_records=_initial_records(),
                            max_state_bytes=capacity_bytes,
                        )
                        stores[arm] = store
                    disposition = "rollback"
                    rolled_back = True
                    rollback_reason = "forced_commit_failure"
                else:
                    receipt = store.commit(proposal)
                    committed = True
                    disposition = "commit"
                    if effect < -NO_PREFERENCE_BOUND or outcome.get("retention_passed") is not True:
                        store.rollback(
                            parent_bytes,
                            transaction_id=str(receipt["transaction_id"]),
                            reason="harmful_update",
                        )
                        committed = False
                        rolled_back = True
                        disposition = "rollback"
                        rollback_reason = "harmful_update"
                    else:
                        result["commit_rows"].append(
                            {
                                "arm": arm,
                                "event_id": event_id,
                                "parent_state_hash": receipt["parent_state_hash"],
                                "new_state_hash": receipt["new_state_hash"],
                                "transaction_id": receipt["transaction_id"],
                                "committed": True,
                            }
                        )
            journal_delta = store.journal_rows()[journal_start:]
            phases = [str(row["phase"]) for row in journal_delta]
            if rolled_back:
                result["rollback_rows"].append(
                    {
                        "arm": arm,
                        "event_id": event_id,
                        "reason": rollback_reason,
                        "parent_state_hash": parent_hash,
                        "restored_state_hash": store.state_hash,
                        "parent_bytes_restored": store.state_bytes == parent_bytes,
                        "journal_phases": phases,
                    }
                )
            after_records = store.records()
            after_keys = {str(row.get("policy_key")) for row in after_records}
            evicted = sorted(before_keys - after_keys)
            result["journal_rows"].append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "disposition": disposition,
                    "parent_state_hash": parent_hash,
                    "final_state_hash": store.state_hash,
                    "journal_phases": phases,
                    "journal_replay_passed": store.replay_journal()["passed"],
                }
            )
            protected_present = _retention_ok(store)
            result["capacity_rows"].append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "state_bytes": store.state_size,
                    "capacity_bytes": capacity_bytes,
                    "within_capacity": store.state_size <= capacity_bytes,
                    "evicted_policy_keys": evicted,
                    "protected_records_present": protected_present,
                }
            )
            result["retention_rows"].append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "protected_retention": float(protected_present),
                }
            )
            exact_quality = _exact_quality(action, outcome)
            source_transfer = any(
                record.experience_id in decision.evidence_ids
                and record.source_group != str(event["source_group"])
                for record in prior_records
            )
            result["chronological_event_rows"].append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "event_time": event_time,
                    "game_id": str(event["game_id"]),
                    "source_group": str(event["source_group"]),
                    "case_kind": str(event["case_kind"]),
                    "authorization": decision.authorization.value,
                    "action": action,
                    "disposition": disposition,
                    "exact_quality": exact_quality,
                    "harmful_update": action == "reuse" and effect < -NO_PREFERENCE_BOUND,
                    "useful_update": action == "reuse" and effect > NO_PREFERENCE_BOUND,
                    "validation_cost": (
                        VALIDATION_COST if decision.authorization is Authorization.VALIDATE else 0.0
                    ),
                    "abstained": action != "reuse",
                    "protected_retention": float(protected_present),
                    "source_group_transfer": source_transfer,
                    "committed": committed,
                    "rolled_back": rolled_back,
                    "decision_budget": DECISION_BUDGET,
                    "validation_budget": VALIDATION_BUDGET,
                    "capacity_bytes": capacity_bytes,
                    "random_seed": random_seed + event_time,
                    "arm_order": offset,
                }
            )
    result.update(recompute_aggregates(result["chronological_event_rows"]))
    return result


def _mean(values: Sequence[float]) -> float | None:
    """Return a stable mean while preserving an empty metric as unsupported."""

    return round(sum(values) / len(values), 12) if values else None


def _paired_interval(rows: Sequence[Mapping[str, Any]], *, seed: int) -> JsonDict:
    """Bootstrap matched event differences with events as the sampling unit."""

    by_event: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_event[str(row["event_id"])][str(row["arm"])] = float(row["exact_quality"])
    deltas = [
        values["context_bound"] - values["no_reuse"]
        for _event_id, values in sorted(by_event.items())
        if {"context_bound", "no_reuse"} <= set(values)
    ]
    if not deltas:
        return {
            "comparison": "context_bound_minus_no_reuse_equal_budget_quality",
            "point_estimate": None,
            "lower": None,
            "upper": None,
            "unit_count": 0,
            "confidence": 0.95,
        }
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(deltas) for _ in deltas) / len(deltas) for _ in range(2_000)
    )
    return {
        "comparison": "context_bound_minus_no_reuse_equal_budget_quality",
        "point_estimate": _mean(deltas),
        "lower": round(means[int(0.025 * (len(means) - 1))], 12),
        "upper": round(means[int(0.975 * (len(means) - 1))], 12),
        "unit_count": len(deltas),
        "confidence": 0.95,
    }


def recompute_aggregates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce every published metric from event rows without aggregate inputs."""

    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_game: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        by_arm[arm].append(row)
        by_game[(str(row["game_id"]), arm)].append(row)
        by_source[(str(row["source_group"]), arm)].append(row)
    rates: JsonDict = {}
    rates["harmful_update_rate_by_arm"] = {
        arm: _mean([float(row["harmful_update"]) for row in by_arm.get(arm, [])]) for arm in ARMS
    }
    rates["useful_update_rate_by_arm"] = {
        arm: _mean([float(row["useful_update"]) for row in by_arm.get(arm, [])]) for arm in ARMS
    }
    rates["validation_cost_by_arm"] = {
        arm: round(sum(float(row["validation_cost"]) for row in by_arm.get(arm, [])), 12)
        for arm in ARMS
    }
    rates["final_exact_quality_by_arm"] = {
        arm: _mean([float(row["exact_quality"]) for row in by_arm.get(arm, [])]) for arm in ARMS
    }
    rates["equal_budget_quality_by_arm"] = deepcopy(rates["final_exact_quality_by_arm"])
    rates["abstention_rate_by_arm"] = {
        arm: _mean([float(row["abstained"]) for row in by_arm.get(arm, [])]) for arm in ARMS
    }
    rates["per_game_results"] = [
        {
            "game_id": game,
            "arm": arm,
            "event_count": len(group),
            "exact_quality": _mean([float(row["exact_quality"]) for row in group]),
            "harmful_update_rate": _mean([float(row["harmful_update"]) for row in group]),
        }
        for (game, arm), group in sorted(by_game.items())
    ]
    rates["per_source_group_rows"] = [
        {
            "source_group": source,
            "arm": arm,
            "event_count": len(group),
            "exact_quality": _mean([float(row["exact_quality"]) for row in group]),
            "harmful_update_rate": _mean([float(row["harmful_update"]) for row in group]),
            "useful_update_rate": _mean([float(row["useful_update"]) for row in group]),
            "transfer_rate": _mean([float(row["source_group_transfer"]) for row in group]),
        }
        for (source, arm), group in sorted(by_source.items())
    ]
    rates["paired_interval_rows"] = [_paired_interval(rows, seed=RANDOM_SEED)]
    return rates


def _arm_summary_rows(comparison: Mapping[str, Any]) -> list[JsonDict]:
    """Build compact arm summaries from already recomputed metric dictionaries."""

    return [
        {
            "arm": arm,
            "harmful_update_rate": comparison["harmful_update_rate_by_arm"][arm],
            "useful_update_rate": comparison["useful_update_rate_by_arm"][arm],
            "validation_cost": comparison["validation_cost_by_arm"][arm],
            "final_exact_quality": comparison["final_exact_quality_by_arm"][arm],
            "equal_budget_quality": comparison["equal_budget_quality_by_arm"][arm],
            "abstention_rate": comparison["abstention_rate_by_arm"][arm],
        }
        for arm in ARMS
    ]


def _comparison_complete(comparison: Mapping[str, Any]) -> bool:
    """Require all planned matched rows, budgets, isolation, and chronology."""

    event_rows = list(comparison.get("chronological_event_rows", []))
    if not event_rows:
        return False
    event_ids = {str(row["event_id"]) for row in event_rows}
    counts = {arm: sum(row["arm"] == arm for row in event_rows) for arm in ARMS}
    expected = len(event_ids)
    return bool(
        expected >= MIN_EVENT_COUNT
        and all(counts[arm] == expected for arm in ARMS)
        and len(comparison.get("decision_snapshot_rows", [])) == expected * len(ARMS)
        and len(comparison.get("exact_outcome_rows", [])) == expected * len(ARMS)
        and len({row["store_path"] for row in comparison.get("arm_definitions", [])}) == len(ARMS)
        and all(
            row.get("max_evidence_event_time", -1) < row.get("event_time", -1)
            for row in comparison.get("decision_snapshot_rows", [])
        )
        and all(row.get("within_capacity") is True for row in comparison.get("capacity_rows", []))
        and all(
            row.get("protected_records_present") is True
            for row in comparison.get("capacity_rows", [])
        )
    )


def _value_gate(artifact: Mapping[str, Any]) -> tuple[bool, JsonDict]:
    """Apply the frozen value rule without credit for lower validation cost."""

    harmful = artifact["harmful_update_rate_by_arm"]
    quality = artifact["equal_budget_quality_by_arm"]
    interval = artifact["paired_interval_rows"][0] if artifact["paired_interval_rows"] else {}
    retention = [
        float(row["protected_retention"])
        for row in artifact["chronological_event_rows"]
        if row["arm"] == "context_bound"
    ]
    conditions = {
        "context_bound_harmful_lower_than_flat": (
            harmful.get("context_bound") is not None
            and harmful.get("flat_reuse") is not None
            and harmful["context_bound"] < harmful["flat_reuse"]
        ),
        "context_bound_equal_budget_quality_above_no_reuse": (
            quality.get("context_bound") is not None
            and quality.get("no_reuse") is not None
            and quality["context_bound"] > quality["no_reuse"]
        ),
        "paired_interval_lower_above_zero": (
            interval.get("lower") is not None and interval["lower"] > 0.0
        ),
        "protected_retention_within_bound": (
            _mean(retention) is not None and _mean(retention) >= PROTECTED_RETENTION_BOUND
        ),
    }
    return bool(
        artifact["bcit_comparison_complete_score"] == 1 and all(conditions.values())
    ), conditions


def _source_hashes(repo_root: Path) -> JsonDict:
    """Hash every imported artifact and code file needed to interpret the run."""

    paths = (
        EXP6978_RELATIVE_PATH,
        EXP7021_RELATIVE_PATH,
        UPSTREAM_RELATIVE_PATH,
        AUTHORIZATION_MODULE_RELATIVE_PATH,
        UPSTREAM_MODULE_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        WRAPPER_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    )
    return {str(path): _sha256_path(repo_root / path) for path in paths}


def _nearest_existing_parent(path: Path) -> Path:
    """Find the existing directory that controls creation of a requested path."""

    candidate = path if path.is_dir() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _isolated_stores_writable(work_root: Path) -> bool:
    """Probe all four private store layouts without leaving experiment state."""

    parent = _nearest_existing_parent(work_root)
    if not os.access(parent, os.W_OK):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="exp7070-preflight-", dir=parent) as directory:
            stores = [
                PolicyStore(
                    Path(directory) / arm,
                    initial_records=_initial_records(),
                    max_state_bytes=DEFAULT_CAPACITY_BYTES,
                )
                for arm in ARMS
            ]
            return len({str(store.root) for store in stores}) == len(ARMS)
    except (OSError, ValueError):
        return False


def collect_preconditions(
    *, repo_root: Path, output_path: Path, work_root: Path
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Check exact upstream authorization, stream, code, size, and storage gates."""

    upstream_path = repo_root / UPSTREAM_RELATIVE_PATH
    upstream = _read_object(upstream_path)
    manifest = upstream.get("chronological_stream_manifest", {})
    events = manifest.get("events", []) if isinstance(manifest, Mapping) else []
    events = events if isinstance(events, list) else []
    event_times = [row.get("event_time") for row in events if isinstance(row, Mapping)]
    groups = {
        str(row.get("source_group"))
        for row in events
        if isinstance(row, Mapping) and row.get("source_group")
    }
    declared = upstream.get("source_artifact_hashes", {})
    declared = declared if isinstance(declared, Mapping) else {}
    stream_observed = sha256_json(manifest)
    event_order_valid = bool(
        len(event_times) == len(events)
        and event_times == sorted(event_times)
        and len(set(event_times)) == len(event_times)
    )
    checks = [
        _gate("upstream_artifact_readable", True, bool(upstream)),
        _gate(
            "context_authorization_contract_ready_score",
            1,
            upstream.get("context_authorization_contract_ready_score"),
        ),
        _gate(
            "chronological_stream_manifest_hash",
            upstream.get("chronological_stream_manifest_hash"),
            stream_observed,
        ),
        _gate(
            "context_authorization_code_hash",
            declared.get(str(AUTHORIZATION_MODULE_RELATIVE_PATH)),
            _sha256_path(repo_root / AUTHORIZATION_MODULE_RELATIVE_PATH),
        ),
        _gate(
            "upstream_contract_code_hash",
            declared.get(str(UPSTREAM_MODULE_RELATIVE_PATH)),
            _sha256_path(repo_root / UPSTREAM_MODULE_RELATIVE_PATH),
        ),
        _gate("chronological_event_order", True, event_order_valid),
        _gate(
            "minimum_chronological_event_count",
            f">={MIN_EVENT_COUNT}",
            len(events),
            passed=len(events) >= MIN_EVENT_COUNT,
        ),
        _gate(
            "minimum_source_group_count",
            f">={MIN_SOURCE_GROUP_COUNT}",
            len(groups),
            passed=len(groups) >= MIN_SOURCE_GROUP_COUNT,
        ),
        _gate("isolated_store_paths_writable", True, _isolated_stores_writable(work_root)),
        _gate(
            "artifact_path_writable",
            True,
            os.access(_nearest_existing_parent(output_path), os.W_OK),
        ),
    ]
    return checks, _source_hashes(repo_root), upstream


def _citations(source_hashes: Mapping[str, Any]) -> list[JsonDict]:
    """Cite prior nulls and the contract without converting them into value evidence."""

    return [
        {
            "experiment_id": 6978,
            "fields_imported": ["verdict_class", "honest_verdict"],
            "sha256": source_hashes.get(str(EXP6978_RELATIVE_PATH)),
        },
        {
            "experiment_id": 7021,
            "fields_imported": ["verdict_class", "honest_verdict"],
            "sha256": source_hashes.get(str(EXP7021_RELATIVE_PATH)),
        },
        {
            "experiment_id": 7069,
            "fields_imported": [
                "context_authorization_contract_ready_score",
                "chronological_stream_manifest",
                "chronological_stream_manifest_hash",
                "protected_retention_manifest",
            ],
            "sha256": source_hashes.get(str(UPSTREAM_RELATIVE_PATH)),
        },
    ]


def _artifact_base(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build fields shared by blocked, partial, null, and positive results."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "cited_upstream_artifacts": _citations(source_hashes),
        "upstream_gate_rows": deepcopy(list(preconditions)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "verifier_is_oracle": False,
    }


def _without_timing(value: Any) -> Any:
    """Remove measured time before the deterministic checksum is computed."""

    if isinstance(value, Mapping):
        return {
            key: _without_timing(child)
            for key, child in value.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_without_timing(child) for child in value]
    return deepcopy(value)


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all deterministic evidence while excluding measured duration."""

    return sha256_json(_without_timing(artifact))


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a schema-complete block without inventing comparison rows."""

    artifact = _artifact_base(
        run_date=run_date,
        duration_s=duration_s,
        source_hashes=source_hashes,
        preconditions=preconditions,
    )
    for field in ROW_FIELDS:
        artifact[field] = []
    for field in AGGREGATE_FIELDS:
        artifact[field] = {}
    artifact.update(
        {
            "bcit_comparison_complete_score": 0,
            "bcit_self_learning_value_score": 0,
            "gate_check_summary": _gate_summary(preconditions),
            "verdict_class": "blocked",
            "honest_verdict": "blocked_insufficient_frozen_bcit_stream",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def complete_artifact_from_comparison(
    comparison: Mapping[str, Any],
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce a completed engine run into one row-owned terminal artifact."""

    artifact = _artifact_base(
        run_date=run_date,
        duration_s=duration_s,
        source_hashes=source_hashes,
        preconditions=preconditions,
    )
    for field in ROW_FIELDS:
        if field == "rows":
            continue
        artifact[field] = deepcopy(comparison.get(field, []))
    for field in AGGREGATE_FIELDS:
        artifact[field] = deepcopy(comparison[field])
    artifact["rows"] = _arm_summary_rows(comparison)
    complete = _comparison_complete(comparison)
    artifact["bcit_comparison_complete_score"] = int(complete)
    artifact["bcit_self_learning_value_score"] = 0
    positive, conditions = _value_gate(artifact)
    artifact["bcit_self_learning_value_score"] = int(positive)
    complete_gate = _gate("bcit_comparison_complete_score", 1, int(complete))
    value_gate = _gate("bcit_self_learning_value_score", 1, int(positive))
    value_gate["condition_details"] = conditions
    artifact["gate_check_summary"] = _gate_summary([*preconditions, complete_gate, value_gate])
    if positive:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = "complete_positive_bcit_context_bound_self_learning"
    elif complete:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "complete_null_bcit_self_learning_repeated_null_retire"
    else:
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial_bcit_self_learning_comparison"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    work_root: Path | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Check preconditions, run valid evidence, or return an exact block."""

    started = time.perf_counter()
    output = output_path or repo_root / RESULT_RELATIVE_PATH
    work = work_root or repo_root / WORK_RELATIVE_PATH
    checks, source_hashes, upstream = collect_preconditions(
        repo_root=repo_root,
        output_path=output,
        work_root=work,
    )
    if any(row["passed"] is not True for row in checks):
        return build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            source_hashes=source_hashes,
            preconditions=checks,
        )
    events = deepcopy(upstream["chronological_stream_manifest"]["events"])
    outcomes = deepcopy(upstream.get("exact_outcome_rows", []))
    comparison = run_comparison(events, outcomes, work_root=work, random_seed=RANDOM_SEED)
    return complete_artifact_from_comparison(
        comparison,
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        source_hashes=source_hashes,
        preconditions=checks,
    )


def validate_artifact(artifact: Any) -> list[str]:
    """Recompute structure, aggregates, scores, and terminal verdict consistency."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"required_fields_missing:{','.join(missing)}"]
    if not isinstance(artifact["field_principles"], Mapping) or set(
        artifact["field_principles"]
    ) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle_mismatch")
    for field, label in (
        ("bcit_comparison_complete_score", "complete_score"),
        ("bcit_self_learning_value_score", "value_score"),
    ):
        value = artifact[field]
        if type(value) is not int or value not in {0, 1}:
            errors.append(f"{label}_must_be_bare_integer")
    verdict_class = artifact.get("verdict_class")
    prefixes = {
        "positive": "complete_positive_",
        "circular_positive": "complete_circular_positive_",
        "null": "complete_null_",
        "blocked": "blocked_",
        "disqualified": "disqualified_",
        "partial": "partial_",
    }
    if verdict_class not in prefixes:
        errors.append("verdict_class_invalid")
    elif not str(artifact.get("honest_verdict", "")).startswith(prefixes[verdict_class]):
        errors.append("verdict_prefix_mismatch")
    for field in ROW_FIELDS:
        if not isinstance(artifact[field], list):
            errors.append(f"{field}_not_list")
    for field in AGGREGATE_FIELDS:
        if not isinstance(artifact[field], Mapping):
            errors.append(f"{field}_not_mapping")
    if artifact["reproducibility_checksum"] != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    gate = artifact.get("gate_check_summary")
    if not isinstance(gate, Mapping) or "failed_check" not in gate:
        errors.append("gate_check_summary_invalid")
    if verdict_class == "blocked":
        if (
            artifact["bcit_comparison_complete_score"] != 0
            or artifact["bcit_self_learning_value_score"] != 0
        ):
            errors.append("blocked_scores_must_be_zero")
        if artifact["chronological_event_rows"]:
            errors.append("blocked_event_rows_must_be_empty")
        if (
            not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not gate.get("failed_checks")
        ):
            errors.append("blocked_gate_summary_invalid")
    elif all(isinstance(artifact[field], list) for field in ROW_FIELDS) and all(
        isinstance(artifact[field], Mapping) for field in AGGREGATE_FIELDS
    ):
        recomputed = recompute_aggregates(artifact["chronological_event_rows"])
        if any(artifact[field] != recomputed[field] for field in AGGREGATE_FIELDS) or any(
            artifact[field] != recomputed[field]
            for field in ("per_game_results", "per_source_group_rows", "paired_interval_rows")
        ):
            errors.append("aggregate_recomputation_mismatch")
        comparison = {field: artifact[field] for field in ROW_FIELDS if field != "rows"}
        expected_complete = int(_comparison_complete(comparison))
        if artifact["bcit_comparison_complete_score"] != expected_complete:
            errors.append("comparison_complete_score_mismatch")
        expected_positive, _conditions = _value_gate(artifact)
        if artifact["bcit_self_learning_value_score"] != int(expected_positive):
            errors.append("self_learning_value_score_mismatch")
    return errors


def _atomic_write(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish complete validated bytes through one filesystem replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_artifact(
    *,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    work_root: Path | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build, validate, and atomically write one terminal artifact."""

    artifact = build_artifact(
        repo_root=repo_root,
        output_path=output_path,
        work_root=work_root,
        run_date=run_date,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"Exp7070 artifact validation failed: {errors}")
    _atomic_write(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the fixed date and run the deterministic comparison command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / WORK_RELATIVE_PATH)
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"execution date must be {RUN_DATE}")
    artifact = write_artifact(
        output_path=args.output,
        repo_root=REPO_ROOT,
        work_root=args.work_root,
        run_date=args.date,
    )
    print(
        json.dumps(
            {
                "bcit_comparison_complete_score": artifact["bcit_comparison_complete_score"],
                "bcit_self_learning_value_score": artifact["bcit_self_learning_value_score"],
                "honest_verdict": artifact["honest_verdict"],
                "result": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns command execution.
    raise SystemExit(main())
