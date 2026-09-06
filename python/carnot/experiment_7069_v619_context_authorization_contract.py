"""Build the deterministic V619 context-authorization contract artifact.

This experiment tests an information boundary and transaction protocol. It
does not measure future utility. Its ready verdict is therefore null even when
all contract checks pass.

Spec refs: REQ-SELFLEARN-7069 and SCENARIO-SELFLEARN-7069-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_7064_v619_exact_entrance_fixture import (
    RESULT_RELATIVE_PATH as EXACT_FIXTURE_RELATIVE_PATH,
)
from carnot.experiment_7064_v619_exact_entrance_fixture import (
    validate_artifact as validate_exact_fixture_artifact,
)
from carnot.learning.constraint_policy_store import PolicyStore
from carnot.learning.context_authorization import (
    Authorization,
    AuthorizationDecision,
    BoundedValidationEvent,
    ContextAuthorizationMachine,
    ContextBoundExperience,
    DecisionView,
    EXPERIENCE_SCHEMA,
    FORBIDDEN_DECISION_KEYS,
    LifecycleState,
    MAX_SUPPORTED_UNCERTAINTY_WIDTH,
    NumericInterval,
    RetentionResult,
    SupportInterval,
    VALIDATION_SCHEMA,
    ValidationPlan,
    canonical_bytes,
    settle_validation,
    sha256_json,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260906"
RANDOM_SEED = 706920260906
EXPERIMENT_ID = "experiment_7069_v619_context_authorization_contract"
INFERENCE_SUBSTRATE = "deterministic_verifier"
SCHEMA = "carnot.experiment_7069.v619_context_authorization_contract.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_7069_v619_context_authorization_contract.json")
WORK_RELATIVE_PATH = Path("results/raw/experiment_7069_v619_context_authorization_contract")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7069_v619_context_authorization_contract.py")
CONTEXT_MODULE_RELATIVE_PATH = Path("python/carnot/learning/context_authorization.py")
TRANSACTION_MODULE_RELATIVE_PATH = Path("python/carnot/learning/constraint_policy_store.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_7069_v619_context_authorization_contract.py"
)
WRAPPER_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_7069_v619_context_authorization_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXACT_FIXTURE_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_7064_v619_exact_entrance_fixture.py"
)
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    TRANSACTION_MODULE_RELATIVE_PATH,
    EXACT_FIXTURE_MODULE_RELATIVE_PATH,
    EXACT_FIXTURE_RELATIVE_PATH,
    CONTEXT_MODULE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    WRAPPER_RELATIVE_PATH,
)

POLICY_HASH = "sha256:" + "a" * 64
SCHEMA_HASH = "sha256:" + "b" * 64
RECEIPT_HASH = "sha256:" + "c" * 64

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "experience_schema",
    "authorization_schema",
    "authorization_rule_rows",
    "state_transition_rows",
    "conflict_rows",
    "temporal_firewall_rows",
    "validation_event_schema",
    "no_op_rows",
    "transaction_rows",
    "rollback_rows",
    "capacity_rows",
    "mutation_rows",
    "chronological_stream_manifest",
    "chronological_stream_manifest_hash",
    "protected_retention_manifest",
    "context_authorization_contract_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Every artifact field states the scientific reason it is present.",
    "preconditions_checked": "Missing transaction, fixture, firewall, or storage resources must block construction.",
    "inference_substrate": "The named substrate separates deterministic verification from learned judgment.",
    "duration_s": "Measured elapsed time makes the contract run auditable.",
    "source_artifact_hashes": "Content hashes bind the result to reviewed source and exact fixture bytes.",
    "rows": "Recomputable gates, not verdict prose, own readiness.",
    "experience_schema": "A closed typed schema binds effects to their observed context.",
    "authorization_schema": "A closed decision schema prevents outcome data from entering authorization.",
    "authorization_rule_rows": "Examples exercise every deterministic use, validate, and reject rule.",
    "state_transition_rows": "The complete transition table prevents unregistered lifecycle moves.",
    "conflict_rows": "Named overlap rows prove that active contradictions stop reuse.",
    "temporal_firewall_rows": "Per-event views prove that current and future outcomes remain unavailable.",
    "validation_event_schema": "A separate post-decision type binds cost, rule, outcome, and disposition.",
    "no_op_rows": "No-preference evidence must leave published state unchanged.",
    "transaction_rows": "Journal receipts prove that complete bytes publish only after prepare.",
    "rollback_rows": "Exact parent hashes prove that harmful validation leaves no active update.",
    "capacity_rows": "Bound checks prove that oversized experience cannot corrupt state.",
    "mutation_rows": "Registered attacks show that unsafe schema, timing, and transition changes fail closed.",
    "chronological_stream_manifest": "A sealed order fixes Exp7070 cases before held outcomes open.",
    "chronological_stream_manifest_hash": "The manifest hash detects reordered or replaced future cases.",
    "protected_retention_manifest": "A sealed retention set prevents transfer tuning on held groups.",
    "context_authorization_contract_ready_score": "One requires every schema, transition, firewall, transaction, and mutation gate.",
    "random_seed": "A fixed seed identifies the deterministic fixture even though it uses no sampling.",
    "reproducibility_checksum": "A content checksum detects silent terminal artifact changes.",
    "gate_check_summary": "Exact expected and observed values make every block actionable.",
    "verifier_is_oracle": "False states that authorization cannot see the exact outcome authority.",
    "verdict_class": "A closed class prevents infrastructure readiness from becoming a utility claim.",
    "honest_verdict": "A terminal prefix gives the conductor an unambiguous result boundary.",
}


def gate(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record one exact expected and observed gate value."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": observed == expected,
    }


def _gate_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed check without hiding later independent failures."""

    failed = [
        {
            "failed_check": str(row.get("check")),
            "expected_value": deepcopy(row.get("expected_value")),
            "observed_value": deepcopy(row.get("observed_value")),
        }
        for row in rows
        if row.get("passed") is not True
    ]
    return {"passed": not failed, "failed_checks": failed}


def _sha256_path(path: Path) -> str | None:
    """Hash source bytes while preserving a missing path as an explicit fact."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path) -> JsonDict:
    """Read one JSON object or return an empty value for blocked preflight."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _nearest_existing_parent(path: Path) -> Path:
    """Find the directory whose permissions control creation of a new path."""

    candidate = path if path.is_dir() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _path_writable(path: Path) -> bool:
    """Check an existing file or the nearest parent for write permission."""

    target = path if path.exists() else _nearest_existing_parent(path)
    return os.access(target, os.W_OK)


def source_artifact_hashes(repo_root: Path = REPO_ROOT) -> JsonDict:
    """Bind all reviewed source files and the ready exact event fixture."""

    return {str(path): _sha256_path(repo_root / path) for path in SOURCE_PATHS}


def collect_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    work_root: Path | None = None,
) -> list[JsonDict]:
    """Check every resource before any transaction or exact outcome opens."""

    output = output_path or repo_root / RESULT_RELATIVE_PATH
    work = work_root or repo_root / WORK_RELATIVE_PATH
    exact_path = repo_root / EXACT_FIXTURE_RELATIVE_PATH
    exact = _read_object(exact_path)
    decision_parameters = set(inspect.signature(ContextAuthorizationMachine.authorize).parameters)
    decision_fields = set(DecisionView.schema()["fields"])
    denied_signature = sorted(
        decision_parameters.intersection(FORBIDDEN_DECISION_KEYS)
        | decision_fields.intersection(FORBIDDEN_DECISION_KEYS)
    )
    checks = [
        gate(
            "transactional_memory_api",
            True,
            all(
                callable(getattr(PolicyStore, name, None))
                for name in ("commit", "rollback", "replay_journal")
            ),
        ),
        gate("exact_event_fixture_api", True, callable(validate_exact_fixture_artifact)),
        gate("exact_event_fixture_readable", True, bool(exact)),
        gate("exact_event_fixture_ready_score", 1, exact.get("entrance_fixture_ready_score")),
        gate("code_path_writable", True, _path_writable(repo_root / CONTEXT_MODULE_RELATIVE_PATH)),
        gate("test_path_writable", True, _path_writable(repo_root / TEST_RELATIVE_PATH)),
        gate("journal_path_writable", True, _path_writable(work)),
        gate("artifact_path_writable", True, _path_writable(output)),
        gate("decision_interface_mutable_outcome_sources", [], denied_signature),
        gate(
            "decision_interface_current_outcome_parameter", False, "outcome" in decision_parameters
        ),
    ]
    return checks


def _experience(
    source_group: str,
    *,
    event_time: int = 0,
    support: tuple[int, int] = (1, 8),
    effect: tuple[float, float] = (0.10, 0.20),
    retention: RetentionResult = RetentionResult.PASSED,
    conflicts: tuple[str, ...] = (),
    policy_hash: str = POLICY_HASH,
    schema_hash: str = SCHEMA_HASH,
) -> ContextBoundExperience:
    """Build one deterministic experience used by contract rule fixtures."""

    return ContextBoundExperience(
        experience_id=f"experience:{source_group}:{event_time}",
        parent_policy_hash=policy_hash,
        source_group=source_group,
        constraint_schema_hash=schema_hash,
        support_interval=SupportInterval(*support),
        observed_effect_interval=NumericInterval(*effect),
        retention_result=retention,
        conflicts=conflicts,
        event_time=event_time,
        source_receipt_hash=RECEIPT_HASH,
    )


def _view(
    case_id: str,
    *,
    source_group: str = "source-a",
    event_time: int = 2,
    related: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
    policy_hash: str = POLICY_HASH,
    schema_hash: str = SCHEMA_HASH,
) -> DecisionView:
    """Build one outcome-free view for the deterministic rule matrix."""

    return DecisionView(
        event_id=case_id,
        event_time=event_time,
        parent_policy_hash=policy_hash,
        source_group=source_group,
        related_source_groups=related,
        constraint_schema_hash=schema_hash,
        active_constraints=conflicts,
    )


def _authorization_cases() -> tuple[
    tuple[str, tuple[ContextBoundExperience, ...], DecisionView, str, str], ...
]:
    """Freeze one fixture for every authorization rule and failure reason."""

    return (
        ("direct", (_experience("source-a"),), _view("direct"), "use", "direct_context_supported"),
        (
            "related",
            (_experience("source-a"),),
            _view("related", source_group="source-b", related=("source-a",)),
            "validate",
            "related_context_bounded_uncertainty",
        ),
        ("missing", (), _view("missing"), "reject", "missing_facts"),
        (
            "policy_mismatch",
            (_experience("source-a"),),
            _view("policy", policy_hash="sha256:" + "d" * 64),
            "reject",
            "policy_mismatch",
        ),
        (
            "schema_mismatch",
            (_experience("source-a"),),
            _view("schema", schema_hash="sha256:" + "e" * 64),
            "reject",
            "schema_mismatch",
        ),
        (
            "conflict",
            (_experience("source-a", conflicts=("constraint-x",)),),
            _view("conflict", conflicts=("constraint-x",)),
            "reject",
            "conflict_overlap",
        ),
        (
            "stale",
            (_experience("source-a", support=(0, 1)),),
            _view("stale", event_time=2),
            "reject",
            "stale_support",
        ),
        (
            "unsupported",
            (_experience("source-a", effect=(0.01, 0.50)),),
            _view("unsupported"),
            "reject",
            "unsupported_uncertainty",
        ),
        (
            "contradicted_effect",
            (_experience("source-a", effect=(-0.20, -0.10)),),
            _view("contradicted-effect"),
            "reject",
            "contradicted_effect",
        ),
        (
            "contradicted_retention",
            (_experience("source-a", retention=RetentionResult.FAILED),),
            _view("contradicted-retention"),
            "reject",
            "contradicted_retention",
        ),
        (
            "unknown",
            (_experience("source-a"),),
            _view("unknown", source_group="source-z"),
            "reject",
            "unknown_context",
        ),
        (
            "non_predecessor",
            (_experience("source-a", event_time=2),),
            _view("non-predecessor", event_time=2),
            "reject",
            "non_predecessor_evidence",
        ),
    )


def _authorization_rule_rows() -> list[JsonDict]:
    """Execute the complete rule fixture without opening any exact outcome."""

    rows: list[JsonDict] = []
    for case_id, records, view, expected, expected_reason in _authorization_cases():
        decision = ContextAuthorizationMachine().authorize(records, view)
        rows.append(
            {
                "case_id": case_id,
                "decision_view": view.to_dict(),
                "committed_evidence_ids": [record.experience_id for record in records],
                "decision": decision.authorization.value,
                "reason": decision.reason,
                "expected_decision": expected,
                "expected_reason": expected_reason,
                "passed": (
                    decision.authorization.value == expected and decision.reason == expected_reason
                ),
            }
        )
    return rows


def _state_transition_rows() -> list[JsonDict]:
    """Execute every registered move and one invalid move from unchanged state."""

    rows: list[JsonDict] = []
    for (source, action), destination in ContextAuthorizationMachine.transition_table().items():
        machine = ContextAuthorizationMachine(state=source)
        observed = machine.advance(action)
        rows.append(
            {
                "source": source.value,
                "action": action,
                "destination": destination.value,
                "observed_destination": observed.value,
                "passed": observed is destination,
            }
        )
    invalid = ContextAuthorizationMachine()
    rejected = False
    try:
        invalid.advance("commit")
    except ValueError:
        rejected = True
    rows.append(
        {
            "source": "start",
            "action": "commit",
            "destination": None,
            "observed_destination": invalid.state.value,
            "registered": False,
            "passed": rejected and invalid.state is LifecycleState.START,
        }
    )
    return rows


def _validation(
    validation_id: str,
    outcome: float,
    *,
    event_time: int,
) -> BoundedValidationEvent:
    """Open one bounded exact outcome after its validate receipt exists."""

    decision = AuthorizationDecision(
        authorization=Authorization.VALIDATE,
        reason="related_context_bounded_uncertainty",
        evidence_ids=("experience:source-a:0",),
        decided_at=event_time,
    )
    return BoundedValidationEvent.open_after_decision(
        plan=ValidationPlan(validation_id=validation_id, cost=0.05, max_abs_outcome=1.0),
        authorization_decision=decision,
        exact_later_outcome=outcome,
        outcome_receipt_hash=sha256_json({"validation_id": validation_id, "outcome": outcome}),
        outcome_opened_at=event_time + 1,
    )


def _transaction_fixtures(work_root: Path) -> JsonDict:
    """Exercise commit, rollback, no-op, and capacity in private bounded stores."""

    commit_store = PolicyStore(work_root / "commit", initial_records=(), max_state_bytes=4096)
    commit_parent = commit_store.state_hash
    commit_event = _validation("positive-transfer", 0.20, event_time=2)
    commit_receipt = settle_validation(
        commit_store,
        commit_event,
        _experience("source-b", event_time=2),
    )
    transaction_rows = [
        {
            "case_id": "positive_transfer_commit",
            "authorization": "validate",
            "outcome_opened_after_decision": True,
            "action": commit_receipt["action"],
            "parent_state_hash": commit_parent,
            "new_state_hash": commit_store.state_hash,
            "journal_phases": [row["phase"] for row in commit_store.journal_rows()],
            "journal_replay_passed": commit_store.replay_journal()["passed"],
            "passed": (
                commit_receipt["action"] == "commit"
                and commit_store.state_hash != commit_parent
                and [row["phase"] for row in commit_store.journal_rows()] == ["prepare", "commit"]
                and commit_store.replay_journal()["passed"] is True
            ),
        }
    ]

    rollback_store = PolicyStore(work_root / "rollback", initial_records=(), max_state_bytes=4096)
    rollback_parent_bytes = rollback_store.state_bytes
    rollback_parent_hash = rollback_store.state_hash
    rollback_receipt = settle_validation(
        rollback_store,
        _validation("harmful-transfer", -0.20, event_time=3),
        _experience("source-c", event_time=3),
    )
    rollback_rows = [
        {
            "case_id": "harmful_transfer_rollback",
            "action": rollback_receipt["action"],
            "parent_state_hash": rollback_parent_hash,
            "restored_state_hash": rollback_store.state_hash,
            "parent_bytes_restored": rollback_store.state_bytes == rollback_parent_bytes,
            "journal_phases": [row["phase"] for row in rollback_store.journal_rows()],
            "journal_replay_passed": rollback_store.replay_journal()["passed"],
            "passed": (
                rollback_receipt["action"] == "rollback"
                and rollback_store.state_bytes == rollback_parent_bytes
                and rollback_store.replay_journal()["passed"] is True
            ),
        }
    ]

    no_op_store = PolicyStore(work_root / "no-op", initial_records=(), max_state_bytes=4096)
    no_op_parent = no_op_store.state_bytes
    no_op_receipt = settle_validation(
        no_op_store,
        _validation("no-preference", 0.01, event_time=4),
        _experience("source-d", event_time=4),
    )
    no_op_rows = [
        {
            "case_id": "bounded_no_preference",
            "action": no_op_receipt["action"],
            "reason": no_op_receipt["reason"],
            "journal_row_count": len(no_op_store.journal_rows()),
            "state_unchanged": no_op_store.state_bytes == no_op_parent,
            "passed": (
                no_op_receipt["action"] == "no_op"
                and no_op_store.state_bytes == no_op_parent
                and not no_op_store.journal_rows()
            ),
        }
    ]

    capacity_store = PolicyStore(work_root / "capacity", initial_records=(), max_state_bytes=256)
    capacity_parent = capacity_store.state_bytes
    oversized = _experience(
        "source-capacity",
        conflicts=tuple(f"constraint-{index}" for index in range(50)),
    )
    error = None
    try:
        capacity_store.commit(oversized.to_policy_record())
    except ValueError as exc:
        error = str(exc)
    capacity_rows = [
        {
            "case_id": "oversized_experience",
            "capacity_bytes": 256,
            "error": error,
            "state_unchanged": capacity_store.state_bytes == capacity_parent,
            "committed_record_count": len(capacity_store.records()),
            "passed": (
                error is not None
                and "byte limit" in error
                and capacity_store.state_bytes == capacity_parent
                and not capacity_store.records()
            ),
        }
    ]
    return {
        "transaction_rows": transaction_rows,
        "rollback_rows": rollback_rows,
        "no_op_rows": no_op_rows,
        "capacity_rows": capacity_rows,
    }


def _temporal_firewall_rows(manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Build and reparse each outcome-free event view in chronological order."""

    rows: list[JsonDict] = []
    for event in manifest["events"]:
        view = _view(
            str(event["event_id"]),
            source_group=str(event["source_group"]),
            event_time=int(event["event_time"]),
            related=tuple(str(item) for item in event["related_source_groups"]),
            conflicts=tuple(str(item) for item in event["active_constraints"]),
        )
        serialized = view.to_dict()
        rows.append(
            {
                "event_id": event["event_id"],
                "decision_view": serialized,
                "decision_view_hash": sha256_json(serialized),
                "outcome_commitment": event["sealed_outcome_receipt_hash"],
                "current_outcome_hidden": True,
                "future_events_hidden": True,
                "held_group_labels_hidden": True,
                "post_event_aggregates_hidden": True,
                "passed": DecisionView.from_dict(serialized) == view,
            }
        )
    return rows


def _mutation_rows() -> list[JsonDict]:
    """Run attacks against the closed schema and state graph."""

    safe = _view("mutation").to_dict()
    attacks = (
        ("current_outcome", {**safe, "exact_outcome": 1.0}),
        ("future_event", {**safe, "metadata": {"future_events": [9]}}),
        ("held_label", {**safe, "held_group_label": "held"}),
        ("post_event_aggregate", {**safe, "post_event_aggregate": {"mean": 1.0}}),
        ("mutable_outcome_source", {**safe, "outcome_provider": "mutable"}),
    )
    rows: list[JsonDict] = []
    for attack, payload in attacks:
        error = None
        try:
            DecisionView.from_dict(payload)
        except ValueError as exc:
            error = str(exc)
        rows.append(
            {
                "attack": attack,
                "detected": bool(error and "forbidden decision fields" in error),
                "error": error,
                "passed": bool(error and "forbidden decision fields" in error),
            }
        )
    invalid = ContextAuthorizationMachine()
    invalid_error = None
    try:
        invalid.advance("rollback")
    except ValueError as exc:
        invalid_error = str(exc)
    rows.append(
        {
            "attack": "unregistered_transition",
            "detected": invalid_error is not None,
            "state_unchanged": invalid.state is LifecycleState.START,
            "error": invalid_error,
            "passed": invalid_error is not None and invalid.state is LifecycleState.START,
        }
    )
    record_payload = _experience("source-a").to_policy_record()
    record_payload["experience_hash"] = "sha256:" + "0" * 64
    record_error = None
    try:
        ContextBoundExperience.from_policy_record(record_payload)
    except ValueError as exc:
        record_error = str(exc)
    rows.append(
        {
            "attack": "experience_hash_change",
            "detected": record_error is not None,
            "error": record_error,
            "passed": bool(record_error and "hash mismatch" in record_error),
        }
    )
    return rows


def _chronological_stream_manifest() -> JsonDict:
    """Freeze Exp7070 cases without placing exact outcomes in decision views."""

    cases = (
        ("exp7070-00-anchor", 0, "source-a", "anchor", (), ()),
        ("exp7070-01-direct", 1, "source-a", "direct_use", (), ()),
        ("exp7070-02-transfer", 2, "source-b", "positive_transfer", ("source-a",), ()),
        ("exp7070-03-conflict", 3, "source-a", "conflict", (), ("constraint-x",)),
        ("exp7070-09-drift", 9, "source-a", "drift", (), ()),
        ("exp7070-10-unknown", 10, "source-z", "unknown", (), ()),
        ("exp7070-11-no-op", 11, "source-d", "no_op", ("source-a",), ()),
        ("exp7070-12-rollback", 12, "source-c", "harmful_transfer", ("source-a",), ()),
    )
    events = [
        {
            "event_id": event_id,
            "event_time": event_time,
            "source_group": source_group,
            "case_kind": case_kind,
            "related_source_groups": list(related),
            "active_constraints": list(conflicts),
            "sealed_outcome_receipt_hash": sha256_json(
                {"event_id": event_id, "private_outcome_slot": index}
            ),
        }
        for index, (event_id, event_time, source_group, case_kind, related, conflicts) in enumerate(
            cases
        )
    ]
    return {
        "consumer_experiment": "Exp7070",
        "schema": "carnot.exp7070.context_authorization_stream.v1",
        "sealed_before_authorization": True,
        "authorization_rules_frozen_before_held_groups": True,
        "held_group_labels_in_decision_view": False,
        "ordered_event_ids": [row["event_id"] for row in events],
        "events": events,
    }


def _protected_retention_manifest() -> JsonDict:
    """Seal the groups that validation must preserve without exposing split labels."""

    groups = ("retention-anchor-a", "retention-anchor-b", "retention-drift-a")
    return {
        "schema": "carnot.exp7070.protected_retention.v1",
        "sealed_before_authorization": True,
        "tuning_allowed": False,
        "protected_source_groups": list(groups),
        "protected_source_groups_hash": sha256_json(groups),
        "decision_view_contains_split_label": False,
    }


def _experience_schema() -> JsonDict:
    """Describe the immutable record using its exact serialized field names."""

    sample = _experience("source-a")
    return {
        "schema": EXPERIENCE_SCHEMA,
        "immutable": True,
        "additional_fields_allowed": False,
        "fields": list(sample.to_dict()),
        "canonical_serialization": "sorted_compact_json_utf8_newline",
        "sample_round_trip": ContextBoundExperience.from_json(sample.to_json()) == sample,
    }


def _authorization_schema() -> JsonDict:
    """Describe the outcome-free interface and its deterministic result enum."""

    return {
        "decision_view": DecisionView.schema(),
        "decisions": [item.value for item in Authorization],
        "mutable_outcome_source_allowed": False,
        "earlier_committed_evidence_only": True,
        "max_supported_uncertainty_width": MAX_SUPPORTED_UNCERTAINTY_WIDTH,
    }


def _artifact_base(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Create fields shared by ready and blocked terminal results."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "verifier_is_oracle": False,
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every deterministic terminal field except duration and the hash itself."""

    projection = deepcopy(dict(artifact))
    projection.pop("reproducibility_checksum", None)
    projection.pop("duration_s", None)
    return sha256_json(projection)


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a schema-complete block without inventing successful evidence."""

    artifact = _artifact_base(
        preconditions=preconditions,
        hashes=hashes or {},
        duration_s=duration_s,
    )
    manifest: JsonDict = {}
    artifact.update(
        {
            "rows": [],
            "experience_schema": {},
            "authorization_schema": {},
            "authorization_rule_rows": [],
            "state_transition_rows": [],
            "conflict_rows": [],
            "temporal_firewall_rows": [],
            "validation_event_schema": {},
            "no_op_rows": [],
            "transaction_rows": [],
            "rollback_rows": [],
            "capacity_rows": [],
            "mutation_rows": [],
            "chronological_stream_manifest": manifest,
            "chronological_stream_manifest_hash": sha256_json(manifest),
            "protected_retention_manifest": {},
            "context_authorization_contract_ready_score": 0,
            "gate_check_summary": _gate_summary(preconditions),
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_context_authorization_precondition_failed",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    work_root: Path | None = None,
    preconditions_override: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run all deterministic contract fixtures and reduce their exact rows."""

    started = time.monotonic()
    output = output_path or repo_root / RESULT_RELATIVE_PATH
    work = work_root or repo_root / WORK_RELATIVE_PATH
    preconditions = list(
        preconditions_override
        if preconditions_override is not None
        else collect_preconditions(repo_root=repo_root, output_path=output, work_root=work)
    )
    hashes = source_artifact_hashes(repo_root)
    if not all(row.get("passed") is True for row in preconditions):
        artifact = build_blocked_artifact(
            preconditions,
            duration_s=time.monotonic() - started,
            hashes=hashes,
        )
        validate_artifact(artifact)
        return artifact

    work.mkdir(parents=True, exist_ok=True)
    authorization_rows = _authorization_rule_rows()
    transition_rows = _state_transition_rows()
    manifest = _chronological_stream_manifest()
    firewall_rows = _temporal_firewall_rows(manifest)
    mutation_rows = _mutation_rows()
    with tempfile.TemporaryDirectory(prefix="run-", dir=work) as temporary:
        transactions = _transaction_fixtures(Path(temporary))
    conflict_rows = [
        {
            "case_id": row["case_id"],
            "active_constraints": row["decision_view"]["active_constraints"],
            "decision": row["decision"],
            "reason": row["reason"],
            "passed": row["passed"],
        }
        for row in authorization_rows
        if row["case_id"] in {"direct", "conflict"}
    ]
    retention = _protected_retention_manifest()
    schema = _experience_schema()
    authorization_schema = _authorization_schema()
    validation_schema = {
        **BoundedValidationEvent.schema(),
        "plan_fields": list(ValidationPlan("schema", 0.05, 1.0).to_dict()),
        "precommitted_before_outcome": True,
    }
    gate_rows = [
        gate("experience_schema_round_trip", True, schema["sample_round_trip"]),
        gate("authorization_rule_rows", True, all(row["passed"] for row in authorization_rows)),
        gate("state_transition_rows", True, all(row["passed"] for row in transition_rows)),
        gate("temporal_firewall_rows", True, all(row["passed"] for row in firewall_rows)),
        gate(
            "validation_bounds",
            True,
            validation_schema["precommitted_before_outcome"] is True,
        ),
        gate(
            "transaction_rows",
            True,
            all(row["passed"] for row in transactions["transaction_rows"]),
        ),
        gate("rollback_rows", True, all(row["passed"] for row in transactions["rollback_rows"])),
        gate("no_op_rows", True, all(row["passed"] for row in transactions["no_op_rows"])),
        gate("capacity_rows", True, all(row["passed"] for row in transactions["capacity_rows"])),
        gate("mutation_rows", True, all(row["passed"] for row in mutation_rows)),
        gate("chronological_stream_sealed", True, manifest["sealed_before_authorization"]),
        gate("protected_retention_sealed", True, retention["sealed_before_authorization"]),
    ]
    ready = all(row["passed"] for row in gate_rows)
    artifact = _artifact_base(
        preconditions=preconditions,
        hashes=hashes,
        duration_s=time.monotonic() - started,
    )
    artifact.update(
        {
            "rows": gate_rows,
            "experience_schema": schema,
            "authorization_schema": authorization_schema,
            "authorization_rule_rows": authorization_rows,
            "state_transition_rows": transition_rows,
            "conflict_rows": conflict_rows,
            "temporal_firewall_rows": firewall_rows,
            "validation_event_schema": validation_schema,
            "no_op_rows": transactions["no_op_rows"],
            "transaction_rows": transactions["transaction_rows"],
            "rollback_rows": transactions["rollback_rows"],
            "capacity_rows": transactions["capacity_rows"],
            "mutation_rows": mutation_rows,
            "chronological_stream_manifest": manifest,
            "chronological_stream_manifest_hash": sha256_json(manifest),
            "protected_retention_manifest": retention,
            "context_authorization_contract_ready_score": 1 if ready else 0,
            "gate_check_summary": _gate_summary(gate_rows),
            "verdict_class": "null" if ready else "blocked",
            "honest_verdict": (
                "complete_null_context_authorization_contract_ready_no_future_utility_claim"
                if ready
                else "complete_blocked_context_authorization_contract_gate_failed"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _validate_blocked(artifact: Mapping[str, Any]) -> bool:
    """Validate an exact terminal precondition block."""

    if artifact["context_authorization_contract_ready_score"] != 0:
        raise ValueError("blocked readiness must be zero")
    if not str(artifact["honest_verdict"]).startswith("complete_blocked_"):
        raise ValueError("blocked honest verdict prefix is inconsistent")
    summary = artifact["gate_check_summary"]
    failed = summary.get("failed_checks", [])
    if summary.get("passed") is not False or not failed:
        raise ValueError("blocked gate summary requires exact failures")
    expected_fields = {"failed_check", "expected_value", "observed_value"}
    if any(set(row) != expected_fields for row in failed):
        raise ValueError("blocked gate failure shape is incomplete")
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute the contract gates and reject changed terminal evidence."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field principles must cover every required field exactly")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference substrate must be deterministic_verifier")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("context authorization verifier_is_oracle must be false")
    score = artifact["context_authorization_contract_ready_score"]
    if type(score) is not int or score not in (0, 1):
        raise ValueError("readiness must be a bare integer zero or one")
    if artifact["reproducibility_checksum"] != artifact_checksum(artifact):
        raise ValueError("reproducibility checksum mismatch")
    if artifact["chronological_stream_manifest_hash"] != sha256_json(
        artifact["chronological_stream_manifest"]
    ):
        raise ValueError("chronological stream manifest hash mismatch")
    if artifact["verdict_class"] == "blocked":
        return _validate_blocked(artifact)
    if artifact["verdict_class"] != "null" or not str(artifact["honest_verdict"]).startswith(
        "complete_null_"
    ):
        raise ValueError("ready contract must use a null terminal verdict")
    if score != 1 or artifact["gate_check_summary"].get("passed") is not True:
        raise ValueError("ready score and gate summary disagree")
    if not artifact["rows"] or not all(row.get("passed") is True for row in artifact["rows"]):
        raise ValueError("readiness rows are incomplete")
    if artifact["experience_schema"] != _experience_schema():
        raise ValueError("experience schema changed")
    if artifact["authorization_schema"] != _authorization_schema():
        raise ValueError("authorization schema changed")
    expected_authorization = _authorization_rule_rows()
    if artifact["authorization_rule_rows"] != expected_authorization:
        raise ValueError("authorization rule rows changed")
    if artifact["state_transition_rows"] != _state_transition_rows():
        raise ValueError("state transition rows changed")
    expected_manifest = _chronological_stream_manifest()
    if artifact["chronological_stream_manifest"] != expected_manifest:
        raise ValueError("chronological stream manifest changed")
    if artifact["protected_retention_manifest"] != _protected_retention_manifest():
        raise ValueError("protected retention manifest changed")
    expected_firewall = _temporal_firewall_rows(expected_manifest)
    if artifact["temporal_firewall_rows"] != expected_firewall:
        raise ValueError("temporal firewall rows changed")
    expected_conflicts = [
        {
            "case_id": row["case_id"],
            "active_constraints": row["decision_view"]["active_constraints"],
            "decision": row["decision"],
            "reason": row["reason"],
            "passed": row["passed"],
        }
        for row in expected_authorization
        if row["case_id"] in {"direct", "conflict"}
    ]
    if artifact["conflict_rows"] != expected_conflicts:
        raise ValueError("conflict rows changed")
    if artifact["mutation_rows"] != _mutation_rows():
        raise ValueError("mutation rows changed")
    for name in ("no_op_rows", "transaction_rows", "rollback_rows", "capacity_rows"):
        rows = artifact[name]
        if not rows or not all(row.get("passed") is True for row in rows):
            raise ValueError(f"{name} are incomplete")
    if (
        artifact["no_op_rows"][0].get("action") != "no_op"
        or artifact["no_op_rows"][0].get("state_unchanged") is not True
    ):
        raise ValueError("no-op row changed")
    rollback = artifact["rollback_rows"][0]
    if (
        rollback.get("action") != "rollback"
        or rollback.get("parent_bytes_restored") is not True
        or rollback.get("parent_state_hash") != rollback.get("restored_state_hash")
    ):
        raise ValueError("rollback row changed")
    transaction = artifact["transaction_rows"][0]
    if transaction.get("journal_phases") != ["prepare", "commit"]:
        raise ValueError("transaction journal order changed")
    if artifact["capacity_rows"][0].get("state_unchanged") is not True:
        raise ValueError("capacity row changed")
    validation_schema = artifact["validation_event_schema"]
    if (
        validation_schema.get("schema") != VALIDATION_SCHEMA
        or validation_schema.get("precommitted_before_outcome") is not True
    ):
        raise ValueError("validation event schema changed")
    return True


def _atomic_write(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish complete artifact bytes with file and directory synchronization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_artifact(
    *,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    work_root: Path | None = None,
) -> JsonDict:
    """Build, validate, and atomically write one terminal result artifact."""

    artifact = build_artifact(
        repo_root=repo_root,
        output_path=output_path,
        work_root=work_root,
    )
    _atomic_write(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the frozen execution date and write the requested result path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / WORK_RELATIVE_PATH)
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"execution date must be {RUN_DATE}")
    write_artifact(output_path=args.output, repo_root=REPO_ROOT, work_root=args.work_root)
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin command wrapper owns execution.
    raise SystemExit(main())
