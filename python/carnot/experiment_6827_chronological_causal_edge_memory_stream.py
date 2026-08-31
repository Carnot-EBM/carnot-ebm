"""Build a frozen external-memory stream from authentic proposal evidence.

Spec refs: REQ-CL-6827, SCENARIO-CL-6827-PRECONDITIONS,
SCENARIO-CL-6827-OPERATIONS, SCENARIO-CL-6827-VISIBILITY,
SCENARIO-CL-6827-SNAPSHOTS, SCENARIO-CL-6827-ORDERS,
SCENARIO-CL-6827-COUNTERFACTUALS, SCENARIO-CL-6827-SERIALIZATION,
and SCENARIO-CL-6827-READINESS.

This module does not learn and does not invoke a model. It converts frozen
proposal rows into family-isolated transaction opportunities. Future outcomes
stay outside each decision snapshot, so a later learner can use this stream
without receiving its own acceptance labels as input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
from random import Random
import tempfile
import time
from types import MappingProxyType
from typing import Any


JsonDict = dict[str, Any]
RUN_DATE = "20260831"
SCHEMA = "carnot.experiment_6827.chronological_causal_edge_memory_stream.v1"
STATE_SCHEMA = "carnot.exp6827.fixed_capacity_memory.v1"
BLOCKED_STATUS = "complete_blocked_chronological_causal_edge_memory_stream"
COMPLETE_STATUS = "complete_chronological_causal_edge_memory_stream"
INFERENCE_SUBSTRATE = "CPU transformation of frozen authentic outputs, no LLM"
RESTORE_AUTHORITY = "sealed_harness"
CAPACITY = 2
EXPECTED_EVENT_COUNT = 288
EVENTS_PER_FAMILY = 96
ORDER_COUNT = 5

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6827_chronological_causal_edge_memory_stream.py"
)
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6827_chronological_causal_edge_memory_stream.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6827_chronological_causal_edge_memory_stream.json")
SOURCE_RELATIVE_PATHS = {
    "exp6812": Path("results/experiment_6812_sota_operational_handoff_corpus_v2.json"),
    "exp6826": Path("results/experiment_6826_selective_arbiter_sealed_adoption.json"),
}

MANDATED_FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
COMPONENT_FIELDS = (
    "hard_safety_decision",
    "safe_action_preservation_decision",
    "utility_decision",
    "certificate_truth_decision",
)
TERMINAL_COMPONENT_DECISIONS = frozenset({"pass", "fail", "insufficient"})
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
COUNTERFACTUAL_KINDS = ("remove", "substitute", "reorder")
RANDOM_SEEDS = MappingProxyType(
    {
        "chronology": 6_827_001,
        "split": 6_827_002,
        "counterfactual": 6_827_003,
    }
)
ORDER_SEEDS = tuple(RANDOM_SEEDS["chronology"] + index for index in range(ORDER_COUNT))

FEATURE_ALLOWLIST = (
    "event_id",
    "chronological_key",
    "scenario_family",
    "scenario_id",
    "arm",
    "raw_output_sha256",
    "candidate_ids",
    "candidate_hashes",
    "past_state_sha256",
)
FEATURE_DENYLIST = (
    "source_family",
    "model_id",
    "selected_action",
    "selected_candidate_id",
    "soft_score",
    "utility",
    "legal_support",
    "retention",
    "hard_case",
    "hard_violation_count",
    "later_read",
    "audit_decision",
    "deployment_adoption_decision",
    "future_outcome",
    "final_acceptance",
)

OPEN_SPEC_IDS = (
    "REQ-CL-6827",
    "SCENARIO-CL-6827-PRECONDITIONS",
    "SCENARIO-CL-6827-OPERATIONS",
    "SCENARIO-CL-6827-VISIBILITY",
    "SCENARIO-CL-6827-SNAPSHOTS",
    "SCENARIO-CL-6827-ORDERS",
    "SCENARIO-CL-6827-COUNTERFACTUALS",
    "SCENARIO-CL-6827-SERIALIZATION",
    "SCENARIO-CL-6827-READINESS",
)
REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6827_chronological_causal_edge_memory_stream.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_6827_chronological_causal_edge_memory_stream.py' -m pytest tests/python/test_experiment_6827_chronological_causal_edge_memory_stream.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6827_chronological_causal_edge_memory_stream.py scripts/experiments/experiment_6827_chronological_causal_edge_memory_stream.py tests/python/test_experiment_6827_chronological_causal_edge_memory_stream.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6827_chronological_causal_edge_memory_stream.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6827_chronological_causal_edge_memory_stream.json",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6827_chronological_causal_edge_memory_stream.json",
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
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "operation_schema",
    "local_receipt_schema",
    "capacity_contract",
    "split_manifest",
    "sealed_field_manifest",
    "order_hashes",
    "feature_allowlist",
    "feature_denylist",
    "causal_edge_schema",
    "counterfactual_manifest",
    "rows",
    "headroom_metrics",
    "admissible_operation_count",
    "rejected_operation_count",
    "later_read_opportunity_count",
    "verified_memory_stream_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents silent reinterpretation of stream bytes.",
    "experiment_id": "A stable identifier joins the stream to its task and result path.",
    "title": "A plain title states which frozen learning input this artifact supplies.",
    "run_date": "The fixed execution date prevents an unnoticed chronology change.",
    "status": "A terminal status separates a complete stream from a blocked input gate.",
    "openspec_requirement_ids": "Requirement links make each stream rule independently testable.",
    "replay_commands": "Exact commands let another operator repeat every verification layer.",
    "field_principles": "One reason per field keeps the artifact contract understandable.",
    "inference_substrate": "The substrate states that this CPU transform invokes no model.",
    "duration_s": "Measured wall time helps detect a skipped stream construction.",
    "random_seed": "Separate frozen seeds control orders, splits, and counterfactuals.",
    "reproducibility_checksum": "One digest binds sources, schema, rows, commands, and output.",
    "source_artifact_hashes": "Raw file hashes pin authentic proposals and sealed audit authority.",
    "operation_schema": "The public grammar limits every future learner to six typed operations.",
    "local_receipt_schema": "Exact receipts make each precondition and state effect replayable.",
    "capacity_contract": "Equal finite active capacity prevents one family from receiving more memory.",
    "split_manifest": "Frozen future, hard-case, and rotation splits prevent evaluation leakage.",
    "sealed_field_manifest": "The seal keeps outcomes and final acceptance outside learner features.",
    "order_hashes": "Five hashes bind every family-specific chronological order.",
    "feature_allowlist": "An allowlist limits decisions to current proposals and past state.",
    "feature_denylist": "A denylist names future, family-oracle, and outcome fields that stay hidden.",
    "causal_edge_schema": "Typed write-read-action-outcome identity makes edge tests auditable.",
    "counterfactual_manifest": "Remove, substitute, and reorder units calibrate each candidate edge.",
    "rows": "One row per required unit makes chronology and counterfactual coverage countable.",
    "headroom_metrics": "Positive stress counts prove the stream contains useful operation choices.",
    "admissible_operation_count": "A nonzero count proves the stream contains accepted operations.",
    "rejected_operation_count": "A nonzero count proves exact preconditions reject unsafe writes.",
    "later_read_opportunity_count": "A nonzero count supplies chronological evidence for causal edges.",
    "verified_memory_stream_ready": "This exact downstream gate depends on completeness, not effect sign.",
    "gate_check_summary": "Expected and observed values make every blocked or ready gate auditable.",
    "verifier_is_oracle": "False prevents a frozen stream builder from becoming acceptance authority.",
    "verdict_class": "A closed class preserves blocked and adverse outcomes without invention.",
    "honest_verdict": "A terminal row-supported sentence states readiness without a learning claim.",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return one compact, sorted, newline-terminated JSON representation."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the project SHA-256 form for exact byte identity."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value only after canonical serialization."""

    return sha256_bytes(canonical_json_bytes(value))


def _deep_freeze(value: Any) -> Any:
    """Copy JSON data into immutable containers so nested writes also fail."""

    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


class OperationKind(StrEnum):
    """Closed public grammar for bounded external memory."""

    ADD = "add"
    REVISE = "revise"
    SOFT_DELETE = "soft_delete"
    RETRIEVE = "retrieve"
    FILTER = "filter"
    RESTORE = "restore"


@dataclass(frozen=True)
class MemoryOperation:
    """One immutable operation proposal evaluated against exact parent bytes."""

    operation_id: str
    kind: OperationKind
    key: str
    value: JsonDict | None = None
    expected_revision: int | None = None
    authority: str = "stream_builder"

    def to_dict(self) -> JsonDict:
        """Expose all grammar fields so omitted values cannot change hash meaning."""

        return {
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "key": self.key,
            "value": deepcopy(self.value),
            "expected_revision": self.expected_revision,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class ReadOnlySnapshot:
    """Deeply immutable state bytes and records for one decision boundary."""

    state_bytes: bytes
    state_sha256: str
    version: int
    records: tuple[Any, ...]

    def retrieve(self, key: str, *, include_deleted: bool = False) -> Any | None:
        """Return one frozen record while hiding tombstones by default."""

        for record in self.records:
            if record["key"] == key and (include_deleted or not record["deleted"]):
                return record
        return None


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash every receipt field except the digest that contains that hash."""

    return sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})


class FixedCapacityMemory:
    """Deterministic family-scoped memory with finite active-record capacity."""

    def __init__(self, *, capacity: int, scope: str) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.scope = scope
        self.version = 0
        self._records: dict[str, JsonDict] = {}
        self._operation_ids: set[str] = set()

    def _state(self) -> JsonDict:
        """Sort records by key so insertion history cannot alter state bytes."""

        return {
            "schema": STATE_SCHEMA,
            "scope": self.scope,
            "capacity": self.capacity,
            "version": self.version,
            "records": [deepcopy(self._records[key]) for key in sorted(self._records)],
        }

    def state_bytes(self) -> bytes:
        """Return canonical bytes for the complete bounded active state."""

        return canonical_json_bytes(self._state())

    def snapshot(self) -> ReadOnlySnapshot:
        """Freeze exact bytes and deeply immutable record copies before a decision."""

        state = self._state()
        data = canonical_json_bytes(state)
        records = tuple(_deep_freeze(record) for record in state["records"])
        return ReadOnlySnapshot(data, sha256_bytes(data), self.version, records)

    def active_records(self) -> list[JsonDict]:
        """Return sorted active copies for deterministic operation scheduling."""

        return [
            deepcopy(self._records[key])
            for key in sorted(self._records)
            if not self._records[key]["deleted"]
        ]

    def deleted_records(self) -> list[JsonDict]:
        """Return sorted tombstone copies for deterministic restore tests."""

        return [
            deepcopy(self._records[key])
            for key in sorted(self._records)
            if self._records[key]["deleted"]
        ]

    def _active_count(self) -> int:
        return sum(not record["deleted"] for record in self._records.values())

    def _reject_reason(self, operation: MemoryOperation) -> str | None:
        """Evaluate the closed precondition order before any state mutation."""

        record = self._records.get(operation.key)
        if operation.operation_id in self._operation_ids:
            return "duplicate_operation"
        if operation.kind == OperationKind.ADD:
            if record is not None:
                return "key_conflict"
            if operation.value is None:
                return "value_required"
            if self._active_count() >= self.capacity:
                return "capacity_exceeded"
        elif operation.kind == OperationKind.REVISE:
            if record is None or record["deleted"]:
                return "active_record_required"
            if operation.expected_revision != record["revision"]:
                return "stale_revision"
            if operation.value is None:
                return "value_required"
        elif operation.kind == OperationKind.SOFT_DELETE:
            if record is None or record["deleted"]:
                return "active_record_required"
            if operation.expected_revision != record["revision"]:
                return "stale_revision"
        elif operation.kind == OperationKind.RESTORE:
            if operation.authority != RESTORE_AUTHORITY:
                return "restore_authority_required"
            if record is None or not record["deleted"]:
                return "deleted_record_required"
            if operation.expected_revision != record["revision"]:
                return "stale_revision"
            if self._active_count() >= self.capacity:
                return "capacity_exceeded"
        return None

    def apply(self, operation: MemoryOperation) -> JsonDict:
        """Apply one operation and bind its exact precondition, effect, and inverse."""

        parent_bytes = self.state_bytes()
        operation_dict = operation.to_dict()
        operation_bytes = canonical_json_bytes(operation_dict)
        reason = self._reject_reason(operation)
        accepted = reason is None
        effect: JsonDict = {"state_changed": False}
        inverse: JsonDict = {"kind": "none"}

        if accepted and operation.kind == OperationKind.ADD:
            self.version += 1
            self._records[operation.key] = {
                "key": operation.key,
                "value": deepcopy(operation.value),
                "revision": 1,
                "deleted": False,
                "created_operation_id": operation.operation_id,
                "last_operation_id": operation.operation_id,
            }
            effect = {"state_changed": True, "key": operation.key, "revision": 1}
            inverse = {"kind": "soft_delete", "key": operation.key, "expected_revision": 1}
        elif accepted and operation.kind == OperationKind.REVISE:
            record = self._records[operation.key]
            old_value = deepcopy(record["value"])
            record["value"] = deepcopy(operation.value)
            record["revision"] += 1
            record["last_operation_id"] = operation.operation_id
            self.version += 1
            effect = {
                "state_changed": True,
                "key": operation.key,
                "revision": record["revision"],
            }
            inverse = {
                "kind": "revise",
                "key": operation.key,
                "value": old_value,
                "expected_revision": record["revision"],
            }
        elif accepted and operation.kind == OperationKind.SOFT_DELETE:
            record = self._records[operation.key]
            record["deleted"] = True
            record["revision"] += 1
            record["last_operation_id"] = operation.operation_id
            self.version += 1
            effect = {
                "state_changed": True,
                "key": operation.key,
                "revision": record["revision"],
            }
            inverse = {
                "kind": "restore",
                "key": operation.key,
                "expected_revision": record["revision"],
                "authority": RESTORE_AUTHORITY,
            }
        elif accepted and operation.kind == OperationKind.RESTORE:
            record = self._records[operation.key]
            record["deleted"] = False
            record["revision"] += 1
            record["last_operation_id"] = operation.operation_id
            self.version += 1
            effect = {
                "state_changed": True,
                "key": operation.key,
                "revision": record["revision"],
            }
            inverse = {
                "kind": "soft_delete",
                "key": operation.key,
                "expected_revision": record["revision"],
            }
        elif accepted and operation.kind == OperationKind.RETRIEVE:
            stored = self._records.get(operation.key)
            record = stored if stored is not None and not stored["deleted"] else None
            effect = {
                "state_changed": False,
                "record": deepcopy(record) if record is not None else None,
            }
        elif accepted and operation.kind == OperationKind.FILTER:
            records = [
                record
                for record in self.active_records()
                if record["key"].startswith(operation.key)
            ]
            effect = {"state_changed": False, "records": records}

        if accepted:
            self._operation_ids.add(operation.operation_id)
        new_bytes = self.state_bytes()
        receipt: JsonDict = {
            "operation_id": operation.operation_id,
            "kind": operation.kind.value,
            "accepted": accepted,
            "reason": "preconditions_passed" if accepted else reason,
            "preconditions": {
                "evaluated_before_effect": True,
                "failure": reason,
            },
            "effect": effect,
            "inverse": inverse,
            "rollback_target": sha256_bytes(parent_bytes),
            "parent_state_bytes": parent_bytes.decode("utf-8"),
            "parent_state_sha256": sha256_bytes(parent_bytes),
            "operation_bytes": operation_bytes.decode("utf-8"),
            "operation_sha256": sha256_bytes(operation_bytes),
            "new_state_bytes": new_bytes.decode("utf-8"),
            "new_state_sha256": sha256_bytes(new_bytes),
        }
        receipt["receipt_sha256"] = receipt_sha256(receipt)
        return receipt


def source_paths_for_root(root: Path) -> dict[str, Path]:
    """Resolve source paths for a checkout without hard-coded write targets."""

    return {name: root / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Load both source artifacts while preserving readable failure evidence."""

    loaded: dict[str, JsonDict] = {}
    for name, path in paths.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            loaded[name] = value if isinstance(value, dict) else {"_load_error": "not_object"}
        except (OSError, json.JSONDecodeError) as error:
            loaded[name] = {"_load_error": type(error).__name__}
    return loaded


def _source_hashes(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Hash raw source files so parsed data cannot replace source identity."""

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
        result[name] = {"path": display_path, "sha256": digest}
    return result


def _group_source_rows(source: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Group the two candidate rows that came from one authentic raw output."""

    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    rows = source.get("rows", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict):
            grouped[str(row.get("cell_id", ""))].append(row)
    return dict(grouped)


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def check_preconditions(
    sources: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
    *,
    decision_snapshot_fields: Sequence[str] | None = None,
) -> JsonDict:
    """Check all frozen evidence before constructing any public stream row."""

    proposal = sources.get("exp6812", {})
    audit = sources.get("exp6826", {})
    readable_observed = {
        name: "unreadable" if source.get("_load_error") else "readable"
        for name, source in sorted(sources.items())
    }
    readable = set(sources) == set(SOURCE_RELATIVE_PATHS) and all(
        value == "readable" for value in readable_observed.values()
    )
    grouped = _group_source_rows(proposal)
    rows = proposal.get("rows", []) if isinstance(proposal.get("rows", []), list) else []
    family_counts = Counter(str(row.get("model_id")) for row in rows if isinstance(row, dict))
    raw_hash_cells = {
        cell_id: sorted({str(row.get("raw_output_sha256", "")) for row in cell_rows})
        for cell_id, cell_rows in grouped.items()
    }
    raw_hashes_valid = bool(
        len(raw_hash_cells) == EXPECTED_EVENT_COUNT
        and all(
            len(hashes) == 1
            and hashes[0].startswith("sha256:")
            and len(hashes[0]) == len("sha256:") + 64
            for hashes in raw_hash_cells.values()
        )
    )
    component_decisions = {field: audit.get(field) for field in COMPONENT_FIELDS}
    cell_ids = [str(row.get("cell_id", "")) for row in rows if isinstance(row, dict)]
    chronology_valid = bool(
        len(rows) == EXPECTED_EVENT_COUNT * 2
        and len(grouped) == EXPECTED_EVENT_COUNT
        and all(cell_ids)
        and all(len(cell_rows) == 2 for cell_rows in grouped.values())
    )
    alternatives = {
        cell_id: sorted(int(row.get("candidate_index", -1)) for row in cell_rows)
        for cell_id, cell_rows in grouped.items()
    }
    alternatives_valid = bool(
        len(alternatives) == EXPECTED_EVENT_COUNT
        and all(indices == [0, 1] for indices in alternatives.values())
    )
    snapshot_fields = tuple(decision_snapshot_fields or FEATURE_ALLOWLIST)
    future_fields = sorted(set(snapshot_fields).intersection(FEATURE_DENYLIST))
    checks = [
        _check(
            "source_artifacts_readable",
            {name: "readable" for name in sorted(SOURCE_RELATIVE_PATHS)},
            readable_observed,
            readable,
        ),
        _check(
            "source_artifact_hashes",
            "two readable sha256 identities",
            _source_hashes(paths),
            all(row["sha256"].startswith("sha256:") for row in _source_hashes(paths).values()),
        ),
        _check(
            "selective_arbiter_audit_complete",
            True,
            audit.get("selective_arbiter_audit_complete"),
            audit.get("selective_arbiter_audit_complete") is True,
        ),
        _check(
            "three_source_families",
            {family: EVENTS_PER_FAMILY * 2 for family in MANDATED_FAMILIES},
            dict(family_counts),
            family_counts
            == Counter({family: EVENTS_PER_FAMILY * 2 for family in MANDATED_FAMILIES}),
        ),
        _check(
            "raw_output_hashes",
            {"cell_count": EXPECTED_EVENT_COUNT, "hashes_per_cell": 1},
            {
                "cell_count": len(raw_hash_cells),
                "valid_cell_count": sum(
                    len(hashes) == 1 and bool(hashes[0]) for hashes in raw_hash_cells.values()
                ),
            },
            raw_hashes_valid,
        ),
        _check(
            "terminal_component_decisions",
            sorted(TERMINAL_COMPONENT_DECISIONS),
            component_decisions,
            all(value in TERMINAL_COMPONENT_DECISIONS for value in component_decisions.values()),
        ),
        _check(
            "chronological_keys",
            {"event_count": EXPECTED_EVENT_COUNT, "candidate_rows_per_event": 2},
            {"event_count": len(grouped), "source_row_count": len(rows)},
            chronology_valid,
        ),
        _check(
            "legal_alternatives",
            {"event_count": EXPECTED_EVENT_COUNT, "candidate_indices": [0, 1]},
            {
                "event_count": len(alternatives),
                "valid_event_count": sum(indices == [0, 1] for indices in alternatives.values()),
            },
            alternatives_valid,
        ),
        _check(
            "decision_snapshot_no_future_fields",
            [],
            future_fields,
            not future_fields,
        ),
    ]
    failed = [row["check"] for row in checks if row["passed"] is not True]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def build_events(source: Mapping[str, Any]) -> list[JsonDict]:
    """Convert each authentic two-candidate output into one immutable event."""

    events: list[JsonDict] = []
    for cell_id, cell_rows in sorted(_group_source_rows(source).items()):
        candidates = sorted(cell_rows, key=lambda row: int(row["candidate_index"]))
        first = candidates[0]
        family = str(first["model_id"])
        arm_rank = {"direct_typed": 0, "compressed_prose": 1}.get(str(first["arm"]), 2)
        chronological_key = f"{int(first['random_seed']):07d}:{arm_rank}:{cell_id}"
        candidate_ids = [
            str(row["candidate"]["candidate_id"])
            if isinstance(row.get("candidate"), dict)
            else f"candidate_{int(row['candidate_index'])}"
            for row in candidates
        ]
        candidate_hashes = [str(row["row_sha256"]) for row in candidates]
        hidden = {
            "selected_action": [row.get("selected_action") for row in candidates],
            "selected_candidate_id": [row.get("selected_candidate_id") for row in candidates],
            "soft_score": [row.get("soft_score") for row in candidates],
            "legal_support": [row.get("legal_support") for row in candidates],
            "hard_violation_count": [row.get("hard_violation_count") for row in candidates],
            "audit_decision": "sealed_exp6826_component_receipt",
        }
        events.append(
            {
                "event_id": cell_id,
                "chronological_key": chronological_key,
                "source_family": family,
                "scenario_family": str(first["scenario_family"]),
                "scenario_id": str(first["scenario_id"]),
                "arm": str(first["arm"]),
                "raw_output_sha256": str(first["raw_output_sha256"]),
                "candidate_ids": candidate_ids,
                "candidate_hashes": candidate_hashes,
                "legal_alternative_count": len(candidates) - 1,
                "proposal_value": {
                    "source_event_id": cell_id,
                    "candidate_sha256": candidate_hashes[0],
                    "alternative_sha256": candidate_hashes[1],
                    "raw_output_sha256": str(first["raw_output_sha256"]),
                },
                "hidden_outcome": hidden,
                "hidden_outcome_sha256": sha256_json(hidden),
            }
        )
    return sorted(events, key=lambda event: (event["source_family"], event["chronological_key"]))


def freeze_chronology(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze five orders, future splits, hard cases, and family rotations."""

    by_family = {
        family: sorted(
            [dict(event) for event in events if event["source_family"] == family],
            key=lambda event: str(event["chronological_key"]),
        )
        for family in MANDATED_FAMILIES
    }
    split_by_family: dict[str, JsonDict] = {}
    for family, family_events in by_family.items():
        development = family_events[:72]
        held_future = family_events[72:]
        hard_case = [
            event
            for event in family_events
            if event["scenario_family"] in {"stale_prerequisites", "competing_authorities"}
        ]
        split_by_family[family] = {
            "development": [event["event_id"] for event in development],
            "held_future": [event["event_id"] for event in held_future],
            "hard_case": [event["event_id"] for event in hard_case],
        }

    orders: list[JsonDict] = []
    order_hashes: dict[str, str] = {}
    for order_index, seed in enumerate(ORDER_SEEDS, start=1):
        order_id = f"order_{order_index}"
        event_ids_by_family: dict[str, list[str]] = {}
        for family_index, family in enumerate(MANDATED_FAMILIES):
            development_ids = list(split_by_family[family]["development"])
            held_ids = list(split_by_family[family]["held_future"])
            if order_index > 1:
                rng = Random(seed + family_index)
                rng.shuffle(development_ids)
                rng.shuffle(held_ids)
            event_ids_by_family[family] = [*development_ids, *held_ids]
        order_hash = sha256_json(
            {"order_id": order_id, "seed": seed, "event_ids_by_family": event_ids_by_family}
        )
        orders.append(
            {
                "order_id": order_id,
                "seed": seed,
                "event_ids_by_family": event_ids_by_family,
                "order_hash": order_hash,
            }
        )
        order_hashes[order_id] = order_hash

    rotations = [
        {
            "rotation_id": f"leave_out_{index + 1}",
            "held_out_family": held,
            "development_families": [family for family in MANDATED_FAMILIES if family != held],
        }
        for index, held in enumerate(MANDATED_FAMILIES)
    ]
    return {
        "orders": orders,
        "order_hashes": order_hashes,
        "split_manifest": {
            "seed": RANDOM_SEEDS["split"],
            "development_count_per_family": 72,
            "held_future_count_per_family": 24,
            "by_family": split_by_family,
            "rotations": rotations,
        },
        "rotations": rotations,
    }


def _operation_for_event(
    event: Mapping[str, Any], position: int, memory: FixedCapacityMemory
) -> MemoryOperation:
    """Choose a deterministic operation using only position and past memory."""

    phase = (position - 1) % 12
    operation_id = f"{memory.scope}::{position:03d}"
    new_key = f"mem:{event['event_id']}"
    active = memory.active_records()
    deleted = memory.deleted_records()
    active_key = active[0]["key"] if active else "mem:missing"
    deleted_key = deleted[-1]["key"] if deleted else "mem:missing"
    value = deepcopy(dict(event["proposal_value"]))

    if phase == 0:
        return MemoryOperation(operation_id, OperationKind.ADD, new_key, value=value)
    if phase == 1:
        if len(active) >= memory.capacity:
            return MemoryOperation(
                operation_id,
                OperationKind.SOFT_DELETE,
                active_key,
                expected_revision=active[0]["revision"],
            )
        return MemoryOperation(operation_id, OperationKind.ADD, new_key, value=value)
    if phase == 2:
        if deleted and len(active) < memory.capacity:
            return MemoryOperation(operation_id, OperationKind.ADD, new_key, value=value)
        return MemoryOperation(operation_id, OperationKind.RETRIEVE, active_key)
    if phase == 3:
        return MemoryOperation(operation_id, OperationKind.RETRIEVE, active_key)
    if phase == 4:
        return MemoryOperation(
            operation_id,
            OperationKind.REVISE,
            active_key,
            value=value,
            expected_revision=active[0]["revision"] if active else None,
        )
    if phase == 5:
        return MemoryOperation(operation_id, OperationKind.FILTER, "mem:")
    if phase == 6:
        return MemoryOperation(
            operation_id,
            OperationKind.SOFT_DELETE,
            active_key,
            expected_revision=active[0]["revision"] if active else None,
        )
    if phase == 7:
        return MemoryOperation(operation_id, OperationKind.RETRIEVE, deleted_key)
    if phase == 8:
        return MemoryOperation(
            operation_id,
            OperationKind.RESTORE,
            deleted_key,
            expected_revision=deleted[-1]["revision"] if deleted else None,
            authority=RESTORE_AUTHORITY,
        )
    if phase == 9:
        return MemoryOperation(operation_id, OperationKind.ADD, active_key, value=value)
    if phase == 10:
        return MemoryOperation(
            operation_id,
            OperationKind.REVISE,
            active_key,
            value=value,
            expected_revision=(active[0]["revision"] + 99) if active else 99,
        )
    return MemoryOperation(
        operation_id,
        OperationKind.RESTORE,
        deleted_key,
        expected_revision=deleted[-1]["revision"] if deleted else None,
        authority="learner",
    )


def _decision_features(event: Mapping[str, Any], state_sha256: str) -> JsonDict:
    """Build the complete public snapshot allowlist and no other input."""

    return {
        "event_id": event["event_id"],
        "chronological_key": event["chronological_key"],
        "scenario_family": event["scenario_family"],
        "scenario_id": event["scenario_id"],
        "arm": event["arm"],
        "raw_output_sha256": event["raw_output_sha256"],
        "candidate_ids": list(event["candidate_ids"]),
        "candidate_hashes": list(event["candidate_hashes"]),
        "past_state_sha256": state_sha256,
    }


def _edge_for_receipt(
    receipt: Mapping[str, Any],
    event: Mapping[str, Any],
    position: int,
    writers: Mapping[str, Mapping[str, Any]],
) -> JsonDict | None:
    """Create one write-read-action-outcome edge from a later visible read."""

    if receipt["kind"] == OperationKind.RETRIEVE.value:
        record = receipt["effect"].get("record")
        records = [record] if record is not None else []
    elif receipt["kind"] == OperationKind.FILTER.value:
        records = list(receipt["effect"].get("records", []))
    else:
        return None
    if not records:
        return None
    record = records[0]
    writer = writers.get(str(record["key"]))
    if writer is None or int(writer["position"]) >= position:
        return None
    material = {
        "write_operation_id": writer["operation_id"],
        "read_operation_id": receipt["operation_id"],
        "action_identity": sha256_json(receipt["effect"]),
        "outcome_identity": event["hidden_outcome_sha256"],
    }
    return {
        "edge_id": sha256_json(material),
        **material,
        "writer_position": writer["position"],
        "read_position": position,
    }


def _counterfactual_transform(
    kind: str, event: Mapping[str, Any], edge: Mapping[str, Any] | None
) -> JsonDict:
    """Describe one public transformation without exposing harness acceptance."""

    if edge is None:
        return {"operation": "safe_no_op", "reason": "no_write_read_edge"}
    if kind == "remove":
        return {"operation": "remove_writer", "expected_read": "absent"}
    if kind == "substitute":
        return {
            "operation": "substitute_writer",
            "alternative_sha256": event["candidate_hashes"][1],
        }
    return {
        "operation": "move_read_before_write",
        "writer_position": edge["writer_position"],
        "read_position": edge["read_position"],
    }


def _split_for_event(split_manifest: Mapping[str, Any], family: str, event_id: str) -> str:
    held = set(split_manifest["by_family"][family]["held_future"])
    return "held_future" if event_id in held else "development"


def _operation_schema() -> JsonDict:
    return {
        "version": "carnot.exp6827.operation.v1",
        "kinds": [kind.value for kind in OperationKind],
        "fields": [
            "operation_id",
            "kind",
            "key",
            "value",
            "expected_revision",
            "authority",
        ],
        "preconditions": {
            "add": "absent key, present value, and active count below capacity",
            "revise": "active key, exact revision, and present replacement value",
            "soft_delete": "active key and exact revision",
            "retrieve": "always legal and hides soft-deleted records",
            "filter": "always legal and returns active keys in canonical order",
            "restore": "deleted key, exact revision, sealed authority, and active headroom",
        },
        "deterministic_effect": True,
        "canonical_encoding": "sorted compact ASCII JSON with one trailing newline",
    }


def _receipt_schema() -> JsonDict:
    return {
        "version": "carnot.exp6827.local_receipt.v1",
        "fields": [
            "operation_id",
            "kind",
            "accepted",
            "reason",
            "preconditions",
            "effect",
            "inverse",
            "rollback_target",
            "parent_state_bytes",
            "parent_state_sha256",
            "operation_bytes",
            "operation_sha256",
            "new_state_bytes",
            "new_state_sha256",
            "receipt_sha256",
        ],
        "precondition_before_effect": True,
        "rejection_preserves_parent_bytes": True,
    }


def _edge_schema() -> JsonDict:
    return {
        "version": "carnot.exp6827.causal_edge.v1",
        "identity_fields": [
            "write_operation_id",
            "read_operation_id",
            "action_identity",
            "outcome_identity",
        ],
        "ownership": "one source family and one chronological order",
        "acceptance_location": "hidden_harness_view",
    }


def _artifact_base(
    *,
    run_date: str,
    duration_s: float,
    source_hashes: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Create a complete blocked shape before any stream row can exist."""

    return {
        "schema": SCHEMA,
        "experiment_id": "6827",
        "title": "Frozen Chronological Causal-Edge Memory Stream",
        "run_date": run_date,
        "status": BLOCKED_STATUS,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "operation_schema": _operation_schema(),
        "local_receipt_schema": _receipt_schema(),
        "capacity_contract": {
            "active_records_per_family_order": CAPACITY,
            "equal_capacity": True,
            "soft_deleted_visible": False,
        },
        "split_manifest": {},
        "sealed_field_manifest": {
            "sealed_until_after_decision": list(FEATURE_DENYLIST),
            "final_acceptance_location": "hidden_harness_view",
            "hidden_view_sha256": sha256_json([]),
        },
        "order_hashes": {},
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "causal_edge_schema": _edge_schema(),
        "counterfactual_manifest": {
            "seed": RANDOM_SEEDS["counterfactual"],
            "kinds": list(COUNTERFACTUAL_KINDS),
            "unit_count_by_kind": {kind: 0 for kind in COUNTERFACTUAL_KINDS},
            "applicable_count_by_kind": {kind: 0 for kind in COUNTERFACTUAL_KINDS},
        },
        "rows": [],
        "headroom_metrics": {},
        "admissible_operation_count": 0,
        "rejected_operation_count": 0,
        "later_read_opportunity_count": 0,
        "verified_memory_stream_ready": False,
        "gate_check_summary": deepcopy(dict(preconditions)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": f"{BLOCKED_STATUS}: one or more frozen input gates failed",
    }


def _build_rows(events: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> JsonDict:
    """Replay every isolated chronology and emit its three public counterfactual units."""

    event_by_id = {str(event["event_id"]): event for event in events}
    rows: list[JsonDict] = []
    hidden_acceptance: list[JsonDict] = []
    admitted = 0
    rejected = 0
    later_reads = 0
    safe_no_ops = 0
    conflicts = 0
    pressure = 0
    recoveries = 0
    canonical_receipts = 0

    for order in manifest["orders"]:
        order_id = str(order["order_id"])
        for family in MANDATED_FAMILIES:
            memory = FixedCapacityMemory(capacity=CAPACITY, scope=f"{order_id}::{family}")
            writers: dict[str, JsonDict] = {}
            pressure_pending = False
            pressure_released = False
            for position, event_id in enumerate(order["event_ids_by_family"][family], start=1):
                event = event_by_id[str(event_id)]
                snapshot = memory.snapshot()
                features = _decision_features(event, snapshot.state_sha256)
                decision_snapshot_sha256 = sha256_json(
                    {"features": features, "past_state_sha256": snapshot.state_sha256}
                )
                operation = _operation_for_event(event, position, memory)
                receipt = memory.apply(operation)
                if receipt["accepted"]:
                    admitted += 1
                else:
                    rejected += 1
                if receipt["receipt_sha256"] == receipt_sha256(receipt):
                    canonical_receipts += 1
                if receipt["reason"] in {"key_conflict", "stale_revision"}:
                    conflicts += 1
                if receipt["reason"] == "capacity_exceeded":
                    pressure += 1
                    pressure_pending = True
                if (
                    pressure_pending
                    and receipt["accepted"]
                    and receipt["kind"] == OperationKind.SOFT_DELETE.value
                ):
                    pressure_released = True
                if (
                    pressure_released
                    and receipt["accepted"]
                    and receipt["kind"] == OperationKind.ADD.value
                ):
                    recoveries += 1
                    pressure_pending = False
                    pressure_released = False
                if receipt["accepted"] and receipt["kind"] in {
                    OperationKind.RETRIEVE.value,
                    OperationKind.FILTER.value,
                }:
                    empty = receipt["effect"].get("record") is None and not receipt["effect"].get(
                        "records", []
                    )
                    safe_no_ops += int(empty)

                edge = _edge_for_receipt(receipt, event, position, writers)
                if edge is not None:
                    later_reads += 1
                if receipt["accepted"] and receipt["kind"] in {
                    OperationKind.ADD.value,
                    OperationKind.REVISE.value,
                    OperationKind.RESTORE.value,
                }:
                    writers[operation.key] = {
                        "operation_id": operation.operation_id,
                        "position": position,
                    }
                elif receipt["accepted"] and receipt["kind"] == OperationKind.SOFT_DELETE.value:
                    writers.pop(operation.key, None)

                split = _split_for_event(manifest["split_manifest"], family, str(event_id))
                for kind in COUNTERFACTUAL_KINDS:
                    transform = _counterfactual_transform(kind, event, edge)
                    row_id = f"{order_id}::{family}::{event_id}::{kind}"
                    row = {
                        "row_id": row_id,
                        "order_id": order_id,
                        "source_family": family,
                        "memory_scope": memory.scope,
                        "event_id": event_id,
                        "chronological_position": position,
                        "chronological_key": event["chronological_key"],
                        "split": split,
                        "counterfactual_kind": kind,
                        "counterfactual_applicable": edge is not None,
                        "counterfactual_transformation": transform,
                        "learner_proposed_test": f"test_{kind}_edge",
                        "operation_kind": receipt["kind"],
                        "operation_id": receipt["operation_id"],
                        "operation_admitted": receipt["accepted"],
                        "operation_reason": receipt["reason"],
                        "parent_state_sha256": receipt["parent_state_sha256"],
                        "new_state_sha256": receipt["new_state_sha256"],
                        "receipt_sha256": receipt["receipt_sha256"],
                        "decision_snapshot_sha256": decision_snapshot_sha256,
                        "decision_feature_keys": sorted(features),
                        "source_proposal_sha256": event["candidate_hashes"][0],
                        "legal_alternative_count": event["legal_alternative_count"],
                        "causal_edge_id": edge["edge_id"] if edge else None,
                        "write_operation_id": edge["write_operation_id"] if edge else None,
                        "read_operation_id": edge["read_operation_id"] if edge else None,
                        "action_identity": edge["action_identity"] if edge else None,
                        "outcome_identity": edge["outcome_identity"] if edge else None,
                    }
                    rows.append(row)
                    hidden_acceptance.append(
                        {
                            "row_id": row_id,
                            "accepted": edge is not None,
                            "outcome_sha256": event["hidden_outcome_sha256"],
                        }
                    )

    headroom = {
        "legal_alternative_count": sum(int(event["legal_alternative_count"]) for event in events),
        "safe_no_op_event_count": safe_no_ops,
        "conflict_event_count": conflicts,
        "capacity_pressure_case_count": pressure,
        "stale_pressure_recovery_count": recoveries,
        "canonical_receipt_count": canonical_receipts,
        "later_read_opportunity_count": later_reads,
    }
    return {
        "rows": rows,
        "hidden_acceptance_sha256": sha256_json(hidden_acceptance),
        "admissible": admitted,
        "rejected": rejected,
        "later_reads": later_reads,
        "headroom": headroom,
    }


def _ready_gates(
    *,
    preconditions: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> JsonDict:
    """Compute readiness from completeness and isolation, never adoption sign."""

    expected_rows = EXPECTED_EVENT_COUNT * ORDER_COUNT * len(COUNTERFACTUAL_KINDS)
    checks = [
        _check(
            "source_preconditions", True, preconditions["passed"], preconditions["passed"] is True
        ),
        _check(
            "complete_chronology",
            {"orders": ORDER_COUNT, "rows": expected_rows},
            {"orders": len(manifest["orders"]), "rows": len(rows)},
            len(manifest["orders"]) == ORDER_COUNT and len(rows) == expected_rows,
        ),
        _check(
            "canonical_rows",
            expected_rows,
            len({str(row["row_id"]) for row in rows}),
            len({str(row["row_id"]) for row in rows}) == expected_rows,
        ),
        _check(
            "sealed_fields",
            [],
            sorted(set(FEATURE_ALLOWLIST).intersection(FEATURE_DENYLIST)),
            set(FEATURE_ALLOWLIST).isdisjoint(FEATURE_DENYLIST)
            and all("final_acceptance" not in row for row in rows),
        ),
        _check(
            "family_isolation",
            True,
            all(
                str(row["memory_scope"]) == f"{row['order_id']}::{row['source_family']}"
                for row in rows
            ),
            all(
                str(row["memory_scope"]) == f"{row['order_id']}::{row['source_family']}"
                for row in rows
            ),
        ),
        _check(
            "positive_headroom",
            "all metrics > 0",
            dict(metrics["headroom"]),
            all(int(value) > 0 for value in metrics["headroom"].values()),
        ),
        _check(
            "operation_outcomes",
            "admissible, rejected, and later reads are nonzero",
            {
                "admissible": metrics["admissible"],
                "rejected": metrics["rejected"],
                "later_reads": metrics["later_reads"],
            },
            all(int(metrics[key]) > 0 for key in ("admissible", "rejected", "later_reads")),
        ),
    ]
    failed = [row["check"] for row in checks if row["passed"] is not True]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path],
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    decision_snapshot_fields: Sequence[str] | None = None,
) -> JsonDict:
    """Build a blocked artifact or the complete frozen chronological stream."""

    preconditions = check_preconditions(
        sources,
        source_paths,
        decision_snapshot_fields=decision_snapshot_fields,
    )
    artifact = _artifact_base(
        run_date=run_date,
        duration_s=duration_s,
        source_hashes=_source_hashes(source_paths),
        preconditions=preconditions,
    )
    if not preconditions["passed"]:
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    events = build_events(sources["exp6812"])
    manifest = freeze_chronology(events)
    built = _build_rows(events, manifest)
    rows = built["rows"]
    gates = _ready_gates(
        preconditions=preconditions,
        rows=rows,
        manifest=manifest,
        metrics=built,
    )
    ready = gates["passed"] is True
    applicable = sum(row["counterfactual_applicable"] for row in rows)
    artifact.update(
        {
            "status": COMPLETE_STATUS if ready else BLOCKED_STATUS,
            "split_manifest": manifest["split_manifest"],
            "sealed_field_manifest": {
                "sealed_until_after_decision": list(FEATURE_DENYLIST),
                "final_acceptance_location": "hidden_harness_view",
                "hidden_view_sha256": built["hidden_acceptance_sha256"],
            },
            "order_hashes": manifest["order_hashes"],
            "counterfactual_manifest": {
                "seed": RANDOM_SEEDS["counterfactual"],
                "kinds": list(COUNTERFACTUAL_KINDS),
                "unit_count_by_kind": {
                    kind: sum(row["counterfactual_kind"] == kind for row in rows)
                    for kind in COUNTERFACTUAL_KINDS
                },
                "applicable_count_by_kind": {
                    kind: sum(
                        row["counterfactual_kind"] == kind and row["counterfactual_applicable"]
                        for row in rows
                    )
                    for kind in COUNTERFACTUAL_KINDS
                },
                "applicable_unit_count": applicable,
            },
            "rows": rows,
            "headroom_metrics": built["headroom"],
            "admissible_operation_count": built["admissible"],
            "rejected_operation_count": built["rejected"],
            "later_read_opportunity_count": built["later_reads"],
            "verified_memory_stream_ready": ready,
            "gate_check_summary": gates,
            "verdict_class": "positive" if ready else "blocked",
            "honest_verdict": (
                "complete: frozen chronological causal-edge memory stream is ready; no learning ran"
                if ready
                else f"{BLOCKED_STATUS}: one or more stream readiness gates failed"
            ),
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind all deterministic output while excluding measured wall time and this digest."""

    material = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(material)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return closed schema and readiness errors without changing the artifact."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    ready = artifact.get("verified_memory_stream_ready") is True
    if ready and len(artifact.get("rows", [])) != (
        EXPECTED_EVENT_COUNT * ORDER_COUNT * len(COUNTERFACTUAL_KINDS)
    ):
        errors.append("ready row count mismatch")
    if ready and artifact.get("gate_check_summary", {}).get("failed_checks"):
        errors.append("ready artifact has failed gates")
    if not ready and artifact.get("rows"):
        errors.append("blocked artifact must not expose partial rows")
    if any("final_acceptance" in row for row in artifact.get("rows", [])):
        errors.append("hidden final acceptance exposed")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and publish one stable JSON file with an atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6827 or validate an existing task-owned artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return 0

    started = time.monotonic()
    paths = source_paths_for_root(REPO_ROOT)
    sources = load_sources(paths)
    artifact = build_artifact(
        sources,
        source_paths=paths,
        run_date=args.date,
        duration_s=args.duration_s or 0.0,
    )
    if args.duration_s is None:
        artifact["duration_s"] = round(time.monotonic() - started, 6)
    write_artifact(result_path, artifact)
    return 0
