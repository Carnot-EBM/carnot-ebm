"""Build the V633 delayed-feedback supersession stream.

This module creates a controlled CPU fixture. It does not run a learner or
measure accuracy. Separate immutable views let later experiments enforce the
three-event feedback delay without putting hidden truth in decision inputs.

Spec refs: REQ-CL-7183 and SCENARIO-CL-7183-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7183
SCHEMA = "carnot.exp7183.v633_supersession_stream.v1"
MILESTONE = "2026.09.633"
RUN_DATE = "20260910"
RANDOM_SEED = 7_183_202_609_10
POISON_SEED = 71_830_024
EXPECTED_EVENT_COUNT = 240
FEEDBACK_DELAY = 3
CORRUPTION_COUNT = 24
REGIME_SIZE = 60
REGIMES = (
    "stable",
    "superseded_rule",
    "recurrence",
    "conflicting_poisoned_feedback",
)
FAMILIES = (
    "lower_bound",
    "upper_bound",
    "parity_class",
    "modular_residue",
    "interval_band",
    "affine_balance",
)
ADAPTATION_FAMILIES = FAMILIES[:4]
TRANSFER_FAMILIES = FAMILIES[4:]
ARMS = ("no_memory", "static_rule", "fifo_replay", "revocable_template")
EXPECTED_ARM_ROW_COUNT = EXPECTED_EVENT_COUNT * len(ARMS)
INFERENCE_SUBSTRATE = (
    "deterministic CPU integer predicates with independent exact replay; no model or learner"
)
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

BYTE_BUDGETS: JsonDict = {
    "decision_context_bytes": 4_096,
    "charged_memory_bytes": 4_096,
    "record_inspection_slots": 2,
    "template_limit": 8,
}
TEMPLATE_OPERATION_CONTRACT: JsonDict = {
    "grammar_version": "revocable_constraint_template.v1",
    "allowed_operations": ["add_template", "replace_template", "revoke_template"],
    "required_fields": ["operation", "family_id", "source_version", "evidence_event_ids"],
    "max_templates": 8,
    "requires_released_feedback": True,
    "allows_weight_updates": False,
}

DEFAULT_DECISION_PATH = Path("results/streams/experiment_7183_v633_decision_view.jsonl")
DEFAULT_FEEDBACK_PATH = Path("results/streams/experiment_7183_v633_feedback_view.jsonl")
DEFAULT_TRUTH_PATH = Path("results/streams/experiment_7183_v633_evaluator_truth.jsonl")
DEFAULT_AVAILABILITY_PATH = Path("results/streams/experiment_7183_v633_availability_matrix.jsonl")
DEFAULT_MANIFEST_DIR = Path("results/streams/experiment_7183_v633_manifests")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7183_v633_supersession_stream.json")

SOURCE_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7105_v623_exact_constraint_stream.py"),
    Path("python/carnot/experiment_7106_v623_procedural_memory_csl.py"),
    Path("python/carnot/experiment_5616_exact_nonstationary_constraint_stream.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/experiment_7183_v633_supersession_stream.py"),
    Path("scripts/experiments/experiment_7183_v633_supersession_stream.py"),
    Path("openspec/capabilities/continuous-learning/spec.md"),
    Path("research-roadmap.yaml"),
)
MANIFEST_NAMES = (
    "family",
    "regime",
    "heldout",
    "operation",
    "arm",
    "budget",
    "feedback",
    "recurrence",
)
FORBIDDEN_DECISION_FIELDS = {
    "label",
    "exact_label",
    "observed_label",
    "current_label",
    "future_label",
    "regime",
    "regime_id",
    "source_version",
    "supersession",
    "supersession_flag",
    "revocation",
    "revocation_receipt",
    "revocation_receipt_id",
    "feedback",
    "future_feedback",
    "evaluator_truth",
    "truth",
    "feedback_corrupted",
    "corruption_status",
    "witness",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "inference_substrate_class",
    "stream_ready_score",
    "decision_view_path",
    "feedback_view_path",
    "evaluator_truth_path",
    "availability_matrix_path",
    "sealed_view_paths",
    "sealed_view_hashes",
    "manifest_paths",
    "manifest_hashes",
    "feedback_schedule",
    "heldout_manifest",
    "regime_rows",
    "template_operation_contract",
    "family_manifest",
    "regime_manifest",
    "arm_manifest",
    "byte_budgets",
    "event_rows",
    "availability_rows",
    "exact_agreement_rows",
    "corruption_witness_rows",
    "revocation_rows",
    "recurrence_rows",
    "transfer_rows",
    "leakage_rows",
    "arm_materialization_rows",
    "learning_executed",
    "accuracy_claimed",
)

_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed experiment ID prevents another task from supplying this stream.",
    "milestone": "The milestone binds the stream to the V633 frozen contract.",
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": "Name each resource and record its actual availability before measurement.",
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": "Hash input contracts, code, seeds, and raw rows to expose drift.",
    "gate_check_summary": "Every blocked verdict names the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Declare whether the scored verifier uses the same authority that labels the outcome.",
    "verdict_class": "Use positive, circular_positive, null, blocked, disqualified, or partial.",
    "honest_verdict": "A terminal description distinguishes useful evidence, null findings, disqualification, and external blocks.",
    "inference_substrate_class": "Use the closed CPU class only after the declared exact work runs.",
    "stream_ready_score": "One requires a complete, sealed, causally ordered stream.",
    "decision_view_path": "The controller can read only information available at decision time.",
    "feedback_view_path": "A separate path lets the scheduler release feedback after decisions.",
    "evaluator_truth_path": "A sealed truth sidecar preserves independent audit authority.",
    "availability_matrix_path": "The matrix makes every chronological release decision replayable.",
    "sealed_view_paths": "Named view paths keep decision, feedback, truth, and timing bytes separate.",
    "sealed_view_hashes": "View hashes detect any post-construction change.",
    "manifest_paths": "Each frozen contract has a separately reviewable immutable path.",
    "manifest_hashes": "Manifest hashes prevent outcome-driven contract changes.",
    "feedback_schedule": "Release times prevent future-label leakage.",
    "heldout_manifest": "Evaluation labels cannot select updates.",
    "regime_rows": "Shift and recurrence remain visible to the evaluator only.",
    "template_operation_contract": "A fixed operation grammar bounds the learned structural changes.",
    "family_manifest": "The family split prevents transfer labels from becoming adaptation support.",
    "regime_manifest": "Fixed chronological regimes prevent outcome-driven reorderings.",
    "arm_manifest": "Frozen arms support a later matched comparison without changing this stream.",
    "byte_budgets": "Equal declared charges prevent resource differences from explaining later results.",
    "event_rows": "Complete event projections bind each separated view to one event identity.",
    "availability_rows": "Per-decision rows expose which released records were eligible for each use.",
    "exact_agreement_rows": "Two exact calculations must agree before truth can be sealed.",
    "corruption_witness_rows": "Contradictory witnesses make every poisoned label independently auditable.",
    "revocation_rows": "Signed receipts make source replacement explicit and chronological.",
    "recurrence_rows": "Frozen recurrence links preserve later forgetting measurements.",
    "transfer_rows": "All held-out family and regime cells remain visible to the evaluator.",
    "leakage_rows": "Field-level checks prove hidden truth is absent from decision inputs.",
    "arm_materialization_rows": "One row per event and arm proves matched stream materialization.",
    "learning_executed": "False prevents fixture construction from becoming a learning claim.",
    "accuracy_claimed": "False prevents exact label agreement from becoming a prediction claim.",
}
FIELD_PRINCIPLES = {field: _PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS}


class ImmutableSealError(RuntimeError):
    """Report an attempt to replace a sealed file with different bytes."""


@dataclass(frozen=True)
class StreamPaths:
    """Keep every output path explicit for isolated tests and the public command."""

    decisions: Path
    feedback: Path
    truth: Path
    availability: Path
    manifest_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> StreamPaths:
        """Return the repository-relative paths required by the task."""

        return cls(
            DEFAULT_DECISION_PATH,
            DEFAULT_FEEDBACK_PATH,
            DEFAULT_TRUTH_PATH,
            DEFAULT_AVAILABILITY_PATH,
            DEFAULT_MANIFEST_DIR,
            DEFAULT_ARTIFACT_PATH,
        )

    @classmethod
    def under(cls, root: Path) -> StreamPaths:
        """Put all task outputs under one private root."""

        return cls(
            root / "decision_view.jsonl",
            root / "feedback_view.jsonl",
            root / "evaluator_truth.jsonl",
            root / "availability_matrix.jsonl",
            root / "manifests",
            root / "experiment_7183_v633_supersession_stream.json",
        )

    def manifest_path(self, name: str) -> Path:
        """Return one manifest path from the frozen manifest roster."""

        return self.manifest_dir / f"{name}.json"


@dataclass(frozen=True)
class StreamViews:
    """Hold separated in-memory views before immutable bytes are written."""

    decisions: list[JsonDict]
    feedback: list[JsonDict]
    truths: list[JsonDict]
    events: list[JsonDict]
    availability: list[JsonDict]
    revocations: list[JsonDict]
    corruption_witnesses: list[JsonDict]
    exact_agreements: list[JsonDict]
    recurrence: list[JsonDict]
    transfer: list[JsonDict]
    arm_rows: list[JsonDict]
    manifests: JsonDict


def canonical_json(value: Any) -> bytes:
    """Serialize values once so equal evidence always has equal bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Prefix SHA-256 values so they cannot be mistaken for raw labels."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash structured evidence with the canonical JSON encoding."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash one readable source while leaving a missing source explicit."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode one canonical object per line for immutable stream views."""

    return b"".join(canonical_json(row) + b"\n" for row in rows)


def _immutable_write(path: Path, payload: bytes) -> str:
    """Create a seal once and accept later writes only when bytes match."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ImmutableSealError(f"immutable_seal_mismatch:{path}") from None
    return sha256_bytes(payload)


def write_immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Seal a chronological row view with canonical line bytes."""

    return _immutable_write(path, _jsonl_bytes(rows))


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    """Seal one manifest without formatting-dependent whitespace."""

    return _immutable_write(path, canonical_json(value) + b"\n")


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish the terminal artifact only after its complete bytes exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
) -> JsonDict:
    """Record both sides and ownership of one exact preflight comparison."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first failed comparison without dropping later checks."""

    copied = [dict(row) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": copied,
        "failed_check": None if failed is None else failed.get("check"),
        "upstream": None if failed is None else failed.get("upstream"),
        "field": None if failed is None else failed.get("field"),
        "expected_value": None if failed is None else deepcopy(failed.get("expected_value")),
        "observed_value": None if failed is None else deepcopy(failed.get("observed_value")),
    }


def _rule_for(family: str, version: str) -> JsonDict:
    """Return one frozen integer rule, including recurrent V1 semantics in V3."""

    recurrent = version in {"v1", "v3"}
    rules: dict[str, tuple[JsonDict, JsonDict]] = {
        "lower_bound": ({"modulus": 20, "minimum": 10}, {"modulus": 20, "minimum": 14}),
        "upper_bound": ({"modulus": 20, "maximum": 9}, {"modulus": 20, "maximum": 5}),
        "parity_class": ({"parity": 0}, {"parity": 1}),
        "modular_residue": ({"modulus": 5, "residues": [0, 1]}, {"modulus": 5, "residues": [3, 4]}),
        "interval_band": (
            {"modulus": 30, "lower": 5, "upper": 20},
            {"modulus": 30, "lower": 10, "upper": 25},
        ),
        "affine_balance": (
            {"multiplier": 3, "offset": 1, "modulus": 7, "maximum": 2},
            {"multiplier": 3, "offset": 4, "modulus": 7, "maximum": 1},
        ),
    }
    stable, changed = rules[family]
    return {
        "rule_id": f"{family}:{version}",
        "family_id": family,
        "source_version": version,
        "parameters": deepcopy(stable if recurrent else changed),
        "recurrent_of": f"{family}:v1" if version == "v3" else None,
    }


def _rule_version(regime: str) -> str:
    """Map public regime order to hidden source versions."""

    return "v1" if regime == "stable" else "v2" if regime == "superseded_rule" else "v3"


def exact_label(family: str, numeric_value: int, rule: Mapping[str, Any]) -> str:
    """Evaluate one bounded integer predicate with no tolerance or model call."""

    parameters = rule["parameters"]
    if family == "lower_bound":
        valid = numeric_value % parameters["modulus"] >= parameters["minimum"]
    elif family == "upper_bound":
        valid = numeric_value % parameters["modulus"] <= parameters["maximum"]
    elif family == "parity_class":
        valid = numeric_value % 2 == parameters["parity"]
    elif family == "modular_residue":
        valid = numeric_value % parameters["modulus"] in parameters["residues"]
    elif family == "interval_band":
        remainder = numeric_value % parameters["modulus"]
        valid = parameters["lower"] <= remainder <= parameters["upper"]
    else:
        remainder = (parameters["multiplier"] * numeric_value + parameters["offset"]) % parameters[
            "modulus"
        ]
        valid = remainder <= parameters["maximum"]
    return "accept" if valid else "reject"


def independent_exact_label(family: str, numeric_value: int, rule: Mapping[str, Any]) -> str:
    """Recompute the label with alternate equivalent integer expressions."""

    parameters = rule["parameters"]
    if family == "lower_bound":
        valid = not numeric_value % parameters["modulus"] < parameters["minimum"]
    elif family == "upper_bound":
        valid = not numeric_value % parameters["modulus"] > parameters["maximum"]
    elif family == "parity_class":
        valid = (numeric_value & 1) == parameters["parity"]
    elif family == "modular_residue":
        remainder = numeric_value % parameters["modulus"]
        valid = any(remainder == candidate for candidate in parameters["residues"])
    elif family == "interval_band":
        remainder = numeric_value % parameters["modulus"]
        valid = not (remainder < parameters["lower"] or remainder > parameters["upper"])
    else:
        _, remainder = divmod(
            parameters["multiplier"] * numeric_value + parameters["offset"],
            parameters["modulus"],
        )
        valid = not remainder > parameters["maximum"]
    return "accept" if valid else "reject"


def _selection_role(family: str, decision_index: int) -> str:
    """Freeze support, rolling validation, transfer, and audit ownership."""

    if family in TRANSFER_FAMILIES:
        return "transfer_evaluation"
    if decision_index >= 180:
        return "final_audit"
    family_occurrence = decision_index // len(FAMILIES)
    return "rolling_validation" if family_occurrence % 5 == 4 else "commit_support"


def _numeric_value(decision_index: int, family_index: int) -> int:
    """Give every new entity a deterministic integer input."""

    return 10_000 + decision_index * 37 + family_index * 11


def _poison_indices() -> set[int]:
    """Select exactly 10 percent of all events inside the frozen poison regime."""

    return set(random.Random(POISON_SEED).sample(range(180, 240), CORRUPTION_COUNT))


def _rule_semantics(rule: Mapping[str, Any]) -> JsonDict:
    """Project rule behavior without its source-version identity."""

    return {"family_id": rule["family_id"], "parameters": deepcopy(rule["parameters"])}


def _revocation_receipt(
    family: str,
    revoked: str,
    replacement: str,
    trigger_event_id: str,
    trigger_index: int,
) -> JsonDict:
    """Sign one source replacement for release with its triggering feedback."""

    body: JsonDict = {
        "receipt_id": f"revoke:{family}:{revoked}:{replacement}",
        "family_id": family,
        "revoked_version": revoked,
        "replacement_version": replacement,
        "trigger_event_id": trigger_event_id,
        "trigger_decision_index": trigger_index,
        "release_index": trigger_index + FEEDBACK_DELAY,
        "operation": "revoke_template",
        "authority": "independent_exact_evaluator",
    }
    body["receipt_hash"] = sha256_json(body)
    return body


def verify_revocation_receipt(row: Mapping[str, Any]) -> bool:
    """Verify that a revocation signature owns the complete receipt body."""

    body = {key: deepcopy(value) for key, value in row.items() if key != "receipt_hash"}
    return row.get("receipt_hash") == sha256_json(body)


def replay_truth(row: Mapping[str, Any]) -> bool:
    """Replay both exact calculations from one evaluator-truth row."""

    first = exact_label(str(row["family_id"]), int(row["numeric_value"]), row["rule"])
    second = independent_exact_label(str(row["family_id"]), int(row["numeric_value"]), row["rule"])
    return first == second == row.get("exact_label")


def replay_corruption_witness(row: Mapping[str, Any]) -> bool:
    """Confirm that one poisoned observation exactly contradicts replayed truth."""

    exact = exact_label(str(row["family_id"]), int(row["numeric_value"]), row["rule"])
    return (
        exact == row.get("exact_label")
        and row.get("observed_feedback_label") != exact
        and row.get("contradiction_confirmed") is True
    )


def _nested_keys(value: Any) -> set[str]:
    """Collect nested keys so hidden fields cannot evade a top-level check."""

    if isinstance(value, Mapping):
        return set(value) | {key for item in value.values() for key in _nested_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def leakage_errors(decisions: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name each decision row that contains evaluator-only information."""

    errors = []
    for index, row in enumerate(decisions):
        forbidden = sorted(_nested_keys(row) & FORBIDDEN_DECISION_FIELDS)
        if forbidden:
            errors.append(f"decision_row_{index}:{','.join(forbidden)}")
    return errors


def _manifests(
    truths: Sequence[Mapping[str, Any]], recurrence: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build all contracts before any later learner can select an update."""

    validation: JsonDict = {}
    support: JsonDict = {}
    for family in ADAPTATION_FAMILIES:
        validation[family] = [
            row["event_id"]
            for row in truths
            if row["family_id"] == family and row["selection_role"] == "rolling_validation"
        ]
        support[family] = [
            row["event_id"]
            for row in truths
            if row["family_id"] == family and row["selection_role"] == "commit_support"
        ]
    return {
        "family": {
            "families": list(FAMILIES),
            "adaptation_families": list(ADAPTATION_FAMILIES),
            "transfer_families": list(TRANSFER_FAMILIES),
            "whole_family_split": True,
        },
        "regime": {
            "regime_order": list(REGIMES),
            "rows_per_regime": REGIME_SIZE,
            "boundaries": [
                {"regime_id": regime, "start": index * 60, "stop": (index + 1) * 60}
                for index, regime in enumerate(REGIMES)
            ],
        },
        "heldout": {
            "rolling_validation_event_ids": validation,
            "commit_support_event_ids": support,
            "transfer_families": list(TRANSFER_FAMILIES),
            "transfer_labels_available_for_commit_selection": False,
            "final_audit_start_index": 180,
            "final_audit_stop_index": 240,
            "final_audit_labels_available_for_commit_selection": False,
            "partitions_disjoint": True,
        },
        "operation": deepcopy(TEMPLATE_OPERATION_CONTRACT),
        "arm": {
            "arms": list(ARMS),
            "treatment_field": "arm",
            "identical_stream_required": True,
        },
        "budget": deepcopy(BYTE_BUDGETS),
        "feedback": {
            "delay_events": FEEDBACK_DELAY,
            "release_rule": "release_index=decision_index+3",
            "feedback_count": EXPECTED_EVENT_COUNT,
            "corruption_seed": POISON_SEED,
            "corruption_count": CORRUPTION_COUNT,
        },
        "recurrence": {
            "subset_event_ids": [row["recurrence_event_id"] for row in recurrence],
            "stable_anchor_event_ids": [row["stable_event_id"] for row in recurrence],
            "families": list(FAMILIES),
            "rows_per_family": 2,
        },
    }


def build_stream_views() -> StreamViews:
    """Construct all frozen views without running a policy or learning update."""

    poison_indices = _poison_indices()
    decisions: list[JsonDict] = []
    feedback: list[JsonDict] = []
    truths: list[JsonDict] = []
    events: list[JsonDict] = []
    revocations: list[JsonDict] = []
    corruption_witnesses: list[JsonDict] = []
    exact_agreements: list[JsonDict] = []

    for decision_index in range(EXPECTED_EVENT_COUNT):
        regime = REGIMES[decision_index // REGIME_SIZE]
        family_index = decision_index % len(FAMILIES)
        family = FAMILIES[family_index]
        source_version = _rule_version(regime)
        rule = _rule_for(family, source_version)
        value = _numeric_value(decision_index, family_index)
        event_id = f"exp7183-event-{decision_index:03d}"
        primary_label = exact_label(family, value, rule)
        independent_label = independent_exact_label(family, value, rule)
        corrupted = decision_index in poison_indices
        observed_label = (
            "reject" if primary_label == "accept" else "accept" if corrupted else primary_label
        )
        if not corrupted:
            observed_label = primary_label
        role = _selection_role(family, decision_index)
        decision: JsonDict = {
            "event_id": event_id,
            "decision_index": decision_index,
            "family_id": family,
            "entity_id": f"numeric-entity-{decision_index:03d}",
            "numeric_value": value,
            "candidate_actions": ["accept", "reject"],
            "observable_source_id": f"source:{family}",
        }
        feedback_row: JsonDict = {
            "event_id": event_id,
            "decision_index": decision_index,
            "release_index": decision_index + FEEDBACK_DELAY,
            "observed_label": observed_label,
            "source_version": source_version,
            "feedback_authority": "delayed_source_report",
        }
        truth: JsonDict = {
            "event_id": event_id,
            "decision_index": decision_index,
            "feedback_release_index": decision_index + FEEDBACK_DELAY,
            "family_id": family,
            "family_role": "adaptation" if family in ADAPTATION_FAMILIES else "transfer",
            "selection_role": role,
            "entity_id": decision["entity_id"],
            "numeric_value": value,
            "regime_id": regime,
            "source_version": source_version,
            "rule": rule,
            "exact_label": primary_label,
            "independent_exact_label": independent_label,
            "feedback_corrupted": corrupted,
            "exact_witness": {
                "operation": family,
                "parameters": deepcopy(rule["parameters"]),
                "numeric_value": value,
                "exact_label": primary_label,
            },
        }
        decision_hash = sha256_json(decision)
        feedback_hash = sha256_json(feedback_row)
        truth_hash = sha256_json(truth)
        decisions.append(decision)
        feedback.append(feedback_row)
        truths.append(truth)
        events.append(
            {
                "event_id": event_id,
                "chronology_index": decision_index,
                "decision": deepcopy(decision),
                "feedback": deepcopy(feedback_row),
                "truth": deepcopy(truth),
                "decision_hash": decision_hash,
                "feedback_hash": feedback_hash,
                "truth_hash": truth_hash,
            }
        )
        exact_agreements.append(
            {
                "event_id": event_id,
                "primary_label": primary_label,
                "independent_label": independent_label,
                "agreement": primary_label == independent_label,
            }
        )
        if corrupted:
            corruption_witnesses.append(
                {
                    "event_id": event_id,
                    "family_id": family,
                    "numeric_value": value,
                    "rule": deepcopy(rule),
                    "exact_label": primary_label,
                    "observed_feedback_label": observed_label,
                    "contradiction_confirmed": observed_label != primary_label,
                }
            )
        regime_local_index = decision_index % REGIME_SIZE
        if regime_local_index < len(FAMILIES) and regime in {"superseded_rule", "recurrence"}:
            revoked, replacement = ("v1", "v2") if regime == "superseded_rule" else ("v2", "v3")
            receipt = _revocation_receipt(family, revoked, replacement, event_id, decision_index)
            revocations.append(receipt)
            feedback_row["revocation_receipt_id"] = receipt["receipt_id"]
            events[-1]["feedback"] = deepcopy(feedback_row)
            events[-1]["feedback_hash"] = sha256_json(feedback_row)

    recurrence: list[JsonDict] = []
    for family_index, family in enumerate(FAMILIES):
        for offset in (0, 1):
            stable = truths[family_index + offset * len(FAMILIES)]
            recurrent = truths[120 + family_index + offset * len(FAMILIES)]
            recurrence.append(
                {
                    "family_id": family,
                    "stable_event_id": stable["event_id"],
                    "recurrence_event_id": recurrent["event_id"],
                    "stable_rule_id": stable["rule"]["rule_id"],
                    "recurrence_rule_id": recurrent["rule"]["rule_id"],
                    "recurrent_rule_matches_stable_rule": _rule_semantics(stable["rule"])
                    == _rule_semantics(recurrent["rule"]),
                }
            )
    transfer = [
        {
            "event_id": row["event_id"],
            "family_id": row["family_id"],
            "regime_id": row["regime_id"],
            "source_version": row["source_version"],
            "exact_label": row["exact_label"],
        }
        for row in truths
        if row["family_role"] == "transfer"
    ]
    manifests = _manifests(truths, recurrence)
    availability: list[JsonDict] = []
    for decision_index in range(EXPECTED_EVENT_COUNT):
        released = [row for row in feedback if row["release_index"] <= decision_index]
        released_ids = [row["event_id"] for row in released]
        truth_by_id = {row["event_id"]: row for row in truths}
        availability.append(
            {
                "decision_index": decision_index,
                "event_id": decisions[decision_index]["event_id"],
                "newly_released_event_ids": [
                    row["event_id"] for row in feedback if row["release_index"] == decision_index
                ],
                "released_feedback_event_ids": released_ids,
                "released_feedback_hash": sha256_json(released),
                "commit_support_event_ids": [
                    event_id
                    for event_id in released_ids
                    if truth_by_id[event_id]["selection_role"] == "commit_support"
                ],
                "rolling_validation_event_ids": [
                    event_id
                    for event_id in released_ids
                    if truth_by_id[event_id]["selection_role"] == "rolling_validation"
                ],
                "available_revocation_receipt_ids": [
                    row["receipt_id"]
                    for row in revocations
                    if row["release_index"] <= decision_index
                ],
            }
        )
    arm_rows: list[JsonDict] = []
    for event in events:
        decision = event["decision"]
        for arm in ARMS:
            arm_rows.append(
                {
                    "event_id": event["event_id"],
                    "decision_index": event["chronology_index"],
                    "decision_hash": event["decision_hash"],
                    "feedback_hash": event["feedback_hash"],
                    "feedback_release_index": event["feedback"]["release_index"],
                    "decision_context_actual_bytes": len(canonical_json(decision)),
                    "decision_context_budget_bytes": BYTE_BUDGETS["decision_context_bytes"],
                    "charged_memory_bytes": BYTE_BUDGETS["charged_memory_bytes"],
                    "record_inspection_slots": BYTE_BUDGETS["record_inspection_slots"],
                    "template_limit": BYTE_BUDGETS["template_limit"],
                    "arm": arm,
                }
            )
    return StreamViews(
        decisions,
        feedback,
        truths,
        events,
        availability,
        revocations,
        corruption_witnesses,
        exact_agreements,
        recurrence,
        transfer,
        arm_rows,
        manifests,
    )


def stream_conformance_errors(views: StreamViews) -> list[str]:
    """Return stable failures for every readiness condition."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(len(views.events) != EXPECTED_EVENT_COUNT, "event_count")
    add(
        [row.get("chronology_index") for row in views.events] != list(range(EXPECTED_EVENT_COUNT)),
        "chronology",
    )
    add(len({row.get("event_id") for row in views.events}) != EXPECTED_EVENT_COUNT, "identity")
    add(bool(leakage_errors(views.decisions)), "decision_leakage")
    add(
        any(
            row.get("release_index") != row.get("decision_index", -FEEDBACK_DELAY) + FEEDBACK_DELAY
            for row in views.feedback
        ),
        "feedback_delay",
    )
    add(
        [sum(row.get("regime_id") == regime for row in views.truths) for regime in REGIMES]
        != [REGIME_SIZE] * len(REGIMES),
        "regime_coverage",
    )
    add({row.get("family_id") for row in views.truths} != set(FAMILIES), "family_coverage")
    add(any(not replay_truth(row) for row in views.truths), "exact_label_agreement")
    add(
        len(views.corruption_witnesses) != CORRUPTION_COUNT
        or any(not replay_corruption_witness(row) for row in views.corruption_witnesses),
        "corruption_witnesses",
    )
    add(
        len(views.revocations) != 12
        or any(not verify_revocation_receipt(row) for row in views.revocations),
        "revocation_receipts",
    )
    add(
        len(views.recurrence) != 12
        or {row.get("family_id") for row in views.recurrence} != set(FAMILIES)
        or any(
            row.get("recurrent_rule_matches_stable_rule") is not True for row in views.recurrence
        ),
        "recurrence_cells",
    )
    transfer_cells = {(row.get("family_id"), row.get("regime_id")) for row in views.transfer}
    add(
        len(views.transfer) != 80
        or transfer_cells
        != {(family, regime) for family in TRANSFER_FAMILIES for regime in REGIMES},
        "transfer_cells",
    )
    truth_by_id = {row["event_id"]: row for row in views.truths}
    for row in views.availability:
        index = row["decision_index"]
        expected_released = [
            feedback["event_id"]
            for feedback in views.feedback
            if feedback["release_index"] <= index
        ]
        add(row.get("released_feedback_event_ids") != expected_released, "availability_matrix")
        eligible = row.get("commit_support_event_ids", []) + row.get(
            "rolling_validation_event_ids", []
        )
        add(
            any(
                truth_by_id[event_id]["family_role"] != "adaptation"
                or truth_by_id[event_id]["decision_index"] >= 180
                or truth_by_id[event_id]["feedback_release_index"] > index
                for event_id in eligible
            ),
            "heldout_selection",
        )
        add(
            bool(
                set(row.get("commit_support_event_ids", []))
                & set(row.get("rolling_validation_event_ids", []))
            ),
            "validation_support_overlap",
        )
    add(len(views.arm_rows) != EXPECTED_ARM_ROW_COUNT, "arm_row_count")
    for event in views.events:
        arm_rows = [row for row in views.arm_rows if row.get("event_id") == event["event_id"]]
        projections = [
            {key: value for key, value in row.items() if key != "arm"} for row in arm_rows
        ]
        add(
            [row.get("arm") for row in arm_rows] != list(ARMS)
            or len(projections) != len(ARMS)
            or any(row != projections[0] for row in projections),
            "arm_matching",
        )
        add(
            any(
                row.get("decision_context_actual_bytes", 0) > BYTE_BUDGETS["decision_context_bytes"]
                for row in arm_rows
            ),
            "decision_byte_budget",
        )
    return errors


def _task_block(roadmap_text: str) -> str:
    """Extract the Exp7183 YAML item without parsing prompt text as a gate."""

    start = roadmap_text.find("- id: exp7183-supersession-stream")
    if start < 0:
        return ""
    end = roadmap_text.find("\n- id:", start + 1)
    return roadmap_text[start:] if end < 0 else roadmap_text[start:end]


def _path_writable(path: Path) -> bool:
    """Probe a destination directory without changing its requested file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7183-probe-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
    except OSError:
        return False
    return True


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind all required sources, implementation bytes, and frozen contracts."""

    return {str(path): sha256_path(repo_root / path) for path in SOURCE_PATHS}


def collect_preconditions(repo_root: Path, paths: StreamPaths) -> list[JsonDict]:
    """Check external inputs and exact gates before constructing stream rows."""

    spec_path = repo_root / "openspec/capabilities/continuous-learning/spec.md"
    roadmap_path = repo_root / "research-roadmap.yaml"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.is_file() else ""
    task_block = _task_block(roadmap_text)
    gate_lines = [
        line.strip() for line in task_block.splitlines() if re.match(r"^\s*gated_on:\s*", line)
    ]
    sizes = {
        str(path): (repo_root / path).stat().st_size if (repo_root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    hashes = source_artifact_hashes(repo_root)
    destinations = {
        "decision_view": _path_writable(paths.decisions),
        "feedback_view": _path_writable(paths.feedback),
        "evaluator_truth": _path_writable(paths.truth),
        "availability_matrix": _path_writable(paths.availability),
        "manifest_directory": _path_writable(paths.manifest_dir / "probe.json"),
        "artifact": _path_writable(paths.artifact),
    }
    evaluator_probe = all(
        exact_label(family, 12_345, _rule_for(family, version))
        == independent_exact_label(family, 12_345, _rule_for(family, version))
        for family in FAMILIES
        for version in ("v1", "v2", "v3")
    )
    identity = {
        "id": "exp7183-supersession-stream" if task_block else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in task_block else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH)
            if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in task_block
            else None
        ),
    }
    expected_identity = {
        "id": "exp7183-supersession-stream",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    tools = {
        "python_executable": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    return [
        gate_check(
            "driving_capability_spec",
            "openspec/capabilities/continuous-learning/spec.md",
            "REQ-CL-7183",
            True,
            "REQ-CL-7183" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            "openspec/capabilities/continuous-learning/spec.md",
            "SCENARIO-CL-7183-*",
            6,
            spec_text.count("### SCENARIO-CL-7183-"),
            spec_text.count("### SCENARIO-CL-7183-") >= 6,
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            sizes,
            all(isinstance(size, int) and size > 0 for size in sizes.values()),
        ),
        gate_check(
            "required_source_hashes",
            "repository",
            "SOURCE_PATHS.sha256",
            "sha256:<64 hex> for every source",
            hashes,
            all(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in hashes.values()
            ),
        ),
        gate_check(
            "v633_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            identity,
        ),
        gate_check(
            "same_milestone_upstream_gates",
            "research-roadmap.yaml",
            "exp7183.gated_on",
            [],
            gate_lines,
        ),
        gate_check(
            "deterministic_exact_evaluators",
            "local_integer_predicates",
            "all_family_version_probes",
            True,
            evaluator_probe,
        ),
        gate_check(
            "required_local_tools",
            "host",
            "python,sha256sum",
            {key: True for key in tools},
            tools,
        ),
        gate_check(
            "output_destinations_writable",
            "host_filesystem",
            "stream,manifest,artifact_paths",
            {key: True for key in destinations},
            destinations,
        ),
    ]


def _empty_evidence() -> JsonDict:
    """Keep blocked artifacts schema-complete without inventing measurements."""

    return {
        "rows": [],
        "event_rows": [],
        "availability_rows": [],
        "exact_agreement_rows": [],
        "corruption_witness_rows": [],
        "revocation_rows": [],
        "recurrence_rows": [],
        "transfer_rows": [],
        "leakage_rows": [],
        "arm_materialization_rows": [],
        "regime_rows": [],
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash timing-free evidence while blanking the self-referential digest."""

    stable = deepcopy(dict(artifact))
    stable["duration_s"] = None
    stable["reproducibility_checksum"] = None
    return sha256_json(stable)


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    paths: StreamPaths,
    *,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Return a terminal no-run record for an external prerequisite failure."""

    summary = gate_summary(checks)
    failed = summary.get("failed_check") or "unknown_precondition"
    empty_paths = {
        "decision": str(paths.decisions),
        "feedback": str(paths.feedback),
        "truth": str(paths.truth),
        "availability": str(paths.availability),
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "preconditions_checked": [dict(row) for row in checks],
        "run_date": run_date,
        "inference_substrate": "blocked before qualifying stream construction",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": f"complete_blocked_supersession_stream:{failed}",
        "inference_substrate_class": "blocked_no_run",
        "stream_ready_score": 0,
        "decision_view_path": str(paths.decisions),
        "feedback_view_path": str(paths.feedback),
        "evaluator_truth_path": str(paths.truth),
        "availability_matrix_path": str(paths.availability),
        "sealed_view_paths": empty_paths,
        "sealed_view_hashes": {key: None for key in empty_paths},
        "manifest_paths": {name: str(paths.manifest_path(name)) for name in MANIFEST_NAMES},
        "manifest_hashes": {name: None for name in MANIFEST_NAMES},
        "feedback_schedule": {},
        "heldout_manifest": {},
        "template_operation_contract": deepcopy(TEMPLATE_OPERATION_CONTRACT),
        "family_manifest": {},
        "regime_manifest": {},
        "arm_manifest": {},
        "byte_budgets": deepcopy(BYTE_BUDGETS),
        "learning_executed": False,
        "accuracy_claimed": False,
        **_empty_evidence(),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _regime_rows(truths: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Summarize evaluator-only shift rows without exposing regimes to decisions."""

    return [
        {
            "regime_id": regime,
            "start_index": index * REGIME_SIZE,
            "stop_index": (index + 1) * REGIME_SIZE,
            "event_count": sum(row["regime_id"] == regime for row in truths),
            "family_counts": {
                family: sum(
                    row["regime_id"] == regime and row["family_id"] == family for row in truths
                )
                for family in FAMILIES
            },
        }
        for index, regime in enumerate(REGIMES)
    ]


def build_and_seal(
    repo_root: Path,
    paths: StreamPaths,
    *,
    run_date: str,
    duration_s: float | None = None,
    progress: bool = False,
) -> JsonDict:
    """Check preconditions, build all views, and write immutable evidence."""

    started = time.monotonic()
    if progress:
        print("PHASE 0 START: verify sources, gates, tools, destinations, and hashes", flush=True)
    checks = collect_preconditions(repo_root, paths)
    if not all(row["passed"] for row in checks):
        elapsed = time.monotonic() - started if duration_s is None else duration_s
        return build_blocked_artifact(checks, paths, run_date=run_date, duration_s=elapsed)
    if progress:
        print("PHASE 1 START: activate progress and bounded execution contract", flush=True)
        print("PHASE 2 START: construct 240 frozen chronological events", flush=True)
    views = build_stream_views()
    if progress:
        print("PHASE 3 START: verify separated delayed feedback and supersession", flush=True)
        print("PHASE 4 START: verify held-out partitions, recurrence, and matched arms", flush=True)
    conformance = stream_conformance_errors(views)
    if progress:
        print(
            "PHASE 5 START: write immutable views, manifests, budgets, and availability", flush=True
        )
    view_paths = {
        "decision": paths.decisions,
        "feedback": paths.feedback,
        "truth": paths.truth,
        "availability": paths.availability,
    }
    view_rows = {
        "decision": views.decisions,
        "feedback": views.feedback,
        "truth": views.truths,
        "availability": views.availability,
    }
    view_hashes = {
        name: write_immutable_jsonl(view_paths[name], view_rows[name]) for name in view_paths
    }
    manifest_paths = {name: paths.manifest_path(name) for name in MANIFEST_NAMES}
    manifest_hashes = {
        name: write_immutable_json(manifest_paths[name], views.manifests[name])
        for name in MANIFEST_NAMES
    }
    if progress:
        print("PHASE 6 START: reduce rows against REQ-CL-7183 readiness", flush=True)
    ready = int(not conformance)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "complete",
        "preconditions_checked": checks,
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": elapsed,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "rows": deepcopy(views.arm_rows),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "honest_verdict": (
            "complete: sealed supersession stream ready; no learning or accuracy claim"
            if ready
            else "complete_disqualified_supersession_stream"
        ),
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "stream_ready_score": ready,
        "decision_view_path": str(paths.decisions),
        "feedback_view_path": str(paths.feedback),
        "evaluator_truth_path": str(paths.truth),
        "availability_matrix_path": str(paths.availability),
        "sealed_view_paths": {name: str(path) for name, path in view_paths.items()},
        "sealed_view_hashes": view_hashes,
        "manifest_paths": {name: str(path) for name, path in manifest_paths.items()},
        "manifest_hashes": manifest_hashes,
        "feedback_schedule": deepcopy(views.manifests["feedback"]),
        "heldout_manifest": deepcopy(views.manifests["heldout"]),
        "regime_rows": _regime_rows(views.truths),
        "template_operation_contract": deepcopy(views.manifests["operation"]),
        "family_manifest": deepcopy(views.manifests["family"]),
        "regime_manifest": deepcopy(views.manifests["regime"]),
        "arm_manifest": deepcopy(views.manifests["arm"]),
        "byte_budgets": deepcopy(views.manifests["budget"]),
        "event_rows": deepcopy(views.events),
        "availability_rows": deepcopy(views.availability),
        "exact_agreement_rows": deepcopy(views.exact_agreements),
        "corruption_witness_rows": deepcopy(views.corruption_witnesses),
        "revocation_rows": deepcopy(views.revocations),
        "recurrence_rows": deepcopy(views.recurrence),
        "transfer_rows": deepcopy(views.transfer),
        "leakage_rows": [
            {
                "event_id": row["event_id"],
                "forbidden_fields": sorted(_nested_keys(row) & FORBIDDEN_DECISION_FIELDS),
                "passed": not bool(_nested_keys(row) & FORBIDDEN_DECISION_FIELDS),
            }
            for row in views.decisions
        ],
        "arm_materialization_rows": deepcopy(views.arm_rows),
        "learning_executed": False,
        "accuracy_claimed": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _resolve(repo_root: Path, value: str) -> Path:
    """Resolve repository-relative evidence paths for cold file checks."""

    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check schema, rows, gates, hashes, and terminal claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(
        set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS),
        "field_principles_mismatch",
    )
    add(artifact["schema"] != SCHEMA, "schema_mismatch")
    add(artifact["experiment_id"] != EXPERIMENT_ID, "experiment_id_mismatch")
    add(artifact["milestone"] != MILESTONE, "milestone_mismatch")
    add(artifact["run_date"] != RUN_DATE, "run_date_mismatch")
    add(type(artifact["stream_ready_score"]) is not int, "stream_ready_score_not_bare_integer")
    add(artifact["verifier_is_oracle"] is not True, "verifier_is_oracle_mismatch")
    add(artifact["learning_executed"] is not False, "learning_claim_mismatch")
    add(artifact["accuracy_claimed"] is not False, "accuracy_claim_mismatch")
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(artifact["status"] != "blocked", "blocked_status_mismatch")
        add(artifact["inference_substrate_class"] != "blocked_no_run", "blocked_substrate_mismatch")
        add(artifact["stream_ready_score"] != 0, "stream_ready_score_mismatch")
        add(bool(artifact["rows"]), "blocked_rows_present")
        add(
            artifact["gate_check_summary"].get("passed") is not False,
            "blocked_gate_summary_mismatch",
        )
        add(not artifact["gate_check_summary"].get("failed_check"), "blocked_failed_check_missing")
        add(not artifact["gate_check_summary"].get("upstream"), "blocked_upstream_missing")
        add(not artifact["gate_check_summary"].get("field"), "blocked_field_missing")
    else:
        expected = build_stream_views()
        leaked_decisions = [row.get("decision", {}) for row in artifact["event_rows"]]
        add(bool(leakage_errors(leaked_decisions)), "stream_conformance:decision_leakage")
        comparisons = {
            "rows": expected.arm_rows,
            "arm_materialization_rows": expected.arm_rows,
            "event_rows": expected.events,
            "availability_rows": expected.availability,
            "exact_agreement_rows": expected.exact_agreements,
            "corruption_witness_rows": expected.corruption_witnesses,
            "revocation_rows": expected.revocations,
            "recurrence_rows": expected.recurrence,
            "transfer_rows": expected.transfer,
            "regime_rows": _regime_rows(expected.truths),
        }
        for field, expected_value in comparisons.items():
            add(artifact[field] != expected_value, f"{field}_mismatch")
        add(
            artifact["feedback_schedule"] != expected.manifests["feedback"],
            "feedback_schedule_mismatch",
        )
        add(
            artifact["heldout_manifest"] != expected.manifests["heldout"],
            "heldout_manifest_mismatch",
        )
        add(artifact["family_manifest"] != expected.manifests["family"], "family_manifest_mismatch")
        add(artifact["regime_manifest"] != expected.manifests["regime"], "regime_manifest_mismatch")
        add(artifact["arm_manifest"] != expected.manifests["arm"], "arm_manifest_mismatch")
        add(artifact["byte_budgets"] != expected.manifests["budget"], "byte_budgets_mismatch")
        add(
            artifact["template_operation_contract"] != expected.manifests["operation"],
            "operation_contract_mismatch",
        )
        expected_ready = int(not stream_conformance_errors(expected))
        add(artifact["stream_ready_score"] != expected_ready, "stream_ready_score_mismatch")
        add(artifact["status"] != "complete", "status_mismatch")
        add(artifact["inference_substrate"] != INFERENCE_SUBSTRATE, "inference_substrate_mismatch")
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class_mismatch",
        )
        add(artifact["execution_venue"] != EXECUTION_VENUE, "execution_venue_mismatch")
        add(
            artifact["verdict_class"]
            != ("circular_positive" if expected_ready else "disqualified"),
            "verdict_class_mismatch",
        )
        add(
            expected_ready == 1 and "no learning" not in artifact["honest_verdict"],
            "honest_verdict_mismatch",
        )
        expected_view_hashes = {
            "decision": sha256_bytes(_jsonl_bytes(expected.decisions)),
            "feedback": sha256_bytes(_jsonl_bytes(expected.feedback)),
            "truth": sha256_bytes(_jsonl_bytes(expected.truths)),
            "availability": sha256_bytes(_jsonl_bytes(expected.availability)),
        }
        add(artifact["sealed_view_hashes"] != expected_view_hashes, "sealed_view_hashes_mismatch")
        expected_manifest_hashes = {
            name: sha256_bytes(canonical_json(expected.manifests[name]) + b"\n")
            for name in MANIFEST_NAMES
        }
        add(artifact["manifest_hashes"] != expected_manifest_hashes, "manifest_hashes_mismatch")
        root = repo_root or Path(__file__).resolve().parents[2]
        if repo_root is not None:
            add(
                artifact["source_artifact_hashes"] != source_artifact_hashes(root),
                "source_hashes_mismatch",
            )
        if check_files:
            add(
                any(
                    sha256_path(_resolve(root, artifact["sealed_view_paths"][name]))
                    != expected_hash
                    for name, expected_hash in expected_view_hashes.items()
                ),
                "sealed_view_file_mismatch",
            )
            add(
                any(
                    sha256_path(_resolve(root, artifact["manifest_paths"][name])) != expected_hash
                    for name, expected_hash in expected_manifest_hashes.items()
                ),
                "manifest_file_mismatch",
            )
    add(
        artifact["reproducibility_checksum"] != reproducibility_checksum(artifact),
        "reproducibility_checksum_mismatch",
    )
    return errors


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and an optional private output root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Construct, validate, and atomically publish the terminal artifact."""

    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    paths = (
        StreamPaths.defaults() if args.output_root is None else StreamPaths.under(args.output_root)
    )
    artifact = build_and_seal(
        repo_root,
        paths,
        run_date=str(args.date),
        progress=True,
    )
    print("PHASE 7 START: validate full file-to-parser-to-gate path", flush=True)
    print("PHASE 8 START: verify required principle-annotated artifact fields", flush=True)
    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        check_files=artifact["verdict_class"] != "blocked",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    print("FINAL ATOMIC WRITE START", flush=True)
    write_json_atomic(paths.artifact, artifact)
    print("FINAL ATOMIC WRITE END", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command wrapper owns this path.
    raise SystemExit(main())
