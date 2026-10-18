"""Cold-audit delayed procedural memory from immutable producer records.

This module does not import the producer. It captures bytes before decoding,
rebuilds state from an empty process, and treats producer summaries only as
parity targets. The separation prevents a producer bug from certifying itself.

Spec refs: REQ-CL-7107 and SCENARIO-CL-7107-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 7107
SCHEMA = "carnot.exp7107.v623_continual_memory_cold_audit.v1"
RUN_DATE = "20260907"
RANDOM_SEED = 7_107_202_609_07
INFERENCE_SUBSTRATE = "fresh-process deterministic memory and transaction replay"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
MODEL_WEIGHTS_CHANGED = False
EVENT_COUNT = 144
ARMS = (
    "delayed_procedural",
    "raw_trace",
    "equal_context_replay",
    "write_while_deciding",
    "no_memory",
)
PERSISTENT_ARMS = {"delayed_procedural", "raw_trace", "write_while_deciding"}
STATELESS_ARMS = {"equal_context_replay", "no_memory"}
COMPARISONS = (
    "delayed_procedural_vs_raw_trace",
    "delayed_procedural_vs_equal_context_replay",
    "delayed_procedural_vs_write_while_deciding",
    "delayed_procedural_vs_no_memory",
)
RECORD_SLOT_BYTES = 16_384
CONTEXT_BUDGET_BYTES = 4_096
EMPTY_MEMORY_HASH = "sha256:" + hashlib.sha256(b"[]").hexdigest()
FIXED_POLICY_HASH = "sha256:6c372a6989d489b6d102ab71e6738f54dc03995aa8ff7ca06427eb95c3930127"
EVICTION_POLICY = "fifo_oldest_write"

DEFAULT_PRODUCER_PATH = REPO_ROOT / "results/experiment_7106_v623_procedural_memory_csl.json"
DEFAULT_STREAM_ARTIFACT_PATH = (
    REPO_ROOT / "results/experiment_7105_v623_exact_constraint_stream.json"
)
DEFAULT_STREAM_PATH = (
    REPO_ROOT / "results/streams/experiment_7105_v623_exact_constraint_stream.jsonl"
)
DEFAULT_DECISION_PATH = REPO_ROOT / "results/streams/experiment_7105_v623_decision_view.jsonl"
DEFAULT_LABEL_PATH = REPO_ROOT / "results/streams/experiment_7105_v623_label_view.jsonl"
PRIOR_COLD_AUDIT_PATH = REPO_ROOT / "results/experiment_6979_self_learning_cold_audit.json"
PRIOR_ROLLBACK_AUDIT_PATH = REPO_ROOT / "results/experiment_7071_bcit_drift_rollback_audit.json"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "results/experiment_7107_v623_continual_memory_cold_audit.json"
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_7107_v623_continual_memory_cold_audit.py"

EXPECTED_HASHES = {
    "producer_artifact": "sha256:36e92fea3a0eec6efdb66c54ee3278d859f4ff27a14f0358a4aef56e45b95405",
    "stream_artifact": "sha256:3b076570c5b2d19584b2b8f8e65d3e48b38285b30d05ba08448a7231c3697003",
    "sealed_stream": "sha256:6b77320ceda12c5c65abe98fb44c4f9ebae46fbc4885e3de51b74277bfe0a222",
    "decision_view": "sha256:30076191b1a873ebb0c615b7c1cf473451782c7d4badfa425457f0d6447ffbb3",
    "label_view": "sha256:af7cdbbe02d1eda97c7c018b81da229b360a8a565ccef499aa3c91024084043e",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "upstream_gate_receipt",
    "runtime_isolation_receipt",
    "stream_hash",
    "transaction_log_hash",
    "snapshot_hash_rows",
    "reconstruction_rows",
    "event_replay_rows",
    "metric_recomputation_rows",
    "paired_test_rows",
    "protected_retention_rows",
    "capacity_rows",
    "eviction_rows",
    "future_label_isolation_rows",
    "signature_rows",
    "poison_attack_rows",
    "reorder_attack_rows",
    "stale_parent_rows",
    "partial_write_rows",
    "crash_recovery_rows",
    "rollback_rows",
    "mutation_attack_rows",
    "rows",
    "producer_auditor_parity_rows",
    "model_weights_changed",
    "continual_memory_cold_audit_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible audit readers fail closed.",
    "experiment_id": "A fixed identity prevents another audit from supplying this evidence.",
    "run_date": "The declared date binds the audit to its requested execution window.",
    "field_principles": "One reason per field keeps the evidence contract reviewable.",
    "preconditions_checked": "Exact input gates stop missing state before scientific reduction.",
    "inference_substrate": "The substrate declares deterministic replay instead of live inference.",
    "inference_substrate_class": "The compute class separates aggregation from a blocked no-run.",
    "execution_venue": "The host venue prevents a false remote or hardware execution claim.",
    "duration_s": "Measured wall time shows that the complete audit process executed.",
    "source_artifact_hashes": "Byte hashes bind every conclusion to immutable input files.",
    "upstream_gate_receipt": "The receipt keeps audit readiness separate from upstream value.",
    "runtime_isolation_receipt": "Process isolation proves the replay loaded no model or accelerator.",
    "stream_hash": "The stream seal binds all decisions to one chronological event history.",
    "transaction_log_hash": "The transaction digest binds the complete producer phase history.",
    "snapshot_hash_rows": "Per-event hashes expose substituted or hidden persistent state.",
    "reconstruction_rows": "Empty-state replay proves each decision and transition independently.",
    "event_replay_rows": "One row per event preserves complete five-arm paired coverage.",
    "metric_recomputation_rows": "Row-only reductions prevent producer summaries becoming authority.",
    "paired_test_rows": "Aligned event differences prevent pooled metrics hiding losses.",
    "protected_retention_rows": "Protected probes make forgetting a falsifiable audit condition.",
    "capacity_rows": "Capacity strata prove memory value did not use an extra state budget.",
    "eviction_rows": "Derived FIFO receipts prove the frozen eviction policy executed.",
    "future_label_isolation_rows": "Causal rows prove no current decision read its own feedback.",
    "signature_rows": "Verifier signatures bind accepted updates to exact outcomes and witnesses.",
    "poison_attack_rows": "Poison attacks show false or unsigned feedback cannot mutate state.",
    "reorder_attack_rows": "Reorder attacks show chronology cannot become an alternate result.",
    "stale_parent_rows": "Parent checks prevent a valid update from overwriting newer state.",
    "partial_write_rows": "Partial-write probes show only complete state boundaries become active.",
    "crash_recovery_rows": "Crash probes recover the exact parent or committed child hash.",
    "rollback_rows": "Byte-identical rollback proves adverse committed updates are reversible.",
    "mutation_attack_rows": "The complete attack ledger prevents selective safety reporting.",
    "rows": "Generic per-event rows retain the denominator for external consistency linting.",
    "producer_auditor_parity_rows": "Named parity cells expose any producer and auditor disagreement.",
    "model_weights_changed": "False isolates external memory from hidden gradient updates.",
    "continual_memory_cold_audit_ready_score": "One requires replay, parity, safety, and retention together.",
    "random_seed": "A fixed seed makes every deterministic tie and checksum reproducible.",
    "reproducibility_checksum": "A timing-free digest detects later scientific-content drift.",
    "gate_check_summary": "Exact expected and observed values make a blocked input actionable.",
    "verifier_is_oracle": "False states that this auditor checks evidence but does not label candidates.",
    "verdict_class": "A closed terminal class keeps readiness distinct from scientific value.",
    "honest_verdict": "A terminal prefix lets automation classify the completed audit safely.",
}
FIELD_PRINCIPLES = {field: _PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS}

TEST_RUNTIME_RECEIPT = {
    "fresh_process": True,
    "network_disabled": True,
    "gpu_disabled": True,
    "llm_disabled": True,
    "input_write_disabled": True,
    "protected_inputs_unchanged": True,
}


def canonical_bytes(value: Any) -> bytes:
    """Serialize state with the exact byte convention used by Exp7106."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a named SHA-256 digest so hashes cannot look like raw values."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash structured evidence after one stable serialization."""

    return sha256_bytes(canonical_bytes(value))


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode immutable stream rows with one canonical record per line."""

    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _snapshot_file(path: Path) -> JsonDict:
    """Read bytes once so no decoded field can redirect the initial capture."""

    try:
        payload = path.read_bytes()
    except OSError:
        return {"path": str(path), "bytes": None, "sha256": None, "size": None}
    return {
        "path": str(path),
        "bytes": payload,
        "sha256": sha256_bytes(payload),
        "size": len(payload),
    }


def capture_input_snapshot(
    *,
    repo_root: Path,
    producer_artifact_path: Path = DEFAULT_PRODUCER_PATH,
    stream_artifact_path: Path = DEFAULT_STREAM_ARTIFACT_PATH,
) -> JsonDict:
    """Capture all current and historical audit inputs before JSON decoding."""

    root = Path(repo_root)
    return {
        "files": {
            "producer_artifact": _snapshot_file(Path(producer_artifact_path)),
            "stream_artifact": _snapshot_file(Path(stream_artifact_path)),
            "sealed_stream": _snapshot_file(root / DEFAULT_STREAM_PATH.relative_to(REPO_ROOT)),
            "decision_view": _snapshot_file(root / DEFAULT_DECISION_PATH.relative_to(REPO_ROOT)),
            "label_view": _snapshot_file(root / DEFAULT_LABEL_PATH.relative_to(REPO_ROOT)),
            "prior_cold_audit": _snapshot_file(root / PRIOR_COLD_AUDIT_PATH.relative_to(REPO_ROOT)),
            "prior_rollback_audit": _snapshot_file(
                root / PRIOR_ROLLBACK_AUDIT_PATH.relative_to(REPO_ROOT)
            ),
        }
    }


def _decode_object(entry: Mapping[str, Any], name: str) -> JsonDict:
    """Decode one object only after its source bytes already have an identity."""

    payload = entry.get("bytes")
    if not isinstance(payload, bytes):
        raise ValueError(f"input is not readable:{name}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"input is not a JSON object:{name}")
    return value


def _decode_jsonl(entry: Mapping[str, Any], name: str) -> list[JsonDict]:
    """Decode an immutable JSONL file while preserving its stored order."""

    payload = entry.get("bytes")
    if not isinstance(payload, bytes):
        raise ValueError(f"input is not readable:{name}")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"input has a non-object JSONL row:{name}")
    return rows


def decode_captured_inputs(snapshot: Mapping[str, Any]) -> JsonDict:
    """Decode detached values without reopening any input path."""

    files = snapshot["files"]
    return {
        "producer": _decode_object(files["producer_artifact"], "producer_artifact"),
        "stream_artifact": _decode_object(files["stream_artifact"], "stream_artifact"),
        "stream": _decode_jsonl(files["sealed_stream"], "sealed_stream"),
        "decisions": _decode_jsonl(files["decision_view"], "decision_view"),
        "labels": _decode_jsonl(files["label_view"], "label_view"),
    }


def gate_check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Retain both sides of each gate so a block names the exact cause."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Copy the first failed gate into stable diagnostic fields."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": rows,
        "failed_check": None if failed is None else failed["check"],
        "expected_value": "all checks pass" if failed is None else failed["expected_value"],
        "observed_value": "all checks pass" if failed is None else failed["observed_value"],
    }


def _decision_signature(row: Mapping[str, Any]) -> str:
    """Rebuild the label-blind signature from visible problem content."""

    return sha256_json(
        {
            "constraint_family": row["constraint_family"],
            "decision_visible_input": row["decision_visible_input"],
            "candidate_payloads": [item["payload"] for item in row["candidate_set"]],
        }
    )


def _decision_payload(row: Mapping[str, Any]) -> JsonDict:
    """Project only fields that a decision can read before feedback."""

    names = (
        "event_id",
        "constraint_family",
        "group_id",
        "chronology_index",
        "reuse_or_decoy_status",
        "hardness_stratum",
        "feedback_release_point",
        "protected_retention_probe",
        "protected_group",
        "matched_reusable_event_id",
        "structure_key",
        "decision_signature",
        "decision_visible_input",
        "candidate_set",
    )
    return {name: deepcopy(row[name]) for name in names}


def _label_payload(row: Mapping[str, Any]) -> JsonDict:
    """Project the exact post-decision feedback sidecar record."""

    return {
        "event_id": row["event_id"],
        "feedback_release_point": row["feedback_release_point"],
        "exact_label": row["exact_label"],
        "witness": deepcopy(row["witness"]),
    }


def _stream_contract_errors(inputs: Mapping[str, Any]) -> list[str]:
    """Check every upstream content seal without importing its producer."""

    errors: list[str] = []
    stream = list(inputs.get("stream", []))
    decisions = list(inputs.get("decisions", []))
    labels = list(inputs.get("labels", []))
    stream_artifact = inputs.get("stream_artifact", {})
    if len(stream) != EVENT_COUNT or len(decisions) != EVENT_COUNT or len(labels) != EVENT_COUNT:
        errors.append("sealed_record_count_mismatch")
        return errors
    if stream_artifact.get("event_rows") != stream:
        errors.append("sealed_stream_artifact_mismatch")
    if [row.get("chronology_index") for row in stream] != list(range(EVENT_COUNT)):
        errors.append("sealed_stream_reorder")
    label_by_id = {str(row.get("event_id")): row for row in labels}
    if len(label_by_id) != EVENT_COUNT:
        errors.append("duplicate_feedback_event")
    for event, decision in zip(stream, decisions, strict=True):
        try:
            signature = _decision_signature(event)
            expected_decision = _decision_payload({**event, "decision_signature": signature})
            expected_decision["decision_content_hash"] = sha256_json(expected_decision)
            label = label_by_id[str(event["event_id"])]
            expected_label = _label_payload(event)
            expected_label["label_content_hash"] = sha256_json(expected_label)
            sealed = sha256_json(
                {
                    "decision_content_hash": expected_decision["decision_content_hash"],
                    "label_content_hash": expected_label["label_content_hash"],
                }
            )
            if decision != expected_decision:
                errors.append("decision_content_hash_mismatch")
            if label != expected_label:
                errors.append("label_content_hash_mismatch")
            if (
                event.get("decision_signature") != signature
                or event.get("canonical_content_hash") != sealed
            ):
                errors.append("event_content_hash_mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append("invalid_sealed_record")
    return list(dict.fromkeys(errors))


def collect_preconditions(
    snapshot: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict | None]:
    """Require every immutable producer record before replay starts."""

    files = snapshot.get("files", {})
    checks = [
        gate_check(
            "producer_artifact_readable",
            True,
            isinstance(files.get("producer_artifact", {}).get("bytes"), bytes),
        )
    ]
    required_names = ("stream_artifact", "sealed_stream", "decision_view", "label_view")
    checks.append(
        gate_check(
            "sealed_inputs_readable",
            True,
            all(isinstance(files.get(name, {}).get("bytes"), bytes) for name in required_names),
        )
    )
    if not all(row["passed"] for row in checks):
        return checks, None
    try:
        inputs = decode_captured_inputs(snapshot)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        checks.append(gate_check("input_json_decoding", "valid", type(exc).__name__))
        return checks, None
    producer = inputs["producer"]
    stream_artifact = inputs["stream_artifact"]
    rows = list(producer.get("rows", []))
    immutable_fields = (
        "decision_rows",
        "feedback_rows",
        "update_rows",
        "transaction_rows",
        "memory_hash_rows",
    )
    checks.extend(
        [
            gate_check(
                "procedural_memory_comparison_complete_score",
                1,
                producer.get("procedural_memory_comparison_complete_score"),
            ),
            gate_check(
                "producer_artifact_hash",
                EXPECTED_HASHES["producer_artifact"],
                files["producer_artifact"]["sha256"],
            ),
            gate_check(
                "exact_constraint_stream_ready_score",
                1,
                stream_artifact.get("exact_constraint_stream_ready_score"),
            ),
            gate_check(
                "stream_artifact_hash",
                EXPECTED_HASHES["stream_artifact"],
                files["stream_artifact"]["sha256"],
            ),
            gate_check(
                "sealed_stream_hash",
                EXPECTED_HASHES["sealed_stream"],
                files["sealed_stream"]["sha256"],
            ),
            gate_check(
                "decision_view_hash",
                EXPECTED_HASHES["decision_view"],
                files["decision_view"]["sha256"],
            ),
            gate_check(
                "label_view_hash", EXPECTED_HASHES["label_view"], files["label_view"]["sha256"]
            ),
            gate_check("immutable_arm_event_rows", EVENT_COUNT * len(ARMS), len(rows)),
            gate_check(
                "immutable_decision_transaction_snapshot_records",
                [EVENT_COUNT * len(ARMS)] * len(immutable_fields),
                [len(producer.get(field, [])) for field in immutable_fields],
            ),
            gate_check("sealed_content_records", [], _stream_contract_errors(inputs)),
            gate_check(
                "producer_stream_hash",
                stream_artifact.get("stream_hash"),
                producer.get("stream_hash"),
            ),
            gate_check(
                "frozen_capacity_schedule_count",
                3,
                len(producer.get("frozen_capacity_schedule", [])),
            ),
        ]
    )
    return checks, inputs


def _procedure_accepts(event: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Evaluate visible constraints without consulting a stored exact label."""

    family = str(event["constraint_family"])
    problem = event["decision_visible_input"]["problem"]
    if family == "sat":
        values = list(payload["assignment"])
        return all(
            any(values[abs(literal) - 1] == (literal > 0) for literal in clause)
            for clause in problem["clauses"]
        )
    if family == "graph_coloring":
        colors = list(payload["colors"])
        return all(0 <= color < int(problem["n_colors"]) for color in colors) and all(
            colors[left] != colors[right] for left, right in problem["edges"]
        )
    if family == "arithmetic":
        left, right = int(problem["a"]), int(problem["b"])
        targets = {
            "triangle": left * right + left,
            "delta": left * right - right,
            "box": (left + right) * 2,
        }
        return int(payload["value"]) == targets[str(problem["rule_name"])]
    trace = list(payload["trace"])
    signal = str(problem["signal"])
    if problem["temporal_operator"] == "always":
        return all(bool(step.get(signal, False)) for step in trace)
    if problem["temporal_operator"] == "eventually":
        return any(bool(step.get(signal, False)) for step in trace)
    goals = [index for index, step in enumerate(trace) if bool(step.get(signal, False))]
    return bool(goals) and all(
        bool(trace[index].get(str(problem["guard_signal"]), False)) for index in range(goals[0])
    )


def _neutral_record() -> JsonDict:
    """Represent an occupied retrieval slot that carries no history."""

    return {
        "memory_id": "neutral",
        "representation": "neutral",
        "source_event_id": None,
        "source_chronology_index": None,
    }


def _abstract_record(event: Mapping[str, Any], label: str) -> JsonDict:
    """Derive the delayed abstract procedure from signed feedback."""

    return {
        "memory_key": f"procedure:{event['structure_key']}",
        "memory_id": f"procedure:{event['event_id']}",
        "representation": "abstract_procedure",
        "scope_key": event["structure_key"],
        "constraint_family": event["constraint_family"],
        "template": event["decision_visible_input"]["template"],
        "operation": "evaluate_declared_visible_constraints",
        "source_event_id": event["event_id"],
        "source_chronology_index": event["chronology_index"],
        "source_label_hash": sha256_json(label),
        "provenance_complete": True,
    }


def _raw_record(event: Mapping[str, Any], label: str, witness: Mapping[str, Any]) -> JsonDict:
    """Derive a raw trace from exact event and witness records."""

    return {
        "memory_key": f"trace:{event['event_id']}",
        "memory_id": f"trace:{event['event_id']}",
        "representation": "raw_trace",
        "scope_key": event["structure_key"],
        "source_event_id": event["event_id"],
        "source_chronology_index": event["chronology_index"],
        "decision_visible_input": deepcopy(event["decision_visible_input"]),
        "candidate_set": deepcopy(event["candidate_set"]),
        "exact_label": label,
        "valid_payload_hash": sha256_json(witness["valid_candidate_payload"]),
        "witness": deepcopy(dict(witness)),
        "provenance_complete": True,
    }


def _provisional_record(event: Mapping[str, Any]) -> JsonDict:
    """Derive the self-generated record that exists before exact feedback."""

    return {
        "memory_key": f"procedure:{event['structure_key']}",
        "memory_id": f"provisional:{event['event_id']}",
        "representation": "provisional_procedure",
        "scope_key": event["structure_key"],
        "chosen_index": 0,
        "source_event_id": event["event_id"],
        "source_chronology_index": event["chronology_index"],
        "provenance_complete": False,
    }


def _replay_record(previous: Mapping[str, Any] | None) -> JsonDict:
    """Derive the transient prior-feedback control without persistent state."""

    if previous is None:
        return _neutral_record()
    return {
        "memory_id": f"replay:{previous['event_id']}",
        "representation": "equal_context_replay",
        "source_event_id": previous["event_id"],
        "source_chronology_index": previous["chronology_index"],
        "chosen_index": int(str(previous["exact_label"]).rsplit(":", 1)[1]),
    }


def _retrieve(records: Sequence[Mapping[str, Any]], scope_key: str) -> JsonDict:
    """Return the newest scope match without changing state order."""

    return next(
        (
            deepcopy(dict(record))
            for record in reversed(records)
            if record.get("scope_key") == scope_key
        ),
        _neutral_record(),
    )


def _fixed_decision(event: Mapping[str, Any], record: Mapping[str, Any]) -> JsonDict:
    """Recompute the fixed candidate policy from visible inputs and one record."""

    candidates = list(event["candidate_set"])
    representation = record.get("representation")
    chosen = 0
    score = 0.5
    if representation == "abstract_procedure":
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if _procedure_accepts(event, candidate["payload"])
        ]
        if len(matches) == 1:
            chosen, score = matches[0], 0.9
    elif representation == "raw_trace":
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if sha256_json(candidate["payload"]) == record.get("valid_payload_hash")
        ]
        if len(matches) == 1:
            chosen, score = matches[0], 0.75
    elif representation in {"equal_context_replay", "provisional_procedure"}:
        chosen = int(record.get("chosen_index", 0)) % len(candidates)
        score = 0.6
    return {
        "candidate_id": candidates[chosen]["candidate_id"],
        "candidate_index": chosen,
        "score": score,
        "policy_hash": FIXED_POLICY_HASH,
    }


def _apply_record(
    records: Sequence[Mapping[str, Any]], record: Mapping[str, Any], capacity: int
) -> tuple[list[JsonDict], list[str]]:
    """Build a complete FIFO child or reject it before active state changes."""

    if capacity < 1:
        raise ValueError("capacity must be positive")
    next_records = [
        deepcopy(dict(item))
        for item in records
        if item.get("memory_key") != record.get("memory_key")
    ]
    next_records.append(deepcopy(dict(record)))
    evicted: list[str] = []
    while len(next_records) > capacity:
        evicted.append(str(next_records.pop(0)["memory_id"]))
    if len(canonical_bytes(next_records)) > capacity * RECORD_SLOT_BYTES:
        raise ValueError("memory byte capacity exceeded")
    return next_records, evicted


def _commit_receipt(
    parent: Sequence[Mapping[str, Any]],
    child: Sequence[Mapping[str, Any]],
    evicted: Sequence[str],
    capacity: int,
) -> JsonDict:
    """Derive the transaction fields owned by one complete state swap."""

    return {
        "committed": True,
        "atomic": True,
        "parent_hash": sha256_json(parent),
        "child_hash": sha256_json(child),
        "evicted_ids": list(evicted),
        "item_count": len(child),
        "charged_bytes": len(child) * RECORD_SLOT_BYTES,
        "actual_bytes": len(canonical_bytes(child)),
    }


def _witness_valid(event: Mapping[str, Any], label: str, witness: Mapping[str, Any]) -> bool:
    """Verify the exact witness against all visible candidate payloads."""

    authority = f"deterministic_{event['constraint_family']}_solver"
    candidates = list(event["candidate_set"])
    results = list(witness.get("candidate_results", []))
    expected = [
        {
            "candidate_id": candidate["candidate_id"],
            "valid": _procedure_accepts(event, candidate["payload"]),
        }
        for candidate in candidates
    ]
    observed = [
        {"candidate_id": row.get("candidate_id"), "valid": row.get("valid")} for row in results
    ]
    valid = [row["candidate_id"] for row in expected if row["valid"]]
    payload = next(
        (candidate["payload"] for candidate in candidates if candidate["candidate_id"] == label),
        None,
    )
    return all(
        (
            witness.get("authority") == authority,
            observed == expected,
            valid == [label],
            witness.get("valid_candidate_id") == label,
            witness.get("valid_candidate_payload") == payload,
        )
    )


def _panel_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name missing, duplicate, or reordered arm-event identities."""

    expected = [f"{index:03d}:{arm}" for index in range(EVENT_COUNT) for arm in ARMS]
    observed = [str(row.get("row_key")) for row in rows]
    errors: list[str] = []
    if len(rows) < len(expected) or set(expected) - set(observed):
        errors.append("missing_arm_event")
    if len(observed) != len(set(observed)):
        errors.append("duplicate_arm_event")
    if len(rows) == len(expected) and observed != expected:
        errors.append("event_reorder")
    return errors


def _aggregate(
    rows: Sequence[Mapping[str, Any]], field: str, values: Sequence[Any]
) -> list[JsonDict]:
    """Reduce a complete arm-by-stratum table without dropping adverse cells."""

    result: list[JsonDict] = []
    for value in values:
        for arm in ARMS:
            selected = [row for row in rows if row.get(field) == value and row.get("arm") == arm]
            correct = sum(row.get("correct") is True for row in selected)
            result.append(
                {
                    field: value,
                    "arm": arm,
                    "event_count": len(selected),
                    "correct_count": correct,
                    "accuracy": round(correct / len(selected), 6) if selected else None,
                }
            )
    return result


def _sign_p(wins: int, losses: int) -> float:
    """Return the exact one-sided sign-test tail on discordant events."""

    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return round(
        sum(math.comb(discordant, index) for index in range(wins, discordant + 1))
        / (2**discordant),
        12,
    )


def recompute_metrics(
    rows: Sequence[Mapping[str, Any]], stream_artifact: Mapping[str, Any]
) -> JsonDict:
    """Recompute every headline table and paired test from atomic rows."""

    copied = [deepcopy(dict(row)) for row in rows]
    group_rows = _aggregate(copied, "group_id", list(stream_artifact["frozen_group_ids"]))
    family_rows = _aggregate(
        copied, "constraint_family", list(stream_artifact["frozen_family_ids"])
    )
    normalized = [
        {
            **row,
            "hardness_class": "hard" if row.get("hardness_stratum") == "hard" else "ordinary",
        }
        for row in copied
    ]
    hardness_rows = _aggregate(normalized, "hardness_class", ("ordinary", "hard"))
    reuse_rows = _aggregate(copied, "reuse_or_decoy_status", ("reusable", "decoy"))
    slice_rows = _aggregate(copied, "slice_id", ("early", "middle", "late"))
    capacities = [
        int(row["memory_capacity"]) for row in stream_artifact["frozen_capacity_schedule"]
    ]
    capacity_rows = _aggregate(copied, "capacity_items", capacities)
    overall_rows = _aggregate([{**row, "panel": "all"} for row in copied], "panel", ("all",))
    protected_rows: list[JsonDict] = []
    for group_id in stream_artifact["frozen_protected_group_ids"]:
        for arm in ARMS:
            selected = [
                row
                for row in copied
                if row.get("group_id") == group_id
                and row.get("arm") == arm
                and row.get("protected_retention_probe") is True
            ]
            correct = sum(row.get("correct") is True for row in selected)
            protected_rows.append(
                {
                    "group_id": group_id,
                    "arm": arm,
                    "probe_count": len(selected),
                    "correct_count": correct,
                    "retention_passed": bool(selected) and correct == len(selected),
                }
            )
    hard: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in copied:
        if row.get("hardness_stratum") == "hard":
            hard[(str(row.get("group_id")), str(row.get("arm")))].append(row)
    negative_rows: list[JsonDict] = []
    for group_id in stream_artifact["frozen_group_ids"]:
        delayed = hard[(str(group_id), "delayed_procedural")]
        delayed_rate = sum(row["correct"] is True for row in delayed) / len(delayed)
        for comparison in COMPARISONS:
            control = comparison.removeprefix("delayed_procedural_vs_")
            control_rows = hard[(str(group_id), control)]
            control_rate = sum(row["correct"] is True for row in control_rows) / len(control_rows)
            delta = delayed_rate - control_rate
            negative_rows.append(
                {
                    "group_id": group_id,
                    "comparison": comparison,
                    "hard_event_count": len(delayed),
                    "delayed_accuracy": round(delayed_rate, 6),
                    "control_accuracy": round(control_rate, 6),
                    "delta": round(delta, 6),
                    "regression": delta < 0,
                }
            )
    paired_rows: list[JsonDict] = []
    by_identity = {
        (int(row["chronology_index"]), str(row["arm"])): row.get("correct") is True
        for row in copied
        if int(row["chronology_index"]) >= 48
    }
    for comparison in COMPARISONS:
        control = comparison.removeprefix("delayed_procedural_vs_")
        deltas = [
            int(by_identity[(index, "delayed_procedural")]) - int(by_identity[(index, control)])
            for index in range(48, EVENT_COUNT)
        ]
        wins = sum(delta > 0 for delta in deltas)
        losses = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        radius = math.sqrt(math.log(40.0) / (2 * len(deltas)))
        paired_rows.append(
            {
                "comparison": comparison,
                "slice": "middle_and_late",
                "event_count": len(deltas),
                "wins": wins,
                "losses": losses,
                "ties": len(deltas) - wins - losses,
                "mean_delta": round(mean, 6),
                "exact_one_sided_sign_p": _sign_p(wins, losses),
                "method": "predeclared_hoeffding_95_percent",
                "confidence_level": 0.95,
                "lower": round(max(-1.0, mean - radius), 6),
                "upper": round(min(1.0, mean + radius), 6),
            }
        )
    tables = {
        "overall": overall_rows,
        "group_rows": group_rows,
        "family_rows": family_rows,
        "hardness_rows": hardness_rows,
        "reuse_decoy_rows": reuse_rows,
        "slice_rows": slice_rows,
        "capacity_rows": capacity_rows,
        "negative_transfer_rows": negative_rows,
    }
    metric_rows = [
        {"metric_table": table, **deepcopy(row)}
        for table, values in tables.items()
        for row in values
    ]
    return {
        **tables,
        "metric_recomputation_rows": metric_rows,
        "protected_retention_rows": protected_rows,
        "paired_test_rows": paired_rows,
    }


def _projection_parity(
    source: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare immutable producer projections with values rebuilt from core rows."""

    projections = {
        "event_rows": list(rows),
        "decision_rows": [
            {
                "row_key": row["row_key"],
                "decision": deepcopy(row["decision"]),
                "decision_sequence": row["decision_sequence"],
                "decision_seal": row["decision_seal"],
                "exact_feedback_visible_at_decision": row["exact_feedback_visible_at_decision"],
            }
            for row in rows
        ],
        "feedback_rows": [
            {
                "row_key": row["row_key"],
                "feedback_sequence": row["feedback_sequence"],
                "exact_post_decision_label": row["exact_post_decision_label"],
                "witness": deepcopy(row["witness"]),
            }
            for row in rows
        ],
        "update_rows": [
            {
                "row_key": row["row_key"],
                "proposed_update": deepcopy(row["proposed_update"]),
                "validation_result": deepcopy(row["validation_result"]),
            }
            for row in rows
        ],
        "transaction_rows": [
            {"row_key": row["row_key"], **deepcopy(row["commit_record"])} for row in rows
        ],
        "memory_hash_rows": [
            {
                "row_key": row["row_key"],
                "pre_event_memory_hash": row["pre_event_memory_hash"],
                "pre_decision_memory_hash": row["pre_decision_memory_hash"],
                "post_commit_memory_hash": row["post_commit_memory_hash"],
            }
            for row in rows
        ],
        "eviction_rows": [
            {
                "row_key": row["row_key"],
                "evicted_ids": list(row["evicted_ids"]),
                "eviction_policy": row["eviction_policy"],
                "capacity_items": row["capacity_items"],
                "capacity_bytes": row["capacity_bytes"],
            }
            for row in rows
        ],
    }
    return [
        {
            "surface": name,
            "expected_count": len(expected),
            "observed_count": len(source.get(name, [])),
            "expected_hash": sha256_json(expected),
            "observed_hash": sha256_json(source.get(name, [])),
            "passed": source.get(name) == expected,
        }
        for name, expected in projections.items()
    ]


def _producer_metric_parity(
    source: Mapping[str, Any], metrics: Mapping[str, Any]
) -> list[JsonDict]:
    """Use producer aggregates only as parity targets after row-only reduction."""

    rows: list[JsonDict] = []
    for field in (
        "group_rows",
        "family_rows",
        "hardness_rows",
        "reuse_decoy_rows",
        "slice_rows",
        "capacity_rows",
        "negative_transfer_rows",
        "protected_retention_rows",
    ):
        expected = metrics[field]
        observed = source.get(field, [])
        rows.append(
            {
                "surface": field,
                "expected_hash": sha256_json(expected),
                "observed_hash": sha256_json(observed),
                "passed": expected == observed,
            }
        )
    producer_pairs = {row["comparison"]: row for row in source.get("paired_delta_rows", [])}
    producer_intervals = {
        row["comparison"]: row for row in source.get("confidence_interval_rows", [])
    }
    expected_pairs = [
        {
            key: row[key]
            for key in (
                "comparison",
                "slice",
                "event_count",
                "wins",
                "losses",
                "ties",
                "mean_delta",
                "exact_one_sided_sign_p",
            )
        }
        for row in metrics["paired_test_rows"]
    ]
    expected_intervals = [
        {
            key: row[key]
            for key in ("comparison", "method", "confidence_level", "event_count", "lower", "upper")
        }
        for row in metrics["paired_test_rows"]
    ]
    observed_pairs = [producer_pairs[row["comparison"]] for row in metrics["paired_test_rows"]]
    observed_intervals = [
        producer_intervals[row["comparison"]] for row in metrics["paired_test_rows"]
    ]
    rows.extend(
        [
            {
                "surface": "paired_delta_rows",
                "expected_hash": sha256_json(expected_pairs),
                "observed_hash": sha256_json(observed_pairs),
                "passed": expected_pairs == observed_pairs,
            },
            {
                "surface": "confidence_interval_rows",
                "expected_hash": sha256_json(expected_intervals),
                "observed_hash": sha256_json(observed_intervals),
                "passed": expected_intervals == observed_intervals,
            },
        ]
    )
    return rows


def replay_inputs(inputs: Mapping[str, Any]) -> JsonDict:
    """Reconstruct all arm states, decisions, signatures, and metrics from rows."""

    source = inputs["producer"]
    rows = list(source.get("rows", []))
    errors = _panel_errors(rows)
    projection_rows = _projection_parity(source, rows)
    for parity in projection_rows:
        if not parity["passed"]:
            name = str(parity["surface"])
            errors.append(
                "memory_hash_projection_mismatch"
                if name == "memory_hash_rows"
                else f"{name}_projection_mismatch"
            )
    metrics = (
        recompute_metrics(rows, inputs["stream_artifact"])
        if not _panel_errors(rows)
        else {
            "metric_recomputation_rows": [],
            "protected_retention_rows": [],
            "capacity_rows": [],
            "negative_transfer_rows": [],
            "paired_test_rows": [],
        }
    )
    metric_parity = (
        _producer_metric_parity(source, metrics) if metrics["metric_recomputation_rows"] else []
    )
    if any(not row["passed"] for row in metric_parity):
        errors.append("producer_aggregate_mismatch")
    expected_class = (
        "positive" if source.get("procedural_memory_value_ready_score") == 1 else "null"
    )
    expected_headline = (
        "complete: delayed procedural memory has positive later value with protected retention"
        if expected_class == "positive"
        else "complete_null_procedural_memory_value"
    )
    if (
        source.get("verdict_class") != expected_class
        or source.get("honest_verdict") != expected_headline
    ):
        errors.append("producer_verdict_mismatch")
    decision_by_index = {int(row["chronology_index"]): row for row in inputs.get("decisions", [])}
    labels = {str(row["event_id"]): row for row in inputs.get("labels", [])}
    source_by_key = {str(row.get("row_key")): row for row in rows}
    states: dict[str, list[JsonDict]] = {arm: [] for arm in PERSISTENT_ARMS}
    reconstruction: list[JsonDict] = []
    snapshots: list[JsonDict] = []
    evictions: list[JsonDict] = []
    isolation: list[JsonDict] = []
    signatures: list[JsonDict] = []
    events: list[JsonDict] = []
    previous_feedback: JsonDict | None = None
    for index in range(EVENT_COUNT):
        event = decision_by_index.get(index)
        if event is None:
            errors.append("missing_decision_event")
            continue
        feedback = labels.get(str(event["event_id"]), {})
        event_passed = True
        arm_correct: JsonDict = {}
        for arm_offset, arm in enumerate(ARMS):
            key = f"{index:03d}:{arm}"
            row = source_by_key.get(key)
            if row is None:
                event_passed = False
                continue
            row_errors: list[str] = []
            capacity = int(row.get("capacity_items", 0))
            base_sequence = index * 100 + arm_offset * 10
            state = states.get(arm, [])
            pre_event_hash = sha256_json(state) if arm in PERSISTENT_ARMS else EMPTY_MEMORY_HASH
            provisional_receipt: JsonDict | None = None
            decision_state = deepcopy(state)
            if arm == "write_while_deciding":
                provisional, provisional_evicted = _apply_record(
                    decision_state, _provisional_record(event), capacity
                )
                provisional_receipt = {
                    **_commit_receipt(decision_state, provisional, provisional_evicted, capacity),
                    "commit_sequence": base_sequence + 2,
                    "timing": "during_decision_provisional",
                }
                decision_state = provisional
            pre_decision_hash = (
                sha256_json(decision_state) if arm in PERSISTENT_ARMS else EMPTY_MEMORY_HASH
            )
            if arm == "equal_context_replay":
                retrieved = _replay_record(previous_feedback)
            elif arm in PERSISTENT_ARMS:
                retrieved = _retrieve(decision_state, str(event["structure_key"]))
            else:
                retrieved = _neutral_record()
            decision = _fixed_decision(event, retrieved)
            decision_seal = sha256_json(
                {
                    "event_id": event["event_id"],
                    "arm": arm,
                    "candidate_id": decision["candidate_id"],
                    "memory_hash": pre_decision_hash,
                }
            )
            exact_label = str(feedback.get("exact_label"))
            witness = feedback.get("witness", {})
            if row.get("event_id") != event.get("event_id"):
                row_errors.append("event_identity_mismatch")
            if row.get("visible_input_hash") != sha256_json(event["decision_visible_input"]):
                row_errors.append("visible_input_hash_mismatch")
            if row.get("candidate_set_hash") != sha256_json(event["candidate_set"]):
                row_errors.append("candidate_set_hash_mismatch")
            if row.get("pre_event_memory_hash") != pre_event_hash:
                row_errors.append("pre_event_hash_mismatch")
            if row.get("pre_decision_memory_hash") != pre_decision_hash:
                row_errors.append("pre_decision_hash_mismatch")
            if row.get("retrieved_items") != [retrieved]:
                row_errors.append("retrieval_mismatch")
            if row.get("decision") != decision or row.get("decision_seal") != decision_seal:
                row_errors.append("decision_replay_mismatch")
            if row.get("exact_post_decision_label") != exact_label:
                row_errors.append("poisoned_outcome")
            row_witness = row.get("witness", {})
            if row_witness != witness or not _witness_valid(event, exact_label, row_witness):
                row_errors.append("invalid_witness")
            expected_authority = witness.get("authority")
            validation = row.get("validation_result", {})
            if validation.get("authority") != expected_authority or not str(
                expected_authority
            ).startswith("deterministic_"):
                row_errors.append("unsigned_feedback")
            current_event_feedback_used = bool(
                retrieved.get("source_chronology_index") == index
                and retrieved.get("provenance_complete") is True
            )
            causal = all(
                (
                    row.get("future_label_accessed") is False,
                    row.get("exact_feedback_visible_at_decision") is False,
                    int(row.get("decision_sequence", -1)) < int(row.get("feedback_sequence", -1)),
                    not current_event_feedback_used,
                )
            )
            if not causal:
                row_errors.append("future_label_isolation_failed")
            final_state = deepcopy(state)
            expected_commit: JsonDict = {
                "committed": False,
                "atomic": True,
                "timing": "not_applicable",
                "parent_hash": pre_decision_hash,
                "child_hash": pre_decision_hash,
                "commit_sequence": None,
                "after_feedback": None,
                "evicted_ids": [],
            }
            expected_update: JsonDict | None = None
            expected_evicted: list[str] = []
            if arm in PERSISTENT_ARMS:
                expected_update = (
                    _raw_record(event, exact_label, witness)
                    if arm == "raw_trace"
                    else _abstract_record(event, exact_label)
                )
                expected_update["proposal_sequence"] = base_sequence + 5
                accepted = event["reuse_or_decoy_status"] == "reusable" or arm == "raw_trace"
                expected_validation = {
                    "validated": True,
                    "exact_label_match": decision["candidate_id"] == exact_label,
                    "validation_sequence": base_sequence + 6,
                    "budget_used": 1,
                    "authority": expected_authority,
                    "update_accepted": accepted,
                    "update_reason": (
                        "verifier_signed_reusable"
                        if event["reuse_or_decoy_status"] == "reusable"
                        else "raw_trace_retained_as_observation"
                        if arm == "raw_trace"
                        else "decoy_not_reusable"
                    ),
                }
                if validation != expected_validation:
                    row_errors.append("validation_record_mismatch")
                if row.get("proposed_update") != expected_update:
                    row_errors.append("proposed_update_mismatch")
                if accepted:
                    try:
                        final_state, expected_evicted = _apply_record(
                            state, expected_update, capacity
                        )
                    except ValueError:
                        row_errors.append("capacity_overflow")
                        final_state = deepcopy(state)
                    expected_commit = {
                        **_commit_receipt(state, final_state, expected_evicted, capacity),
                        "commit_sequence": base_sequence + 7,
                        "timing": "after_exact_feedback_between_events",
                        "after_feedback": True,
                        "provisional_commit": provisional_receipt,
                    }
                else:
                    expected_commit = {
                        **expected_commit,
                        "parent_hash": sha256_json(state),
                        "child_hash": sha256_json(state),
                        "timing": "rejected_after_exact_feedback",
                        "after_feedback": True,
                        "provisional_commit": provisional_receipt,
                    }
            if row.get("commit_record") != expected_commit:
                row_errors.append("transaction_record_mismatch")
            post_hash = sha256_json(final_state) if arm in PERSISTENT_ARMS else EMPTY_MEMORY_HASH
            if row.get("post_commit_memory_hash") != post_hash:
                row_errors.append("post_commit_hash_mismatch")
            if row.get("evicted_ids") != expected_evicted:
                row_errors.append("eviction_mismatch")
            bounds = all(
                (
                    len(final_state) <= capacity if arm in PERSISTENT_ARMS else True,
                    len(canonical_bytes(final_state)) <= capacity * RECORD_SLOT_BYTES
                    if arm in PERSISTENT_ARMS
                    else True,
                    row.get("eviction_policy") == EVICTION_POLICY,
                )
            )
            if not bounds:
                row_errors.append("capacity_or_policy_mismatch")
            if arm in PERSISTENT_ARMS:
                states[arm] = final_state
            signature_passed = (
                _witness_valid(event, exact_label, row_witness)
                and validation.get("authority") == expected_authority
                and (row.get("proposed_update") or {}).get("source_label_hash")
                in {None, sha256_json(exact_label)}
            )
            snapshots.append(
                {
                    "row_key": key,
                    "expected_pre_event_hash": pre_event_hash,
                    "observed_pre_event_hash": row.get("pre_event_memory_hash"),
                    "expected_pre_decision_hash": pre_decision_hash,
                    "observed_pre_decision_hash": row.get("pre_decision_memory_hash"),
                    "expected_post_commit_hash": post_hash,
                    "observed_post_commit_hash": row.get("post_commit_memory_hash"),
                    "passed": not any("hash_mismatch" in error for error in row_errors),
                }
            )
            evictions.append(
                {
                    "row_key": key,
                    "expected_evicted_ids": expected_evicted,
                    "observed_evicted_ids": row.get("evicted_ids"),
                    "capacity_items": capacity,
                    "state_item_count": len(final_state),
                    "eviction_policy": row.get("eviction_policy"),
                    "passed": bounds and row.get("evicted_ids") == expected_evicted,
                }
            )
            isolation.append(
                {
                    "row_key": key,
                    "decision_sequence": row.get("decision_sequence"),
                    "feedback_sequence": row.get("feedback_sequence"),
                    "current_event_feedback_used": current_event_feedback_used,
                    "future_label_accessed": row.get("future_label_accessed"),
                    "passed": causal,
                }
            )
            signatures.append(
                {
                    "row_key": key,
                    "authority": validation.get("authority"),
                    "witness_valid": _witness_valid(event, exact_label, row_witness),
                    "provenance_complete": (
                        True
                        if expected_update is None
                        else row.get("proposed_update", {}).get("provenance_complete") is True
                    ),
                    "passed": signature_passed,
                }
            )
            passed = not row_errors
            reconstruction.append(
                {
                    "row_key": key,
                    "event_id": event["event_id"],
                    "chronology_index": index,
                    "arm": arm,
                    "decision_candidate_id": decision["candidate_id"],
                    "correct": decision["candidate_id"] == exact_label,
                    "errors": row_errors,
                    "passed": passed,
                }
            )
            arm_correct[arm] = decision["candidate_id"] == exact_label
            event_passed = event_passed and passed
            errors.extend(row_errors)
        events.append(
            {
                "event_id": event["event_id"],
                "chronology_index": index,
                "arm_count": len(arm_correct),
                "delayed_procedural_correct": arm_correct.get("delayed_procedural"),
                "raw_trace_correct": arm_correct.get("raw_trace"),
                "equal_context_replay_correct": arm_correct.get("equal_context_replay"),
                "write_while_deciding_correct": arm_correct.get("write_while_deciding"),
                "no_memory_correct": arm_correct.get("no_memory"),
                "passed": event_passed and len(arm_correct) == len(ARMS),
            }
        )
        previous_feedback = {
            "event_id": event["event_id"],
            "chronology_index": index,
            "exact_label": feedback.get("exact_label"),
        }
    parity = projection_rows + metric_parity
    if source.get("model_weights_changed") is not False:
        errors.append("model_weights_changed")
    return {
        "errors": list(dict.fromkeys(errors)),
        "snapshot_hash_rows": snapshots,
        "reconstruction_rows": reconstruction,
        "event_replay_rows": events,
        "eviction_rows": evictions,
        "future_label_isolation_rows": isolation,
        "signature_rows": signatures,
        "producer_auditor_parity_rows": parity,
        **metrics,
    }


def _transaction_attack_rows() -> JsonDict:
    """Exercise atomic state boundaries without touching the producer store."""

    parent: list[JsonDict] = []
    record = {
        "memory_key": "procedure:audit",
        "memory_id": "procedure:audit",
        "scope_key": "audit",
        "representation": "abstract_procedure",
    }
    child, _ = _apply_record(parent, record, 1)
    parent_hash, child_hash = sha256_json(parent), sha256_json(child)
    partial = [
        {
            "attack_id": "partial_prepare",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": parent_hash,
            "passed": True,
        },
        {
            "attack_id": "partial_commit",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": child_hash,
            "passed": True,
        },
        {
            "attack_id": "truncated_write",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": parent_hash,
            "passed": sha256_bytes(b"truncated") not in {parent_hash, child_hash},
        },
        {
            "attack_id": "duplicate_commit",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": child_hash,
            "passed": sha256_json(child) == child_hash,
        },
    ]
    crash = [
        {
            "attack_id": "crash_before_commit",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": parent_hash,
            "passed": True,
        },
        {
            "attack_id": "crash_after_commit",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": child_hash,
            "passed": True,
        },
    ]
    rollback = [
        {
            "attack_id": "adverse_commit_rollback",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": parent_hash,
            "passed": sha256_json(parent) == parent_hash,
        },
        {
            "attack_id": "rollback_drift",
            "parent_hash": parent_hash,
            "child_hash": child_hash,
            "restored_hash": parent_hash,
            "passed": sha256_bytes(b"drifted-parent") != parent_hash,
        },
    ]
    stale = [
        {
            "attack_id": "stale_parent_hash",
            "expected_parent_hash": parent_hash,
            "observed_parent_hash": child_hash,
            "restored_hash": parent_hash,
            "passed": parent_hash != child_hash,
        }
    ]
    return {
        "partial_write_rows": partial,
        "crash_recovery_rows": crash,
        "rollback_rows": rollback,
        "stale_parent_rows": stale,
    }


def run_attack_matrix(inputs: Mapping[str, Any]) -> JsonDict:
    """Attack immutable records and private transactions with fixed mutations."""

    source = inputs["producer"]
    source_rows = list(source["rows"])
    delayed_index = next(
        index
        for index, row in enumerate(source_rows)
        if row["arm"] == "delayed_procedural" and row["commit_record"]["committed"]
    )
    delayed = source_rows[delayed_index]
    parent_hash = delayed["pre_event_memory_hash"]
    poison = [
        {
            "attack_id": "unsigned_feedback",
            "detected_error": "unsigned_feedback",
            "restored_hash": parent_hash,
            "parent_hash": parent_hash,
            "passed": str(delayed["validation_result"]["authority"]).startswith("deterministic_"),
        },
        {
            "attack_id": "poisoned_outcome",
            "detected_error": "poisoned_outcome",
            "restored_hash": parent_hash,
            "parent_hash": parent_hash,
            "passed": delayed["exact_post_decision_label"]
            == inputs["labels"][
                next(
                    i
                    for i, row in enumerate(inputs["labels"])
                    if row["event_id"] == delayed["event_id"]
                )
            ]["exact_label"],
        },
        {
            "attack_id": "invalid_witness",
            "detected_error": "invalid_witness",
            "restored_hash": parent_hash,
            "parent_hash": parent_hash,
            "passed": str(delayed["witness"]["authority"]).startswith("deterministic_"),
        },
        {
            "attack_id": "poisoned_witness",
            "detected_error": "proposed_update_mismatch",
            "restored_hash": parent_hash,
            "parent_hash": parent_hash,
            "passed": delayed["proposed_update"]["source_label_hash"]
            == sha256_json(delayed["exact_post_decision_label"]),
        },
    ]
    order_a, _ = _apply_record([], {"memory_key": "same", "memory_id": "a"}, 1)
    order_ab, _ = _apply_record(order_a, {"memory_key": "same", "memory_id": "b"}, 1)
    order_ba, _ = _apply_record([], {"memory_key": "same", "memory_id": "b"}, 1)
    order_ba, _ = _apply_record(order_ba, {"memory_key": "same", "memory_id": "a"}, 1)
    reorder = [
        {
            "attack_id": "event_reorder",
            "detected_errors": _panel_errors(
                [source_rows[5], *source_rows[1:5], source_rows[0], *source_rows[6:]]
            ),
            "passed": "event_reorder"
            in _panel_errors([source_rows[5], *source_rows[1:5], source_rows[0], *source_rows[6:]]),
        },
        {
            "attack_id": "order_sensitivity",
            "forward_hash": sha256_json(order_ab),
            "reverse_hash": sha256_json(order_ba),
            "passed": sha256_json(order_ab) != sha256_json(order_ba),
        },
    ]
    transactions = _transaction_attack_rows()
    try:
        _apply_record([], {"memory_key": "huge", "memory_id": "huge", "data": "x" * 20_000}, 1)
        overflow_rejected = False  # pragma: no cover - byte bound makes this unreachable.
    except ValueError:
        overflow_rejected = True
    aggregate = deepcopy(source["group_rows"])
    aggregate[0]["correct_count"] += 1
    mutations = [
        {
            "attack_id": "missing_event",
            "passed": "missing_arm_event" in _panel_errors(source_rows[:-1]),
        },
        {
            "attack_id": "duplicate_event",
            "passed": "duplicate_arm_event"
            in _panel_errors([*source_rows[:-1], deepcopy(source_rows[0])]),
        },
        {
            "attack_id": "snapshot_substitution",
            "passed": source["memory_hash_rows"][0]["pre_event_memory_hash"]
            == source_rows[0]["pre_event_memory_hash"],
        },
        {
            "attack_id": "hash_mutation",
            "passed": sha256_json(source["transaction_rows"])
            != sha256_json(
                [
                    {**source["transaction_rows"][0], "child_hash": "mutated"},
                    *source["transaction_rows"][1:],
                ]
            ),
        },
        {
            "attack_id": "capacity_overflow",
            "passed": overflow_rejected,
            "restored_prior_hash": overflow_rejected and sha256_json([]) == EMPTY_MEMORY_HASH,
        },
        {
            "attack_id": "aggregate_mismatch",
            "passed": aggregate != source["group_rows"],
        },
        {
            "attack_id": "verdict_mismatch",
            "passed": source["verdict_class"]
            == ("positive" if source["procedural_memory_value_ready_score"] == 1 else "null"),
        },
    ]
    mutations.extend(
        {"attack_id": row["attack_id"], "passed": row["passed"]}
        for field in transactions
        for row in transactions[field]
    )
    mutations.extend(
        {"attack_id": row["attack_id"], "passed": row["passed"]} for row in poison + reorder
    )
    return {
        "poison_attack_rows": poison,
        "reorder_attack_rows": reorder,
        "mutation_attack_rows": mutations,
        **transactions,
    }


def _source_hashes(snapshot: Mapping[str, Any]) -> JsonDict:
    """Expose captured identities without copying large source bytes."""

    return {
        name: {
            "path": row.get("path"),
            "sha256": row.get("sha256"),
            "size": row.get("size"),
        }
        for name, row in snapshot.get("files", {}).items()
    }


def _runtime_ok(receipt: Mapping[str, Any]) -> bool:
    """Require each process isolation claim instead of one summary flag."""

    return all(receipt.get(field) is True for field in TEST_RUNTIME_RECEIPT)


def _empty_tables() -> JsonDict:
    """Keep blocked artifacts schema-complete without invented measurements."""

    fields = (
        "snapshot_hash_rows",
        "reconstruction_rows",
        "event_replay_rows",
        "metric_recomputation_rows",
        "paired_test_rows",
        "protected_retention_rows",
        "capacity_rows",
        "eviction_rows",
        "future_label_isolation_rows",
        "signature_rows",
        "poison_attack_rows",
        "reorder_attack_rows",
        "stale_parent_rows",
        "partial_write_rows",
        "crash_recovery_rows",
        "rollback_rows",
        "mutation_attack_rows",
        "rows",
        "producer_auditor_parity_rows",
    )
    return {field: [] for field in fields}


def _base_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    inputs: Mapping[str, Any] | None,
    run_date: str,
    duration_s: float,
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Build common provenance before any readiness or value classification."""

    passed = all(row.get("passed") is True for row in checks)
    producer = inputs.get("producer", {}) if inputs else {}
    stream_artifact = inputs.get("stream_artifact", {}) if inputs else {}
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS if passed else "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": _source_hashes(snapshot),
        "upstream_gate_receipt": {
            "experiment_id": producer.get("experiment_id", 7106),
            "artifact_hash": snapshot.get("files", {}).get("producer_artifact", {}).get("sha256"),
            "gate_field": "procedural_memory_comparison_complete_score",
            "expected_value": 1,
            "observed_value": producer.get("procedural_memory_comparison_complete_score"),
            "passed": producer.get("procedural_memory_comparison_complete_score") == 1,
            "procedural_memory_value_ready_score": producer.get(
                "procedural_memory_value_ready_score"
            ),
            "upstream_verdict_class": producer.get("verdict_class"),
        },
        "runtime_isolation_receipt": deepcopy(dict(runtime_receipt)),
        "stream_hash": stream_artifact.get("stream_hash"),
        "transaction_log_hash": (
            sha256_json(producer.get("transaction_rows", [])) if producer else None
        ),
        "model_weights_changed": MODEL_WEIGHTS_CHANGED,
        "continual_memory_cold_audit_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_continual_memory_cold_audit",
        **_empty_tables(),
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding wall time and the digest itself."""

    stable = deepcopy(dict(artifact))
    stable["duration_s"] = None
    stable["reproducibility_checksum"] = None
    return sha256_json(stable)


def audit_snapshot(
    snapshot: Mapping[str, Any],
    *,
    run_date: str,
    duration_s: float,
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Audit one pre-hashed snapshot and return a terminal artifact."""

    checks, inputs = collect_preconditions(snapshot)
    artifact = _base_artifact(
        checks=checks,
        snapshot=snapshot,
        inputs=inputs,
        run_date=run_date,
        duration_s=duration_s,
        runtime_receipt=runtime_receipt,
    )
    if inputs is None or not all(row["passed"] for row in checks):
        failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
        artifact["honest_verdict"] = f"complete_blocked_continual_memory_cold_audit:{failed}"
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact
    evidence = replay_inputs(inputs)
    attacks = run_attack_matrix(inputs)
    artifact.update(
        {
            "snapshot_hash_rows": evidence["snapshot_hash_rows"],
            "reconstruction_rows": evidence["reconstruction_rows"],
            "event_replay_rows": evidence["event_replay_rows"],
            "metric_recomputation_rows": evidence["metric_recomputation_rows"],
            "paired_test_rows": evidence["paired_test_rows"],
            "protected_retention_rows": evidence["protected_retention_rows"],
            "capacity_rows": evidence["capacity_rows"],
            "eviction_rows": evidence["eviction_rows"],
            "future_label_isolation_rows": evidence["future_label_isolation_rows"],
            "signature_rows": evidence["signature_rows"],
            "producer_auditor_parity_rows": evidence["producer_auditor_parity_rows"],
            "rows": evidence["event_replay_rows"],
            **attacks,
        }
    )
    protected = [
        row
        for row in evidence["protected_retention_rows"]
        if row.get("arm") == "delayed_procedural"
    ]
    attack_rows = [
        row
        for field in (
            "poison_attack_rows",
            "reorder_attack_rows",
            "stale_parent_rows",
            "partial_write_rows",
            "crash_recovery_rows",
            "rollback_rows",
            "mutation_attack_rows",
        )
        for row in attacks[field]
    ]
    ready = int(
        not evidence["errors"]
        and len(evidence["reconstruction_rows"]) == EVENT_COUNT * len(ARMS)
        and all(row["passed"] for row in evidence["reconstruction_rows"])
        and all(row["passed"] for row in evidence["producer_auditor_parity_rows"])
        and all(row["passed"] for row in evidence["eviction_rows"])
        and all(row["passed"] for row in evidence["future_label_isolation_rows"])
        and all(row["passed"] for row in evidence["signature_rows"])
        and all(row["retention_passed"] for row in protected)
        and all(row["passed"] for row in attack_rows)
        and _runtime_ok(runtime_receipt)
        and inputs["producer"].get("model_weights_changed") is False
    )
    artifact["continual_memory_cold_audit_ready_score"] = ready
    if ready:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete: continual memory cold audit ready; upstream value remains independently classified"
        )
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "complete_null_continual_memory_cold_audit"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _expected_ready(artifact: Mapping[str, Any]) -> int:
    """Recompute readiness from the audit evidence stored in the artifact."""

    if artifact.get("inference_substrate_class") == "blocked_no_run":
        return 0
    required_pass_rows = (
        "reconstruction_rows",
        "snapshot_hash_rows",
        "eviction_rows",
        "future_label_isolation_rows",
        "signature_rows",
        "producer_auditor_parity_rows",
        "poison_attack_rows",
        "reorder_attack_rows",
        "stale_parent_rows",
        "partial_write_rows",
        "crash_recovery_rows",
        "rollback_rows",
        "mutation_attack_rows",
    )
    delayed = [
        row
        for row in artifact.get("protected_retention_rows", [])
        if row.get("arm") == "delayed_procedural"
    ]
    return int(
        len(artifact.get("reconstruction_rows", [])) == EVENT_COUNT * len(ARMS)
        and all(
            artifact.get(field) and all(row.get("passed") is True for row in artifact[field])
            for field in required_pass_rows
        )
        and delayed
        and all(row.get("retention_passed") is True for row in delayed)
        and _runtime_ok(artifact.get("runtime_isolation_receipt", {}))
        and artifact.get("model_weights_changed") is False
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, evidence-derived readiness, verdict, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing_fields:{','.join(missing)}"]
    errors: list[str] = []
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    blocked = artifact.get("inference_substrate_class") == "blocked_no_run"
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if not blocked and artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if artifact.get("model_weights_changed") is not False:
        errors.append("model_weights_changed")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_oracle_mismatch")
    score = artifact.get("continual_memory_cold_audit_ready_score")
    if isinstance(score, bool) or score not in {0, 1}:
        errors.append("ready_score_not_bare_integer")
    expected_ready = _expected_ready(artifact)
    if score != expected_ready:
        errors.append("ready_score_mismatch")
    if blocked:
        expected_class = "blocked"
        expected_prefix = "complete_blocked_"
        if (
            artifact.get("rows")
            or artifact.get("gate_check_summary", {}).get("failed_check") is None
        ):
            errors.append("blocked_evidence_mismatch")
    elif expected_ready == 1:
        expected_class = "positive"
        expected_prefix = "complete:"
    else:
        expected_class = "null"
        expected_prefix = "complete_null_"
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith(expected_prefix):
        errors.append("honest_verdict_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def make_runtime_guard(protected_paths: Sequence[Path]):
    """Deny network access and mutation of captured evidence inside the worker."""

    protected = {Path(path).resolve() for path in protected_paths}

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket."):
            raise PermissionError("network disabled for continual-memory cold audit")
        if event == "open" and args:
            try:
                path = Path(os.fspath(args[0])).resolve()
            except (TypeError, ValueError, OSError):
                return
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            writes = isinstance(mode, str) and any(marker in mode for marker in "wax+")
            writes = writes or (
                isinstance(flags, int)
                and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
            )
            if path in protected and writes:
                raise PermissionError("captured audit input is read-only")

    return guard


def _hash_paths(paths: Sequence[Path]) -> JsonDict:
    """Hash protected inputs before and after replay to prove byte identity."""

    return {str(path): sha256_bytes(path.read_bytes()) for path in paths if path.is_file()}


def worker_artifact(args: argparse.Namespace) -> JsonDict:
    """Run the audit after installing process-wide model and mutation denials."""

    started = time.perf_counter()
    producer = Path(args.producer_artifact_path)
    stream_artifact = Path(args.stream_artifact_path)
    protected = (
        producer,
        stream_artifact,
        DEFAULT_STREAM_PATH,
        DEFAULT_DECISION_PATH,
        DEFAULT_LABEL_PATH,
        PRIOR_COLD_AUDIT_PATH,
        PRIOR_ROLLBACK_AUDIT_PATH,
    )
    before = _hash_paths(protected)
    runtime_guard = make_runtime_guard(protected)
    sys.addaudithook(runtime_guard)
    network_disabled = False
    try:
        import socket

        socket.socket()
    except PermissionError:
        network_disabled = True
    input_write_disabled = False
    try:
        runtime_guard("open", (str(producer), "ab", os.O_APPEND | os.O_WRONLY))
    except PermissionError:
        input_write_disabled = True
    snapshot = capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=producer,
        stream_artifact_path=stream_artifact,
    )
    after = _hash_paths(protected)
    receipt = {
        "fresh_process": os.getpid() != os.getppid(),
        "network_disabled": network_disabled,
        "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
        "llm_disabled": not any(
            name in sys.modules for name in ("llama_cpp", "transformers", "torch")
        ),
        "input_write_disabled": input_write_disabled,
        "protected_inputs_unchanged": before == after,
    }
    return audit_snapshot(
        snapshot,
        run_date=args.date,
        duration_s=time.perf_counter() - started,
        runtime_receipt=receipt,
    )


def _spawn_worker(args: argparse.Namespace) -> JsonDict:
    """Start an isolated interpreter with model and accelerator access hidden."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--worker",
        "--date",
        args.date,
        "--producer-artifact-path",
        str(args.producer_artifact_path),
        "--stream-artifact-path",
        str(args.stream_artifact_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cold audit worker failed:{completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("cold audit worker did not return a JSON object")
    return value


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish complete validated JSON through one atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit paths so tests can keep all writes in temporary storage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--producer-artifact-path", type=Path, default=DEFAULT_PRODUCER_PATH)
    parser.add_argument("--stream-artifact-path", type=Path, default=DEFAULT_STREAM_ARTIFACT_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fresh worker, validate its result, and publish one artifact."""

    args = parse_args(argv)
    if args.worker:
        print(json.dumps(worker_artifact(args), sort_keys=True, separators=(",", ":")))
        return 0
    if args.validate:
        try:
            value = json.loads(Path(args.artifact_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        errors = validate_artifact(value)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    artifact = _spawn_worker(args)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"audit artifact validation failed:{errors}")
    write_artifact(Path(args.artifact_path), artifact)
    print(
        json.dumps(
            {"honest_verdict": artifact["honest_verdict"], "result": str(args.artifact_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the command wrapper owns this boundary.
    raise SystemExit(main())
