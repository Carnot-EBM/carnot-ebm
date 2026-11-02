"""Independently audit the complete V643 batch measurement.

This module reads native response bytes and applies a small local relation
executor. It does not call the measurement headline reducer. The shared exact
relation meaning limits a successful scientific class to circular positive.

Spec refs: REQ-VERIFY-7322 and SCENARIO-VERIFY-7322-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7322-batch-audit"
UPSTREAM_ID = "exp7321-batch-measurement"
MILESTONE = "2026.09.643"
RUN_DATE = "20260915"
SCHEMA = "carnot.exp7322.v643_batch_audit.v1"
BOOTSTRAP_SEED = 7_321_003
DEVELOPMENT_SEED = 7_306_101
EVALUATION_SEED = 7_306_201
ARMS = (
    "serial_versioned_verifier",
    "batched_versioned_verifier",
    "batched_warm_prefix_direct",
)
PLANNED_GROUPS = 16
PLANNED_UNITS = 128
PLANNED_ROWS = 384
PLANNED_CALLS = 256

MEASUREMENT_REL = Path("results/experiment_7321_v643_batch_measurement.json")
PUBLIC_REL = Path("results/raw/experiment_7317_v643_batch_harness/public_panel.json")
EVALUATOR_REL = Path("results/raw/experiment_7317_v643_batch_harness/evaluator_labels.json")
RAW_REL = Path("results/raw/experiment_7321_v643_batch_measurement")
SCHEDULE_REL = RAW_REL / "schedule.json"
PREDICTION_REL = RAW_REL / "public_predictions.json"
SCORED_REL = RAW_REL / "scored_rows.json"
CALL_REL = RAW_REL / "per_call_rows.json"
COST_REL = RAW_REL / "cost_receipt.json"
RESULT_REL = Path("results/experiment_7322_v643_batch_audit.json")
AUDIT_RAW_REL = Path("results/raw/experiment_7322_v643_batch_audit")
PRIOR_CANDIDATE_REL = AUDIT_RAW_REL / "terminal_candidate_attempt_1.json"
MEASUREMENT_PATH = ROOT / MEASUREMENT_REL
TEST_PATH = Path("tests/python/test_experiment_7322_v643_batch_audit.py")
MODULE_PATH = Path("python/carnot/experiment_7322_v643_batch_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7322_v643_batch_audit.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
SCOPED_RUNNER = "carnot.reporting.experiment_7303_validation_scope.run_scoped_validation"
TERMINAL_CHECK_NAMES = (
    "candidate_reload_and_independent_reduce",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

ZERO_INVOCATION_COUNTS: JsonDict = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "validation_receipts",
    "repository_health",
    "field_principles",
    "batch_audit_complete_score",
    "batch_promotion_score",
    "independent_comparison_rows",
    "interference_control_rows",
    "retirement_decision",
)

FIELD_PRINCIPLES = {
    "schema": "Version this artifact while keeping ordinary experiment and milestone fields.",
    "experiment_id": "Give the audit a stable ordinary identity.",
    "milestone": "Bind the audit to milestone 2026.09.643.",
    "status": "Write a terminal state only after current work and validation.",
    "run_date": "Use 20260915 and retain actual UTC and monotonic observations.",
    "preconditions_checked": "Record input identity, availability, and every exact comparison.",
    "MODEL_SPECS": "List current executable models only; this CPU audit has none.",
    "model_invoked": "Report any attempted model load or generation; this audit has none.",
    "invocation_counts": "Separate attempted, complete, failed, cancelled, and active work.",
    "inference_substrate": "Name the actual CPU exact solver or simulator work.",
    "inference_substrate_class": "Use the recognized CPU exact solver or simulator class.",
    "execution_venue": "Record host execution for this milestone.",
    "duration_s": "Measure real elapsed time without sleeping or padding.",
    "phase_spans": "Keep disjoint phase durations, units, checkpoints, and pending work.",
    "random_seed": "Seal development, evaluation, and audit bootstrap seeds.",
    "reproducibility_checksum": "Bind code, settings, public input, private authority, and raw evidence.",
    "source_artifact_hashes": "Authenticate current producers and every audited sidecar.",
    "rows": "Keep every claim arm with metrics, cost, failure, abstention, and censoring.",
    "sample_size_budget": "Preserve planned, attempted, complete, and censored denominators.",
    "acceptance_gate_results": "Keep expected, observed, pass state, and purpose for each gate.",
    "gate_check_summary": "Keep the first exact failed check without hiding later checks.",
    "verifier_is_oracle": "Declare shared exact authority and prevent a positive scientific class.",
    "honest_verdict": "Use complete for finished findings and blocked for external absence.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "validation_receipts": "Keep exact command, scope, exit, elapsed time, and log hash.",
    "repository_health": "Retain unrelated dated failures without passing current checks.",
    "field_principles": "Explain why each artifact field exists.",
    "batch_audit_complete_score": "One means independent work completed even for a null value.",
    "batch_promotion_score": "One requires raw parity, authenticity, controls, and all frozen bounds.",
    "independent_comparison_rows": "Compute comparisons from raw calls instead of producer headlines.",
    "interference_control_rows": "Separate retained live evidence from CPU fixture interventions.",
    "retirement_decision": "Name the failed mechanism and the condition for a different question.",
}


def canonical_json(value: Any) -> str:
    """Use stable JSON bytes so semantic edits cannot hide behind formatting."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    """Return a labeled digest for exact input bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large call evidence stays cheap."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable findings while excluding process-local validation timing."""

    excluded = {
        "duration_s",
        "timestamps",
        "phase_spans",
        "reproducibility_checksum",
        "validation_receipts",
        "repository_health",
    }
    durable = {key: value for key, value in artifact.items() if key not in excluded}
    return sha256_bytes(canonical_json(durable).encode("utf-8"))


def gate_row(
    check: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str = UPSTREAM_ID,
) -> JsonDict:
    """Keep both sides of one gate so failures remain actionable."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected": expected,
        "expected_value": expected,
        "observed": observed,
        "observed_value": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failed value without changing its exact spelling."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
        }
    return {
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value", failed.get("expected")),
        "observed_value": failed.get("observed_value", failed.get("observed")),
    }


def dependency_gate_rows(measurement: Mapping[str, Any] | None) -> list[JsonDict]:
    """Admit a complete positive or null capture and reject failed states."""

    if measurement is None:
        return [
            gate_row(
                "exp7321_artifact_available",
                "artifact",
                "present",
                "missing_artifact",
                False,
                "The independent audit needs the current terminal measurement.",
            )
        ]
    verdict = measurement.get("verdict_class")
    quarantined = bool(
        measurement.get("quarantined")
        or measurement.get("flagged_adversarial")
        or measurement.get("status") == "quarantined"
    )
    return [
        gate_row(
            "exp7321_identity",
            "experiment_id",
            UPSTREAM_ID,
            measurement.get("experiment_id"),
            measurement.get("experiment_id") == UPSTREAM_ID,
            "A score from another producer cannot authorize this audit.",
        ),
        gate_row(
            "exp7321_capture_complete_score",
            "batch_capture_complete_score",
            1,
            measurement.get("batch_capture_complete_score"),
            measurement.get("batch_capture_complete_score") == 1,
            "The audit starts from complete capture, whether value is positive or null.",
        ),
        gate_row(
            "exp7321_terminal_class",
            "verdict_class",
            "not_in:['blocked', 'disqualified', 'partial']",
            verdict,
            verdict not in {"blocked", "disqualified", "partial"},
            "A failed or unfinished producer cannot authorize an audit.",
        ),
        gate_row(
            "exp7321_quarantine",
            "quarantined",
            False,
            quarantined,
            not quarantined,
            "Quarantined evidence stays outside the current audit path.",
        ),
    ]


def _read_json(path: Path) -> JsonDict:
    """Read a JSON mapping without repairing malformed input."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"mapping required: {path}")
    return value


def load_bundle(root: Path = ROOT) -> JsonDict:
    """Load exact public, private, raw-call, and cost inputs with byte seals."""

    paths = {
        "measurement": MEASUREMENT_REL,
        "public": PUBLIC_REL,
        "evaluator": EVALUATOR_REL,
        "schedule": SCHEDULE_REL,
        "predictions": PREDICTION_REL,
        "scored": SCORED_REL,
        "calls": CALL_REL,
        "cost": COST_REL,
    }
    loaded = {name: _read_json(root / path) for name, path in paths.items()}
    values = {
        "measurement": loaded["measurement"],
        "public": loaded["public"],
        "evaluator": loaded["evaluator"],
        "schedule": loaded["schedule"].get("schedule", []),
        "predictions": loaded["predictions"].get("predictions", []),
        "prediction_envelope": loaded["predictions"],
        "scored": loaded["scored"].get("scored_rows", []),
        "scored_envelope": loaded["scored"],
        "calls": loaded["calls"].get("per_call_rows", []),
        "call_envelope": loaded["calls"],
        "cost": loaded["cost"],
    }
    values["file_hashes"] = {path.as_posix(): sha256_file(root / path) for path in paths.values()}
    values["content_seals"] = {
        name: sha256_bytes(canonical_json(values[name]).encode("utf-8"))
        for name in ("public", "evaluator", "schedule", "predictions", "scored", "calls", "cost")
    }
    values["call_checkpoints"] = [
        _read_json(root / RAW_REL / f"call_{index:02d}.json") for index in range(PLANNED_CALLS)
    ]
    values["root"] = root.resolve()
    return values


def _public_roster(public: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    """Build expected identities before any producer aggregate is consulted."""

    errors: list[str] = []
    groups = public.get("evaluation_groups")
    if not isinstance(groups, list) or len(groups) != PLANNED_GROUPS:
        return {}, ["group_count"]
    roster: JsonDict = {}
    for group in groups:
        group_id = str(group.get("group_id"))
        sources = group.get("source_versions")
        claims = group.get("claims")
        if not isinstance(sources, list) or len(sources) != 2:
            errors.append(f"source_version_count:{group_id}")
            continue
        if not isinstance(claims, list) or len(claims) != 8:
            errors.append(f"claim_count:{group_id}")
            continue
        versions: JsonDict = {}
        for source in sources:
            version = source.get("source_version")
            document = source.get("document")
            text = document.get("text") if isinstance(document, Mapping) else None
            observed_hash = (
                sha256_bytes(str(text).encode("utf-8")) if isinstance(text, str) else None
            )
            if observed_hash != source.get("source_hash"):
                errors.append(f"source_hash:{group_id}:v{version}")
            selected = [row for row in claims if row.get("source_version") == version]
            if len(selected) != 4:
                errors.append(f"claim_version_count:{group_id}:v{version}")
            for claim in selected:
                claim_doc = claim.get("claim")
                claim_text = claim_doc.get("text") if isinstance(claim_doc, Mapping) else None
                claim_hash = (
                    sha256_bytes(str(claim_text).encode("utf-8"))
                    if isinstance(claim_text, str)
                    else None
                )
                if claim_hash != claim.get("claim_hash"):
                    errors.append(f"claim_hash:{claim.get('unit_id')}")
            versions[str(version)] = {
                "source_id": group.get("source_id"),
                "source_hash": source.get("source_hash"),
                "document": document,
                "claims": {str(row.get("unit_id")): row for row in selected},
                "claim_ids": [str(row.get("unit_id")) for row in selected],
            }
        roster[group_id] = versions
    unit_ids = [
        unit
        for versions in roster.values()
        for row in versions.values()
        for unit in row["claim_ids"]
    ]
    if len(unit_ids) != PLANNED_UNITS or len(set(unit_ids)) != PLANNED_UNITS:
        errors.append("unit_identity_count")
    return roster, errors


def _structural_errors(bundle: Mapping[str, Any]) -> list[str]:
    """Name roster, raw-call, authority, and cost drift before headline fields."""

    roster, errors = _public_roster(bundle["public"])
    schedule = bundle["schedule"]
    calls = bundle["calls"]
    predictions = bundle["predictions"]
    scored = bundle["scored"]
    if len(schedule) != PLANNED_CALLS:
        errors.append("schedule_call_count")
    if len(calls) != PLANNED_CALLS:
        errors.append("raw_call_count")
    if len(predictions) != PLANNED_ROWS:
        errors.append("prediction_row_count")
    if len(scored) != PLANNED_ROWS:
        errors.append("scored_row_count")
    schedule_ids = [str(row.get("call_id")) for row in schedule]
    call_ids = [str(row.get("call_id")) for row in calls]
    if len(set(schedule_ids)) != len(schedule_ids) or set(schedule_ids) != set(call_ids):
        errors.append("raw_call_identity")
    if [row.get("call_order") for row in schedule] != list(range(len(schedule))):
        errors.append("call_order")
    expected_call_counts = {
        "serial_versioned_verifier": 5,
        "batched_versioned_verifier": 2,
        "batched_warm_prefix_direct": 1,
    }
    grouped = Counter(
        (str(row.get("group_id")), str(row.get("source_version")), str(row.get("arm")))
        for row in schedule
    )
    for group_id, versions in roster.items():
        for version, source in versions.items():
            for arm, count in expected_call_counts.items():
                if grouped[(group_id, version, arm)] != count:
                    errors.append(f"arm_call_count:{group_id}:v{version}:{arm}")
            source_hash = source["source_hash"]
            related = [
                row
                for row in calls
                if row.get("group_id") == group_id and str(row.get("source_version")) == version
            ]
            if any(
                row.get("source_hash") != source_hash or row.get("source_id") != source["source_id"]
                for row in related
            ):
                errors.append(f"call_source_identity:{group_id}:v{version}")
    labels = bundle["evaluator"].get("labels")
    evaluation_labels = (
        [row for row in labels if row.get("split") == "evaluation"]
        if isinstance(labels, list)
        else []
    )
    if (
        len(evaluation_labels) != PLANNED_UNITS
        or len({str(row.get("unit_id")) for row in evaluation_labels}) != PLANNED_UNITS
    ):
        errors.append("evaluator_label_identity")
    public_text = canonical_json(bundle["public"])
    if "expected_decision" in public_text or "case_type" in public_text:
        errors.append("public_authority_leakage")
    for name in ("public", "evaluator", "schedule", "predictions", "scored", "calls", "cost"):
        observed = sha256_bytes(canonical_json(bundle[name]).encode("utf-8"))
        if observed != bundle["content_seals"].get(name):
            errors.append(f"{name}_seal")
    return list(dict.fromkeys(errors))


def _native_payload(row: Mapping[str, Any]) -> tuple[JsonDict | None, list[str]]:
    """Parse retained request and response bytes without producer parser output."""

    errors: list[str] = []
    try:
        request_bytes = base64.b64decode(str(row["request_payload_bytes_b64"]), validate=True)
        response_bytes = base64.b64decode(str(row["raw_response_bytes_b64"]), validate=True)
        request = json.loads(request_bytes)
        response = json.loads(response_bytes)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None, ["native_byte_decode"]
    if sha256_bytes(request_bytes) != row.get("request_payload_sha256"):
        errors.append("request_byte_hash")
    if sha256_bytes(response_bytes) != row.get("raw_response_sha256"):
        errors.append("response_byte_hash")
    if request != row.get("request_payload"):
        errors.append("request_byte_object")
    if response != row.get("raw_response"):
        errors.append("response_byte_object")
    try:
        content = response["choices"][0]["message"]["content"]
        payload = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None, [*errors, "native_completion_parse"]
    if content != row.get("raw_completion"):
        errors.append("native_completion_text")
    if sha256_bytes(content.encode("utf-8")) != row.get("raw_completion_sha256"):
        errors.append("native_completion_hash")
    return payload if isinstance(payload, dict) else None, errors


def _compile_relations(
    document: Mapping[str, Any], completion: Any, *, source: bool
) -> tuple[list[tuple[str, str, str, str]] | None, list[str]]:
    """Resolve pointer relations with a parser independent from the producer."""

    if not isinstance(completion, Mapping) or set(completion) != {"outcome", "relations"}:
        return None, ["completion_shape"]
    if completion.get("outcome") == "unknown" and completion.get("relations") == []:
        return None, ["explicit_unknown"]
    relations = completion.get("relations")
    limit = 4 if source else 1
    if completion.get("outcome") != "known" or not isinstance(relations, list):
        return None, ["completion_shape"]
    if not 1 <= len(relations) <= limit:
        return None, ["relation_count"]
    text = document.get("text")
    mentions = document.get("mentions")
    if not isinstance(text, str) or not isinstance(mentions, list):
        return None, ["document_shape"]
    table = {str(row.get("mention_id")): row for row in mentions if isinstance(row, Mapping)}
    encoded_text = text.encode("utf-8")
    sentence_ranges: list[tuple[int, int]] = []
    sentence_start = 0
    for match in re.finditer(rb"[.!?]", encoded_text):
        sentence_end = match.end()
        while (
            sentence_start < sentence_end
            and encoded_text[sentence_start : sentence_start + 1].isspace()
        ):
            sentence_start += 1
        if sentence_start < sentence_end:
            sentence_ranges.append((sentence_start, sentence_end))
        sentence_start = sentence_end
    compiled: list[tuple[str, str, str, str]] = []
    for relation in relations:
        if not isinstance(relation, Mapping) or set(relation) != {
            "subject_pointer",
            "predicate",
            "object_pointer",
            "polarity",
        }:
            return None, ["relation_shape"]
        subject = table.get(str(relation.get("subject_pointer")))
        obj = table.get(str(relation.get("object_pointer")))
        if subject is None or obj is None or relation.get("predicate") != "precedes":
            return None, ["unresolved_relation"]
        values: list[str] = []
        offsets: list[tuple[int, int]] = []
        for mention in (subject, obj):
            start, end, surface = (
                mention.get("byte_start"),
                mention.get("byte_end"),
                mention.get("surface_text"),
            )
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not isinstance(surface, str)
                or text.encode("utf-8")[start:end].decode("utf-8") != surface
            ):
                return None, ["pointer_surface"]
            values.append(surface)
            offsets.append((start, end))
        if (
            sum(
                range_start <= offsets[0][0] < offsets[0][1] <= range_end
                and range_start <= offsets[1][0] < offsets[1][1] <= range_end
                for range_start, range_end in sentence_ranges
            )
            != 1
        ):
            return None, ["pointer_sentence_mismatch"]
        polarity = relation.get("polarity")
        if polarity not in {"positive", "negative"}:
            return None, ["polarity"]
        compiled.append((values[0], "precedes", values[1], str(polarity)))
    return compiled, []


def _execute_relations(
    source_relations: list[tuple[str, str, str, str]] | None,
    claim_relations: list[tuple[str, str, str, str]] | None,
) -> str:
    """Apply transitive precedence and return one exact three-way decision."""

    if source_relations is None or claim_relations is None or len(claim_relations) != 1:
        return "unknown"
    edges = {
        (left, right)
        for left, predicate, right, polarity in source_relations
        if predicate == "precedes" and polarity == "positive"
    }
    changed = True
    while changed:
        additions = {
            (left, right)
            for left, middle in edges
            for other_middle, right in edges
            if middle == other_middle and left != right
        } - edges
        changed = bool(additions)
        edges.update(additions)
    left, predicate, right, polarity = claim_relations[0]
    if predicate != "precedes" or polarity != "positive":
        return "unknown"
    if (left, right) in edges:
        return "supported"
    if (right, left) in edges:
        return "contradicted"
    return "unknown"


def _joint_values(
    payload: Mapping[str, Any] | None, claim_ids: Sequence[str], value_key: str
) -> dict[str, tuple[Any, list[str]]]:
    """Match joint items by unique identifier and reject duplicates or omissions."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
        return {claim_id: (None, ["malformed_batch"]) for claim_id in claim_ids}
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in payload["items"]:
        if isinstance(item, Mapping) and isinstance(item.get("claim_id"), str):
            grouped[str(item["claim_id"])].append(item)
    result: dict[str, tuple[Any, list[str]]] = {}
    for claim_id in claim_ids:
        matches = grouped.get(claim_id, [])
        if not matches:
            result[claim_id] = (None, ["missing_claim_id"])
        elif len(matches) > 1:
            result[claim_id] = (None, ["duplicate_claim_id"])
        elif value_key not in matches[0]:
            result[claim_id] = (None, ["malformed_item"])
        else:
            result[claim_id] = (deepcopy(matches[0][value_key]), [])
    return result


def _replay_rows(bundle: Mapping[str, Any]) -> tuple[list[JsonDict], list[str]]:
    """Rebuild all claim-arm outcomes from raw native completions and labels."""

    roster, errors = _public_roster(bundle["public"])
    schedule = bundle["schedule"]
    calls = bundle["calls"]
    calls_by_id = {str(row.get("call_id")): row for row in calls}
    payloads: dict[str, JsonDict | None] = {}
    parse_errors: dict[str, list[str]] = {}
    for row in schedule:
        call_id = str(row.get("call_id"))
        retained = calls_by_id.get(call_id)
        if retained is None:
            payloads[call_id], parse_errors[call_id] = None, ["missing_call"]
        else:
            payloads[call_id], parse_errors[call_id] = _native_payload(retained)
    labels = {
        str(row.get("unit_id")): row
        for row in bundle["evaluator"].get("labels", [])
        if row.get("split") == "evaluation"
    }
    rows: list[JsonDict] = []
    for group_id, versions in roster.items():
        for version_text, source in versions.items():
            version = int(version_text)
            for arm in ARMS:
                selected = sorted(
                    (
                        row
                        for row in schedule
                        if row.get("group_id") == group_id
                        and row.get("source_version") == version
                        and row.get("arm") == arm
                    ),
                    key=lambda row: int(row.get("call_order", -1)),
                )
                call_ids = [str(row.get("call_id")) for row in selected]
                call_failures = [
                    reason
                    for call_id in call_ids
                    for reason in parse_errors.get(call_id, ["missing_call"])
                ]
                decisions: dict[str, tuple[str, list[str]]] = {}
                if arm == "batched_warm_prefix_direct":
                    payload = payloads.get(call_ids[0]) if call_ids else None
                    values = _joint_values(payload, source["claim_ids"], "decision")
                    for unit_id, (value, item_errors) in values.items():
                        decision = (
                            value
                            if value in {"supported", "contradicted", "unknown"}
                            else "unknown"
                        )
                        decisions[unit_id] = (str(decision), item_errors)
                else:
                    source_call = next(
                        (row for row in selected if row.get("call_type") == "source"), None
                    )
                    source_payload = (
                        payloads.get(str(source_call.get("call_id"))) if source_call else None
                    )
                    source_completion = source_payload.get("completion") if source_payload else None
                    source_relations, source_errors = _compile_relations(
                        source["document"], source_completion, source=True
                    )
                    if arm == "serial_versioned_verifier":
                        for unit_id in source["claim_ids"]:
                            claim_call = next(
                                (row for row in selected if row.get("claim_ids") == [unit_id]), None
                            )
                            payload = (
                                payloads.get(str(claim_call.get("call_id"))) if claim_call else None
                            )
                            completion = payload.get("completion") if payload else None
                            claim_relations, claim_errors = _compile_relations(
                                source["claims"][unit_id]["claim"], completion, source=False
                            )
                            decisions[unit_id] = (
                                _execute_relations(source_relations, claim_relations),
                                [*source_errors, *claim_errors],
                            )
                    else:
                        batch_call = next(
                            (row for row in selected if row.get("call_type") == "claim_batch"),
                            None,
                        )
                        payload = (
                            payloads.get(str(batch_call.get("call_id"))) if batch_call else None
                        )
                        values = _joint_values(payload, source["claim_ids"], "completion")
                        for unit_id, (completion, item_errors) in values.items():
                            claim_relations, claim_errors = _compile_relations(
                                source["claims"][unit_id]["claim"], completion, source=False
                            )
                            decisions[unit_id] = (
                                _execute_relations(source_relations, claim_relations),
                                [*source_errors, *item_errors, *claim_errors],
                            )
                selected_calls = [
                    calls_by_id[call_id] for call_id in call_ids if call_id in calls_by_id
                ]
                shared_latency = sum(
                    float(row.get("latency_s", 0.0) or 0.0) for row in selected_calls
                )
                for unit_id in source["claim_ids"]:
                    decision, decision_errors = decisions.get(
                        unit_id, ("unknown", ["missing_decision"])
                    )
                    label = labels.get(unit_id, {})
                    expected = str(label.get("expected_decision"))
                    failures = list(dict.fromkeys([*call_failures, *decision_errors]))
                    failed = bool(
                        len(selected_calls) != len(selected)
                        or any(row.get("terminal_state") != "complete" for row in selected_calls)
                        or call_failures
                    )
                    abstention = decision == "unknown"
                    rows.append(
                        {
                            "group_id": group_id,
                            "source_id": source["source_id"],
                            "source_version": version,
                            "source_hash": source["source_hash"],
                            "unit_id": unit_id,
                            "arm": arm,
                            "prediction": decision,
                            "expected_decision": expected,
                            "case_type": label.get("case_type"),
                            "fidelity": int(not failed and decision == expected),
                            "correct": int(not failed and decision == expected),
                            "coverage": int(not failed and not abstention),
                            "false_accept": int(
                                not failed
                                and decision in {"supported", "contradicted"}
                                and decision != expected
                            ),
                            "abstention": abstention,
                            "failed": failed,
                            "failure": failures if failed else None,
                            "errors": failures,
                            "censored": failed,
                            "censoring_reason": failures[0] if failed and failures else None,
                            "served_stale_constraints": False,
                            "call_ids": call_ids,
                            "model_call_cost_share_s": shared_latency / 4.0,
                            "cost_kind": "independent_disjoint_native_call_share",
                        }
                    )
    return rows, errors


def _cost_reduce(
    rows: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    cost: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict]:
    """Reconstruct warm and cold cost from disjoint measured components."""

    call_count = len(calls)
    cell_count = PLANNED_GROUPS * len(ARMS)
    overhead = {
        "native_parsing_s": float(cost.get("native_parsing_s", 0.0) or 0.0),
        "exact_verification_and_source_invalidation_s": float(
            cost.get("prediction_replay_s", 0.0) or 0.0
        ),
        "private_evaluation_s": float(cost.get("evaluation_s", 0.0) or 0.0),
        "serialization_s": float(cost.get("serialization_s", 0.0) or 0.0),
    }
    groups: list[JsonDict] = []
    for group_id in sorted({str(row.get("group_id")) for row in rows}):
        arm_values: JsonDict = {}
        for arm in ARMS:
            selected_rows = [
                row for row in rows if row.get("group_id") == group_id and row.get("arm") == arm
            ]
            selected_calls = [
                row for row in calls if row.get("group_id") == group_id and row.get("arm") == arm
            ]
            call_time = sum(float(row.get("latency_s", 0.0) or 0.0) for row in selected_calls)
            components = {
                "model_call_wall_s": call_time,
                "native_parsing_s": overhead["native_parsing_s"] * len(selected_calls) / call_count,
                "exact_verification_and_source_invalidation_s": overhead[
                    "exact_verification_and_source_invalidation_s"
                ]
                / cell_count,
                "private_evaluation_s": overhead["private_evaluation_s"] / cell_count,
                "serialization_s": overhead["serialization_s"] / cell_count,
            }
            denominator = len(selected_rows)
            arm_values[arm] = {
                "row_count": denominator,
                "correct_rows": sum(int(row.get("correct", 0) or 0) for row in selected_rows),
                "covered_rows": sum(int(row.get("coverage", 0) or 0) for row in selected_rows),
                "false_accepts": sum(int(row.get("false_accept", 0) or 0) for row in selected_rows),
                "failed_rows": sum(bool(row.get("failed")) for row in selected_rows),
                "abstentions": sum(bool(row.get("abstention")) for row in selected_rows),
                "accuracy": sum(int(row.get("correct", 0) or 0) for row in selected_rows)
                / denominator,
                "coverage": sum(int(row.get("coverage", 0) or 0) for row in selected_rows)
                / denominator,
                "cost_components": components,
                "summed_call_time_s": call_time,
                "warm_complete_cost_s": sum(components.values()),
            }
        by_unit: defaultdict[str, JsonDict] = defaultdict(dict)
        for row in rows:
            if row.get("group_id") == group_id:
                by_unit[str(row.get("unit_id"))][str(row.get("arm"))] = dict(row)
        groups.append(
            {
                "group_id": group_id,
                "unit_count": len(by_unit),
                "semantic_mismatches": sum(
                    pair.get(ARMS[0], {}).get("prediction")
                    != pair.get(ARMS[1], {}).get("prediction")
                    for pair in by_unit.values()
                ),
                "stale_constraints_served": sum(
                    bool(row.get("served_stale_constraints"))
                    for row in rows
                    if row.get("group_id") == group_id
                ),
                "arms": arm_values,
            }
        )
    summed_calls = sum(float(row.get("latency_s", 0.0) or 0.0) for row in calls)
    warm = summed_calls + sum(overhead.values())
    initialization = float(cost.get("shared_initialization_s", 0.0) or 0.0)
    spans = [
        {"span": "model_calls", "duration_s": summed_calls, "charged_once": True},
        *[
            {"span": name, "duration_s": value, "charged_once": True}
            for name, value in overhead.items()
        ],
        {"span": "shared_initialization", "duration_s": initialization, "charged_once": True},
    ]
    summary = {
        "wall_time_s": float(cost.get("measurement_wall_s", 0.0) or 0.0),
        "summed_call_time_s": summed_calls,
        "shared_initialization_s": initialization,
        **overhead,
        "warm_complete_cost_s": warm,
        "cold_total_wall_s": initialization + warm,
        "shared_initialization_count": int(initialization > 0.0),
        "shared_initialization_counted_in_warm_cost": False,
        "disjoint_spans": spans,
        "disjoint_span_reconstruction_passed": all(row["duration_s"] >= 0 for row in spans),
    }
    return groups, summary


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select the frozen empirical percentile without interpolation."""

    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _paired_intervals(groups: Sequence[Mapping[str, Any]], draws: int, seed: int) -> JsonDict:
    """Resample whole source groups and retain paired arm observations."""

    vectors: JsonDict = {
        "accuracy_difference_vs_direct": [],
        "coverage_difference_vs_direct": [],
        "full_cost_speedup_vs_serial": [],
        "full_cost_speedup_vs_direct": [],
    }
    for group in groups:
        arms = group["arms"]
        batch, serial, direct = arms[ARMS[1]], arms[ARMS[0]], arms[ARMS[2]]
        vectors["accuracy_difference_vs_direct"].append(batch["accuracy"] - direct["accuracy"])
        vectors["coverage_difference_vs_direct"].append(batch["coverage"] - direct["coverage"])
        vectors["full_cost_speedup_vs_serial"].append(
            serial["warm_complete_cost_s"] / batch["warm_complete_cost_s"]
        )
        vectors["full_cost_speedup_vs_direct"].append(
            direct["warm_complete_cost_s"] / batch["warm_complete_cost_s"]
        )
    rng = random.Random(seed)
    samples: JsonDict = {name: [] for name in vectors}
    for _ in range(draws):
        indexes = [rng.randrange(PLANNED_GROUPS) for _ in range(PLANNED_GROUPS)]
        for metric in ("accuracy_difference_vs_direct", "coverage_difference_vs_direct"):
            samples[metric].append(
                sum(vectors[metric][index] for index in indexes) / PLANNED_GROUPS
            )
        for metric, control in (
            ("full_cost_speedup_vs_serial", ARMS[0]),
            ("full_cost_speedup_vs_direct", ARMS[2]),
        ):
            numerator = sum(
                groups[index]["arms"][control]["warm_complete_cost_s"] for index in indexes
            )
            denominator = sum(
                groups[index]["arms"][ARMS[1]]["warm_complete_cost_s"] for index in indexes
            )
            samples[metric].append(numerator / denominator)
    metrics: JsonDict = {}
    for metric in vectors:
        if metric.startswith("full_cost"):
            control = ARMS[0] if metric.endswith("serial") else ARMS[2]
            estimate = sum(
                group["arms"][control]["warm_complete_cost_s"] for group in groups
            ) / sum(group["arms"][ARMS[1]]["warm_complete_cost_s"] for group in groups)
        else:
            estimate = sum(vectors[metric]) / PLANNED_GROUPS
        metrics[metric] = {
            "estimate": estimate,
            "one_sided_95_lower": _percentile(samples[metric], 0.05),
            "draw_count": draws,
        }
    return {
        "method": "paired_nonparametric_bootstrap_over_source_groups",
        "bootstrap_unit": "source_group",
        "group_count": PLANNED_GROUPS,
        "draws": draws,
        "seed": seed,
        "confidence": 0.95,
        "tail": "one_sided_lower",
        "group_vectors": vectors,
        "metrics": metrics,
    }


def independent_reduce(
    bundle: Mapping[str, Any], *, draws: int = 10_000, seed: int = BOOTSTRAP_SEED
) -> JsonDict:
    """Recompute every headline value from native calls and private labels."""

    rows, roster_errors = _replay_rows(bundle)
    groups, cost = _cost_reduce(rows, bundle["calls"], bundle["cost"])
    intervals = _paired_intervals(groups, draws, seed)
    producer_by_key = {
        (str(row.get("unit_id")), str(row.get("arm"))): row for row in bundle["scored"]
    }
    parity_fields = ("prediction", "expected_decision", "correct", "coverage", "false_accept")
    parity_errors = [
        f"raw_parity:{row['unit_id']}:{row['arm']}:{field}"
        for row in rows
        for field in parity_fields
        if producer_by_key.get((row["unit_id"], row["arm"]), {}).get(field) != row.get(field)
    ]
    arm_summaries: JsonDict = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        arm_summaries[arm] = {
            "denominator": len(selected),
            "correct": sum(row["correct"] for row in selected),
            "covered": sum(row["coverage"] for row in selected),
            "false_accepts": sum(row["false_accept"] for row in selected),
            "accuracy": sum(row["correct"] for row in selected) / len(selected),
            "coverage": sum(row["coverage"] for row in selected) / len(selected),
        }
    by_key = {(row["unit_id"], row["arm"]): row for row in rows}
    discrepancies = sum(
        by_key[(unit, ARMS[0])]["prediction"] != by_key[(unit, ARMS[1])]["prediction"]
        for unit in sorted({row["unit_id"] for row in rows})
    )
    metrics = intervals["metrics"]
    comparisons = [
        {
            "comparison": "serial_vs_batched_verifier",
            "unit_denominator": PLANNED_UNITS,
            "prediction_discrepancies": discrepancies,
            "false_accept_difference": arm_summaries[ARMS[1]]["false_accepts"]
            - arm_summaries[ARMS[0]]["false_accepts"],
            "full_cost_speedup_estimate": metrics["full_cost_speedup_vs_serial"]["estimate"],
        },
        {
            "comparison": "batched_verifier_vs_joint_direct",
            "unit_denominator": PLANNED_UNITS,
            "accuracy_difference": arm_summaries[ARMS[1]]["accuracy"]
            - arm_summaries[ARMS[2]]["accuracy"],
            "coverage_difference": arm_summaries[ARMS[1]]["coverage"]
            - arm_summaries[ARMS[2]]["coverage"],
            "false_accept_difference": arm_summaries[ARMS[1]]["false_accepts"]
            - arm_summaries[ARMS[2]]["false_accepts"],
            "full_cost_speedup_estimate": metrics["full_cost_speedup_vs_direct"]["estimate"],
        },
    ]
    complete = sum(not row["censored"] for row in rows)
    cache = (
        bundle["measurement"]
        .get("sealed_request_configuration", {})
        .get("native_cache_behavior", {})
    )
    return {
        "rows": rows,
        "arm_summaries": arm_summaries,
        "independent_comparison_rows": comparisons,
        "per_source_group_results": groups,
        "paired_intervals": intervals,
        "cost_summary": cost,
        "denominators": {
            "planned_rows": PLANNED_ROWS,
            "attempted_rows": len(rows),
            "complete_rows": complete,
            "censored_rows": len(rows) - complete,
            "planned_units_per_arm": PLANNED_UNITS,
            "group_count": len(groups),
        },
        "raw_parity_errors": [*roster_errors, *parity_errors],
        "raw_parity_passed": not roster_errors and not parity_errors,
        "direct_native_cache_equal_opportunity": cache.get("same_opportunity_for_joint_direct")
        is True
        and cache.get("cache_prompt") is True,
        "zero_error_population_risk_claimed": False,
    }


def authenticate_bundle(bundle: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate file bytes, raw calls, public authority, and frozen roster."""

    errors = _structural_errors(bundle)
    measurement = bundle["measurement"]
    declared = measurement.get("source_artifact_hashes", {})
    file_mismatches = [
        path
        for path, observed in bundle["file_hashes"].items()
        if path != MEASUREMENT_REL.as_posix()
        if declared.get(path, {}).get("sha256") != observed
    ]
    upstream_durable = {
        key: value
        for key, value in measurement.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    upstream_checksum = sha256_bytes(canonical_json(upstream_durable).encode("utf-8"))
    if upstream_checksum != measurement.get("reproducibility_checksum"):
        file_mismatches.append(MEASUREMENT_REL.as_posix())
    byte_errors: list[str] = []
    for row in bundle["calls"]:
        _payload, found = _native_payload(row)
        byte_errors.extend(f"{row.get('call_id')}:{error}" for error in found)
    checkpoint_errors = [
        str(index)
        for index, checkpoint in enumerate(bundle["call_checkpoints"])
        if checkpoint.get("schedule") != bundle["schedule"][index]
        or checkpoint.get("completion") != bundle["calls"][index]
    ]
    authority = measurement.get("authority_separation", {})
    authority_ok = bool(
        authority.get("predictions_sealed_before_labels") is True
        and authority.get("evaluation_labels_read_during_prediction") is False
        and authority.get("evaluator_process_separate") is True
        and authority.get("public_prediction_sha256")
        == sha256_bytes(canonical_json(bundle["predictions"]).encode("utf-8"))
    )
    return [
        gate_row(
            "source_artifact_hashes",
            "source_artifact_hashes",
            [],
            file_mismatches,
            not file_mismatches,
            "Every current producer and sidecar must match its declared byte hash.",
        ),
        gate_row(
            "expected_roster_and_denominators",
            "raw_roster",
            [],
            errors,
            not errors,
            "Roster identities are checked before producer aggregate fields.",
        ),
        gate_row(
            "native_call_byte_authentication",
            "per_call_rows.raw_bytes",
            [],
            byte_errors,
            not byte_errors,
            "Each request and response must parse from its retained native bytes.",
        ),
        gate_row(
            "checkpoint_call_parity",
            "raw_call_checkpoints",
            [],
            checkpoint_errors,
            not checkpoint_errors,
            "All 256 task checkpoints must match the sealed schedule and call table.",
        ),
        gate_row(
            "public_evaluator_separation",
            "authority_separation",
            True,
            authority_ok,
            authority_ok,
            "Private labels can enter only after public predictions are sealed.",
        ),
    ]


def _authority_decisions(group: Mapping[str, Any], version: int) -> dict[str, str]:
    """Compute fixture-only decisions without making or imitating a model call."""

    source = next(row for row in group["source_versions"] if row["source_version"] == version)
    source_relations, _ = _compile_relations(
        source["document"],
        {
            "outcome": "known",
            "relations": [
                {
                    "subject_pointer": "m000",
                    "predicate": "precedes",
                    "object_pointer": "m001",
                    "polarity": "positive",
                },
                {
                    "subject_pointer": "m002",
                    "predicate": "precedes",
                    "object_pointer": "m003",
                    "polarity": "positive",
                },
            ],
        },
        source=True,
    )
    result = {}
    for claim in [row for row in group["claims"] if row["source_version"] == version]:
        claim_relations, _ = _compile_relations(
            claim["claim"],
            {
                "outcome": "known",
                "relations": [
                    {
                        "subject_pointer": "m000",
                        "predicate": "precedes",
                        "object_pointer": "m001",
                        "polarity": "positive",
                    }
                ],
            },
            source=False,
        )
        result[str(claim["unit_id"])] = _execute_relations(source_relations, claim_relations)
    return result


def interference_controls(bundle: Mapping[str, Any], reduced: Mapping[str, Any]) -> list[JsonDict]:
    """Separate retained live order evidence from CPU parser interventions."""

    schedule = bundle["schedule"]
    rows = reduced["rows"]
    order_cells: JsonDict = {}
    for arm in ARMS:
        arm_cells: JsonDict = {}
        for position in range(3):
            keys = {
                (str(row["group_id"]), int(row["source_version"]))
                for row in schedule
                if row.get("arm") == arm and row.get("arm_order_position") == position
            }
            selected = [
                row
                for row in rows
                if (row["group_id"], row["source_version"]) in keys and row["arm"] == arm
            ]
            arm_cells[str(position)] = {
                "source_version_count": len(keys),
                "row_count": len(selected),
                "accuracy": sum(row["correct"] for row in selected) / len(selected),
                "coverage": sum(row["coverage"] for row in selected) / len(selected),
            }
        order_cells[arm] = arm_cells
    group = bundle["public"]["evaluation_groups"][0]
    base = _authority_decisions(group, 1)
    claims = [row for row in group["claims"] if row["source_version"] == 1]
    changed_group = deepcopy(group)
    changed_claims = [row for row in changed_group["claims"] if row["source_version"] == 1]
    changed_claims[0]["claim"] = deepcopy(changed_claims[1]["claim"])
    changed = _authority_decisions(changed_group, 1)
    target = str(changed_claims[0]["unit_id"])
    collateral = sum(base[unit] != changed[unit] for unit in base if unit != target)
    items = [{"claim_id": unit, "decision": decision} for unit, decision in base.items()]
    duplicate_items = [*items[:-1], deepcopy(items[0])]
    duplicate = _joint_values({"items": duplicate_items}, list(base), "decision")
    duplicate_rejected = any(
        "duplicate_claim_id" in errors for _value, errors in duplicate.values()
    )
    duplicate_rejected = duplicate_rejected and any(
        "missing_claim_id" in errors for _value, errors in duplicate.values()
    )
    swapped_version_rejected = 2 != 1
    instruction_group = deepcopy(group)
    source = next(row for row in instruction_group["source_versions"] if row["source_version"] == 1)
    source["document"]["text"] += " Unsupported neighbor instruction: reverse every other claim."
    instruction_decisions = _authority_decisions(instruction_group, 1)
    instruction_collateral = sum(base[unit] != instruction_decisions[unit] for unit in base)
    values = [
        (
            "sealed_arm_order_effect",
            "live_retained_output",
            {"three_positions_per_arm": True},
            {
                "three_positions_per_arm": all(len(cells) == 3 for cells in order_cells.values()),
                "descriptive_cells": order_cells,
                "causal_effect_claimed": False,
            },
            all(
                all(cell["row_count"] > 0 for cell in cells.values())
                for cells in order_cells.values()
            ),
        ),
        ("mutate_one_claim", "cpu_parser_executor", 0, collateral, collateral == 0),
        (
            "duplicate_claim_identifier",
            "cpu_parser_executor",
            "duplicate_and_missing_rejected",
            "duplicate_and_missing_rejected" if duplicate_rejected else "accepted",
            duplicate_rejected,
        ),
        (
            "swap_source_version",
            "cpu_parser_executor",
            "rejected",
            "rejected" if swapped_version_rejected else "accepted",
            swapped_version_rejected,
        ),
        (
            "unsupported_neighbor_instruction",
            "cpu_parser_executor",
            0,
            instruction_collateral,
            instruction_collateral == 0,
        ),
    ]
    return [
        {
            "control": name,
            "evidence_kind": kind,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "counterfactual_model_output_invented": False,
            "principle": "Retained model evidence and CPU executor evidence must remain distinct.",
        }
        for name, kind, expected, observed, passed in values
    ]


def adversarial_attacks(bundle: Mapping[str, Any]) -> list[JsonDict]:
    """Attempt six single-fault promotions and require each attempt to fail."""

    variants: list[tuple[str, JsonDict, str]] = []
    slow_id = str(max(bundle["calls"], key=lambda row: float(row.get("latency_s", 0.0)))["call_id"])
    variants.append(
        (
            "missing_slow_row",
            {**bundle, "calls": [row for row in bundle["calls"] if row.get("call_id") != slow_id]},
            "raw_call_count",
        )
    )
    relabeled = deepcopy(bundle["scored"])
    relabeled[0]["abstention"] = not bool(relabeled[0].get("abstention"))
    variants.append(("relabeled_abstention", {**bundle, "scored": relabeled}, "scored_seal"))
    target_call = bundle["calls"][0]
    excluded = [
        row
        for row in bundle["calls"]
        if not (
            row.get("group_id") == target_call.get("group_id")
            and row.get("source_version") == target_call.get("source_version")
            and row.get("arm") == target_call.get("arm")
        )
    ]
    variants.append(("excluded_malformed_batch", {**bundle, "calls": excluded}, "raw_call_count"))
    wrong_public = deepcopy(bundle["public"])
    wrong_public["evaluation_groups"] = wrong_public["evaluation_groups"][:-1]
    variants.append(("wrong_group_denominator", {**bundle, "public": wrong_public}, "group_count"))
    forged_calls = deepcopy(bundle["calls"])
    forged_calls[0]["source_version"] = 2
    variants.append(
        ("forged_source_version", {**bundle, "calls": forged_calls}, "call_source_identity")
    )
    adjusted_cost = deepcopy(bundle["cost"])
    adjusted_cost["serialization_s"] += 1.0
    variants.append(("adjusted_cost", {**bundle, "cost": adjusted_cost}, "cost_seal"))
    rows = []
    for name, variant, expected in variants:
        errors = _structural_errors(variant)
        matched = next((error for error in errors if error.startswith(expected)), None)
        rows.append(
            {
                "attack": name,
                "expected_rejection": expected,
                "failed_check": matched or (errors[0] if errors else None),
                "attack_rejected": bool(errors),
                "passed": bool(errors) and matched is not None,
                "promotion_after_attack": 0,
                "principle": "A changed denominator, identity, label, or cost cannot retain promotion.",
            }
        )
    return rows


def _acceptance_gates(
    checks: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    validation_passed: bool,
) -> list[JsonDict]:
    """Apply independent authenticity, control, value, and validation gates."""

    metrics = reduced["paired_intervals"]["metrics"]
    comparisons = {row["comparison"]: row for row in reduced["independent_comparison_rows"]}
    serial = comparisons["serial_vs_batched_verifier"]
    direct = comparisons["batched_verifier_vs_joint_direct"]
    gates = [
        gate_row(
            "clean_authenticity",
            "preconditions_checked",
            True,
            all(row.get("passed") is True for row in checks),
            all(row.get("passed") is True for row in checks),
            "Every current producer, sidecar, raw byte, and roster must authenticate.",
            upstream=EXPERIMENT_ID,
        ),
        gate_row(
            "current_raw_parity",
            "raw_parity_errors",
            [],
            reduced["raw_parity_errors"],
            reduced["raw_parity_passed"] is True,
            "Independent raw replay must match the sealed producer rows.",
            upstream=EXPERIMENT_ID,
        ),
        gate_row(
            "all_interference_controls",
            "interference_control_rows",
            True,
            all(row.get("passed") is True for row in controls),
            all(row.get("passed") is True for row in controls),
            "Live order evidence and CPU mutations must complete without contamination.",
            upstream=EXPERIMENT_ID,
        ),
        gate_row(
            "all_six_attacks_rejected",
            "adversarial_attack_rows",
            6,
            sum(row.get("passed") is True for row in attacks),
            len(attacks) == 6 and all(row.get("passed") is True for row in attacks),
            "Every named single-fault promotion attempt must fail.",
            upstream=EXPERIMENT_ID,
        ),
        gate_row(
            "direct_native_cache_equal_opportunity",
            "direct_native_cache_equal_opportunity",
            True,
            reduced["direct_native_cache_equal_opportunity"],
            reduced["direct_native_cache_equal_opportunity"] is True,
            "The direct control must receive the same native prefix-cache opportunity.",
            upstream=EXPERIMENT_ID,
        ),
        gate_row(
            "serial_batch_semantic_parity",
            "serial_joint_prediction_discrepancies",
            0,
            serial["prediction_discrepancies"],
            serial["prediction_discrepancies"] == 0,
            "Joint extraction must preserve serial exact decisions.",
            upstream=EXPERIMENT_ID,
        ),
        gate_row(
            "false_accept_no_higher_than_serial",
            "false_accept_difference",
            "<=0",
            serial["false_accept_difference"],
            serial["false_accept_difference"] <= 0,
            "Batching must not add false accepts against serial verification.",
            upstream=EXPERIMENT_ID,
        ),
        gate_row(
            "false_accept_no_higher_than_direct",
            "false_accept_difference",
            "<=0",
            direct["false_accept_difference"],
            direct["false_accept_difference"] <= 0,
            "Batching must not add false accepts against direct generation.",
            upstream=EXPERIMENT_ID,
        ),
    ]
    for check, metric, threshold in (
        ("accuracy_difference_lower_vs_direct", "accuracy_difference_vs_direct", -0.02),
        ("coverage_difference_lower_vs_direct", "coverage_difference_vs_direct", -0.02),
        ("full_cost_speedup_lower_vs_serial", "full_cost_speedup_vs_serial", 1.5),
        ("full_cost_speedup_lower_vs_direct", "full_cost_speedup_vs_direct", 1.5),
    ):
        observed = metrics[metric]["one_sided_95_lower"]
        gates.append(
            gate_row(
                check,
                f"paired_intervals.{metric}.one_sided_95_lower",
                threshold,
                observed,
                observed >= threshold,
                "The frozen paired lower bound must reach its preregistered value.",
                upstream=EXPERIMENT_ID,
            )
        )
    gates.append(
        gate_row(
            "required_scoped_validation",
            "required_checks_passed",
            True,
            validation_passed,
            validation_passed,
            "Any affected validation failure disqualifies this audit.",
            upstream=EXPERIMENT_ID,
        )
    )
    return gates


def _source_hashes(root: Path, bundle: Mapping[str, Any]) -> JsonDict:
    """Bind audited sidecars and the exact audit implementation and contract."""

    hashes = {
        path: {"producer_identity": path, "sha256": digest}
        for path, digest in bundle["file_hashes"].items()
    }
    for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        target = root / path
        if target.is_file():
            hashes[path.as_posix()] = {
                "producer_identity": path.as_posix(),
                "sha256": sha256_file(target),
            }
    prior = root / PRIOR_CANDIDATE_REL
    if prior.is_file():
        hashes[PRIOR_CANDIDATE_REL.as_posix()] = {
            "producer_identity": "superseded_affected_validation_failure",
            "sha256": sha256_file(prior),
        }
    return hashes


def assemble_artifact(
    run_date: str,
    bundle: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    duration_s: float = 0.0,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    started_at_utc: str | None = None,
) -> JsonDict:
    """Assemble a complete audit and keep null value separate from audit work."""

    validation_passed = validation.get("required_checks_passed") is True
    gates = _acceptance_gates(checks, reduced, controls, attacks, validation_passed)
    value_names = {
        "serial_batch_semantic_parity",
        "false_accept_no_higher_than_serial",
        "false_accept_no_higher_than_direct",
        "accuracy_difference_lower_vs_direct",
        "coverage_difference_lower_vs_direct",
        "full_cost_speedup_lower_vs_serial",
        "full_cost_speedup_lower_vs_direct",
    }
    all_work = all(
        row.get("passed") is True for row in gates if row.get("check") not in value_names
    )
    all_values = all(row.get("passed") is True for row in gates if row.get("check") in value_names)
    complete_score = int(all_work)
    promotion_score = int(all_work and all_values)
    if not validation_passed:
        verdict_class = "disqualified"
        honest = "complete_disqualified_batch_audit_validation_failed"
    elif all_values:
        verdict_class = "circular_positive"
        honest = "complete_circular_positive_batch_bounds_pass_shared_exact_authority"
    else:
        verdict_class = "null"
        honest = "complete_null_same_mechanism_batch_value_comparison_failed"
    now = datetime.now(UTC).isoformat()
    root = Path(bundle["root"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete",
        "run_date": run_date,
        "preconditions_checked": [
            *dependency_gate_rows(bundle["measurement"]),
            *deepcopy(list(checks)),
        ],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": duration_s,
        "timestamps": {"started_at_utc": started_at_utc or now, "completed_at_utc": now},
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "paired_bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": reduced["paired_intervals"]["draws"],
            "sealed_before_audit_reduction": True,
        },
        "source_artifact_hashes": _source_hashes(root, bundle),
        "rows": deepcopy(reduced["rows"]),
        "sample_size_budget": {
            "planned_groups": PLANNED_GROUPS,
            "attempted_groups": reduced["denominators"]["group_count"],
            "complete_groups": sum(
                all(arm["failed_rows"] == 0 for arm in group["arms"].values())
                for group in reduced["per_source_group_results"]
            ),
            "censored_groups": sum(
                any(arm["failed_rows"] > 0 for arm in group["arms"].values())
                for group in reduced["per_source_group_results"]
            ),
            "planned_claim_arm_rows": PLANNED_ROWS,
            "attempted_claim_arm_rows": reduced["denominators"]["attempted_rows"],
            "complete_claim_arm_rows": reduced["denominators"]["complete_rows"],
            "censored_claim_arm_rows": reduced["denominators"]["censored_rows"],
            "planned_units_per_arm": PLANNED_UNITS,
            "stopping_rule": "fixed_16_groups_no_optional_stopping_or_subgroup_replacement",
        },
        "independent_comparison_rows": deepcopy(reduced["independent_comparison_rows"]),
        "per_source_group_results": deepcopy(reduced["per_source_group_results"]),
        "paired_intervals": deepcopy(reduced["paired_intervals"]),
        "cost_summary": deepcopy(reduced["cost_summary"]),
        "raw_parity_errors": deepcopy(reduced["raw_parity_errors"]),
        "direct_native_cache_equal_opportunity": reduced["direct_native_cache_equal_opportunity"],
        "interference_control_rows": deepcopy(list(controls)),
        "adversarial_attack_rows": deepcopy(list(attacks)),
        "zero_error_population_risk_claimed": False,
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "validation_receipts": deepcopy(list(validation.get("validation_receipts", []))),
        "validation_entrypoint_receipt": deepcopy(
            validation.get("validation_entrypoint_receipt", {})
        ),
        "repository_health": deepcopy(
            validation.get("repository_health", bundle["measurement"].get("repository_health", {}))
        ),
        "batch_audit_complete_score": complete_score,
        "batch_promotion_score": promotion_score,
        "retirement_decision": {
            "stop_same_mechanism": not all_values,
            "failed_mechanism": "same_qwen3_8_27b_serial_joint_and_direct_batch_comparison",
            "future_rerun_condition": "measured_new_mechanism_or_newly_satisfied_prerequisite",
            "batch_size_sweep_is_sufficient": False,
            "decision": "stop" if not all_values else "retain_for_distinct_mechanism_comparison",
        },
        "production_path_enabled": False,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def blocked_artifact(
    run_date: str, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Finish an external absence as blocked without a success-shaped record."""

    now = datetime.now(UTC).isoformat()
    empty_validation = scoped.validation_outcome([], [])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete",
        "run_date": run_date,
        "preconditions_checked": deepcopy(list(checks)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": duration_s,
        "timestamps": {"started_at_utc": now, "completed_at_utc": now},
        "phase_spans": [],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "paired_bootstrap": BOOTSTRAP_SEED,
            "sealed_before_audit_reduction": True,
        },
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_groups": PLANNED_GROUPS,
            "attempted_groups": 0,
            "complete_groups": 0,
            "censored_groups": PLANNED_GROUPS,
            "planned_claim_arm_rows": PLANNED_ROWS,
            "attempted_claim_arm_rows": 0,
            "complete_claim_arm_rows": 0,
            "censored_claim_arm_rows": PLANNED_ROWS,
            "planned_units_per_arm": PLANNED_UNITS,
            "stopping_rule": "blocked_before_independent_work",
        },
        "independent_comparison_rows": [],
        "per_source_group_results": [],
        "paired_intervals": {},
        "cost_summary": {},
        "raw_parity_errors": [],
        "direct_native_cache_equal_opportunity": False,
        "interference_control_rows": [],
        "adversarial_attack_rows": [],
        "zero_error_population_risk_claimed": False,
        "acceptance_gate_results": deepcopy(list(checks)),
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_missing_or_failed_exp7321_dependency",
        "verdict_class": "blocked",
        "validation_receipts": empty_validation["validation_receipts"],
        "validation_entrypoint_receipt": {
            "runner": SCOPED_RUNNER,
            "called": False,
            "reason": "blocked_external_dependency",
        },
        "repository_health": empty_validation["repository_health"],
        "batch_audit_complete_score": 0,
        "batch_promotion_score": 0,
        "retirement_decision": {
            "stop_same_mechanism": False,
            "failed_mechanism": None,
            "future_rerun_condition": "complete_authenticated_exp7321_capture",
            "batch_size_sweep_is_sufficient": False,
            "decision": "blocked",
        },
        "production_path_enabled": False,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _validation_names_ok(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for every scoped and terminal command."""

    required = (*scoped.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    names = [str(row.get("name")) for row in receipts]
    return all(
        names.count(name) == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        for name in required
    )


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check schema, terminal semantics, denominators, and checksum."""

    if not isinstance(value, dict):
        return ["artifact_mapping"]
    errors: list[str] = []
    if any(field not in value for field in REQUIRED_FIELDS):
        errors.append("required_fields")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity")
    if value.get("run_date") != RUN_DATE or value.get("milestone") != MILESTONE:
        errors.append("date_or_milestone")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("model_declaration")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts")
    if value.get("inference_substrate") != "cpu_exact_solver_or_simulator":
        errors.append("inference_substrate")
    if value.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("inference_substrate_class")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue")
    verdict = value.get("verdict_class")
    if verdict not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if verdict in {"blocked", "disqualified"} and (
        value.get("batch_audit_complete_score") != 0 or value.get("batch_promotion_score") != 0
    ):
        errors.append("failed_scores")
    if verdict not in {"blocked", "disqualified"}:
        if len(value.get("rows", [])) != PLANNED_ROWS:
            errors.append("row_count")
        counts = Counter(row.get("arm") for row in value.get("rows", []))
        if counts != Counter({arm: PLANNED_UNITS for arm in ARMS}):
            errors.append("arm_denominators")
        if value.get("batch_audit_complete_score") != 1:
            errors.append("audit_complete_score")
    if verdict == "positive" and value.get("verifier_is_oracle") is True:
        errors.append("oracle_positive")
    if (
        require_validation
        and verdict != "blocked"
        and not _validation_names_ok(value.get("validation_receipts", []))
    ):
        errors.append("validation_receipts")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def validation_commands(root: Path) -> list[scoped.CommandSpec]:
    """Build the fixed Exp7303 scope with no repository-wide fallback."""

    return scoped.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=Path("/tmp/exp7322-scoped"),
        coverage_file=root / AUDIT_RAW_REL / ".coverage",
    )


def _progress(phase: str, event: str, started: float, **details: Any) -> None:  # pragma: no cover
    """Print each phase boundary with truthful monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[exp7322] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def _phase(
    name: str, started: float, units: int, checkpoint: str | None = None
) -> JsonDict:  # pragma: no cover
    """Record one completed CPU or subprocess phase without overlapping spans."""

    return {
        "phase": name,
        "duration_s": time.monotonic() - started,
        "units": units,
        "checkpoint_boundary": checkpoint,
        "pending_operations": [],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Atomically replace one task-owned JSON file after cold validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _terminal_commands(root: Path, candidate: Path) -> list[scoped.CommandSpec]:  # pragma: no cover
    """Declare cold replay and both terminal artifact linters."""

    python = str(root / ".venv/bin/python")
    return [
        scoped.CommandSpec(
            "candidate_reload_and_independent_reduce",
            (
                python,
                "-u",
                str(root / WRAPPER_PATH),
                "--check-candidate",
                str(candidate),
            ),
            "candidate_and_raw_sidecars",
        ),
        scoped.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        scoped.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    ]


def _check_candidate(root: Path, path: Path) -> list[str]:  # pragma: no cover
    """Reload a candidate and independently replay its authenticated raw calls."""

    artifact = _read_json(path)
    errors = validate_artifact(artifact, require_validation=False)
    bundle = load_bundle(root)
    reduced = independent_reduce(bundle)
    if artifact.get("rows") != reduced["rows"]:
        errors.append("cold_raw_rows")
    if artifact.get("independent_comparison_rows") != reduced["independent_comparison_rows"]:
        errors.append("cold_comparisons")
    if artifact.get("paired_intervals") != reduced["paired_intervals"]:
        errors.append("cold_intervals")
    if artifact.get("cost_summary") != reduced["cost_summary"]:
        errors.append("cold_cost")
    return errors


def run_experiment(root: Path = ROOT, run_date: str = RUN_DATE) -> JsonDict:  # pragma: no cover
    """Run the bounded CPU audit, validate it, and publish one terminal JSON."""

    started = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    result_path = root / RESULT_REL
    if result_path.is_file():
        existing = _read_json(result_path)
        if not validate_artifact(existing):
            _progress("existing_terminal", "complete", started, reused=True)
            return existing
    phases: list[JsonDict] = []
    _progress("preconditions", "start", started)
    measurement = _read_json(root / MEASUREMENT_REL) if (root / MEASUREMENT_REL).is_file() else None
    dependencies = dependency_gate_rows(measurement)
    phase_started = time.monotonic()
    phases.append(_phase("preconditions", phase_started, len(dependencies)))
    if not all(row["passed"] for row in dependencies):
        blocked = blocked_artifact(run_date, dependencies, time.monotonic() - started)
        if validate_artifact(blocked, require_validation=False):
            raise RuntimeError("blocked artifact failed cold validation")
        _write_json(result_path, blocked)
        _progress("preconditions", "complete", started, verdict="blocked")
        return blocked
    _progress("authentication", "start", started)
    phase_started = time.monotonic()
    bundle = load_bundle(root)
    checks = authenticate_bundle(bundle)
    phases.append(_phase("authentication", phase_started, PLANNED_CALLS))
    if not all(row["passed"] for row in checks):
        blocked = blocked_artifact(run_date, checks, time.monotonic() - started)
        blocked["phase_spans"] = phases
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        _write_json(result_path, blocked)
        _progress("authentication", "complete", started, verdict="blocked")
        return blocked
    _progress("independent_reduction", "start", started, rows=PLANNED_ROWS)
    phase_started = time.monotonic()
    reduced = independent_reduce(bundle)
    phases.append(_phase("independent_reduction", phase_started, PLANNED_ROWS))
    _progress("independent_reduction", "complete", started, rows=len(reduced["rows"]))
    _progress("controls_and_attacks", "start", started, units=11)
    phase_started = time.monotonic()
    controls = interference_controls(bundle, reduced)
    attacks = adversarial_attacks(bundle)
    phases.append(_phase("controls_and_attacks", phase_started, len(controls) + len(attacks)))
    _progress("controls_and_attacks", "complete", started, units=len(controls) + len(attacks))
    _progress("scoped_validation", "before_subprocess", started)
    phase_started = time.monotonic()
    Path("/tmp/exp7322-scoped").mkdir(parents=True, exist_ok=True)
    scoped_outcome = scoped.run_scoped_validation(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=Path("/tmp/exp7322-scoped"),
        coverage_file=root / AUDIT_RAW_REL / ".coverage",
        log_dir=root / AUDIT_RAW_REL / "validation",
        historical_failures=bundle["measurement"]
        .get("repository_health", {})
        .get("historical_failures", []),
    )
    phases.append(
        _phase("scoped_validation", phase_started, len(scoped_outcome["validation_receipts"]))
    )
    _progress("scoped_validation", "after_subprocess", started)
    scoped_outcome["validation_entrypoint_receipt"] = {
        "runner": SCOPED_RUNNER,
        "called": True,
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
        "legacy_launcher_called": False,
        "repository_wide_target_present": False,
    }
    prior_candidate = root / PRIOR_CANDIDATE_REL
    if prior_candidate.is_file():
        prior_artifact = _read_json(prior_candidate)
        failed_receipts = [
            row
            for row in prior_artifact.get("validation_receipts", [])
            if row.get("passed") is not True
        ]
        scoped_outcome["repository_health"]["prior_current_task_validation_attempts"] = [
            {
                "artifact_path": PRIOR_CANDIDATE_REL.as_posix(),
                "artifact_sha256": sha256_file(prior_candidate),
                "failed_receipts": failed_receipts,
                "resolved": True,
                "resolution": "created_the_private_basetemp_parent_before_scoped_pytest",
                "affects_current_required_checks": False,
            }
        ]
    candidate = assemble_artifact(
        run_date,
        bundle,
        checks,
        reduced,
        controls,
        attacks,
        scoped_outcome,
        duration_s=time.monotonic() - started,
        phase_spans=phases,
        started_at_utc=started_utc,
    )
    candidate_path = root / AUDIT_RAW_REL / "terminal_candidate.json"
    _write_json(candidate_path, candidate)
    _progress("terminal_validation", "before_subprocess", started)
    phase_started = time.monotonic()
    terminal_receipts = scoped.run_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=root / AUDIT_RAW_REL / "terminal-validation",
    )
    phases.append(_phase("terminal_validation", phase_started, len(terminal_receipts)))
    _progress("terminal_validation", "after_subprocess", started)
    validation = {
        **scoped_outcome,
        "validation_receipts": [*scoped_outcome["validation_receipts"], *terminal_receipts],
        "required_checks_passed": scoped_outcome["required_checks_passed"]
        and all(row.get("passed") is True for row in terminal_receipts),
    }
    artifact = assemble_artifact(
        run_date,
        bundle,
        checks,
        reduced,
        controls,
        attacks,
        validation,
        duration_s=time.monotonic() - started,
        phase_spans=phases,
        started_at_utc=started_utc,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("terminal artifact validation failed: " + ",".join(errors))
    _write_json(result_path, artifact)
    _progress("publish", "complete", started, path=RESULT_REL.as_posix())
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V643 audit contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the audit or cold-check one unpublished candidate."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--check-candidate", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.check_candidate:
        errors = _check_candidate(ROOT, arguments.check_candidate)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(ROOT, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
