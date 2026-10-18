"""Measure the frozen V643 joint-claim approach on held-out source groups.

This module reuses the qualified V643 fixture, parser, exact executor, and
native CUDA transport. It adds held-out scheduling, private-process scoring,
complete cost reduction, source-group bootstrap intervals, and terminal gates.

Spec refs: REQ-VERIFY-7321 and SCENARIO-VERIFY-7321-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7209_v635_span_canary as native
from carnot import experiment_7306_v642_batch_fixture as fixture
from carnot import experiment_7317_v643_batch_harness as harness
from carnot import experiment_7320_v643_batch_canary as canary
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root
from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]

RUN_DATE = "20260915"
MILESTONE = "2026.09.643"
EXPERIMENT_ID = "exp7321-batch-measurement"
TASK_ID = "experiment_7321_v643_batch_measurement"
SCHEMA = "carnot.exp7321.v643_batch_measurement.v1"
MODEL_ID = canary.MODEL_ID
QUANTIZATION = canary.QUANTIZATION
MODEL_SPECS = deepcopy(canary.MODEL_SPECS)
ZERO_INVOCATION_COUNTS = deepcopy(canary.ZERO_INVOCATION_COUNTS)

DEVELOPMENT_SEED = fixture.DEVELOPMENT_SEED
EVALUATION_SEED = fixture.EVALUATION_SEED
BOOTSTRAP_SEED = 7_321_003
RANDOM_SEED = {
    "development": DEVELOPMENT_SEED,
    "evaluation": EVALUATION_SEED,
    "bootstrap": BOOTSTRAP_SEED,
}
PLANNED_GROUPS = 16
PLANNED_SOURCE_VERSIONS = 32
PLANNED_UNITS = 128
PLANNED_ROWS = 384
PLANNED_CALLS = 256
PLANNED_OUTPUT_TOKENS = 122_880
MODEL_SESSION_CAP_S = 3_000.0
MODEL_LOAD_CAP_S = canary.MODEL_LOAD_CAP_S
REQUEST_CAP_S = canary.REQUEST_CAP_S
FULL_GENERATION_FLOOR_S = 60.0
BOOTSTRAP_DRAWS = 10_000
ARM_ORDER = tuple(fixture.ARMS)

HARNESS_PATH = Path("results/experiment_7317_v643_batch_harness.json")
CANARY_PATH = Path("results/experiment_7320_v643_batch_canary.json")
PUBLIC_PATH = Path("results/raw/experiment_7317_v643_batch_harness/public_panel.json")
LABEL_PATH = Path("results/raw/experiment_7317_v643_batch_harness/evaluator_labels.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7321_v643_batch_measurement.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7321_v643_batch_measurement.py")
TEST_PATH = Path("tests/python/test_experiment_7321_v643_batch_measurement.py")
RESULT_PATH = Path("results/experiment_7321_v643_batch_measurement.json")
RAW_DIR = Path("results/raw/experiment_7321_v643_batch_measurement")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7321_v643_batch_measurement")
SCHEDULE_PATH = RAW_DIR / "schedule.json"
PER_CALL_PATH = RAW_DIR / "per_call_rows.json"
PREDICTION_PATH = RAW_DIR / "public_predictions.json"
SCORED_PATH = RAW_DIR / "scored_rows.json"
COST_PATH = RAW_DIR / "cost_receipt.json"
CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"

TERMINAL_CHECK_NAMES = (
    "candidate_reload_and_independent_reduce",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version this artifact and keep ordinary experiment and milestone fields.",
    "status": "Publish terminal output only after current work and required validation.",
    "run_date": "Use 20260915 and preserve actual UTC and monotonic phase spans.",
    "preconditions_checked": "Record input identities, availability, and each exact failed check.",
    "MODEL_SPECS": "List the current executable Qwen3.8 GGUF and its native identities.",
    "model_invoked": "Record every attempted model load or generation, including failures.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and active work.",
    "inference_substrate": "Describe actual computation with a recognized substrate literal.",
    "inference_substrate_class": "Classify actual model work, including a load-only attempt.",
    "execution_venue": "This milestone executes on the host.",
    "duration_s": "Measure real elapsed time without sleeps or padding.",
    "phase_spans": "Keep disjoint spans, units, checkpoint boundaries, and pending operations.",
    "random_seed": "Seal independent development, evaluation, and bootstrap seeds.",
    "reproducibility_checksum": "Bind code, public input, evaluator, settings, and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers and task-owned evidence.",
    "rows": "Keep every claim arm with decisions, costs, failures, and abstentions.",
    "sample_size_budget": "Record fixed planned, attempted, complete, and censored counts.",
    "acceptance_gate_results": "Keep expected, observed, passed, and purpose for each check.",
    "gate_check_summary": "Preserve the first exact failed check and both compared values.",
    "verifier_is_oracle": "Shared executor authority forbids a positive scientific class.",
    "honest_verdict": "Use complete_ for findings and blocked_ for external absence.",
    "verdict_class": "Use only the closed terminal class values.",
    "validation_receipts": "Keep commands, scopes, exits, elapsed times, and log hashes.",
    "repository_health": "Keep unrelated dated failures separate from current checks.",
    "field_principles": "Explain why each field exists without wrapping its value.",
    "batch_capture_complete_score": "One requires complete authenticated measurement evidence.",
    "batch_value_score": "One requires every frozen semantic, safety, coverage, and cost gate.",
    "per_source_group_results": "Keep paired metrics and complete costs for sixteen groups.",
    "paired_intervals": "Expose source-group vectors, seed, draws, estimates, and lower bounds.",
    "call_budget_contract": "Record planned and actual calls and tokens for every arm.",
}

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        *FIELD_PRINCIPLES,
        "experiment_id",
        "milestone",
        "timestamps",
        "inference_mode",
        "execution_host",
        "sealed_request_configuration",
        "runtime_model_identity",
        "gpu_receipts",
        "runner_receipt",
        "per_call_rows",
        "parser_replay_receipt",
        "cost_summary",
        "authority_separation",
        "validation_entrypoint_receipt",
        "production_path_enabled",
        "methodology_note",
    }
)


def _utc_now() -> str:
    """Record a real UTC boundary without using it as performance evidence."""

    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    """Use one compact JSON spelling for hashes and byte comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Label exact byte hashes so a digest cannot be mistaken for raw data."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without normalizing durable evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def gate_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    value_gate: bool = False,
    capture_gate: bool = False,
) -> JsonDict:
    """Keep both values and the role of one prerequisite or acceptance gate."""

    return {
        "check": check,
        "criterion": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "expected": expected,
        "observed_value": observed,
        "observed": observed,
        "passed": passed,
        "principle": principle,
        "value_gate": value_gate,
        "capture_gate": capture_gate,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failed check without changing its observed value."""

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
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
    }


def _dependency_rows(
    artifact: Mapping[str, Any] | None,
    *,
    prefix: str,
    upstream: str,
    score_field: str,
) -> list[JsonDict]:
    """Reject one absent, quarantined, failed, or unready current producer."""

    if artifact is None:
        return [
            gate_row(
                f"{prefix}_available",
                upstream,
                "artifact",
                "available",
                "missing_artifact",
                False,
                "The named current producer must exist before model work.",
            )
        ]
    quarantined = bool(
        artifact.get("flagged_adversarial") is True or artifact.get("quarantined") is True
    )
    terminal = artifact.get("verdict_class")
    failed_classes = {"blocked", "partial", "disqualified"}
    return [
        gate_row(
            f"{prefix}_status",
            upstream,
            "status",
            "complete",
            artifact.get("status"),
            artifact.get("status") == "complete",
            "The producer must be terminal before it authorizes held-out work.",
        ),
        gate_row(
            f"{prefix}_ready_score",
            upstream,
            score_field,
            1,
            artifact.get(score_field),
            artifact.get(score_field) == 1,
            "The producer readiness score must qualify the frozen next step.",
        ),
        gate_row(
            f"{prefix}_quarantine",
            upstream,
            "flagged_adversarial",
            False,
            artifact.get("flagged_adversarial") if quarantined else False,
            not quarantined,
            "Quarantined evidence cannot authorize current model work.",
        ),
        gate_row(
            f"{prefix}_terminal_class",
            upstream,
            "verdict_class",
            "not blocked, partial, or disqualified",
            terminal,
            terminal not in failed_classes,
            "A failed terminal class remains blocking even when readiness is one.",
        ),
    ]


def dependency_gate_rows(
    harness_artifact: Mapping[str, Any] | None,
    canary_artifact: Mapping[str, Any] | None,
) -> list[JsonDict]:
    """Authenticate both same-milestone readiness chains before model loading."""

    return [
        *_dependency_rows(
            harness_artifact,
            prefix="exp7317",
            upstream=harness.EXPERIMENT_ID,
            score_field="batch_harness_ready_score",
        ),
        *_dependency_rows(
            canary_artifact,
            prefix="exp7320",
            upstream=canary.EXPERIMENT_ID,
            score_field="batch_canary_ready_score",
        ),
    ]


def _held_out_panel(public_panel: Mapping[str, Any]) -> JsonDict:
    """Present only held-out public groups to the reused canary builders."""

    groups = public_panel.get("evaluation_groups")
    if not isinstance(groups, list) or len(groups) != PLANNED_GROUPS:
        raise ValueError("held_out_group_denominator")
    return {"development_groups": deepcopy(groups)}


def _balanced_arm_order(source_version_index: int) -> tuple[str, ...]:
    """Rotate which arm runs first without changing any arm request."""

    offset = source_version_index % len(ARM_ORDER)
    return (*ARM_ORDER[offset:], *ARM_ORDER[:offset])


def build_schedule(public_panel: Mapping[str, Any]) -> list[JsonDict]:
    """Build 256 held-out calls with frozen claim and balanced arm order."""

    held_out = _held_out_panel(public_panel)
    groups = held_out["development_groups"]
    base: list[JsonDict] = []
    for start in range(0, PLANNED_GROUPS, 2):
        chunk = {"development_groups": deepcopy(groups[start : start + 2])}
        ids = [str(row["group_id"]) for row in chunk["development_groups"]]
        base.extend(canary.build_schedule(chunk, ids))
    by_key: defaultdict[tuple[str, int, str], list[JsonDict]] = defaultdict(list)
    for row in base:
        by_key[(str(row["group_id"]), int(row["source_version"]), str(row["arm"]))].append(row)
    schedule: list[JsonDict] = []
    source_version_index = 0
    for group in groups:
        for version in (1, 2):
            arm_order = _balanced_arm_order(source_version_index)
            source_version_index += 1
            for arm_position, arm in enumerate(arm_order):
                for row in by_key[(str(group["group_id"]), version, arm)]:
                    schedule.append(
                        {
                            **deepcopy(row),
                            "call_order": len(schedule),
                            "split": "held_out",
                            "arm_order_position": arm_position,
                            "source_version_arm_order": list(arm_order),
                        }
                    )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]], public_panel: Mapping[str, Any]
) -> list[str]:
    """Rebuild the held-out schedule and name any identity or budget drift."""

    errors: list[str] = []
    try:
        expected = build_schedule(public_panel)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    if canonical_json(list(schedule)) != canonical_json(expected):
        errors.append("schedule_mismatch")
    if len(schedule) != PLANNED_CALLS:
        errors.append("call_count")
    allocated = sum(int(row.get("output_token_budget", 0) or 0) for row in schedule)
    if allocated != PLANNED_OUTPUT_TOKENS:
        errors.append("allocated_output_tokens")
    if Counter(row.get("arm") for row in schedule) != {
        "serial_versioned_verifier": 160,
        "batched_versioned_verifier": 64,
        "batched_warm_prefix_direct": 32,
    }:
        errors.append("arm_call_counts")
    for group_id in [str(row["group_id"]) for row in public_panel.get("evaluation_groups", [])]:
        for version in (1, 2):
            for arm in ARM_ORDER:
                budget = sum(
                    int(row.get("output_token_budget", 0) or 0)
                    for row in schedule
                    if row.get("group_id") == group_id
                    and row.get("source_version") == version
                    and row.get("arm") == arm
                )
                if budget != 1_280:
                    errors.append(f"arm_version_budget:{group_id}:v{version}:{arm}")
    return list(dict.fromkeys(errors))


def call_budget_contract(
    per_call_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Record the frozen allocation and every actual call and token outcome."""

    planned = {
        "source_groups": PLANNED_GROUPS,
        "source_versions": PLANNED_SOURCE_VERSIONS,
        "claims_per_source_version": 4,
        "claim_arm_rows": PLANNED_ROWS,
        "generation_calls": PLANNED_CALLS,
        "allocated_output_tokens": PLANNED_OUTPUT_TOKENS,
        "per_arm_source_version_output_tokens": 1_280,
        "per_arm_calls": {
            "serial_versioned_verifier": 160,
            "batched_versioned_verifier": 64,
            "batched_warm_prefix_direct": 32,
        },
        "retry_calls": 0,
        "repair_calls": 0,
        "stopping_rule": "fixed_16_groups_no_optional_stopping_or_subgroup_replacement",
    }
    per_arm: JsonDict = {}
    for arm in ARM_ORDER:
        selected = [row for row in per_call_rows if row.get("arm") == arm]
        per_arm[arm] = {
            "attempted_calls": len(selected),
            "completed_calls": sum(row.get("terminal_state") == "complete" for row in selected),
            "failed_calls": sum(row.get("terminal_state") == "failed" for row in selected),
            "cancelled_calls": planned["per_arm_calls"][arm] - len(selected),
            "allocated_output_tokens": sum(
                int(row.get("allocated_output_tokens", 0) or 0) for row in selected
            ),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in selected),
            "completion_tokens": sum(int(row.get("completion_tokens", 0) or 0) for row in selected),
            "summed_call_time_s": sum(float(row.get("latency_s", 0.0) or 0.0) for row in selected),
        }
    actual = {
        "attempted_calls": len(per_call_rows),
        "completed_calls": sum(row.get("terminal_state") == "complete" for row in per_call_rows),
        "failed_calls": sum(row.get("terminal_state") == "failed" for row in per_call_rows),
        "cancelled_calls": PLANNED_CALLS - len(per_call_rows),
        "in_flight_calls": 0,
        "allocated_output_tokens": sum(
            int(row.get("allocated_output_tokens", 0) or 0) for row in per_call_rows
        ),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in per_call_rows),
        "completion_tokens": sum(
            int(row.get("completion_tokens", 0) or 0) for row in per_call_rows
        ),
        "per_arm": per_arm,
    }
    return {"planned": planned, "actual": actual}


def _source_spans(
    prediction: Mapping[str, Any],
    public_panel: Mapping[str, Any],
    call_rows: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Recover public source mention spans from retained source completions."""

    if prediction.get("arm") == "batched_warm_prefix_direct":
        return []
    groups = {str(row["group_id"]): row for row in public_panel.get("evaluation_groups", [])}
    group = groups.get(str(prediction.get("group_id")), {})
    version = int(prediction.get("source_version", 0) or 0)
    sources = [
        row for row in group.get("source_versions", []) if row.get("source_version") == version
    ]
    if len(sources) != 1:
        return []
    document = sources[0].get("document", {})
    mentions = {str(row.get("mention_id")): row for row in document.get("mentions", [])}
    pointers: set[str] = set()
    for call_id in prediction.get("call_ids", []):
        call = call_rows.get(str(call_id), {})
        if call.get("call_type") != "source":
            continue
        completion = dict(call.get("normalized_response") or {}).get("completion", {})
        for relation in completion.get("relations", []):
            if isinstance(relation, Mapping):
                pointers.update(
                    str(relation.get(field))
                    for field in ("subject_pointer", "object_pointer")
                    if relation.get(field) is not None
                )
    return [
        {
            "mention_id": pointer,
            "byte_start": mentions[pointer].get("byte_start"),
            "byte_end": mentions[pointer].get("byte_end"),
            "surface_text": mentions[pointer].get("surface_text"),
        }
        for pointer in sorted(pointers)
        if pointer in mentions
    ]


def replay_public_predictions(
    public_panel: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    per_call_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Cold-replay public bytes without retaining any evaluator-only value."""

    held_out = _held_out_panel(public_panel)
    groups = held_out["development_groups"]
    retained_by_id = {str(row.get("call_id")): row for row in per_call_rows}
    predictions: list[JsonDict] = []
    replayed_calls: list[JsonDict] = []
    chunk_receipts: list[JsonDict] = []
    for start in range(0, PLANNED_GROUPS, 2):
        ids = {str(row["group_id"]) for row in groups[start : start + 2]}
        chunk_schedule = [row for row in schedule if str(row.get("group_id")) in ids]
        chunk_calls = [
            retained_by_id[str(row["call_id"])]
            for row in chunk_schedule
            if str(row["call_id"]) in retained_by_id
        ]
        replay = canary.cold_replay(held_out, chunk_schedule, chunk_calls)
        chunk_receipts.append(deepcopy(dict(replay["receipt"])))
        replayed_calls.extend(deepcopy(list(replay["replayed_per_call_rows"])))
        call_map = {str(row["call_id"]): row for row in replay["replayed_per_call_rows"]}
        for source_row in replay["rows"]:
            row = deepcopy(dict(source_row))
            for field in ("expected_prediction", "metric", "false_accept", "error"):
                row.pop(field, None)
            row["source_spans"] = _source_spans(row, public_panel, call_map)
            row["cost_kind"] = "measured_native_call_share_before_private_scoring"
            predictions.append(row)
    decisions = sorted(
        {str(row["prediction"]) for row in predictions if row.get("censored") is not True}
    )
    receipt = {
        "call_count": len(replayed_calls),
        "prediction_row_count": len(predictions),
        "transport_complete": bool(
            len(per_call_rows) == PLANNED_CALLS
            and len(replayed_calls) == PLANNED_CALLS
            and all(row.get("terminal_state") == "complete" for row in replayed_calls)
        ),
        "required_field_omission_count": sum(
            bool(row.get("required_field_omissions")) for row in replayed_calls
        ),
        "truncation_count": sum(bool(row.get("truncated")) for row in replayed_calls),
        "source_claim_identity_swap_count": sum(
            bool(row.get("source_claim_identity_swap")) for row in replayed_calls
        ),
        "replay_parser_match": all(
            row.get("retained_parser_match") is True for row in replayed_calls
        ),
        "decision_classes_observed": decisions,
        "model_calls_made_during_replay": 0,
        "evaluation_labels_read": False,
        "chunk_receipts": chunk_receipts,
    }
    return {
        "predictions": predictions,
        "replayed_per_call_rows": replayed_calls,
        "receipt": receipt,
    }


def score_predictions(
    predictions: Sequence[Mapping[str, Any]], evaluator: Mapping[str, Any]
) -> list[JsonDict]:
    """Join private labels after prediction sealing and retain all failures."""

    labels = evaluator.get("labels")
    if not isinstance(labels, list):
        raise ValueError("label_identity")
    held_out = [row for row in labels if row.get("split") == "evaluation"]
    by_id: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in held_out:
        by_id[str(row.get("unit_id"))].append(row)
    predicted_ids = {str(row.get("unit_id")) for row in predictions}
    if len(by_id) != PLANNED_UNITS or any(len(by_id[unit]) != 1 for unit in predicted_ids):
        raise ValueError("label_identity")
    rows: list[JsonDict] = []
    for prediction in predictions:
        row = deepcopy(dict(prediction))
        label = by_id[str(row["unit_id"])][0]
        expected = str(label["expected_decision"])
        failed = bool(row.get("censored") or row.get("failed_call_ids"))
        predicted = str(row.get("prediction"))
        confident = predicted in {"supported", "contradicted"}
        row.update(
            {
                "case_type": label.get("case_type"),
                "expected_decision": expected,
                "support_status": "supported" if expected == "supported" else "unsupported",
                "correct": int(not failed and predicted == expected),
                "coverage": int(not failed and not bool(row.get("abstention"))),
                "false_accept": int(not failed and confident and predicted != expected),
                "failed": failed,
                "failure": row.get("censoring_reason") if failed else None,
            }
        )
        rows.append(row)
    return rows


def per_source_group_results(
    rows: Sequence[Mapping[str, Any]],
    per_call_rows: Sequence[Mapping[str, Any]],
    cost_receipt: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict]:
    """Reduce paired metrics and allocate every warm cost exactly once."""

    group_ids = sorted({str(row.get("group_id")) for row in rows})
    if len(group_ids) != PLANNED_GROUPS:
        raise ValueError("source_group_denominator")
    call_count = max(1, len(per_call_rows))
    cell_count = PLANNED_GROUPS * len(ARM_ORDER)
    native_parsing = float(cost_receipt.get("native_parsing_s", 0.0) or 0.0)
    replay_total = float(cost_receipt.get("prediction_replay_s", 0.0) or 0.0)
    evaluation_total = float(cost_receipt.get("evaluation_s", 0.0) or 0.0)
    serialization_total = float(cost_receipt.get("serialization_s", 0.0) or 0.0)
    group_results: list[JsonDict] = []
    for group_id in group_ids:
        arms: JsonDict = {}
        for arm in ARM_ORDER:
            selected_rows = [
                row for row in rows if row.get("group_id") == group_id and row.get("arm") == arm
            ]
            selected_calls = [
                row
                for row in per_call_rows
                if row.get("group_id") == group_id and row.get("arm") == arm
            ]
            call_time = sum(float(row.get("latency_s", 0.0) or 0.0) for row in selected_calls)
            components = {
                "model_call_wall_s": call_time,
                "native_parsing_s": native_parsing * len(selected_calls) / call_count,
                "exact_verification_and_source_invalidation_s": replay_total / cell_count,
                "private_evaluation_s": evaluation_total / cell_count,
                "serialization_s": serialization_total / cell_count,
            }
            warm = sum(components.values())
            denominator = max(1, len(selected_rows))
            arms[arm] = {
                "row_count": len(selected_rows),
                "correct_rows": sum(int(row.get("correct", 0) or 0) for row in selected_rows),
                "covered_rows": sum(int(row.get("coverage", 0) or 0) for row in selected_rows),
                "failed_rows": sum(bool(row.get("failed")) for row in selected_rows),
                "abstentions": sum(bool(row.get("abstention")) for row in selected_rows),
                "false_accepts": sum(int(row.get("false_accept", 0) or 0) for row in selected_rows),
                "accuracy": sum(int(row.get("correct", 0) or 0) for row in selected_rows)
                / denominator,
                "coverage": sum(int(row.get("coverage", 0) or 0) for row in selected_rows)
                / denominator,
                "cost_components": components,
                "summed_call_time_s": call_time,
                "warm_complete_cost_s": warm,
            }
        by_unit = defaultdict(dict)
        for row in rows:
            if row.get("group_id") == group_id:
                by_unit[str(row["unit_id"])][str(row["arm"])] = row
        mismatches = sum(
            pair.get("serial_versioned_verifier", {}).get("prediction")
            != pair.get("batched_versioned_verifier", {}).get("prediction")
            for pair in by_unit.values()
        )
        group_results.append(
            {
                "group_id": group_id,
                "unit_count": len(by_unit),
                "semantic_mismatches": mismatches,
                "stale_constraints_served": sum(
                    bool(row.get("served_stale_constraints"))
                    for row in rows
                    if row.get("group_id") == group_id
                ),
                "arms": arms,
            }
        )
    summed_calls = sum(float(row.get("latency_s", 0.0) or 0.0) for row in per_call_rows)
    warm_complete = (
        summed_calls + native_parsing + replay_total + evaluation_total + serialization_total
    )
    shared_initialization = float(cost_receipt.get("shared_initialization_s", 0.0) or 0.0)
    summary = {
        "wall_time_s": float(cost_receipt.get("measurement_wall_s", 0.0) or 0.0),
        "summed_call_time_s": summed_calls,
        "shared_initialization_s": shared_initialization,
        "native_parsing_s": native_parsing,
        "exact_verification_and_source_invalidation_s": replay_total,
        "private_evaluation_s": evaluation_total,
        "serialization_s": serialization_total,
        "warm_complete_cost_s": warm_complete,
        "cold_total_wall_s": shared_initialization + warm_complete,
        "shared_initialization_counted_in_warm_cost": False,
        "shared_initialization_count": int(shared_initialization > 0.0),
        "speedup_cost_basis": "warm_complete_cost_excluding_one_shared_initialization",
        "wall_time_and_summed_call_time_distinct": True,
    }
    return group_results, summary


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select a deterministic empirical percentile without interpolation."""

    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def paired_intervals(
    group_results: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> JsonDict:
    """Bootstrap paired estimates by resampling all claims in a source group."""

    if len(group_results) != PLANNED_GROUPS:
        raise ValueError("source_group_denominator")
    vectors = {
        "accuracy_difference_vs_direct": [],
        "coverage_difference_vs_direct": [],
        "full_cost_speedup_vs_serial": [],
        "full_cost_speedup_vs_direct": [],
    }
    for group in group_results:
        arms = group["arms"]
        batch = arms["batched_versioned_verifier"]
        serial = arms["serial_versioned_verifier"]
        direct = arms["batched_warm_prefix_direct"]
        batch_cost = float(batch["warm_complete_cost_s"])
        vectors["accuracy_difference_vs_direct"].append(
            float(batch["accuracy"]) - float(direct["accuracy"])
        )
        vectors["coverage_difference_vs_direct"].append(
            float(batch["coverage"]) - float(direct["coverage"])
        )
        vectors["full_cost_speedup_vs_serial"].append(
            float(serial["warm_complete_cost_s"]) / batch_cost
        )
        vectors["full_cost_speedup_vs_direct"].append(
            float(direct["warm_complete_cost_s"]) / batch_cost
        )
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in vectors}
    for _draw in range(draws):
        indices = [rng.randrange(PLANNED_GROUPS) for _ in range(PLANNED_GROUPS)]
        samples["accuracy_difference_vs_direct"].append(
            sum(vectors["accuracy_difference_vs_direct"][index] for index in indices)
            / PLANNED_GROUPS
        )
        samples["coverage_difference_vs_direct"].append(
            sum(vectors["coverage_difference_vs_direct"][index] for index in indices)
            / PLANNED_GROUPS
        )
        for metric, control in (
            ("full_cost_speedup_vs_serial", "serial_versioned_verifier"),
            ("full_cost_speedup_vs_direct", "batched_warm_prefix_direct"),
        ):
            numerator = sum(
                float(group_results[index]["arms"][control]["warm_complete_cost_s"])
                for index in indices
            )
            denominator = sum(
                float(
                    group_results[index]["arms"]["batched_versioned_verifier"][
                        "warm_complete_cost_s"
                    ]
                )
                for index in indices
            )
            samples[metric].append(numerator / denominator)
    metrics: JsonDict = {}
    for metric in vectors:
        if metric.startswith("full_cost"):
            control = (
                "serial_versioned_verifier"
                if metric.endswith("serial")
                else "batched_warm_prefix_direct"
            )
            estimate = sum(
                float(group["arms"][control]["warm_complete_cost_s"]) for group in group_results
            ) / sum(
                float(group["arms"]["batched_versioned_verifier"]["warm_complete_cost_s"])
                for group in group_results
            )
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


def acceptance_gates(
    rows: Sequence[Mapping[str, Any]],
    group_results: Sequence[Mapping[str, Any]],
    intervals: Mapping[str, Any],
    *,
    capture_complete: bool,
    validation_passed: bool,
    provenance_passed: bool = True,
    duration_s: float = FULL_GENERATION_FLOOR_S,
) -> list[JsonDict]:
    """Apply the exact harness acceptance contract to measured held-out rows."""

    metrics = intervals["metrics"]
    false_accepts = {
        arm: sum(int(row.get("false_accept", 0) or 0) for row in rows if row.get("arm") == arm)
        for arm in ARM_ORDER
    }
    mismatches = sum(int(group.get("semantic_mismatches", 0) or 0) for group in group_results)
    stale = sum(int(group.get("stale_constraints_served", 0) or 0) for group in group_results)
    gates = [
        gate_row(
            "complete_authenticated_measurement",
            EXPERIMENT_ID,
            "capture_complete",
            True,
            capture_complete,
            capture_complete is True,
            "Capture needs every frozen group, call, row, and raw receipt.",
            capture_gate=True,
        ),
        gate_row(
            "owned_cuda_provenance",
            EXPERIMENT_ID,
            "gpu_receipts.provenance_ok",
            True,
            provenance_passed,
            provenance_passed is True,
            "A current live result needs owned CUDA generation evidence.",
            capture_gate=True,
        ),
        gate_row(
            "model_full_generation_duration_floor",
            EXPERIMENT_ID,
            "duration_s",
            f">={FULL_GENERATION_FLOOR_S}",
            duration_s,
            duration_s >= FULL_GENERATION_FLOOR_S,
            "Real full-generation work must meet the declared duration floor.",
            capture_gate=True,
        ),
        gate_row(
            "required_scoped_validation",
            EXPERIMENT_ID,
            "required_checks_passed",
            True,
            validation_passed,
            validation_passed is True,
            "A current affected failure disqualifies both measurement scores.",
            capture_gate=True,
        ),
        gate_row(
            "serial_batch_semantic_parity",
            EXPERIMENT_ID,
            "semantic_mismatches",
            0,
            mismatches,
            mismatches == 0,
            "Joint extraction must preserve each serial exact decision.",
            value_gate=True,
        ),
        gate_row(
            "zero_stale_constraints_served",
            EXPERIMENT_ID,
            "stale_constraints_served",
            0,
            stale,
            stale == 0,
            "A changed source version must invalidate older compiled state.",
            value_gate=True,
        ),
        gate_row(
            "false_accept_no_higher_than_serial",
            EXPERIMENT_ID,
            "false_accept_difference",
            "<=0",
            false_accepts["batched_versioned_verifier"]
            - false_accepts["serial_versioned_verifier"],
            false_accepts["batched_versioned_verifier"]
            <= false_accepts["serial_versioned_verifier"],
            "Batching must not add false accepts against serial verification.",
            value_gate=True,
        ),
        gate_row(
            "false_accept_no_higher_than_direct",
            EXPERIMENT_ID,
            "false_accept_difference",
            "<=0",
            false_accepts["batched_versioned_verifier"]
            - false_accepts["batched_warm_prefix_direct"],
            false_accepts["batched_versioned_verifier"]
            <= false_accepts["batched_warm_prefix_direct"],
            "Batching must not add false accepts against joint direct generation.",
            value_gate=True,
        ),
    ]
    interval_contract = (
        (
            "accuracy_difference_lower_vs_direct",
            "accuracy_difference_vs_direct",
            -0.02,
            "The accuracy lower bound limits batch degradation.",
        ),
        (
            "coverage_difference_lower_vs_direct",
            "coverage_difference_vs_direct",
            -0.02,
            "The coverage lower bound limits extra abstention.",
        ),
        (
            "full_cost_speedup_lower_vs_serial",
            "full_cost_speedup_vs_serial",
            1.5,
            "Complete warm batch cost must beat serial cost with margin.",
        ),
        (
            "full_cost_speedup_lower_vs_direct",
            "full_cost_speedup_vs_direct",
            1.5,
            "Complete warm batch cost must beat joint direct cost with margin.",
        ),
    )
    for check, metric, threshold, principle in interval_contract:
        observed = float(metrics[metric]["one_sided_95_lower"])
        gates.append(
            gate_row(
                check,
                EXPERIMENT_ID,
                f"paired_intervals.{metric}.one_sided_95_lower",
                threshold,
                observed,
                observed >= threshold,
                principle,
                value_gate=True,
            )
        )
    return gates


def classify_inference(counts: Mapping[str, Any]) -> JsonDict:
    """Derive model use and the full-generation class from recorded attempts."""

    loads = int(counts.get("model_loads_attempted", 0) or 0)
    generations = int(counts.get("generation_calls_attempted", 0) or 0)
    if generations:
        return {
            "model_invoked": True,
            "inference_substrate": "model_full_generation",
            "inference_substrate_class": "model_full_generation",
            "inference_mode": "live_gpu",
        }
    if loads:
        return {
            "model_invoked": True,
            "inference_substrate": "model_load_no_generation",
            "inference_substrate_class": "model_load_no_generation",
            "inference_mode": "live_gpu_load_only",
        }
    return {
        "model_invoked": False,
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_run",
    }


def _invocation_counts(
    per_call_rows: Sequence[Mapping[str, Any]], capture: Mapping[str, Any]
) -> JsonDict:
    """Count every actual model load and generation boundary."""

    attempted = len(per_call_rows)
    model_attempted = bool(capture)
    loaded = capture.get("model_loaded") is True
    return {
        "model_loads_attempted": int(model_attempted),
        "model_loads_completed": int(loaded),
        "model_loads_failed": int(model_attempted and not loaded),
        "model_loads_cancelled": 0,
        "model_loads_in_flight": 0,
        "generation_calls_attempted": attempted,
        "generation_calls_completed": sum(
            row.get("terminal_state") == "complete" for row in per_call_rows
        ),
        "generation_calls_failed": sum(
            row.get("terminal_state") == "failed" for row in per_call_rows
        ),
        "generation_calls_cancelled": max(0, PLANNED_CALLS - attempted),
        "generation_calls_in_flight": 0,
    }


def _model_specs(model_identity: Mapping[str, Any]) -> list[JsonDict]:
    """Materialize current model and native binary identities in MODEL_SPECS."""

    return [
        {
            **deepcopy(MODEL_SPECS[0]),
            "model_path": model_identity.get("gguf_path"),
            "revision": model_identity.get("revision"),
            "gguf_sha256": model_identity.get("gguf_sha256"),
            "binary_sha256": model_identity.get("binary_sha256"),
        }
    ]


def blocked_artifact(
    run_date: str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    timestamps: Mapping[str, Any],
    model_identity: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a terminal external block without success-shaped measurement rows."""

    summary = gate_check_summary(checks)
    identity = deepcopy(dict(model_identity or {}))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": run_date,
        "timestamps": deepcopy(dict(timestamps)),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": _model_specs(identity),
        **classify_inference(ZERO_INVOCATION_COUNTS),
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": duration_s,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "pending",
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
            "stopping_rule": "fixed_16_groups_no_optional_stopping_or_subgroup_replacement",
        },
        "acceptance_gate_results": [deepcopy(dict(row)) for row in checks],
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{summary['failed_check']}",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "repository_health": {"status": "not_observed", "affects_required_checks": False},
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "batch_capture_complete_score": 0,
        "batch_value_score": 0,
        "per_source_group_results": [],
        "paired_intervals": {},
        "call_budget_contract": call_budget_contract(),
        "sealed_request_configuration": None,
        "runtime_model_identity": identity,
        "gpu_receipts": {},
        "runner_receipt": {},
        "per_call_rows": [],
        "parser_replay_receipt": {},
        "cost_summary": {},
        "authority_separation": {
            "predictions_sealed_before_labels": False,
            "evaluation_labels_read_during_prediction": False,
            "evaluator_process_separate": True,
        },
        "validation_entrypoint_receipt": {
            "runner": harness.SCOPED_RUNNER,
            "called": False,
            "test_paths": [TEST_PATH.as_posix()],
            "changed_modules": [MODULE_PATH.as_posix()],
            "static_paths": [WRAPPER_PATH.as_posix()],
            "legacy_launcher_called": False,
            "repository_wide_target_present": False,
        },
        "production_path_enabled": False,
        "methodology_note": "An external prerequisite failed before model loading.",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def assemble_artifact(
    run_date: str,
    *,
    preconditions: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    per_call_rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    group_results: Sequence[Mapping[str, Any]],
    intervals: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
    validation: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    timestamps: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    cost_summary: Mapping[str, Any],
    authority_separation: Mapping[str, Any],
) -> JsonDict:
    """Assemble measured rows while keeping capture and value outcomes separate."""

    counts = _invocation_counts(per_call_rows, capture)
    inference = classify_inference(counts)
    capture_ready = all(row.get("passed") is True for row in gates if row.get("capture_gate"))
    value_ready = capture_ready and all(
        row.get("passed") is True for row in gates if row.get("value_gate")
    )
    validation_ready = validation.get("required_checks_passed") is True
    if not validation_ready:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_batch_measurement_validation_failed"
    elif capture.get("runtime_error") and len(per_call_rows) < PLANNED_CALLS:
        verdict_class = "blocked"
        verdict = "blocked_external_native_runtime_failure"
    elif not capture_ready:
        verdict_class = "null"
        verdict = "complete_null_batch_capture_incomplete"
    elif value_ready:
        verdict_class = "circular_positive"
        verdict = "complete_circular_positive_batch_value_gates_passed"
    else:
        verdict_class = "null"
        verdict = "complete_null_batch_value_gates_failed"
    budget = call_budget_contract(per_call_rows)
    schedule_hash = sha256_bytes(canonical_json(list(schedule)).encode("utf-8"))
    complete_groups = sum(
        all(row.get("failed") is not True for row in rows if row.get("group_id") == group_id)
        for group_id in {str(row.get("group_id")) for row in rows}
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked" if verdict_class == "blocked" else "complete",
        "run_date": run_date,
        "timestamps": deepcopy(dict(timestamps)),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": _model_specs(model_identity),
        **inference,
        "invocation_counts": counts,
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": deepcopy(list(rows)),
        "sample_size_budget": {
            "planned_groups": PLANNED_GROUPS,
            "attempted_groups": len({str(row.get("group_id")) for row in rows}),
            "complete_groups": complete_groups,
            "censored_groups": PLANNED_GROUPS - complete_groups,
            "planned_source_versions": PLANNED_SOURCE_VERSIONS,
            "planned_units_per_arm": PLANNED_UNITS,
            "planned_claim_arm_rows": PLANNED_ROWS,
            "attempted_claim_arm_rows": len(rows),
            "complete_claim_arm_rows": sum(row.get("failed") is not True for row in rows),
            "censored_claim_arm_rows": sum(row.get("failed") is True for row in rows),
            "stopping_rule": "fixed_16_groups_no_optional_stopping_or_subgroup_replacement",
            "post_hoc_larger_n": False,
            "narrowed_primary_subgroup": False,
        },
        "acceptance_gate_results": [deepcopy(dict(row)) for row in gates],
        "gate_check_summary": gate_check_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "validation_receipts": deepcopy(list(validation.get("validation_receipts") or [])),
        "repository_health": deepcopy(
            dict(validation.get("repository_health") or {"affects_required_checks": False})
        ),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "batch_capture_complete_score": int(capture_ready),
        "batch_value_score": int(value_ready),
        "per_source_group_results": deepcopy(list(group_results)),
        "paired_intervals": deepcopy(dict(intervals)),
        "call_budget_contract": budget,
        "sealed_request_configuration": {
            "schema": "carnot.exp7321.sealed_request.v1",
            "held_out_group_ids": [str(row["group_id"]) for row in group_results],
            "schedule_sha256": schedule_hash,
            "arm_order": list(ARM_ORDER),
            "balanced_arm_orders": [
                list(_balanced_arm_order(index)) for index in range(PLANNED_SOURCE_VERSIONS)
            ],
            "decoding_parameters": deepcopy(canary.DECODING_PARAMETERS),
            "grammar_sha256": sha256_bytes(canary.JSON_GRAMMAR.encode("utf-8")),
            "native_cache_behavior": {
                "cache_prompt": True,
                "single_server_resident_model": True,
                "replica_count": 1,
                "same_opportunity_for_joint_direct": True,
            },
            "model_session_cap_s": MODEL_SESSION_CAP_S,
            "request_cap_s": REQUEST_CAP_S,
            "outcome_driven_tuning": False,
            "retry_calls": 0,
            "repair_calls": 0,
        },
        "runtime_model_identity": deepcopy(dict(model_identity)),
        "gpu_receipts": deepcopy(dict(capture.get("gpu_receipts") or {})),
        "runner_receipt": deepcopy(dict(capture.get("runner_receipt") or {})),
        "per_call_rows": deepcopy(list(per_call_rows)),
        "parser_replay_receipt": deepcopy(dict(replay.get("receipt") or {})),
        "cost_summary": deepcopy(dict(cost_summary)),
        "authority_separation": deepcopy(dict(authority_separation)),
        "validation_entrypoint_receipt": deepcopy(
            dict(validation.get("validation_entrypoint_receipt") or {})
        ),
        "production_path_enabled": False,
        "methodology_note": (
            "This held-out result compares frozen serial, joint-claim, and joint-direct arms. "
            "Shared exact executor authority prevents a positive scientific class."
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check terminal identity, denominators, scores, and evidence hashes."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("schema")
    if (
        value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        errors.append("run_identity")
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    specs = value.get("MODEL_SPECS")
    if (
        not isinstance(specs, list)
        or len(specs) != 1
        or not isinstance(specs[0], Mapping)
        or specs[0].get("hf_id") != MODEL_ID
        or specs[0].get("quantization") != QUANTIZATION
    ):
        errors.append("MODEL_SPECS")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_identity")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if value.get("production_path_enabled") is not False:
        errors.append("production_path_enabled")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    counts = value.get("invocation_counts")
    if not isinstance(counts, Mapping):
        errors.append("invocation_counts")
    elif classify_inference(counts) != {
        field: value.get(field)
        for field in (
            "model_invoked",
            "inference_substrate",
            "inference_substrate_class",
            "inference_mode",
        )
    }:
        errors.append("inference_classification")
    if value.get("status") == "blocked" and not value.get("per_call_rows"):
        if (
            value.get("batch_capture_complete_score") != 0
            or value.get("batch_value_score") != 0
            or value.get("verdict_class") != "blocked"
        ):
            errors.append("blocked_terminal_state")
        return list(dict.fromkeys(errors))
    rows = value.get("rows")
    calls = value.get("per_call_rows")
    groups = value.get("per_source_group_results")
    intervals = value.get("paired_intervals")
    if not isinstance(rows, list) or len(rows) != PLANNED_ROWS:
        errors.append("row_denominator")
    if not isinstance(calls, list) or len(calls) != PLANNED_CALLS:
        errors.append("call_denominator")
    if not isinstance(groups, list) or len(groups) != PLANNED_GROUPS:
        errors.append("group_denominator")
    if not isinstance(intervals, Mapping) or intervals.get("bootstrap_unit") != "source_group":
        errors.append("paired_intervals")
    gates = list(value.get("acceptance_gate_results") or [])
    expected_capture = int(
        bool(gates) and all(row.get("passed") is True for row in gates if row.get("capture_gate"))
    )
    expected_value = int(
        expected_capture == 1
        and all(row.get("passed") is True for row in gates if row.get("value_gate"))
    )
    if value.get("batch_capture_complete_score") != expected_capture:
        errors.append("batch_capture_complete_score")
    if value.get("batch_value_score") != expected_value:
        errors.append("batch_value_score")
    receipts = list(value.get("validation_receipts") or [])
    names = Counter(str(row.get("name")) for row in receipts)
    required_names = (*scoped.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES, "private_evaluator")
    validation_ready = all(names[name] == 1 for name in required_names) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in receipts
    )
    if require_validation and expected_capture and not validation_ready:
        errors.append("validation_receipts")
    if value.get("batch_value_score") == 1 and value.get("verdict_class") != "circular_positive":
        errors.append("verdict_class")
    if value.get("verdict_class") == "positive":
        errors.append("oracle_positive_forbidden")
    if require_validation and value.get("verdict_class") == "disqualified" and validation_ready:
        errors.append("disqualified_without_validation_failure")
    return list(dict.fromkeys(errors))


def _progress(phase: str, event: str, started: float, **details: Any) -> None:  # pragma: no cover
    """Flush every boundary with measured monotonic elapsed time."""

    print(
        canonical_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 6),
                **details,
            }
        ),
        flush=True,
    )


def _phase_row(
    phase: str,
    started: float,
    units: int,
    checkpoint: str | None = None,
) -> JsonDict:  # pragma: no cover
    """Close one disjoint phase with units and a durable boundary."""

    return {
        "phase": phase,
        "duration_s": time.monotonic() - started,
        "units": units,
        "checkpoint_boundary": checkpoint,
        "pending_operations": [],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write task-owned JSON atomically through the repository helper."""

    atomic_write_json(path, value, allow_override=True, sort_keys=True)


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover
    """Hash exact producers, settings, private authority, and raw evidence."""

    paths = {
        "AGENTS.md": Path("AGENTS.md"),
        "CLAUDE.md": Path("CLAUDE.md"),
        "CODEX.md": Path("CODEX.md"),
        "research-program.md": Path("research-program.md"),
        "research-references.md": Path("research-references.md"),
        EXCLUSION_PATH.as_posix(): EXCLUSION_PATH,
        "ops/e2e-test-plan.md": Path("ops/e2e-test-plan.md"),
        SPEC_PATH.as_posix(): SPEC_PATH,
        HARNESS_PATH.as_posix(): HARNESS_PATH,
        CANARY_PATH.as_posix(): CANARY_PATH,
        PUBLIC_PATH.as_posix(): PUBLIC_PATH,
        LABEL_PATH.as_posix(): LABEL_PATH,
        "python/carnot/experiment_7306_v642_batch_fixture.py": Path(
            "python/carnot/experiment_7306_v642_batch_fixture.py"
        ),
        "python/carnot/experiment_7317_v643_batch_harness.py": Path(
            "python/carnot/experiment_7317_v643_batch_harness.py"
        ),
        "python/carnot/experiment_7320_v643_batch_canary.py": Path(
            "python/carnot/experiment_7320_v643_batch_canary.py"
        ),
        "python/carnot/reporting/experiment_7303_validation_scope.py": Path(
            "python/carnot/reporting/experiment_7303_validation_scope.py"
        ),
        MODULE_PATH.as_posix(): MODULE_PATH,
        WRAPPER_PATH.as_posix(): WRAPPER_PATH,
        TEST_PATH.as_posix(): TEST_PATH,
        SCHEDULE_PATH.as_posix(): SCHEDULE_PATH,
        PER_CALL_PATH.as_posix(): PER_CALL_PATH,
        PREDICTION_PATH.as_posix(): PREDICTION_PATH,
        SCORED_PATH.as_posix(): SCORED_PATH,
        COST_PATH.as_posix(): COST_PATH,
        "scripts/adversarial_verify.py": Path("scripts/adversarial_verify.py"),
        "scripts/verdict_row_consistency_lint.py": Path("scripts/verdict_row_consistency_lint.py"),
    }
    return {
        name: {
            "sha256": sha256_file(root / path) if (root / path).is_file() else "missing",
            "producer_identity": name,
        }
        for name, path in paths.items()
    }


def _authenticate_inputs(
    root: Path, run_date: str, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate both producers, public bytes, sealed settings, cache, and GPU."""

    checks: list[JsonDict] = []
    context: JsonDict = {}

    def record(row: JsonDict) -> None:
        _progress("preconditions", "check_start", started, check=row["check"])
        checks.append(row)
        _progress("preconditions", "check_end", started, check=row["check"], passed=row["passed"])

    record(
        gate_row(
            "run_date",
            EXPERIMENT_ID,
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            "The execution date is fixed by the milestone contract.",
        )
    )
    required = {
        "harness": HARNESS_PATH,
        "canary": CANARY_PATH,
        "public_panel": PUBLIC_PATH,
        "evaluator_labels": LABEL_PATH,
        "exclusion_manifest": EXCLUSION_PATH,
        "verification_spec": SPEC_PATH,
        "module": MODULE_PATH,
        "entrypoint": WRAPPER_PATH,
        "focused_tests": TEST_PATH,
    }
    availability = {name: (root / path).is_file() for name, path in required.items()}
    record(
        gate_row(
            "required_paths",
            "repository",
            "paths",
            {name: True for name in availability},
            availability,
            all(availability.values()),
            "Every named input must exist before model loading.",
        )
    )
    if not all(availability.values()):
        return checks, context
    try:
        harness_artifact = json.loads((root / HARNESS_PATH).read_text(encoding="utf-8"))
        canary_artifact = json.loads((root / CANARY_PATH).read_text(encoding="utf-8"))
        public_panel = json.loads((root / PUBLIC_PATH).read_text(encoding="utf-8"))
        exclusions = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        record(
            gate_row(
                "public_input_serialization",
                "repository",
                "public_json_and_yaml",
                "valid",
                f"{type(exc).__name__}:{exc}",
                False,
                "Malformed public evidence cannot be repaired during authentication.",
            )
        )
        return checks, context
    record(
        gate_row(
            "exp7317_cold_validation",
            harness.EXPERIMENT_ID,
            "schema_and_checksum",
            [],
            harness.validate_artifact(harness_artifact),
            not harness.validate_artifact(harness_artifact),
            "The shipped harness validator must accept the exact producer bytes.",
        )
    )
    record(
        gate_row(
            "exp7320_cold_validation",
            canary.EXPERIMENT_ID,
            "schema_and_checksum",
            [],
            canary.validate_artifact(canary_artifact),
            not canary.validate_artifact(canary_artifact),
            "The shipped canary validator must accept the exact producer bytes.",
        )
    )
    for row in dependency_gate_rows(harness_artifact, canary_artifact):
        record(row)
    capture_inputs = harness.build_capture_inputs(harness_artifact)
    record(
        gate_row(
            "exp7317_capture_builder",
            harness.EXPERIMENT_ID,
            "build_capture_inputs.ok",
            True,
            capture_inputs.get("ok"),
            capture_inputs.get("ok") is True,
            "The harness owns the held-out selection and explicit validation scope.",
        )
    )
    expected_public_hash = capture_inputs.get("public_panel_sha256")
    observed_public_hash = sha256_bytes(canonical_json(public_panel).encode("utf-8"))
    record(
        gate_row(
            "public_panel_authentication",
            harness.EXPERIMENT_ID,
            "public_panel_sha256",
            expected_public_hash,
            observed_public_hash,
            observed_public_hash == expected_public_hash,
            "Predictions must use the exact public bytes qualified by the harness.",
        )
    )
    sealed = dict(canary_artifact.get("sealed_transport_config") or {})
    expected_sealed = {
        "decoding_parameters": canary.DECODING_PARAMETERS,
        "grammar_sha256": sha256_bytes(canary.JSON_GRAMMAR.encode("utf-8")),
        "arm_order": list(ARM_ORDER),
        "retry_calls": 0,
        "repair_calls": 0,
        "cache_prompt": True,
    }
    observed_sealed = {
        "decoding_parameters": sealed.get("decoding_parameters"),
        "grammar_sha256": sealed.get("grammar_sha256"),
        "arm_order": sealed.get("arm_order"),
        "retry_calls": sealed.get("retry_calls"),
        "repair_calls": sealed.get("repair_calls"),
        "cache_prompt": dict(sealed.get("native_cache_behavior") or {}).get("cache_prompt"),
    }
    record(
        gate_row(
            "sealed_request_configuration",
            canary.EXPERIMENT_ID,
            "sealed_transport_config",
            expected_sealed,
            observed_sealed,
            observed_sealed == expected_sealed,
            "Held-out requests must reuse canary-qualified decoding and cache settings.",
        )
    )
    excluded = any(
        fixture.reuse._manifest_lists_experiment(exclusions, experiment)
        for experiment in (EXPERIMENT_ID, harness.EXPERIMENT_ID, canary.EXPERIMENT_ID)
    )
    record(
        gate_row(
            "exclusion_manifest",
            EXCLUSION_PATH.as_posix(),
            "retired_or_quarantined",
            False,
            excluded,
            not excluded,
            "Retired or quarantined work cannot silently resume.",
        )
    )
    try:
        schedule = build_schedule(public_panel)
        schedule_problem = schedule_errors(schedule, public_panel)
    except (KeyError, TypeError, ValueError) as exc:
        schedule = []
        schedule_problem = [f"{type(exc).__name__}:{exc}"]
    record(
        gate_row(
            "frozen_held_out_schedule",
            harness.EXPERIMENT_ID,
            "schedule",
            {"calls": PLANNED_CALLS, "tokens": PLANNED_OUTPUT_TOKENS, "errors": []},
            {
                "calls": len(schedule),
                "tokens": sum(int(row.get("output_token_budget", 0)) for row in schedule),
                "errors": schedule_problem,
            },
            not schedule_problem,
            "All sixteen groups and equal arm budgets must be fixed before generation.",
        )
    )
    if any(row["passed"] is not True for row in checks):
        context["model_identity"] = deepcopy(
            dict(canary_artifact.get("runtime_model_identity") or {})
        )
        return checks, context
    runtime_checks, runtime = canary._authenticate_inputs(root, run_date, started)
    checks.extend(deepcopy(runtime_checks))
    runtime.update(
        {
            "harness_artifact": harness_artifact,
            "canary_artifact": canary_artifact,
            "public_panel": public_panel,
            "capture_inputs": capture_inputs,
            "schedule": schedule,
            "task_id": TASK_ID,
            "live_window_cap_s": MODEL_SESSION_CAP_S,
            "request_cap_s": REQUEST_CAP_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
            "completion_builder": canary.build_per_call_row,
        }
    )
    return checks, runtime


def _checkpointing_builder(
    checkpoint_dir: Path, schedule: Sequence[Mapping[str, Any]]
) -> Any:  # pragma: no cover
    """Persist all completed rows whenever one source-group boundary closes."""

    last_order = {
        group_id: max(
            int(row["call_order"]) for row in schedule if str(row["group_id"]) == group_id
        )
        for group_id in {str(row["group_id"]) for row in schedule}
    }
    completed: list[JsonDict] = []

    def build(
        sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
    ) -> JsonDict:
        row = canary.build_per_call_row(sealed, response, resource)
        completed.append(deepcopy(row))
        group_id = str(sealed["group_id"])
        if int(sealed["call_order"]) == last_order[group_id]:
            _write_json(
                checkpoint_dir / f"group_{group_id}.json",
                {
                    "group_id": group_id,
                    "completed_calls": len(completed),
                    "rows": completed,
                    "pending_operation": None,
                },
            )
        return row

    return build


def _scoped_validation(
    root: Path,
    scope: Mapping[str, Any],
    historical_failures: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover
    """Run the shipped explicit-scope validator with a private temp directory."""

    basetemp = Path(tempfile.mkdtemp(prefix="carnot-exp7321-"))
    result = scoped.run_scoped_validation(
        root,
        list(scope["test_paths"]),
        list(scope["changed_modules"]),
        static_paths=list(scope["static_paths"]),
        basetemp=basetemp,
        coverage_file=root / RAW_DIR / ".coverage",
        log_dir=root / RAW_DIR / "validation",
        historical_failures=historical_failures,
    )
    result["validation_entrypoint_receipt"] = {
        "runner": harness.SCOPED_RUNNER,
        "called": True,
        "test_paths": list(scope["test_paths"]),
        "changed_modules": list(scope["changed_modules"]),
        "static_paths": list(scope["static_paths"]),
        "legacy_launcher_called": False,
        "repository_wide_target_present": False,
    }
    return result


def _terminal_commands(root: Path, candidate: Path) -> list[scoped.CommandSpec]:  # pragma: no cover
    """Build independent artifact checks without a broad test target."""

    python = str(root / ".venv/bin/python")
    return [
        scoped.CommandSpec(
            TERMINAL_CHECK_NAMES[0],
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--check-artifact",
                str(candidate),
                "--raw-rows",
                str(root / SCORED_PATH),
                "--raw-calls",
                str(root / PER_CALL_PATH),
                "--cost-receipt",
                str(root / COST_PATH),
            ),
            "terminal_candidate_and_raw_evidence",
        ),
        scoped.CommandSpec(
            TERMINAL_CHECK_NAMES[1],
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        scoped.CommandSpec(
            TERMINAL_CHECK_NAMES[2],
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


def _private_evaluator_command(root: Path) -> scoped.CommandSpec:  # pragma: no cover
    """Give private labels to a separate process only after predictions are sealed."""

    return scoped.CommandSpec(
        "private_evaluator",
        (
            str(root / ".venv/bin/python"),
            "-u",
            WRAPPER_PATH.as_posix(),
            "--date",
            RUN_DATE,
            "--score-predictions",
            str(root / PREDICTION_PATH),
            "--labels",
            str(root / LABEL_PATH),
            "--scored-output",
            str(root / SCORED_PATH),
        ),
        "sealed_predictions_and_private_labels",
    )


def _span_duration(spans: Sequence[Mapping[str, Any]], name: str) -> float:  # pragma: no cover
    """Read one native measured phase without treating summed phases as wall time."""

    return sum(float(row.get("duration_s", 0.0) or 0.0) for row in spans if row.get("name") == name)


def run_experiment(
    root: Path | None = None, run_date: str = RUN_DATE
) -> JsonDict:  # pragma: no cover
    """Authenticate, capture, score separately, validate, and publish once."""

    repository = root or find_repo_root(start=__file__)
    result_path = repository / RESULT_PATH
    print("[exp7321] startup", flush=True)
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(existing)
        if errors:
            raise ValueError("existing Exp7321 artifact invalid: " + ",".join(errors))
        print("[exp7321] stable terminal exists", flush=True)
        return dict(existing)
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    started = time.monotonic()
    measurement_started = started
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = repository / RAW_DIR
    checkpoint_dir = repository / CHECKPOINT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    phase = time.monotonic()
    _progress("preconditions", "start", started)
    checks, context = _authenticate_inputs(repository, run_date, started)
    spans.append(_phase_row("preconditions", phase, len(checks)))
    _progress("preconditions", "complete", started, units=len(checks))
    failure = next((row for row in checks if row.get("passed") is not True), None)
    if failure is not None:
        blocked = blocked_artifact(
            run_date,
            checks,
            duration_s=time.monotonic() - started,
            timestamps={"started_at_utc": started_at, "completed_at_utc": _utc_now()},
            model_identity=context.get("model_identity", {}),
        )
        blocked["phase_spans"] = spans
        blocked["source_artifact_hashes"] = _source_hashes(repository)
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        errors = validate_artifact(blocked, require_validation=False)
        if errors:
            raise ValueError("invalid blocked Exp7321 artifact: " + ",".join(errors))
        _progress("publish", "before", started, verdict=blocked["honest_verdict"])
        _write_json(result_path, blocked)
        _progress("publish", "after", started, verdict=blocked["honest_verdict"])
        return blocked

    schedule = list(context["schedule"])
    phase = time.monotonic()
    _progress("tokenizer", "before_model_load", started, operation="embedded_vocab_only")
    owner, tokenizer, tokenizer_receipt = native._load_embedded_tokenizer(
        Path(context["model_path"]), context["tokenizer_loader"]
    )
    _progress(
        "tokenizer",
        "after_model_load",
        started,
        operation="embedded_vocab_only",
        available=tokenizer is not None,
    )
    prompt_counts = (
        [len(tokenizer(str(row["prompt"]).encode("utf-8"))) for row in schedule]
        if tokenizer
        else []
    )
    if owner is not None:
        owner.close()
    context["model_identity"]["embedded_tokenizer_load_receipt"] = tokenizer_receipt
    context["model_identity"]["frozen_prompt_token_counts"] = prompt_counts
    spans.append(_phase_row("embedded_tokenizer", phase, len(prompt_counts)))
    if tokenizer is None:
        checks.append(
            gate_row(
                "embedded_tokenizer_load",
                MODEL_ID,
                "embedded_tokenizer",
                True,
                False,
                False,
                "The GGUF tokenizer must load before native generation.",
            )
        )
        blocked = blocked_artifact(
            run_date,
            checks,
            duration_s=time.monotonic() - started,
            timestamps={"started_at_utc": started_at, "completed_at_utc": _utc_now()},
            model_identity=context["model_identity"],
        )
        blocked["phase_spans"] = spans
        blocked["source_artifact_hashes"] = _source_hashes(repository)
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        _write_json(result_path, blocked)
        return blocked

    phase = time.monotonic()
    _progress("sealed_public_schedule", "start", started, calls=len(schedule))
    serialization_started = time.monotonic()
    _write_json(repository / SCHEDULE_PATH, {"schedule": schedule})
    serialization_s = time.monotonic() - serialization_started
    spans.append(
        _phase_row("sealed_public_schedule", phase, len(schedule), SCHEDULE_PATH.as_posix())
    )
    _progress("sealed_public_schedule", "complete", started, calls=len(schedule))

    context["completion_builder"] = _checkpointing_builder(checkpoint_dir, schedule)
    _progress("live_capture", "before_model_load_and_generation", started, calls=len(schedule))
    capture = native._live_capture(context, checkpoint_dir, raw_dir, spans)
    _progress(
        "live_capture",
        "after_model_load_and_generation",
        started,
        completed_units=len(capture["rows"]),
        total_units=len(schedule),
    )
    per_call_rows = list(capture["rows"])
    serialization_started = time.monotonic()
    _write_json(repository / PER_CALL_PATH, {"per_call_rows": per_call_rows})
    serialization_s += time.monotonic() - serialization_started

    phase = time.monotonic()
    replay_started = time.monotonic()
    _progress("public_replay", "start", started, calls=len(per_call_rows))
    replay = replay_public_predictions(context["public_panel"], schedule, per_call_rows)
    prediction_replay_s = time.monotonic() - replay_started
    serialization_started = time.monotonic()
    _write_json(
        repository / PREDICTION_PATH,
        {
            "schema": "carnot.exp7321.sealed_public_predictions.v1",
            "predictions": replay["predictions"],
            "prediction_sha256": sha256_bytes(
                canonical_json(replay["predictions"]).encode("utf-8")
            ),
            "sealed_at_utc": _utc_now(),
            "evaluation_labels_read": False,
        },
    )
    serialization_s += time.monotonic() - serialization_started
    spans.append(
        _phase_row("public_replay", phase, len(replay["predictions"]), PREDICTION_PATH.as_posix())
    )
    _progress("public_replay", "complete", started, rows=len(replay["predictions"]))

    phase = time.monotonic()
    _progress("private_evaluator", "before_subprocess", started, rows=len(replay["predictions"]))
    evaluator_receipts = scoped.run_commands(
        repository,
        [_private_evaluator_command(repository)],
        log_dir=repository / RAW_DIR / "private-evaluator",
    )
    _progress("private_evaluator", "after_subprocess", started)
    spans.append(_phase_row("private_evaluator", phase, 1, SCORED_PATH.as_posix()))
    evaluator_receipt = evaluator_receipts[0]
    if evaluator_receipt.get("passed") is not True or not (repository / SCORED_PATH).is_file():
        raise RuntimeError("private evaluator failed before terminal assembly")
    scored_payload = json.loads((repository / SCORED_PATH).read_text(encoding="utf-8"))
    rows = list(scored_payload["scored_rows"])
    evaluation_s = float(scored_payload["evaluator_receipt"]["duration_s"])
    authority = {
        "prediction_process_inputs": [PUBLIC_PATH.as_posix(), SCHEDULE_PATH.as_posix()],
        "evaluator_process_inputs": [PREDICTION_PATH.as_posix(), LABEL_PATH.as_posix()],
        "predictions_sealed_before_labels": True,
        "evaluation_labels_read_during_prediction": False,
        "evaluator_process_separate": True,
        "prediction_pid": os.getpid(),
        "evaluator_pid": scored_payload["evaluator_receipt"]["pid"],
        "public_prediction_sha256": scored_payload["evaluator_receipt"]["public_prediction_sha256"],
        "evaluator_label_file_sha256": scored_payload["evaluator_receipt"][
            "evaluator_label_file_sha256"
        ],
        "evaluator_label_canonical_sha256": scored_payload["evaluator_receipt"][
            "evaluator_label_canonical_sha256"
        ],
    }

    cost_receipt = {
        "shared_initialization_s": _span_duration(spans, "task_owned_lease_and_model_load"),
        "native_parsing_s": _span_duration(spans, "parsing_aggregate"),
        "prediction_replay_s": prediction_replay_s,
        "evaluation_s": evaluation_s,
        "serialization_s": serialization_s,
        "measurement_wall_s": time.monotonic() - measurement_started,
    }
    groups, cost_summary = per_source_group_results(rows, per_call_rows, cost_receipt)
    intervals = paired_intervals(groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    serialization_started = time.monotonic()
    _write_json(repository / COST_PATH, cost_receipt)
    serialization_s += time.monotonic() - serialization_started
    cost_receipt["serialization_s"] = serialization_s
    groups, cost_summary = per_source_group_results(rows, per_call_rows, cost_receipt)
    intervals = paired_intervals(groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    _write_json(repository / COST_PATH, cost_receipt)

    phase = time.monotonic()
    _progress("scoped_validation", "before_subprocesses", started, units=8)
    historical = harness.repository_health(repository).get("historical_failures", [])
    validation = _scoped_validation(
        repository,
        context["capture_inputs"]["validation_scope"],
        historical,
    )
    validation["validation_receipts"] = [
        evaluator_receipt,
        *validation["validation_receipts"],
    ]
    if evaluator_receipt.get("passed") is not True:
        validation["required_checks_passed"] = False
        validation["failed_required_commands"] = [
            *validation.get("failed_required_commands", []),
            "private_evaluator",
        ]
    spans.append(_phase_row("scoped_validation", phase, len(validation["validation_receipts"])))
    _progress("scoped_validation", "after_subprocesses", started)

    capture_complete = bool(
        len(per_call_rows) == PLANNED_CALLS
        and replay["receipt"]["transport_complete"] is True
        and len(rows) == PLANNED_ROWS
    )
    provenance_passed = dict(capture.get("gpu_receipts") or {}).get("provenance_ok") is True
    duration = time.monotonic() - started
    gates = acceptance_gates(
        rows,
        groups,
        intervals,
        capture_complete=capture_complete,
        validation_passed=validation.get("required_checks_passed") is True,
        provenance_passed=provenance_passed,
        duration_s=duration,
    )
    source_hashes = _source_hashes(repository)
    candidate = assemble_artifact(
        run_date,
        preconditions=checks,
        schedule=schedule,
        per_call_rows=per_call_rows,
        replay=replay,
        rows=rows,
        group_results=groups,
        intervals=intervals,
        gates=gates,
        capture=capture,
        validation=validation,
        source_hashes=source_hashes,
        duration_s=duration,
        phase_spans=spans,
        timestamps={"started_at_utc": started_at, "completed_at_utc": _utc_now()},
        model_identity=context["model_identity"],
        cost_summary=cost_summary,
        authority_separation=authority,
    )
    candidate_errors = validate_artifact(candidate, require_validation=False)
    if candidate_errors:
        raise ValueError("Exp7321 candidate invalid: " + ",".join(candidate_errors))
    _write_json(repository / CANDIDATE_PATH, candidate)

    phase = time.monotonic()
    _progress("terminal_validation", "before_subprocesses", started, units=3)
    terminal_receipts = scoped.run_commands(
        repository,
        _terminal_commands(repository, repository / CANDIDATE_PATH),
        log_dir=repository / RAW_DIR / "terminal-validation",
    )
    spans.append(_phase_row("terminal_validation", phase, len(terminal_receipts)))
    _progress("terminal_validation", "after_subprocesses", started)
    validation["validation_receipts"] = [
        *validation["validation_receipts"],
        *terminal_receipts,
    ]
    failed_terminal = [row["name"] for row in terminal_receipts if row.get("passed") is not True]
    if failed_terminal:
        validation["required_checks_passed"] = False
        validation["failed_required_commands"] = [
            *validation.get("failed_required_commands", []),
            *failed_terminal,
        ]
    final_duration = time.monotonic() - started
    gates = acceptance_gates(
        rows,
        groups,
        intervals,
        capture_complete=capture_complete,
        validation_passed=validation.get("required_checks_passed") is True,
        provenance_passed=provenance_passed,
        duration_s=final_duration,
    )
    artifact = assemble_artifact(
        run_date,
        preconditions=checks,
        schedule=schedule,
        per_call_rows=per_call_rows,
        replay=replay,
        rows=rows,
        group_results=groups,
        intervals=intervals,
        gates=gates,
        capture=capture,
        validation=validation,
        source_hashes=source_hashes,
        duration_s=final_duration,
        phase_spans=spans,
        timestamps={"started_at_utc": started_at, "completed_at_utc": _utc_now()},
        model_identity=context["model_identity"],
        cost_summary=cost_summary,
        authority_separation=authority,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("Exp7321 terminal invalid: " + ",".join(errors))
    _progress("publish", "before", started, verdict=artifact["honest_verdict"])
    _write_json(result_path, artifact)
    _progress("publish", "after", started, verdict=artifact["honest_verdict"])
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V643 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"--date must be {RUN_DATE}")
    return value


def _score_sidecar(
    prediction_path: Path, label_path: Path, output_path: Path
) -> None:  # pragma: no cover
    """Read private labels only inside the evaluator process after sealing."""

    started = time.monotonic()
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    rows = score_predictions(predictions["predictions"], labels)
    _write_json(
        output_path,
        {
            "schema": "carnot.exp7321.private_evaluation.v1",
            "scored_rows": rows,
            "evaluator_receipt": {
                "pid": os.getpid(),
                "duration_s": time.monotonic() - started,
                "public_prediction_sha256": sha256_bytes(
                    canonical_json(predictions["predictions"]).encode("utf-8")
                ),
                "evaluator_label_file_sha256": sha256_file(label_path),
                "evaluator_label_canonical_sha256": sha256_bytes(
                    canonical_json(labels).encode("utf-8")
                ),
                "label_count": len(labels["labels"]),
                "scored_row_count": len(rows),
            },
        },
    )


def _check_raw_reduction(
    artifact: Mapping[str, Any], raw_rows: Path, raw_calls: Path, cost_path: Path
) -> list[str]:  # pragma: no cover
    """Independently reduce raw rows without trusting artifact aggregates."""

    rows = json.loads(raw_rows.read_text(encoding="utf-8"))["scored_rows"]
    calls = json.loads(raw_calls.read_text(encoding="utf-8"))["per_call_rows"]
    costs = json.loads(cost_path.read_text(encoding="utf-8"))
    groups, summary = per_source_group_results(rows, calls, costs)
    intervals = paired_intervals(groups, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    errors = []
    if rows != artifact.get("rows"):
        errors.append("independent_raw_rows")
    if groups != artifact.get("per_source_group_results"):
        errors.append("independent_group_reduction")
    if intervals != artifact.get("paired_intervals"):
        errors.append("independent_paired_intervals")
    if summary != artifact.get("cost_summary"):
        errors.append("independent_cost_summary")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run measurement, private scoring, or independent terminal validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--score-predictions", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--scored-output", type=Path)
    parser.add_argument("--check-artifact", type=Path)
    parser.add_argument("--raw-rows", type=Path)
    parser.add_argument("--raw-calls", type=Path)
    parser.add_argument("--cost-receipt", type=Path)
    arguments = parser.parse_args(argv)
    print("[exp7321] startup", flush=True)
    if arguments.score_predictions is not None:
        if arguments.labels is None or arguments.scored_output is None:
            parser.error("--labels and --scored-output are required for scoring")
        _score_sidecar(arguments.score_predictions, arguments.labels, arguments.scored_output)
        print("[exp7321] private_evaluator=complete", flush=True)
        return 0
    if arguments.check_artifact is not None:
        value = json.loads(arguments.check_artifact.read_text(encoding="utf-8"))
        errors = validate_artifact(value, require_validation=False)
        if arguments.raw_rows and arguments.raw_calls and arguments.cost_receipt:
            errors.extend(
                _check_raw_reduction(
                    value, arguments.raw_rows, arguments.raw_calls, arguments.cost_receipt
                )
            )
        if errors:
            raise ValueError("artifact check failed: " + ",".join(dict.fromkeys(errors)))
        print("[exp7321] artifact_check=pass", flush=True)
        return 0
    artifact = run_experiment(run_date=arguments.date)
    print(
        f"[exp7321] verdict={artifact['honest_verdict']} "
        f"capture={artifact['batch_capture_complete_score']} "
        f"value={artifact['batch_value_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
