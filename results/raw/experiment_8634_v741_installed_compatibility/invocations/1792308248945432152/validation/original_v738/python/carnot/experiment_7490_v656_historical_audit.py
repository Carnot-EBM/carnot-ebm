"""Requalify V655 probability and feedback evidence without rewriting history.

The module reuses the corrected Exp7484 raw reducer. It adds the missing audit
of where each shuffled feedback label came from. No current model is loaded.

Spec refs: REQ-REPORT-7490 and SCENARIO-REPORT-7490-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from typing import Any, TypeVar

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7484_v655_decision_audit import audit_sources
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


Json = dict[str, Any]
T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7490-historical-audit"
SCHEMA = "carnot.exp7490.v656_historical_audit.v1"
RESULT_PATH = Path("results/experiment_7490_v656_historical_audit.json")
RAW_DIR = Path("results/raw/experiment_7490_v656_historical_audit")
MODULE_PATH = Path("python/carnot/experiment_7490_v656_historical_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7490_v656_historical_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7490_v656_historical_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EVALUATOR_PATH = Path("results/raw/experiment_7462_v654_option_protocol/cohort_evaluators.jsonl")
HISTORICAL_ARTIFACTS = (
    Path("results/experiment_7481_v655_typed_calibration.json"),
    Path("results/experiment_7483_v655_continuous_learning.json"),
    Path("results/experiment_7484_v655_decision_audit.json"),
)
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
VALIDITY_PRINCIPLE = "A favorable metric cannot excuse invalid evidence."
READINESS_PRINCIPLE = "A valid scientific null must not block independent measurements."
BENEFIT_PRINCIPLE = "A favorable seed, fixture or low-support result cannot replace held-out value."
REQUIRED_RECEIPTS = (
    *validation_scope.REQUIRED_CHECK_NAMES,
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def resolve_label_polarity(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Resolve label meaning from the pinned annotation policy, not a score."""

    relevant = [
        row
        for row in rows
        if row.get("label_policy") == "one_if_no_human_unsupported_span"
        and row.get("annotation_disposition") in {"supported", "contains_unsupported"}
    ]
    policies = {str(row.get("label_policy")) for row in relevant}
    supported = {
        int(row["label"]) for row in relevant if row["annotation_disposition"] == "supported"
    }
    unsupported = {
        int(row["label"])
        for row in relevant
        if row["annotation_disposition"] == "contains_unsupported"
    }
    valid = (
        bool(relevant)
        and policies == {"one_if_no_human_unsupported_span"}
        and supported == {1}
        and unsupported == {0}
    )
    return {
        "annotation_policy": "one_if_no_human_unsupported_span",
        "evaluator_label_one": "supported",
        "evaluator_label_zero": "contains_unsupported",
        "audit_target_one": "contains_unsupported",
        "audit_target_zero": "supported",
        "transform": "audit_target=1-evaluator_label",
        "supported_row_count": sum(
            row.get("annotation_disposition") == "supported" for row in relevant
        ),
        "unsupported_row_count": sum(
            row.get("annotation_disposition") == "contains_unsupported" for row in relevant
        ),
        "valid": valid,
    }


def recompute_probability_accounting(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Average fit seeds inside each external group before proper-score reduction."""

    external = [
        row for row in rows if row.get("role") == "external" and row.get("failed") is not True
    ]
    expected_groups = sorted({str(row["group_id"]) for row in external})
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in external:
        grouped[(str(row["arm"]), str(row["group_id"]))].append(row)
    arms: Json = {}
    for arm in sorted({key[0] for key in grouped}):
        values: list[tuple[float, int]] = []
        seeds: set[int] = set()
        raw_count = 0
        for group_id in expected_groups:
            group_rows = grouped.get((arm, group_id), [])
            if not group_rows:
                continue
            labels = {int(row["label"]) for row in group_rows}
            if len(labels) != 1:
                raise ValueError(f"probability_label_disagreement:{arm}:{group_id}")
            probabilities = [float(row["probability"]) for row in group_rows]
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
                raise ValueError(f"probability_invalid:{arm}:{group_id}")
            values.append((math.fsum(probabilities) / len(probabilities), labels.pop()))
            seeds.update(int(row["seed"]) for row in group_rows if row.get("seed") is not None)
            raw_count += len(group_rows)
        brier = math.fsum((probability - label) ** 2 for probability, label in values) / len(values)
        log_loss = -math.fsum(
            label * math.log(min(max(probability, 1e-12), 1.0 - 1e-12))
            + (1 - label) * math.log1p(-min(max(probability, 1e-12), 1.0 - 1e-12))
            for probability, label in values
        ) / len(values)
        arms[arm] = {
            "prediction_row_count": raw_count,
            "group_count": len(values),
            "fit_seed_count": len(seeds),
            "fit_seeds": sorted(seeds),
            "coverage": len(values) / len(expected_groups),
            "brier": brier,
            "log_loss": log_loss,
            "class_support": {
                "supported": sum(label == 0 for _, label in values),
                "contains_unsupported": sum(label == 1 for _, label in values),
            },
        }
    return {
        "role": "external",
        "expected_group_count": len(expected_groups),
        "failed_prediction_row_count": sum(
            row.get("role") == "external" and row.get("failed") is True for row in rows
        ),
        "arms": arms,
    }


def completed_probability_rows(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Add terminal unit accounting without changing independent reducer rows."""

    output = [deepcopy(dict(row)) for row in rows]
    for row in output:
        row.update(
            {
                "unit_id": f"external_probability:{row['group_id']}",
                "attempted": True,
                "complete": True,
                "failed": False,
                "censored": False,
                "excluded": False,
                "unstarted": False,
            }
        )
    return output


def reconstruct_shuffled_chronology(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    order_seeds: Sequence[int],
    audit_seeds: Sequence[int],
    delays: Sequence[int],
) -> list[Json]:
    """Reproduce the historical global shuffle and retain every label origin."""

    if len(order_seeds) != len(audit_seeds):
        raise ValueError("seed_count_mismatch")
    output: list[Json] = []
    for order_seed, audit_seed in zip(order_seeds, audit_seeds, strict=True):
        for delay in delays:
            stream = sorted(
                (
                    row
                    for row in ledger_rows
                    if int(row.get("order_seed", -1)) == int(order_seed)
                    and int(row.get("delay", -1)) == int(delay)
                    and row.get("arm") == "shuffled_feedback"
                ),
                key=lambda row: int(row["event_time"]),
            )
            audited = [row for row in stream if row.get("label_revealed") is True]
            origins = [
                {
                    "group_id": str(row["group_id"]),
                    "event_time": int(row["event_time"]),
                    "available_time": int(row["feedback_time"]),
                    "label": int(row["label"]),
                }
                for row in audited
            ]
            np.random.default_rng(int(audit_seed)).shuffle(origins)
            for target, origin in zip(audited, origins, strict=True):
                assignment_time = int(target["feedback_time"])
                output.append(
                    {
                        "construction": "historical_global_pre_replay_shuffle",
                        "order_seed": int(order_seed),
                        "audit_seed": int(audit_seed),
                        "delay": int(delay),
                        "target_group_id": str(target["group_id"]),
                        "target_event_time": int(target["event_time"]),
                        "target_prediction_time": int(target["prediction_time"]),
                        "assignment_time": assignment_time,
                        "assigned_control_label": int(origin["label"]),
                        "origin_group_id": origin["group_id"],
                        "origin_event_time": origin["event_time"],
                        "origin_available_time": origin["available_time"],
                        "origin_available_at_assignment": origin["available_time"]
                        <= assignment_time,
                        "from_future_event": origin["event_time"] > int(target["event_time"]),
                        "update_accepted": target.get("update_accepted") is True,
                    }
                )
    return output


def chronology_preserving_block_permutation(
    stream_rows: Sequence[Mapping[str, Any]], *, block_size: int
) -> list[Json]:
    """Build a causal control from the previous completed block of labels.

    The first block uses each label at its own release time as a warm-up. Later
    blocks draw only from the complete prior block. This tests the chronology
    guard. It is not a replacement efficacy result.
    """

    if block_size <= 0:
        raise ValueError("block_size_must_be_positive")
    ordered = sorted(
        (row for row in stream_rows if row.get("label_revealed") is True),
        key=lambda row: int(row["event_time"]),
    )
    output: list[Json] = []
    for index, target in enumerate(ordered):
        block_start = (index // block_size) * block_size
        if block_start == 0:
            origin = target
        else:
            prior = ordered[max(0, block_start - block_size) : block_start]
            origin = prior[(index - block_start + 1) % len(prior)]
        assignment_time = int(target["feedback_time"])
        origin_available = int(origin["feedback_time"])
        output.append(
            {
                "construction": "chronology_preserving_previous_block_permutation",
                "order_seed": int(target.get("order_seed", 0)),
                "audit_seed": int(target.get("audit_seed", 0)),
                "delay": int(target.get("delay", 0)),
                "target_group_id": str(target["group_id"]),
                "target_event_time": int(target["event_time"]),
                "target_prediction_time": int(target["prediction_time"]),
                "assignment_time": assignment_time,
                "assigned_control_label": int(origin["label"]),
                "origin_group_id": str(origin["group_id"]),
                "origin_event_time": int(origin["event_time"]),
                "origin_available_time": origin_available,
                "origin_available_at_assignment": origin_available <= assignment_time,
                "from_future_event": int(origin["event_time"]) > int(target["event_time"]),
                "update_accepted": target.get("update_accepted") is True,
            }
        )
    return output


def validate_feedback_chronology(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Count control assignments whose origin was unavailable when assigned."""

    missing = sum(
        any(
            field not in row
            for field in (
                "origin_group_id",
                "origin_event_time",
                "origin_available_time",
                "assignment_time",
            )
        )
        for row in rows
    )
    future = sum(
        not bool(row.get("origin_available_at_assignment"))
        or int(row.get("origin_available_time", 1)) > int(row.get("assignment_time", 0))
        or bool(row.get("from_future_event"))
        for row in rows
        if all(
            field in row
            for field in ("origin_event_time", "origin_available_time", "assignment_time")
        )
    )
    return {
        "assignment_count": len(rows),
        "missing_origin_count": missing,
        "future_origin_assignment_count": future,
        "valid": bool(rows) and missing == 0 and future == 0,
    }


def _main_feedback_summary(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Check only the main importance learner's own revealed feedback timing."""

    selected = [
        row
        for row in rows
        if row.get("arm") == "importance_anchor" and row.get("update_accepted") is True
    ]
    violations = sum(
        row.get("label_revealed") is not True
        or row.get("feedback_time") is None
        or int(row["feedback_time"]) < int(row["prediction_time"])
        for row in selected
    )
    return {
        "accepted_update_count": len(selected),
        "future_label_violation_count": violations,
        "source_branch": "importance_anchor_uses_own_source_label_after_release",
    }


def historical_claim_limits(
    *,
    probability_benefit: bool,
    typed_utility: bool,
    passed_cost_cells: int,
    total_cost_cells: int,
    online_benefit: bool,
    future_control_assignments: int,
    main_feedback_violations: int,
) -> Json:
    """Keep utility, probability quality, control validity, and leakage separate."""

    return {
        "typed_utility": {
            "finding": "historical_observed" if typed_utility else "null",
            "scope": "previously_evaluated_diagnostic_history",
            "passed_cost_cells": passed_cost_cells,
            "total_cost_cells": total_cost_cells,
            "fresh_holdout": False,
        },
        "probability_quality": {
            "finding": "positive" if probability_benefit else "null",
            "scope": "external_faithbench_diagnostic_history",
            "fresh_holdout": False,
        },
        "online_learning": {
            "finding": "positive" if online_benefit else "null",
            "registered_thresholds_preserved": True,
        },
        "shuffled_control_validity": {
            "finding": "invalid_negative_control"
            if future_control_assignments
            else "chronology_valid",
            "future_origin_assignment_count": future_control_assignments,
        },
        "main_learner_future_label_use": {
            "finding": "observed" if main_feedback_violations else "not_observed",
            "violation_count": main_feedback_violations,
            "invalid_negative_control_is_not_main_leakage_evidence": True,
        },
        "counterfactual_control_used_for_efficacy": False,
    }


def classify_terminal(
    *,
    inputs_available: bool,
    current_validation_passed: bool,
    reduction_complete: bool,
    scientific_benefit: bool,
) -> Json:
    """Classify external absence, invalid work, and a complete null separately."""

    if not inputs_available:
        return {
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_missing_historical_input",
            "historical_audit_complete_score": 0,
        }
    if not current_validation_passed or not reduction_complete:
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_historical_audit_validation",
            "historical_audit_complete_score": 0,
        }
    return {
        "verdict_class": "positive" if scientific_benefit else "null",
        "honest_verdict": (
            "complete_positive_requalified_historical_evidence"
            if scientific_benefit
            else "complete_null_historical_probability_limit_and_invalid_shuffled_control"
        ),
        "historical_audit_complete_score": 1,
    }


def _gate(check: str, category: str, expected: Any, observed: Any, op: str, passed: bool) -> Json:
    principle = {
        "validity": VALIDITY_PRINCIPLE,
        "readiness": READINESS_PRINCIPLE,
        "benefit": BENEFIT_PRINCIPLE,
    }[category]
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principle,
    }


FIELD_PRINCIPLES = {
    "schema": "Versioned schema, exact experiment identity, milestone and terminal status prevent reader drift.",
    "run_date": "The fixed run date plus measured UTC and monotonic identity prevent replay drift.",
    "preconditions_checked": "Exact paths, values, ownership and input validity prevent guessed prerequisites.",
    "MODEL_SPECS": "An empty current model list prevents historical model evidence from becoming current inference.",
    "model_specs": "The matching lowercase empty list prevents reader-specific model identity drift.",
    "model_invoked": "False distinguishes zero current calls from archived model activity.",
    "invocation_counts": "Balanced zero call states expose any attempted or unfinished current inference.",
    "inference_substrate": "The canonical aggregation name prevents methodology misclassification.",
    "inference_substrate_class": "The aggregation class applies the correct evidence and duration rules.",
    "execution_venue": "Host identity stays separate from archived board or CUDA evidence.",
    "duration_s": "Measured elapsed work prevents a synthetic duration floor.",
    "phase_spans": "Flushed measured boundaries expose unfinished operations and stalls.",
    "random_seed": "Frozen audit, order and interval seeds prevent favorable reruns.",
    "reproducibility_checksum": "The checksum binds code, data roles, raw shards and validation scope.",
    "source_artifact_hashes": "Exact upstream bytes, verdicts and flags prevent historical laundering.",
    "rows": "Per-group rows with completion state permit independent headline reduction.",
    "sample_size_budget": "Separate unit states prevent planned or repeated rows from inflating support.",
    "acceptance_gate_results": "Typed gates keep validity, readiness and benefit independent.",
    "gate_check_summary": "Exact failed operands prevent a blocked or null result from losing its cause.",
    "honest_verdict": "A complete terminal finding distinguishes a valid null from unfinished work.",
    "verdict_class": "The closed enum prevents prose from changing machine classification.",
    "verifier_is_oracle": "False prevents the validation reader from becoming positive scientific evidence.",
    "flagged_adversarial": "Actual current reader failures cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, exits and log hashes establish the required scope.",
    "field_principles": "One purpose per emitted field makes silent semantic drift visible.",
    "historical_audit_complete_score": "Completion measures valid reduction only and authorizes no historical claim.",
    "historical_probability_rows": "Raw external group comparisons keep probability findings reproducible.",
    "feedback_chronology_rows": "Each control label origin and availability time exposes future access.",
    "historical_claim_limits": "Separate utility, probability and control findings prevent claim transfer.",
}


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Give every emitted field a one-line failure-prevention purpose."""

    return {
        key: FIELD_PRINCIPLES.get(
            key,
            "This field preserves measured audit evidence and prevents silent omission.",
        )
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash all terminal content except measured clocks and this hash itself."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "ended_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "phase_spans",
        "process_identity",
        "device_identity",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        len(
            [
                row
                for row in receipts
                if row.get("name") == name
                and row.get("passed") is True
                and int(row.get("exit_code", 1)) == 0
                and row.get("timed_out") is not True
            ]
        )
        == 1
        for name in REQUIRED_RECEIPTS
    )


def _build_gates(value: Mapping[str, Any]) -> list[Json]:
    reduction_errors = list(value.get("reduction_errors") or [])
    claims = value.get("historical_claim_limits") or {}
    chronology = value.get("feedback_chronology_summary") or {}
    inputs = bool(value.get("preconditions_checked")) and all(
        row.get("passed") is True for row in value.get("preconditions_checked") or []
    )
    validation = _required_receipts_pass(value.get("validation_receipts") or [])
    complete = int(value.get("historical_audit_complete_score", 0))
    return [
        _gate("historical_inputs_authentic", "validity", True, inputs, "==", inputs),
        _gate(
            "independent_reduction_errors",
            "validity",
            [],
            reduction_errors,
            "==",
            not reduction_errors,
        ),
        _gate(
            "canonical_current_provenance",
            "validity",
            "aggregation_from_upstream_artifacts",
            value.get("inference_substrate"),
            "==",
            value.get("inference_substrate") == "aggregation_from_upstream_artifacts",
        ),
        _gate("required_current_validation", "validity", True, validation, "==", validation),
        _gate("historical_audit_complete", "readiness", 1, complete, "==", complete == 1),
        _gate(
            "historical_probability_benefit",
            "benefit",
            "positive",
            claims.get("probability_quality", {}).get("finding"),
            "==",
            claims.get("probability_quality", {}).get("finding") == "positive",
        ),
        _gate(
            "historical_online_benefit",
            "benefit",
            "positive",
            claims.get("online_learning", {}).get("finding"),
            "==",
            claims.get("online_learning", {}).get("finding") == "positive",
        ),
        _gate(
            "shuffled_negative_control_chronology",
            "benefit",
            0,
            chronology.get("future_origin_assignment_count"),
            "==",
            chronology.get("future_origin_assignment_count") == 0,
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    failed = [
        {
            "check": row["check"],
            "category": row["category"],
            "upstream": "current_historical_audit",
            "field_path": row["check"],
            "expected": row["expected"],
            "observed": row["observed"],
            "op": row["op"],
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "all_validity_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "validity"
        ),
        "all_readiness_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "readiness"
        ),
        "all_benefit_passed": all(
            row.get("passed") is True for row in gates if row.get("category") == "benefit"
        ),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def _base_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    reduction: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
) -> Json:
    inputs_available = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    validation = _required_receipts_pass(receipts)
    reduction_complete = (
        not reduction.get("reduction_errors")
        and bool(reduction.get("historical_probability_rows"))
        and bool(reduction.get("feedback_chronology_rows"))
    )
    claims = reduction.get("historical_claim_limits") or {}
    scientific_benefit = bool(
        claims.get("probability_quality", {}).get("finding") == "positive"
        and claims.get("online_learning", {}).get("finding") == "positive"
        and claims.get("shuffled_control_validity", {}).get("finding") == "chronology_valid"
        and claims.get("typed_utility", {}).get("fresh_holdout") is True
    )
    terminal = classify_terminal(
        inputs_available=inputs_available,
        current_validation_passed=validation,
        reduction_complete=reduction_complete,
        scientific_benefit=scientific_benefit,
    )
    probability_rows = completed_probability_rows(
        reduction.get("historical_probability_rows") or []
    )
    feedback_rows = reduction.get("feedback_chronology_rows") or []
    duration = (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000
    value: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": terminal["honest_verdict"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "clock_identity": {"utc": "datetime.now(UTC)", "monotonic": "time.monotonic_ns"},
        "process_identity": {"pid": os.getpid(), "ppid": os.getppid(), "python": sys.executable},
        "device_identity": {
            "venue": "host",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cuda_used": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration,
        "duration_components_s": {
            "model_load_s": 0.0,
            "forward_s": 0.0,
            "generation_s": 0.0,
            "optimization_s": 0.0,
            "aggregation_s": max(
                0.0, duration - math.fsum(float(row.get("duration_s", 0.0)) for row in receipts)
            ),
            "validation_s": math.fsum(float(row.get("duration_s", 0.0)) for row in receipts),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "role": "pinned_upstream_roles",
            "optimizer": None,
            "audit": list(reduction.get("audit_seeds") or []),
            "order": list(reduction.get("order_seeds") or []),
            "interval": [6_551_481, 6_551_581, 6_551_981, 7_483_401, 7_483_402],
        },
        "source_artifact_hashes": deepcopy(list(reduction.get("source_artifact_hashes") or [])),
        "rows": probability_rows,
        "sample_size_budget": {
            "planned": len(probability_rows),
            "attempted": len(probability_rows),
            "complete": len(probability_rows),
            "failed": 0,
            "excluded": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "external_source_group",
            "feedback_control_assignments": len(feedback_rows),
            "feedback_assignments_are_repeated_protocol_units": True,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "field_principles": {},
        "historical_audit_complete_score": terminal["historical_audit_complete_score"],
        "historical_probability_rows": probability_rows,
        "feedback_chronology_rows": deepcopy(list(feedback_rows)),
        "historical_probability_summary": deepcopy(
            dict(reduction.get("historical_probability_summary") or {})
        ),
        "feedback_chronology_summary": deepcopy(
            dict(reduction.get("feedback_chronology_summary") or {})
        ),
        "chronology_preserving_control": deepcopy(
            dict(reduction.get("chronology_preserving_control") or {})
        ),
        "main_feedback_summary": deepcopy(dict(reduction.get("main_feedback_summary") or {})),
        "historical_claim_limits": deepcopy(dict(claims)),
        "historical_source_states": deepcopy(list(reduction.get("historical_source_states") or [])),
        "original_exp7484_reader_output": deepcopy(
            dict(reduction.get("original_exp7484_reader_output") or {})
        ),
        "label_semantics": deepcopy(dict(reduction.get("label_semantics") or {})),
        "reduction_errors": deepcopy(list(reduction.get("reduction_errors") or [])),
        "small_ebm_training": {"performed": False, "current_llm_calls": 0},
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": validation,
            "numbered_runtime_e2e": [],
            "reason": "Pure reporting changed no shared sampler, training, binding, ARC, telemetry, or Rust code.",
        },
        "external_publication_authorized": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "reproducibility_checksum": "",
    }
    value["acceptance_gate_results"] = _build_gates(value)
    value["gate_check_summary"] = _gate_summary(value["acceptance_gate_results"])
    value["field_principles"] = _field_principles(tuple(value))
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def fixture_artifact() -> Json:
    """Build a small contract-complete null for mutation tests."""

    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": f"fixture:{name}",
            "log_sha256": canonical_hash(name),
            "duration_s": 0.0,
        }
        for name in REQUIRED_RECEIPTS
    ]
    probability_rows = [
        {
            "row_kind": "static_group",
            "role": "external",
            "group_id": "fixture-group",
            "label": 1,
            "probabilities": {"gibbs": 0.8, "temperature": 0.7, "logistic": 0.75},
            "brier_losses": {"gibbs": 0.04, "temperature": 0.09, "logistic": 0.0625},
            "log_losses": {
                "gibbs": -math.log(0.8),
                "temperature": -math.log(0.7),
                "logistic": -math.log(0.75),
            },
        }
    ]
    feedback = [
        {
            "construction": "historical_global_pre_replay_shuffle",
            "target_group_id": "g0",
            "target_event_time": 0,
            "target_prediction_time": 0,
            "assignment_time": 0,
            "origin_group_id": "g1",
            "origin_event_time": 1,
            "origin_available_time": 1,
            "origin_available_at_assignment": False,
            "from_future_event": True,
        }
    ]
    claims = historical_claim_limits(
        probability_benefit=False,
        typed_utility=True,
        passed_cost_cells=1,
        total_cost_cells=9,
        online_benefit=False,
        future_control_assignments=1,
        main_feedback_violations=0,
    )
    now = time.monotonic_ns()
    return _base_artifact(
        preconditions=[{"check": "fixture", "passed": True}],
        reduction={
            "source_artifact_hashes": [],
            "historical_probability_rows": probability_rows,
            "feedback_chronology_rows": feedback,
            "historical_probability_summary": {"cost_cells": [{} for _ in range(9)]},
            "feedback_chronology_summary": validate_feedback_chronology(feedback),
            "chronology_preserving_control": {"valid": True},
            "main_feedback_summary": {"future_label_violation_count": 0},
            "historical_claim_limits": claims,
            "historical_source_states": [],
            "original_exp7484_reader_output": {"exit_code": 1, "flag": "METHODOLOGY_MISSING"},
            "label_semantics": {"valid": True},
            "reduction_errors": [],
            "audit_seeds": [7],
            "order_seeds": [1],
        },
        receipts=receipts,
        started_at_utc="2026-09-21T00:00:00+00:00",
        started_monotonic_ns=now,
        ended_monotonic_ns=now + 1,
        phase_spans=[],
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_files: bool = True
) -> list[str]:  # pragma: no cover - exercised by fresh-process readers and contract tests.
    """Cold-check identity, provenance, rows, gates, source bytes, and checksum."""

    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "experiment_id",
        "milestone",
        "status",
        "model_specs",
    }
    errors.extend(f"missing_field:{name}" for name in sorted(required - value.keys()))
    if (
        value.get("schema"),
        value.get("experiment_id"),
        value.get("milestone"),
        value.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or value.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or value.get("inference_substrate_class") != "aggregation"
        or value.get("execution_venue") != "host"
    ):
        errors.append("current_provenance_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_invalid")
    principles = value.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(value)
        or any(not isinstance(item, str) or not item.strip() for item in principles.values())
    ):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results") or []
    if not gates or any(
        set(("check", "category", "expected", "observed", "op", "passed", "principle")) - row.keys()
        for row in gates
    ):
        errors.append("gate_contract_invalid")
    if value.get("verdict_class") == "blocked":
        if value.get("historical_audit_complete_score") != 0:
            errors.append("blocked_complete_score_nonzero")
    elif (
        value.get("historical_audit_complete_score") != 1
        or not value.get("historical_probability_rows")
        or not value.get("feedback_chronology_rows")
        or value.get("rows") != value.get("historical_probability_rows")
        or not _required_receipts_pass(value.get("validation_receipts") or [])
    ):
        errors.append("historical_audit_incomplete")
    if verify_files:
        for row in value.get("source_artifact_hashes") or []:
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_invalid:{path}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _load_object(path: Path) -> Json:  # pragma: no cover - runtime I/O boundary.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[Json]:  # pragma: no cover - runtime I/O boundary.
    rows: list[Json] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}")
            rows.append(value)
    return rows


def _source_row(
    root: Path, path: Path, *, evidence_class: str, artifact: Mapping[str, Any] | None = None
) -> Json:  # pragma: no cover - runtime byte authentication.
    resolved = path if path.is_absolute() else root / path
    return {
        "path": str(path) if path.is_absolute() else path.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "evidence_class": evidence_class,
        "original_verdict_class": (artifact or {}).get("verdict_class"),
        "original_honest_verdict": (artifact or {}).get("honest_verdict"),
        "original_flagged_adversarial": (artifact or {}).get("flagged_adversarial"),
    }


def collect_preconditions(root: Path) -> tuple[list[Json], dict[str, Json]]:  # pragma: no cover
    """Authenticate exact historical bytes and the original failed reader."""

    artifacts = {path.stem: _load_object(root / path) for path in HISTORICAL_ARTIFACTS}
    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7484_v655_decision_audit.py"),
        Path("python/carnot/experiment_7481_v655_typed_calibration.py"),
        Path("python/carnot/experiment_7483_v655_continuous_learning.py"),
        *HISTORICAL_ARTIFACTS,
        EVALUATOR_PATH,
        SPEC_PATH,
    )
    checks: list[Json] = []
    hashes: dict[str, Json] = {}
    for relative in paths:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "path": relative.as_posix(),
                "ownership": "repository_or_upstream_immutable",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "op": "==",
                "passed": available,
                "owner_uid": path.stat().st_uid if available else None,
                "size_bytes": path.stat().st_size if available else None,
            }
        )
        if available:
            artifact = artifacts.get(relative.stem)
            hashes[relative.as_posix()] = _source_row(
                root,
                relative,
                evidence_class="historical_terminal_artifact"
                if artifact
                else "instruction_code_or_raw_input",
                artifact=artifact,
            )
    expected = {
        "experiment_7481_v655_typed_calibration": ("exp7481-typed-calibration", "positive", False),
        "experiment_7483_v655_continuous_learning": ("exp7483-continuous-learning", "null", False),
        "experiment_7484_v655_decision_audit": ("exp7484-decision-audit", "null", False),
    }
    for stem, (experiment_id, verdict, flagged) in expected.items():
        artifact = artifacts[stem]
        for field, wanted in (
            ("experiment_id", experiment_id),
            ("milestone", "2026.09.655"),
            ("verdict_class", verdict),
            ("flagged_adversarial", flagged),
        ):
            checks.append(
                {
                    "check": f"historical_identity:{stem}:{field}",
                    "path": f"results/{stem}.json",
                    "ownership": "upstream_immutable",
                    "expected": wanted,
                    "observed": artifact.get(field),
                    "op": "==",
                    "passed": artifact.get(field) == wanted,
                }
            )
    old_audit = artifacts["experiment_7484_v655_decision_audit"]
    reader = next(
        (
            row
            for row in old_audit.get("validation_receipts") or []
            if row.get("name") == "adversarial_verify"
        ),
        {},
    )
    checks.append(
        {
            "check": "original_exp7484_methodology_failure",
            "path": HISTORICAL_ARTIFACTS[2].as_posix(),
            "ownership": "upstream_immutable",
            "expected": {"exit_code": 1, "flag": "METHODOLOGY_MISSING"},
            "observed": {
                "exit_code": reader.get("exit_code"),
                "flag_present": "METHODOLOGY_MISSING" in str(reader.get("output_tail") or ""),
            },
            "op": "matches",
            "passed": reader.get("exit_code") == 1
            and "METHODOLOGY_MISSING" in str(reader.get("output_tail") or ""),
        }
    )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        {
            "check": "driving_requirement",
            "path": SPEC_PATH.as_posix(),
            "ownership": "current_spec",
            "expected": "REQ-REPORT-7490",
            "observed": "REQ-REPORT-7490" if "REQ-REPORT-7490" in spec_text else None,
            "op": "==",
            "passed": "REQ-REPORT-7490" in spec_text,
        }
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7490" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        {
            "check": "current_task_not_excluded",
            "path": "ops/exclusion_manifest.yaml",
            "ownership": "repository_instruction",
            "expected": False,
            "observed": excluded,
            "op": "==",
            "passed": not excluded,
        }
    )
    return checks, hashes


def reduce_historical_sources(
    root: Path, initial_hashes: Mapping[str, Json]
) -> Json:  # pragma: no cover
    """Run corrected raw reduction and add probability and chronology accounting."""

    corrected = audit_sources(root)
    typed = _load_object(root / HISTORICAL_ARTIFACTS[0])
    learning = _load_object(root / HISTORICAL_ARTIFACTS[1])
    old_audit = _load_object(root / HISTORICAL_ARTIFACTS[2])
    prediction_ref = typed.get("prediction_row_shard") or {}
    prediction_path = root / str(prediction_ref.get("path") or "")
    ledger_ref = learning.get("feedback_ledger") or {}
    ledger_path = root / str(ledger_ref.get("path") or "")
    prediction_rows = _load_jsonl(prediction_path)
    ledger_rows = _load_jsonl(ledger_path)
    evaluator_rows = _load_jsonl(root / EVALUATOR_PATH)
    accounting = recompute_probability_accounting(prediction_rows)
    semantics = resolve_label_polarity(evaluator_rows)
    evaluator = {str(row["group_id"]): int(row["label"]) for row in evaluator_rows}
    polarity_mismatches = 0
    for row in prediction_rows:
        group_id = str(row["group_id"])
        if row.get("role") == "external" and group_id in evaluator:
            polarity_mismatches += int(int(row["label"]) != 1 - evaluator[group_id])
    protocol = learning.get("protocol") or {}
    chronology_rows = reconstruct_shuffled_chronology(
        ledger_rows,
        order_seeds=[int(seed) for seed in protocol.get("order_seeds") or []],
        audit_seeds=[int(seed) for seed in protocol.get("audit_seeds") or []],
        delays=[int(delay) for delay in protocol.get("delays") or []],
    )
    chronology = validate_feedback_chronology(chronology_rows)
    causal_summaries = []
    for order_seed in protocol.get("order_seeds") or []:
        for delay in protocol.get("delays") or []:
            stream = [
                row
                for row in ledger_rows
                if row.get("arm") == "shuffled_feedback"
                and int(row["order_seed"]) == int(order_seed)
                and int(row["delay"]) == int(delay)
            ]
            causal_summaries.append(
                validate_feedback_chronology(
                    chronology_preserving_block_permutation(stream, block_size=8)
                )
            )
    causal = {
        "scheme": "chronology_preserving_previous_block_permutation",
        "stream_count": len(causal_summaries),
        "assignment_count": sum(row["assignment_count"] for row in causal_summaries),
        "future_origin_assignment_count": sum(
            row["future_origin_assignment_count"] for row in causal_summaries
        ),
        "valid": bool(causal_summaries) and all(row["valid"] for row in causal_summaries),
        "used_for_efficacy": False,
    }
    main_feedback = _main_feedback_summary(ledger_rows)
    static = corrected.get("static_summary") or {}
    online = corrected.get("online_summary") or {}
    cost_cells = list((static.get("decision_cost_grid") or {}).get("cells") or [])
    probability_benefit = bool(
        (static.get("probability_comparisons") or {}).get("probability_benefit_passed")
    )
    typed_utility = bool((static.get("decision_cost_grid") or {}).get("decision_benefit_passed"))
    online_benefit = bool(online.get("scientific_benefit_passed"))
    claims = historical_claim_limits(
        probability_benefit=probability_benefit,
        typed_utility=typed_utility,
        passed_cost_cells=sum(row.get("benefit_passed") is True for row in cost_cells),
        total_cost_cells=len(cost_cells),
        online_benefit=online_benefit,
        future_control_assignments=int(chronology["future_origin_assignment_count"]),
        main_feedback_violations=int(main_feedback["future_label_violation_count"]),
    )
    external_rows = [
        deepcopy(dict(row))
        for row in corrected.get("independent_metric_rows") or []
        if row.get("row_kind") == "static_group" and row.get("role") == "external"
    ]
    errors = list(corrected.get("errors") or [])
    if not semantics["valid"] or polarity_mismatches:
        errors.append(f"label_polarity_invalid:{polarity_mismatches}")
    if len(external_rows) != accounting.get("expected_group_count"):
        errors.append("external_group_count_mismatch")
    if len(cost_cells) != 9:
        errors.append("cost_cell_count_mismatch")
    if not causal["valid"]:
        errors.append("chronology_preserving_control_invalid")
    for arm in ("gibbs", "logistic", "temperature"):
        raw = accounting.get("arms", {}).get(arm, {})
        independent = (static.get("proper_scores") or {}).get("external", {}).get(arm, {})
        if (
            not math.isclose(
                float(raw.get("brier", math.nan)),
                float(independent.get("brier", math.nan)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(raw.get("log_loss", math.nan)),
                float(independent.get("log_loss", math.nan)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or float(raw.get("coverage", 0.0)) != 1.0
        ):
            errors.append(f"probability_reduction_mismatch:{arm}")
    old_reader = next(
        (
            row
            for row in old_audit.get("validation_receipts") or []
            if row.get("name") == "adversarial_verify"
        ),
        {},
    )
    hashes = dict(initial_hashes)
    for row in corrected.get("source_artifact_hashes") or []:
        hashes.setdefault(str(row["path"]), deepcopy(dict(row)))
    for path, evidence_class in (
        (Path(str(prediction_ref["path"])), "historical_probability_rows"),
        (Path(str(ledger_ref["path"])), "historical_feedback_ledger"),
    ):
        hashes[path.as_posix()] = _source_row(root, path, evidence_class=evidence_class)
    states = [
        {
            "path": path.as_posix(),
            "experiment_id": artifact.get("experiment_id"),
            "original_honest_verdict": artifact.get("honest_verdict"),
            "original_verdict_class": artifact.get("verdict_class"),
            "original_flagged_adversarial": artifact.get("flagged_adversarial"),
            "original_inference_substrate": artifact.get("inference_substrate"),
        }
        for path, artifact in (
            (HISTORICAL_ARTIFACTS[0], typed),
            (HISTORICAL_ARTIFACTS[1], learning),
            (HISTORICAL_ARTIFACTS[2], old_audit),
        )
    ]
    return {
        "source_artifact_hashes": sorted(hashes.values(), key=lambda row: str(row["path"])),
        "historical_source_states": states,
        "original_exp7484_reader_output": {
            "name": old_reader.get("name"),
            "exit_code": old_reader.get("exit_code"),
            "passed": old_reader.get("passed"),
            "output_tail": old_reader.get("output_tail"),
            "historical_only": True,
        },
        "label_semantics": {**semantics, "prediction_label_mismatch_count": polarity_mismatches},
        "historical_probability_rows": external_rows,
        "historical_probability_summary": {
            "corpus": "faithbench",
            "evaluation_status": "previously_evaluated_diagnostic_history",
            "fresh_holdout": False,
            "accounting": accounting,
            "proper_scores": (static.get("proper_scores") or {}).get("external", {}),
            "probability_comparisons": static.get("probability_comparisons"),
            "decision_cost_grid": static.get("decision_cost_grid"),
        },
        "feedback_chronology_rows": chronology_rows,
        "feedback_chronology_summary": chronology,
        "chronology_preserving_control": causal,
        "main_feedback_summary": main_feedback,
        "historical_claim_limits": claims,
        "reduction_errors": sorted(set(errors)),
        "audit_seeds": list(protocol.get("audit_seeds") or []),
        "order_seeds": list(protocol.get("order_seeds") or []),
    }


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:  # pragma: no cover
    """Re-read all historical rows and compare each stable reduction field."""

    value = _load_object(path)
    if not value:
        return ["candidate_unreadable"]
    preconditions, hashes = collect_preconditions(root)
    current = reduce_historical_sources(root, hashes)
    errors = validate_artifact(value, root=root)
    for field in (
        "source_artifact_hashes",
        "historical_source_states",
        "original_exp7484_reader_output",
        "label_semantics",
        "historical_probability_rows",
        "historical_probability_summary",
        "feedback_chronology_rows",
        "feedback_chronology_summary",
        "chronology_preserving_control",
        "main_feedback_summary",
        "historical_claim_limits",
        "reduction_errors",
    ):
        current_value = current.get(field)
        if field == "historical_probability_rows":
            current_value = completed_probability_rows(current_value or [])
        if value.get(field) != current_value:
            errors.append(f"independent_replay_mismatch:{field}")
    if value.get("preconditions_checked") != preconditions:
        errors.append("independent_replay_mismatch:preconditions_checked")
    return sorted(set(errors))


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print one flushed phase boundary with measured monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7490] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _with_heartbeat(
    operation: str, fn: Callable[[], T], *, started: float, heartbeat_s: float = 60.0
) -> T:  # pragma: no cover
    """Emit truthful pending messages while one in-process reduction runs."""

    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(heartbeat_s):
            progress(started, operation, "pending")

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _span(
    phase: str, phase_started: int, run_started: int, completed_units: int
) -> Json:  # pragma: no cover
    ended = time.monotonic_ns()
    return {
        "phase": phase,
        "start_s": (phase_started - run_started) / 1e9,
        "end_s": (ended - run_started) / 1e9,
        "duration_s": (ended - phase_started) / 1e9,
        "completed_units": completed_units,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = ".venv/bin/python"
    common = ("--date", RUN_DATE, "--root", ".")
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "capability_end_to_end",
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "raw_historical_rows",
            timeout_s=1_200.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> Json:  # pragma: no cover - declared capability E2E.
    """Authenticate, reduce, validate in fresh processes, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(UTC).isoformat()
    spans: list[Json] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic_ns()
    preconditions, hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started_ns, len(preconditions)))
    failed = [row for row in preconditions if row.get("passed") is not True]
    progress(started, "preconditions", "complete", completed=len(preconditions), failed=len(failed))
    if failed:
        raise RuntimeError(f"blocked_missing_historical_input:{failed[0]}")

    for phase in ("model_load", "generation", "optimization"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic_ns()
        spans.append(_span(phase, phase_started, started_ns, 0))
        progress(started, phase, "after", completed=0)

    progress(started, "historical_reduction", "before_benchmark")
    phase_started = time.monotonic_ns()
    reduction = _with_heartbeat(
        "historical_reduction",
        lambda: reduce_historical_sources(root, hashes),
        started=started,
    )
    spans.append(
        _span(
            "historical_reduction",
            phase_started,
            started_ns,
            len(reduction["historical_probability_rows"])
            + len(reduction["feedback_chronology_rows"]),
        )
    )
    progress(
        started,
        "historical_reduction",
        "after_benchmark",
        probability_groups=len(reduction["historical_probability_rows"]),
        chronology_rows=len(reduction["feedback_chronology_rows"]),
    )
    if reduction["reduction_errors"]:
        raise RuntimeError(f"historical_reduction_failed:{reduction['reduction_errors']}")

    private_root = Path(tempfile.mkdtemp(prefix="exp7490-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic_ns()
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started_ns, len(affected)))
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    if not affected_reduction["passed"]:
        raise RuntimeError(f"affected_validation_failed:{affected_reduction}")

    provisional = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "provisional_for_candidate_reader": True,
        }
        for name in REQUIRED_RECEIPTS
        if name not in validation_scope.REQUIRED_CHECK_NAMES
    ]
    candidate = _base_artifact(
        preconditions=preconditions,
        reduction=reduction,
        receipts=[*affected, *provisional],
        started_at_utc=started_utc,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_atomic_write", path=candidate_path)
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_atomic_write", path=candidate_path)

    progress(started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic_ns()
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started_ns, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("terminal_validation_failed")

    final = _base_artifact(
        preconditions=preconditions,
        reduction=reduction,
        receipts=[*affected, *terminal],
        started_at_utc=started_utc,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(started, "publish", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed run date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the audit or one exact fresh-process terminal reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = (
            validate_artifact(value, root=root, verify_files=not args.no_source_check)
            if value
            else ["candidate_unreadable"]
        )
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce, root=root)
        print(json.dumps({"errors": errors, "passed": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
