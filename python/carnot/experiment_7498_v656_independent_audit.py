"""Independently audit V656 window and causal-learning evidence.

The reducer reads roadmap-selected artifacts and immutable rows. It never uses
an upstream verdict as a numeric operand and never loads or fits a model.

Spec refs: REQ-REPORT-7498 and SCENARIO-REPORT-7498-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

import yaml

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


Json = dict[str, Any]
T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7498-independent-audit"
SCHEMA = "carnot.exp7498.v656_independent_audit.v1"
RESULT_PATH = Path("results/experiment_7498_v656_independent_audit.json")
RAW_DIR = Path("results/raw/experiment_7498_v656_independent_audit")
MODULE_PATH = Path("python/carnot/experiment_7498_v656_independent_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7498_v656_independent_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7498_v656_independent_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
OPTION_IDS = {"supported", "contains_unsupported"}
VALIDITY_PRINCIPLE = "A favorable metric cannot excuse invalid evidence."
READINESS_PRINCIPLE = "A valid scientific null must not block independent measurements."
BENEFIT_PRINCIPLE = "A favorable seed, fixture or low-support result cannot replace held-out value."
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
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


def resolve_roadmap_producers(roadmap: Mapping[str, Any]) -> list[Json]:
    """Read exact producer paths from the matching roadmap task sequence."""

    tasks = roadmap.get("tasks")
    selected: list[Json] = []
    if roadmap.get("milestone") == MILESTONE and isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            task_id = str(task.get("id") or "")
            for number in range(7491, 7498):
                if task_id.startswith(f"exp{number}-"):
                    selected.append(
                        {
                            "number": number,
                            "task_id": task_id,
                            "deliverable": str(task.get("deliverable") or ""),
                        }
                    )
                    break
    if [row["number"] for row in selected] != list(range(7491, 7498)) or any(
        not row["deliverable"].startswith("results/") for row in selected
    ):
        raise ValueError("roadmap_producer_sequence_invalid")
    return selected


def reduce_protocol_rows(
    *,
    groups: Sequence[Mapping[str, Any]],
    predictors: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
) -> Json:
    """Rebuild role, response, option-order, and label-free request boundaries."""

    roles: dict[str, set[str]] = defaultdict(set)
    for collection in (groups, predictors, evaluators):
        for row in collection:
            roles[str(row.get("group_id"))].add(str(row.get("role")))
    role_disjoint = bool(roles) and all(len(values) == 1 for values in roles.values())
    predictor_index = {str(row.get("group_id")): row for row in predictors}
    windows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in windows:
        windows_by_group[str(row.get("group_id"))].append(row)
    coverage_errors: list[str] = []
    for group_id, predictor in predictor_index.items():
        encoded = str(predictor.get("response_text") or "").encode()
        ordered = sorted(
            windows_by_group.get(group_id, []), key=lambda row: int(row.get("window_index", -1))
        )
        cursor = 0
        for index, row in enumerate(ordered):
            start = int(row.get("byte_start", -1))
            end = int(row.get("byte_end", -1))
            if int(row.get("window_index", -1)) != index or start != cursor or end <= start:
                coverage_errors.append(group_id)
                break
            cursor = end
        if not ordered or cursor != len(encoded):
            coverage_errors.append(group_id)
    option_orders = {tuple(str(item) for item in row.get("option_order") or []) for row in requests}
    option_mapping_valid = bool(requests) and option_orders == {
        ("supported", "contains_unsupported"),
        ("contains_unsupported", "supported"),
    }
    frozen_label_boundary = bool(requests) and all(
        row.get("gold_label") is None for row in requests
    )
    eligible = [row for row in groups if row.get("eligible") is True]
    counts = Counter(str(row.get("role")) for row in eligible)
    unit_rows = [
        {
            "unit_id": str(row.get("group_id")),
            "group_id": str(row.get("group_id")),
            "role": str(row.get("role")),
            "eligible": row.get("eligible") is True,
        }
        for row in groups
    ]
    passed = (
        role_disjoint
        and not coverage_errors
        and option_mapping_valid
        and frozen_label_boundary
        and set(predictor_index) == {str(row.get("group_id")) for row in groups}
    )
    return {
        "passed": passed,
        "role_disjoint": role_disjoint,
        "complete_response_coverage": not coverage_errors,
        "coverage_error_groups": sorted(set(coverage_errors)),
        "option_mapping_valid": option_mapping_valid,
        "frozen_label_boundary": frozen_label_boundary,
        "group_counts": dict(sorted(counts.items())),
        "request_count": len(requests),
        "evaluator_count": len(evaluators),
        "unit_rows": unit_rows,
    }


def reduce_capture_rows(
    plan: Sequence[Mapping[str, Any]], observed_rows: Sequence[Mapping[str, Any]]
) -> Json:
    """Reconcile native rows by request ID and remap display labels to stable options."""

    expected = {str(row.get("request_id")): row for row in plan}
    observed = {str(row.get("request_id")): row for row in observed_rows}
    duplicate_rows = len(observed) != len(observed_rows)
    complete_roster = not duplicate_rows and set(observed) == set(expected)
    option_mapping_valid = True
    complete_calls = 0
    for request_id, planned in expected.items():
        row = observed.get(request_id)
        if row is None:
            option_mapping_valid = False
            continue
        order = tuple(str(item) for item in planned.get("option_order") or [])
        mapping = {" A": order[0], " B": order[1]} if len(order) == 2 else {}
        logits = row.get("raw_logits_by_option_id")
        identity_valid = all(
            row.get(field) == planned.get(field)
            for field in ("group_id", "role", "arm", "option_order", "eligible")
        )
        if planned.get("eligible") is False:
            valid = row.get("disposition") == "excluded" and identity_valid
        else:
            valid = (
                row.get("disposition") == "complete"
                and row.get("label_to_option_id") == mapping
                and isinstance(logits, Mapping)
                and set(logits) == OPTION_IDS
                and all(math.isfinite(float(value)) for value in logits.values())
                and identity_valid
            )
        option_mapping_valid &= valid
        complete_calls += int(valid and planned.get("eligible") is not False)
    role_groups: dict[str, set[str]] = defaultdict(set)
    for row in plan:
        role_groups[str(row.get("role"))].add(str(row.get("group_id")))
    unit_rows: list[Json] = []
    for role, group_ids in sorted(role_groups.items()):
        for group_id in sorted(group_ids):
            planned = [row for row in plan if str(row.get("group_id")) == group_id]
            eligible = any(row.get("eligible") is True for row in planned)
            calls = [observed.get(str(row.get("request_id"))) for row in planned]
            complete = eligible and all(
                row is not None and row.get("disposition") == "complete" for row in calls
            )
            excluded = not eligible and all(
                row is not None and row.get("disposition") == "excluded" for row in calls
            )
            unit_rows.append(
                {
                    "unit_id": group_id,
                    "group_id": group_id,
                    "role": role,
                    "attempted": complete,
                    "complete": complete,
                    "failed": eligible and not complete,
                    "excluded": excluded,
                    "censored": False,
                    "unstarted": eligible and not complete and all(row is None for row in calls),
                }
            )
    passed = complete_roster and option_mapping_valid
    return {
        "passed": passed,
        "complete_roster": complete_roster,
        "duplicate_rows": duplicate_rows,
        "option_mapping_valid": option_mapping_valid,
        "planned_calls": len(plan),
        "complete_calls": complete_calls,
        "group_counts": {role: len(groups) for role, groups in sorted(role_groups.items())},
        "unit_rows": unit_rows,
    }


def replay_feedback_rows(events: Sequence[Mapping[str, Any]]) -> Json:
    """Replay label availability and adjacent state hashes without a summarizer."""

    ordered = sorted(events, key=lambda row: int(row.get("prediction_time", -1)))
    future = 0
    state_errors = 0
    changed = 0
    for index, row in enumerate(ordered):
        future += int(int(row.get("label_available_time", 1)) > int(row.get("feedback_time", 0)))
        changed += int(row.get("actual_label") != row.get("shuffled_label"))
        if index + 1 < len(ordered):
            state_errors += int(
                row.get("state_after_hash") != ordered[index + 1].get("state_before_hash")
            )
    return {
        "passed": bool(ordered) and future == 0 and state_errors == 0,
        "event_count": len(ordered),
        "future_label_count": future,
        "state_transition_error_count": state_errors,
        "class_changing_permutation_count": changed,
        "class_changing_permutation_supported": changed > 0,
    }


def make_checkpoint_receipt(payload: Mapping[str, Any]) -> Json:
    """Bind a terminal state payload so a later mutation is visible."""

    copied = deepcopy(dict(payload))
    return {"payload": copied, "payload_sha256": canonical_hash(copied)}


def verify_checkpoint_receipt(receipt: Mapping[str, Any]) -> bool:
    """Recompute a private terminal checkpoint identity."""

    payload = receipt.get("payload")
    return isinstance(payload, Mapping) and receipt.get("payload_sha256") == canonical_hash(payload)


def run_attack_controls() -> Json:
    """Prove that future-label and terminal-state mutations are rejected."""

    events = [
        {
            "prediction_time": 0,
            "feedback_time": 1,
            "label_available_time": 1,
            "state_before_hash": "s0",
            "state_after_hash": "s1",
            "actual_label": 0,
            "shuffled_label": 1,
        },
        {
            "prediction_time": 2,
            "feedback_time": 3,
            "label_available_time": 3,
            "state_before_hash": "s1",
            "state_after_hash": "s2",
            "actual_label": 1,
            "shuffled_label": 0,
        },
    ]
    future = deepcopy(events)
    future[0]["label_available_time"] = 2
    checkpoint = make_checkpoint_receipt({"terminal_state_hash": "s2"})
    mutated = deepcopy(checkpoint)
    mutated["payload"]["terminal_state_hash"] = "changed"
    output = {
        "future_label_mutation_detected": replay_feedback_rows(future)["passed"] is False,
        "terminal_checkpoint_mutation_detected": verify_checkpoint_receipt(mutated) is False,
    }
    return {**output, "passed": all(output.values())}


def classify_terminal(dispositions: Sequence[str], reduction_errors: Sequence[str]) -> Json:
    """Keep external absence, invalid evidence, and completed nulls distinct."""

    invalid = bool(reduction_errors) or any(
        state in {"invalid", "disqualified", "flagged"} for state in dispositions
    )
    missing = any(state in {"absent", "pre_gate", "blocked"} for state in dispositions)
    if invalid:
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_v656_independent_science_audit",
            "science_audit_complete_score": 0,
        }
    if missing:
        return {
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_missing_v656_scientific_inputs",
            "science_audit_complete_score": 1,
        }
    return {
        "verdict_class": "null",
        "honest_verdict": "complete_null_v656_independent_science_audit",
        "science_audit_complete_score": 1,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    *,
    upstream: str,
    field_path: str,
) -> Json:
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
        "upstream": upstream,
        "field_path": field_path,
        "principle": principle,
    }


FIELD_PRINCIPLES = {
    "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic clock/process identity.",
    "preconditions_checked": "Record exact resource paths, observed values, ownership and input validity.",
    "MODEL_SPECS": "An empty list prevents historical model evidence from becoming current inference.",
    "model_specs": "The lowercase empty list prevents reader-specific model identity drift.",
    "model_invoked": "False separates zero current calls from archived model events.",
    "invocation_counts": "Balanced zero call states expose attempted or unfinished current inference.",
    "inference_substrate": "The canonical aggregation name prevents methodology misclassification.",
    "inference_substrate_class": "The aggregation class applies the correct evidence and duration rules.",
    "execution_venue": "Host CPU identity stays distinct from archived board or CUDA evidence.",
    "duration_s": "Measured work and components prevent a synthetic time floor.",
    "phase_spans": "Flushed progress and real checkpoints expose unfinished operations and stalls.",
    "random_seed": "Frozen role, optimizer, audit, order and interval seeds prevent favorable reruns.",
    "reproducibility_checksum": "Bind code, model identity, prompts, roles, raw shards and validation scope.",
    "source_artifact_hashes": "Preserve original upstream bytes, verdicts and flags without laundering.",
    "rows": "Per-unit failures and censoring permit independent headline reduction.",
    "sample_size_budget": "Separate unit states prevent repeated or missing rows from inflating support.",
    "acceptance_gate_results": "Typed gates keep validity, readiness and benefit independent.",
    "gate_check_summary": "Exact failed paths prevent blocked or null operands from disappearing.",
    "honest_verdict": "A complete terminal finding distinguishes external absence from unfinished work.",
    "verdict_class": "The closed enum prevents prose from changing machine classification.",
    "verifier_is_oracle": "False prevents this audit from becoming positive scientific evidence.",
    "flagged_adversarial": "Actual upstream and reader flags cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, exits and log hashes make current checks reviewable.",
    "field_principles": "Every field states its failure-prevention purpose.",
    "science_audit_complete_score": "Branch accounting is distinct from aggregate scientific success.",
    "independent_probability_rows": "Per-group probability comparisons prevent summary-only score claims.",
    "independent_learning_rows": "Event replay prevents a producer summary from hiding future feedback.",
    "claim_dispositions": "Each producer remains valid, null, absent, blocked or disqualified.",
}


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    return {
        key: FIELD_PRINCIPLES.get(
            key, "This field preserves measured audit evidence and prevents silent omission."
        )
        for key in keys
    }


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in REQUIRED_RECEIPTS
    )


def _build_gates(value: Mapping[str, Any]) -> list[Json]:
    dispositions = list(value.get("claim_dispositions") or [])
    by_number = {row.get("producer_number"): row for row in dispositions}
    errors = list(value.get("reduction_errors") or [])
    attacks = value.get("attack_controls") or {}
    gates = [
        _gate(
            "independent_reduction_errors",
            "validity",
            [],
            errors,
            "==",
            not errors,
            upstream="current_audit",
            field_path="reduction_errors",
        ),
        _gate(
            "private_attack_controls",
            "validity",
            True,
            attacks.get("passed"),
            "==",
            attacks.get("passed") is True,
            upstream="current_audit",
            field_path="attack_controls.passed",
        ),
        _gate(
            "required_current_validation",
            "validity",
            True,
            _required_receipts_pass(value.get("validation_receipts") or []),
            "==",
            _required_receipts_pass(value.get("validation_receipts") or []),
            upstream="current_audit",
            field_path="validation_receipts",
        ),
    ]
    names = {
        7491: "window_protocol_available",
        7492: "window_pilot_available",
        7493: "fit_capture_available",
        7494: "evaluation_capture_available",
        7495: "probability_and_utility_available",
        7496: "causal_fixture_available",
        7497: "causal_learning_and_retention_available",
    }
    for number, name in names.items():
        disposition = by_number.get(number, {})
        observed = disposition.get("disposition")
        gates.append(
            _gate(
                name,
                "readiness",
                "valid",
                observed,
                "==",
                observed == "valid",
                upstream=str(disposition.get("path") or f"exp{number}"),
                field_path="claim_dispositions.disposition",
            )
        )
    gates.extend(
        [
            _gate(
                "probability_support_effect_and_multiplicity",
                "benefit",
                "passed",
                by_number.get(7495, {}).get("benefit_state", "absent"),
                "==",
                by_number.get(7495, {}).get("benefit_state") == "passed",
                upstream=str(by_number.get(7495, {}).get("path") or "exp7495"),
                field_path="probability_summary.benefit_state",
            ),
            _gate(
                "nine_utility_cells",
                "benefit",
                9,
                len(value.get("independent_utility_rows") or []),
                "==",
                len(value.get("independent_utility_rows") or []) == 9,
                upstream=str(by_number.get(7495, {}).get("path") or "exp7495"),
                field_path="independent_utility_rows",
            ),
            _gate(
                "causal_learning_and_retention",
                "benefit",
                "passed",
                by_number.get(7497, {}).get("benefit_state", "absent"),
                "==",
                by_number.get(7497, {}).get("benefit_state") == "passed",
                upstream=str(by_number.get(7497, {}).get("path") or "exp7497"),
                field_path="learning_summary.benefit_state",
            ),
        ]
    )
    return gates


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    failed = [
        {
            "check": row["check"],
            "category": row["category"],
            "upstream": row["upstream"],
            "field_path": row["field_path"],
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


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks and this checksum."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "ended_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "duration_breakdown_s",
        "phase_spans",
        "process_identity",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _sample_budget(rows: Sequence[Mapping[str, Any]]) -> Json:
    return {
        "planned": len(rows),
        "attempted": sum(row.get("attempted") is True for row in rows),
        "complete": sum(row.get("complete") is True for row in rows),
        "failed": sum(row.get("failed") is True for row in rows),
        "excluded": sum(row.get("excluded") is True for row in rows),
        "censored": sum(row.get("censored") is True for row in rows),
        "unstarted": sum(row.get("unstarted") is True for row in rows),
        "independent_unit": "source_group",
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
    dispositions = [str(row.get("availability")) for row in reduction["claim_dispositions"]]
    terminal = classify_terminal(dispositions, reduction.get("reduction_errors") or [])
    rows = deepcopy(list(reduction.get("rows") or []))
    duration = (ended_monotonic_ns - started_monotonic_ns) / 1e9
    value: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
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
        "duration_breakdown_s": {
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "optimization": 0.0,
            "aggregation": max(
                0.0,
                duration - math.fsum(float(row.get("duration_s", 0.0)) for row in receipts),
            ),
            "validation": math.fsum(float(row.get("duration_s", 0.0)) for row in receipts),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "role": "pinned_upstream_roles",
            "optimizer": None,
            "audit": 7_498_656,
            "order": "pinned_upstream_order",
            "interval": "deterministic_exact_reduction_no_resampling_when_branch_absent",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(list(reduction.get("source_artifact_hashes") or [])),
        "rows": rows,
        "sample_size_budget": _sample_budget(rows),
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "verifier_is_oracle": False,
        "flagged_adversarial": any(
            row.get("original_flagged_adversarial") is True
            for row in reduction["claim_dispositions"]
        ),
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "field_principles": {},
        "science_audit_complete_score": terminal["science_audit_complete_score"],
        "independent_probability_rows": deepcopy(
            list(reduction.get("independent_probability_rows") or [])
        ),
        "independent_learning_rows": deepcopy(
            list(reduction.get("independent_learning_rows") or [])
        ),
        "independent_utility_rows": deepcopy(list(reduction.get("independent_utility_rows") or [])),
        "claim_dispositions": deepcopy(list(reduction["claim_dispositions"])),
        "protocol_reduction": deepcopy(dict(reduction.get("protocol_reduction") or {})),
        "capture_reductions": deepcopy(dict(reduction.get("capture_reductions") or {})),
        "attack_controls": deepcopy(dict(reduction.get("attack_controls") or {})),
        "reduction_errors": deepcopy(list(reduction.get("reduction_errors") or [])),
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": _required_receipts_pass(receipts),
            "numbered_runtime_e2e": [],
            "reason": "Pure reporting changed no shared sampler, training, binding, ARC, telemetry, or Rust code.",
        },
        "external_publication_authorized": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
    }
    value["acceptance_gate_results"] = _build_gates(value)
    value["gate_check_summary"] = _gate_summary(value["acceptance_gate_results"])
    value["field_principles"] = _field_principles(tuple(value))
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def fixture_artifact() -> Json:
    """Build a small blocked artifact for contract mutation tests."""

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
    dispositions = [
        {
            "producer_number": number,
            "path": f"results/exp{number}.json",
            "availability": "available" if number < 7495 else "absent",
            "disposition": "valid" if number < 7495 else "absent",
            "benefit_state": "not_claimed" if number < 7495 else "absent",
            "original_flagged_adversarial": False,
        }
        for number in range(7491, 7498)
    ]
    rows = [
        {
            "unit_id": "g1",
            "group_id": "g1",
            "role": "test",
            "attempted": True,
            "complete": True,
            "failed": False,
            "excluded": False,
            "censored": False,
            "unstarted": False,
        }
    ]
    now = time.monotonic_ns()
    return _base_artifact(
        preconditions=[{"check": "fixture", "passed": True}],
        reduction={
            "claim_dispositions": dispositions,
            "source_artifact_hashes": [],
            "rows": rows,
            "protocol_reduction": {"passed": True},
            "capture_reductions": {},
            "attack_controls": run_attack_controls(),
            "reduction_errors": [],
            "independent_probability_rows": [],
            "independent_learning_rows": [],
            "independent_utility_rows": [],
        },
        receipts=receipts,
        started_at_utc="2026-09-21T00:00:00+00:00",
        started_monotonic_ns=now,
        ended_monotonic_ns=now + 1,
        phase_spans=[],
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_files: bool = True
) -> list[str]:
    """Cold-check identity, provenance, fields, gates, hashes, and checksum."""

    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "experiment_id",
        "milestone",
        "terminal_status",
        "status",
    }
    errors.extend(f"missing_field:{name}" for name in sorted(required - value.keys()))
    if (
        value.get("schema"),
        value.get("experiment_id"),
        value.get("milestone"),
        value.get("run_date"),
        value.get("terminal_status"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE, "complete"):
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
    principles = value.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(value)
        or any(not isinstance(item, str) or not item.strip() for item in principles.values())
    ):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results")
    required_gate_fields = {
        "check",
        "category",
        "expected",
        "observed",
        "op",
        "passed",
        "upstream",
        "field_path",
        "principle",
    }
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(row, Mapping) or required_gate_fields - row.keys() for row in gates)
    ):
        errors.append("gate_contract_invalid")
    if value.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_invalid")
    if value.get("science_audit_complete_score") not in {0, 1}:
        errors.append("science_audit_score_invalid")
    dispositions = value.get("claim_dispositions") or []
    if len(dispositions) != 7 or [row.get("producer_number") for row in dispositions] != list(
        range(7491, 7498)
    ):
        errors.append("claim_dispositions_invalid")
    if value.get("verdict_class") == "blocked" and not any(
        row.get("disposition") in {"absent", "blocked"} for row in dispositions
    ):
        errors.append("blocked_without_missing_branch")
    if not _required_receipts_pass(value.get("validation_receipts") or []):
        errors.append("required_validation_failed")
    if value.get("sample_size_budget") != _sample_budget(value.get("rows") or []):
        errors.append("sample_size_budget_mismatch")
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
    root: Path,
    path: Path,
    *,
    evidence_class: str,
    producer_number: int,
    artifact: Mapping[str, Any] | None = None,
) -> Json:  # pragma: no cover - runtime byte authentication.
    resolved = path if path.is_absolute() else root / path
    return {
        "path": str(path) if path.is_absolute() else path.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "evidence_class": evidence_class,
        "producer_number": producer_number,
        "original_verdict_class": (artifact or {}).get("verdict_class"),
        "original_honest_verdict": (artifact or {}).get("honest_verdict"),
        "original_flagged_adversarial": (artifact or {}).get("flagged_adversarial"),
    }


def _roadmap(root: Path) -> list[Json]:  # pragma: no cover - runtime I/O boundary.
    value = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("roadmap_not_object")
    return resolve_roadmap_producers(value)


def _locate(root: Path, spec: Mapping[str, Any]) -> Json:  # pragma: no cover
    relative = Path(str(spec["deliverable"]))
    value = _load_object(root / relative)
    if not value:
        return {**spec, "availability": "absent", "path": relative.as_posix(), "artifact": {}}
    if value.get("milestone") != MILESTONE or not str(value.get("experiment_id", "")).startswith(
        f"exp{spec['number']}"
    ):
        return {**spec, "availability": "invalid", "path": relative.as_posix(), "artifact": value}
    availability = "pre_gate" if value.get("blocked_at_layer") else "available"
    return {**spec, "availability": availability, "path": relative.as_posix(), "artifact": value}


def collect_preconditions(root: Path) -> tuple[list[Json], list[Json]]:  # pragma: no cover
    """Authenticate instructions, roadmap slots, exclusions, and original flags."""

    sources = (
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
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
        SPEC_PATH,
        ROADMAP_PATH,
    )
    checks: list[Json] = []
    for relative in sources:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "path": relative.as_posix(),
                "ownership": "repository_instruction_or_current_code",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "op": "==",
                "passed": available,
                "owner_uid": path.stat().st_uid if available else None,
                "size_bytes": path.stat().st_size if available else None,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        {
            "check": "driving_requirement",
            "path": SPEC_PATH.as_posix(),
            "ownership": "current_spec",
            "expected": "REQ-REPORT-7498",
            "observed": "REQ-REPORT-7498" if "REQ-REPORT-7498" in spec_text else None,
            "op": "==",
            "passed": "REQ-REPORT-7498" in spec_text,
        }
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7498" in exclusion or EXPERIMENT_ID in exclusion
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
    located = [_locate(root, spec) for spec in _roadmap(root)]
    for row in located:
        checks.append(
            {
                "check": f"producer_slot_exp{row['number']}",
                "path": row["path"],
                "ownership": "upstream_or_external_absence",
                "expected": "available_pre_gate_or_explicit_absence",
                "observed": row["availability"],
                "op": "in",
                "passed": row["availability"] in {"available", "pre_gate", "absent"},
            }
        )
    return checks, located


def _hash_ref(
    root: Path,
    path: Path,
    expected_sha: Any,
    *,
    producer_number: int,
    evidence_class: str,
) -> tuple[Json | None, str | None]:  # pragma: no cover
    resolved = root / path
    if not resolved.is_file():
        return None, f"source_missing:{path}"
    row = _source_row(root, path, evidence_class=evidence_class, producer_number=producer_number)
    if expected_sha is not None and row["sha256"] != expected_sha:
        return row, f"source_hash_mismatch:{path}"
    return row, None


def _artifact_refs(
    root: Path, number: int, artifact: Mapping[str, Any]
) -> tuple[list[Json], list[str]]:  # pragma: no cover
    refs: list[tuple[Path, Any, str]] = []
    role_manifest = artifact.get("role_manifest") or {}
    raw_dir = Path(str(role_manifest.get("raw_directory") or ""))
    for receipt in (role_manifest.get("raw_shards") or {}).values():
        if isinstance(receipt, Mapping):
            refs.append((raw_dir / str(receipt.get("path")), receipt.get("sha256"), "raw_protocol"))
    for receipt in (artifact.get("raw_shards") or {}).values():
        if isinstance(receipt, Mapping):
            refs.append((Path(str(receipt.get("path"))), receipt.get("sha256"), "raw_pilot"))
    raw_root = Path(str(artifact.get("raw_logit_root") or ""))
    for receipt in artifact.get("raw_logit_shards") or []:
        if isinstance(receipt, Mapping):
            refs.append((raw_root / str(receipt.get("path")), receipt.get("sha256"), "raw_capture"))
    checkpoint_root = root / f"results/raw/experiment_{number}_v656_"
    matches = sorted(
        (root / "results/raw").glob(f"experiment_{number}_v656_*/checkpoints/checkpoint-index.json")
    )
    del checkpoint_root
    for index in matches:
        refs.append((index.relative_to(root), None, "checkpoint_index"))
        value = _load_object(index)
        for receipt in (value.get("groups") or {}).values():
            if isinstance(receipt, Mapping):
                refs.append(
                    (
                        index.parent.relative_to(root) / str(receipt.get("path")),
                        receipt.get("sha256"),
                        "terminal_checkpoint",
                    )
                )
    rows: list[Json] = []
    errors: list[str] = []
    seen: set[str] = set()
    for path, expected, evidence_class in refs:
        if path.as_posix() in seen:
            continue
        seen.add(path.as_posix())
        row, error = _hash_ref(
            root,
            path,
            expected,
            producer_number=number,
            evidence_class=evidence_class,
        )
        if row:
            rows.append(row)
        if error:
            errors.append(error)
    return rows, errors


def _protocol_from_artifact(root: Path, artifact: Mapping[str, Any]) -> Json:  # pragma: no cover
    role_manifest = artifact["role_manifest"]
    raw_dir = root / str(role_manifest["raw_directory"])
    shards = role_manifest["raw_shards"]
    reduced = reduce_protocol_rows(
        groups=_load_jsonl(raw_dir / shards["groups"]["path"]),
        predictors=_load_jsonl(raw_dir / shards["predictors"]["path"]),
        windows=_load_jsonl(raw_dir / shards["windows"]["path"]),
        requests=_load_jsonl(raw_dir / shards["requests"]["path"]),
        evaluators=_load_jsonl(raw_dir / shards["evaluators"]["path"]),
    )
    if reduced["group_counts"] != role_manifest.get("eligible_role_counts"):
        reduced["passed"] = False
        reduced.setdefault("errors", []).append("eligible_role_counts_mismatch")
    if reduced["request_count"] != artifact.get("request_manifest", {}).get("request_count"):
        reduced["passed"] = False
        reduced.setdefault("errors", []).append("request_count_mismatch")
    if artifact.get("role_manifest", {}).get("labels_opened_after_prediction_freeze") is not True:
        reduced["passed"] = False
        reduced.setdefault("errors", []).append("label_freeze_declaration_invalid")
    return reduced


def _capture_from_artifact(root: Path, artifact: Mapping[str, Any]) -> Json:  # pragma: no cover
    raw_root = root / str(artifact["raw_logit_root"])
    plan: list[Json] = []
    rows: list[Json] = []
    for receipt in artifact["raw_logit_shards"]:
        values = _load_jsonl(raw_root / str(receipt["path"]))
        (plan if receipt["kind"] == "plan" else rows).extend(values)
    reduced = reduce_capture_rows(plan, rows)
    declared_counts = artifact.get("role_counts") or {}
    observed_counts = {
        role: int(values.get("planned", -1)) for role, values in declared_counts.items()
    }
    if reduced["group_counts"] != observed_counts:
        reduced["passed"] = False
        reduced.setdefault("errors", []).append("role_group_counts_mismatch")
    capture = artifact.get("capture_reduction") or {}
    if reduced["complete_roster"] != bool(capture.get("capture_complete_score")):
        reduced["passed"] = False
        reduced.setdefault("errors", []).append("capture_complete_mismatch")
    return reduced


def reduce_sources(root: Path, located: Sequence[Mapping[str, Any]]) -> Json:  # pragma: no cover
    """Hash all evidence and independently reduce each available branch."""

    source_hashes: list[Json] = []
    errors: list[str] = []
    dispositions: list[Json] = []
    protocol: Json = {}
    captures: dict[str, Json] = {}
    rows_by_id: dict[str, Json] = {}
    for slot in located:
        number = int(slot["number"])
        artifact = slot["artifact"]
        availability = str(slot["availability"])
        branch_errors: list[str] = []
        if availability in {"available", "pre_gate", "invalid"} and artifact:
            source_hashes.append(
                _source_row(
                    root,
                    Path(str(slot["path"])),
                    evidence_class="upstream_terminal_or_pregate",
                    producer_number=number,
                    artifact=artifact,
                )
            )
            refs, ref_errors = _artifact_refs(root, number, artifact)
            source_hashes.extend(refs)
            branch_errors.extend(ref_errors)
        if availability == "available" and number == 7491:
            protocol = _protocol_from_artifact(root, artifact)
            branch_errors.extend(protocol.get("errors") or [])
            if protocol.get("passed") is not True:
                branch_errors.append("protocol_reduction_failed")
        elif availability == "available" and number in {7493, 7494}:
            capture = _capture_from_artifact(root, artifact)
            captures[str(number)] = capture
            branch_errors.extend(capture.get("errors") or [])
            if capture.get("passed") is not True:
                branch_errors.append(f"capture_reduction_failed:{number}")
            for row in capture["unit_rows"]:
                rows_by_id[str(row["group_id"])] = row
        elif availability == "available" and number == 7492:
            raw = artifact.get("raw_shards", {}).get("pilot_call_rows", {})
            pilot_rows = _load_jsonl(root / str(raw.get("path"))) if raw else []
            mapping_valid = bool(pilot_rows) and all(
                set(row.get("raw_logits_by_option_id") or {}) == OPTION_IDS
                and set(row.get("label_to_option_id") or {}) == {" A", " B"}
                for row in pilot_rows
                if row.get("disposition") == "complete"
            )
            captures["7492"] = {
                "passed": mapping_valid,
                "raw_call_count": len(pilot_rows),
                "option_mapping_valid": mapping_valid,
            }
            if not mapping_valid:
                branch_errors.append("pilot_option_mapping_invalid")
        elif availability == "available" and number in {7495, 7496, 7497}:
            branch_errors.append(f"unsupported_present_producer_schema:{number}")
        if availability == "invalid":
            branch_errors.append(f"producer_identity_invalid:{number}")
        original_class = artifact.get("verdict_class")
        original_flag = artifact.get("flagged_adversarial")
        if availability == "absent":
            disposition = "absent"
        elif availability == "pre_gate":
            disposition = "blocked"
        elif (
            availability == "invalid"
            or original_class == "disqualified"
            or original_flag is True
            or branch_errors
        ):
            disposition = "disqualified"
        else:
            disposition = "valid"
        dispositions.append(
            {
                "producer_number": number,
                "task_id": slot["task_id"],
                "path": slot["path"],
                "availability": availability,
                "disposition": disposition,
                "benefit_state": "absent" if availability == "absent" else "not_claimed",
                "original_honest_verdict": artifact.get("honest_verdict"),
                "original_verdict_class": original_class,
                "original_flagged_adversarial": original_flag,
                "reduction_errors": sorted(set(branch_errors)),
            }
        )
        errors.extend(branch_errors)
    return {
        "source_artifact_hashes": sorted(
            source_hashes, key=lambda row: (int(row["producer_number"]), str(row["path"]))
        ),
        "claim_dispositions": dispositions,
        "protocol_reduction": protocol,
        "capture_reductions": captures,
        "rows": list(rows_by_id.values()) or protocol.get("unit_rows", []),
        "independent_probability_rows": [],
        "independent_learning_rows": [],
        "independent_utility_rows": [],
        "attack_controls": run_attack_controls(),
        "reduction_errors": sorted(set(errors)),
    }


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:  # pragma: no cover
    """Re-read every available raw input and compare stable audit fields."""

    value = _load_object(path)
    if not value:
        return ["candidate_unreadable"]
    preconditions, located = collect_preconditions(root)
    current = reduce_sources(root, located)
    errors = validate_artifact(value, root=root)
    for field in (
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "independent_probability_rows",
        "independent_learning_rows",
        "independent_utility_rows",
        "claim_dispositions",
        "protocol_reduction",
        "capture_reductions",
        "attack_controls",
        "reduction_errors",
    ):
        current_value = current.get(field)
        if field == "sample_size_budget":
            current_value = _sample_budget(current.get("rows") or [])
        if value.get(field) != current_value:
            errors.append(f"independent_replay_mismatch:{field}")
    if value.get("preconditions_checked") != preconditions:
        errors.append("independent_replay_mismatch:preconditions_checked")
    return sorted(set(errors))


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print one flushed boundary with measured monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7498] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _with_heartbeat(
    operation: str, fn: Callable[[], T], *, started: float, heartbeat_s: float = 60.0
) -> T:  # pragma: no cover
    """Emit truthful pending messages during a long in-process reduction."""

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
            "raw_upstream_rows",
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
    """Reduce, validate in fresh processes, and publish one terminal JSON."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = datetime.now(UTC).isoformat()
    spans: list[Json] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic_ns()
    preconditions, located = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started_ns, len(preconditions)))
    failed = [row for row in preconditions if row.get("passed") is not True]
    progress(started, "preconditions", "complete", completed=len(preconditions), failed=len(failed))
    if failed:
        raise RuntimeError(f"precondition_authentication_failed:{failed[0]}")

    for phase in ("model_load", "generation", "optimization"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic_ns()
        spans.append(_span(phase, phase_started, started_ns, 0))
        progress(started, phase, "after", completed=0)

    progress(started, "raw_reduction", "before_benchmark")
    phase_started = time.monotonic_ns()
    reduction = _with_heartbeat(
        "raw_reduction", lambda: reduce_sources(root, located), started=started
    )
    spans.append(_span("raw_reduction", phase_started, started_ns, len(reduction["rows"])))
    progress(
        started,
        "raw_reduction",
        "after_benchmark",
        completed_units=len(reduction["rows"]),
        missing=sum(row["availability"] == "absent" for row in reduction["claim_dispositions"]),
    )

    private_root = Path(tempfile.mkdtemp(prefix="exp7498-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    manifest_path = root / RAW_DIR / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        },
    )
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
