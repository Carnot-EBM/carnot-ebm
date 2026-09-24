"""Reconcile fourteen V664 outcomes without merging scientific branches.

This read-only report authenticates existing artifacts. It performs no model
inference, hardware operation, roadmap change, or external publication.

Spec refs: REQ-REPORT-7614 and SCENARIO-REPORT-7614-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7600_v663_capstone as prior
from carnot import experiment_7601_v664_contract_methods as contract_reader
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7572_v661_capstone import (
    _receipt_set_passed,
    evaluate_publication_gates,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.664"
EXPERIMENT_ID = "exp7614-capstone"
SCHEMA = "carnot.exp7614.v664.capstone.v1"

EXPECTED_TASK_IDS = contract_reader.EXPECTED_TASK_IDS
TERMINAL_CLASSES = contract_reader.TERMINAL_CLASSES
ZERO_INVOCATION_COUNTS = contract_reader.ZERO_INVOCATION_COUNTS
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v664-capstone.md")
GAPS_PATH = Path("ops/verifier_gaps.md")
RESULT_PATH = Path("results/experiment_7614_v664_capstone.json")
RAW_DIR = Path("results/raw/experiment_7614_v664_capstone")
MODULE_PATH = Path("python/carnot/experiment_7614_v664_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7614_v664_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7614_v664_capstone.py")
V663_ARCHIVE_PATH = Path("openspec/change-proposals/research-roadmap-v663-preserved-20260924.md")

DIAGNOSTIC_PATHS = {
    "exp7605-fit-evidence": Path("results/experiment_7605_fit_evidence.json"),
    "exp7606-test-online-evidence": Path("results/experiment_7606_test_online_evidence.json"),
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7600_v663_capstone.py"),
    Path("scripts/publication_gate.py"),
    Path("scripts/recurring_blocker_ledger.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("ops/verifier_gaps.md"),
    DESIGN_PATH,
    SPEC_PATH,
    V663_ARCHIVE_PATH,
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def load_json(path: Path) -> JsonDict:
    """Require a JSON object because scalar evidence has no named operands."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_contract(root: Path) -> JsonDict:
    """Resolve matching V664 authority, including consumed staging."""

    selected, roadmap, candidates = contract_reader.resolve_v664_roadmap(root)
    comparison = contract_reader.compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V664 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": deepcopy(candidates),
        "tasks": deepcopy(tasks),
    }


def _failed_check(
    check: str,
    upstream: str,
    path: object,
    field: str,
    expected: object,
    observed: object,
    *,
    operator: str = "==",
) -> JsonDict:
    """Retain every operand needed to investigate a blocked prerequisite."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": False,
    }


def _base_source(task: Mapping[str, Any]) -> JsonDict:
    """Represent an absent producer before an optional pre-gate is examined."""

    task_id = str(task.get("id") or "")
    path = str(task.get("deliverable") or "")
    return {
        "task_id": task_id,
        "expected_path": path,
        "evidence_path": None,
        "source_sha256": None,
        "source_size_bytes": 0,
        "evidence_state": "missing",
        "authenticated": False,
        "valid": False,
        "honest_verdict": "complete_blocked_missing_external_producer",
        "verdict_class": "blocked",
        "original_verdict_class": None,
        "flagged_adversarial": False,
        "inference_substrate_class": None,
        "model_invoked": False,
        "gate_check_summary": _failed_check(
            "producer_artifact_exists",
            task_id,
            path,
            "path",
            True,
            False,
            operator="exists",
        ),
        "payload": {},
    }


def _diagnostic_failure(task_id: str, payload: Mapping[str, Any]) -> JsonDict:
    """Normalize a conductor pre-gate record without calling it a run."""

    raw = payload.get("blocked_diagnostic_contract")
    row = raw if isinstance(raw, Mapping) else payload
    return _failed_check(
        "conductor_pre_gate",
        str(row.get("failed_upstream") or task_id),
        row.get("failed_evidence_path"),
        str(row.get("failed_field") or "unknown"),
        row.get("failed_expected"),
        row.get("failed_observed"),
        operator=str(row.get("failed_operator") or "=="),
    )


def _producer_failure(task_id: str, path: str, payload: Mapping[str, Any]) -> JsonDict:
    """Name the first exact validity defect in a present producer."""

    if payload.get("flagged_adversarial") is True:
        field, expected, observed = "flagged_adversarial", False, True
    elif payload.get("verdict_class") not in TERMINAL_CLASSES:
        field, expected, observed = (
            "verdict_class",
            sorted(TERMINAL_CLASSES),
            payload.get("verdict_class"),
        )
    else:
        field, expected, observed = (
            "identity_and_milestone",
            task_id,
            payload.get("experiment_id", payload.get("experiment")),
        )
    return _failed_check("producer_required_validity", task_id, path, field, expected, observed)


def load_producer(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate a terminal producer, conductor pre-gate, or absence."""

    task_id = str(task.get("id") or "")
    relative = Path(str(task.get("deliverable") or ""))
    path = root / relative
    source = _base_source(task)
    if not path.is_file():
        diagnostic_relative = DIAGNOSTIC_PATHS.get(task_id)
        diagnostic = root / diagnostic_relative if diagnostic_relative else None
        if diagnostic is None or not diagnostic.is_file():
            return source
        payload = load_json(diagnostic)
        source.update(
            evidence_path=diagnostic_relative.as_posix(),
            source_sha256=sha256_file(diagnostic),
            source_size_bytes=diagnostic.stat().st_size,
            evidence_state="conductor_gate_blocked",
            authenticated=True,
            honest_verdict=str(payload.get("honest_verdict") or "blocked_gate_check_failed"),
            verdict_class="blocked",
            inference_substrate_class="blocked_no_run",
            gate_check_summary=_diagnostic_failure(task_id, payload),
            payload=payload,
        )
        return source
    try:
        payload = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        source.update(
            evidence_path=relative.as_posix(),
            source_sha256=sha256_file(path),
            source_size_bytes=path.stat().st_size,
            evidence_state="invalid",
            honest_verdict="complete_disqualified_unreadable_producer_evidence",
            verdict_class="disqualified",
            gate_check_summary=_failed_check(
                "producer_json_object",
                task_id,
                relative.as_posix(),
                "json_object",
                "mapping",
                str(error),
            ),
        )
        return source
    number = task_id[3:7]
    observed_id = payload.get("experiment_id", payload.get("experiment"))
    original_class = payload.get("verdict_class")
    authenticated = bool(
        number in str(observed_id)
        and payload.get("milestone") == MILESTONE
        and original_class in TERMINAL_CLASSES
        and isinstance(payload.get("flagged_adversarial"), bool)
    )
    valid = bool(
        authenticated
        and original_class != "disqualified"
        and payload.get("flagged_adversarial") is False
    )
    state = "terminal_blocked" if valid and original_class == "blocked" else "terminal"
    if not valid:
        state = "invalid"
    source.update(
        evidence_path=relative.as_posix(),
        source_sha256=sha256_file(path),
        source_size_bytes=path.stat().st_size,
        evidence_state=state,
        authenticated=authenticated,
        valid=valid,
        honest_verdict=str(payload.get("honest_verdict") or payload.get("status") or "absent"),
        verdict_class=str(original_class) if valid else "disqualified",
        original_verdict_class=original_class,
        flagged_adversarial=payload.get("flagged_adversarial") is True,
        inference_substrate_class=payload.get("inference_substrate_class"),
        model_invoked=payload.get("model_invoked") is True,
        gate_check_summary=(
            deepcopy(payload.get("gate_check_summary"))
            if valid
            else _producer_failure(task_id, relative.as_posix(), payload)
        ),
        payload=payload,
    )
    return source


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Inventory all thirteen upstream tasks in authority order."""

    return {str(task.get("id")): load_producer(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, affected_complete: bool, terminal_complete: bool
) -> JsonDict:
    """Reserve partial for unfinished work owned by this capstone."""

    if affected_complete and not terminal_complete:
        verdict, honest = "partial", "partial_retryable_current_capstone_validation_unfinished"
    elif not affected_complete:
        verdict, honest = "disqualified", "complete_disqualified_required_v664_validation"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict, honest = "disqualified", "complete_disqualified_required_v664_evidence"
    elif any(
        row.get("evidence_state") in {"missing", "conductor_gate_blocked", "terminal_blocked"}
        or row.get("verdict_class") == "blocked"
        for row in rows
    ):
        verdict, honest = "blocked", "complete_blocked_required_v664_external_evidence"
    else:
        verdict, honest = "null", "complete_null_v664_accounting_without_aggregate_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _disposition(source: Mapping[str, Any], task: Mapping[str, Any], order: int) -> JsonDict:
    """Reduce one source to an auditable custody row, never a science row."""

    state = source.get("evidence_state")
    if state in {"missing", "conductor_gate_blocked"}:
        disposition = "absent_external_or_pre_gated"
    elif state == "invalid":
        disposition = "disqualified"
    else:
        disposition = "valid_terminal"
    excluded = bool(
        disposition != "valid_terminal"
        or source.get("verdict_class") in {"blocked", "disqualified"}
        or source.get("flagged_adversarial") is True
    )
    return {
        "order": order,
        "task_id": str(task.get("id")),
        "arm": "authenticated_task_disposition",
        "disposition": disposition,
        "expected_artifact_path": source.get("expected_path"),
        "evidence_path": source.get("evidence_path"),
        "artifact_sha256": source.get("source_sha256"),
        "evidence_state": state,
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("verdict_class"),
        "original_verdict_class": source.get("original_verdict_class"),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
        "producer_validation_state": "authenticated" if source.get("authenticated") else state,
        "claim_scope": "descriptive_reuse",
        "excluded_from_scientific_metrics": excluded,
        "disposition_attempted": True,
        "disposition_completed": True,
        "producer_started": state not in {"missing", "conductor_gate_blocked"},
        "producer_completed": state in {"terminal", "terminal_blocked"},
        "producer_failed": state == "invalid",
        "producer_unstarted": state in {"missing", "conductor_gate_blocked"},
        "raw_numerator": 1,
        "raw_denominator": 1,
        "direction": "exact_custody_is_required",
        "seed": None,
        "missing": state in {"missing", "conductor_gate_blocked"},
        "censored": state == "missing",
        "provenance": source.get("evidence_path") or source.get("expected_path"),
        "principle": "Exact custody keeps absent, invalid, blocked, and flagged evidence distinct.",
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    *,
    current_validation_complete: bool,
) -> list[JsonDict]:
    """Build thirteen upstream rows and one self row without self-reading."""

    rows = [
        _disposition(evidence[str(task.get("id"))], task, order)
        for order, task in enumerate(tasks[:-1], 1)
    ]
    rows.append(
        {
            "order": 14,
            "task_id": str(tasks[-1].get("id")),
            "arm": "authenticated_task_disposition",
            "disposition": "valid_terminal" if current_validation_complete else "owned_incomplete",
            "expected_artifact_path": str(tasks[-1].get("deliverable")),
            "evidence_path": None,
            "artifact_sha256": None,
            "evidence_state": (
                "current_terminal" if current_validation_complete else "current_running"
            ),
            "honest_verdict": terminal.get("honest_verdict"),
            "verdict_class": terminal.get("verdict_class"),
            "original_verdict_class": terminal.get("verdict_class"),
            "flagged_adversarial": False,
            "producer_validation_state": "current_owned_validation",
            "claim_scope": "current_accounting",
            "excluded_from_scientific_metrics": True,
            "disposition_attempted": True,
            "disposition_completed": current_validation_complete,
            "producer_started": True,
            "producer_completed": current_validation_complete,
            "producer_failed": False,
            "producer_unstarted": False,
            "raw_numerator": int(current_validation_complete),
            "raw_denominator": 1,
            "direction": "owned_validation_complete_is_required",
            "seed": None,
            "missing": not current_validation_complete,
            "censored": not current_validation_complete,
            "provenance": "current_exp7614_validation_receipts",
            "principle": "Current work cannot authenticate itself through its future result path.",
        }
    )
    return rows


def _source_ref(source: Mapping[str, Any]) -> JsonDict:
    """Bind branch evidence by exact bytes without copying all producer rows."""

    return {
        "task_id": source.get("task_id"),
        "path": source.get("evidence_path") or source.get("expected_path"),
        "sha256": source.get("source_sha256"),
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("original_verdict_class") or source.get("verdict_class"),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
        "evidence_state": source.get("evidence_state"),
        "claim_scope": "descriptive_reuse",
    }


def _audit_branch(audit: Mapping[str, Any], source_name: str, branch: str) -> JsonDict:
    """Copy only an Exp7610 independent conclusion for one science branch."""

    payload = audit.get("payload") or {}
    rows = payload.get("branch_conclusions") or []
    selected = next(
        (
            deepcopy(dict(row))
            for row in rows
            if isinstance(row, Mapping) and row.get("branch") == source_name
        ),
        None,
    )
    blocked = selected is None or str(selected.get("validity", "")).startswith("blocked")
    if source_name == "information_value" and selected is not None:
        blocked = (
            selected.get("conclusion") == "transport_failure_is_not_an_incremental_information_null"
        )
    return {
        "branch": branch,
        "source_task": "exp7610-evidence-audit",
        "sources": [_source_ref(audit)],
        "validity": selected.get("validity") if selected else "blocked_missing_independent_row",
        "readiness": selected.get("readiness") if selected else "not_established",
        "benefit": "not_measured" if blocked else selected.get("benefit"),
        "verdict_class": "blocked" if blocked else "null",
        "independent_conclusion": selected,
        "sample_size_budget": deepcopy(payload.get("sample_size_budget") or []),
        "source_exposure": "historically_exposed_descriptive_reuse",
        "intervals": "unavailable_without_complete_static_or_online_rows",
        "complete_cost": "unavailable_without_decision_rows"
        if branch == "action_cost"
        else "not_applicable",
        "claim_scope": "independent_exp7610_reduction",
    }


def build_branch_conclusions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep eight scientific and operational conclusions independent."""

    audit = evidence["exp7610-evidence-audit"]
    arc = evidence["exp7612-arc-history-measurement"]
    service = evidence["exp7613-service-attribution"]
    arc_payload = arc.get("payload") or {}
    service_payload = service.get("payload") or {}
    arc_budget = arc_payload.get("sample_size_budget") or {}
    arc_reduction = arc_payload.get("support_reduction") or {}
    service_rows = service_payload.get("rows") or []
    consumer_rows = [
        row
        for row in service_rows
        if isinstance(row, Mapping) and row.get("row_type") == "consumer_stage"
    ]
    spans_authenticated = bool(
        service.get("valid")
        and len(consumer_rows) == 240
        and all(
            isinstance(row.get("denominator"), (int, float))
            and row.get("denominator") == row.get("whole_service_ns")
            for row in consumer_rows
        )
    )
    return [
        _audit_branch(audit, "information_value", "information_value"),
        _audit_branch(audit, "calibration", "probability"),
        _audit_branch(audit, "decision_value", "action_cost"),
        _audit_branch(audit, "learning", "causal_learning"),
        _audit_branch(audit, "retention", "retention"),
        {
            "branch": "arc_support",
            "source_task": "exp7612-arc-history-measurement",
            "sources": [_source_ref(arc)],
            "validity": "authenticated_bounded_runtime_rows",
            "readiness": False,
            "benefit": False,
            "verdict_class": "blocked",
            "observed_independent_units": arc_budget.get("observed_independent_units"),
            "excluded_independent_units": arc_budget.get("excluded_independent_units"),
            "selected_matched_keys": arc_budget.get("selected_matched_keys"),
            "history_support_score": arc_reduction.get("history_support_score"),
            "support_sufficient": False,
            "runtime_rows": deepcopy(arc_payload.get("rows") or []),
            "interval": arc_reduction.get("game_cluster_interval_95"),
            "new_solve_claimed": False,
            "conclusion": "insufficient_matched_support_stop_unchanged_collector",
            "claim_scope": "authenticated_exp7612_runtime_rows_only",
        },
        {
            "branch": "service_placement",
            "source_task": "exp7613-service-attribution",
            "sources": [_source_ref(service)],
            "validity": spans_authenticated,
            "readiness": None,
            "benefit": False,
            "verdict_class": "null",
            "whole_service_spans_authenticated": spans_authenticated,
            "paired_stage_blocks": deepcopy(
                (service_payload.get("sample_size_budget") or {}).get("paired_stage_blocks")
            ),
            "stage_reduction": deepcopy(service_payload.get("stage_reduction") or {}),
            "instrumentation_overhead": deepcopy(
                service_payload.get("instrumentation_overhead") or {}
            ),
            "amdahl_upper_bound": deepcopy(service_payload.get("amdahl_upper_bound") or {}),
            "prior_exp7598_verdict": service_payload.get("prior_exp7598_honest_verdict"),
            "cold_start_regression_preserved": (
                service_payload.get("prior_exp7598_aggregate_verdict_preserved") is True
            ),
            "automatic_native_binding_authorized": False,
            "automatic_accelerator_follow_up_authorized": False,
            "conclusion": "measured_stages_do_not_replace_failed_aggregate_speed_gate",
            "claim_scope": "whole_durable_service",
        },
        {
            "branch": "hardware",
            "source_task": "exp7613-service-attribution",
            "sources": [_source_ref(service)],
            "validity": service.get("valid") is True,
            "readiness": None,
            "benefit": "not_measured",
            "verdict_class": "null",
            "board_rows": deepcopy(service_payload.get("board_rows") or []),
            "board_prerequisite_count": len(service_payload.get("board_rows") or []),
            "operator_held_confirmations": {"E0": "operator_held", "Kaggle": "operator_held"},
            "automatic_follow_up_authorized": False,
            "purchase_authorized": False,
            "conclusion": "three_board_prerequisites_preserved_without_current_probe",
            "claim_scope": "historical_board_continuity",
        },
    ]


def retirement_decisions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Close only measured scopes and stop unchanged unsupported collection."""

    return [
        {
            "branch": "eight_feature_evidence_construction",
            "decision": "defer",
            "scope": "v664_exact_eight_feature_source_link_head",
            "scientific_hypothesis_retired": False,
            "resource_or_custody_block": True,
            "reopening_condition": "New valid source-link observations pass schema and complete static evaluation.",
            "source": _source_ref(evidence["exp7610-evidence-audit"]),
        },
        {
            "branch": "guarded_updates",
            "decision": "defer",
            "scope": "v664_guarded_delayed_residual_update",
            "scientific_hypothesis_retired": False,
            "resource_or_custody_block": True,
            "reopening_condition": "Authenticated causal-benefit and evaluator-retention rows become available.",
            "source": _source_ref(evidence["exp7610-evidence-audit"]),
        },
        {
            "branch": "matched_prefix_collector",
            "decision": "stop_unchanged_mechanism",
            "scope": "v664_six_game_matched_prefix_collector",
            "scientific_hypothesis_retired": False,
            "resource_or_custody_block": False,
            "reopening_condition": "A materially changed mechanism justifies new matched support.",
            "source": _source_ref(evidence["exp7612-arc-history-measurement"]),
        },
        {
            "branch": "prior_capstone_scope",
            "decision": "retire_narrow_repeated_scope",
            "scope": "aggregate_capstone_missing_external_evidence_disposition",
            "prior_verdict": "complete_blocked_required_v663_external_evidence",
            "current_verdict": "complete_blocked_required_v664_external_evidence",
            "normalized_scope_match": True,
            "scientific_hypothesis_retired": False,
            "reopening_condition": "Required external producers become authenticated and eligible.",
        },
    ]


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Bind current sources, authority, and every producer or diagnostic byte."""

    paths = [*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH, NOTE_PATH]
    selected = Path(str(contract.get("selected_roadmap_path")))
    if selected not in paths:
        paths.append(selected)
    rows = [
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "evidence_class": "current_source",
        }
        for path in dict.fromkeys(paths)
    ]
    rows.extend(
        {
            "path": source.get("evidence_path") or source.get("expected_path"),
            "sha256": source.get("source_sha256"),
            "evidence_class": source.get("evidence_state"),
            "task_id": task_id,
            "original_honest_verdict": source.get("honest_verdict"),
            "original_verdict_class": source.get("original_verdict_class"),
            "original_flagged_adversarial": source.get("flagged_adversarial") is True,
        }
        for task_id, source in evidence.items()
    )
    return rows


def _source_hashes_match(value: Mapping[str, Any], root: Path) -> bool:
    """Recheck every row that claims present exact bytes."""

    rows = value.get("source_artifact_hashes")
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        expected = row.get("sha256")
        if expected is None:
            continue
        path = root / str(row.get("path"))
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def _gate(
    check: str,
    category: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    operator: str = "==",
) -> JsonDict:
    """Keep one acceptance result separate from unrelated branch results."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": principle,
    }


def acceptance_gates(
    contract: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep validity, readiness, benefit, retention, and freshness distinct."""

    by_branch = {str(row.get("branch")): row for row in branches}
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    invalid = "Invalid or incomplete evidence cannot support a scientific claim."
    readiness = "Completion and interface readiness do not establish benefit."
    benefit = "Each empirical claim needs its own independently passing gate."
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v664_contract",
            str(contract.get("selected_roadmap_path")),
            "comparison_passed",
            True,
            contract.get("comparison_passed"),
            contract.get("comparison_passed") is True,
            invalid,
        ),
        _gate(
            "current_validation",
            "validity",
            EXPERIMENT_ID,
            "validation_receipts",
            "all_required_passed",
            True,
            current_valid,
            current_valid,
            invalid,
        ),
        _gate(
            "fourteen_dispositions",
            "readiness",
            EXPERIMENT_ID,
            "task_dispositions",
            "length",
            14,
            len(dispositions),
            len(dispositions) == 14,
            readiness,
        ),
        _gate(
            "incremental_information",
            "benefit",
            "exp7610-evidence-audit",
            "branch_conclusions.information_value",
            "benefit",
            True,
            by_branch["information_value"].get("benefit"),
            by_branch["information_value"].get("benefit") is True,
            benefit,
        ),
        _gate(
            "probability_benefit",
            "benefit",
            "exp7610-evidence-audit",
            "branch_conclusions.probability",
            "benefit",
            True,
            by_branch["probability"].get("benefit"),
            by_branch["probability"].get("benefit") is True,
            benefit,
        ),
        _gate(
            "action_cost_benefit",
            "benefit",
            "exp7610-evidence-audit",
            "branch_conclusions.action_cost",
            "benefit",
            True,
            by_branch["action_cost"].get("benefit"),
            by_branch["action_cost"].get("benefit") is True,
            benefit,
        ),
        _gate(
            "causal_learning_benefit",
            "benefit",
            "exp7610-evidence-audit",
            "branch_conclusions.causal_learning",
            "benefit",
            True,
            by_branch["causal_learning"].get("benefit"),
            by_branch["causal_learning"].get("benefit") is True,
            benefit,
        ),
        _gate(
            "evaluator_retention",
            "retention",
            "exp7610-evidence-audit",
            "branch_conclusions.retention",
            "benefit",
            True,
            by_branch["retention"].get("benefit"),
            by_branch["retention"].get("benefit") is True,
            benefit,
        ),
        _gate(
            "arc_support_floor",
            "readiness",
            "exp7612-arc-history-measurement",
            "support_reduction",
            "support_sufficient",
            True,
            by_branch["arc_support"].get("support_sufficient"),
            by_branch["arc_support"].get("support_sufficient") is True,
            readiness,
        ),
        _gate(
            "whole_service_spans",
            "validity",
            "exp7613-service-attribution",
            "stage_reduction",
            "whole_service_spans_authenticated",
            True,
            by_branch["service_placement"].get("whole_service_spans_authenticated"),
            by_branch["service_placement"].get("whole_service_spans_authenticated") is True,
            invalid,
        ),
        _gate(
            "board_prerequisites_carried",
            "readiness",
            "exp7613-service-attribution",
            "board_rows",
            "count",
            3,
            by_branch["hardware"].get("board_prerequisite_count"),
            by_branch["hardware"].get("board_prerequisite_count") == 3,
            readiness,
        ),
        _gate(
            "fresh_confirmatory_claim",
            "freshness",
            EXPERIMENT_ID,
            "branch_conclusions",
            "claim_scope",
            "descriptive_reuse",
            "descriptive_reuse",
            True,
            "Historical exposure cannot become fresh confirmation.",
        ),
    ]


def failure_rows(
    contract: Mapping[str, Any], evidence: Mapping[str, JsonDict], validation: Mapping[str, Any]
) -> list[JsonDict]:
    """List only exact defects that determine the aggregate terminal class."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            _failed_check(
                "contract_authorities_agree",
                "v664_contract",
                contract.get("selected_roadmap_path"),
                "comparison_passed",
                True,
                contract.get("comparison_passed"),
            )
        )
    for field in ("required_checks_passed", "terminal_validation_passed"):
        if validation.get(field) is not True:
            failures.append(
                _failed_check(
                    f"current_{field}",
                    EXPERIMENT_ID,
                    "validation_receipts",
                    field,
                    True,
                    validation.get(field),
                )
            )
    for task_id, source in evidence.items():
        state = source.get("evidence_state")
        summary = source.get("gate_check_summary")
        if state in {"missing", "conductor_gate_blocked", "invalid"}:
            if isinstance(summary, Mapping):
                failure = deepcopy(dict(summary))
                if "operator" not in failure:
                    failure["operator"] = failure.pop("op", "==")
                failure["upstream"] = task_id
                failures.append(failure)
            else:  # pragma: no cover - normalized sources always have a mapping.
                failures.append(
                    _failed_check(
                        "producer_state",
                        task_id,
                        source.get("expected_path"),
                        "evidence_state",
                        "terminal",
                        state,
                    )
                )
        elif state == "terminal_blocked" or source.get("verdict_class") == "blocked":
            failures.append(
                _failed_check(
                    "producer_terminal_not_blocked",
                    task_id,
                    source.get("evidence_path"),
                    "verdict_class",
                    "not blocked",
                    "blocked",
                    operator="!=",
                )
            )
    return failures


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every aggregate failure and its first exact cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


FIELD_PRINCIPLES = {
    "honest_verdict": "A complete prefix records terminal work without implying benefit.",
    "verdict_class": "The closed class reserves partial for unfinished owned work.",
    "flagged_adversarial": "Flagged evidence cannot open readiness or benefit.",
    "gate_check_summary": "Every block retains exact upstream operands.",
    "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
    "rows": "One ordered task row keeps absence distinct from zero.",
    "sample_size_budget": "Seeds, windows, and replay arms never multiply independent units.",
    "inference_substrate": "Current aggregation cannot become historical model work.",
    "inference_substrate_class": "Planned and actual execution classes stay explicit.",
    "MODEL_SPECS": "An empty list prevents historical identities becoming current calls.",
    "invocation_counts": "Typed zeros expose invented loads, forwards, generations, or tokens.",
    "duration_s": "Monotonic current time excludes inherited work and artificial delay.",
    "random_seed": "Each stochastic stage has a seed or an explicit absence.",
    "reproducibility_checksum": "Configuration, immutable evidence, and reductions bind identity.",
    "source_artifact_hashes": "Byte hashes distinguish producers, pre-gates, and absences.",
    "validation_receipts": "Commands, exits, worktree, logs, and readers bind validation.",
    "verifier_is_oracle": "Exact fixtures cannot establish learned semantic correctness.",
    "field_principles": "Every top-level field names the inference failure it prevents.",
    "task_dispositions": "Exactly fourteen full IDs retain authority order and custody.",
    "branch_conclusions": "Independent science, ARC, and service branches cannot promote one another.",
    "retirement_rows": "Only a measured failed construction can be retired scientifically.",
    "publication_gates": "Stable G1-G4 eligibility never authorizes submission.",
    "submitted_externally": "False preserves operator-only publication authority.",
}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give each top-level field a short failure-prevention principle."""

    return {
        field: FIELD_PRINCIPLES.get(
            field, f"Exact {field} evidence prevents an unstated inference."
        )
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks and process identity."""

    excluded = {
        "reproducibility_checksum",
        "field_principles",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "phase_spans",
        "process_identity",
    }
    return canonical_hash({key: item for key, item in value.items() if key not in excluded})


def _sample_budget(dispositions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count capstone dispositions separately from producer measurements."""

    upstream = dispositions[:-1]
    return {
        "independent_unit": "ordered_v664_task_disposition",
        "planned": 14,
        "intended": 14,
        "observed": sum(int(row.get("disposition_completed") is True) for row in dispositions),
        "excluded": sum(
            int(row.get("excluded_from_scientific_metrics") is True) for row in dispositions
        ),
        "censored": sum(int(row.get("censored") is True) for row in dispositions),
        "unstarted": sum(int(row.get("producer_unstarted") is True) for row in dispositions),
        "seeds_and_windows_multiply_units": False,
        "underlying_producer_work": {
            "planned_before_capstone": 13,
            "started": sum(int(row.get("producer_started") is True) for row in upstream),
            "completed_terminal_artifacts": sum(
                int(row.get("producer_completed") is True) for row in upstream
            ),
            "gate_blocked_without_run": sum(
                int(row.get("evidence_state") == "conductor_gate_blocked") for row in upstream
            ),
            "absent_without_artifact": sum(
                int(row.get("evidence_state") == "missing") for row in upstream
            ),
        },
    }


def _passing_receipts() -> list[JsonDict]:
    """Create deterministic passing receipts for pure test construction."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "duration_s": 0.0,
            "worktree": str(REPO_ROOT),
            "log_path": f"unit-test/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
        }
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _test_spans() -> list[JsonDict]:
    """Return deterministic phase records for pure artifact tests."""

    phases = (
        ("preconditions", 13),
        ("publication_gate", 4),
        ("model_load", 0),
        ("generation", 0),
        ("reduction", 14),
        ("validation", 8),
        ("terminal_validation", 5),
        ("write", 1),
    )
    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "completed_units": units,
            "checkpoint": f"{phase}_complete",
        }
        for phase, units in phases
    ]


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one terminal record from authenticated V664 evidence."""

    affected_complete = validation.get("required_checks_passed") is True
    terminal_complete = validation.get("terminal_validation_passed") is True
    current_complete = affected_complete and terminal_complete
    terminal = classify_terminal(
        list(evidence.values()),
        affected_complete=affected_complete,
        terminal_complete=terminal_complete,
    )
    dispositions = task_dispositions(
        contract["tasks"], evidence, terminal, current_validation_complete=current_complete
    )
    branches = build_branch_conclusions(evidence)
    retirements = retirement_decisions(evidence)
    failures = failure_rows(contract, evidence, validation)
    receipts = deepcopy(validation.get("validation_receipts") or [])
    publication = deepcopy(validation.get("publication_gates") or evaluate_publication_gates(root))
    complete_score = int(
        contract.get("comparison_passed") is True
        and len(dispositions) == 14
        and all(row.get("disposition_completed") is True for row in dispositions)
        and current_complete
    )
    terminal_readers = [
        deepcopy(dict(row))
        for row in receipts
        if isinstance(row, Mapping) and row.get("name") in TERMINAL_CHECK_NAMES
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7614,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "positive_claim": False,
        "flagged_adversarial": False,
        "readiness": None,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "phase_spans": deepcopy(list(phase_spans)),
        "process_identity": {
            "pid": os.getpid(),
            "node": platform.node(),
            "source_revision": prior.prior._source_revision(root),
        },
        "random_seed": {
            "current_aggregation": None,
            "exp7610_audit": 7_610_664_01,
            "exp7612_episode_seeds": [7_612_001, 7_612_002],
            "exp7613_stage_seed_range": [7_613_101, 7_613_430],
            "explanation": "Current deterministic aggregation makes no stochastic draw.",
        },
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {
            "planned_class": "aggregation",
            "actual_class": "aggregation",
            "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
            "historical_identity_counted_as_current": False,
            "generated_tokens": 0,
        },
        "execution_venue": "host",
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "rows": dispositions,
        "task_dispositions": dispositions,
        "sample_size_budget": _sample_budget(dispositions),
        "branch_conclusions": branches,
        "retirement_rows": retirements,
        "continuation_rows": [row for row in retirements if row.get("decision") == "defer"],
        "acceptance_gate_results": acceptance_gates(contract, dispositions, branches, validation),
        "gate_check_summary": _gate_summary(failures),
        "capstone_complete_score": complete_score,
        "publication_gates": publication,
        "paper_ready": publication["paper_ready"],
        "unmet_gates": deepcopy(publication["unmet_gates"]),
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "spec_paths": [SPEC_PATH.as_posix()],
            "note_paths": [NOTE_PATH.as_posix(), GAPS_PATH.as_posix()],
            "frozen_before_checks": True,
        },
        "validation_receipts": receipts,
        "terminal_reader_outcomes": terminal_readers,
        "worktree": str(root),
        "repository_health": deepcopy(
            validation.get("repository_health") or {"status": "outside_current_affected_validity"}
        ),
        "capability_e2e": {
            "raw_to_report_fresh_process_replay": "required",
            "independent_reduction": "required",
            "adversarial_reader": "required",
            "strict_row_consistency": "required",
            "numbered_runtime_e2e": "not_applicable_read_only_reporting",
            "unrelated_runtime_suite_executed": False,
        },
        "source_exposure": {
            "evidence_groups": "historically_exposed_descriptive_reuse",
            "arc_levels": "registered_public_development_history",
            "fresh_confirmatory_claim_allowed": False,
        },
        "independent_sample_sizes": {
            "evidence_pilot_groups": 8,
            "static_evaluation_groups": 0,
            "online_learning_groups": 0,
            "arc_games_observed": 1,
            "service_paired_blocks": 120,
            "service_telemetry_off_blocks": 40,
        },
        "intervals": {
            "information_probability_action": None,
            "causal_learning_retention": None,
            "arc_game_cluster": None,
            "service_amdahl": deepcopy(
                next(row for row in branches if row["branch"] == "service_placement")[
                    "amdahl_upper_bound"
                ]
            ),
        },
        "complete_cost": {
            "evidence_decision_cost": "unavailable_missing_decision_rows",
            "current_aggregation_duration_s": duration_s,
            "service_denominator": "whole_durable_request",
        },
        "board_prerequisites": deepcopy(
            next(row for row in branches if row["branch"] == "hardware")["board_rows"]
        ),
        "operator_held_confirmations": {"E0": "operator_held", "Kaggle": "operator_held"},
        "verifier_is_oracle": True,
        "oracle_defined_fixture_verdict_class": "circular_positive",
        "oracle_distinct_positive_claimed": False,
        "fresh_confirmatory_claim_allowed": False,
        "roadmap_activation_performed": False,
        "publication_performed": False,
        "submission_performed": False,
        "submitted_externally": False,
        "external_contact_performed": False,
        "purchase_performed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "native_binding_promoted": False,
        "accelerator_follow_up_authorized": False,
        "push_performed": False,
        "research_conductor_modified": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic terminal candidate from tracked repository bytes."""

    contract = load_contract(REPO_ROOT)
    evidence = collect_evidence(REPO_ROOT, contract["tasks"])
    publication = evaluate_publication_gates(REPO_ROOT)
    return build_artifact(
        REPO_ROOT,
        contract,
        evidence,
        {
            "required_checks_passed": True,
            "terminal_validation_passed": True,
            "validation_receipts": _passing_receipts(),
            "publication_gates": publication,
            "repository_health": {"status": "outside_current_affected_validity"},
        },
        started_at_utc="2026-09-24T00:00:00+00:00",
        completed_at_utc="2026-09-24T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        duration_s=0.0,
        phase_spans=_test_spans(),
    )


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = False
) -> list[str]:
    """Cold-check identity, custody, reductions, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    required = {
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "invocation_counts",
        "duration_s",
        "random_seed",
        "source_artifact_hashes",
        "validation_receipts",
        "verifier_is_oracle",
        "field_principles",
        "task_dispositions",
        "branch_conclusions",
        "retirement_rows",
        "publication_gates",
        "submitted_externally",
        "reproducibility_checksum",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    identity = (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    )
    if identity != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if artifact.get("verdict_class") not in TERMINAL_CLASSES or not str(
        artifact.get("honest_verdict")
    ).startswith(("complete_", "partial_")):
        errors.append("terminal_identity_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_specs") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("planned_inference_substrate_class") != "aggregation"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    if (
        artifact.get("positive_claim") is not False
        or artifact.get("flagged_adversarial") is not False
        or artifact.get("readiness") is not None
        or artifact.get("oracle_distinct_positive_claimed") is not False
        or artifact.get("submitted_externally") is not False
    ):
        errors.append("claim_contract_invalid")
    dispositions = artifact.get("task_dispositions")
    ids = (
        [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        if isinstance(dispositions, list)
        else []
    )
    if ids != list(EXPECTED_TASK_IDS) or artifact.get("rows") != dispositions:
        errors.append("task_dispositions_invalid")
    if not _source_hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        receipts = artifact.get("validation_receipts")
        validation = {
            "required_checks_passed": _receipt_set_passed(receipts, REQUIRED_CHECK_NAMES),
            "terminal_validation_passed": _receipt_set_passed(receipts, TERMINAL_CHECK_NAMES),
            "validation_receipts": receipts,
            "publication_gates": evaluate_publication_gates(root),
            "repository_health": artifact.get("repository_health", {}),
        }
        current_complete = bool(
            validation["required_checks_passed"] and validation["terminal_validation_passed"]
        )
        terminal = classify_terminal(
            list(evidence.values()),
            affected_complete=bool(validation["required_checks_passed"]),
            terminal_complete=bool(validation["terminal_validation_passed"]),
        )
        expected_rows = task_dispositions(
            contract["tasks"], evidence, terminal, current_validation_complete=current_complete
        )
        branches = build_branch_conclusions(evidence)
        retirements = retirement_decisions(evidence)
        failures = failure_rows(contract, evidence, validation)
        if dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        if artifact.get("branch_conclusions") != branches:
            errors.append("branch_conclusions_invalid")
        if artifact.get("retirement_rows") != retirements:
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != [
            row for row in retirements if row.get("decision") == "defer"
        ]:
            errors.append("retirement_rows_invalid")
        if (
            artifact.get("status") != terminal["status"]
            or artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != _gate_summary(failures)
        ):
            errors.append("terminal_reduction_invalid")
        if artifact.get("acceptance_gate_results") != acceptance_gates(
            contract, expected_rows, branches, validation
        ):
            errors.append("acceptance_gates_invalid")
        if artifact.get("sample_size_budget") != _sample_budget(expected_rows):
            errors.append("sample_size_budget_invalid")
        expected_complete = int(
            contract.get("comparison_passed") is True
            and len(expected_rows) == 14
            and all(row.get("disposition_completed") is True for row in expected_rows)
            and current_complete
        )
        if artifact.get("capstone_complete_score") != expected_complete:
            errors.append("capstone_score_invalid")
        publication = validation["publication_gates"]
        if (
            artifact.get("publication_gates") != publication
            or artifact.get("paper_ready") != publication["paper_ready"]
            or artifact.get("unmet_gates") != publication["unmet_gates"]
        ):
            errors.append("publication_gates_invalid")
        if require_terminal and not validation["terminal_validation_passed"]:
            errors.append("terminal_validation_incomplete")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        errors.append("independent_reduction_failed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }:
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def independent_reduce(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Replay every pure reduction without requiring future reader receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the shared scoped plan for only affected Exp7614 Python files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, public scratch paths, or command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V664 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def root_argument(value: str | Path) -> Path:
    """Require an absolute root that resolves to this exact worktree."""

    supplied = Path(value)
    resolved = supplied.resolve()
    if (
        not supplied.is_absolute()
        or resolved != REPO_ROOT
        or not (resolved / "AGENTS.md").is_file()
    ):
        raise ValueError(f"repository root must resolve absolutely to {REPO_ROOT}")
    return resolved


def _utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return an aware UTC timestamp for one durable boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7614] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - real clock boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:
    """Close one measured phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
    }


def _terminal_commands(  # pragma: no cover - subprocess plan boundary.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build replay, independent reduction, and strict terminal readers."""

    python = str(root / ".venv/bin/python")
    base = (python, "-u", WRAPPER_PATH.as_posix(), "--root", str(root))
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint",
            (*base, "--validate", str(candidate)),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (*base, "--cold-replay", str(candidate)),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (*base, "--independent-reduce", str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def _manifest_payload() -> JsonDict:
    """Freeze the exact affected file set before any validation subprocess."""

    return {
        "experiment_id": VALIDATION_MANIFEST.experiment_id,
        "test_paths": list(VALIDATION_MANIFEST.test_paths),
        "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
        "static_paths": list(VALIDATION_MANIFEST.static_paths),
        "spec_paths": [SPEC_PATH.as_posix()],
        "note_paths": [NOTE_PATH.as_posix(), GAPS_PATH.as_posix()],
        "frozen_before_checks": True,
    }


def run_experiment(  # pragma: no cover - exercised through the declared entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run scoped checks and atomically publish one validated capstone."""

    root = root_argument(root)
    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7614-", dir="/tmp"))

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in INPUT_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    if contract.get("comparison_passed") is not True:
        raise RuntimeError("v664_authorities_do_not_match")
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            completed_units=len(evidence),
            checkpoint="thirteen_upstreams_authenticated",
        )
    )
    progress(started, "preconditions", "after", completed_units=len(evidence))

    phase_started = time.monotonic()
    progress(started, "publication_gate", "before_subprocess")
    publication = evaluate_publication_gates(root)
    if publication.get("exit_code") != 0:
        raise RuntimeError("publication_gate_reader_failed")
    spans.append(
        _span(
            "publication_gate",
            phase_started,
            started,
            completed_units=4,
            checkpoint="g1_g4_read_only",
        )
    )
    progress(
        started, "publication_gate", "after_subprocess", paper_ready=publication["paper_ready"]
    )

    phase_started = time.monotonic()
    progress(started, "plan", "before")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    atomic_json(raw_dir / "affected_validation_manifest.json", _manifest_payload())
    spans.append(
        _span(
            "plan",
            phase_started,
            started,
            completed_units=len(commands),
            checkpoint="affected_validation_manifest_frozen",
        )
    )
    progress(started, "plan", "after", completed_units=len(commands))

    for phase, checkpoint in (
        ("model_load", "no_current_model_load"),
        ("generation", "no_current_generation"),
    ):
        phase_started = time.monotonic()
        progress(started, phase, "before", completed_units=0)
        spans.append(_span(phase, phase_started, started, completed_units=0, checkpoint=checkpoint))
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "validation", "before_subprocesses", completed_units=0)
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=private_root / "logs/affected",
        heartbeat_s=60.0,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    validation: JsonDict = {
        **affected_reduction,
        "required_checks_passed": affected_reduction["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": affected,
        "publication_gates": publication,
        "repository_health": {"status": "outside_current_affected_validity"},
    }
    spans.append(
        _span(
            "validation",
            phase_started,
            started,
            completed_units=len(affected),
            checkpoint="affected_checks_complete",
        )
    )
    progress(
        started,
        "validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    phase_started = time.monotonic()
    progress(started, "reduction", "before", completed_units=0)
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    spans.append(
        _span(
            "reduction",
            phase_started,
            started,
            completed_units=14,
            checkpoint="fourteen_dispositions_and_eight_branches_reduced",
        )
    )
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = private_root / "terminal/candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(candidate_path, candidate)
    progress(started, "reduction", "after", completed_units=14)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal_receipts = run_categorized_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=private_root / "logs/terminal",
        heartbeat_s=60.0,
    )
    validation["terminal_validation_passed"] = all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in terminal_receipts
    )
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    atomic_json(raw_dir / "exact_terminal_validation_receipts.json", terminal_receipts)
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            completed_units=len(terminal_receipts),
            checkpoint="replay_reduction_and_strict_readers_complete",
        )
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=validation["terminal_validation_passed"],
    )

    phase_started = time.monotonic()
    progress(started, "write", "before_atomic", path=output_path.as_posix())
    final_spans = [
        *spans,
        _span(
            "write",
            phase_started,
            started,
            completed_units=1,
            checkpoint="terminal_artifact_ready",
        ),
    ]
    final = build_artifact(
        root,
        contract,
        evidence,
        validation,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(candidate_path, final)
    destination = output_path if output_path.is_absolute() else root / output_path
    atomic_json(destination, final)
    progress(started, "write", "after_atomic", path=destination.as_posix())
    return final


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary.
    """Parse frozen root, date, output, and read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the capstone or one read-only fresh-process replay."""

    print("[exp7614] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    root = root_argument(args.root)
    candidate_path = args.validate or args.cold_replay or args.independent_reduce
    if candidate_path is not None:
        try:
            value = load_json(candidate_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            print(json.dumps({"errors": [str(error)]}, sort_keys=True), flush=True)
            return 1
        errors = (
            independent_reduce(value, root=root)
            if args.independent_reduce is not None
            else validate_artifact(value, root=root, require_terminal=False)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, date_argument(args.date), output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI boundary.
    raise SystemExit(main())
