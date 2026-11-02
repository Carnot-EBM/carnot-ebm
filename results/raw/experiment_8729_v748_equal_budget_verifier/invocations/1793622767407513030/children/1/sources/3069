"""Reconcile fourteen V663 outcomes without merging scientific branches.

This report authenticates existing bytes. It performs no model inference,
hardware operation, roadmap change, or publication.

Spec refs: REQ-REPORT-7600 and SCENARIO-REPORT-7600-*.
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
import shutil
import tempfile
import time
from typing import Any

from carnot import experiment_7586_v662_capstone as prior
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
from carnot.experiment_7587_v663_contract_methods import (
    EXPECTED_TASK_IDS,
    compare_contract_authorities,
    resolve_v663_roadmap,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.663"
EXPERIMENT_ID = "exp7600-capstone"
SCHEMA = "carnot.exp7600.v663.capstone.v1"

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v663-capstone.md")
RESULT_PATH = Path("results/experiment_7600_v663_capstone.json")
RAW_DIR = Path("results/raw/experiment_7600_v663_capstone")
MODULE_PATH = Path("python/carnot/experiment_7600_v663_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7600_v663_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7600_v663_capstone.py")

TERMINAL_CLASSES = prior.TERMINAL_CLASSES
ZERO_INVOCATION_COUNTS = prior.ZERO_INVOCATION_COUNTS
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
PRODUCER_PATHS = {
    task_id: Path(path)
    for task_id, path in (
        ("exp7587-contract-methods", "results/experiment_7587_v663_contract_methods.json"),
        ("exp7588-evidence-protocol", "results/experiment_7588_v663_evidence_protocol.json"),
        ("exp7589-arc-output-boundary", "results/experiment_7589_v663_arc_output_boundary.json"),
        ("exp7590-evidence-pilot", "results/experiment_7590_v663_evidence_pilot.json"),
        ("exp7591-fit-evidence", "results/experiment_7591_v663_fit_evidence.json"),
        ("exp7592-test-online-evidence", "results/experiment_7592_v663_test_online_evidence.json"),
        ("exp7593-evidence-energy", "results/experiment_7593_v663_evidence_energy.json"),
        ("exp7594-decision-evaluation", "results/experiment_7594_v663_decision_evaluation.json"),
        ("exp7595-guarded-learning", "results/experiment_7595_v663_guarded_learning.json"),
        ("exp7596-evidence-audit", "results/experiment_7596_v663_evidence_audit.json"),
        (
            "exp7597-arc-history-generalization",
            "results/experiment_7597_v663_arc_history_generalization.json",
        ),
        ("exp7598-rust-consumer", "results/experiment_7598_v663_rust_consumer.json"),
        ("exp7599-board-continuity", "results/experiment_7599_v663_board_continuity.json"),
    )
}
DIAGNOSTIC_PATHS = {"exp7590-evidence-pilot": Path("results/experiment_7590_evidence_pilot.json")}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7586_v662_capstone.py"),
    Path("python/carnot/experiment_7587_v663_contract_methods.py"),
    Path("scripts/publication_gate.py"),
    Path("scripts/recurring_blocker_ledger.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    DESIGN_PATH,
    SPEC_PATH,
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


load_json = prior.load_json


def load_contract(root: Path) -> JsonDict:
    """Resolve matching V663 authority and compare every contract field."""

    selected, roadmap, candidates = resolve_v663_roadmap(root)
    comparison = compare_contract_authorities((root / DESIGN_PATH).read_text(), roadmap)
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V663 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": deepcopy(candidates),
        "roadmap": deepcopy(roadmap),
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
    op: str = "==",
) -> JsonDict:
    """Name every operand needed to investigate one failed prerequisite."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": False,
    }


def _base_source(task: Mapping[str, Any]) -> JsonDict:
    """Represent an absent producer before any optional diagnostic is read."""

    task_id = str(task.get("id") or "")
    path = str(task.get("deliverable") or "")
    return {
        "task_id": task_id,
        "expected_path": path,
        "artifact_path": None,
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
        "ready_value_fields": {},
        "validation_receipts": [],
        "gate_check_summary": _failed_check(
            "producer_artifact_exists", task_id, path, "path", True, False, op="exists"
        ),
        "payload": {},
    }


def _diagnostic_failure(task_id: str, payload: Mapping[str, Any]) -> JsonDict:
    """Normalize a conductor gate diagnostic without treating it as a run."""

    raw = payload.get("blocked_diagnostic_contract")
    row = raw if isinstance(raw, Mapping) else payload
    failure = _failed_check(
        "conductor_pre_gate",
        task_id,
        row.get("failed_evidence_path"),
        str(row.get("failed_field") or "unknown"),
        row.get("failed_expected"),
        row.get("failed_observed"),
        op=str(row.get("failed_operator") or "=="),
    )
    failure["failed_dependency"] = row.get("failed_upstream")
    return failure


def _producer_failure(task_id: str, path: str, payload: Mapping[str, Any]) -> JsonDict:
    """Name the exact defect in present producer evidence."""

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


def _ready_values(payload: Mapping[str, Any]) -> JsonDict:
    """Keep producer readiness fields without copying scientific rows."""

    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key.endswith("_score") and isinstance(value, (bool, int, float))
    }


def load_producer(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one producer, pre-gate artifact, or explicit absence."""

    task_id = str(task.get("id") or "")
    relative = PRODUCER_PATHS.get(task_id, Path(str(task.get("deliverable") or "")))
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
            artifact_path=relative.as_posix(),
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
                op="is",
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
    evidence_state = "terminal_blocked" if valid and original_class == "blocked" else "terminal"
    if not valid:
        evidence_state = "invalid"
    producer_summary = payload.get("gate_check_summary")
    source.update(
        artifact_path=relative.as_posix(),
        evidence_path=relative.as_posix(),
        source_sha256=sha256_file(path),
        source_size_bytes=path.stat().st_size,
        evidence_state=evidence_state,
        authenticated=authenticated,
        valid=valid,
        honest_verdict=str(payload.get("honest_verdict") or payload.get("status") or "absent"),
        verdict_class=str(original_class) if valid else "disqualified",
        original_verdict_class=original_class,
        flagged_adversarial=payload.get("flagged_adversarial") is True,
        inference_substrate_class=payload.get("inference_substrate_class"),
        model_invoked=payload.get("model_invoked") is True,
        ready_value_fields=_ready_values(payload),
        validation_receipts=deepcopy(payload.get("validation_receipts") or []),
        gate_check_summary=(
            deepcopy(producer_summary)
            if valid and isinstance(producer_summary, Mapping)
            else _producer_failure(task_id, relative.as_posix(), payload)
        ),
        payload=payload,
    )
    return source


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Inventory all thirteen upstream tasks in conductor order."""

    return {str(task.get("id")): load_producer(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, affected_complete: bool, terminal_complete: bool
) -> JsonDict:
    """Apply owned-work precedence before invalid or unavailable evidence."""

    if affected_complete and not terminal_complete:
        verdict, honest = "partial", "partial_retryable_current_capstone_validation_unfinished"
    elif not affected_complete:
        verdict, honest = "disqualified", "complete_disqualified_required_v663_validation"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict, honest = "disqualified", "complete_disqualified_required_v663_evidence"
    elif any(
        row.get("evidence_state") in {"missing", "conductor_gate_blocked", "terminal_blocked"}
        or row.get("verdict_class") == "blocked"
        for row in rows
    ):
        verdict, honest = "blocked", "complete_blocked_required_v663_external_evidence"
    else:
        verdict, honest = "null", "complete_null_v663_accounting_without_aggregate_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _disposition(source: Mapping[str, Any], task: Mapping[str, Any], order: int) -> JsonDict:
    """Reduce one source into a custody row without inventing measurements."""

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
        "artifact_path": source.get("artifact_path"),
        "evidence_path": source.get("evidence_path"),
        "artifact_sha256": source.get("source_sha256"),
        "evidence_state": state,
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("verdict_class"),
        "original_verdict_class": source.get("original_verdict_class"),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
        "inference_substrate_class": source.get("inference_substrate_class"),
        "ready_value_fields": deepcopy(source.get("ready_value_fields") or {}),
        "producer_validation_state": "authenticated" if source.get("authenticated") else state,
        "gate_check_summary": deepcopy(source.get("gate_check_summary")),
        "claim_scope": "descriptive_reuse",
        "excluded_from_scientific_metrics": excluded,
        "disposition_attempted": True,
        "disposition_completed": True,
        "producer_started": state not in {"missing", "conductor_gate_blocked"},
        "producer_completed": state in {"terminal", "terminal_blocked"},
        "producer_failed": state == "invalid",
        "producer_censored": False,
        "producer_unstarted": state in {"missing", "conductor_gate_blocked"},
        "raw_numerator": 1,
        "raw_denominator": 1,
        "metric_direction": "exact_custody_is_required",
        "seed": None,
        "missing": state in {"missing", "conductor_gate_blocked"},
        "censored": state == "missing",
        "provenance": source.get("evidence_path") or source.get("expected_path"),
        "principle": "Exact custody keeps absent, invalid, and blocked evidence distinct.",
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    *,
    current_validation_complete: bool,
) -> list[JsonDict]:
    """Build thirteen source rows and one current row without self-reading."""

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
            "artifact_path": None,
            "evidence_path": None,
            "artifact_sha256": None,
            "evidence_state": "current_terminal"
            if current_validation_complete
            else "current_running",
            "honest_verdict": terminal.get("honest_verdict"),
            "verdict_class": terminal.get("verdict_class"),
            "original_verdict_class": terminal.get("verdict_class"),
            "flagged_adversarial": False,
            "inference_substrate_class": "aggregation",
            "ready_value_fields": {"capstone_complete_score": int(current_validation_complete)},
            "producer_validation_state": "current_owned_validation",
            "gate_check_summary": {},
            "claim_scope": "current_accounting",
            "excluded_from_scientific_metrics": True,
            "disposition_attempted": True,
            "disposition_completed": current_validation_complete,
            "producer_started": True,
            "producer_completed": current_validation_complete,
            "producer_failed": False,
            "producer_censored": False,
            "producer_unstarted": False,
            "raw_numerator": int(current_validation_complete),
            "raw_denominator": 1,
            "metric_direction": "owned_validation_complete_is_required",
            "seed": None,
            "missing": not current_validation_complete,
            "censored": not current_validation_complete,
            "provenance": "current_exp7600_validation_receipts",
            "principle": "Current work cannot authenticate itself through its future result path.",
        }
    )
    return rows


def _source_ref(source: Mapping[str, Any]) -> JsonDict:
    """Bind one branch input without copying its measurement rows."""

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


def _audit_branch(
    audit: Mapping[str, Any], names: Sequence[str], branch: str, conclusion: str
) -> JsonDict:
    """Reduce named Exp7596 findings without consulting blocked producers."""

    payload = audit.get("payload") or {}
    rows = payload.get("branch_conclusions") or []
    selected = [
        deepcopy(dict(row))
        for row in rows
        if isinstance(row, Mapping) and row.get("branch") in names
    ]
    blocked = not selected or any(
        str(row.get("validity", "")).startswith("blocked") for row in selected
    )
    return {
        "branch": branch,
        "source_task": "exp7596-evidence-audit",
        "sources": [_source_ref(audit)],
        "validity": "blocked_missing_capture" if blocked else "valid",
        "readiness": False if blocked else all(row.get("readiness") is True for row in selected),
        "benefit": "not_measured"
        if blocked
        else all(row.get("benefit") is True for row in selected),
        "retention": "not_measured" if blocked else "not_applicable",
        "freshness": "not_assessed" if blocked else False,
        "verdict_class": "blocked" if blocked else "null",
        "claim_scope": "descriptive_reuse",
        "independent_findings": selected,
        "conclusion": conclusion,
    }


def build_branch_conclusions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reduce eight conclusions while keeping readiness and benefit separate."""

    audit = evidence["exp7596-evidence-audit"]
    arc = evidence["exp7597-arc-history-generalization"]
    consumer = evidence["exp7598-rust-consumer"]
    boards = evidence["exp7599-board-continuity"]
    arc_payload = arc.get("payload") or {}
    consumer_payload = consumer.get("payload") or {}
    board_payload = boards.get("payload") or {}
    consumer_reduction = consumer_payload.get("independent_reduction") or {}
    comparison = consumer_reduction.get("comparison_summary") or {}
    return [
        _audit_branch(
            audit,
            ("incremental_evidence", "probability"),
            "evidence_gain_and_probability",
            "Required source groups were not captured, so gain and probability remain unmeasured.",
        ),
        _audit_branch(
            audit,
            ("decision_cost",),
            "decision_quality",
            "Required decision rows are absent, so decision quality remains unmeasured.",
        ),
        _audit_branch(
            audit,
            ("causal_update_behavior",),
            "delayed_learning",
            "The guarded learner did not run, so no causal update benefit was measured.",
        ),
        _audit_branch(
            audit,
            ("retention",),
            "retention",
            "The evaluator-retention branch did not run and cannot support a safety claim.",
        ),
        {
            "branch": "arc_observability",
            "source_task": "exp7597-arc-history-generalization",
            "sources": [_source_ref(arc)],
            "validity": bool(
                arc.get("valid") and arc_payload.get("arc_history_measurement_ready_score") == 1
            ),
            "readiness": arc_payload.get("arc_history_measurement_ready_score") == 1,
            "benefit": False,
            "retention": "not_applicable",
            "freshness": False,
            "verdict_class": str(arc_payload.get("verdict_class") or "null"),
            "claim_scope": "descriptive_reuse",
            "observed_independent_units": (arc_payload.get("sample_size_budget") or {}).get(
                "observed_independent_units"
            ),
            "history_support_ready_score": arc_payload.get("history_support_ready_score"),
            "support_versus_conflict_curve": deepcopy(
                arc_payload.get("support_versus_conflict_curve") or []
            ),
            "independent_reduction": deepcopy(arc_payload.get("independent_reduction") or {}),
            "next_model_interface_finding": (
                "Observable histories remove conflicts on repeated keys, but repeated-key support "
                "is too sparse for a safe acceptance-gate claim."
            ),
            "new_solve_claimed": False,
            "near_exact_gate_safety_claimed": False,
            "conclusion": "History support is a next-model interface finding, not a solve.",
        },
        {
            "branch": "rust_consumer_readiness",
            "source_task": "exp7598-rust-consumer",
            "sources": [_source_ref(consumer)],
            "validity": bool(consumer.get("valid")),
            "readiness": consumer_reduction.get("consumer_ready_score") == 1,
            "benefit": "not_applicable",
            "retention": "not_applicable",
            "freshness": False,
            "verdict_class": "null",
            "claim_scope": "descriptive_reuse",
            "opt_in": True,
            "subprocess_service_integration": deepcopy(
                consumer_payload.get("e2e_applicability") or {}
            ),
            "conclusion": "The typed Rust consumer is ready and remains opt-in.",
        },
        {
            "branch": "rust_consumer_speed",
            "source_task": "exp7598-rust-consumer",
            "sources": [_source_ref(consumer)],
            "validity": bool(
                consumer.get("valid") and consumer_reduction.get("measurement_complete")
            ),
            "readiness": consumer_reduction.get("consumer_ready_score") == 1,
            "benefit": consumer_reduction.get("consumer_speed_benefit_score") == 1,
            "retention": "not_applicable",
            "freshness": False,
            "verdict_class": "positive"
            if consumer_reduction.get("consumer_speed_benefit_score") == 1
            else "null",
            "claim_scope": "descriptive_reuse",
            "comparison_summary": deepcopy(comparison),
            "independent_pair_count": sum(
                int((row or {}).get("pair_count") or 0)
                for row in comparison.values()
                if isinstance(row, Mapping)
            ),
            "hardware_speed_claimed": False,
            "conclusion": "Consumer parity passed, but the registered aggregate speed gate failed.",
        },
        {
            "branch": "board_continuity",
            "source_task": "exp7599-board-continuity",
            "sources": [_source_ref(boards)],
            "validity": bool(boards.get("valid")),
            "readiness": board_payload.get("board_continuity_complete_score") == 1,
            "benefit": "not_measured",
            "retention": "not_applicable",
            "freshness": False,
            "verdict_class": "null",
            "claim_scope": "descriptive_reuse",
            "board_rows": deepcopy(board_payload.get("board_rows") or []),
            "placement_scope": board_payload.get("placement_scope"),
            "amdahl_upper_bound": deepcopy(board_payload.get("amdahl_upper_bound")),
            "host_speed_reported_as_board_or_tsu_speed": False,
            "conclusion": "Three board dispositions remain separate; placement is unmeasured.",
        },
    ]


def retirement_decisions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Retire only measured constructions and preserve external hypotheses."""

    prior_row = {
        "experiment_id": "exp7586-capstone",
        "verdict": "complete_blocked_required_v662_external_evidence",
        "retire_if_same_verdict": True,
    }
    current = "complete_blocked_required_v663_external_evidence"
    return [
        {
            "branch": "evidence_link_feature_construction",
            "decision": "defer",
            "scope": "v663_exact_eight_feature_sentence_link_construction",
            "independent_benefit_measured": False,
            "resource_or_transport_blocker": True,
            "scientific_hypothesis_retired": False,
            "reopening_condition": "Complete independent gain evaluation from new eligible rows.",
            "fresh_corpus_reopening_condition": (
                "A genuinely unexposed inventory and new preregistration, not a reshuffle."
            ),
            "forbidden_continuation": "another penalty, seed, or capture with unchanged inputs",
            "source": _source_ref(evidence["exp7596-evidence-audit"]),
        },
        {
            "branch": "guarded_update_construction",
            "decision": "defer",
            "scope": "v663_delayed_evidence_guarded_update_construction",
            "independent_benefit_measured": False,
            "retention_measured": False,
            "resource_or_transport_blocker": True,
            "scientific_hypothesis_retired": False,
            "reopening_condition": "Complete benefit and evaluator-retention evaluation.",
            "fresh_corpus_reopening_condition": (
                "A genuinely unexposed inventory and new preregistration, not a reshuffle."
            ),
            "source": _source_ref(evidence["exp7596-evidence-audit"]),
        },
        {
            "branch": "prior_capstone_verdict",
            "decision": "retire_narrow_repeated_capstone_scope",
            "scope": "aggregate_capstone_missing_external_evidence_disposition",
            "literal_prior_verdict": prior_row["verdict"],
            "literal_current_verdict": current,
            "literal_exact_match": False,
            "normalized_scope_match": True,
            "retire_if_same_verdict": prior_row["retire_if_same_verdict"],
            "scientific_hypothesis_retired": False,
            "resource_or_transport_blocker": True,
            "reopening_condition": "Required external producers become authenticated and eligible.",
        },
    ]


def collect_preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record source custody, requirement, authority, host, and producer states."""

    rows: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "op": "is",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else "absent_or_empty",
                "passed": available,
                "sha256": sha256_file(path) if available else None,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "path": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "op": "contains",
            "expected": "REQ-REPORT-7600",
            "observed": "REQ-REPORT-7600" if "REQ-REPORT-7600" in spec_text else None,
            "passed": "REQ-REPORT-7600" in spec_text,
        }
    )
    rows.append(
        {
            "check": "roadmap_authority",
            "upstream": "V663 authority resolution",
            "path": contract.get("selected_roadmap_path"),
            "field": "comparison_passed",
            "op": "==",
            "expected": True,
            "observed": contract.get("comparison_passed"),
            "passed": contract.get("comparison_passed") is True,
        }
    )
    disk = shutil.disk_usage(root)
    rows.append(
        {
            "check": "aggregation_resource",
            "upstream": "current_host",
            "path": str(root),
            "field": "cpu_and_storage",
            "op": "satisfies",
            "expected": "cpu_count>0 and free_bytes>0",
            "observed": {"cpu_count": os.cpu_count(), "free_bytes": disk.free},
            "passed": bool((os.cpu_count() or 0) > 0 and disk.free > 0),
        }
    )
    for task_id, source in evidence.items():
        rows.append(
            {
                "check": f"producer_state:{task_id}",
                "upstream": task_id,
                "path": source.get("evidence_path") or source.get("expected_path"),
                "field": "evidence_state",
                "op": "observed",
                "expected": "explicit observed state",
                "observed": source.get("evidence_state"),
                "passed": True,
                "sha256": source.get("source_sha256"),
            }
        )
    return rows


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Bind current code, authority, and every producer or diagnostic byte."""

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
    """Recheck each source row that claims existing exact bytes."""

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
    op: str = "==",
) -> JsonDict:
    """Keep one acceptance operand and its failure-prevention principle."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def acceptance_gates(
    contract: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep validity, readiness, benefit, retention, and freshness separate."""

    by_branch = {str(row.get("branch")): row for row in branches}
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    evidence = by_branch["evidence_gain_and_probability"]
    learning = by_branch["delayed_learning"]
    retention = by_branch["retention"]
    arc = by_branch["arc_observability"]
    speed = by_branch["rust_consumer_speed"]
    boards = by_branch["board_continuity"]
    validity = "Invalid or incomplete evidence cannot support science."
    readiness = "Accounting or interface readiness does not establish benefit."
    benefit = "One branch cannot promote another branch's effect claim."
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v663_contract",
            str(contract.get("selected_roadmap_path")),
            "comparison_passed",
            True,
            contract.get("comparison_passed"),
            contract.get("comparison_passed") is True,
            validity,
        ),
        _gate(
            "current_scoped_and_terminal_validation",
            "validity",
            EXPERIMENT_ID,
            "validation_receipts",
            "all_required_passed",
            True,
            current_valid,
            current_valid,
            validity,
        ),
        _gate(
            "fourteen_honest_dispositions",
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
            "independent_evidence_gain_available",
            "validity",
            "exp7596-evidence-audit",
            "branch_conclusions.evidence_gain_and_probability",
            "validity",
            "valid",
            evidence.get("validity"),
            evidence.get("validity") == "valid",
            validity,
        ),
        _gate(
            "evidence_incremental_benefit",
            "benefit",
            "exp7596-evidence-audit",
            "branch_conclusions.evidence_gain_and_probability",
            "benefit",
            True,
            evidence.get("benefit"),
            evidence.get("benefit") is True,
            benefit,
        ),
        _gate(
            "guarded_learning_benefit",
            "benefit",
            "exp7596-evidence-audit",
            "branch_conclusions.delayed_learning",
            "benefit",
            True,
            learning.get("benefit"),
            learning.get("benefit") is True,
            benefit,
        ),
        _gate(
            "evaluator_retention",
            "retention",
            "exp7596-evidence-audit",
            "branch_conclusions.retention",
            "retention",
            True,
            retention.get("retention"),
            retention.get("retention") is True,
            benefit,
        ),
        _gate(
            "arc_history_measurement_ready",
            "readiness",
            "exp7597-arc-history-generalization",
            "arc_history_measurement_ready_score",
            "readiness",
            True,
            arc.get("readiness"),
            arc.get("readiness") is True,
            readiness,
        ),
        _gate(
            "rust_consumer_speed_benefit",
            "benefit",
            "exp7598-rust-consumer",
            "consumer_speed_benefit_score",
            "benefit",
            True,
            speed.get("benefit"),
            speed.get("benefit") is True,
            benefit,
        ),
        _gate(
            "board_continuity_ready",
            "readiness",
            "exp7599-board-continuity",
            "board_continuity_complete_score",
            "readiness",
            True,
            boards.get("readiness"),
            boards.get("readiness") is True,
            readiness,
        ),
        _gate(
            "fresh_confirmatory_claim_allowed",
            "freshness",
            EXPERIMENT_ID,
            "branch_conclusions",
            "source_data_claim_scope",
            "descriptive_reuse",
            "descriptive_reuse",
            True,
            "Historical exposure stays visible and cannot become fresh confirmation.",
        ),
    ]


def failure_rows(
    contract: Mapping[str, Any], evidence: Mapping[str, JsonDict], validation: Mapping[str, Any]
) -> list[JsonDict]:
    """List only failures that determine the aggregate terminal class."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            _failed_check(
                "contract_authorities_agree",
                "v663_contract",
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
            failures.append(
                deepcopy(dict(summary))
                if isinstance(summary, Mapping)
                else _failed_check(
                    "producer_state",
                    task_id,
                    source.get("expected_path"),
                    "state",
                    "terminal",
                    state,
                )
            )
        elif state == "terminal_blocked" or source.get("verdict_class") == "blocked":
            first = summary.get("first_failure") if isinstance(summary, Mapping) else None
            if isinstance(first, Mapping):
                nested = deepcopy(dict(first))
                nested["failed_dependency"] = nested.get("upstream")
                nested["upstream"] = task_id
                failures.append(nested)
            else:
                failures.append(
                    _failed_check(
                        "producer_terminal_not_blocked",
                        task_id,
                        source.get("evidence_path"),
                        "verdict_class",
                        "not blocked",
                        "blocked",
                        op="!=",
                    )
                )
    return failures


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain each terminal failure and its first exact cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


FIELD_PRINCIPLES = {
    "honest_verdict": "A complete prefix records terminal execution without implying benefit.",
    "verdict_class": "The closed class reserves partial for unfinished owned work.",
    "flagged_adversarial": "Flagged evidence cannot open a gate.",
    "gate_check_summary": "Every block retains exact upstream operands.",
    "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
    "rows": "One row per task keeps absence distinct from zero.",
    "sample_size_budget": "Independent units exclude multiplication by seeds or windows.",
    "inference_substrate": "The actual aggregation mode cannot imply current inference.",
    "inference_substrate_class": "Actual and planned classes stay separate.",
    "MODEL_SPECS": "An empty list prevents historical models from becoming current calls.",
    "invocation_counts": "Typed zeros expose invented loads, forwards, generations, or tokens.",
    "duration_s": "Monotonic current time excludes inherited work and artificial sleep.",
    "random_seed": "Every stochastic stage has an explicit seed or an explicit absence.",
    "reproducibility_checksum": "Immutable evidence and terminal reduction determine identity.",
    "source_artifact_hashes": "Exact byte hashes bind producer, diagnostic, and missing states.",
    "validation_receipts": "Commands, exits, worktree paths, and log hashes bind validation.",
    "verifier_is_oracle": "Exact labels and hand-built controls cannot support an oracle-distinct claim.",
    "field_principles": "Every top-level field states the inference failure it prevents.",
    "task_dispositions": "Exactly fourteen rows retain the contract order and custody state.",
    "branch_conclusions": "Independent branches cannot promote one another.",
    "retirement_rows": "Only a measured construction closes for a concrete cause.",
    "publication_gates": "Stable G1-G4 remain unchanged and do not publish anything.",
    "submitted_externally": "False preserves operator-only publication authority.",
}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give every top-level field one plain failure-prevention reason."""

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
        "device_compute",
        "duration_components_s",
    }
    return canonical_hash({key: item for key, item in value.items() if key not in excluded})


def _sample_budget(dispositions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate capstone accounting from underlying producer execution."""

    upstream = dispositions[:-1]
    return {
        "independent_unit": "ordered_v663_task_disposition",
        "planned": 14,
        "attempted": sum(int(row.get("disposition_attempted") is True) for row in dispositions),
        "observed": sum(int(row.get("disposition_completed") is True) for row in dispositions),
        "completed": sum(int(row.get("disposition_completed") is True) for row in dispositions),
        "excluded": sum(
            int(row.get("excluded_from_scientific_metrics") is True) for row in dispositions
        ),
        "failed": sum(int(row.get("producer_failed") is True) for row in dispositions),
        "censored": sum(int(row.get("producer_censored") is True) for row in dispositions),
        "unstarted": sum(int(row.get("producer_unstarted") is True) for row in dispositions),
        "seeds_and_windows_multiply_source_groups": False,
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
    """Return deterministic phase rows for pure test construction."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "completed_units": units,
            "checkpoint": checkpoint,
        }
        for phase, units, checkpoint in (
            ("preconditions", 13, "thirteen_upstreams_observed"),
            ("publication_gate", 4, "g1_g4_read_only"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("reduction", 14, "fourteen_dispositions_reduced"),
            ("validation", 8, "affected_checks_complete"),
            ("terminal_validation", 5, "cold_readers_complete"),
            ("write", 1, "terminal_artifact_ready"),
        )
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
    """Build one terminal record from authenticated V663 evidence."""

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
    validation_s = sum(
        float(row.get("duration_s") or 0.0) for row in receipts if isinstance(row, Mapping)
    )
    historical_s = sum(
        float((source.get("payload") or {}).get("duration_s") or 0.0)
        for source in evidence.values()
    )
    complete_score = int(
        contract.get("comparison_passed") is True
        and len(dispositions) == 14
        and all(row.get("disposition_completed") is True for row in dispositions)
        and current_complete
    )
    publication = evaluate_publication_gates(root)
    current_task = contract["tasks"][-1]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7600,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "positive_claim": False,
        "flagged_adversarial": False,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "phase_spans": deepcopy(list(phase_spans)),
        "process_identity": {
            "pid": os.getpid(),
            "node": platform.node(),
            "source_revision": prior._source_revision(root),
        },
        "device_compute": {
            "execution_venue": "host",
            "device_identity": platform.processor() or "CPU",
            "machine": platform.machine(),
            "cuda_used": False,
            "model_compute_used": False,
        },
        "duration_components_s": {
            "aggregation": max(0.0, duration_s - validation_s),
            "validation": validation_s,
            "historical_upstream": historical_s,
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
        },
        "random_seed": {
            "selection": None,
            "fitting": None,
            "ordering": None,
            "bootstrap": None,
            "audit": 7_600_663_01,
            "historical_arc": 7_597_003,
            "historical_consumer_bootstrap": [7_598_002, 7_598_009, 7_598_102, 7_598_109],
            "explanation": "Current deterministic aggregation makes no outcome-dependent draw.",
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
            "actual_class": "aggregation",
            "planned_class": "aggregation",
            "readout_kind": "upstream_artifact_reduction",
            "generated_tokens": 0,
            "blocked_no_run_means_zero_model_work": True,
        },
        "execution_venue": "host",
        "preconditions_checked": collect_preconditions(root, contract, evidence),
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_model_receipts": [
            {
                "task_id": task_id,
                "path": source.get("artifact_path"),
                "sha256": source.get("source_sha256"),
                "historical_substrate_class": source.get("inference_substrate_class"),
                "historical_model_invoked": source.get("model_invoked") is True,
                "counted_as_current_invocation": False,
            }
            for task_id, source in evidence.items()
            if source.get("artifact_path")
        ],
        "rows": dispositions,
        "task_dispositions": dispositions,
        "sample_size_budget": _sample_budget(dispositions),
        "comparative_row_reduction": {
            "row_count": len(dispositions),
            "sign_counts": {"positive": 0, "negative": 0, "zero": 0, "missing": 14},
            "censored_count": sum(int(row.get("censored") is True) for row in dispositions),
            "principle": "Custody rows have no scientific effect sign; missing is explicit.",
        },
        "branch_conclusions": branches,
        "branch_dispositions": branches,
        "prior_failure_rows": deepcopy(current_task.get("prior_failures") or []),
        "retirement_decisions": retirements,
        "retirement_rows": retirements,
        "continuation_rows": [row for row in retirements if row.get("decision") == "defer"],
        "acceptance_gate_results": acceptance_gates(contract, dispositions, branches, validation),
        "gate_check_summary": _gate_summary(failures),
        "capstone_complete_score": complete_score,
        "publication_gates": publication,
        "publication_gate_results": deepcopy(publication),
        "paper_ready": publication["paper_ready"],
        "unmet_gates": deepcopy(publication["unmet_gates"]),
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "spec_paths": [SPEC_PATH.as_posix()],
            "note_paths": [NOTE_PATH.as_posix()],
            "frozen_before_checks": True,
        },
        "validation_receipts": receipts,
        "repository_health": deepcopy(
            validation.get("repository_health") or {"status": "outside_current_affected_validity"}
        ),
        "capability_e2e": {
            "declared_entrypoint": "required",
            "fresh_process_cold_replay": "required",
            "independent_reduction": "required",
            "learning_lifecycle": {
                "current_execution_required": False,
                "upstream_task": "exp7598-rust-consumer",
                "exercise": "predict-release-update-persist-reload-with-duplicate-rejection",
                "completed": True,
            },
            "arc_runtime": {
                "current_execution_required": False,
                "upstream_task": "exp7597-arc-history-generalization",
                "checks": ["E2E-009", "E2E-010", "E2E-011", "E2E-012", "E2E-013"],
                "private_llm_off_smoke": True,
                "immutable_evidence_path_rejection_preserved": True,
            },
            "bindings": {
                "E2E-003": "not_applicable_no_pyo3_binding",
                "E2E-004": "not_applicable_no_safetensors_cross_language_serialization",
            },
            "numbered_runtime_e2e": "not_applicable_read_only_reporting",
        },
        "verifier_is_oracle": True,
        "oracle_distinct_positive_claimed": False,
        "fresh_confirmatory_claim_allowed": False,
        "roadmap_activation_performed": False,
        "roadmap_archive_performed": False,
        "publication_performed": False,
        "submission_performed": False,
        "submitted_externally": False,
        "external_contact_performed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "push_performed": False,
        "research_conductor_modified": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic terminal artifact from current repository evidence."""

    contract = load_contract(REPO_ROOT)
    evidence = collect_evidence(REPO_ROOT, contract["tasks"])
    return build_artifact(
        REPO_ROOT,
        contract,
        evidence,
        {
            "required_checks_passed": True,
            "terminal_validation_passed": True,
            "validation_receipts": _passing_receipts(),
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
    """Cold-check identity, sources, reductions, principles, and checksum."""

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
        or artifact.get("oracle_distinct_positive_claimed") is not False
        or artifact.get("submitted_externally") is not False
    ):
        errors.append("current_claim_contract_invalid")
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
        if (
            artifact.get("branch_conclusions") != branches
            or artifact.get("branch_dispositions") != branches
        ):
            errors.append("branch_conclusions_invalid")
        if (
            artifact.get("retirement_decisions") != retirements
            or artifact.get("retirement_rows") != retirements
        ):
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != [
            row for row in retirements if row.get("decision") == "defer"
        ]:
            errors.append("continuation_rows_invalid")
        if artifact.get("prior_failure_rows") != deepcopy(
            contract["tasks"][-1].get("prior_failures") or []
        ):
            errors.append("prior_failure_rows_invalid")
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
        if (
            type(artifact.get("capstone_complete_score")) is not int
            or artifact.get("capstone_complete_score") != expected_complete
        ):
            errors.append("capstone_score_invalid")
        publication = evaluate_publication_gates(root)
        if (
            artifact.get("publication_gates") != publication
            or artifact.get("publication_gate_results") != publication
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
    """Replay every pure reduction without requiring final terminal receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the shared scoped plan for only affected Exp7600 files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, or command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V663 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def root_argument(value: str | Path) -> Path:
    """Require an absolute repository root that resolves to this worktree."""

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
    """Return one aware UTC timestamp for a durable runtime boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed phase line with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7600] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
    """Build entrypoint, cold replay, independent reduction, and strict guards."""

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


def run_experiment(  # pragma: no cover - exercised through declared entrypoint.
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
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7600-", dir="/tmp"))

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in INPUT_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    preconditions = collect_preconditions(root, contract, evidence)
    required = [
        row
        for row in preconditions
        if str(row.get("check")).startswith("source_bytes:")
        or row.get("check") in {"driving_requirement", "roadmap_authority", "aggregation_resource"}
    ]
    if any(row.get("passed") is not True for row in required):
        raise RuntimeError("required_capstone_precondition_failed")
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            completed_units=len(evidence),
            checkpoint="thirteen_upstreams_observed",
        )
    )
    progress(started, "preconditions", "after", completed_units=len(evidence))

    phase_started = time.monotonic()
    progress(started, "publication_gate", "before_subprocess")
    publication = evaluate_publication_gates(root)
    if publication["exit_code"] != 0:
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
    atomic_json(
        raw_dir / "affected_validation_manifest.json",
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "spec_paths": [SPEC_PATH.as_posix()],
            "note_paths": [NOTE_PATH.as_posix()],
            "frozen_before_checks": True,
        },
    )
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
            checkpoint="entrypoint_cold_reduction_and_strict_guards_complete",
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

    print("[exp7600] phase=startup event=flushed", flush=True)
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
