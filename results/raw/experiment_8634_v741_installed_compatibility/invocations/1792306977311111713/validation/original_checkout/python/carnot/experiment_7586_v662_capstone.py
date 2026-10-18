"""Reconcile fourteen V662 outcomes without merging scientific branches.

This report authenticates existing bytes. It performs no model inference,
hardware operation, roadmap change, or publication.

Spec refs: REQ-REPORT-7586 and SCENARIO-REPORT-7586-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7572_v661_capstone import (
    _ready_values,
    _receipt_rows,
    _receipt_set_passed,
    evaluate_publication_gates,
)
from carnot.experiment_7573_v662_contract_methods import (
    EXPECTED_TASK_IDS,
    compare_contract_authorities,
    resolve_v662_roadmap,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7586-capstone"
SCHEMA = "carnot.exp7586.v662.capstone.v1"

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7586_v662_capstone.json")
RAW_DIR = Path("results/raw/experiment_7586_v662_capstone")
MODULE_PATH = Path("python/carnot/experiment_7586_v662_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7586_v662_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7586_v662_capstone.py")

TERMINAL_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls", "tokens")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
PRODUCER_PATHS = {
    task_id: Path(f"results/experiment_{number}_v662_{slug.replace('-', '_')}.json")
    for task_id, number, slug in (
        ("exp7573-contract-methods", 7573, "contract-methods"),
        ("exp7574-measurement-requalification", 7574, "measurement-requalification"),
        ("exp7575-cached-learning-protocol", 7575, "cached-learning-protocol"),
        ("exp7576-proper-loss-energy", 7576, "proper-loss-energy"),
        ("exp7577-proper-loss-evaluation", 7577, "proper-loss-evaluation"),
        ("exp7578-continuous-proper-loss", 7578, "continuous-proper-loss"),
        ("exp7579-decision-learning-audit", 7579, "decision-learning-audit"),
        ("exp7580-arc-verifier-support", 7580, "arc-verifier-support"),
        ("exp7581-arc-bounded-canary", 7581, "arc-bounded-canary"),
        ("exp7582-arc-panel-a", 7582, "arc-panel-a"),
        ("exp7583-arc-panel-b", 7583, "arc-panel-b"),
        ("exp7584-arc-independent-audit", 7584, "arc-independent-audit"),
        ("exp7585-portable-service", 7585, "portable-service"),
    )
}
DIAGNOSTIC_PATHS = {"exp7582-arc-panel-a": Path("results/experiment_7582_arc_panel_a.json")}
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
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7572_v661_capstone.py"),
    Path("scripts/publication_gate.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("ops/known-issues.md"),
    Path("ops/verifier_gaps.md"),
    DESIGN_PATH,
    SPEC_PATH,
)
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def load_json(path: Path) -> JsonDict:
    """Read one JSON mapping so malformed evidence fails closed."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_contract(root: Path) -> JsonDict:
    """Resolve the matching V662 authority and compare all contract fields."""

    selected, roadmap, candidates = resolve_v662_roadmap(root)
    comparison = compare_contract_authorities((root / DESIGN_PATH).read_text(), roadmap)
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):  # pragma: no cover - guarded by the authority reader.
        raise ValueError("active V662 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": deepcopy(candidates),
        "roadmap": deepcopy(roadmap),
        "tasks": deepcopy(tasks),
    }


def _base_source(task: Mapping[str, Any]) -> JsonDict:
    """Create one explicit absent-producer row before reading source bytes."""

    task_id = str(task.get("id") or "")
    expected = str(task.get("deliverable") or "")
    return {
        "task_id": task_id,
        "expected_path": expected,
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
        "gate_check_summary": {
            "check": "producer_artifact_exists",
            "upstream": task_id,
            "path": expected,
            "field": "path",
            "op": "exists",
            "expected": True,
            "observed": False,
            "passed": False,
        },
        "payload": {},
    }


def _diagnostic_failure(task_id: str, payload: Mapping[str, Any]) -> JsonDict:
    """Normalize a conductor pre-gate artifact without treating it as a run."""

    raw = payload.get("blocked_diagnostic_contract")
    row = raw if isinstance(raw, Mapping) else payload
    return {
        "check": "conductor_pre_gate",
        "upstream": task_id,
        "path": row.get("failed_evidence_path"),
        "field": row.get("failed_field"),
        "op": row.get("failed_operator"),
        "expected": deepcopy(row.get("failed_expected")),
        "observed": deepcopy(row.get("failed_observed")),
        "passed": False,
    }


def _producer_failure(task_id: str, path: str, payload: Mapping[str, Any]) -> JsonDict:
    """Name the exact present-evidence defect that disqualifies a producer."""

    if payload.get("flagged_adversarial") is True:
        field, expected, observed = "flagged_adversarial", False, True
    elif payload.get("verdict_class") == "disqualified":
        field, expected, observed = "verdict_class", "not disqualified", "disqualified"
    else:
        field, expected, observed = "required_validation", True, False
    return {
        "check": "producer_required_validity",
        "upstream": task_id,
        "path": path,
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": False,
    }


def _producer_receipts_pass(payload: Mapping[str, Any]) -> bool:
    """Require artifact checks while allowing an honestly failed science E2E."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list):
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    by_name = {str(row.get("name")): row for row in rows}
    integrity_names = set(REQUIRED_CHECK_NAMES)
    if not integrity_names.issubset(by_name):
        return False
    required_rows = [by_name[name] for name in integrity_names]
    for optional in ("adversarial_verify", "verdict_row_consistency_strict"):
        if optional in by_name:
            required_rows.append(by_name[optional])
    return all(row.get("passed") is True and row.get("exit_code") == 0 for row in required_rows)


def load_producer(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one producer or its matching conductor diagnostic."""

    task_id = str(task.get("id") or "")
    relative = PRODUCER_PATHS.get(task_id, Path(str(task.get("deliverable") or "")))
    path = root / relative
    row = _base_source(task)
    if not path.is_file():
        diagnostic_relative = DIAGNOSTIC_PATHS.get(task_id)
        diagnostic = root / diagnostic_relative if diagnostic_relative else None
        if diagnostic is None or not diagnostic.is_file():
            return row
        payload = load_json(diagnostic)
        row.update(
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
        return row
    try:
        payload = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        row.update(
            artifact_path=relative.as_posix(),
            evidence_path=relative.as_posix(),
            source_sha256=sha256_file(path),
            source_size_bytes=path.stat().st_size,
            evidence_state="invalid",
            honest_verdict="complete_disqualified_unreadable_producer_evidence",
            verdict_class="disqualified",
            gate_check_summary={
                "check": "producer_json_object",
                "upstream": task_id,
                "path": relative.as_posix(),
                "field": "json_object",
                "op": "is",
                "expected": "mapping",
                "observed": str(error),
                "passed": False,
            },
        )
        return row

    expected_number = int(task_id[3:7])
    observed_id = payload.get("experiment", payload.get("experiment_id"))
    original_class = payload.get("verdict_class")
    authenticated = bool(
        str(expected_number) in str(observed_id)
        and payload.get("milestone") == MILESTONE
        and original_class in TERMINAL_CLASSES
        and isinstance(payload.get("flagged_adversarial"), bool)
    )
    valid = bool(
        authenticated
        and _producer_receipts_pass(payload)
        and original_class != "disqualified"
        and payload.get("flagged_adversarial") is False
    )
    row.update(
        artifact_path=relative.as_posix(),
        evidence_path=relative.as_posix(),
        source_sha256=sha256_file(path),
        source_size_bytes=path.stat().st_size,
        evidence_state="terminal" if valid else "invalid",
        authenticated=authenticated,
        valid=valid,
        honest_verdict=str(payload.get("honest_verdict") or payload.get("status") or "absent"),
        verdict_class=str(original_class) if valid else "disqualified",
        original_verdict_class=original_class,
        flagged_adversarial=payload.get("flagged_adversarial") is True,
        inference_substrate_class=payload.get("inference_substrate_class"),
        model_invoked=payload.get("model_invoked") is True,
        ready_value_fields=_ready_values(payload),
        validation_receipts=_receipt_rows(payload),
        gate_check_summary=(
            deepcopy(payload.get("gate_check_summary"))
            if valid
            else _producer_failure(task_id, relative.as_posix(), payload)
        ),
        payload=payload,
    )
    return row


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Inventory all thirteen upstream tasks in conductor order."""

    return {str(task.get("id")): load_producer(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, affected_complete: bool, terminal_complete: bool
) -> JsonDict:
    """Apply owned-work precedence before upstream invalidity or absence."""

    if affected_complete and not terminal_complete:
        verdict, honest = "partial", "partial_retryable_current_capstone_validation_unfinished"
    elif not affected_complete:
        verdict, honest = "disqualified", "complete_disqualified_required_v662_capstone_validation"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict, honest = "disqualified", "complete_disqualified_required_v662_evidence"
    elif any(
        row.get("evidence_state") in {"missing", "conductor_gate_blocked"}
        or row.get("verdict_class") == "blocked"
        for row in rows
    ):
        verdict, honest = "blocked", "complete_blocked_required_v662_external_evidence"
    else:
        verdict, honest = "null", "complete_null_v662_accounting_without_aggregate_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _disposition(source: Mapping[str, Any], task: Mapping[str, Any], order: int) -> JsonDict:
    """Reduce one upstream source into one allowed task disposition."""

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
        "validation_receipts": deepcopy(source.get("validation_receipts") or []),
        "gate_check_summary": deepcopy(source.get("gate_check_summary")),
        "claim_scope": "descriptive_reuse",
        "excluded_from_scientific_metrics": excluded,
        "disposition_attempted": True,
        "disposition_completed": True,
        "producer_started": state not in {"missing", "conductor_gate_blocked"},
        "producer_completed": state == "terminal",
        "producer_failed": state == "invalid",
        "producer_censored": False,
        "producer_unstarted": state in {"missing", "conductor_gate_blocked"},
        "raw_numerator": 1,
        "raw_denominator": 1,
        "metric_direction": "exact_custody_is_required",
        "seed": None,
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
            "validation_receipts": [],
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
            "censored": not current_validation_complete,
            "provenance": "current_exp7586_validation_receipts",
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
        "inference_substrate_class": source.get("inference_substrate_class"),
        "evidence_state": source.get("evidence_state"),
        "claim_scope": "descriptive_reuse",
    }


def build_branch_conclusions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reduce four branches while keeping validity and benefit independent."""

    learning = evidence["exp7579-decision-learning-audit"]
    learning_payload = learning.get("payload") or {}
    audited = learning_payload.get("branch_conclusions")
    audit_rows = audited if isinstance(audited, Mapping) else {}
    static = audit_rows.get("static") if isinstance(audit_rows.get("static"), Mapping) else {}
    causal = audit_rows.get("causal") if isinstance(audit_rows.get("causal"), Mapping) else {}
    retention = (
        audit_rows.get("retention") if isinstance(audit_rows.get("retention"), Mapping) else {}
    )

    support = evidence["exp7580-arc-verifier-support"]
    canary = evidence["exp7581-arc-bounded-canary"]
    panel_a = evidence["exp7582-arc-panel-a"]
    panel_b = evidence["exp7583-arc-panel-b"]
    arc_audit = evidence["exp7584-arc-independent-audit"]
    support_payload = support.get("payload") or {}
    canary_payload = canary.get("payload") or {}
    arc_payload = arc_audit.get("payload") or {}

    service = evidence["exp7585-portable-service"]
    service_payload = service.get("payload") or {}
    speedup = service_payload.get("whole_service_speedup") or {}
    warm = speedup.get("warm") if isinstance(speedup, Mapping) else {}
    service_valid = bool(
        service.get("valid")
        and service_payload.get("portable_parity_score") == 1
        and service_payload.get("service_measurement_complete_score") == 1
        and service_payload.get("board_continuity_complete_score") == 1
    )
    service_benefit = bool(
        service_valid and isinstance(warm, Mapping) and float(warm.get("lower95") or 0.0) > 1.0
    )
    return [
        {
            "branch": "static_proper_loss",
            "sources": [_source_ref(learning)],
            "validity": bool(learning.get("valid") and static.get("readiness") is True),
            "readiness": static.get("readiness") is True,
            "benefit": static.get("benefit") is True,
            "freshness": static.get("freshness") is True,
            "failure_source": static.get("failure_source"),
            "verdict_class": "null" if static.get("benefit") is False else "positive",
            "claim_scope": "descriptive_reuse",
            "verifier_is_oracle": True,
            "oracle_distinct_positive_claimed": False,
            "conclusion": "The frozen proper-loss map is valid but did not beat the registered controls.",
        },
        {
            "branch": "delayed_learning_and_retention",
            "sources": [_source_ref(learning)],
            "validity": bool(learning.get("valid") and causal.get("readiness") is True),
            "readiness": causal.get("readiness") is True,
            "benefit": causal.get("benefit") is True,
            "retention_passed": retention.get("benefit") is True,
            "freshness": causal.get("freshness") is True,
            "failure_source": {
                "learning": causal.get("failure_source"),
                "retention": retention.get("failure_source"),
            },
            "verdict_class": "null",
            "claim_scope": "descriptive_reuse",
            "predict_release_update_persist_reload_complete": bool(
                learning_payload.get("learning_claims_qualified_score") == 1
            ),
            "conclusion": "Delayed learning showed no supported advantage and failed retention.",
        },
        {
            "branch": "live_verifier_support_and_plan_execution",
            "sources": [
                _source_ref(source) for source in (support, canary, panel_a, panel_b, arc_audit)
            ],
            "validity": arc_payload.get("arc_claims_qualified_score") == 1,
            "readiness": {
                "fixture_support": support_payload.get("verifier_support_ready_score") == 1,
                "transport": canary_payload.get("arc_transport_ready_score") == 1,
                "live_panels": arc_payload.get("arc_claims_qualified_score") == 1,
            },
            "benefit": "not_measured",
            "freshness": False,
            "verdict_class": "blocked",
            "claim_scope": "descriptive_reuse",
            "semantic_null_claimed": False,
            "fixture_positive_is_circular": True,
            "official_score_claimed": False,
            "new_solve_credit_claimed": False,
            "required_mechanism_change": (
                "Repair explicit-root ARC E2E and private environment plumbing before any model "
                "load; then run owned non-truncated panel generations with authenticated plan/action joins."
            ),
            "conclusion": "Fixture support is circular; failed plumbing and absent panels provide no semantic null.",
        },
        {
            "branch": "rust_service_and_board_continuity",
            "sources": [_source_ref(service)],
            "validity": service_valid,
            "readiness": service_valid,
            "benefit": service_benefit,
            "freshness": True,
            "verdict_class": "positive" if service_benefit else "null",
            "claim_scope": "descriptive_reuse",
            "portable_parity_score": service_payload.get("portable_parity_score"),
            "service_measurement_complete_score": service_payload.get(
                "service_measurement_complete_score"
            ),
            "whole_service_speedup": deepcopy(speedup),
            "board_rows": deepcopy(service_payload.get("board_rows") or []),
            "kernel_branch_disposition": service_payload.get("kernel_branch_disposition"),
            "promotes_other_branches": False,
            "hardware_expansion_claimed": False,
            "conclusion": "Rust improved the equal-durability host service; board states remain independent.",
        },
    ]


def retirement_decisions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Close only named constructions and preserve external hypotheses."""

    audit_verdict = evidence["exp7579-decision-learning-audit"].get("honest_verdict")
    return [
        {
            "branch": "static_proper_loss",
            "decision": "retire",
            "scope": "v662_frozen_nine_knot_proper_loss_construction",
            "literal_prior_experiment": "exp7567-source-evaluation",
            "literal_prior_verdict": "complete_null_source_evaluation_no_supported_benefit",
            "literal_current_experiment": "exp7579-decision-learning-audit",
            "literal_current_verdict": audit_verdict,
            "literal_exact_match": False,
            "literal_prior_rule_triggered": False,
            "construction_stop_rule_triggered": True,
            "hypothesis_retired": False,
            "explicit_addressed_cause": (
                "The frozen proper-loss map completed valid evaluation and did not beat raw or "
                "strong controls; Exp7577 predeclared stopping this construction on no benefit."
            ),
            "exact_next_prerequisite": "A different information source, not a new penalty or seed.",
        },
        {
            "branch": "delayed_learning_and_retention",
            "decision": "retire",
            "scope": "v662_direct_constrained_proper_loss_online_construction",
            "literal_prior_experiment": "exp7549-count-learning",
            "literal_prior_verdict": "complete_null_count_learning_valid_benefit_gate_failed",
            "literal_current_experiment": "exp7579-decision-learning-audit",
            "literal_current_verdict": audit_verdict,
            "literal_exact_match": False,
            "literal_prior_rule_triggered": False,
            "construction_stop_rule_triggered": True,
            "hypothesis_retired": False,
            "explicit_addressed_cause": (
                "The delayed learner had no supported advantage and retention was harmful."
            ),
            "exact_next_prerequisite": (
                "A new learner must add information and predeclare a retention-safe mechanism."
            ),
        },
        {
            "branch": "live_verifier_support_and_plan_execution",
            "decision": "defer",
            "scope": "v662_arc_adapter_withheld_live_panels",
            "literal_prior_experiment": "exp7570-arc-live-lineage",
            "literal_prior_verdict": "complete_disqualified_required_validation_failure",
            "literal_current_experiment": "exp7584-arc-independent-audit",
            "literal_current_verdict": evidence["exp7584-arc-independent-audit"].get(
                "honest_verdict"
            ),
            "literal_exact_match": False,
            "literal_prior_rule_triggered": False,
            "construction_stop_rule_triggered": False,
            "hypothesis_retired": False,
            "explicit_addressed_cause": (
                "The canary failed before inference and both live panel producers were unavailable."
            ),
            "exact_next_prerequisite": (
                "Pass E2E-009, E2E-010, E2E-011, E2E-012, E2E-013 and the private "
                "LLM-off smoke, then complete non-truncated owned panel generations."
            ),
        },
        {
            "branch": "rust_service_and_board_continuity",
            "decision": "continue",
            "scope": "rust_recalibration_complete_service_boundary",
            "literal_prior_experiment": "exp7571-portable-calibration",
            "literal_prior_verdict": "complete_blocked_exp7561_recalibration_ready_score",
            "literal_current_experiment": "exp7585-portable-service",
            "literal_current_verdict": evidence["exp7585-portable-service"].get("honest_verdict"),
            "literal_exact_match": False,
            "literal_prior_rule_triggered": False,
            "construction_stop_rule_triggered": False,
            "hypothesis_retired": False,
            "explicit_addressed_cause": (
                "The Rust process achieved parity and complete-service improvement on the host."
            ),
            "exact_next_prerequisite": (
                "Demonstrate an actual consumer boundary before PyO3 work; require a new operator "
                "physical-change receipt before any GateMate probe."
            ),
        },
    ]


def collect_preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record exact sources, requirement, authority, host, and producer states."""

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
            "expected": "REQ-REPORT-7586",
            "observed": "REQ-REPORT-7586" if "REQ-REPORT-7586" in spec_text else None,
            "passed": "REQ-REPORT-7586" in spec_text,
        }
    )
    rows.append(
        {
            "check": "roadmap_authority",
            "upstream": "V662 authority resolution",
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
    """Bind code, authority, and every producer or diagnostic byte."""

    paths = [*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH]
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
            "original_substrate_class": source.get("inference_substrate_class"),
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
    """Keep validity, readiness, effect, retention, and freshness separate."""

    by_branch = {str(row.get("branch")): row for row in branches}
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    static = by_branch["static_proper_loss"]
    learning = by_branch["delayed_learning_and_retention"]
    arc = by_branch["live_verifier_support_and_plan_execution"]
    service = by_branch["rust_service_and_board_continuity"]
    validity = "Invalid or incomplete evidence cannot support science."
    readiness = "Accounting completion is independent from empirical benefit."
    benefit = "One branch cannot promote another branch's effect claim."
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v662_contract",
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
            "static_proper_loss_valid",
            "validity",
            "exp7579-decision-learning-audit",
            "branch_conclusions.static_proper_loss",
            "validity",
            True,
            static.get("validity"),
            static.get("validity") is True,
            validity,
        ),
        _gate(
            "static_proper_loss_benefit",
            "benefit",
            "exp7579-decision-learning-audit",
            "branch_conclusions.static_proper_loss",
            "benefit",
            True,
            static.get("benefit"),
            static.get("benefit") is True,
            benefit,
        ),
        _gate(
            "delayed_learning_benefit",
            "benefit",
            "exp7579-decision-learning-audit",
            "branch_conclusions.delayed_learning_and_retention",
            "benefit",
            True,
            learning.get("benefit"),
            learning.get("benefit") is True,
            benefit,
        ),
        _gate(
            "retention_non_regression",
            "retention",
            "exp7579-decision-learning-audit",
            "branch_conclusions.delayed_learning_and_retention",
            "retention_passed",
            True,
            learning.get("retention_passed"),
            learning.get("retention_passed") is True,
            benefit,
        ),
        _gate(
            "live_arc_support_valid",
            "validity",
            "exp7584-arc-independent-audit",
            "branch_conclusions.live_verifier_support_and_plan_execution",
            "validity",
            True,
            arc.get("validity"),
            arc.get("validity") is True,
            validity,
        ),
        _gate(
            "rust_complete_service_benefit",
            "benefit",
            "exp7585-portable-service",
            "branch_conclusions.rust_service_and_board_continuity",
            "benefit",
            True,
            service.get("benefit"),
            service.get("benefit") is True,
            benefit,
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
            "Source exposure remains visible and cannot become fresh confirmation.",
        ),
    ]


def failure_rows(
    contract: Mapping[str, Any], evidence: Mapping[str, JsonDict], validation: Mapping[str, Any]
) -> list[JsonDict]:
    """List only failures that determine the aggregate terminal class."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "check": "contract_authorities_agree",
                "upstream": "v662_contract",
                "path": contract.get("selected_roadmap_path"),
                "field": "comparison_passed",
                "op": "==",
                "expected": True,
                "observed": contract.get("comparison_passed"),
                "passed": False,
            }
        )
    for field in ("required_checks_passed", "terminal_validation_passed"):
        if validation.get(field) is not True:
            failures.append(
                {
                    "check": f"current_{field}",
                    "upstream": EXPERIMENT_ID,
                    "path": "validation_receipts",
                    "field": field,
                    "op": "==",
                    "expected": True,
                    "observed": validation.get(field),
                    "passed": False,
                }
            )
    for task_id, source in evidence.items():
        state = source.get("evidence_state")
        if state in {"missing", "conductor_gate_blocked", "invalid"}:
            summary = source.get("gate_check_summary")
            if isinstance(summary, Mapping):
                failures.append(deepcopy(dict(summary)))
        elif source.get("verdict_class") == "blocked":
            failures.append(
                {
                    "check": "producer_terminal_not_blocked",
                    "upstream": task_id,
                    "path": source.get("evidence_path"),
                    "field": "verdict_class",
                    "op": "!=",
                    "expected": "blocked",
                    "observed": "blocked",
                    "passed": False,
                }
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
    "honest_verdict": "A complete prefix records terminal state without implying benefit.",
    "verdict_class": "The closed class reserves partial for unfinished owned work.",
    "flagged_adversarial": "Flagged evidence cannot open readiness or support science.",
    "gate_check_summary": "Every block retains exact upstream operands.",
    "acceptance_gate_results": "Validity, readiness, effect, retention, and freshness stay separate.",
    "rows": "One row per task keeps absence distinct from a zero.",
    "inference_substrate_class": "Aggregation prevents a current live-inference claim.",
    "MODEL_SPECS": "An empty list prevents historical models from becoming current calls.",
    "invocation_counts": "Typed zeros expose invented loads, forwards, generations, or tokens.",
    "duration_s": "Monotonic current time excludes inherited work and artificial sleep.",
    "source_artifact_hashes": "Exact byte hashes bind each inherited conclusion.",
    "validation_receipts": "Commands, exits, worktree paths, and log hashes bind validation.",
    "field_principles": "Each top-level field states the inference failure it prevents.",
    "verifier_is_oracle": "Label-accessing controls cannot support an oracle-distinct claim.",
    "capstone_complete_score": "One means fourteen dispositions and owned validation, not benefit.",
    "task_dispositions": "All fourteen tasks retain one exact custody state.",
    "branch_conclusions": "No branch can promote another branch's claim.",
    "retirement_decisions": "Only a named construction closes for a concrete cause.",
    "publication_gates": "Stable G1-G4 remain unchanged and do not publish anything.",
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
        "independent_unit": "ordered_v662_task_disposition",
        "planned": 14,
        "attempted": sum(int(row.get("disposition_attempted") is True) for row in dispositions),
        "completed": sum(int(row.get("disposition_completed") is True) for row in dispositions),
        "excluded": sum(
            int(row.get("excluded_from_scientific_metrics") is True) for row in dispositions
        ),
        "failed": sum(int(row.get("producer_failed") is True) for row in dispositions),
        "censored": sum(int(row.get("producer_censored") is True) for row in dispositions),
        "unstarted": sum(int(row.get("producer_unstarted") is True) for row in dispositions),
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


def _source_revision(root: Path) -> str:
    """Read the current Git revision without changing repository state."""

    completed = subprocess.run(  # noqa: S603 - fixed read-only command.
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _passing_receipts() -> list[JsonDict]:
    """Create deterministic passing receipts for pure test construction."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": f"unit-test/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
        }
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _test_spans() -> list[JsonDict]:
    """Return deterministic phase rows for pure test construction."""

    phases = (
        ("preconditions", 13, "thirteen_upstreams_observed"),
        ("publication_gate", 4, "g1_g4_read_only"),
        ("model_load", 0, "no_current_model_load"),
        ("generation", 0, "no_current_generation"),
        ("reduction", 14, "fourteen_dispositions_reduced"),
        ("validation", 8, "affected_checks_complete"),
        ("terminal_validation", 4, "cold_readers_complete"),
        ("write", 1, "terminal_artifact_ready"),
    )
    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "completed_units": units,
            "checkpoint": checkpoint,
        }
        for phase, units, checkpoint in phases
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
    """Build one compact terminal record from authenticated V662 evidence."""

    affected_complete = validation.get("required_checks_passed") is True
    terminal_complete = validation.get("terminal_validation_passed") is True
    current_complete = affected_complete and terminal_complete
    terminal = classify_terminal(
        list(evidence.values()),
        affected_complete=affected_complete,
        terminal_complete=terminal_complete,
    )
    dispositions = task_dispositions(
        contract["tasks"],
        evidence,
        terminal,
        current_validation_complete=current_complete,
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
        "experiment": 7586,
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
            "source_revision": _source_revision(root),
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
            "audit": 7_586_662_01,
            "explanation": "Deterministic aggregation makes no outcome-dependent draw.",
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
            "readout_kind": "upstream_artifact_reduction",
            "generated_tokens": 0,
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
        "branch_conclusions": branches,
        "branch_dispositions": branches,
        "prior_failure_rows": deepcopy(current_task.get("prior_failures") or []),
        "retirement_decisions": retirements,
        "retirement_rows": [row for row in retirements if row["decision"] == "retire"],
        "continuation_rows": [row for row in retirements if row["decision"] != "retire"],
        "permanent_exclusion_entries_added": [],
        "acceptance_gate_results": acceptance_gates(contract, dispositions, branches, validation),
        "gate_check_summary": _gate_summary(failures),
        "capstone_complete_score": complete_score,
        "publication_gates": publication,
        "publication_gate_results": deepcopy(publication),
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
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
                "required": True,
                "exercise": "predict-release-update-persist-reload",
                "completed": branches[1]["predict_release_update_persist_reload_complete"],
                "upstream_task": "exp7579-decision-learning-audit",
            },
            "numbered_runtime_e2e": "not_applicable_read_only_reporting",
        },
        "verifier_is_oracle": True,
        "oracle_distinct_positive_claimed": False,
        "roadmap_activation_performed": False,
        "roadmap_archive_performed": False,
        "publication_performed": False,
        "submission_performed": False,
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
        "status",
        "honest_verdict",
        "verdict_class",
        "positive_claim",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "planned_inference_substrate_class",
        "inference_substrate_class",
        "inference_substrate",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "duration_s",
        "source_artifact_hashes",
        "validation_receipts",
        "field_principles",
        "verifier_is_oracle",
        "capstone_complete_score",
        "task_dispositions",
        "branch_conclusions",
        "retirement_decisions",
        "publication_gates",
        "reproducibility_checksum",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
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
            contract["tasks"],
            evidence,
            terminal,
            current_validation_complete=current_complete,
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
        if artifact.get("retirement_decisions") != retirements:
            errors.append("retirement_decisions_invalid")
        if artifact.get("retirement_rows") != [
            row for row in retirements if row["decision"] == "retire"
        ]:
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != [
            row for row in retirements if row["decision"] != "retire"
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
    """Build the shared scoped plan for only the affected Exp7586 files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, or command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V662 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def root_argument(value: str | Path) -> Path:
    """Require an absolute repository root that resolves to this worktree."""

    resolved = Path(value).resolve()
    if resolved != REPO_ROOT or not (resolved / "AGENTS.md").is_file():
        raise ValueError(f"repository root must resolve to {REPO_ROOT}")
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
        f"[exp7586] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
    """Build entrypoint replay, independent reduction, and strict guards."""

    python = str(root / ".venv/bin/python")
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--root",
                str(root),
                "--validate",
                str(candidate),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--root",
                str(root),
                "--independent-reduce",
                str(candidate),
            ),
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
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7586-", dir="/tmp"))

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
        started,
        "publication_gate",
        "after_subprocess",
        paper_ready=publication["paper_ready"],
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
        log_dir=raw_dir / "validation/affected",
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
            checkpoint="fourteen_dispositions_and_four_branches_reduced",
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
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "reduction", "after", completed_units=14)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal_receipts = run_categorized_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    validation["terminal_validation_passed"] = all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in terminal_receipts
    )
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            completed_units=len(terminal_receipts),
            checkpoint="cold_replay_reduction_and_strict_guards_complete",
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
    """Parse the frozen root, date, and read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the capstone or one read-only fresh-process replay."""

    print("[exp7586] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    root = root_argument(args.root)
    candidate_path = args.validate or args.independent_reduce
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
