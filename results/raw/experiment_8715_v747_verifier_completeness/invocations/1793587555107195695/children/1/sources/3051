"""Reconcile thirteen V661 tasks without merging distinct scientific claims.

This report reads upstream bytes and conductor diagnostics. It performs no
model inference, roadmap change, publication, or hardware operation.

Spec refs: REQ-REPORT-7572 and SCENARIO-REPORT-7572-*.
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
from carnot.experiment_7560_v661_contract_methods import (
    compare_contract_authorities,
    resolve_v661_roadmap,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7572-capstone"
SCHEMA = "carnot.exp7572.v661.capstone.v1"

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7572_v661_capstone.json")
RAW_DIR = Path("results/raw/experiment_7572_v661_capstone")
MODULE_PATH = Path("python/carnot/experiment_7572_v661_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7572_v661_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7572_v661_capstone.py")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7560, "contract-methods"),
        (7561, "recalibration-prototype"),
        (7562, "arc-plan-lineage"),
        (7563, "native-pilot"),
        (7564, "fit-capture"),
        (7565, "test-online-capture"),
        (7566, "energy-fit"),
        (7567, "source-evaluation"),
        (7568, "continuous-recalibration"),
        (7569, "decision-learning-audit"),
        (7570, "arc-live-lineage"),
        (7571, "portable-calibration"),
        (7572, "capstone"),
    )
)
CLOSED_VERDICTS = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
RANDOM_SEED = {
    "selection": None,
    "fitting": None,
    "ordering": None,
    "bootstrap": None,
    "audit": 7_572_661_01,
    "explanation": "Deterministic aggregation makes no outcome-dependent draw.",
}
PRODUCER_PATHS = {
    task_id: Path(f"results/experiment_{number}_v661_{slug.replace('-', '_')}.json")
    for task_id, number, slug in (
        ("exp7560-contract-methods", 7560, "contract-methods"),
        ("exp7561-recalibration-prototype", 7561, "recalibration-prototype"),
        ("exp7562-arc-plan-lineage", 7562, "arc-plan-lineage"),
        ("exp7563-native-pilot", 7563, "native-pilot"),
        ("exp7564-fit-capture", 7564, "fit-capture"),
        ("exp7565-test-online-capture", 7565, "test-online-capture"),
        ("exp7566-energy-fit", 7566, "energy-fit"),
        ("exp7567-source-evaluation", 7567, "source-evaluation"),
        ("exp7568-continuous-recalibration", 7568, "continuous-recalibration"),
        ("exp7569-decision-learning-audit", 7569, "decision-learning-audit"),
        ("exp7570-arc-live-lineage", 7570, "arc-live-lineage"),
        ("exp7571-portable-calibration", 7571, "portable-calibration"),
    )
}
DIAGNOSTIC_PATHS = {
    "exp7568-continuous-recalibration": Path(
        "results/experiment_7568_continuous_recalibration.json"
    )
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
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7559_v660_capstone.py"),
    PUBLICATION_GATE_PATH,
    Path("scripts/summarize_artifact.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("ops/north-star.md"),
    Path("ops/known-issues.md"),
    Path("ops/verifier_gaps.md"),
    DESIGN_PATH,
    SPEC_PATH,
    CONDUCTOR_LOG_PATH,
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
    """Read one JSON object and reject malformed or non-object evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_contract(root: Path) -> JsonDict:
    """Resolve the current V661 authority and compare its full task contract."""

    selected, roadmap, candidates = resolve_v661_roadmap(root)
    comparison = compare_contract_authorities((root / DESIGN_PATH).read_text(), roadmap)
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V661 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": deepcopy(candidates),
        "roadmap": deepcopy(roadmap),
        "tasks": deepcopy(tasks),
    }


def _ready_values(payload: Mapping[str, Any]) -> JsonDict:
    """Retain scalar readiness fields without treating them as benefit."""

    return {
        str(key): deepcopy(value)
        for key, value in sorted(payload.items())
        if str(key).endswith(("_score", "_ready", "_complete"))
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _required_receipts_pass(payload: Mapping[str, Any]) -> bool:
    """Require every producer receipt that the producer marked as required."""

    summary = payload.get("validation_summary")
    if isinstance(summary, Mapping) and "required_checks_passed" in summary:
        return summary.get("required_checks_passed") is True
    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list) or not receipts:
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    required = [row for row in rows if row.get("required") is True] or rows
    return bool(required) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in required
    )


def _receipt_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Keep receipt identity and hashes while omitting large command output."""

    fields = ("name", "required", "passed", "exit_code", "log_path", "log_sha256")
    receipts = payload.get("validation_receipts")
    return (
        [
            {key: deepcopy(row.get(key)) for key in fields}
            for row in receipts
            if isinstance(row, Mapping)
        ]
        if isinstance(receipts, list)
        else []
    )


def _diagnostic_failure(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize the original conductor gate operands without guessing."""

    raw = payload.get("blocked_diagnostic_contract")
    row = raw if isinstance(raw, Mapping) else payload
    return {
        "check": "conductor_pre_gate",
        "upstream": row.get("failed_upstream"),
        "path": row.get("failed_evidence_path"),
        "field": row.get("failed_field"),
        "op": row.get("failed_operator"),
        "expected": deepcopy(row.get("failed_expected")),
        "observed": deepcopy(row.get("failed_observed")),
        "passed": False,
    }


def _base_source(task: Mapping[str, Any]) -> JsonDict:
    """Create the common source row before exact bytes are inspected."""

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
        "source_status": "MISSING",
        "authenticated": False,
        "custody_valid": False,
        "valid": False,
        "honest_verdict": "complete_blocked_missing_declared_producer_evidence",
        "verdict_class": "blocked",
        "original_verdict_class": None,
        "flagged_adversarial": False,
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


def _producer_failure(task_id: str, path: str, payload: Mapping[str, Any]) -> JsonDict:
    """Name the first present-evidence failure that prevents scientific use."""

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


def load_producer(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one producer or its original pre-gate diagnostic."""

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
            source_status="GATE_BLOCK",
            authenticated=True,
            custody_valid=True,
            valid=False,
            honest_verdict=str(payload.get("honest_verdict") or "blocked_gate_check_failed"),
            verdict_class="blocked",
            gate_check_summary=_diagnostic_failure(payload),
            payload=payload,
        )
        return row

    try:
        payload = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        row.update(
            artifact_path=relative.as_posix(),
            source_sha256=sha256_file(path),
            source_size_bytes=path.stat().st_size,
            evidence_state="invalid",
            source_status="INVALID",
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
    identity_ok = str(expected_number) in str(observed_id)
    original_class = payload.get("verdict_class")
    receipts_pass = _required_receipts_pass(payload)
    authenticated = bool(
        identity_ok
        and payload.get("milestone") == MILESTONE
        and original_class in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
    )
    valid = bool(
        authenticated
        and receipts_pass
        and original_class != "disqualified"
        and payload.get("flagged_adversarial") is False
    )
    row.update(
        artifact_path=relative.as_posix(),
        evidence_path=relative.as_posix(),
        source_sha256=sha256_file(path),
        source_size_bytes=path.stat().st_size,
        evidence_state="terminal" if valid else "invalid",
        source_status="TERMINAL" if valid else "INVALID",
        authenticated=authenticated,
        custody_valid=authenticated,
        valid=valid,
        honest_verdict=str(payload.get("honest_verdict") or payload.get("status") or "absent"),
        verdict_class=str(original_class) if valid else "disqualified",
        original_verdict_class=original_class,
        flagged_adversarial=payload.get("flagged_adversarial") is True,
        model_invoked=payload.get("model_invoked") is True,
        ready_value_fields=_ready_values(payload),
        validation_receipts=_receipt_rows(payload),
        gate_check_summary=(
            deepcopy(payload.get("gate_check_summary") or {})
            if valid
            else _producer_failure(task_id, relative.as_posix(), payload)
        ),
        payload=payload,
    )
    return row


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Inventory all twelve upstream tasks in declared order."""

    return {str(task.get("id")): load_producer(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, affected_complete: bool, terminal_complete: bool
) -> JsonDict:
    """Apply owned-work precedence before upstream invalidity or absence."""

    if affected_complete and not terminal_complete:
        verdict = "partial"
        honest = "partial_retryable_current_capstone_validation_unfinished"
    elif not affected_complete:
        verdict = "disqualified"
        honest = "complete_disqualified_required_v661_capstone_validation"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict = "disqualified"
        honest = "complete_disqualified_required_v661_evidence"
    elif any(
        row.get("evidence_state") in {"missing", "conductor_gate_blocked"}
        or row.get("verdict_class") == "blocked"
        for row in rows
    ):
        verdict = "blocked"
        honest = "complete_blocked_required_v661_external_evidence"
    else:
        verdict = "null"
        honest = "complete_null_v661_accounting_without_aggregate_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _disposition(source: Mapping[str, Any], task: Mapping[str, Any], order: int) -> JsonDict:
    """Reduce one source while preserving invalid, blocked, and flagged states."""

    excluded = bool(
        source.get("evidence_state") != "terminal"
        or source.get("verdict_class") in {"blocked", "disqualified"}
        or source.get("flagged_adversarial") is True
    )
    started = source.get("evidence_state") not in {"missing", "conductor_gate_blocked"}
    return {
        "order": order,
        "task_id": str(task.get("id")),
        "expected_artifact_path": source.get("expected_path"),
        "artifact_path": source.get("artifact_path"),
        "evidence_path": source.get("evidence_path"),
        "artifact_sha256": source.get("source_sha256"),
        "evidence_state": source.get("evidence_state"),
        "source_status": source.get("source_status"),
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("verdict_class"),
        "original_verdict_class": source.get("original_verdict_class"),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
        "ready_value_fields": deepcopy(source.get("ready_value_fields") or {}),
        "validation_receipts": deepcopy(source.get("validation_receipts") or []),
        "gate_check_summary": deepcopy(source.get("gate_check_summary") or {}),
        "disposition_attempted": True,
        "disposition_completed": True,
        "producer_started": started,
        "producer_completed": source.get("evidence_state") == "terminal",
        "producer_failed": source.get("evidence_state") == "invalid",
        "producer_censored": False,
        "producer_unstarted": not started,
        "excluded_from_scientific_metrics": excluded,
        "principle": "Exact custody keeps missing or invalid evidence from becoming a benefit.",
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    *,
    current_validation_complete: bool,
) -> list[JsonDict]:
    """Build twelve source rows and one current row without self-reading."""

    rows = [
        _disposition(evidence[str(task.get("id"))], task, order)
        for order, task in enumerate(tasks[:-1], 1)
    ]
    rows.append(
        {
            "order": 13,
            "task_id": str(tasks[-1].get("id")),
            "expected_artifact_path": str(tasks[-1].get("deliverable")),
            "artifact_path": None,
            "evidence_path": None,
            "artifact_sha256": None,
            "evidence_state": "current_terminal"
            if current_validation_complete
            else "current_running",
            "source_status": "TERMINAL" if current_validation_complete else "RUNNING",
            "honest_verdict": terminal.get("honest_verdict"),
            "verdict_class": terminal.get("verdict_class"),
            "original_verdict_class": terminal.get("verdict_class"),
            "flagged_adversarial": False,
            "ready_value_fields": {"capstone_complete_score": int(current_validation_complete)},
            "validation_receipts": [],
            "gate_check_summary": {},
            "disposition_attempted": True,
            "disposition_completed": current_validation_complete,
            "producer_started": True,
            "producer_completed": current_validation_complete,
            "producer_failed": False,
            "producer_censored": False,
            "producer_unstarted": False,
            "excluded_from_scientific_metrics": True,
            "principle": "Current work cannot authenticate itself through its future result path.",
        }
    )
    return rows


def _source_ref(source: Mapping[str, Any]) -> JsonDict:
    """Bind a branch input without copying its large measurement rows."""

    return {
        "task_id": source.get("task_id"),
        "path": source.get("evidence_path") or source.get("expected_path"),
        "sha256": source.get("source_sha256"),
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("original_verdict_class") or source.get("verdict_class"),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
        "evidence_state": source.get("evidence_state"),
    }


def build_branch_conclusions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reduce source, learning, ARC, and portability as independent claims."""

    audit = evidence["exp7569-decision-learning-audit"]
    audit_payload = audit.get("payload") or {}
    branch_payload = audit_payload.get("branch_dispositions") or {}
    source = branch_payload.get("source") if isinstance(branch_payload, Mapping) else {}
    learning = branch_payload.get("learning") if isinstance(branch_payload, Mapping) else {}
    arc = evidence["exp7570-arc-live-lineage"]
    arc_payload = arc.get("payload") or {}
    portable = evidence["exp7571-portable-calibration"]
    portable_payload = portable.get("payload") or {}
    source_qualified = int(bool(audit.get("valid") and source.get("claims_qualified_score") == 1))
    learning_qualified = int(
        bool(audit.get("valid") and learning.get("claims_qualified_score") == 1)
    )
    source_benefit = int(
        source_qualified and audit_payload.get("qualified_source_benefit_score") == 1
    )
    learning_benefit = int(
        learning_qualified and audit_payload.get("qualified_learning_benefit_score") == 1
    )
    return [
        {
            "branch": "source_decisions",
            "sources": [_source_ref(audit)],
            "claims_qualified_score": source_qualified,
            "benefit_score": source_benefit,
            "probability_improvement_score": source_benefit,
            "decision_cost_improvement_score": source_benefit,
            "freshness_measured": source_qualified == 1,
            "verdict_class": source.get("verdict_class"),
            "verifier_is_oracle": True,
            "verification_basis": "execution_grounded_label_check",
            "oracle_distinct_benefit_claimed": False,
            "conclusion": "Independent Exp7569 reduction qualified the source null; no probability or decision-cost benefit passed.",
        },
        {
            "branch": "persistent_learning",
            "sources": [_source_ref(audit)],
            "claims_qualified_score": learning_qualified,
            "benefit_score": learning_benefit,
            "freshness_measured": learning_qualified == 1,
            "retention_measured": learning_qualified == 1,
            "restart_measured": learning_qualified == 1,
            "predict_release_update_persist_reload_complete": learning_qualified == 1,
            "verdict_class": learning.get("verdict_class"),
            "gate_failure": deepcopy(learning.get("gate_failure")),
            "conclusion": "Exp7568 did not start after the prototype gate failed; no learning state or zero effect was invented.",
        },
        {
            "branch": "arc_runtime_lineage",
            "sources": [_source_ref(arc)],
            "source_collection_dependency": False,
            "plan_lineage_measured_score": arc_payload.get("plan_lineage_measured_score"),
            "arc_measurement_complete_score": arc_payload.get("arc_measurement_complete_score"),
            "trajectory_supervisor": deepcopy(arc_payload.get("trajectory_supervisor") or {}),
            "causal_suppression_efficacy_claimed": False,
            "new_solve_credit_claimed": False,
            "official_score_claimed": False,
            "excluded_from_scientific_metrics": not arc.get("valid"),
            "verdict_class": arc.get("verdict_class"),
            "conclusion": "The flagged disqualified runtime row remains visible and supplies no lineage or supervisor benefit claim.",
        },
        {
            "branch": "portable_calibration",
            "sources": [_source_ref(portable)],
            "portable_kernel_ready_score": portable_payload.get("portable_kernel_ready_score"),
            "service_measurement_complete_score": portable_payload.get(
                "service_measurement_complete_score"
            ),
            "board_continuity_complete_score": portable_payload.get(
                "board_continuity_complete_score"
            ),
            "rust_parity": deepcopy(portable_payload.get("rust_parity")),
            "complete_service_costs": deepcopy(
                portable_payload.get("kernel_and_service_costs") or {}
            ),
            "board_rows": deepcopy(portable_payload.get("board_rows") or []),
            "hardware_benefit_claimed": False,
            "verdict_class": portable.get("verdict_class"),
            "conclusion": "Board continuity is complete; Rust parity and complete service costs were not measured.",
        },
    ]


def continuation_rows() -> list[JsonDict]:
    """Require one falsifiable change before any branch runs again."""

    return [
        {
            "branch": "source_decisions",
            "decision": "continue",
            "current_state": "qualified_source_null",
            "changed_prerequisite": "Use fresh held-out source groups and freeze a newly diagnosed source-signal mechanism before outcomes.",
            "reopen_when": "The new mechanism states why it can beat the same-information comparator and freezes its effect and uncertainty gates.",
            "activation_authorized": False,
        },
        {
            "branch": "persistent_learning",
            "decision": "defer",
            "current_state": "blocked_by_disqualified_prototype",
            "changed_prerequisite": "Repair Exp7561 required validation and obtain recalibration_ready_score=1 without changing the frozen lifecycle after outcomes.",
            "reopen_when": "The prototype independently converges and predict-release-update-persist-reload passes before fresh online data are read.",
            "activation_authorized": False,
        },
        {
            "branch": "arc_runtime_lineage",
            "decision": "defer",
            "current_state": "disqualified_required_validation",
            "changed_prerequisite": "Diagnose the validation failure, preserve the withheld roster, and collect authenticated model-plan-action-supervisor joins.",
            "reopen_when": "A corrected unflagged artifact passes required validation and records actual episode outcomes.",
            "activation_authorized": False,
        },
        {
            "branch": "portable_calibration",
            "decision": "defer",
            "current_state": "board_continuity_complete_kernel_blocked",
            "changed_prerequisite": "First qualify the constrained recalibration state, then measure Rust parity and the complete durable service denominator.",
            "reopen_when": "Prototype readiness is one and a paired service plan includes compute, movement, persistence, startup, and acknowledgement.",
            "activation_authorized": False,
        },
    ]


def prior_failure_rows(terminal: Mapping[str, Any]) -> list[JsonDict]:
    """Compare the capstone's literal prior verdict without broad retirement."""

    prior = "complete_blocked_required_v660_source_science_externally_gated"
    current = terminal.get("honest_verdict")
    exact = current == prior
    return [
        {
            "task_id": EXPERIMENT_ID,
            "prior_experiment": "exp7559-capstone",
            "prior_honest_verdict": prior,
            "current_honest_verdict": current,
            "retire_if_same_verdict": True,
            "exact_text_match": exact,
            "same_completed_mechanism": exact and terminal.get("verdict_class") != "blocked",
            "external_absence": terminal.get("verdict_class") == "blocked",
            "retirement_triggered": bool(exact and terminal.get("verdict_class") != "blocked"),
            "addressed_by": "Use current independent branch evidence and a resolved V661 authority.",
        }
    ]


def retirement_rows(priors: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retire only a literal repeated completed mechanism."""

    return [
        {
            "task_id": row.get("task_id"),
            "prior_experiment": row.get("prior_experiment"),
            "prior_honest_verdict": row.get("prior_honest_verdict"),
            "current_honest_verdict": row.get("current_honest_verdict"),
            "scope": "same_completed_scientific_mechanism_only",
            "exclusion_workflow": "ops/exclusion_manifest.yaml",
        }
        for row in priors
        if row.get("retirement_triggered") is True
    ]


def evaluate_publication_gates(root: Path) -> JsonDict:
    """Run the stable G1-G4 reader and recompute its conjunction."""

    command = (str(root / ".venv/bin/python"), "-u", PUBLICATION_GATE_PATH.as_posix(), "--json")
    completed = subprocess.run(  # noqa: S603 - fixed repository command.
        command, cwd=root, text=True, capture_output=True, timeout=120, check=False
    )
    try:
        parsed = json.loads(completed.stdout)
        gates = parsed["gates"]
        if not isinstance(parsed, dict) or not isinstance(gates, dict):
            raise ValueError("publication mapping required")
        stable = {name: deepcopy(gates[name]) for name in ("G1", "G2", "G3", "G4")}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        stable = {
            name: {"pass": False, "detail": "reader_failed"} for name in ("G1", "G2", "G3", "G4")
        }
    unmet = [name for name, gate in stable.items() if gate.get("pass") is not True]
    return {
        "gates": stable,
        "unmet_gates": unmet,
        "paper_ready": not unmet,
        "command_argv": list(command),
        "exit_code": completed.returncode,
        "stdout_sha256": "sha256:"
        + __import__("hashlib").sha256(completed.stdout.encode()).hexdigest(),
        "publication_performed": False,
    }


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
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else "absent_or_empty",
                "passed": available,
                "sha256": sha256_file(path) if available else None,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": SPEC_PATH.as_posix(),
                "path": SPEC_PATH.as_posix(),
                "field": "REQ-*",
                "expected": "REQ-REPORT-7572",
                "observed": "REQ-REPORT-7572" if "REQ-REPORT-7572" in spec_text else None,
                "passed": "REQ-REPORT-7572" in spec_text,
            },
            {
                "check": "roadmap_authority",
                "upstream": "V661 authority resolution",
                "path": contract.get("selected_roadmap_path"),
                "field": "comparison_passed",
                "expected": True,
                "observed": contract.get("comparison_passed"),
                "passed": contract.get("comparison_passed") is True,
            },
        ]
    )
    disk = shutil.disk_usage(root)
    rows.append(
        {
            "check": "aggregation_resource",
            "upstream": "current_host",
            "path": str(root),
            "field": "cpu_and_storage",
            "expected": "cpu_count>0 and free_bytes>0",
            "observed": {
                "cpu_count": os.cpu_count(),
                "free_bytes": disk.free,
                "gpu_required": False,
            },
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
    """Bind code, authorities, and exact producer or diagnostic bytes."""

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
    """Keep each acceptance operand and its failure-prevention principle."""

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
    """Keep validity, readiness, and scientific benefit independent."""

    by_branch = {str(row.get("branch")): row for row in branches}
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    validity = "Invalid evidence cannot support science."
    readiness = "A valid null remains reusable because completion is not benefit."
    benefit = "A missing effect, cost, or support gate cannot become empirical value."
    source = by_branch["source_decisions"]
    learning = by_branch["persistent_learning"]
    arc = by_branch["arc_runtime_lineage"]
    portable = by_branch["portable_calibration"]
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v661_contract",
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
            "thirteen_honest_dispositions",
            "readiness",
            EXPERIMENT_ID,
            "task_dispositions",
            "length",
            13,
            len(dispositions),
            len(dispositions) == 13,
            readiness,
        ),
        _gate(
            "source_claims_qualified",
            "readiness",
            "exp7569-decision-learning-audit",
            "branch_conclusions.source_decisions",
            "claims_qualified_score",
            1,
            source.get("claims_qualified_score"),
            source.get("claims_qualified_score") == 1,
            readiness,
        ),
        _gate(
            "source_probability_and_decision_benefit",
            "benefit",
            "exp7569-decision-learning-audit",
            "branch_conclusions.source_decisions",
            "benefit_scores",
            [1, 1],
            [
                source.get("probability_improvement_score"),
                source.get("decision_cost_improvement_score"),
            ],
            source.get("probability_improvement_score")
            == source.get("decision_cost_improvement_score")
            == 1,
            benefit,
        ),
        _gate(
            "learning_causal_replay",
            "readiness",
            "exp7569-decision-learning-audit",
            "branch_conclusions.persistent_learning",
            "claims_qualified_score",
            1,
            learning.get("claims_qualified_score"),
            learning.get("claims_qualified_score") == 1,
            readiness,
        ),
        _gate(
            "arc_valid_lineage_measurement",
            "readiness",
            "exp7570-arc-live-lineage",
            "branch_conclusions.arc_runtime_lineage",
            "plan_lineage_measured_score",
            1,
            arc.get("plan_lineage_measured_score"),
            arc.get("plan_lineage_measured_score") == 1,
            readiness,
        ),
        _gate(
            "portable_complete_service",
            "readiness",
            "exp7571-portable-calibration",
            "branch_conclusions.portable_calibration",
            "service_measurement_complete_score",
            1,
            portable.get("service_measurement_complete_score"),
            portable.get("service_measurement_complete_score") == 1,
            readiness,
        ),
        _gate(
            "measured_portable_benefit",
            "benefit",
            "exp7571-portable-calibration",
            "branch_conclusions.portable_calibration",
            "hardware_benefit_claimed",
            True,
            portable.get("hardware_benefit_claimed"),
            portable.get("hardware_benefit_claimed") is True,
            benefit,
        ),
    ]


def failure_rows(
    contract: Mapping[str, Any], evidence: Mapping[str, JsonDict], validation: Mapping[str, Any]
) -> list[JsonDict]:
    """List only failures that control terminal classification."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "check": "contract_authorities_agree",
                "upstream": "v661_contract",
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
    for source in evidence.values():
        if source.get("evidence_state") != "terminal":
            failures.append(deepcopy(source.get("gate_check_summary") or {}))
    return failures


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every classification failure and its first exact cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


FIELD_PRINCIPLES = {
    "experiment_id": "The exact ID, milestone, and date prevent cross-run evidence drift.",
    "preconditions_checked": "Exact inputs and resources prevent fabricated fallback.",
    "MODEL_SPECS": "An empty list prevents historical model work from becoming current work.",
    "model_specs": "The resolved-spec alias remains empty because no model loaded.",
    "model_invoked": "False separates aggregation from cited upstream inference.",
    "invocation_counts": "Typed zero counters expose an invented load, forward, or generation.",
    "inference_substrate_class": "Aggregation prevents model-runtime claims.",
    "inference_substrate": "The exact label distinguishes reduction from inference.",
    "execution_venue": "Host is a legal venue and stays separate from device identity.",
    "duration_s": "Monotonic current time does not absorb historical runtime.",
    "random_seed": "Frozen seeds prevent favorable rerun selection.",
    "reproducibility_checksum": "One digest binds source bytes, code, settings, and reductions.",
    "rows": "Thirteen rows keep missing work visible instead of zero-valued.",
    "sample_size_budget": "Attempted, failed, censored, and unstarted units stay distinct.",
    "acceptance_gate_results": "Validity, readiness, and benefit cannot substitute.",
    "gate_check_summary": "Every terminal failure keeps exact operands and provenance.",
    "honest_verdict": "A complete prefix records terminal state without a retry loop.",
    "verdict_class": "The closed class reserves partial for unfinished owned work.",
    "verifier_is_oracle": "Execution-grounded checks cannot become oracle-distinct benefit.",
    "flagged_adversarial": "Current and retained upstream flags remain separate.",
    "validation_receipts": "Commands, exits, scopes, and hashes make validation auditable.",
    "field_principles": "Each field states the inference failure it prevents.",
    "capstone_complete_score": "One requires all dispositions and scoped validation, not benefit.",
    "task_dispositions": "Every planned task stays visible even when it never started.",
    "branch_conclusions": "Independent claims cannot promote one another.",
    "continuation_rows": "A falsifiable changed prerequisite is required before retry.",
    "retirement_rows": "Literal equality retires only the same completed construction.",
    "publication_gates": "Stable G1-G4 replace a redefinable blocker count.",
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


def _receipt_set_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require exactly one passing receipt for each named command."""

    if not isinstance(receipts, list):
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    counts = Counter(str(row.get("name")) for row in rows)
    by_name = {str(row.get("name")): row for row in rows}
    return all(
        counts[name] == 1
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


def _sample_budget(dispositions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate capstone accounting from underlying producer execution."""

    upstream = dispositions[:-1]
    return {
        "independent_unit": "ordered_v661_task_disposition",
        "planned": 13,
        "attempted": sum(int(row.get("disposition_attempted") is True) for row in dispositions),
        "completed": sum(int(row.get("disposition_completed") is True) for row in dispositions),
        "excluded": sum(
            int(row.get("excluded_from_scientific_metrics") is True) for row in dispositions
        ),
        "failed": sum(int(row.get("producer_failed") is True) for row in dispositions),
        "censored": sum(int(row.get("producer_censored") is True) for row in dispositions),
        "unstarted": sum(int(row.get("producer_unstarted") is True) for row in dispositions),
        "underlying_producer_work": {
            "planned_before_capstone": 12,
            "started": sum(int(row.get("producer_started") is True) for row in upstream),
            "completed_terminal_artifacts": sum(
                int(row.get("producer_completed") is True) for row in upstream
            ),
            "gate_blocked_without_run": sum(
                int(row.get("evidence_state") == "conductor_gate_blocked") for row in upstream
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
            ("preconditions", 12, "twelve_upstreams_observed"),
            ("publication_gate", 4, "g1_g4_read_only"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("reduction", 13, "thirteen_dispositions_reduced"),
            ("validation", 8, "affected_checks_complete"),
            ("terminal_validation", 4, "cold_readers_complete"),
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
    """Build one compact terminal record from authenticated V661 evidence."""

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
    priors = prior_failure_rows(terminal)
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
        and len(dispositions) == 13
        and all(row.get("disposition_completed") is True for row in dispositions)
        and current_complete
    )
    publication = evaluate_publication_gates(root)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7572,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "positive_claim": False,
        "flagged_adversarial": False,
        "retained_source_adversarial_flags": [
            {
                "task_id": task_id,
                "flagged_adversarial": True,
                "excluded_from_scientific_metrics": True,
            }
            for task_id, source in evidence.items()
            if source.get("flagged_adversarial") is True
        ],
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "clock_identity": {"wall": "datetime.now(datetime.UTC)", "monotonic": "time.monotonic_ns"},
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
            "authoring": 0.0,
            "aggregation": max(0.0, duration_s - validation_s),
            "validation": validation_s,
            "historical_upstream": historical_s,
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": deepcopy(RANDOM_SEED),
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
        "prior_failure_rows": priors,
        "retirement_rows": retirement_rows(priors),
        "permanent_exclusion_entries_added": [],
        "continuation_rows": continuation_rows(),
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
            "predict_release_update_persist_reload": {
                "required_for_learning_claim": True,
                "completed": branches[1]["predict_release_update_persist_reload_complete"],
                "upstream_task": "exp7569-decision-learning-audit",
            },
            "numbered_runtime_e2e": "not_applicable_read_only_reporting",
        },
        "verifier_is_oracle": False,
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
        started_at_utc="2026-09-23T00:00:00+00:00",
        completed_at_utc="2026-09-23T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        duration_s=0.0,
        phase_spans=_test_spans(),
    )


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = False
) -> list[str]:
    """Cold-check identity, exact sources, reductions, principles, and checksum."""

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
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "planned_inference_substrate_class",
        "inference_substrate_class",
        "inference_substrate",
        "execution_venue",
        "duration_s",
        "phase_spans",
        "random_seed",
        "preconditions_checked",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "validation_receipts",
        "field_principles",
        "capstone_complete_score",
        "task_dispositions",
        "branch_conclusions",
        "continuation_rows",
        "retirement_rows",
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
    if artifact.get("verdict_class") not in CLOSED_VERDICTS or not str(
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
        priors = prior_failure_rows(terminal)
        failures = failure_rows(contract, evidence, validation)
        if dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        if (
            artifact.get("branch_conclusions") != branches
            or artifact.get("branch_dispositions") != branches
        ):
            errors.append("branch_conclusions_invalid")
        if artifact.get("prior_failure_rows") != priors:
            errors.append("prior_failure_rows_invalid")
        if artifact.get("retirement_rows") != retirement_rows(priors):
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != continuation_rows():
            errors.append("continuation_rows_invalid")
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
            and len(expected_rows) == 13
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
    """Replay all pure reductions without requiring final terminal receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the shared scoped plan for only the affected Exp7572 files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, or command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V661 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def _utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return one aware UTC timestamp for a durable runtime boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed phase line with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7572] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
            (python, "-u", WRAPPER_PATH.as_posix(), "--validate", str(candidate)),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(  # pragma: no cover - exercised through declared entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run scoped checks and atomically publish one validated capstone."""

    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7572-", dir="/tmp"))

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
            checkpoint="twelve_upstreams_observed",
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
            completed_units=13,
            checkpoint="thirteen_dispositions_and_four_branches_reduced",
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
    progress(started, "reduction", "after", completed_units=13)

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
            "write", phase_started, started, completed_units=1, checkpoint="terminal_artifact_ready"
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
    atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic", path=output_path.as_posix())
    return final


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary.
    """Parse the frozen date and two read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the capstone or one read-only fresh-process replay."""

    print("[exp7572] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    candidate_path = args.validate or args.independent_reduce
    if candidate_path is not None:
        try:
            value = load_json(candidate_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            print(json.dumps({"errors": [str(error)]}, sort_keys=True), flush=True)
            return 1
        errors = (
            independent_reduce(value)
            if args.independent_reduce is not None
            else validate_artifact(value, require_terminal=False)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, date_argument(args.date), output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI boundary.
    raise SystemExit(main())
