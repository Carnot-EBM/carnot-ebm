"""Close V658 with fourteen dispositions and independent claim scopes.

This reducer reads completed producer artifacts. It loads no model and changes
no roadmap, model weight, production default, or external publication state.

Spec refs: REQ-REPORT-7529 and SCENARIO-REPORT-7529-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
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
from carnot.experiment_7446_v652_capstone import (
    authenticate_row_manifest,
    authenticate_validation_receipts,
    load_json_object,
    numeric_experiment_id,
    terminal_status,
)
from carnot.experiment_7516_v658_contract_methods import (
    compare_contract_authorities,
    resolve_v658_roadmap,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.658"
EXPERIMENT_ID = "exp7529-capstone"
SCHEMA = "carnot.exp7529.v658.capstone.v1"

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7529_v658_capstone.json")
RAW_DIR = Path("results/raw/experiment_7529_v658_capstone")
MODULE_PATH = Path("python/carnot/experiment_7529_v658_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7529_v658_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7529_v658_capstone.py")
NOTE_PATH = Path("docs/research-notes/v658-capstone.md")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7516, "contract-methods"),
        (7517, "source-protocol"),
        (7518, "source-pilot"),
        (7519, "source-fit-capture"),
        (7520, "source-eval-capture"),
        (7521, "consistency-energy"),
        (7522, "source-evaluation"),
        (7523, "count-memory"),
        (7524, "count-online"),
        (7525, "decision-audit"),
        (7526, "arc-eligibility"),
        (7527, "arc-opportunities"),
        (7528, "service-boundary"),
        (7529, "capstone"),
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
    "arrival": None,
    "bootstrap": None,
    "audit": 7_529_658_01,
    "explanation": "Deterministic aggregation makes no selection, fit, arrival, or bootstrap draw.",
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
    Path("python/carnot/experiment_7515_v657_capstone.py"),
    Path("results/experiment_7515_v657_capstone.json"),
    PUBLICATION_GATE_PATH,
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("ops/north-star.md"),
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


def compare_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare the V658 public task fields through the established parser."""

    return compare_contract_authorities(markdown_text, roadmap)


def load_contract(root: Path) -> JsonDict:
    """Resolve the active V658 YAML and compare it with the Markdown table."""

    selected, roadmap, candidates = resolve_v658_roadmap(root)
    comparison = compare_authorities((root / DESIGN_PATH).read_text(), roadmap)
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V658 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": candidates,
        "roadmap": deepcopy(roadmap),
        "tasks": deepcopy(tasks),
    }


def _ready_value_fields(payload: Mapping[str, Any]) -> JsonDict:
    """Retain bare readiness and value numbers without interpreting benefit."""

    suffixes = ("_score", "_ready", "_complete")
    return {
        str(key): deepcopy(value)
        for key, value in sorted(payload.items())
        if str(key).endswith(suffixes)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _receipt_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Keep command identity and log custody without embedding large logs."""

    fields = ("name", "required", "passed", "exit_code", "log_path", "log_sha256")
    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list):
        return []
    return [
        {key: deepcopy(row.get(key)) for key in fields}
        for row in receipts
        if isinstance(row, Mapping)
    ]


def _required_receipts_pass(payload: Mapping[str, Any]) -> bool:
    """Require every explicit required receipt, or every legacy receipt."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list) or not receipts:
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    required = [row for row in rows if row.get("required") is True] or rows
    return bool(required) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in required
    )


def _missing_producer(task: Mapping[str, Any]) -> JsonDict:
    """Describe an absent upstream output as blocked external evidence."""

    task_id = str(task.get("id") or "")
    path = str(task.get("deliverable") or "")
    failure = {
        "check": "producer_artifact_exists",
        "upstream": task_id,
        "path": path,
        "field": "path",
        "op": "exists",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    return {
        "task_id": task_id,
        "expected_path": path,
        "artifact_path": None,
        "source_sha256": None,
        "source_size_bytes": 0,
        "evidence_state": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "honest_verdict": "complete_blocked_missing_declared_producer_evidence",
        "verdict_class": "blocked",
        "original_verdict_class": None,
        "flagged_adversarial": False,
        "model_invoked": False,
        "inference_substrate": None,
        "inference_substrate_class": None,
        "ready_value_fields": {},
        "row_receipt": {},
        "validation_receipt": {},
        "validation_receipts": [],
        "gate_check_summary": failure,
        "payload": {},
    }


def _invalid_producer(task: Mapping[str, Any], path: Path, error: str) -> JsonDict:
    """Keep malformed present bytes separate from external absence."""

    row = _missing_producer(task)
    row.update(
        artifact_path=str(task.get("deliverable") or path),
        source_sha256=sha256_file(path),
        source_size_bytes=path.stat().st_size,
        evidence_state="invalid",
        available=True,
        honest_verdict="complete_disqualified_unreadable_producer_evidence",
        verdict_class="disqualified",
        gate_check_summary={
            "check": "producer_json_object",
            "upstream": str(task.get("id") or ""),
            "path": str(task.get("deliverable") or path),
            "field": "json_object",
            "op": "is",
            "expected": "mapping",
            "observed": error,
            "passed": False,
        },
    )
    return row


def _acceptance_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve producer gate operands without reclassifying their science."""

    gates = payload.get("acceptance_gate_results")
    if not isinstance(gates, list):
        return []
    fields = (
        "check",
        "category",
        "branch",
        "upstream",
        "path",
        "field",
        "required_field",
        "expected",
        "observed",
        "op",
        "passed",
        "principle",
    )
    return [
        {key: deepcopy(row.get(key)) for key in fields if key in row}
        for row in gates
        if isinstance(row, Mapping)
    ]


def _raw_receipts(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Retain hash-bound raw paths named by a producer."""

    rows = payload.get("source_artifact_hashes")
    if not isinstance(rows, list):
        return []
    return [
        deepcopy(dict(row))
        for row in rows
        if isinstance(row, Mapping) and "raw" in str(row.get("path") or "")
    ]


def _producer_failure(
    task_id: str,
    path: str,
    payload: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Name the first exact present-evidence condition that failed validity."""

    for row in payload.get("validation_receipts") or []:
        if (
            isinstance(row, Mapping)
            and row.get("required") is True
            and (row.get("passed") is not True or row.get("exit_code") != 0)
        ):
            return {
                "check": "producer_required_validation",
                "upstream": task_id,
                "path": str(row.get("log_path") or path),
                "field": str(row.get("name") or "validation_receipt"),
                "op": "==",
                "expected": {"passed": True, "exit_code": 0},
                "observed": {"passed": row.get("passed"), "exit_code": row.get("exit_code")},
                "passed": False,
            }
    if payload.get("flagged_adversarial") is True:
        return {
            "check": "producer_adversarial_flag",
            "upstream": task_id,
            "path": path,
            "field": "flagged_adversarial",
            "op": "==",
            "expected": False,
            "observed": True,
            "passed": False,
        }
    return {
        "check": "producer_validity",
        "upstream": task_id,
        "path": path,
        "field": "verdict_class",
        "op": "!=",
        "expected": "disqualified",
        "observed": payload.get("verdict_class"),
        "passed": False,
        "validation_failures": deepcopy(validation.get("failures", [])),
    }


def load_producer(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one completed V658 producer and its supporting receipts."""

    relative = Path(str(task.get("deliverable") or ""))
    path = root / relative
    if not path.is_file():
        return _missing_producer(task)
    try:
        payload = load_json_object(path)
    except ValueError as error:
        return _invalid_producer(task, path, str(error))

    rows = authenticate_row_manifest(root, payload)
    validation = {
        **authenticate_validation_receipts(root, payload),
        "required_passed": _required_receipts_pass(payload),
    }
    original_class = payload.get("verdict_class")
    authenticated = bool(
        numeric_experiment_id(payload.get("experiment_id", payload.get("experiment")))
        == numeric_experiment_id(task.get("id"))
        and payload.get("milestone") == MILESTONE
        and original_class in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
        and terminal_status(payload)
        and rows.get("authenticated") is True
        and validation.get("authenticated") is True
    )
    valid = bool(
        authenticated
        and validation.get("required_passed") is True
        and original_class != "disqualified"
        and payload.get("flagged_adversarial") is False
    )
    state = "terminal" if valid else "invalid"
    return {
        "task_id": str(task.get("id") or ""),
        "expected_path": relative.as_posix(),
        "artifact_path": relative.as_posix(),
        "source_sha256": sha256_file(path),
        "source_size_bytes": path.stat().st_size,
        "evidence_state": state,
        "authenticated": authenticated,
        "available": True,
        "valid": valid,
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "verdict_class": str(original_class) if valid else "disqualified",
        "original_verdict_class": original_class,
        "flagged_adversarial": payload.get("flagged_adversarial") is True,
        "model_invoked": payload.get("model_invoked") is True,
        "inference_substrate": payload.get("inference_substrate"),
        "inference_substrate_class": payload.get("inference_substrate_class"),
        "ready_value_fields": _ready_value_fields(payload),
        "row_receipt": rows,
        "raw_receipts": _raw_receipts(payload),
        "validation_receipt": validation,
        "validation_receipts": _receipt_rows(payload),
        "acceptance_gates": _acceptance_rows(payload),
        "gate_check_summary": (
            deepcopy(payload.get("gate_check_summary"))
            if valid
            else _producer_failure(
                str(task.get("id") or ""), relative.as_posix(), payload, validation
            )
        ),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Inventory the thirteen earlier tasks in declared conductor order."""

    return {str(task["id"]): load_producer(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, affected_complete: bool, terminal_complete: bool
) -> JsonDict:
    """Classify owned validation before invalid, absent, and null science."""

    if affected_complete and not terminal_complete:
        verdict = "partial"
        honest = "partial_retryable_current_capstone_validation_unfinished"
    elif not affected_complete:
        verdict = "disqualified"
        honest = "complete_disqualified_required_validation"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict = "disqualified"
        honest = "complete_disqualified_required_v658_evidence"
    elif any(
        row.get("evidence_state") == "missing" or row.get("verdict_class") == "blocked"
        for row in rows
    ):
        verdict = "blocked"
        honest = "complete_blocked_required_v658_science_absent_or_externally_gated"
    else:
        verdict = "null"
        honest = "complete_null_v658_accounting_without_aggregate_scientific_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _claim(evidence: Mapping[str, JsonDict], task_id: str, claim: str, **values: Any) -> JsonDict:
    """Attach one claim only to its declared producer and current validity."""

    source = evidence[task_id]
    positive = bool(values.get("qualified_value") == 1)
    if not source["valid"] or source["flagged_adversarial"]:
        positive = False
    return {
        "claim": claim,
        "source_task_id": task_id,
        "source_path": source["artifact_path"],
        "source_sha256": source["source_sha256"],
        "source_valid": source["valid"],
        "source_verdict_class": source["original_verdict_class"],
        "source_honest_verdict": source["honest_verdict"],
        "flagged_adversarial": source["flagged_adversarial"],
        "positive_aggregate_eligible": positive,
        **deepcopy(values),
    }


def build_claim_ledger(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep audited decisions, live opportunity, service cost, and boards separate."""

    decision = evidence["exp7525-decision-audit"]["payload"]
    opportunity = evidence["exp7527-arc-opportunities"]["payload"]
    service = evidence["exp7528-service-boundary"]["payload"]
    return [
        _claim(
            evidence,
            "exp7525-decision-audit",
            "static_probability_value",
            ready_value=decision.get("decision_claims_qualified_score"),
            qualified_value=decision.get("qualified_static_value_score"),
            audit_complete=decision.get("audit_complete_score"),
            successful_audit_is_intervention=False,
        ),
        _claim(
            evidence,
            "exp7525-decision-audit",
            "causal_online_learning",
            ready_value=decision.get("online_claims_qualified_score"),
            qualified_value=decision.get("qualified_online_value_score"),
            audit_complete=decision.get("audit_complete_score"),
            successful_audit_is_intervention=False,
        ),
        _claim(
            evidence,
            "exp7527-arc-opportunities",
            "live_agent_opportunity",
            ready_value=opportunity.get("arc_measurement_complete_score"),
            qualified_value=opportunity.get("opportunity_support_score"),
            effect_estimate=(opportunity.get("raw_reduction") or {}).get("effect_estimate"),
            live_opportunity_only=True,
        ),
        _claim(
            evidence,
            "exp7528-service-boundary",
            "exact_cpu_operation_cost",
            ready_value=service.get("service_cost_complete_score"),
            qualified_value=int(service.get("service_cost_complete_score") == 1),
            whole_service_speedup=service.get("whole_service_speedup"),
            execution_venue="host_cpu",
        ),
        _claim(
            evidence,
            "exp7528-service-boundary",
            "dated_board_continuity",
            ready_value=service.get("board_continuity_complete_score"),
            qualified_value=service.get("board_continuity_complete_score"),
            board_rows=deepcopy(service.get("board_rows") or []),
            positive_scientific_claim=False,
        ),
    ]


def _disposition(source: Mapping[str, Any], task: Mapping[str, Any], order: int) -> JsonDict:
    """Reduce one producer without changing its readiness or scientific class."""

    blocked_or_missing = bool(
        source["evidence_state"] in {"missing", "invalid"}
        or source["verdict_class"] in {"blocked", "disqualified"}
        or source["flagged_adversarial"]
    )
    return {
        "order": order,
        "task_id": str(task["id"]),
        "expected_artifact_path": source["expected_path"],
        "artifact_path": source["artifact_path"],
        "artifact_sha256": source["source_sha256"],
        "evidence_state": source["evidence_state"],
        "honest_verdict": source["honest_verdict"],
        "verdict_class": source["verdict_class"],
        "original_verdict_class": source["original_verdict_class"],
        "flagged_adversarial": source["flagged_adversarial"],
        "ready_value_fields": deepcopy(source["ready_value_fields"]),
        "row_receipt": deepcopy(source["row_receipt"]),
        "raw_receipts": deepcopy(source.get("raw_receipts") or []),
        "validation_receipt": deepcopy(source["validation_receipt"]),
        "validation_receipts": deepcopy(source["validation_receipts"]),
        "acceptance_gates": deepcopy(source.get("acceptance_gates") or []),
        "gate_check_summary": deepcopy(source["gate_check_summary"]),
        "attempted": source["evidence_state"] != "missing",
        "completed": source["evidence_state"] != "missing",
        "failed": source["evidence_state"] == "invalid",
        "censored": source["evidence_state"] == "missing",
        "unstarted": source["evidence_state"] == "missing",
        "excluded_from_positive_aggregate": blocked_or_missing,
        "principle": "Literal source custody prevents readiness or verdict laundering.",
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    *,
    current_validation_complete: bool,
) -> list[JsonDict]:
    """Build thirteen source rows and one non-self-dependent current row."""

    rows = [
        _disposition(evidence[str(task["id"])], task, order)
        for order, task in enumerate(tasks[:-1], 1)
    ]
    rows.append(
        {
            "order": 14,
            "task_id": str(tasks[-1]["id"]),
            "expected_artifact_path": str(tasks[-1]["deliverable"]),
            "artifact_path": None,
            "artifact_sha256": None,
            "evidence_state": "current_terminal"
            if current_validation_complete
            else "current_running",
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "original_verdict_class": terminal["verdict_class"],
            "flagged_adversarial": False,
            "ready_value_fields": {"capstone_complete_score": int(current_validation_complete)},
            "row_receipt": {},
            "raw_receipts": [],
            "validation_receipt": {"required_passed": current_validation_complete},
            "validation_receipts": [],
            "acceptance_gates": [],
            "gate_check_summary": {},
            "attempted": True,
            "completed": current_validation_complete,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "excluded_from_positive_aggregate": True,
            "principle": "Current work cannot authenticate itself through its future result path.",
        }
    )
    return rows


def _current_substantive_state(task_id: str, source: Mapping[str, Any] | None) -> str:
    """Name current mechanisms separately from versioned verdict wording."""

    if source is None:
        return "current_capstone_accounting"
    if source.get("evidence_state") == "missing":
        return "external_producer_absence"
    verdict = str(source.get("honest_verdict") or "")
    if task_id == "exp7517-source-protocol":
        return "external_fresh_source_inventory_absence"
    if task_id == "exp7525-decision-audit":
        return "external_required_science_absence"
    if task_id == "exp7526-arc-eligibility":
        return "eligible_recommendations_without_applied_effect_support"
    if task_id == "exp7527-arc-opportunities" and "owned_gpu" in verdict:
        return "environmental_owned_gpu_absence"
    if task_id == "exp7528-service-boundary":
        return "external_count_memory_prototype_absence"
    return verdict or "unknown_current_state"


def _prior_substantive_state(verdict: object) -> str:
    """Normalize only known prior meanings, not arbitrary version strings."""

    text = str(verdict or "")
    meanings = {
        "complete_null_zero_eligible_supervisor_opportunity": "zero_eligible_supervisor_opportunity",
        "complete_null_static_evaluation_exploratory_prior_exposure": "static_exploratory_prior_exposure",
        "complete_null_causal_online_measurement_valid_benefit_gate_failed": "causal_online_benefit_gate_failed",
        "complete_disqualified_required_validation": "required_validation_disqualified",
        "complete_disqualified_required_v657_evidence": "required_v657_evidence_disqualified",
        "FAIL": "unfinished_without_terminal_artifact",
    }
    return meanings.get(text, text)


def reduce_prior_failures(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Compare literal prior verdicts and separately inspect mechanism repeats."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        priors = task.get("prior_failures") or []
        if not priors or any(not isinstance(prior, Mapping) for prior in priors):
            raise ValueError(f"at least one prior failure required for {task_id}")
        source = evidence.get(task_id)
        current_verdict = source.get("honest_verdict") if source else None
        current_class = source.get("verdict_class") if source else None
        current_state = _current_substantive_state(task_id, source)
        environmental = current_state.startswith("environmental_") or current_state.startswith(
            "external_"
        )
        for prior in priors:
            prior_state = _prior_substantive_state(prior.get("verdict"))
            literal = bool(source is not None and current_verdict == prior.get("verdict"))
            substantive = bool(source is not None and current_state == prior_state)
            triggered = bool(
                prior.get("retire_if_same_verdict") is True
                and not environmental
                and (literal or substantive)
            )
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment": prior.get("experiment_id"),
                    "prior_honest_verdict": prior.get("verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "current_honest_verdict": current_verdict,
                    "current_verdict_class": current_class,
                    "exact_text_match": literal,
                    "prior_substantive_state": prior_state,
                    "current_substantive_state": current_state,
                    "substantive_repeat": substantive,
                    "environmental_absence": environmental,
                    "retirement_triggered": triggered,
                }
            )
    return rows


def retirement_rows(prior_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retire only repeated scientific mechanisms, never missing resources."""

    return [
        {
            "retired_task_id": str(row.get("task_id")),
            "prior_experiment": row.get("prior_experiment"),
            "prior_honest_verdict": row.get("prior_honest_verdict"),
            "current_honest_verdict": row.get("current_honest_verdict"),
            "prior_substantive_state": row.get("prior_substantive_state"),
            "current_substantive_state": row.get("current_substantive_state"),
            "changed_mechanism": row.get("addressed_by"),
            "retire_if_same_verdict": row.get("retire_if_same_verdict"),
            "scope": "bounded_scientific_mechanism_only",
            "exclusion_workflow": "ops/exclusion_manifest.yaml",
        }
        for row in prior_rows
        if row.get("retirement_triggered") is True
    ]


def next_conditions() -> list[JsonDict]:
    """Require a measurable change before reopening each closed branch."""

    return [
        {
            "branch": "source_probability_and_causal_learning",
            "state": "blocked_external_source_inventory",
            "reopen_when": "Provide 480 fresh eligible source groups under the sealed Exp7517 roles, then run Exp7518 through Exp7525 in order.",
        },
        {
            "branch": "live_agent_opportunity",
            "state": "blocked_environmental_gpu_ownership",
            "reopen_when": "Acquire exclusive ownership of real CUDA and complete the registered twelve adapter-withheld live episodes; do not infer idleness from an end-state monitor.",
        },
        {
            "branch": "applied_supervisor_effect",
            "state": "closed_unmeasured",
            "reopen_when": "Use a separately registered policy that applies eligible mutations in at least ten independent episodes before estimating efficacy.",
        },
        {
            "branch": "exact_count_memory_service_cost",
            "state": "blocked_missing_count_memory_prototype",
            "reopen_when": "Land the exact eight-bin Exp7523 operation, then measure both durable CPU service arms with thirty independent batches each.",
        },
        {
            "branch": "gatemate_fabric",
            "state": "blocked_unchanged_physical_prerequisite",
            "reopen_when": "Provide a dated operator-authored cable, port, power, board, JTAG, or DirtyJTAG change receipt newer than Exp6559.",
        },
    ]


def build_retrospective(artifact: Mapping[str, Any]) -> JsonDict:
    """Reduce actual rows into three bounded PRD gap updates and timing."""

    dispositions = artifact.get("task_dispositions") or []
    claims = {row["claim"]: row for row in artifact.get("claim_ledger") or []}
    return {
        "task_dispositions": [
            {
                "order": row.get("order"),
                "task_id": row.get("task_id"),
                "verdict_class": row.get("verdict_class"),
                "honest_verdict": row.get("honest_verdict"),
                "artifact_path": row.get("artifact_path") or row.get("expected_artifact_path"),
            }
            for row in dispositions
        ],
        "prd_gap_updates": [
            {
                "gap": "Probability and causal learning",
                "status": "blocked_fresh_source_inventory",
                "evidence": (
                    "Exp7525 completed its audit machinery, but static and online claim "
                    f"qualification remain {claims.get('static_probability_value', {}).get('ready_value')} "
                    f"and {claims.get('causal_online_learning', {}).get('ready_value')}."
                ),
            },
            {
                "gap": "Live-agent opportunity",
                "status": "eligibility_ready_live_panel_blocked",
                "evidence": "Exp7526 found explicit eligible and selected events without applied mutations. Exp7527 did not obtain owned GPU execution.",
            },
            {
                "gap": "Service cost and board continuity",
                "status": "board_continuity_ready_exact_cost_blocked",
                "evidence": "Exp7528 preserved dated board rows, but the missing Exp7523 prototype prevented exact current CPU cost measurement.",
            },
        ],
        "wall_time_phase_budget": {
            "current_duration_s": artifact.get("duration_s"),
            "historical_producer_duration_s": (artifact.get("duration_components_s") or {}).get(
                "historical_capture"
            ),
            "phase_spans": deepcopy(artifact.get("phase_spans") or []),
            "declared_estimated_wall_time_min": 25,
        },
        "aggregate_verdict_class": artifact.get("verdict_class"),
        "capstone_complete_score": artifact.get("capstone_complete_score"),
        "next_conditions": deepcopy(artifact.get("next_conditions") or []),
    }


def retrospective_markdown(value: Mapping[str, Any]) -> str:
    """Render the fourteen dispositions, three gaps, timing, and reopen gates."""

    lines = [
        "# V658 capstone",
        "",
        "Milestone accounting is complete independently from scientific benefit.",
        "",
        "## Fourteen dispositions",
        "",
        "| Task | Class | Literal disposition | Artifact |",
        "|---|---|---|---|",
    ]
    for row in value.get("task_dispositions") or []:
        lines.append(
            f"| {row['task_id']} | {row['verdict_class']} | `{row['honest_verdict']}` | `{row['artifact_path']}` |"
        )
    lines.extend(["", "## Three PRD gaps", ""])
    for row in value.get("prd_gap_updates") or []:
        lines.extend([f"### {row['gap']}", "", f"Status: `{row['status']}`. {row['evidence']}", ""])
    budget = value.get("wall_time_phase_budget") or {}
    lines.extend(
        [
            "## Wall time and phase budget",
            "",
            f"Current measured duration: `{budget.get('current_duration_s')}` seconds.",
            f"Declared planning budget: `{budget.get('declared_estimated_wall_time_min')}` minutes.",
            "",
            "## Reopening conditions",
            "",
            *[
                f"- `{row['branch']}`: {row['reopen_when']}"
                for row in value.get("next_conditions") or []
            ],
            "",
            "Publication and roadmap activation remain operator actions.",
            "",
        ]
    )
    return "\n".join(lines)


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
    """Store each gate operand so a reader can repeat the comparison."""

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
    evidence: Mapping[str, JsonDict],
    dispositions: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep current validity, accounting readiness, and benefit independent."""

    by_claim = {str(row["claim"]): row for row in claims}
    present_valid = all(
        row["valid"] for row in evidence.values() if row["evidence_state"] != "missing"
    )
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    validity = "Favorable science cannot excuse invalid evidence."
    readiness = "A valid null remains eligible for auditing."
    benefit = "A hypothesis or successful audit is not a successful intervention."
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v658_contract",
            str(contract.get("selected_roadmap_path")),
            "comparison_passed",
            True,
            contract.get("comparison_passed"),
            contract.get("comparison_passed") is True,
            validity,
        ),
        _gate(
            "all_present_producer_evidence_valid",
            "validity",
            "v658_producers",
            "task_dispositions",
            "valid",
            True,
            present_valid,
            present_valid,
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
            "fourteen_dispositions_recorded",
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
            "static_probability_claim_qualified",
            "benefit",
            "exp7525-decision-audit",
            "claim_ledger.static_probability_value",
            "qualified_value",
            1,
            by_claim["static_probability_value"]["qualified_value"],
            by_claim["static_probability_value"]["qualified_value"] == 1,
            benefit,
        ),
        _gate(
            "causal_online_claim_qualified",
            "benefit",
            "exp7525-decision-audit",
            "claim_ledger.causal_online_learning",
            "qualified_value",
            1,
            by_claim["causal_online_learning"]["qualified_value"],
            by_claim["causal_online_learning"]["qualified_value"] == 1,
            benefit,
        ),
        _gate(
            "live_agent_opportunity_supported",
            "benefit",
            "exp7527-arc-opportunities",
            "claim_ledger.live_agent_opportunity",
            "qualified_value",
            1,
            by_claim["live_agent_opportunity"]["qualified_value"],
            by_claim["live_agent_opportunity"]["qualified_value"] == 1,
            benefit,
        ),
        _gate(
            "exact_cpu_operation_cost_complete",
            "readiness",
            "exp7528-service-boundary",
            "claim_ledger.exact_cpu_operation_cost",
            "ready_value",
            1,
            by_claim["exact_cpu_operation_cost"]["ready_value"],
            by_claim["exact_cpu_operation_cost"]["ready_value"] == 1,
            readiness,
        ),
        _gate(
            "dated_board_continuity_complete",
            "readiness",
            "exp7528-service-boundary",
            "claim_ledger.dated_board_continuity",
            "ready_value",
            1,
            by_claim["dated_board_continuity"]["ready_value"],
            by_claim["dated_board_continuity"]["ready_value"] == 1,
            readiness,
        ),
    ]


def _normalized_blocked_failure(source: Mapping[str, Any]) -> JsonDict:
    """Give every blocked producer an exact path, field, and operands."""

    summary = source.get("gate_check_summary")
    if isinstance(summary, Mapping):
        candidate = summary.get("first_failure") or summary.get("failed_check") or summary
        if isinstance(candidate, Mapping):
            row = deepcopy(dict(candidate))
            row.setdefault("upstream", source.get("task_id"))
            row.setdefault("path", source.get("artifact_path") or source.get("expected_path"))
            row.setdefault("field", row.get("check") or "verdict_class")
            row.setdefault("op", "==")
            row.setdefault("expected", "unblocked_terminal_evidence")
            row.setdefault("observed", source.get("honest_verdict"))
            row["passed"] = False
            if row.get("path") is None:
                row["path"] = source.get("artifact_path") or source.get("expected_path")
            if row.get("field") is None:
                row["field"] = row.get("check") or "verdict_class"
            return row
    return {
        "check": "producer_terminal_disposition",
        "upstream": source.get("task_id"),
        "path": source.get("artifact_path") or source.get("expected_path"),
        "field": "verdict_class",
        "op": "!=",
        "expected": "blocked",
        "observed": "blocked",
        "passed": False,
    }


def failure_rows(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Return failures that determine invalid, blocked, or unfinished state."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "check": "contract_authorities_agree",
                "upstream": "v658_contract",
                "path": str(contract.get("selected_roadmap_path")),
                "field": "comparison_passed",
                "op": "==",
                "expected": True,
                "observed": contract.get("comparison_passed"),
                "passed": False,
            }
        )
    for source in evidence.values():
        if source["evidence_state"] in {"missing", "invalid"}:
            failures.append(deepcopy(dict(source["gate_check_summary"])))
        elif source["verdict_class"] == "blocked":
            failures.append(_normalized_blocked_failure(source))
    for field in ("required_checks_passed", "terminal_validation_passed"):
        if validation.get(field) is not True:
            failures.append(
                {
                    "check": "current_" + field,
                    "upstream": EXPERIMENT_ID,
                    "path": "validation_receipts",
                    "field": field,
                    "op": "==",
                    "expected": True,
                    "observed": validation.get(field),
                    "passed": False,
                }
            )
    return failures


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every classification failure and its first exact cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


def _preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record repository resources and observed producer states before reduction."""

    rows = []
    for relative in INPUT_PATHS:
        path = root / relative
        exists = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "owner": "repository",
                "field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if exists else "absent_or_empty",
                "passed": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    rows.append(
        {
            "check": "contract_authority_equivalence",
            "upstream": "v658_contract",
            "path": f"{DESIGN_PATH} + {contract['selected_roadmap_path']}",
            "owner": "repository",
            "field": "comparison_passed",
            "expected": True,
            "observed": contract["comparison_passed"],
            "passed": contract["comparison_passed"] is True,
        }
    )
    rows.extend(
        {
            "check": "producer_state_observed",
            "upstream": task_id,
            "path": source["artifact_path"] or source["expected_path"],
            "owner": "upstream_producer",
            "field": "evidence_state",
            "expected": "one explicit terminal, invalid, or missing state",
            "observed": source["evidence_state"],
            "passed": source["evidence_state"] in {"terminal", "invalid", "missing"},
            "sha256": source["source_sha256"],
        }
        for task_id, source in evidence.items()
    )
    return rows


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Bind instructions, authorities, current code, note, and producer bytes."""

    paths = [*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH]
    if (root / NOTE_PATH).is_file():
        paths.append(NOTE_PATH)
    selected = Path(str(contract["selected_roadmap_path"]))
    if selected not in paths:
        paths.append(selected)
    rows = [
        {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "evidence_class": "current_source",
        }
        for relative in paths
    ]
    rows.extend(
        {
            "path": source["artifact_path"] or source["expected_path"],
            "sha256": source["source_sha256"],
            "evidence_class": source["evidence_state"],
            "task_id": task_id,
            "original_honest_verdict": source["honest_verdict"],
            "original_verdict_class": source["original_verdict_class"],
            "original_flagged_adversarial": source["flagged_adversarial"],
        }
        for task_id, source in evidence.items()
    )
    return rows


def _source_hashes_match(value: Mapping[str, Any], root: Path) -> bool:
    """Recheck every source row that claims existing exact bytes."""

    rows = value.get("source_artifact_hashes")
    if not isinstance(rows, list):
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


@lru_cache(maxsize=2)
def evaluate_publication_gates(root: Path) -> JsonDict:
    """Run the fixed G1-G4 reader and retain its read-only result."""

    command = (str(root / ".venv/bin/python"), "-u", PUBLICATION_GATE_PATH.as_posix(), "--json")
    completed = subprocess.run(
        command, cwd=root, text=True, capture_output=True, timeout=120, check=False
    )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {"paper_ready": False, "gates": {}, "unmet_gates": ["reader_failed"]}
    return {
        **parsed,
        "command_argv": list(command),
        "exit_code": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "stdout_sha256": "sha256:" + hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "publication_performed": False,
    }


FIELD_PRINCIPLES = {
    "schema": "Version, experiment identity, and milestone prevent terminal reader drift.",
    "run_date": "The frozen date stays distinct from measured UTC and monotonic boundaries.",
    "preconditions_checked": "Exact resources and hashes prevent invented readiness.",
    "MODEL_SPECS": "An empty declaration prevents historical model calls from becoming current calls.",
    "model_specs": "Both model-spec aliases stay empty for aggregation-only work.",
    "model_invoked": "Current model work stays distinct from historical producer provenance.",
    "invocation_counts": "Balanced zero counters expose invented model operations.",
    "inference_substrate_class": "The aggregation class prevents model or duration implications.",
    "inference_substrate": "The exact aggregation label distinguishes artifact reduction from inference.",
    "execution_venue": "Host aggregation stays distinct from CUDA, board CPU, and FPGA evidence.",
    "duration_s": "Measured current work stays separate from authoring and historical capture.",
    "phase_spans": "Measured boundaries expose waiting, validation, and unfinished work.",
    "random_seed": "Frozen applicable seeds prevent favorable rerun selection.",
    "reproducibility_checksum": "One digest binds code, settings, roles, source rows, and reductions.",
    "source_artifact_hashes": "Exact bytes preserve exposure history, verdicts, and flags.",
    "rows": "Fourteen ordered rows prevent missing tasks from becoming zero-valued evidence.",
    "sample_size_budget": "Planned, attempted, completed, excluded, failed, censored, and unstarted units remain separate.",
    "acceptance_gate_results": "Validity, readiness, and benefit cannot substitute for one another.",
    "gate_check_summary": "Every blocked state names exact expected and observed operands.",
    "honest_verdict": "A complete prefix records terminal accounting without softening absence.",
    "verdict_class": "The closed enum reserves partial for retryable current work only.",
    "verifier_is_oracle": "Fixtures and human labels do not become formal proof.",
    "flagged_adversarial": "A retained source flag cannot be cleared to open a claim gate.",
    "validation_receipts": "Exact commands, exits, scopes, and hashes make current checks auditable.",
    "field_principles": "Every top-level field states the inference failure it prevents.",
    "capstone_complete_score": "A bare one records complete accounting, not scientific success.",
    "task_dispositions": "Exactly fourteen ordered states preserve every absence, null, and flag.",
    "claim_ledger": "Independent scopes prevent audit, intervention, opportunity, and cost promotion.",
    "retirement_rows": "Only repeated unchanged scientific mechanisms retire.",
    "publication_gate_results": "Read-only G1-G4 results do not authorize publication.",
    "next_conditions": "A measurable new prerequisite or mechanism is required before reopening.",
}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give each top-level field one plain failure-prevention explanation."""

    return {
        field: FIELD_PRINCIPLES.get(
            field,
            f"Exact {field} evidence prevents a reader from inferring an unstated result.",
        )
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding real clock and process observations."""

    excluded = {
        "reproducibility_checksum",
        "field_principles",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "duration_components_s",
        "phase_spans",
        "process_identity",
        "device_identity",
        "retrospective",
    }
    return canonical_hash({key: item for key, item in value.items() if key not in excluded})


def _receipt_set_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require exactly one passing receipt for each declared command name."""

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
    """Build one compact terminal record from authenticated V658 evidence."""

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
    claims = build_claim_ledger(evidence)
    priors = reduce_prior_failures(contract["tasks"], evidence)
    failures = failure_rows(contract, evidence, validation)
    receipts = deepcopy(validation.get("validation_receipts") or [])
    validation_s = sum(
        float(row.get("duration_s") or 0.0) for row in receipts if isinstance(row, Mapping)
    )
    historical_s = sum(
        float(source["payload"].get("duration_s") or 0.0)
        for source in evidence.values()
        if source["available"] and isinstance(source["payload"], Mapping)
    )
    capstone_complete = int(
        contract.get("comparison_passed") is True and len(dispositions) == 14 and current_complete
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": any(row["flagged_adversarial"] for row in evidence.values()),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "clock_identity": {
            "wall": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
        },
        "process_identity": {
            "pid": os.getpid(),
            "node": platform.node(),
            "source_revision": _source_revision(root),
        },
        "device_identity": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cuda_used": False,
        },
        "duration_components_s": {
            "authoring": 0.0,
            "computation": max(0.0, duration_s - validation_s),
            "validation": validation_s,
            "historical_capture": historical_s,
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
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "execution_venue_detail": "host_cpu",
        "preconditions_checked": _preconditions(root, contract, evidence),
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_model_receipts": [
            {
                "task_id": task_id,
                "path": source["artifact_path"],
                "sha256": source["source_sha256"],
                "historical_model_invoked": source["model_invoked"],
                "counted_as_current_invocation": False,
            }
            for task_id, source in evidence.items()
            if source["available"]
        ],
    }
    artifact.update(
        {
            "rows": deepcopy(dispositions),
            "task_dispositions": dispositions,
            "sample_size_budget": {
                "independent_unit": "ordered_v658_task_disposition",
                "planned": 14,
                "attempted": sum(int(row["attempted"]) for row in dispositions),
                "completed": sum(int(row["completed"]) for row in dispositions),
                "excluded": sum(
                    int(row["excluded_from_positive_aggregate"]) for row in dispositions
                ),
                "failed": sum(int(row["failed"]) for row in dispositions),
                "censored": sum(int(row["censored"]) for row in dispositions),
                "unstarted": sum(int(row["unstarted"]) for row in dispositions),
            },
            "claim_ledger": claims,
            "prior_failure_rows": priors,
            "retirement_rows": retirement_rows(priors),
            "permanent_exclusion_entries_added": [],
            "next_conditions": next_conditions(),
            "publication_gate_results": deepcopy(evaluate_publication_gates(root)),
            "validation_manifest": {
                "experiment_id": VALIDATION_MANIFEST.experiment_id,
                "test_paths": list(VALIDATION_MANIFEST.test_paths),
                "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
                "static_paths": list(VALIDATION_MANIFEST.static_paths),
                "frozen_before_checks": True,
            },
            "validation_receipts": receipts,
            "repository_health": deepcopy(
                validation.get("repository_health", {"status": "outside_current_affected_validity"})
            ),
            "acceptance_gate_results": acceptance_gates(
                contract, evidence, dispositions, claims, validation
            ),
            "gate_check_summary": _gate_summary(failures),
            "capstone_complete_score": capstone_complete,
            "capability_e2e": {
                "declared_entrypoint": "required",
                "fresh_process_replay": "required",
                "numbered_runtime_e2e": "not_applicable_reporting_only",
            },
            "verifier_is_oracle": False,
            "roadmap_activation_performed": False,
            "publication_performed": False,
            "submission_performed": False,
            "external_contact_performed": False,
            "production_defaults_changed": False,
            "generator_weights_changed": False,
            "push_performed": False,
            "research_conductor_modified": False,
        }
    )
    artifact["retrospective"] = build_retrospective(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _source_revision(root: Path) -> str:
    """Record the exact Git revision without changing repository state."""

    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _passing_receipts() -> list[JsonDict]:
    """Create deterministic successful receipts for pure unit construction."""

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
    """Return deterministic phase checkpoints for unit artifact construction."""

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
            ("preconditions", 13, "thirteen_producers_observed"),
            ("publication_gate", 4, "g1_g4_read_only"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("reduction", 14, "fourteen_dispositions_reduced"),
            ("validation", 8, "affected_checks_complete"),
            ("terminal_validation", 4, "cold_readers_complete"),
            ("write", 1, "terminal_artifact_ready"),
        )
    ]


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic terminal artifact from current repository evidence."""

    contract = load_contract(REPO_ROOT)
    evidence = collect_evidence(REPO_ROOT, contract["tasks"])
    receipts = _passing_receipts()
    return build_artifact(
        REPO_ROOT,
        contract,
        evidence,
        {
            "required_checks_passed": True,
            "terminal_validation_passed": True,
            "validation_receipts": receipts,
            "repository_health": {"status": "outside_current_affected_validity"},
        },
        started_at_utc="2026-09-22T00:00:00+00:00",
        completed_at_utc="2026-09-22T00:00:00+00:00",
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
        "flagged_adversarial",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
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
        "claim_ledger",
        "retirement_rows",
        "publication_gate_results",
        "next_conditions",
        "reproducibility_checksum",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]

    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
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
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
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
        claims = build_claim_ledger(evidence)
        priors = reduce_prior_failures(contract["tasks"], evidence)
        failures = failure_rows(contract, evidence, validation)
        if dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        if artifact.get("claim_ledger") != claims:
            errors.append("claim_ledger_invalid")
        if artifact.get("prior_failure_rows") != priors:
            errors.append("prior_failure_rows_invalid")
        if artifact.get("retirement_rows") != retirement_rows(priors):
            errors.append("retirement_rows_invalid")
        if artifact.get("next_conditions") != next_conditions():
            errors.append("next_conditions_invalid")
        if (
            artifact.get("status") != terminal["status"]
            or artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != _gate_summary(failures)
        ):
            errors.append("terminal_reduction_invalid")
        expected_gates = acceptance_gates(contract, evidence, expected_rows, claims, validation)
        if artifact.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gates_invalid")
        expected_complete = int(
            contract["comparison_passed"] is True and len(expected_rows) == 14 and current_complete
        )
        score = artifact.get("capstone_complete_score")
        if not isinstance(score, int) or isinstance(score, bool) or score != expected_complete:
            errors.append("capstone_score_invalid")
        if artifact.get("retrospective") != build_retrospective(artifact):
            errors.append("retrospective_invalid")
        if artifact.get("publication_gate_results") != evaluate_publication_gates(root):
            errors.append("publication_gate_results_invalid")
        if require_terminal and not validation["terminal_validation_passed"]:
            errors.append("terminal_validation_incomplete")
    except (KeyError, OSError, TypeError, ValueError):
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
    """Build the Exp7358 plan for only the affected Exp7529 files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V658 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def _utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return one aware UTC timestamp for a durable runtime boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit one flushed phase line with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7529] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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


def _atomic_text(path: Path, text: str) -> None:  # pragma: no cover - durable write boundary.
    """Replace the generated note only after its bytes reach local storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _terminal_commands(  # pragma: no cover - subprocess plan boundary.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build replay, independent reduction, and both strict candidate guards."""

    python = str(root / ".venv/bin/python")
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--validate", str(candidate)),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
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

    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7529-", dir="/tmp"))

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in INPUT_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            completed_units=len(evidence),
            checkpoint="thirteen_producers_observed",
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
        spans.append(
            _span(
                phase,
                phase_started,
                started,
                completed_units=0,
                checkpoint=checkpoint,
            )
        )
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
    preliminary = build_artifact(
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
    _atomic_text(root / NOTE_PATH, retrospective_markdown(preliminary["retrospective"]))
    spans.append(
        _span(
            "reduction",
            phase_started,
            started,
            completed_units=14,
            checkpoint="fourteen_dispositions_and_note_reduced",
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
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    validation["terminal_validation_passed"] = bool(
        all(row.get("passed") is True for row in terminal_receipts) and not critical
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

    print("[exp7529] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    candidate_path = args.validate or args.independent_reduce
    if candidate_path is not None:
        try:
            value = load_json_object(candidate_path)
        except ValueError as error:
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
