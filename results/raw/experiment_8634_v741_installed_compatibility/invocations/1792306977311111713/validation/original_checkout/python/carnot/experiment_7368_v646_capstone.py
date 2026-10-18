"""Reconcile V646 without converting valid nulls into missing work.

This module reads existing artifacts and canonical conductor gate records. It
does not invoke a model, contact hardware, or publish a result.

Spec refs: REQ-REPORT-7368 and SCENARIO-REPORT-7368-*.
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
import re
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as quarantine_helpers
from carnot import experiment_7342_v644_capstone as shared
from carnot import experiment_7358_v646_validation_contract as validation_boundary
from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.646"
RUN_DATE = "20260917"
EXPERIMENT_ID = "exp7368-capstone"
SCHEMA = "carnot.experiment_7368.v646_capstone.v1"
RESULT_PATH = Path("results/experiment_7368_v646_capstone.json")
RAW_DIR = Path("results/raw/experiment_7368_v646_capstone")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7368_v646_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_7368_v646_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7368_v646_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7368_v646_capstone.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ROADMAP_PATH = Path("research-roadmap.yaml")
HISTORICAL_ARC_ARTIFACT = Path("results/experiment_7354_v645_arc_transfer.json")
HISTORICAL_ARC_ROWS = Path("results/raw/experiment_7354_v645_arc_transfer/episode_rows.json")
EXCLUSION_RECEIPT_ID = "v646_learning_audit_pregate_repeat_retired_exp7363"

EXPECTED_TASK_IDS = (
    "exp7357-contract",
    "exp7358-validation-contract",
    "exp7359-capture-reducer",
    "exp7360-learning-fixture",
    "exp7361-fresh-plan-capture",
    "exp7362-prospective-learning",
    "exp7363-learning-audit",
    "exp7364-acquisition-adjudication",
    "exp7365-supervisor-support",
    "exp7366-supervisor-live",
    "exp7367-board-disposition",
    EXPERIMENT_ID,
)

EXPECTED_EVIDENCE_STATES: dict[str, JsonDict] = {
    "exp7357-contract": {
        "path": "results/experiment_7357_v646_contract.json",
        "class": "disqualified",
    },
    "exp7358-validation-contract": {
        "path": "results/experiment_7358_v646_validation_contract.json",
        "class": "null",
    },
    "exp7359-capture-reducer": {
        "path": "results/experiment_7359_v646_capture_reducer.json",
        "class": "null",
    },
    "exp7360-learning-fixture": {
        "path": "results/experiment_7360_v646_learning_fixture.json",
        "class": "null",
    },
    "exp7361-fresh-plan-capture": {
        "path": "results/experiment_7361_v646_fresh_plan_capture.json",
        "class": "circular_positive",
    },
    "exp7362-prospective-learning": {
        "path": "results/experiment_7362_v646_prospective_learning.json",
        "class": "disqualified",
    },
    "exp7363-learning-audit": {
        "path": "results/experiment_7363_learning_audit.json",
        "class": "blocked",
    },
    "exp7364-acquisition-adjudication": {
        "path": "results/experiment_7364_v646_acquisition_adjudication.json",
        "class": "null",
    },
    "exp7365-supervisor-support": {
        "path": "results/experiment_7365_v646_supervisor_support.json",
        "class": "null",
    },
    "exp7366-supervisor-live": {
        "path": "results/experiment_7366_supervisor_live.json",
        "class": "blocked",
    },
    "exp7367-board-disposition": {
        "path": "results/experiment_7367_v646_board_disposition.json",
        "class": "blocked",
    },
}

CANONICAL_GATE_RECORDS: dict[str, JsonDict] = {
    "exp7363-learning-audit": {
        "path": "results/experiment_7363_learning_audit.json",
        "upstream": "exp7362-prospective-learning",
        "field": "learning_capture_complete_score",
        "expected": 1,
        "observed": 0,
    },
    "exp7366-supervisor-live": {
        "path": "results/experiment_7366_supervisor_live.json",
        "upstream": "exp7365-supervisor-support",
        "field": "supervisor_trial_ready_score",
        "expected": 1,
        "observed": 0,
    },
}

CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ELIGIBLE_VERDICTS = {"positive", "circular_positive", "null"}
REQUIRED_SCIENCE_TASKS = (
    "exp7361-fresh-plan-capture",
    "exp7362-prospective-learning",
    "exp7363-learning-audit",
)
CLAIM_NAMES = (
    "fresh_source_fidelity",
    "structural_learning",
    "learning_audit",
    "acquisition_adjudication",
    "supervisor_support_and_live",
    "board_dispositions",
)
ZERO_INVOCATION_COUNTS = deepcopy(validation_boundary.ZERO_INVOCATION_COUNTS)
RANDOM_SEED = {
    "development": 7_368_202_609_17,
    "evaluation": 7_368_202_609_18,
    "resampling": 7_368_202_609_19,
}

V646_MANIFEST = validation_boundary.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_VALIDATION_NAMES = scoped.REQUIRED_CHECK_NAMES
REQUIRED_VALIDATION_NAMES = (
    *AFFECTED_VALIDATION_NAMES,
    "full_python_suite",
    "independent_raw_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

FIELD_PRINCIPLES = {
    "schema": "Version this record while keeping ordinary experiment and milestone identities.",
    "status": "Use a terminal status only after the actual reconciliation and affected checks.",
    "run_date": "Use 20260917 and retain actual UTC boundary timestamps.",
    "preconditions_checked": "Record exact paths, fields, classes, flags, and hashes before reduction.",
    "MODEL_SPECS": "Keep this empty because this aggregation task intends no current LLM work.",
    "model_invoked": "Set true for any attempted current model load or generation, including failure.",
    "invocation_counts": "Separate current zero counts from labeled historical receipts.",
    "inference_substrate": "Describe host aggregation; historical inference stays in a sidecar.",
    "inference_substrate_class": "Use the closed aggregation duration class without padding.",
    "execution_venue": "Record host computation; V646 performs no new board execution.",
    "duration_s": "Use measured monotonic elapsed time and no synthetic delay.",
    "phase_spans": "Keep disjoint measured load, generation, evaluation, validation, and write spans.",
    "random_seed": "Freeze development, evaluation, and resampling seeds.",
    "reproducibility_checksum": "Bind code, settings, evaluator, inputs, and raw evidence.",
    "source_artifact_hashes": "Hash each exact producer, pre-gate record, and raw evidence source.",
    "rows": "Keep each claim summary and historical live pair with costs, failures, and censoring.",
    "sample_size_budget": "Keep planned, attempted, completed, censored units and stopping rules.",
    "acceptance_gate_results": "Separate expected, observed, and passed for validation, value, and safety.",
    "gate_check_summary": "Name each upstream, exact field, expected value, and observed value.",
    "verifier_is_oracle": "True because one current source-fidelity evaluator defines public truth.",
    "honest_verdict": "Separate complete accounting from unavailable, null, or disqualified science.",
    "verdict_class": "Use the closed terminal enum; partial is only unfinished task-owned work.",
    "flagged_adversarial": "A current critical verification finding blocks promotion.",
    "validation_receipts": "Keep exact command, scope, exit, elapsed time, and log hash for every check.",
    "repository_health": "Keep dated unrelated failures separate from affected required validation.",
    "field_principles": "Explain fields without wrapping scalar gates or ordinary dictionaries.",
    "milestone_disposition_complete_score": "One means all twelve slots were accounted for, not science success.",
    "disposition_rows": "Keep exact contract order, source path and hash, class, gate, and disposition.",
    "required_science_complete_score": "One requires valid source capture, learning, and independent audit.",
    "publication_gate_results": "Keep G1 through G4 separate with provenance and circularity limits.",
    "retirement_decisions": "Record exact repeats and durable manifest receipts.",
    "next_research_decisions": "Tie continue, retire, or defer to one measured bottleneck.",
}

sha256 = shared.sha256
sha256_bytes = shared.sha256_bytes
read_json = shared.read_json
artifact_checksum = shared.artifact_checksum
atomic_write_json = validation_boundary.atomic_json


def progress(started: float, phase: str, event: str, detail: str = "") -> None:
    """Flush a factual phase boundary before or after long work."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7368] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def _task_number(task_id: str) -> int | None:
    """Return the numeric prefix used by ordinary experiment identities."""

    matched = re.match(r"exp(\d+)", task_id)
    return int(matched.group(1)) if matched else None


def _path_label(path: Path, root: Path) -> str:
    """Keep repository paths stable while private test sidecars stay readable."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_label(root: Path, label: str) -> Path:
    """Resolve one stored repository-relative or absolute path."""

    path = Path(label)
    return path if path.is_absolute() else root / path


def _terminal_status(status: object) -> bool:
    """Accept only states that explicitly close or block work."""

    text = str(status)
    return text in {"complete", "blocked", "disqualified"} or text.startswith(
        ("complete_", "blocked_", "disqualified_")
    )


def load_contract(root: Path) -> JsonDict:
    """Load the active YAML authority and require the exact twelve-task order."""

    path = root / ROADMAP_PATH
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("milestone") != MILESTONE:
        raise ValueError("active roadmap must name milestone 2026.09.646")
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or [row.get("id") for row in tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("active roadmap must contain the exact twelve V646 tasks")
    for task in tasks:
        task_id = str(task["id"])
        if task.get("milestone") != MILESTONE:
            raise ValueError(f"task milestone mismatch: {task_id}")
        if task_id != EXPERIMENT_ID:
            expected = EXPECTED_EVIDENCE_STATES[task_id]["path"]
            declared = str(task.get("deliverable"))
            canonical = CANONICAL_GATE_RECORDS.get(task_id, {}).get("path")
            if declared != expected and canonical != expected:
                raise ValueError(f"task deliverable mismatch: {task_id}")
    return {
        "milestone": MILESTONE,
        "path": ROADMAP_PATH.as_posix(),
        "contract_sha256": sha256(path),
        "tasks": deepcopy(tasks),
    }


def _identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept the task ID or its ordinary integer form for V646 artifacts."""

    number = _task_number(task_id)
    return payload.get("milestone") == MILESTONE and payload.get("experiment_id") in {
        task_id,
        number,
        str(number),
    }


def _canonical_gate_payload(
    root: Path, task_id: str, selected: Path, spec: Mapping[str, Any]
) -> JsonDict:
    """Authenticate one conductor pre-gate record against its producer bytes."""

    payload = read_json(selected)
    upstream_state = EXPECTED_EVIDENCE_STATES[str(spec["upstream"])]
    upstream_path = root / str(upstream_state["path"])
    valid = (
        payload.get("schema") == "blocked_gate_check_v1"
        and payload.get("status") == "blocked"
        and payload.get("blocked_at_layer") == "conductor_pre_gate"
        and payload.get("failed_upstream") == spec["upstream"]
        and payload.get("failed_field") == spec["field"]
        and payload.get("failed_expected") == spec["expected"]
        and payload.get("failed_observed") == spec["observed"]
        and payload.get("failed_evidence_sha256") == sha256(upstream_path)
    )
    if not valid:
        raise ValueError(f"canonical conductor pre-gate record mismatch: {task_id}")
    return payload


def load_evidence_slot(root: Path, task: Mapping[str, Any], manifest: object) -> JsonDict:
    """Select a declared artifact, or one allowed canonical pre-gate record."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    declared_path = root / declared
    canonical = CANONICAL_GATE_RECORDS.get(task_id)
    source_kind = "declared_artifact"
    selected_label: str | None = declared
    payload: JsonDict
    if declared_path.is_file():
        selected_path = declared_path
        payload = read_json(selected_path)
        verdict_class = str(payload.get("verdict_class", ""))
        authenticated = (
            _identity_matches(task_id, payload)
            and _terminal_status(payload.get("status"))
            and verdict_class in CLOSED_VERDICTS
        )
    elif canonical is not None and (root / str(canonical["path"])).is_file():
        selected_label = str(canonical["path"])
        selected_path = root / selected_label
        payload = _canonical_gate_payload(root, task_id, selected_path, canonical)
        verdict_class = "blocked"
        source_kind = "conductor_pre_gate"
        authenticated = True
    else:
        return {
            "task_id": task_id,
            "expected_path": declared,
            "actual_path": None,
            "source_kind": "missing",
            "sha256": None,
            "payload": {},
            "status": "missing",
            "honest_verdict": "blocked_missing_external_evidence",
            "verdict_class": "blocked",
            "flagged_adversarial": False,
            "quarantine_receipt": {"quarantined": False},
            "quarantined": False,
            "authenticated": False,
            "accepted_for_science": False,
        }

    quarantine = quarantine_helpers.quarantine_receipt(payload, task_id, selected_label, manifest)
    flagged = payload.get("flagged_adversarial") is True
    accepted = (
        authenticated
        and not quarantine["quarantined"]
        and not flagged
        and verdict_class in ELIGIBLE_VERDICTS
    )
    return {
        "task_id": task_id,
        "expected_path": declared,
        "actual_path": selected_label,
        "source_kind": source_kind,
        "sha256": sha256(selected_path),
        "payload": payload,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict", "blocked_gate_check_failed"),
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "quarantine_receipt": quarantine,
        "quarantined": quarantine["quarantined"],
        "authenticated": authenticated,
        "accepted_for_science": accepted,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Read one exact evidence slot for each producer before the capstone."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {str(task["id"]): load_evidence_slot(root, task, manifest) for task in tasks[:-1]}


def collect_preconditions(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Record exact input identity and eligibility before scientific reduction."""

    spec_present = "REQ-REPORT-7368" in (root / SPEC_PATH).read_text(encoding="utf-8")
    rows: list[JsonDict] = [
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "artifact_field": "REQ-REPORT-7368",
            "expected": True,
            "observed": spec_present,
            "passed": spec_present,
        },
        {
            "check": "twelve_task_contract",
            "upstream": contract["path"],
            "artifact_field": "tasks[].id",
            "expected": list(EXPECTED_TASK_IDS),
            "observed": [row["id"] for row in contract["tasks"]],
            "passed": [row["id"] for row in contract["tasks"]] == list(EXPECTED_TASK_IDS),
        },
    ]
    for task_id in EXPECTED_TASK_IDS[:-1]:
        actual = evidence[task_id]
        expected = EXPECTED_EVIDENCE_STATES[task_id]
        observed = {
            "path": actual["actual_path"],
            "class": actual["verdict_class"],
            "flagged_adversarial": actual["flagged_adversarial"],
            "quarantined": actual["quarantined"],
            "authenticated": actual["authenticated"],
            "sha256": actual["sha256"],
        }
        expected_value = {
            "path": expected["path"],
            "class": expected["class"],
            "flagged_adversarial": False,
            "authenticated": True,
            "sha256_prefix": "sha256:",
        }
        passed = (
            observed["path"] == expected_value["path"]
            and observed["class"] == expected_value["class"]
            and observed["flagged_adversarial"] is False
            and observed["authenticated"] is True
            and isinstance(observed["sha256"], str)
            and observed["sha256"].startswith("sha256:")
        )
        rows.append(
            {
                "check": "producer_path_field_class_flag",
                "upstream": task_id,
                "artifact_field": "path/identity/verdict_class/flag/hash",
                "expected": expected_value,
                "observed": observed,
                "passed": passed,
                "accepted_for_science": actual["accepted_for_science"],
            }
        )
    return rows


def _claim_row(
    claim: str,
    producer: str,
    *,
    verdict_class: str,
    evidence_period: str,
    cohort: str,
    truth_authority: str,
    completion_score: int,
    value_score: int,
    metrics: Mapping[str, Any],
    costs: Mapping[str, Any],
    failures: Sequence[str],
    censored: bool = False,
) -> JsonDict:
    """Build one uniform claim row without treating diagnostics as promotion."""

    return {
        "unit_id": f"claim:{claim}",
        "arm": claim,
        "claim": claim,
        "producer": producer,
        "verdict_class": verdict_class,
        "evidence_period": evidence_period,
        "cohort": cohort,
        "truth_authority": truth_authority,
        "completion_score": completion_score,
        "value_score": value_score,
        "metric": None,
        "metrics": deepcopy(dict(metrics)),
        "costs": deepcopy(dict(costs)),
        "failures": list(failures),
        "censored": censored,
        "promotes_scientific_value": False,
    }


def reduce_claim_rows(root: Path, evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Independently summarize six claim classes without crossing authorities."""

    del root
    source = evidence["exp7361-fresh-plan-capture"]["payload"]
    source_rows = source.get("source_fidelity_rows")
    if not isinstance(source_rows, list):
        raise ValueError("source_fidelity_rows must be a list")
    all_fidelity = sum(
        all(
            row.get(field) is True
            for field in (
                "request_identity_fidelity",
                "entity_fidelity",
                "quantity_fidelity",
                "ordering_fidelity",
            )
        )
        for row in source_rows
    )
    source_claim = _claim_row(
        "fresh_source_fidelity",
        "exp7361-fresh-plan-capture",
        verdict_class=str(evidence["exp7361-fresh-plan-capture"]["verdict_class"]),
        evidence_period="current",
        cohort="live_model_capture",
        truth_authority="oracle_defined",
        completion_score=int(source.get("plan_capture_complete_score") == 1),
        value_score=int(source.get("value_ready_score") == 1),
        metrics={
            "row_count": len(source_rows),
            "parser_valid_count": sum(row.get("parser_valid") is True for row in source_rows),
            "public_semantic_correct_count": sum(
                row.get("public_semantic_correct") is True for row in source_rows
            ),
            "executor_valid_count": sum(row.get("executor_valid") is True for row in source_rows),
            "all_fidelity_dimension_count": all_fidelity,
        },
        costs={
            "generation_calls": source.get("invocation_counts", {}).get(
                "generation_calls_completed"
            )
        },
        failures=[]
        if source.get("value_ready_score") == 1
        else ["independent_value_not_established"],
    )

    learning_source = evidence["exp7362-prospective-learning"]
    learning = learning_source["payload"]
    raw_rows = learning.get("rows")
    streams = learning.get("per_stream_results")
    witnesses = learning.get("structural_learning_witnesses")
    if (
        not isinstance(raw_rows, list)
        or not isinstance(streams, list)
        or not isinstance(witnesses, list)
    ):
        raise ValueError("learning rows, streams, and witnesses must be lists")
    reduced = learning.get("independent_reduction") or {}
    learning_claim = _claim_row(
        "structural_learning",
        "exp7362-prospective-learning",
        verdict_class=str(learning_source["verdict_class"]),
        evidence_period="current",
        cohort="synthetic_and_live",
        truth_authority="independent_raw_reduction",
        completion_score=int(learning.get("learning_capture_complete_score") == 1),
        value_score=int(learning.get("learning_value_score") == 1),
        metrics={
            "raw_row_count": len(raw_rows),
            "synthetic_stream_count": sum(row.get("cohort") == "synthetic" for row in streams),
            "live_stream_count": sum(row.get("cohort") == "live" for row in streams),
            "structural_witness_count": len(witnesses),
            "maximum_query_ratio_upper_95": (reduced.get("gate_results") or {})
            .get("query_ratio_vs_reset", {})
            .get("observed", {})
            .get("maximum_upper_95"),
            "maximum_complete_cost_upper_95": (reduced.get("gate_results") or {})
            .get("complete_service_cost_ratio", {})
            .get("observed", {})
            .get("maximum_upper_95"),
        },
        costs={"historical_model_amortization_included": True},
        failures=[
            name
            for name, gate in (reduced.get("gate_results") or {}).items()
            if gate.get("passed") is not True
        ],
    )

    audit_source = evidence["exp7363-learning-audit"]
    audit_payload = audit_source["payload"]
    learning_audit_claim = _claim_row(
        "learning_audit",
        "exp7363-learning-audit",
        verdict_class="blocked",
        evidence_period="current",
        cohort="not_run",
        truth_authority="conductor_pre_gate",
        completion_score=0,
        value_score=0,
        metrics={
            "failed_field": audit_payload.get("failed_field"),
            "expected": audit_payload.get("failed_expected"),
            "observed": audit_payload.get("failed_observed"),
        },
        costs={"current_model_calls": 0},
        failures=["independent_learning_audit_unavailable"],
        censored=True,
    )

    acquisition_source = evidence["exp7364-acquisition-adjudication"]
    acquisition = acquisition_source["payload"]
    paired = acquisition.get("paired_cost_rows")
    if not isinstance(paired, list):
        raise ValueError("paired_cost_rows must be a list")
    acquisition_reduction = acquisition.get("independent_reduction") or {}
    acquisition_claim = _claim_row(
        "acquisition_adjudication",
        "exp7364-acquisition-adjudication",
        verdict_class=str(acquisition_source["verdict_class"]),
        evidence_period="current_reduction_of_historical_rows",
        cohort="synthetic_archive",
        truth_authority="independent_raw_reduction",
        completion_score=int(acquisition.get("acquisition_adjudication_complete_score") == 1),
        value_score=int(acquisition.get("acquisition_value_score") == 1),
        metrics={
            "paired_context_count": len(paired),
            "complete_cost_upper_95": (
                acquisition_reduction.get("complete_cost_ratio_ci95") or {}
            ).get("upper"),
            "paid_query_ratio_upper_95": (
                acquisition_reduction.get("paid_query_ratio_ci95") or {}
            ).get("upper"),
            "returned_infeasible_count": acquisition_reduction.get("returned_infeasible_count"),
        },
        costs={"new_generation_calls": 0},
        failures=[]
        if acquisition.get("acquisition_value_score") == 1
        else ["complete_cost_gate_failed"],
    )

    support_source = evidence["exp7365-supervisor-support"]
    support = support_source["payload"]
    support_rows = support.get("support_rows")
    if not isinstance(support_rows, list):
        raise ValueError("support_rows must be a list")
    live = evidence["exp7366-supervisor-live"]["payload"]
    supervisor_claim = _claim_row(
        "supervisor_support_and_live",
        "exp7365-supervisor-support+exp7366-supervisor-live",
        verdict_class="null",
        evidence_period="current",
        cohort="archived_support_with_optional_live_not_run",
        truth_authority="observed_outcomes_only",
        completion_score=1,
        value_score=0,
        metrics={
            "support_row_count": len(support_rows),
            "supported_decision_count": sum(
                row.get("outcome_supported") is True for row in support_rows
            ),
            "unsupported_counterfactual_count": support.get("unsupported_counterfactual_count"),
            "live_trial_executed": (support.get("generalization_activity") or {}).get(
                "live_trial_executed"
            ),
            "live_skip_accounted": live.get("failed_field") == "supervisor_trial_ready_score"
            and live.get("failed_observed") == 0,
        },
        costs={"current_live_generation_calls": 0},
        failures=["insufficient_supported_outcomes", "expected_live_pre_gate_skip"],
        censored=False,
    )

    board_source = evidence["exp7367-board-disposition"]
    board = board_source["payload"]
    board_rows = board.get("board_rows")
    if not isinstance(board_rows, list):
        raise ValueError("board_rows must be a list")
    board_claim = _claim_row(
        "board_dispositions",
        "exp7367-board-disposition",
        verdict_class="blocked",
        evidence_period="historical_and_current_receipt_search",
        cohort="hardware_accounting",
        truth_authority="authenticated_receipts",
        completion_score=int(board.get("board_disposition_complete_score") == 1),
        value_score=int(board.get("hardware_value_score") == 1),
        metrics={
            "board_row_count": len(board_rows),
            "native_tenfold_speed_gate": board.get("native_tenfold_speed_gate"),
            "changed_state_receipt": board.get("changed_state_receipt"),
            "hardware_operation_count": sum(
                value
                for value in (board.get("hardware_operations") or {}).values()
                if isinstance(value, int) and not isinstance(value, bool)
            ),
        },
        costs={"current_hardware_operations": 0},
        failures=["gatemate_changed_physical_state_receipt_missing"],
    )
    return [
        source_claim,
        learning_claim,
        learning_audit_claim,
        acquisition_claim,
        supervisor_claim,
        board_claim,
    ]


def reduce_historical_live_arc_pairs(root: Path) -> list[JsonDict]:
    """Recompute two paired ARC differences from raw historical episode rows."""

    path = root / HISTORICAL_ARC_ROWS
    try:
        payload = read_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("historical ARC raw evidence must contain four rows") from error
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("historical ARC raw evidence must contain four rows")
    by_game: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        game = str(row.get("game"))
        arm = str(row.get("arm"))
        by_game.setdefault(game, {})[arm] = row
    output: list[JsonDict] = []
    for game in ("r11l", "re86"):
        arms = by_game.get(game, {})
        if set(arms) != {"result_resume", "result_withheld"}:
            raise ValueError("historical ARC raw evidence must contain four rows")
        resume = arms["result_resume"]
        withheld = arms["result_withheld"]
        output.append(
            {
                "unit_id": f"historical_arc_pair:{game}",
                "arm": "result_resume_minus_result_withheld",
                "claim": "historical_live_arc_pair",
                "game": game,
                "verdict_class": "null",
                "evidence_period": "historical",
                "cohort": "live",
                "truth_authority": "raw_environment_episode",
                "metric": None,
                "metrics": {
                    "resume_levels": resume.get("levels"),
                    "withheld_levels": withheld.get("levels"),
                    "level_delta_resume_minus_withheld": int(resume.get("levels", 0))
                    - int(withheld.get("levels", 0)),
                    "action_delta_resume_minus_withheld": int(resume.get("action_count", 0))
                    - int(withheld.get("action_count", 0)),
                    "usable_answer_delta_resume_minus_withheld": int(
                        resume.get("usable_answers", 0)
                    )
                    - int(withheld.get("usable_answers", 0)),
                },
                "costs": {
                    "resume": deepcopy(resume.get("compute_cost") or {}),
                    "withheld": deepcopy(withheld.get("compute_cost") or {}),
                },
                "failures": ["no_successful_bound_result", "no_later_policy_action"],
                "censored": bool(resume.get("censored") or withheld.get("censored")),
                "promotes_scientific_value": False,
            }
        )
    return output


def _required_science_failures(
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Name failures for source capture, completed learning, and independent audit."""

    failures: list[JsonDict] = []
    learning = evidence["exp7362-prospective-learning"]["payload"]
    affected = (learning.get("acceptance_gate_results") or {}).get("affected_validation") or {}
    if affected.get("passed") is not True:
        failures.append(
            {
                "upstream": "exp7362-prospective-learning",
                "failed_check": "affected_validation",
                "artifact_field": "acceptance_gate_results.affected_validation",
                "expected": deepcopy(affected.get("expected")),
                "observed": deepcopy(affected.get("observed")),
                "passed": False,
                "terminal_blocking": True,
            }
        )
    if learning.get("learning_capture_complete_score") != 1:
        failures.append(
            {
                "upstream": "exp7362-prospective-learning",
                "failed_check": "required_learning_completion",
                "artifact_field": "learning_capture_complete_score",
                "expected": 1,
                "observed": learning.get("learning_capture_complete_score"),
                "passed": False,
                "terminal_blocking": True,
            }
        )
    audit = evidence["exp7363-learning-audit"]
    if audit.get("verdict_class") != "null" and audit.get("verdict_class") != "positive":
        payload = audit["payload"]
        failures.append(
            {
                "upstream": "exp7363-learning-audit",
                "failed_check": "independent_learning_audit",
                "artifact_field": payload.get("failed_field", "verdict_class"),
                "expected": payload.get("failed_expected", "eligible terminal audit"),
                "observed": payload.get("failed_observed", audit.get("verdict_class")),
                "passed": False,
                "terminal_blocking": True,
            }
        )
    return failures


def terminal_state(
    evidence: Mapping[str, Mapping[str, Any]], *, required_validation_passed: bool
) -> JsonDict:
    """Choose blocked, disqualified, or null without consulting optional branches."""

    for task_id in REQUIRED_SCIENCE_TASKS:
        row = evidence[task_id]
        if row.get("actual_path") is None or row.get("authenticated") is not True:
            failure = {
                "upstream": task_id,
                "failed_check": "required_external_evidence",
                "artifact_field": "declared_deliverable_or_canonical_pre_gate_record",
                "expected": row.get("expected_path"),
                "observed": row.get("actual_path"),
                "passed": False,
                "terminal_blocking": True,
            }
            return {
                "status": "blocked_external_required_science",
                "verdict_class": "blocked",
                "honest_verdict": (
                    f"blocked_{task_id.replace('-', '_')}: required V646 science evidence is "
                    "externally unavailable; accounting cannot continue as a success"
                ),
                "required_science_complete_score": 0,
                "gate_check_summary": {
                    "passed": False,
                    "failed_count": 1,
                    "first_failure": failure,
                    "failures": [failure],
                },
            }
    failures = _required_science_failures(evidence)
    if not required_validation_passed:
        failures.insert(
            0,
            {
                "upstream": EXPERIMENT_ID,
                "failed_check": "current_required_validation",
                "artifact_field": "required_checks_passed",
                "expected": True,
                "observed": False,
                "passed": False,
                "terminal_blocking": True,
            },
        )
    source = evidence["exp7361-fresh-plan-capture"]
    learning = evidence["exp7362-prospective-learning"]
    audit = evidence["exp7363-learning-audit"]
    required_complete = int(
        source.get("accepted_for_science") is True
        and source["payload"].get("plan_capture_complete_score") == 1
        and learning.get("accepted_for_science") is True
        and learning["payload"].get("learning_capture_complete_score") == 1
        and audit.get("accepted_for_science") is True
    )
    if not required_validation_passed or learning.get("verdict_class") == "disqualified":
        verdict_class = "disqualified"
        status = "complete_disqualified_required_science_or_validation_failure"
        verdict = (
            "complete_disqualified_required_science_or_validation_failure: all twelve V646 "
            "dispositions are accounted for, but Exp7362 failed required validation and the "
            "independent learning audit was pre-gated"
        )
    elif required_complete == 0:
        verdict_class = "blocked"
        status = "blocked_required_science_unavailable"
        verdict = "blocked_required_science_unavailable: required learning or audit evidence is unavailable"
    else:
        verdict_class = "null"
        status = "complete_null_no_useful_effect"
        verdict = (
            "complete_null_no_useful_effect: required science completed without a useful effect"
        )
    return {
        "status": status,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
        "required_science_complete_score": required_complete,
        "gate_check_summary": {
            "passed": not failures,
            "failed_count": len(failures),
            "first_failure": failures[0] if failures else None,
            "failures": failures,
        },
    }


def build_disposition_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Create eleven evidence rows before the non-recursive self disposition."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        source = evidence[task_id]
        expected = EXPECTED_EVIDENCE_STATES[task_id]
        payload = source["payload"]
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task["title"],
                "phase": task["phase"],
                "requirement": task.get("track"),
                "expected_artifact_path": task["deliverable"],
                "actual_evidence_path": source["actual_path"],
                "evidence_source": source["source_kind"],
                "artifact_sha256": source["sha256"],
                "expected_class": expected["class"],
                "actual_class": source["verdict_class"],
                "expected_flagged_adversarial": False,
                "actual_flagged_adversarial": source["flagged_adversarial"],
                "quarantined": source["quarantined"],
                "authenticated": source["authenticated"],
                "status": source["status"],
                "honest_verdict": source["honest_verdict"],
                "failed_gate": deepcopy(payload.get("blocked_diagnostic_contract")),
                "disposition": source["verdict_class"],
                "accepted_for_science": source["accepted_for_science"],
            }
        )
    rows.append(
        {
            "order": 12,
            "task_id": EXPERIMENT_ID,
            "title": tasks[-1]["title"],
            "phase": tasks[-1]["phase"],
            "requirement": "REQ-REPORT-7368",
            "expected_artifact_path": tasks[-1]["deliverable"],
            "actual_evidence_path": tasks[-1]["deliverable"],
            "evidence_source": "capstone_self",
            "artifact_sha256": None,
            "expected_class": terminal["verdict_class"],
            "actual_class": terminal["verdict_class"],
            "expected_flagged_adversarial": False,
            "actual_flagged_adversarial": False,
            "quarantined": False,
            "authenticated": True,
            "status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "failed_gate": deepcopy(terminal["gate_check_summary"]["first_failure"]),
            "disposition": terminal["verdict_class"],
            "accepted_for_science": False,
        }
    )
    return rows


def _predecessor(root: Path, experiment_id: str) -> JsonDict:
    """Load one exact predecessor artifact for a declared prior failure."""

    number = _task_number(experiment_id)
    matches = sorted((root / "results").glob(f"experiment_{number}_*.json")) if number else []
    if len(matches) != 1:
        raise ValueError(f"one predecessor artifact required for {experiment_id}")
    payload = read_json(matches[0])
    return {
        "path": _path_label(matches[0], root),
        "sha256": sha256(matches[0]),
        "honest_verdict": str(payload.get("honest_verdict", "")),
    }


def _retirement_manifest_receipt(root: Path) -> JsonDict:
    """Read the durable exclusion entry used for the exact repeated audit block."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    matches = [
        row
        for row in manifest.get("retired_extras", [])
        if isinstance(row, Mapping) and row.get("id") == EXCLUSION_RECEIPT_ID
    ]
    if len(matches) != 1:
        raise ValueError("one V646 learning-audit retirement receipt required")
    return {
        "id": EXCLUSION_RECEIPT_ID,
        "path": EXCLUSION_PATH.as_posix(),
        "sha256": sha256(root / EXCLUSION_PATH),
        "entry": deepcopy(dict(matches[0])),
    }


def retirement_decisions(
    root: Path,
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Compare prior verdict bytes and preserve four existing branch retirements."""

    decisions: list[JsonDict] = []
    manifest_receipt = _retirement_manifest_receipt(root)
    for task in tasks:
        task_id = str(task["id"])
        current = (
            str(terminal["honest_verdict"])
            if task_id == EXPERIMENT_ID
            else str(evidence[task_id]["honest_verdict"])
        )
        for prior in task.get("prior_failures") or []:
            predecessor = _predecessor(root, str(prior["experiment_id"]))
            declared = str(prior["verdict"])
            recorded = predecessor["honest_verdict"]
            declaration_matches = declared == recorded
            exact_repeat = declaration_matches and current == recorded
            if exact_repeat and prior.get("retire_if_same_verdict") is True:
                decision = "retire_exact_repeat"
            elif not declaration_matches:
                decision = "preserve_declaration_mismatch"
            else:
                decision = "changed_verdict_no_retirement"
            decisions.append(
                {
                    "kind": "prior_failure",
                    "task_id": task_id,
                    "prior_experiment_id": prior["experiment_id"],
                    "declared_prior_honest_verdict": declared,
                    "recorded_predecessor_honest_verdict": recorded,
                    "recorded_predecessor_path": predecessor["path"],
                    "recorded_predecessor_sha256": predecessor["sha256"],
                    "current_honest_verdict": current,
                    "declared_prior_matches_recorded": declaration_matches,
                    "exact_repeat": exact_repeat,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "decision": decision,
                    "manifest_receipt": manifest_receipt
                    if decision == "retire_exact_repeat"
                    else None,
                }
            )
    board = evidence["exp7367-board-disposition"]["payload"]
    acquisition = evidence["exp7364-acquisition-adjudication"]["payload"]
    pairs = reduce_historical_live_arc_pairs(root)
    decisions.extend(
        [
            {
                "kind": "branch",
                "branch": "native_tenfold_null",
                "decision": "preserve_retirement",
                "exact_evidence": board.get("native_tenfold_speed_gate"),
                "reopening_condition": "A changed complete boundary has a paired lower 95 percent speedup bound of at least ten with parity.",
            },
            {
                "kind": "branch",
                "branch": "external_text_scoring",
                "decision": "preserve_retirement",
                "exact_evidence": "phase_d_external_text_scorer_retired_exp5163_v474",
                "reopening_condition": "An operator override names a genuinely different mechanism outside external generated-text scoring.",
            },
            {
                "kind": "branch",
                "branch": "unchanged_acquisition",
                "decision": "preserve_retirement",
                "exact_evidence": deepcopy(acquisition.get("unchanged_method_disposition")),
                "reopening_condition": (acquisition.get("unchanged_method_disposition") or {}).get(
                    "future_evidence_required"
                ),
            },
            {
                "kind": "branch",
                "branch": "result_resume",
                "decision": "preserve_retirement",
                "exact_evidence": pairs,
                "reopening_condition": "A changed mechanism yields a successful bound result and a later action within the fixed episode budget.",
            },
        ]
    )
    return decisions


def publication_gate_results(
    evidence: Mapping[str, Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    *,
    current_validation_passed: bool,
) -> dict[str, JsonDict]:
    """Apply V646 G1 through G4 without treating accounting as publication."""

    by_claim = {str(row["claim"]): row for row in claims}
    source_ready = evidence["exp7361-fresh-plan-capture"]["accepted_for_science"] is True
    learning_ready = evidence["exp7362-prospective-learning"]["accepted_for_science"] is True
    audit_ready = evidence["exp7363-learning-audit"]["accepted_for_science"] is True
    cost_honest = (
        by_claim["structural_learning"]["metrics"]["maximum_complete_cost_upper_95"] is not None
        and by_claim["acquisition_adjudication"]["metrics"]["complete_cost_upper_95"] is not None
    )
    provenance_honest = (
        by_claim["fresh_source_fidelity"]["truth_authority"] == "oracle_defined"
        and by_claim["supervisor_support_and_live"]["metrics"]["live_trial_executed"] is False
        and by_claim["board_dispositions"]["metrics"]["hardware_operation_count"] == 0
    )
    return {
        "G1": {
            "name": "reproducible_authenticated_raw_rows",
            "expected": {"source_capture": True, "learning_capture": True},
            "observed": {"source_capture": source_ready, "learning_capture": learning_ready},
            "passed": source_ready and learning_ready,
        },
        "G2": {
            "name": "independent_adversarial_and_raw_reduction",
            "expected": {"learning_audit": True, "current_validation": True},
            "observed": {
                "learning_audit": audit_ready,
                "current_validation": current_validation_passed,
            },
            "passed": audit_ready and current_validation_passed,
        },
        "G3": {
            "name": "complete_cost_and_baseline_honesty",
            "expected": True,
            "observed": cost_honest,
            "passed": cost_honest,
        },
        "G4": {
            "name": "access_venue_circularity_and_solve_provenance",
            "expected": True,
            "observed": provenance_honest,
            "passed": provenance_honest,
        },
    }


def next_research_decisions(claims: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Tie each bounded next action to one measured bottleneck."""

    by_claim = {str(row["claim"]): row for row in claims}
    return [
        {
            "branch": "source_fidelity",
            "decision": "defer_more_generation",
            "measured_bottleneck": {
                "executor_valid_count": by_claim["fresh_source_fidelity"]["metrics"][
                    "executor_valid_count"
                ],
                "row_count": by_claim["fresh_source_fidelity"]["metrics"]["row_count"],
                "oracle_defined": True,
            },
            "next_condition": "Add independent non-oracle truth before using source fidelity as value evidence.",
        },
        {
            "branch": "structural_learning",
            "decision": "retire_unchanged_chain",
            "measured_bottleneck": {
                "structural_witness_count": 0,
                "maximum_query_ratio_upper_95": by_claim["structural_learning"]["metrics"][
                    "maximum_query_ratio_upper_95"
                ],
                "maximum_complete_cost_upper_95": by_claim["structural_learning"]["metrics"][
                    "maximum_complete_cost_upper_95"
                ],
            },
            "next_condition": "A different representation yields an erasure witness and passes complete-cost and query gates.",
        },
        {
            "branch": "learning_audit",
            "decision": "defer_until_eligible_capture",
            "measured_bottleneck": {"audit_completion_score": 0},
            "next_condition": "An eligible prospective capture passes all required checks before a new audit.",
        },
        {
            "branch": "acquisition",
            "decision": "retire_unchanged_method",
            "measured_bottleneck": {
                "complete_cost_upper_95": by_claim["acquisition_adjudication"]["metrics"][
                    "complete_cost_upper_95"
                ],
                "required_upper": 0.9,
            },
            "next_condition": "A changed query rule or representation passes the paired complete-cost gate.",
        },
        {
            "branch": "supervisor_live",
            "decision": "defer_live_generation",
            "measured_bottleneck": {
                "supported_decision_count": by_claim["supervisor_support_and_live"]["metrics"][
                    "supported_decision_count"
                ],
                "live_trial_executed": False,
            },
            "next_condition": "Observed outcomes support ten decisions per arm on at least three development games.",
        },
        {
            "branch": "hardware",
            "decision": "preserve_external_block",
            "measured_bottleneck": {"changed_state_receipt": None},
            "next_condition": "An operator records a dated GateMate physical-state change after Exp6559.",
        },
    ]


def write_historical_inference_sidecar(
    path: Path, evidence: Mapping[str, Mapping[str, Any]], root: Path
) -> JsonDict:
    """Label historical LLM receipts so they cannot count as current inference."""

    capture = evidence["exp7361-fresh-plan-capture"]["payload"]
    arc = read_json(root / HISTORICAL_ARC_ARTIFACT)
    sidecar = {
        "schema": "carnot.experiment_7368.historical_inference_receipts.v1",
        "label": "historical_only_not_current_exp7368_inference",
        "current": {
            "MODEL_SPECS": [],
            "model_invoked": False,
            "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        },
        "historical": [
            {
                "experiment_id": "exp7361-fresh-plan-capture",
                "artifact_path": evidence["exp7361-fresh-plan-capture"]["actual_path"],
                "artifact_sha256": evidence["exp7361-fresh-plan-capture"]["sha256"],
                "MODEL_SPECS": deepcopy(capture.get("MODEL_SPECS")),
                "invocation_counts": deepcopy(capture.get("invocation_counts")),
                "counted_as_current": False,
            },
            {
                "experiment_id": "exp7354-arc-transfer",
                "artifact_path": HISTORICAL_ARC_ARTIFACT.as_posix(),
                "artifact_sha256": sha256(root / HISTORICAL_ARC_ARTIFACT),
                "raw_episode_path": HISTORICAL_ARC_ROWS.as_posix(),
                "raw_episode_sha256": sha256(root / HISTORICAL_ARC_ROWS),
                "MODEL_SPECS": deepcopy(arc.get("MODEL_SPECS")),
                "invocation_counts": deepcopy(arc.get("invocation_counts")),
                "counted_as_current": False,
            },
        ],
    }
    atomic_write_json(path, sidecar)
    return sidecar


def write_independent_reduction_sidecar(
    path: Path, root: Path, evidence: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    """Persist a compact raw reduction that terminal replay can hash-check."""

    sidecar = {
        "schema": "carnot.experiment_7368.independent_raw_reduction.v1",
        "claim_rows": reduce_claim_rows(root, evidence),
        "historical_live_arc_pairs": reduce_historical_live_arc_pairs(root),
        "producer_hashes": {task_id: row["sha256"] for task_id, row in evidence.items()},
    }
    atomic_write_json(path, sidecar)
    return sidecar


def _source_hashes(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    historical_sidecar: Path,
    independent_sidecar: Path | None,
) -> dict[str, str]:
    """Hash exact code, settings, evaluator, producer, and raw-evidence bytes."""

    paths = [
        root / str(contract["path"]),
        root / SPEC_PATH,
        root / EXCLUSION_PATH,
        root / "ops/e2e-test-plan.md",
        root / MODULE_PATH,
        root / TEST_PATH,
        root / WRAPPER_PATH,
        root / "python/carnot/reporting/experiment_7303_validation_scope.py",
        root / "python/carnot/experiment_7358_v646_validation_contract.py",
        root / "scripts/adversarial_verify.py",
        root / "scripts/verdict_row_consistency_lint.py",
        root / HISTORICAL_ARC_ARTIFACT,
        root / HISTORICAL_ARC_ROWS,
        historical_sidecar,
    ]
    if independent_sidecar is not None:
        paths.append(independent_sidecar)
    paths.extend(
        _resolve_label(root, str(row["actual_path"]))
        for row in evidence.values()
        if row.get("actual_path") is not None
    )
    unique = {path.resolve(): path for path in paths}
    return {
        _path_label(path, root): sha256(path)
        for path in sorted(unique.values(), key=lambda item: str(item.resolve()))
    }


def _acceptance_gates(
    *,
    terminal: Mapping[str, Any],
    validation_passed: bool,
    publication: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Keep accounting, required validation, value, and promotion distinct."""

    return {
        "milestone_accounting": {
            "category": "completion",
            "expected": 12,
            "observed": 12,
            "passed": True,
        },
        "required_validation": {
            "category": "required_validation",
            "expected": True,
            "observed": validation_passed,
            "passed": validation_passed,
        },
        "required_science": {
            "category": "completion",
            "expected": 1,
            "observed": terminal["required_science_complete_score"],
            "passed": terminal["required_science_complete_score"] == 1,
        },
        "scientific_value": {
            "category": "efficacy",
            "expected": 1,
            "observed": 0,
            "passed": False,
        },
        "publication": {
            "category": "publication",
            "expected": ["G1", "G2", "G3", "G4"],
            "observed": [name for name, row in publication.items() if row["passed"]],
            "passed": all(row["passed"] for row in publication.values()),
        },
        "promotion": {
            "category": "promotion",
            "expected": 1,
            "observed": 0,
            "passed": False,
        },
    }


def _principles_for(keys: Sequence[str]) -> dict[str, str]:
    """Give each artifact field a separate plain-language principle."""

    return {
        key: FIELD_PRINCIPLES.get(
            key, "Keep this field explicit, hash-bound, and independently replayable."
        )
        for key in keys
    }


def zero_test_phase_spans() -> list[JsonDict]:
    """Return the five required phase names for deterministic unit fixtures."""

    return [
        {"phase": name, "start_s": 0.0, "end_s": 0.0, "duration_s": 0.0, "units": 0}
        for name in ("load", "generation", "evaluation", "validation", "write")
    ]


def build_artifact(
    *,
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    historical_sidecar: Path,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    independent_sidecar: Path | None = None,
) -> JsonDict:
    """Build one terminal capstone from already authenticated evidence."""

    validation_passed = bool(
        validation.get("affected_validation_passed") is True
        and validation.get("full_suite_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    terminal = terminal_state(evidence, required_validation_passed=validation_passed)
    claims = reduce_claim_rows(root, evidence)
    arc_pairs = reduce_historical_live_arc_pairs(root)
    rows = [*claims, *arc_pairs]
    publication = publication_gate_results(
        evidence, claims, current_validation_passed=validation_passed
    )
    dispositions = build_disposition_rows(contract["tasks"], evidence, terminal)
    sidecar_label = _path_label(historical_sidecar, root)
    sidecar_receipt = {
        "label": "historical_only_not_current_exp7368_inference",
        "path": sidecar_label,
        "sha256": sha256(historical_sidecar),
    }
    independent_receipt = (
        {
            "path": _path_label(independent_sidecar, root),
            "sha256": sha256(independent_sidecar),
        }
        if independent_sidecar is not None
        else None
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "status": terminal["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": collect_preconditions(root, contract, evidence),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_INVOCATION_COUNTS),
            "historical": {"sidecar_count": 1, "counted_as_current": False},
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_computation": {
            "node": platform.node(),
            "processor": platform.processor() or "host_cpu",
            "python": platform.python_version(),
            "current_gpu_operations": 0,
            "current_model_operations": 0,
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hashes(
            root, contract, evidence, historical_sidecar, independent_sidecar
        ),
        "historical_inference_sidecars": [sidecar_receipt],
        "independent_reduction_sidecar": independent_receipt,
        "rows": rows,
        "sample_size_budget": {
            "planned_dispositions": 12,
            "attempted_dispositions": 12,
            "completed_dispositions": 12,
            "censored_dispositions": 0,
            "current_claim_rows": len(claims),
            "historical_live_pair_rows": len(arc_pairs),
            "stopping_rule": "Stop after all twelve contract slots and fixed raw summaries reconcile once.",
        },
        "acceptance_gate_results": _acceptance_gates(
            terminal=terminal,
            validation_passed=validation_passed,
            publication=publication,
        ),
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": True,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": validation.get("flagged_adversarial") is True,
        "validation_receipts": deepcopy(validation.get("validation_receipts") or []),
        "required_checks_passed": validation_passed,
        "repository_health": deepcopy(validation.get("repository_health") or {}),
        "field_principles": {},
        "milestone_disposition_complete_score": int(len(dispositions) == 12),
        "disposition_rows": dispositions,
        "required_science_complete_score": terminal["required_science_complete_score"],
        "publication_gate_results": publication,
        "publication_ready_score": int(all(row["passed"] for row in publication.values())),
        "retirement_decisions": retirement_decisions(root, contract["tasks"], evidence, terminal),
        "next_research_decisions": next_research_decisions(claims),
        "readiness_score": 0,
        "value_score": 0,
        "promotion_score": 0,
        "estimated_operational_savings": 0,
        "operational_efficiency": {
            "phase_receipts_read": True,
            "model_count_receipts_read": True,
            "runner_receipts_read": bool(validation.get("validation_receipts")),
            "gpu_receipt": "zero current GPU operations; historical GPU/model work is sidecar-only",
            "measured_counterfactual": None,
            "estimated_savings": 0,
            "claim": "No operational efficiency or savings claim is made.",
        },
        "publication_performed": False,
        "release_performed": False,
        "push_performed": False,
        "automatic_policy_promotion_performed": False,
        "automatic_constraint_promotion_performed": False,
        "hardware_operations_performed": False,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "conductor_modified": False,
    }
    artifact["field_principles"] = _principles_for(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _receipt_names_complete(receipts: object) -> bool:
    """Require one well-shaped receipt for each declared current check."""

    if not isinstance(receipts, list):
        return False
    for name in REQUIRED_VALIDATION_NAMES:
        matches = [row for row in receipts if isinstance(row, Mapping) and row.get("name") == name]
        if len(matches) != 1:
            return False
        if not {"command", "scope", "exit_code", "duration_s", "log_sha256"}.issubset(matches[0]):
            return False
    return True


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    replay: bool = False,
    require_validation_receipts: bool = True,
) -> list[str]:
    """Cold-check identity, raw summaries, gates, retirements, and hashes."""

    if not isinstance(value, Mapping):
        return ["artifact_not_mapping"]
    artifact = dict(value)
    errors: list[str] = []

    def add(condition: bool, error: str) -> None:
        if condition and error not in errors:
            errors.append(error)

    add(
        (
            artifact.get("schema"),
            artifact.get("experiment_id"),
            artifact.get("milestone"),
            artifact.get("run_date"),
        )
        != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE),
        "identity",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or (artifact.get("invocation_counts") or {}).get("current") != ZERO_INVOCATION_COUNTS,
        "model_boundary",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host",
        "substrate",
    )
    dispositions = artifact.get("disposition_rows")
    add(
        not isinstance(dispositions, list)
        or len(dispositions) != 12
        or [row.get("task_id") for row in dispositions] != list(EXPECTED_TASK_IDS),
        "disposition_rows",
    )
    add(
        artifact.get("milestone_disposition_complete_score") != 1
        or artifact.get("required_science_complete_score") != 0
        or artifact.get("publication_ready_score") != 0
        or any(
            artifact.get(name) != 0
            for name in ("readiness_score", "value_score", "promotion_score")
        ),
        "scores",
    )
    if require_validation_receipts:
        add(not _receipt_names_complete(artifact.get("validation_receipts")), "validation_receipts")
    add(
        artifact.get("field_principles") != _principles_for(tuple(artifact)),
        "field_principles",
    )
    add(
        any(
            artifact.get(name) is not False
            for name in (
                "publication_performed",
                "release_performed",
                "push_performed",
                "automatic_policy_promotion_performed",
                "automatic_constraint_promotion_performed",
                "hardware_operations_performed",
                "production_defaults_changed",
                "active_research_roadmap_changed",
                "conductor_modified",
            )
        ),
        "unauthorized_action",
    )
    add(
        artifact.get("reproducibility_checksum") != artifact_checksum(artifact),
        "reproducibility_checksum",
    )
    if replay:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        validation_passed = artifact.get("required_checks_passed") is True
        terminal = terminal_state(evidence, required_validation_passed=validation_passed)
        claims = reduce_claim_rows(root, evidence)
        expected_rows = [*claims, *reduce_historical_live_arc_pairs(root)]
        add(artifact.get("rows") != expected_rows, "rows")
        add(
            artifact.get("disposition_rows")
            != build_disposition_rows(contract["tasks"], evidence, terminal),
            "disposition_rows",
        )
        add(
            artifact.get("preconditions_checked")
            != collect_preconditions(root, contract, evidence),
            "preconditions_checked",
        )
        add(
            artifact.get("publication_gate_results")
            != publication_gate_results(
                evidence, claims, current_validation_passed=validation_passed
            ),
            "publication_gate_results",
        )
        add(
            artifact.get("retirement_decisions")
            != retirement_decisions(root, contract["tasks"], evidence, terminal),
            "retirement_decisions",
        )
        add(
            artifact.get("next_research_decisions") != next_research_decisions(claims),
            "next_research_decisions",
        )
        add(
            any(
                artifact.get(key) != terminal[key]
                for key in ("status", "verdict_class", "honest_verdict", "gate_check_summary")
            ),
            "terminal_state",
        )
        sidecars = artifact.get("historical_inference_sidecars") or []
        historical_path = (
            _resolve_label(root, str(sidecars[0]["path"]))
            if len(sidecars) == 1 and isinstance(sidecars[0], Mapping)
            else root / "missing-historical-sidecar"
        )
        independent = artifact.get("independent_reduction_sidecar")
        independent_path = (
            _resolve_label(root, str(independent["path"]))
            if isinstance(independent, Mapping)
            else None
        )
        if historical_path.is_file() and (independent_path is None or independent_path.is_file()):
            add(
                artifact.get("source_artifact_hashes")
                != _source_hashes(root, contract, evidence, historical_path, independent_path),
                "source_artifact_hashes",
            )
        else:
            add(True, "source_artifact_hashes")
    return errors


def build_validation_plan(root: Path, private_root: Path) -> list[scoped.CommandSpec]:
    """Build the exact Exp7358-bounded affected command plan."""

    return validation_boundary.build_command_plan(root, V646_MANIFEST, private_root)


def validate_validation_plan(root: Path, commands: Sequence[scoped.CommandSpec]) -> list[str]:
    """Reject command expansion outside this module, test, and thin wrapper."""

    return validation_boundary.validate_command_plan(root, V646_MANIFEST, commands)


def build_full_suite_command(root: Path, private_root: Path) -> scoped.CommandSpec:
    """Build the separately reported full Python suite required by this task."""

    basetemp = private_root / "full-suite-basetemp"
    basetemp.mkdir(parents=True, exist_ok=True)
    return scoped.CommandSpec(
        "full_python_suite",
        (
            str(root / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={basetemp}",
        ),
        "repository_python_suite_separate_from_affected_validation",
        timeout_s=900.0,
    )


def run_affected_validation(
    root: Path, commands: Sequence[scoped.CommandSpec], log_dir: Path
) -> tuple[list[JsonDict], JsonDict]:
    """Run the shipped plan while preserving command-local coverage settings."""

    planned = [
        validation_boundary.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    receipts = validation_boundary.run_categorized_commands(root, planned, log_dir=log_dir)
    return receipts, validation_boundary.reduce_affected_receipts(root, V646_MANIFEST, receipts)


def run_terminal_validation(root: Path, candidate: Path, log_dir: Path) -> list[JsonDict]:
    """Run cold replay and the two strict artifact readers against one candidate."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,sys;from pathlib import Path;"
        "from carnot.experiment_7368_v646_capstone import validate_artifact;"
        "v=json.loads(Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v,root=Path.cwd(),replay=True,require_validation_receipts=False);"
        "print(json.dumps({'errors':e}),flush=True);raise SystemExit(bool(e))"
    )
    commands = [
        scoped.CommandSpec(
            "independent_raw_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate_and_raw_sources",
        ),
        scoped.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
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
            "measured_candidate",
        ),
    ]
    return scoped.run_commands(root, commands, log_dir=log_dir)


def _span(name: str, start: float, end: float, run_start: float, units: int) -> JsonDict:
    """Store one measured interval relative to the monotonic run origin."""

    return {
        "phase": name,
        "start_s": start - run_start,
        "end_s": end - run_start,
        "duration_s": end - start,
        "units": units,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by E2E.
    """Run the bounded capstone workflow and atomically publish one terminal JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"Exp7368 requires --date {RUN_DATE}")

    run_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []
    progress(run_started, "startup", "start", f"date={args.date}")

    phase_start = time.monotonic()
    progress(run_started, "preconditions", "start", "read exact contract and eleven producers")
    contract = load_contract(REPO_ROOT)
    evidence = collect_evidence(REPO_ROOT, contract["tasks"])
    phase_end = time.monotonic()
    spans.append(_span("load", phase_start, phase_end, run_started, 12))
    progress(run_started, "preconditions", "end", "producer_slots=11")

    missing = [
        task_id
        for task_id, row in evidence.items()
        if row["actual_path"] is None or row["authenticated"] is not True
    ]
    if missing:  # pragma: no cover - fail-closed external state.
        failure = terminal_state(evidence, required_validation_passed=False)
        blocked = {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "status": failure["status"],
            "run_date": RUN_DATE,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "missing_producers": missing,
            "gate_check_summary": failure["gate_check_summary"],
            "honest_verdict": failure["honest_verdict"],
            "verdict_class": "blocked",
            "MODEL_SPECS": [],
            "model_invoked": False,
            "invocation_counts": {"current": deepcopy(ZERO_INVOCATION_COUNTS)},
            "inference_substrate": "host_cpu_precondition_checks_only",
            "inference_substrate_class": "aggregation",
            "execution_venue": "host",
            "duration_s": time.monotonic() - run_started,
            "phase_spans": spans,
            "milestone_disposition_complete_score": 0,
            "required_science_complete_score": 0,
            "readiness_score": 0,
            "value_score": 0,
            "promotion_score": 0,
        }
        atomic_write_json(REPO_ROOT / RESULT_PATH, blocked)
        progress(run_started, "preconditions", "blocked", f"missing={missing}")
        return 1

    raw_dir = REPO_ROOT / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase_start = phase_end
    spans.append(_span("generation", phase_start, phase_start, run_started, 0))
    progress(run_started, "generation", "skipped", "MODEL_SPECS=[] model_invoked=false")

    phase_start = time.monotonic()
    progress(run_started, "evaluation", "start", "reduce current and historical rows")
    historical_sidecar = raw_dir / "historical_model_receipts.json"
    independent_sidecar = raw_dir / "independent_raw_reduction.json"
    write_historical_inference_sidecar(historical_sidecar, evidence, REPO_ROOT)
    write_independent_reduction_sidecar(independent_sidecar, REPO_ROOT, evidence)
    checkpoint = REPO_ROOT / CHECKPOINT_PATH
    atomic_write_json(
        checkpoint,
        {
            "status": "reconciliation_complete",
            "completed_dispositions": 11,
            "planned_dispositions": 12,
            "started_at_utc": started_at,
        },
    )
    phase_end = time.monotonic()
    spans.append(_span("evaluation", phase_start, phase_end, run_started, 11))
    progress(run_started, "evaluation", "end", "checkpointed_producer_dispositions=11")

    private_root = Path(tempfile.mkdtemp(prefix="exp7368-validation-", dir="/tmp"))
    phase_start = time.monotonic()
    progress(run_started, "validation", "start", "build Exp7358 affected plan")
    commands = build_validation_plan(REPO_ROOT, private_root)
    plan_errors = validate_validation_plan(REPO_ROOT, commands)
    if plan_errors:  # pragma: no cover - fixed builder and validator share the contract.
        raise RuntimeError(f"invalid Exp7358 command plan: {plan_errors}")
    progress(run_started, "validation", "before_subprocess", "affected_checks=8")
    affected_receipts, affected = run_affected_validation(
        REPO_ROOT, commands, raw_dir / "affected_validation"
    )
    progress(
        run_started,
        "validation",
        "after_subprocess",
        f"affected_passed={affected['passed']}",
    )
    full_command = build_full_suite_command(REPO_ROOT, private_root)
    progress(run_started, "validation", "before_subprocess", "full_python_suite")
    full_receipts = scoped.run_commands(
        REPO_ROOT, [full_command], log_dir=raw_dir / "full_python_suite"
    )
    progress(
        run_started,
        "validation",
        "after_subprocess",
        f"full_python_suite_passed={full_receipts[0]['passed']}",
    )
    preliminary_validation: JsonDict = {
        "validation_receipts": [*affected_receipts, *full_receipts],
        "affected_validation_passed": affected["passed"],
        "full_suite_passed": full_receipts[0]["passed"] is True,
        "terminal_validation_passed": False,
        "flagged_adversarial": False,
        "repository_health": {
            "status": "healthy" if full_receipts[0]["passed"] else "degraded_current",
            "as_of": "2026-09-17",
            "affects_required_checks": full_receipts[0]["passed"] is not True,
            "historical_failures": [],
        },
    }
    candidate = build_artifact(
        root=REPO_ROOT,
        contract=contract,
        evidence=evidence,
        validation=preliminary_validation,
        historical_sidecar=historical_sidecar,
        independent_sidecar=independent_sidecar,
        started_at_utc=started_at,
        completed_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - run_started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_write_json(candidate_path, candidate)
    progress(run_started, "terminal_validation", "before_subprocess", "three candidate checks")
    terminal_receipts = run_terminal_validation(
        REPO_ROOT, candidate_path, raw_dir / "terminal_validation"
    )
    progress(run_started, "terminal_validation", "after_subprocess", "checks=3")
    phase_end = time.monotonic()
    spans.append(_span("validation", phase_start, phase_end, run_started, 12))

    validation: JsonDict = {
        **preliminary_validation,
        "validation_receipts": [
            *preliminary_validation["validation_receipts"],
            *terminal_receipts,
        ],
        "terminal_validation_passed": all(row["passed"] is True for row in terminal_receipts),
        "flagged_adversarial": any(
            row["name"] == "adversarial_verify" and row["passed"] is not True
            for row in terminal_receipts
        ),
    }

    phase_start = time.monotonic()
    progress(run_started, "write", "start", "build terminal self-disposition")
    phase_end = time.monotonic()
    spans.append(_span("write", phase_start, phase_end, run_started, 1))
    final = build_artifact(
        root=REPO_ROOT,
        contract=contract,
        evidence=evidence,
        validation=validation,
        historical_sidecar=historical_sidecar,
        independent_sidecar=independent_sidecar,
        started_at_utc=started_at,
        completed_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - run_started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=REPO_ROOT, replay=True)
    if errors:
        progress(run_started, "write", "failed", f"validation_errors={errors}")
        return 1
    output = REPO_ROOT / RESULT_PATH
    atomic_write_json(output, final)
    reloaded = read_json(output)
    reload_errors = validate_artifact(reloaded, root=REPO_ROOT, replay=True)
    progress(run_started, "write", "end", f"output={RESULT_PATH} errors={reload_errors}")
    return int(bool(reload_errors))


if __name__ == "__main__":  # pragma: no cover - thin wrapper is the declared entrypoint.
    raise SystemExit(main())
