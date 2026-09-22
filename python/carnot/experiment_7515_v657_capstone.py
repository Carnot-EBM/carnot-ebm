"""Close V657 with thirteen authenticated dispositions and separate claims.

This reducer reads immutable producer evidence. It does not load a model,
activate a roadmap, change defaults, or publish externally.

Spec refs: REQ-REPORT-7515 and SCENARIO-REPORT-7515-*.
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
from carnot.experiment_7503_v657_contract_methods import (
    compare_contract_authorities,
    resolve_v657_roadmap,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7515-capstone"
SCHEMA = "carnot.exp7515.v657.capstone.v1"

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7515_v657_capstone.json")
RAW_DIR = Path("results/raw/experiment_7515_v657_capstone")
MODULE_PATH = Path("python/carnot/experiment_7515_v657_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7515_v657_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7515_v657_capstone.py")
RETROSPECTIVE_PATH = Path("docs/research-notes/v657-retrospective.md")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = (
    "exp7503-contract-methods",
    "exp7504-evidence-interface",
    "exp7505-energy-fit",
    "exp7506-causal-prototype",
    "exp7507-static-evaluation",
    "exp7508-static-audit",
    "exp7509-causal-online",
    "exp7510-causal-audit",
    "exp7511-arc-evidence-recovery",
    "exp7512-arc-opportunity",
    "exp7513-placement-continuity",
    "exp7514-service-trace",
    "exp7515-capstone",
)
CLOSED_VERDICTS = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
RANDOM_SEED = {
    "fitting": None,
    "arrival": None,
    "audit": 7_515_657_01,
    "bootstrap": None,
    "explanation": "Deterministic aggregation uses no fitting, arrival, or bootstrap draw.",
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7502_v656_capstone.py"),
    Path("results/experiment_7502_v656_capstone.json"),
    PUBLICATION_GATE_PATH,
    Path("ops/north-star.md"),
    Path("research-studying.md"),
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


def load_contract(root: Path) -> JsonDict:
    """Resolve the matching V657 YAML and compare it with the Markdown table."""

    selected, roadmap, candidates = resolve_v657_roadmap(root)
    comparison = compare_contract_authorities((root / DESIGN_PATH).read_text(), roadmap)
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V657 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": candidates,
        "roadmap": deepcopy(roadmap),
        "tasks": deepcopy(tasks),
    }


@lru_cache(maxsize=2)
def evaluate_publication_gates(root: Path) -> JsonDict:
    """Run the fixed G1-G4 reader and retain its actual read-only output."""

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
        "stdout_sha256": "sha256:" + hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "publication_performed": False,
    }


def _ready_value_fields(payload: Mapping[str, Any]) -> JsonDict:
    """Keep bare top-level readiness and value scores without interpreting them."""

    suffixes = ("_score", "_ready", "_complete")
    return {
        str(key): deepcopy(value)
        for key, value in sorted(payload.items())
        if str(key).endswith(suffixes)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _validation_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Retain exact validation outcomes and log identities, not large output tails."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list):
        return []
    fields = ("name", "required", "passed", "exit_code", "log_path", "log_sha256")
    return [
        {key: deepcopy(row.get(key)) for key in fields}
        for row in receipts
        if isinstance(row, Mapping)
    ]


def _gate_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve producer gate operands and principles without recomputing them."""

    gates = payload.get("acceptance_gate_results")
    if not isinstance(gates, list):
        return []
    fields = (
        "check",
        "category",
        "upstream",
        "path",
        "field",
        "field_path",
        "artifact_field",
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


def _producer_receipts_pass(payload: Mapping[str, Any]) -> bool:
    """Use explicit required markers, or the producer's all-receipts convention."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list) or not receipts:
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    explicit = [row for row in rows if row.get("required") is True]
    required = explicit or rows
    return bool(required) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in required
    )


def _missing_producer(task: Mapping[str, Any]) -> JsonDict:
    """Record an absent external producer as blocked, never retryable partial."""

    path = str(task.get("deliverable") or "")
    failure = {
        "check": "producer_artifact_exists",
        "upstream": str(task.get("id")),
        "path": path,
        "field": "path",
        "op": "exists",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    return {
        "task_id": str(task.get("id")),
        "expected_path": path,
        "artifact_path": None,
        "source_sha256": None,
        "source_size_bytes": 0,
        "evidence_state": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "honest_verdict": "blocked_missing_declared_producer_evidence",
        "verdict_class": "blocked",
        "original_verdict_class": None,
        "flagged_adversarial": False,
        "model_invoked": False,
        "inference_substrate": None,
        "inference_substrate_class": None,
        "ready_value_fields": {},
        "row_receipt": {},
        "raw_row_references": [],
        "validation_receipt": {},
        "validation_receipts": [],
        "acceptance_gates": [],
        "gate_check_summary": failure,
        "payload": {},
    }


def _invalid_producer(task: Mapping[str, Any], path: Path, error: str) -> JsonDict:
    """Keep malformed present bytes distinct from external absence."""

    row = _missing_producer(task)
    row.update(
        {
            "artifact_path": path.as_posix(),
            "source_sha256": sha256_file(path),
            "source_size_bytes": path.stat().st_size,
            "evidence_state": "invalid",
            "available": True,
            "honest_verdict": "complete_disqualified_unreadable_producer_evidence",
            "verdict_class": "disqualified",
            "gate_check_summary": {
                "check": "producer_json_object",
                "upstream": str(task.get("id")),
                "path": path.as_posix(),
                "field": "json_object",
                "op": "is",
                "expected": "mapping",
                "observed": error,
                "passed": False,
            },
        }
    )
    return row


def _first_producer_failure(
    task_id: str, relative: Path, payload: Mapping[str, Any], validation: Mapping[str, Any]
) -> JsonDict:
    """Name the exact validation, flag, or verdict condition that invalidated evidence."""

    for row in payload.get("validation_receipts") or []:
        if (
            isinstance(row, Mapping)
            and row.get("required") is True
            and (row.get("passed") is not True or row.get("exit_code") != 0)
        ):
            return {
                "check": "producer_required_validation",
                "upstream": task_id,
                "path": str(row.get("log_path") or relative),
                "field": str(row.get("name")),
                "op": "==",
                "expected": {"passed": True, "exit_code": 0},
                "observed": {"passed": row.get("passed"), "exit_code": row.get("exit_code")},
                "passed": False,
            }
    if payload.get("flagged_adversarial") is True:
        return {
            "check": "producer_adversarial_flag",
            "upstream": task_id,
            "path": relative.as_posix(),
            "field": "flagged_adversarial",
            "op": "==",
            "expected": False,
            "observed": True,
            "passed": False,
        }
    return {
        "check": "producer_validity",
        "upstream": task_id,
        "path": relative.as_posix(),
        "field": "verdict_class",
        "op": "!=",
        "expected": "disqualified",
        "observed": payload.get("verdict_class"),
        "passed": False,
        "validation_failures": deepcopy(validation.get("failures", [])),
    }


def load_producer(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one declared V657 producer and its row and log sidecars."""

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
        "required_passed": _producer_receipts_pass(payload),
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
    row_refs = [
        {
            "path": relative.as_posix(),
            "kind": "inline_rows",
            "rows": rows.get("inline_row_count"),
            "sha256": rows.get("inline_rows_sha256"),
        }
    ]
    row_refs.extend(deepcopy(payload.get("row_shards") or []))
    state = "terminal" if valid else "invalid"
    return {
        "task_id": str(task.get("id")),
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
        "raw_row_references": row_refs,
        "validation_receipt": validation,
        "validation_receipts": _validation_rows(payload),
        "acceptance_gates": _gate_rows(payload),
        "gate_check_summary": (
            deepcopy(payload.get("gate_check_summary"))
            if valid
            else _first_producer_failure(str(task.get("id")), relative, payload, validation)
        ),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Inventory the twelve preceding tasks in conductor order."""

    return {str(task["id"]): load_producer(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, current_validation_complete: bool
) -> JsonDict:
    """Classify owned incompletion before invalid, blocked, and valid science."""

    if not current_validation_complete:
        verdict = "partial"
        honest = "partial_retryable_current_capstone_validation_unfinished"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict = "disqualified"
        honest = "complete_disqualified_required_v657_evidence"
    elif any(
        row.get("evidence_state") == "missing" or row.get("verdict_class") == "blocked"
        for row in rows
    ):
        verdict = "blocked"
        honest = "complete_blocked_required_v657_evidence_absent_or_externally_gated"
    else:
        verdict = "null"
        honest = "complete_null_v657_accounting_without_aggregate_scientific_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _claim(evidence: Mapping[str, JsonDict], task_id: str, claim: str, **values: Any) -> JsonDict:
    """Attach one claim to its exact source and validity boundary."""

    source = evidence[task_id]
    state = "disqualified" if not source["valid"] else str(source["original_verdict_class"])
    return {
        "claim": claim,
        "state": state,
        "source_task_id": task_id,
        "source_path": source["artifact_path"],
        "source_sha256": source["source_sha256"],
        "source_valid": source["valid"],
        "source_honest_verdict": source["honest_verdict"],
        **deepcopy(values),
    }


def build_claim_ledger(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep evidence, value, causality, ARC, placement, and timing independent."""

    field = lambda task, name: evidence[task]["ready_value_fields"].get(name)  # noqa: E731
    return [
        _claim(
            evidence,
            "exp7504-evidence-interface",
            "native_evidence_quality",
            ready_value=field("exp7504-evidence-interface", "evidence_ready_score"),
            historical_native_calls_only=True,
            current_model_invoked=False,
        ),
        _claim(
            evidence,
            "exp7508-static-audit",
            "static_probability_value",
            ready_value=field("exp7508-static-audit", "static_claims_qualified_score"),
            qualified_value=field(
                "exp7508-static-audit", "qualified_static_probability_value_score"
            ),
            independent_audit_propagated=True,
        ),
        _claim(
            evidence,
            "exp7508-static-audit",
            "selective_decision_value",
            ready_value=field("exp7508-static-audit", "static_claims_qualified_score"),
            qualified_value=field(
                "exp7508-static-audit", "qualified_selective_decision_value_score"
            ),
            independent_audit_propagated=True,
        ),
        _claim(
            evidence,
            "exp7510-causal-audit",
            "causal_information",
            ready_value=field("exp7510-causal-audit", "causal_claims_qualified_score"),
            qualified_value=field("exp7510-causal-audit", "qualified_online_benefit_score"),
            independent_audit_propagated=True,
        ),
        _claim(
            evidence,
            "exp7509-causal-online",
            "retention",
            ready_value=field("exp7509-causal-online", "restart_parity_score"),
            qualified_value=field("exp7509-causal-online", "online_benefit_score"),
            reported_separately=True,
        ),
        _claim(
            evidence,
            "exp7511-arc-evidence-recovery",
            "arc_reachability",
            ready_value=field("exp7511-arc-evidence-recovery", "arc_panel_b_qualified_score"),
            qualified_value=field("exp7511-arc-evidence-recovery", "scientific_benefit_score"),
            live_benchmark_result=False,
            historical_raw_recovery=True,
        ),
        _claim(
            evidence,
            "exp7512-arc-opportunity",
            "arc_opportunity",
            ready_value=field("exp7512-arc-opportunity", "arc_opportunity_audit_complete_score"),
            qualified_value=field("exp7512-arc-opportunity", "arc_cost_claim_ready_score"),
            live_benchmark_result=False,
        ),
        _claim(
            evidence,
            "exp7513-placement-continuity",
            "host_quantization",
            ready_value=field("exp7513-placement-continuity", "numeric_placement_ready_score"),
            measured_board_performance=False,
        ),
        _claim(
            evidence,
            "exp7513-placement-continuity",
            "historical_board_continuity",
            ready_value=field("exp7513-placement-continuity", "board_continuity_complete_score"),
            current_board_measurement=False,
        ),
        _claim(
            evidence,
            "exp7514-service-trace",
            "current_machine_service_timing",
            ready_value=field("exp7514-service-trace", "durable_service_claim_ready_score"),
            trace_complete=field("exp7514-service-trace", "service_trace_complete_score"),
            current_model_invoked=evidence["exp7514-service-trace"]["model_invoked"],
            efficacy_claim=False,
            production_sla_claim=False,
        ),
    ]


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    *,
    current_validation_complete: bool,
) -> list[JsonDict]:
    """Build twelve source rows and one non-self-dependent current row."""

    result: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        source = evidence[str(task["id"])]
        result.append(
            {
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
                "raw_row_references": deepcopy(source["raw_row_references"]),
                "validation_receipt": deepcopy(source["validation_receipt"]),
                "validation_receipts": deepcopy(source["validation_receipts"]),
                "acceptance_gates": deepcopy(source["acceptance_gates"]),
                "gate_check_summary": deepcopy(source["gate_check_summary"]),
                "attempted": source["evidence_state"] != "missing",
                "completed": source["evidence_state"] != "missing",
                "failed": source["evidence_state"] == "invalid",
                "censored": source["evidence_state"] == "missing",
                "unstarted": source["evidence_state"] == "missing",
                "excluded_from_positive_aggregate": not source["valid"],
                "principle": "Literal source rows prevent disposition or score laundering.",
            }
        )
    result.append(
        {
            "order": 13,
            "task_id": str(tasks[-1]["id"]),
            "expected_artifact_path": str(tasks[-1]["deliverable"]),
            "artifact_path": None,
            "artifact_sha256": None,
            "evidence_state": "current_work",
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "original_verdict_class": terminal["verdict_class"],
            "flagged_adversarial": False,
            "ready_value_fields": {"capstone_complete_score": int(current_validation_complete)},
            "row_receipt": {},
            "raw_row_references": [],
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
    return result


def reduce_prior_failures(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Apply the declared four-field retirement rule to literal repeated verdicts."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        priors = task.get("prior_failures") or []
        if not priors or any(not isinstance(prior, Mapping) for prior in priors):
            raise ValueError(f"at least one prior failure required for {task_id}")
        source = evidence.get(task_id)
        if source is None:
            comparison = "current_work"
            current_verdict = None
            current_class = None
        elif source["evidence_state"] == "missing":
            comparison = "external_absence"
            current_verdict = source["honest_verdict"]
            current_class = source["verdict_class"]
        elif source["evidence_state"] == "invalid":
            comparison = "current_invalid"
            current_verdict = source["honest_verdict"]
            current_class = source["verdict_class"]
        else:
            comparison = "compared"
            current_verdict = source["honest_verdict"]
            current_class = source["verdict_class"]
        for prior in priors:
            same = bool(comparison == "compared" and current_verdict == prior.get("verdict"))
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment": prior.get("experiment_id"),
                    "prior_honest_verdict": prior.get("verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "comparison_state": comparison,
                    "current_honest_verdict": current_verdict,
                    "current_verdict_class": current_class,
                    "exact_text_match": current_verdict == prior.get("verdict"),
                    "retirement_triggered": bool(
                        prior.get("retire_if_same_verdict") is True and same
                    ),
                }
            )
    return rows


def retirement_rows(prior_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retire only a bounded task whose literal registered failure repeated."""

    return [
        {
            "retired_task_id": str(row.get("task_id")),
            "prior_experiment": row.get("prior_experiment"),
            "prior_honest_verdict": row.get("prior_honest_verdict"),
            "current_honest_verdict": row.get("current_honest_verdict"),
            "changed_mechanism": row.get("addressed_by"),
            "retire_if_same_verdict": row.get("retire_if_same_verdict"),
            "scope": "bounded_mechanism_only",
            "exclusion_workflow": "ops/exclusion_manifest.yaml",
        }
        for row in prior_rows
        if row.get("retirement_triggered") is True
    ]


def next_conditions() -> list[JsonDict]:
    """Name evidence changes needed before closed branches can make new claims."""

    return [
        {
            "branch": "static_probability_and_decision_value",
            "state": "closed_disqualified",
            "reopen_when": "Repair Exp7508 strict row-consistency validation without changing frozen thresholds, then rerun the independent audit.",
        },
        {
            "branch": "causal_feedback_benefit",
            "state": "closed_valid_null",
            "reopen_when": "Use a registered feedback design that supplies at least 12 permutable labels per seed and changes the bounded update mechanism.",
        },
        {
            "branch": "arc_live_benchmark_value",
            "state": "closed_valid_null",
            "reopen_when": "Acquire qualified live hidden-game progress and eligible supervisor opportunities; recovered raw candidates alone do not qualify.",
        },
        {
            "branch": "hardware_service_speedup",
            "state": "closed_scope_limited",
            "reopen_when": "Run the qualified head on a board with the same complete durable service denominator and retain transfer costs.",
        },
    ]


def build_retrospective(artifact: Mapping[str, Any]) -> JsonDict:
    """Reduce the thirteen dispositions into three bounded PRD gap updates."""

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
                "gap": "Static calibrated decisions",
                "status": "disqualified_independent_static_audit",
                "evidence": "Exp7507 is a valid exploratory null, but Exp7508 failed required strict row consistency and remains flagged.",
            },
            {
                "gap": "Causal feedback learning",
                "status": "valid_null_weak_permutation_support",
                "evidence": f"Exp7510 qualified the audit but retained benefit={claims.get('causal_information', {}).get('qualified_value')}.",
            },
            {
                "gap": "ARC service and hardware scope",
                "status": "qualified_components_without_live_arc_or_board_speedup",
                "evidence": "Panel B custody, host quantization, board continuity, and service timing completed; ARC progress, eligible opportunity, and measured board service gain did not.",
            },
        ],
        "aggregate_verdict_class": artifact.get("verdict_class"),
        "capstone_complete_score": artifact.get("capstone_complete_score"),
        "next_conditions": deepcopy(artifact.get("next_conditions") or []),
    }


def retrospective_markdown(value: Mapping[str, Any]) -> str:
    """Render the exact dispositions and PRD gaps as a small durable note."""

    lines = [
        "# V657 retrospective",
        "",
        "Milestone accounting is complete independently from scientific benefit.",
        "",
        "## Thirteen dispositions",
        "",
        "| Task | Class | Literal disposition | Artifact |",
        "|---|---|---|---|",
    ]
    for row in value.get("task_dispositions") or []:
        lines.append(
            f"| {row['task_id']} | {row['verdict_class']} | `{row['honest_verdict']}` | `{row['artifact_path']}` |"
        )
    lines.extend(["", "## PRD gap updates", ""])
    for row in value.get("prd_gap_updates") or []:
        lines.extend([f"### {row['gap']}", "", f"Status: `{row['status']}`. {row['evidence']}", ""])
    lines.extend(
        [
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
    """Describe one gate with the operands needed for an independent replay."""

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
    """Keep evidence validity, completion readiness, and scientific benefit separate."""

    by_claim = {str(row["claim"]): row for row in claims}
    present_valid = all(row["valid"] for row in evidence.values())
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    validity = "Favorable metrics cannot excuse invalid evidence."
    readiness = "A valid null must not prevent complete milestone accounting."
    benefit = "Exploratory or historical evidence cannot become a confirmatory claim."
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v657_contract",
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
            "v657_producers",
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
            "thirteen_dispositions_recorded",
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
            "qualified_static_probability_value",
            "benefit",
            "exp7508-static-audit",
            "claim_ledger.static_probability_value",
            "qualified_value",
            1,
            by_claim["static_probability_value"]["qualified_value"],
            by_claim["static_probability_value"]["qualified_value"] == 1,
            benefit,
        ),
        _gate(
            "qualified_selective_decision_value",
            "benefit",
            "exp7508-static-audit",
            "claim_ledger.selective_decision_value",
            "qualified_value",
            1,
            by_claim["selective_decision_value"]["qualified_value"],
            by_claim["selective_decision_value"]["qualified_value"] == 1,
            benefit,
        ),
        _gate(
            "qualified_causal_information",
            "benefit",
            "exp7510-causal-audit",
            "claim_ledger.causal_information",
            "qualified_value",
            1,
            by_claim["causal_information"]["qualified_value"],
            by_claim["causal_information"]["qualified_value"] == 1,
            benefit,
        ),
        _gate(
            "qualified_live_arc_progress_and_opportunity",
            "benefit",
            "exp7511/exp7512",
            "claim_ledger.arc_reachability+arc_opportunity",
            "qualified_value",
            {"progress": 1, "opportunity": 1},
            {
                "progress": by_claim["arc_reachability"]["qualified_value"],
                "opportunity": by_claim["arc_opportunity"]["qualified_value"],
            },
            bool(
                by_claim["arc_reachability"]["qualified_value"] == 1
                and by_claim["arc_opportunity"]["qualified_value"] == 1
            ),
            benefit,
        ),
    ]


def failure_rows(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Return only failures that determine invalid, blocked, or unfinished state."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "check": "contract_authorities_agree",
                "upstream": "v657_contract",
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
            failures.append(
                {
                    "check": "producer_terminal_disposition",
                    "upstream": source["task_id"],
                    "path": source["artifact_path"],
                    "field": "verdict_class",
                    "op": "!=",
                    "expected": "blocked",
                    "observed": "blocked",
                    "passed": False,
                }
            )
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
    """Retain every classification failure and the first exact cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


def _preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record exact resources, owners, hashes, and observed producer states."""

    rows = [
        {
            "check": f"source_bytes:{relative.as_posix()}",
            "upstream": relative.as_posix(),
            "path": relative.as_posix(),
            "owner": "repository",
            "field": "bytes",
            "expected": "readable_nonempty_bytes",
            "observed": "readable_nonempty_bytes",
            "passed": (root / relative).is_file() and (root / relative).stat().st_size > 0,
            "sha256": sha256_file(root / relative),
        }
        for relative in INPUT_PATHS
    ]
    rows.append(
        {
            "check": "contract_authority_equivalence",
            "upstream": "v657_contract",
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
    """Bind instructions, implementation, note, authorities, and producer bytes."""

    paths = [*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH]
    if (root / RETROSPECTIVE_PATH).is_file():
        paths.append(RETROSPECTIVE_PATH)
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


FIELD_PRINCIPLES = {
    "schema": "Version, experiment identity, milestone, and status prevent reader drift.",
    "run_date": "The frozen date stays distinct from measured UTC and monotonic boundaries.",
    "preconditions_checked": "Exact paths and hashes prevent aggregation over imagined inputs.",
    "MODEL_SPECS": "Empty current model declarations prevent historical calls from becoming current calls.",
    "model_specs": "Both model-spec aliases stay empty for aggregation-only work.",
    "model_invoked": "Current model work remains distinct from producer model provenance.",
    "invocation_counts": "Balanced zero counts expose invented loads, forwards, or generations.",
    "inference_substrate": "The exact aggregation label prevents a cached calculation from becoming inference.",
    "inference_substrate_class": "The aggregation class prevents duration padding or model implications.",
    "execution_venue": "Host aggregation stays distinct from archived FPGA and board CPU evidence.",
    "duration_s": "Measured duration separates current aggregation and validation from historical work.",
    "phase_spans": "Flushed boundaries and checkpoints expose unfinished or silent operations.",
    "random_seed": "Frozen applicable seeds prevent favorable reruns from multiplying support.",
    "reproducibility_checksum": "One checksum binds sources, rows, claims, gates, and validation scope.",
    "source_artifact_hashes": "Exact bytes preserve producer verdicts, flags, exposure, and row identity.",
    "rows": "Thirteen ordered rows make every milestone disposition independently reducible.",
    "sample_size_budget": "Attempted, complete, failed, censored, and unstarted units remain separate.",
    "acceptance_gate_results": "Validity, readiness, and benefit cannot substitute for one another.",
    "gate_check_summary": "Every invalid or blocked state names exact expected and observed operands.",
    "honest_verdict": "A complete prefix records terminal accounting without softening disqualification.",
    "verdict_class": "The closed enum reserves partial for repairable current work only.",
    "verifier_is_oracle": "Oracle-defined evidence cannot become an oracle-distinct universal claim.",
    "flagged_adversarial": "An upstream flag stays visible and cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, exits, scopes, and hashes make current checks auditable.",
    "field_principles": "Every top-level field explains the inference failure it prevents.",
    "capstone_complete_score": "A bare one records complete accounting, not scientific success.",
    "task_dispositions": "Exactly thirteen ordered source states match both authorities.",
    "claim_ledger": "Independent evidence boundaries prevent cross-claim promotion.",
    "retirement_rows": "Only a literal repeated registered failure retires a bounded mechanism.",
    "publication_gate_results": "Actual read-only G1-G4 outcomes do not perform publication.",
    "next_conditions": "New evidence or a changed method is required before reopening scope.",
}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give every top-level field one plain failure-prevention explanation."""

    return {
        field: FIELD_PRINCIPLES.get(
            field,
            f"Exact {field} evidence prevents a later reader from inferring an unstated result.",
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
    """Build one compact terminal record from authenticated V657 evidence."""

    owned_complete = validation.get("required_checks_passed") is True
    terminal_complete = validation.get("terminal_validation_passed") is True
    terminal = classify_terminal(
        list(evidence.values()), current_validation_complete=owned_complete
    )
    dispositions = task_dispositions(
        contract["tasks"], evidence, terminal, current_validation_complete=owned_complete
    )
    claims = build_claim_ledger(evidence)
    priors = reduce_prior_failures(contract["tasks"], evidence)
    failures = failure_rows(contract, evidence, validation)
    receipts = deepcopy(validation.get("validation_receipts") or [])
    validation_s = sum(
        float(row.get("duration_s") or 0.0) for row in receipts if isinstance(row, Mapping)
    )
    capstone_complete = int(
        contract.get("comparison_passed") is True
        and len(dispositions) == 13
        and owned_complete
        and terminal_complete
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
        "clock_identity": {"wall": "datetime.now(datetime.UTC)", "monotonic": "time.monotonic_ns"},
        "process_identity": {"pid": os.getpid(), "node": platform.node()},
        "device_identity": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cuda_used": False,
        },
        "duration_components_s": {
            "authoring": 0.0,
            "computation": max(0.0, duration_s - validation_s),
            "validation": validation_s,
            "historical_capture": 0.0,
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
        "preconditions_checked": _preconditions(root, contract, evidence),
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_evidence_sidecars": [
            {
                "task_id": task_id,
                "path": source["artifact_path"],
                "sha256": source["source_sha256"],
                "historical_model_invoked": source["model_invoked"],
                "counted_as_current_invocation": False,
            }
            for task_id, source in evidence.items()
        ],
        "rows": deepcopy(dispositions),
        "task_dispositions": dispositions,
        "sample_size_budget": {
            "independent_unit": "ordered_v657_task_disposition",
            "planned": 13,
            "attempted": sum(int(row["attempted"]) for row in dispositions),
            "complete": sum(int(row["completed"]) for row in dispositions),
            "excluded": sum(int(row["excluded_from_positive_aggregate"]) for row in dispositions),
            "failed": sum(int(row["failed"]) for row in dispositions),
            "censored": sum(int(row["censored"]) for row in dispositions),
            "unstarted": sum(int(row["unstarted"]) for row in dispositions),
        },
        "claim_ledger": claims,
        "prior_failure_rows": priors,
        "retirement_rows": retirement_rows(priors),
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
        "verifier_is_oracle": False,
        "roadmap_activated": False,
        "publication_performed": False,
        "submission_performed": False,
        "external_contact_performed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "push_performed": False,
        "research_conductor_changed": False,
    }
    artifact["retrospective"] = build_retrospective(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


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
        {"phase": phase, "duration_s": 0.0, "completed_units": units, "checkpoint": checkpoint}
        for phase, units, checkpoint in (
            ("preconditions", 12, "twelve_producers_observed"),
            ("publication_gate", 4, "g1_g4_read_only"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("reduction", 13, "thirteen_dispositions_reduced"),
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
        terminal = classify_terminal(
            list(evidence.values()),
            current_validation_complete=validation["required_checks_passed"],
        )
        expected_rows = task_dispositions(
            contract["tasks"],
            evidence,
            terminal,
            current_validation_complete=validation["required_checks_passed"],
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
            contract["comparison_passed"] is True
            and len(expected_rows) == 13
            and validation["required_checks_passed"]
            and validation["terminal_validation_passed"]
        )
        if artifact.get("capstone_complete_score") != expected_complete:
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
    """Build the fixed Exp7358 plan for only the affected Exp7515 files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V657 execution date."""

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
        f"[exp7515] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
    """Replace one generated note only after its complete bytes reach storage."""

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
    """Build the declared replay, independent reducer, and two strict readers."""

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
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7515-", dir="/tmp"))

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
            checkpoint="twelve_producers_observed",
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
    _atomic_text(root / RETROSPECTIVE_PATH, retrospective_markdown(preliminary["retrospective"]))
    spans.append(
        _span(
            "reduction",
            phase_started,
            started,
            completed_units=13,
            checkpoint="thirteen_dispositions_and_retrospective_reduced",
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

    print("[exp7515] phase=startup event=flushed", flush=True)
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
