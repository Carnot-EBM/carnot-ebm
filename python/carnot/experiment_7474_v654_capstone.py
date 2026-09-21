"""Close V654 with fourteen authenticated task dispositions.

This reducer reads existing artifacts and summarizes their bounded evidence. It
does not invoke a model, fit a numeric head, operate hardware, or publish.

Spec refs: REQ-REPORT-7474 and SCENARIO-REPORT-7474-*.
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
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import (
    parse_markdown_contract,
    parse_yaml_contract,
)
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
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts.publication_gate import evaluate as evaluate_publication_gates


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.654"
EXPERIMENT_ID = "exp7474-capstone"
SCHEMA = "carnot.exp7474.v654.capstone.v1"

ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7474_v654_capstone.json")
RAW_DIR = Path("results/raw/experiment_7474_v654_capstone")
MODULE_PATH = Path("python/carnot/experiment_7474_v654_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7474_v654_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7474_v654_capstone.py")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = (
    "exp7461-contract-methods",
    "exp7462-option-protocol",
    "exp7463-semif-e0-logprob-parity",
    "exp7464-semif-e6-decision-cost-profile",
    "exp7465-source-option-capture",
    "exp7466-typed-energy-calibration",
    "exp7467-factual-span-canary",
    "exp7468-residual-learner",
    "exp7469-continuous-residual-learning",
    "exp7470-independent-audit",
    "exp7471-arc-seam-observation",
    "exp7472-prefix-service",
    "exp7473-board-continuity",
    "exp7474-capstone",
)
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
SOURCE_PATHS = (
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
    SPEC_PATH,
    Path("results/experiment_7460_v653_capstone.json"),
    Path("results/experiment_7470_v654_independent_audit.json"),
    PUBLICATION_GATE_PATH,
    Path("ops/verifier_gaps.md"),
    Path("_bmad/traceability.md"),
    ROADMAP_PATH,
    DESIGN_PATH,
)
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "publication_gate_json",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with exact experiment identity, milestone, and terminal status.",
    "run_date": "Use 20260921 with measured UTC and monotonic boundaries and a named clock.",
    "preconditions_checked": "Record exact paths, ownership, device identity, and observed prerequisites before work.",
    "MODEL_SPECS": "Use an empty list because this aggregation performs no current LLM work.",
    "model_specs": "Repeat the empty model list for lowercase readers.",
    "model_invoked": "Keep current attempted calls separate from archived producer calls.",
    "invocation_counts": "Balance current model loads, forwards, and generations at zero.",
    "inference_substrate": "Name aggregation from upstream artifacts for this pure reducer.",
    "inference_substrate_class": "Use aggregation because no model is loaded or fitted.",
    "execution_venue": "Use host and keep historical CUDA and board evidence in source rows.",
    "duration_s": "Measure current work and separate reduction from validation without padding.",
    "phase_spans": "Bind progress phases to monotonic times and completed-unit checkpoints.",
    "random_seed": "Freeze ordering, audit, and bootstrap seeds while declaring no fit seed.",
    "reproducibility_checksum": "Bind code, protocol, source bytes, rows, decisions, and validation scope.",
    "source_artifact_hashes": "Retain exact upstream bytes, original classes, and original flags.",
    "rows": "Keep one ordered row for every task, including invalid and unavailable units.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Keep validity, completion, scientific benefit, and publication scope separate.",
    "gate_check_summary": "Name each failed check and its exact upstream field and path.",
    "honest_verdict": "Finish the capstone without converting invalid or missing science into benefit.",
    "verdict_class": "Use the closed terminal enum; external absence is never retryable partial work.",
    "verifier_is_oracle": "False because this capstone supplies no evaluation labels.",
    "flagged_adversarial": "Retain source structural flags without clearing them for a gate.",
    "validation_receipts": "Capture exact affected commands, exits, log hashes, and required status.",
    "field_principles": "Explain why each artifact field and acceptance gate exists.",
    "task_dispositions": "Keep exactly fourteen ordered entries, including pre-gates and current work.",
    "continuation_rows": "Name one measured cause and exact changed prerequisite for each branch.",
    "publication_gates": "Keep the stable historical G1-G4 result in its FoVer-only scope.",
    "capstone_complete_score": "Use one only for complete reconciliation, independent of science success.",
    "unresolved_obligations": "Keep forbidden and external work explicit without causing endless retries.",
}


def load_yaml_mapping(path: Path) -> JsonDict:
    """Read one YAML mapping and reject another top-level shape."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def _public_task(task: Mapping[str, Any]) -> JsonDict:
    """Keep only fields independently represented by both authorities."""

    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
    }


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare every V654 contract field without using an older capstone."""

    try:
        markdown = parse_markdown_contract(markdown_text)
        parsed_yaml = parse_yaml_contract(roadmap)
    except (TypeError, ValueError) as error:
        return {"comparison_passed": False, "errors": [f"parse_error:{error}"], "rows": []}
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    width = max(len(EXPECTED_TASK_IDS), len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else {}
        observed = yaml_tasks[index] if index < len(yaml_tasks) else {}
        checks = {
            field: expected.get(field) == observed.get(field) and bool(expected) and bool(observed)
            for field in ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
        }
        rows.append(
            {
                "order": index + 1,
                "task_id": str(observed.get("id") or expected.get("id") or f"missing-{index}"),
                "markdown": _public_task(expected),
                "yaml": _public_task(observed),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    errors: list[str] = []
    if markdown.get("milestone") != MILESTONE:
        errors.append("markdown_milestone")
    if parsed_yaml.get("milestone") != MILESTONE:
        errors.append("yaml_milestone")
    if [row.get("id") for row in markdown_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("markdown_task_order")
    if [row.get("id") for row in yaml_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("yaml_task_order")
    if len(markdown_tasks) != len(EXPECTED_TASK_IDS) or len(yaml_tasks) != len(EXPECTED_TASK_IDS):
        errors.append("task_count")
    if any(not row["passed"] for row in rows):
        errors.append("row_mismatch")
    return {
        "comparison_passed": not errors,
        "errors": list(dict.fromkeys(errors)),
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": parsed_yaml.get("milestone"),
        "rows": rows,
    }


def load_contract(root: Path) -> JsonDict:
    """Resolve the V654 roadmap and compare it with the Markdown authority."""

    candidates: list[JsonDict] = []
    selected = ROADMAP_PATH
    for path in (NEXT_ROADMAP_PATH, ROADMAP_PATH):
        exists = (root / path).is_file()
        value = load_yaml_mapping(root / path) if exists else {}
        matches = value.get("milestone") == MILESTONE
        candidates.append(
            {
                "path": path.as_posix(),
                "exists": exists,
                "observed_milestone": value.get("milestone"),
                "matches": matches,
            }
        )
        if matches:
            selected = path
            roadmap = value
            break
    else:
        roadmap = load_yaml_mapping(root / ROADMAP_PATH)
    comparison = compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V654 task list required")
    return {
        **comparison,
        "tasks": deepcopy(tasks),
        "selected_roadmap_path": selected.as_posix(),
        "resolution_candidates": candidates,
    }


def _fallback_pre_gate_path(task: Mapping[str, Any]) -> Path:
    """Derive the conductor's alternate result name from the task ID."""

    task_id = str(task.get("id") or "")
    number = numeric_experiment_id(task_id)
    slug = task_id.split("-", 1)[1].replace("-", "_")
    return Path(f"results/experiment_{number}_{slug}.json")


def _validation_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Retain validation meaning and log identity without copying output tails."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list):
        return []
    return [
        {
            key: deepcopy(row.get(key))
            for key in ("name", "required", "passed", "exit_code", "log_path", "log_sha256")
        }
        for row in receipts
        if isinstance(row, Mapping)
    ]


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Give absent external producer bytes a terminal blocked disposition."""

    expected = str(task.get("deliverable") or "")
    return {
        "task_id": str(task.get("id")),
        "expected_path": expected,
        "found_path": None,
        "evidence_state": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "raw_experiment_id": None,
        "raw_milestone": None,
        "raw_status": None,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_missing_declared_producer_evidence",
        "flagged_adversarial": False,
        "gate_values": [],
        "validation_receipts": [],
        "validation_receipts_authenticated": False,
        "validation_authentication_failures": ["producer_evidence_missing"],
        "required_validation_passed": False,
        "source_sha256": None,
        "row_count": 0,
        "payload": {},
        "gate_check_summary": {
            "check": "producer_evidence",
            "upstream": str(task.get("id")),
            "path": expected,
            "field": "path",
            "op": "exists",
            "expected": True,
            "observed": False,
            "passed": False,
        },
    }


def _pre_gate_evidence(
    root: Path, task: Mapping[str, Any], relative: Path, payload: JsonDict
) -> JsonDict:
    """Authenticate one conductor pre-gate record and its cited source bytes."""

    evidence_path = Path(str(payload.get("failed_evidence_path") or ""))
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    expected_hash = payload.get("failed_evidence_sha256")
    evidence_matches = (
        evidence_path.is_file()
        and isinstance(expected_hash, str)
        and sha256_file(evidence_path) == expected_hash
    )
    required = (
        "failed_upstream",
        "failed_field",
        "failed_operator",
        "failed_expected",
        "failed_observed",
    )
    authenticated = bool(
        numeric_experiment_id(payload.get("experiment")) == numeric_experiment_id(task.get("id"))
        and payload.get("blocked_at_layer") == "conductor_pre_gate"
        and payload.get("status") == "blocked"
        and all(field in payload for field in required)
        and evidence_matches
    )
    gate_values = deepcopy(payload.get("gates_evaluated") or [])
    return {
        "task_id": str(task.get("id")),
        "expected_path": str(task.get("deliverable") or ""),
        "found_path": relative.as_posix(),
        "evidence_state": "pre_gate" if authenticated else "invalid",
        "authenticated": authenticated,
        "available": False,
        "valid": authenticated,
        "raw_experiment_id": payload.get("experiment_id", payload.get("experiment")),
        "raw_milestone": payload.get("milestone"),
        "raw_status": payload.get("status"),
        "verdict_class": "blocked" if authenticated else "disqualified",
        "honest_verdict": str(payload.get("honest_verdict") or "blocked_invalid_pre_gate"),
        "flagged_adversarial": False,
        "gate_values": gate_values,
        "validation_receipts": [],
        "validation_receipts_authenticated": authenticated,
        "validation_authentication_failures": [],
        "required_validation_passed": authenticated,
        "source_sha256": sha256_file(root / relative),
        "row_count": 0,
        "payload": payload,
        "gate_check_summary": {
            "check": "structured_pre_gate",
            "upstream": payload.get("failed_upstream"),
            "path": relative.as_posix(),
            "field": payload.get("failed_field"),
            "op": payload.get("failed_operator"),
            "expected": payload.get("failed_expected"),
            "observed": payload.get("failed_observed"),
            "passed": False,
        },
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one exact producer, pre-gate record, or missing slot."""

    expected = Path(str(task.get("deliverable") or ""))
    relative = expected
    if not (root / relative).is_file():
        fallback = _fallback_pre_gate_path(task)
        if not (root / fallback).is_file():
            return _missing_evidence(task)
        relative = fallback
    payload = load_json_object(root / relative)
    if payload.get("schema") == "blocked_gate_check_v1":
        return _pre_gate_evidence(root, task, relative, payload)
    row_receipt = authenticate_row_manifest(root, payload)
    validation = authenticate_validation_receipts(root, payload)
    raw_id = payload.get("experiment_id", payload.get("experiment"))
    verdict = payload.get("verdict_class")
    authenticated = bool(
        numeric_experiment_id(raw_id) == numeric_experiment_id(task.get("id"))
        and payload.get("milestone") == MILESTONE
        and verdict in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
        and terminal_status(payload)
        and row_receipt["authenticated"]
        and validation["receipt_count"] > 0
    )
    flagged = payload.get("flagged_adversarial") is True
    valid = bool(
        authenticated
        and validation["required_passed"]
        and verdict != "disqualified"
        and not flagged
    )
    evidence_state = "terminal" if authenticated and valid else "invalid"
    return {
        "task_id": str(task.get("id")),
        "expected_path": expected.as_posix(),
        "found_path": relative.as_posix(),
        "evidence_state": evidence_state,
        "authenticated": authenticated,
        "available": authenticated,
        "valid": valid,
        "raw_experiment_id": raw_id,
        "raw_milestone": payload.get("milestone"),
        "raw_status": payload.get("status"),
        "verdict_class": verdict if authenticated else "disqualified",
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "flagged_adversarial": flagged,
        "gate_values": deepcopy(payload.get("acceptance_gate_results") or []),
        "validation_receipts": _validation_rows(payload),
        "validation_receipts_authenticated": validation["authenticated"],
        "validation_authentication_failures": deepcopy(validation["failures"]),
        "required_validation_passed": validation["required_passed"],
        "source_sha256": sha256_file(root / relative),
        "row_count": row_receipt["inline_row_count"] + row_receipt["raw_shard_row_count"],
        "payload": payload,
        "gate_check_summary": deepcopy(payload.get("gate_check_summary")),
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Authenticate the thirteen predecessor slots in contract order."""

    return {str(task["id"]): load_evidence_slot(root, task) for task in tasks[:-1]}


def _audit_branch(payload: Mapping[str, Any], name: str) -> JsonDict:
    """Return one independent audit branch without merging its benefit state."""

    branches = payload.get("branch_rows")
    if not isinstance(branches, list):
        return {}
    return deepcopy(
        next(
            (row for row in branches if isinstance(row, Mapping) and row.get("branch") == name),
            {},
        )
    )


def reduce_science(evidence: Mapping[str, JsonDict]) -> dict[str, JsonDict]:
    """Reduce raw and independent rows while keeping claim families separate."""

    e0 = evidence["exp7463-semif-e0-logprob-parity"]["payload"]
    e6 = evidence["exp7464-semif-e6-decision-cost-profile"]["payload"]
    audit = evidence["exp7470-independent-audit"]["payload"]
    arc = evidence["exp7471-arc-seam-observation"]["payload"]
    board = evidence["exp7473-board-continuity"]["payload"]
    local = e0.get("local_runtime_reduction") or {}
    extraction = audit.get("extraction_audit") or {}
    counts = extraction.get("counts") or {}
    update = audit.get("independent_update_replay") or {}
    replaceable = e6.get("replaceable_share_bounds") or {}
    stage = e6.get("stage_attribution") or {}
    arc_rows = arc.get("rows") if isinstance(arc.get("rows"), list) else []
    audit_rows = audit.get("rows") if isinstance(audit.get("rows"), list) else []
    dispositions = Counter(str(row.get("disposition") or "missing") for row in audit_rows)
    typed = _audit_branch(audit, "typed_decision")
    residual = _audit_branch(audit, "residual_learning")
    return {
        "probability_evidence": {
            "source_task": "exp7463-semif-e0-logprob-parity",
            "complete_pairs": local.get("complete"),
            "argmax_agreement": local.get("argmax_agreement"),
            "median_tv_distance": local.get("median_tv_distance"),
            "local_runtime_parity": e0.get("local_runtime_parity_score") == 1,
            "scored_runtime_parity": e0.get("scored_runtime_parity_score") == 1,
            "scored_runtime_evidence_available": e0.get("scored_runtime_evidence") is not None,
        },
        "typed_decision_utility": {
            "source_task": "exp7470-independent-audit",
            "availability": typed.get("availability", "missing"),
            "validity": typed.get("validity", "not_auditable"),
            "measured_benefit": typed.get("measured_benefit"),
            "positive_probability_gain": False,
            "typed_utility_established": typed.get("measured_benefit") == "measured_improvement",
        },
        "online_retention_and_benefit": {
            "source_task": "exp7470-independent-audit",
            "availability": residual.get("availability", "missing"),
            "updates_replayed": int(update.get("updates_replayed") or 0),
            "retention_predictions_checked": int(update.get("retention_predictions_checked") or 0),
            "measured_benefit": residual.get("measured_benefit"),
            "benefit_established": residual.get("measured_benefit") == "measured_improvement",
        },
        "extraction_coverage": {
            "source_task": "exp7470-independent-audit",
            "planned": int(counts.get("planned") or 0),
            "attempted": int(counts.get("attempted") or 0),
            "completed": int(counts.get("completed") or 0),
            "failed": int(counts.get("failed") or 0),
            "censored": int(counts.get("censored") or 0),
            "unstarted": int(counts.get("unstarted") or 0),
            "raw_disposition_counts": dict(sorted(dispositions.items())),
            "constructed_exact_pairs": extraction.get("constructed_exact_pairs"),
            "natural_annotation_uncertainty": extraction.get("natural_annotation_uncertainty"),
            "natural_text_value_established": False,
        },
        "decision_cost": {
            "source_task": "exp7464-semif-e6-decision-cost-profile",
            "episode_count": len(e6.get("episode_rows") or []),
            "trace_time_fraction": stage.get("trace_time_fraction"),
            "replaceable_share_lower": replaceable.get("lower"),
            "replaceable_share_upper": replaceable.get("upper"),
            "downstream_generation_directly_replaceable": replaceable.get(
                "downstream_generation_directly_replaceable"
            ),
            "efficacy_established": False,
        },
        "arc_self_discovery": {
            "source_task": "exp7471-arc-seam-observation",
            "current_live_episode_count": len(arc_rows),
            "progressed_episode_count": sum(row.get("progressed") is True for row in arc_rows),
            "hidden_game_efficacy_claim": arc.get("hidden_game_efficacy_claim") is True,
            "public_development_generalization_proxy": arc.get(
                "public_development_generalization_proxy"
            )
            is True,
            "new_level_credit": arc.get("new_level_credit"),
        },
        "board_continuity": {
            "source_task": "exp7473-board-continuity",
            "hardware_value_score": board.get("hardware_value_score"),
            "gatemate_changed_state_score": board.get("gatemate_changed_state_score"),
            "hardware_operation_count": board.get("hardware_operation_count"),
        },
        "external_text_verifier_moat": {
            "source_task": "ops/verifier_gaps.md",
            "retired_scope": "general external generated-text and logprob scorer construction",
            "source_conditioned_option_scores_are_equivalent": False,
            "reopened": False,
        },
    }


def classify_terminal(rows: Sequence[Mapping[str, Any]], *, validation_complete: bool) -> JsonDict:
    """Apply owned validity, invalid evidence, external absence, then benefit."""

    if not validation_complete:
        verdict_class = "partial"
        verdict = "partial_retryable_current_validation_incomplete"
    elif any(
        row.get("verdict_class") == "disqualified" or row.get("evidence_state") == "invalid"
        for row in rows
    ):
        verdict_class = "disqualified"
        verdict = "complete_disqualified_required_v654_science_with_fourteen_dispositions"
    elif any(
        row.get("verdict_class") == "blocked"
        or row.get("evidence_state") in {"missing", "pre_gate"}
        for row in rows
    ):
        verdict_class = "blocked"
        verdict = "complete_blocked_required_v654_science_with_fourteen_dispositions"
    elif any(row.get("verdict_class") in {"positive", "circular_positive"} for row in rows):
        verdict_class = "null"
        verdict = "complete_null_v654_capstone_no_independent_in_scope_benefit"
    else:
        verdict_class = "null"
        verdict = "complete_null_v654_capstone_with_fourteen_dispositions"
    return {"status": verdict, "honest_verdict": verdict, "verdict_class": verdict_class}


def _failure_rows(
    contract: Mapping[str, Any], evidence: Mapping[str, JsonDict], validation_complete: bool
) -> list[JsonDict]:
    """Name exact failures in classification precedence order."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "check": "contract_authority_equivalence",
                "category": "current_validity",
                "upstream": "v654-contract-authorities",
                "path": f"{DESIGN_PATH.as_posix()} + {contract.get('selected_roadmap_path')}",
                "field": "comparison_passed",
                "op": "==",
                "expected": True,
                "observed": contract.get("comparison_passed"),
            }
        )
    if not validation_complete:
        failures.append(
            {
                "check": "current_required_validation",
                "category": "current_validity",
                "upstream": EXPERIMENT_ID,
                "path": TEST_PATH.as_posix(),
                "field": "required_checks_passed",
                "op": "==",
                "expected": True,
                "observed": False,
            }
        )
    for task_id, row in evidence.items():
        if row["evidence_state"] == "invalid" or row["verdict_class"] == "disqualified":
            failures.append(
                {
                    "check": "required_evidence_validity",
                    "category": "scientific_validity",
                    "upstream": task_id,
                    "path": row["found_path"] or row["expected_path"],
                    "field": "valid",
                    "op": "==",
                    "expected": True,
                    "observed": row["valid"],
                }
            )
    for task_id, row in evidence.items():
        if row["evidence_state"] in {"missing", "pre_gate"} or row["verdict_class"] == "blocked":
            summary = row.get("gate_check_summary") or {}
            failures.append(
                {
                    "check": summary.get("check", "required_evidence_availability"),
                    "category": "scientific_availability",
                    "upstream": task_id,
                    "path": row["found_path"] or row["expected_path"],
                    "field": summary.get("field", "available"),
                    "op": summary.get("op", "=="),
                    "expected": summary.get("expected", True),
                    "observed": summary.get("observed", row["available"]),
                }
            )
    return failures


def retirement_rows(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Apply the four-field prior-failure rule without broad retirement."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        current = evidence.get(task_id, {})
        current_verdict = current.get("honest_verdict")
        for prior in task.get("prior_failures") or []:
            if not isinstance(prior, Mapping):
                continue
            previous = str(prior.get("verdict"))
            exact = current_verdict == previous
            valid_null = bool(
                current.get("valid") is True
                and current.get("verdict_class") == "null"
                and previous.startswith("complete_null")
            )
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "previous_verdict": previous,
                    "current_verdict": current_verdict,
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                    "same_exact_verdict": exact,
                    "same_valid_null_class": valid_null,
                    "decision": "retire" if exact and valid_null else "continue",
                    "retired_scope": task_id if exact and valid_null else None,
                }
            )
    return rows


def continuation_rows(
    evidence: Mapping[str, JsonDict], science: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Give each branch one decision and the exact condition for new work."""

    del evidence
    extraction = science["extraction_coverage"]
    return [
        {
            "branch": "typed_decision",
            "decision": "defer",
            "measured_cause": "The option protocol is disqualified, capture is pre-gated, and calibration is absent.",
            "changed_prerequisite": "Repair Exp7462 required validation, then capture authenticated source-option rows.",
            "scope": "source-conditioned typed decisions only",
        },
        {
            "branch": "probability_parity",
            "decision": "defer",
            "measured_cause": "Local runtime parity missed its gate and the exact scored runtime is unavailable.",
            "changed_prerequisite": "Provide an exact scored-runtime receipt and a new local parity result that meets the frozen bounds.",
            "scope": "E0 option-logprob runtime parity",
        },
        {
            "branch": "decision_cost",
            "decision": "continue",
            "measured_cause": "E6 measured a wide replaceable share with only partial stage attribution.",
            "changed_prerequisite": "Collect at least 30 attributed episodes across 10 games before an efficacy or savings claim.",
            "scope": "E6 cost attribution, not intervention efficacy",
        },
        {
            "branch": "compact_extraction",
            "decision": "retire",
            "measured_cause": f"The changed factual canary still left {extraction['unstarted']} of {extraction['planned']} evaluation cells unstarted.",
            "changed_prerequisite": "Use a different extraction mechanism or an independently labeled natural corpus.",
            "scope": "unchanged compact-span canary thresholds and bounded decoding only",
        },
        {
            "branch": "residual_learning",
            "decision": "defer",
            "measured_cause": "The analytic learner is circular-positive, while real continuous-learning evidence is absent.",
            "changed_prerequisite": "Supply chronological source-support errors and replayable delayed-update checkpoints.",
            "scope": "single residual energy head",
        },
        {
            "branch": "four_expert_mixture",
            "decision": "retire",
            "measured_cause": "V653 independently confirmed the repeated valid no-benefit result.",
            "changed_prerequisite": "Use a different expert set or update mechanism.",
            "scope": "unchanged four-expert mixture only",
        },
        {
            "branch": "external_text_verifier_moat",
            "decision": "retire",
            "measured_cause": "Source-conditioned option scores do not supply a general external-text selector result.",
            "changed_prerequisite": "Test an oracle-distinct mechanism outside the retired generated-text and logprob scorer class.",
            "scope": "general external generated-text and logprob scorer construction",
        },
        {
            "branch": "arc_self_discovery",
            "decision": "continue",
            "measured_cause": "Eight current live episodes recorded seams but no reproducible progress or hidden-game efficacy.",
            "changed_prerequisite": "Change the self-discovery mechanism or hidden-game evidence, not only the public proxy budget.",
            "scope": "live ARC discovery path",
        },
        {
            "branch": "prefix_service",
            "decision": "defer",
            "measured_cause": "The source capture prerequisite failed before prefix-service evidence could run.",
            "changed_prerequisite": "Produce valid shared-prefix source rows after the option protocol is repaired.",
            "scope": "exact-prefix service cost",
        },
        {
            "branch": "board_continuity",
            "decision": "defer",
            "measured_cause": "GateMate has no dated changed physical-state receipt after Exp6559.",
            "changed_prerequisite": "Provide an operator-authored cable, port, power, board, JTAG, or DirtyJTAG change receipt.",
            "scope": "GateMate physical retry only",
        },
    ]


def unresolved_obligations() -> list[JsonDict]:
    """Keep external and forbidden work visible without retrying this capstone."""

    return [
        {
            "obligation_id": "option_protocol_validation",
            "state": "disqualified",
            "path": "results/experiment_7462_v654_option_protocol.json",
            "required_change": "repair the failed required full-suite receipt without weakening tests",
        },
        {
            "obligation_id": "typed_decision_and_online_inputs",
            "state": "blocked_external_producer_evidence",
            "paths": [
                "results/experiment_7466_v654_typed_energy_calibration.json",
                "results/experiment_7469_v654_continuous_residual_learning.json",
                "results/experiment_7472_v654_prefix_service.json",
            ],
            "required_change": "produce exact authenticated artifacts after their changed prerequisites",
        },
        {
            "obligation_id": "gatemate_hardware_access",
            "state": "blocked_unchanged_physical_prerequisite",
            "required_change": "provide a dated operator-authored changed physical-state receipt",
        },
        {
            "obligation_id": "conductor_change",
            "state": "user_forbidden",
            "prohibited_path": "scripts/research_conductor.py",
            "current_task_action": "none",
        },
    ]


def publication_gate_row(value: Mapping[str, Any]) -> JsonDict:
    """Preserve G1-G4 while denying any V654 certification inference."""

    return {
        **deepcopy(dict(value)),
        "headline_scope": "FoVer dual-condition AUROC",
        "certifies_v654": False,
        "v654_science_scope": "not_evaluated_by_G1_G4",
    }


def _current_validation_complete(validation: Mapping[str, Any]) -> bool:
    """Require affected and terminal command sets exactly once."""

    return bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )


def _receipt_set_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require one successful receipt for every named current command."""

    if not isinstance(receipts, list):
        return False
    selected = [row for row in receipts if isinstance(row, Mapping)]
    counts = Counter(str(row.get("name")) for row in selected)
    by_name = {str(row.get("name")): row for row in selected}
    return all(
        counts[name] == 1
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


def _task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Build fourteen plain ordered rows without reading this result as input."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        source = evidence[str(task["id"])]
        rows.append(
            {
                "order": order,
                "task_id": str(task["id"]),
                "expected_path": source["expected_path"],
                "found_path": source["found_path"],
                "evidence_state": source["evidence_state"],
                "raw_experiment_id": source["raw_experiment_id"],
                "raw_milestone": source["raw_milestone"],
                "raw_status": source["raw_status"],
                "honest_verdict": source["honest_verdict"],
                "verdict_class": source["verdict_class"],
                "flagged_adversarial": source["flagged_adversarial"],
                "authenticated": source["authenticated"],
                "valid": source["valid"],
                "gate_values": source["gate_values"],
                "validation_receipts": source["validation_receipts"],
                "validation_receipts_authenticated": source["validation_receipts_authenticated"],
                "validation_authentication_failures": source["validation_authentication_failures"],
                "source_sha256": source["source_sha256"],
                "row_count": source["row_count"],
                "attempted": True,
                "completed": True,
                "failed": source["evidence_state"] == "invalid",
                "censored": source["verdict_class"] == "blocked",
                "unstarted": False,
            }
        )
    current_complete = _current_validation_complete(validation)
    rows.append(
        {
            "order": len(tasks),
            "task_id": str(tasks[-1]["id"]),
            "expected_path": str(tasks[-1]["deliverable"]),
            "found_path": None,
            "evidence_state": "current_work",
            "raw_experiment_id": EXPERIMENT_ID,
            "raw_milestone": MILESTONE,
            "raw_status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "flagged_adversarial": terminal["verdict_class"] == "disqualified",
            "authenticated": current_complete,
            "valid": current_complete,
            "gate_values": [],
            "validation_receipts": [
                {
                    key: deepcopy(row.get(key))
                    for key in (
                        "name",
                        "required",
                        "passed",
                        "exit_code",
                        "log_path",
                        "log_sha256",
                    )
                }
                for row in validation.get("validation_receipts", [])
                if isinstance(row, Mapping)
            ],
            "validation_receipts_authenticated": current_complete,
            "validation_authentication_failures": [],
            "source_sha256": None,
            "row_count": 1,
            "attempted": True,
            "completed": current_complete,
            "failed": not current_complete,
            "censored": False,
            "unstarted": False,
        }
    )
    return rows


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool,
    principle: str,
    op: str = "==",
) -> JsonDict:
    """Create one plain gate row with an auditable purpose."""

    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "operator": op,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    science: Mapping[str, JsonDict],
    dispositions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep completion, validity, benefit, and publication gates separate."""

    return [
        _gate(
            "contract_authorities_agree",
            "current_validity",
            True,
            contract.get("comparison_passed"),
            passed=contract.get("comparison_passed") is True,
            principle="Both current authorities must agree on all fourteen tasks.",
        ),
        _gate(
            "fourteen_dispositions",
            "completion",
            14,
            len(dispositions),
            passed=len(dispositions) == 14,
            principle="Every promised slot needs one disposition even when evidence is absent.",
        ),
        _gate(
            "required_present_evidence_valid",
            "scientific_validity",
            True,
            all(row["valid"] for row in evidence.values() if row["available"]),
            passed=all(row["valid"] for row in evidence.values() if row["available"]),
            principle="Present invalid evidence disqualifies instead of becoming a scientific null.",
        ),
        _gate(
            "typed_decision_utility",
            "benefit",
            True,
            science["typed_decision_utility"]["typed_utility_established"],
            passed=science["typed_decision_utility"]["typed_utility_established"],
            principle="Probability scores do not establish typed-decision utility.",
        ),
        _gate(
            "online_retention_benefit",
            "benefit",
            True,
            science["online_retention_and_benefit"]["benefit_established"],
            passed=science["online_retention_and_benefit"]["benefit_established"],
            principle="An analytic learner does not establish delayed online benefit.",
        ),
        _gate(
            "natural_extraction_value",
            "benefit",
            True,
            science["extraction_coverage"]["natural_text_value_established"],
            passed=science["extraction_coverage"]["natural_text_value_established"],
            principle="Constructed exact pairs do not establish natural extraction value.",
        ),
        _gate(
            "arc_hidden_game_efficacy",
            "benefit",
            True,
            science["arc_self_discovery"]["hidden_game_efficacy_claim"],
            passed=science["arc_self_discovery"]["hidden_game_efficacy_claim"],
            principle="Public development episodes remain separate from hidden-game efficacy.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            True,
            validation.get("required_checks_passed"),
            passed=validation.get("required_checks_passed") is True,
            principle="The frozen affected-file plan controls current code validity.",
        ),
        _gate(
            "terminal_validation",
            "required_validation",
            True,
            validation.get("terminal_validation_passed"),
            passed=validation.get("terminal_validation_passed") is True,
            principle="Cold replay and strict readers control atomic publication.",
        ),
        _gate(
            "promotion_zero",
            "promotion",
            0,
            0,
            passed=True,
            principle="A host aggregation cannot publish or change production state.",
        ),
    ]


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Bind required source files and every found predecessor artifact."""

    rows = [
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "evidence_class": "required_source",
        }
        for path in SOURCE_PATHS
    ]
    rows.extend(
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "evidence_class": "current_implementation",
        }
        for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH)
    )
    selected = str(contract["selected_roadmap_path"])
    if selected != ROADMAP_PATH.as_posix():
        rows.append(
            {
                "path": selected,
                "sha256": sha256_file(root / selected),
                "evidence_class": "selected_contract_authority",
            }
        )
    rows.extend(
        {
            "path": row["found_path"] or row["expected_path"],
            "sha256": row["source_sha256"],
            "evidence_class": row["evidence_state"],
            "task_id": task_id,
            "raw_experiment_id": row["raw_experiment_id"],
            "raw_milestone": row["raw_milestone"],
            "original_verdict_class": row["verdict_class"],
            "original_flagged_adversarial": row["flagged_adversarial"],
        }
        for task_id, row in evidence.items()
    )
    return rows


def _historical_sidecars(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep all producer activity outside the current invocation counters."""

    return [
        {
            "task_id": task_id,
            "path": row["found_path"],
            "sha256": row["source_sha256"],
            "scope": "historical_model_receipts",
            "counted_as_current_invocation": False,
            "original_model_invoked": row["payload"].get("model_invoked"),
            "original_inference_substrate_class": row["payload"].get("inference_substrate_class"),
        }
        for task_id, row in evidence.items()
        if row["found_path"] is not None
    ]


def _preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record exact source, ownership, host, contract, and evidence observations."""

    rows = [
        {
            "check": f"source_bytes:{path.as_posix()}",
            "upstream": path.as_posix(),
            "path": path.as_posix(),
            "field": "bytes",
            "expected": "readable_nonempty_bytes",
            "observed": "readable_nonempty_bytes"
            if (root / path).is_file() and (root / path).stat().st_size > 0
            else None,
            "passed": (root / path).is_file() and (root / path).stat().st_size > 0,
        }
        for path in SOURCE_PATHS
    ]
    rows.extend(
        [
            {
                "check": "execution_venue",
                "upstream": EXPERIMENT_ID,
                "path": "/proc/self",
                "field": "host_identity",
                "expected": "host",
                "observed": "host",
                "passed": True,
                "device_identity": {
                    "node": platform.node(),
                    "machine": platform.machine(),
                    "pid": os.getpid(),
                    "cuda_used": False,
                },
            },
            {
                "check": "contract_authority_equivalence",
                "upstream": "v654-contract-authorities",
                "path": f"{DESIGN_PATH.as_posix()} + {contract['selected_roadmap_path']}",
                "field": "comparison_passed",
                "expected": True,
                "observed": contract["comparison_passed"],
                "passed": contract["comparison_passed"] is True,
            },
        ]
    )
    rows.extend(
        {
            "check": "predecessor_disposition_observed",
            "upstream": task_id,
            "path": row["found_path"] or row["expected_path"],
            "field": "evidence_state",
            "expected": "one explicit state",
            "observed": row["evidence_state"],
            "passed": row["evidence_state"] in {"terminal", "pre_gate", "missing", "invalid"},
        }
        for task_id, row in evidence.items()
    )
    return rows


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain every field while ordinary values remain plain scalars."""

    return {
        key: FIELD_PRINCIPLES.get(key, f"Retain plain audited evidence for {key}.") for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable protocol, source, row, decision, and validation evidence."""

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
        "clock_identity",
    }
    return canonical_hash({key: item for key, item in value.items() if key not in excluded})


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all failures and the first exact classification cause."""

    return {
        "passed": not failures,
        "failure_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failures": deepcopy([dict(row) for row in failures]),
    }


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    publication: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one schema-complete artifact from current authenticated evidence."""

    required_complete = validation.get("required_checks_passed") is True
    terminal = classify_terminal(list(evidence.values()), validation_complete=required_complete)
    science = reduce_science(evidence)
    dispositions = _task_dispositions(contract["tasks"], evidence, terminal, validation)
    failures = _failure_rows(contract, evidence, required_complete)
    validation_s = sum(
        float(row.get("duration_s") or 0.0)
        for row in validation.get("validation_receipts", [])
        if isinstance(row, Mapping)
    )
    complete = bool(
        len(dispositions) == len(EXPECTED_TASK_IDS)
        and contract.get("comparison_passed") is True
        and _current_validation_complete(validation)
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
        "flagged_adversarial": terminal["verdict_class"] == "disqualified"
        or any(row["flagged_adversarial"] for row in evidence.values()),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "clock_identity": {
            "utc": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
            "host": platform.node(),
        },
        "duration_s": float(duration_s),
        "duration_components_s": {
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "numeric_fitting": 0.0,
            "aggregation": max(0.0, float(duration_s) - validation_s),
            "validation": validation_s,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "inference_substrate_details": {
            "current_llm_calls": 0,
            "current_cuda_calls": 0,
            "current_board_calls": 0,
            "historical_events_counted_as_current": False,
        },
        "execution_venue": "host",
        "random_seed": {
            "fitting": None,
            "ordering": 6_547_474,
            "audit": 6_547_475,
            "bootstrap": 6_547_476,
            "reason": "The reducer performs no fit; fixed seeds bind ordering and future replay.",
        },
        "small_ebm_training": {
            "attempted": 0,
            "completed": 0,
            "performed": False,
            "duration_s": 0.0,
        },
        "preconditions_checked": _preconditions(root, contract, evidence),
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_evidence_sidecars": _historical_sidecars(evidence),
        "rows": deepcopy(dispositions),
        "sample_size_budget": {
            "planned": 14,
            "attempted": 14,
            "complete": 14,
            "failed": sum(row["failed"] for row in dispositions),
            "censored": sum(row["censored"] for row in dispositions),
            "unstarted": sum(row["unstarted"] for row in dispositions),
            "independent_units": 14,
            "stopping_rule": "Reduce each ordered V654 slot once; do not retry terminal external absence.",
        },
        "acceptance_gate_results": _acceptance_gates(
            contract, evidence, science, dispositions, validation
        ),
        "gate_check_summary": _gate_summary(failures),
        "verifier_is_oracle": False,
        "validation_receipts": deepcopy(validation.get("validation_receipts", [])),
        "repository_health": deepcopy(validation.get("repository_health", {})),
        "task_dispositions": dispositions,
        "science_reductions": science,
        "retirement_rows": retirement_rows(contract["tasks"], evidence),
        "continuation_rows": continuation_rows(evidence, science),
        "publication_gates": publication_gate_row(publication),
        "unresolved_obligations": unresolved_obligations(),
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "declared_entrypoint_cold_replay",
            "independent_raw_reduction": "independent_cold_reducer",
            "numbered_runtime_e2e": [],
        },
        "roadmap_activated": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_contact_performed": False,
        "publication_performed": False,
        "submission_performed": False,
        "push_performed": False,
        "promotion_score": 0,
        "capstone_complete_score": int(complete),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _source_hashes_match(value: Mapping[str, Any], root: Path) -> bool:
    """Recheck every source entry that claims an existing byte hash."""

    rows = value.get("source_artifact_hashes")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        digest = row.get("sha256")
        if digest is None:
            continue
        path = Path(str(row.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != digest:
            return False
    return True


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, sources, reductions, decisions, and checksum."""

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
        "task_dispositions",
        "science_reductions",
        "continuation_rows",
        "publication_gates",
        "capstone_complete_score",
        "unresolved_obligations",
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
    if not str(artifact.get("status")).startswith(("complete_", "partial_")):
        errors.append("lifecycle_invalid")
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
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_invalid")
    dispositions = artifact.get("task_dispositions")
    if (
        not isinstance(dispositions, list)
        or [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        != list(EXPECTED_TASK_IDS)
        or artifact.get("rows") != dispositions
    ):
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
            list(evidence.values()), validation_complete=validation["required_checks_passed"]
        )
        expected_rows = _task_dispositions(contract["tasks"], evidence, terminal, validation)
        if dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        science = reduce_science(evidence)
        if artifact.get("science_reductions") != science:
            errors.append("science_reductions_invalid")
        if artifact.get("continuation_rows") != continuation_rows(evidence, science):
            errors.append("continuation_rows_invalid")
        if artifact.get("retirement_rows") != retirement_rows(contract["tasks"], evidence):
            errors.append("retirement_rows_invalid")
        if artifact.get("unresolved_obligations") != unresolved_obligations():
            errors.append("unresolved_obligations_invalid")
        failures = _failure_rows(contract, evidence, validation["required_checks_passed"])
        if (
            artifact.get("status") != terminal["status"]
            or artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != _gate_summary(failures)
        ):
            errors.append("terminal_reduction_invalid")
        expected_gates = _acceptance_gates(contract, evidence, science, expected_rows, validation)
        if artifact.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gates_invalid")
        if artifact.get("publication_gates") != publication_gate_row(evaluate_publication_gates()):
            errors.append("publication_gates_invalid")
        expected_complete = int(
            len(expected_rows) == len(EXPECTED_TASK_IDS)
            and contract["comparison_passed"] is True
            and _current_validation_complete(validation)
        )
        if artifact.get("capstone_complete_score") != expected_complete:
            errors.append("capstone_score_invalid")
        expected_flag = terminal["verdict_class"] == "disqualified" or any(
            row["flagged_adversarial"] for row in evidence.values()
        )
        if artifact.get("flagged_adversarial") is not expected_flag:
            errors.append("flagged_adversarial_invalid")
        if require_terminal and validation["terminal_validation_passed"] is not True:
            errors.append("terminal_validation_incomplete")
    except (KeyError, OSError, TypeError, ValueError):
        errors.append("independent_reduction_failed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact) - {
        "field_principles"
    }:
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def independent_reduce(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Replay all pure reductions without requiring terminal receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def _passing_receipts() -> list[JsonDict]:
    """Create deterministic unit-test receipts for both command sets."""

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
    """Return zero-duration phase boundaries for pure unit construction."""

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
            ("preconditions", 13, "thirteen_predecessor_slots_observed"),
            ("plan", 8, "affected_validation_manifest_frozen"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("validation", 8, "affected_checks_complete"),
            ("reduction", 14, "fourteen_dispositions_reduced"),
            ("terminal_validation", 5, "cold_readers_complete"),
            ("write", 1, "terminal_artifact_ready"),
        )
    ]


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic artifact with passing current validation."""

    contract = load_contract(REPO_ROOT)
    evidence = collect_evidence(REPO_ROOT, contract["tasks"])
    receipts = _passing_receipts()
    validation = {
        "required_checks_passed": True,
        "terminal_validation_passed": True,
        "validation_receipts": receipts,
        "repository_health": {"status": "outside_current_affected_validity"},
    }
    return build_artifact(
        REPO_ROOT,
        contract,
        evidence,
        validation,
        evaluate_publication_gates(),
        started_at_utc="2026-09-21T00:00:00+00:00",
        completed_at_utc="2026-09-21T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        duration_s=0.0,
        phase_spans=_test_spans(),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7358 plan for only the current affected files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V654 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def _utc_now() -> str:  # pragma: no cover - real process clock boundary.
    """Return a measured aware UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush one phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7474] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - real process clock boundary.
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


def _terminal_commands(  # pragma: no cover - subprocess boundary.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, independent reduction, strict readers, and G1-G4."""

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
        validation_scope.CommandSpec(
            "publication_gate_json",
            (python, "-u", PUBLICATION_GATE_PATH.as_posix(), "--json"),
            "unchanged_publication_g1_g4",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(  # pragma: no cover - exercised through the declared entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run exact reads, scoped checks, cold readers, and atomic publication."""

    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7474-", dir="/tmp"))

    phase_started = time.monotonic()
    _progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    publication = evaluate_publication_gates()
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            completed_units=len(evidence),
            checkpoint="thirteen_predecessor_slots_observed",
        )
    )
    _progress(started, "preconditions", "after", completed_units=len(evidence))

    phase_started = time.monotonic()
    _progress(started, "plan", "before")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    spans.append(
        _span(
            "plan",
            phase_started,
            started,
            completed_units=len(commands),
            checkpoint="affected_validation_manifest_frozen",
        )
    )
    _progress(started, "plan", "after", completed_units=len(commands))

    for phase, checkpoint in (
        ("model_load", "no_current_model_load"),
        ("generation", "no_current_generation"),
    ):
        phase_started = time.monotonic()
        _progress(started, phase, "before", completed_units=0)
        spans.append(
            _span(
                phase,
                phase_started,
                started,
                completed_units=0,
                checkpoint=checkpoint,
            )
        )
        _progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    _progress(started, "validation", "before_subprocesses", completed_units=0)
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
        "repository_health": {
            "status": "outside_current_affected_validity",
            "historical_failures": [],
        },
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
    _progress(
        started,
        "validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    phase_started = time.monotonic()
    _progress(started, "reduction", "before", completed_units=0)
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=[
            *spans,
            _span(
                "reduction",
                phase_started,
                started,
                completed_units=14,
                checkpoint="fourteen_dispositions_reduced",
            ),
        ],
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    _progress(started, "reduction", "after", completed_units=14)

    phase_started = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate_path)
    _progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal_receipts = run_categorized_commands(
        root,
        terminal_commands,
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
            checkpoint="cold_readers_complete",
        )
    )
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=validation["terminal_validation_passed"],
        critical=critical,
    )

    phase_started = time.monotonic()
    _progress(started, "write", "before_atomic", path=output_path.as_posix())
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
        publication,
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
    _progress(started, "write", "after_atomic", path=output_path.as_posix())
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
    """Run the capstone or one read-only cold replay."""

    print("[exp7474] phase=startup event=flushed", flush=True)
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


if __name__ == "__main__":  # pragma: no cover - CLI boundary.
    raise SystemExit(main())
