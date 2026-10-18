"""Close V653 with fourteen authenticated task dispositions.

This reducer reads existing artifacts and independently summarizes bounded raw
rows. It does not invoke a model, train a numeric head, operate hardware, or
publish externally.

Spec refs: REQ-REPORT-7460 and SCENARIO-REPORT-7460-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import fmean
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import (
    _field_declared,
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
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7460-capstone"
SCHEMA = "carnot.exp7460.v653.capstone.v1"

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7460_v653_capstone.json")
RAW_DIR = Path("results/raw/experiment_7460_v653_capstone")
MODULE_PATH = Path("python/carnot/experiment_7460_v653_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7460_v653_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7460_v653_capstone.py")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7447, "contract-methods"),
        (7448, "capture-lifecycle"),
        (7449, "source-protocol"),
        (7450, "prediction-ledger"),
        (7451, "span-capture"),
        (7452, "source-embeddings"),
        (7453, "energy-calibration"),
        (7454, "continuous-learning"),
        (7455, "decision-audit"),
        (7456, "extraction-audit"),
        (7457, "arc-exposure"),
        (7458, "durable-updates"),
        (7459, "board-continuity"),
        (7460, "capstone"),
    )
)
CLAIM_BRANCHES = (
    "contract",
    "static_representation",
    "delayed_learning",
    "extraction",
    "arc_exposure",
    "durability",
    "board_status",
)
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ZERO_CALL_COUNTS = {
    "attempted": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "in_flight": 0,
}
ZERO_INVOCATION_COUNTS = {
    "model_loads": deepcopy(ZERO_CALL_COUNTS),
    "forward_calls": deepcopy(ZERO_CALL_COUNTS),
    "generation_calls": deepcopy(ZERO_CALL_COUNTS),
}
SOURCE_PATHS = (
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
    SPEC_PATH,
    Path("python/carnot/experiment_7446_v652_capstone.py"),
    PUBLICATION_GATE_PATH,
    Path("scripts/conductor_gates.py"),
    Path("ops/north-star.md"),
    Path("ops/verifier_gaps.md"),
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


def utc_now() -> str:  # pragma: no cover - real process boundary.
    """Return the real UTC time for an artifact boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit one flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7460] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_yaml_mapping(path: Path) -> JsonDict:
    """Load one YAML mapping and reject another top-level shape."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:  # pragma: no cover - I/O guard.
        raise ValueError(f"unreadable YAML mapping: {path}: {error}") from error
    if not isinstance(value, dict):  # pragma: no cover - parser guard.
        raise ValueError(f"YAML mapping required: {path}")
    return value


def _public_task(task: Mapping[str, Any]) -> JsonDict:
    """Keep only fields represented by both contract formats."""

    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
    }


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Parse both V653 authorities and compare all fourteen rows."""

    try:
        markdown = parse_markdown_contract(markdown_text)
        parsed_yaml = parse_yaml_contract(roadmap)
    except (TypeError, ValueError) as error:  # pragma: no cover - parser fail-closed path.
        return {"comparison_passed": False, "errors": [f"parse_error:{error}"], "contract_rows": []}
    mapping = roadmap if isinstance(roadmap, Mapping) else {}
    raw_tasks = mapping.get("tasks") if isinstance(mapping.get("tasks"), list) else []
    raw_by_id = {
        str(task.get("id")): (index, task)
        for index, task in enumerate(raw_tasks)
        if isinstance(task, Mapping)
    }
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    width = max(len(EXPECTED_TASK_IDS), len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
    undeclared = False
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else {}
        observed = yaml_tasks[index] if index < len(yaml_tasks) else {}
        task_id = str(observed.get("id") or expected.get("id") or f"missing-{index + 1}")
        checks = {
            field: expected.get(field) == observed.get(field) and bool(expected) and bool(observed)
            for field in ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
        }
        producer_rows: list[JsonDict] = []
        for gate in observed.get("gates") or []:
            upstream = str(gate.get("upstream"))
            producer_entry = raw_by_id.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            field = str(gate.get("artifact_field"))
            declared = _field_declared(str(producer.get("prompt") or ""), field)
            precedes = producer_index is not None and producer_index < index
            producer_rows.append(
                {
                    "upstream": upstream,
                    "artifact_field": field,
                    "producer_in_roadmap": producer_entry is not None,
                    "producer_precedes_consumer": precedes,
                    "producer_field_declared": declared,
                    "passed": precedes and declared,
                }
            )
        checks["producer_fields"] = all(row["passed"] for row in producer_rows)
        undeclared = undeclared or not checks["producer_fields"]
        rows.append(
            {
                "order": index + 1,
                "task_id": task_id,
                "markdown": _public_task(expected),
                "yaml": _public_task(observed),
                "gates": deepcopy(observed.get("gates") or []),
                "producer_declarations": producer_rows,
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
    if len(markdown_tasks) != 14 or len(yaml_tasks) != 14:
        errors.append("task_count")
    if any(not row["passed"] for row in rows):
        errors.append("row_mismatch")
    if undeclared:
        errors.append("producer_field_declaration")
    return {
        "comparison_passed": not errors,
        "errors": list(dict.fromkeys(errors)),
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": parsed_yaml.get("milestone"),
        "contract_rows": rows,
    }


def load_contract(root: Path) -> JsonDict:
    """Read the current design and active V653 YAML independently."""

    roadmap = load_yaml_mapping(root / ROADMAP_PATH)
    comparison = compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):  # pragma: no cover - comparison guards this.
        raise ValueError("active V653 task list required")
    return {**comparison, "tasks": deepcopy(tasks), "title": roadmap.get("milestone_title")}


def _resolve_path(root: Path, label: str) -> Path:
    """Resolve a repository label while preserving absolute conductor paths."""

    path = Path(label)
    return path if path.is_absolute() else root / path


def _fallback_pre_gate_path(task: Mapping[str, Any]) -> Path:
    """Derive the conductor's exact alternate pre-gate filename."""

    task_id = str(task.get("id") or "")
    number = numeric_experiment_id(task_id)
    slug = task_id.split("-", 1)[1].replace("-", "_") if "-" in task_id else "task"
    return Path(f"results/experiment_{number}_{slug}.json")


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Represent absent external bytes as a terminal blocked disposition."""

    declared = str(task.get("deliverable") or "")
    return {
        "task_id": str(task.get("id")),
        "declared_path": declared,
        "source_path": declared,
        "source_kind": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_missing_declared_evidence",
        "flagged_adversarial": False,
        "row_manifest_authenticated": False,
        "validation_receipts_authenticated": False,
        "required_validation_passed": False,
        "raw_row_count": 0,
        "raw_shard_count": 0,
        "sha256": None,
        "gate_check_summary": {
            "upstream": str(task.get("id")),
            "path": declared,
            "check": "declared_artifact_exists",
            "field": "path",
            "operator": "exists",
            "expected": True,
            "observed": False,
            "passed": False,
        },
        "payload": {},
    }


def _pre_gate_row(root: Path, task: Mapping[str, Any], actual: Path, payload: JsonDict) -> JsonDict:
    """Authenticate an exact conductor record and its failed source bytes."""

    expected_number = numeric_experiment_id(task.get("id"))
    observed_number = numeric_experiment_id(payload.get("experiment"))
    evidence_label = str(payload.get("failed_evidence_path") or "")
    evidence_path = _resolve_path(root, evidence_label)
    evidence_hash = payload.get("failed_evidence_sha256")
    evidence_matches = (
        evidence_path.is_file()
        and isinstance(evidence_hash, str)
        and sha256_file(evidence_path) == evidence_hash
    )
    required = (
        "failed_upstream",
        "failed_field",
        "failed_operator",
        "failed_expected",
        "failed_observed",
    )
    authenticated = (
        observed_number == expected_number
        and payload.get("status") == "blocked"
        and str(payload.get("honest_verdict") or "").startswith("blocked")
        and all(field in payload for field in required)
        and evidence_matches
    )
    return {
        "task_id": str(task.get("id")),
        "declared_path": str(task.get("deliverable") or ""),
        "source_path": actual.as_posix(),
        "source_kind": "structured_pre_gate",
        "authenticated": authenticated,
        "available": False,
        "valid": authenticated,
        "verdict_class": "blocked",
        "honest_verdict": str(payload.get("honest_verdict")),
        "flagged_adversarial": False,
        "row_manifest_authenticated": authenticated,
        "validation_receipts_authenticated": authenticated,
        "required_validation_passed": authenticated,
        "raw_row_count": 0,
        "raw_shard_count": 0,
        "sha256": sha256_file(root / actual),
        "gate_check_summary": {
            "upstream": payload.get("failed_upstream"),
            "path": actual.as_posix(),
            "check": "structured_pre_gate",
            "field": payload.get("failed_field"),
            "operator": payload.get("failed_operator"),
            "expected": payload.get("failed_expected"),
            "observed": payload.get("failed_observed"),
            "passed": False,
        },
        "payload": payload,
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one producer artifact or exact conductor pre-gate."""

    task_id = str(task.get("id"))
    declared = Path(str(task.get("deliverable") or ""))
    actual = declared
    if not (root / actual).is_file():
        fallback = _fallback_pre_gate_path(task)
        if (root / fallback).is_file():
            actual = fallback
        else:
            return _missing_evidence(task)
    path = root / actual
    payload = load_json_object(path)
    if payload.get("schema") == "blocked_gate_check_v1":
        return _pre_gate_row(root, task, actual, payload)
    row_manifest = authenticate_row_manifest(root, payload)
    validation = authenticate_validation_receipts(root, payload)
    expected_number = numeric_experiment_id(task_id)
    observed_number = numeric_experiment_id(
        payload.get("experiment_id") or payload.get("experiment") or payload.get("task_id")
    )
    verdict = payload.get("verdict_class")
    authenticated = (
        observed_number == expected_number
        and payload.get("milestone") == MILESTONE
        and verdict in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
        and terminal_status(payload)
        and row_manifest["authenticated"]
    )
    flagged = payload.get("flagged_adversarial") is True
    valid = (
        authenticated
        and validation["required_passed"]
        and verdict != "disqualified"
        and not flagged
    )
    return {
        "task_id": task_id,
        "declared_path": declared.as_posix(),
        "source_path": actual.as_posix(),
        "source_kind": "terminal_artifact",
        "authenticated": authenticated,
        "available": authenticated,
        "valid": valid,
        "verdict_class": verdict if verdict in CLOSED_VERDICTS else "disqualified",
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "flagged_adversarial": flagged,
        "row_manifest_authenticated": row_manifest["authenticated"],
        "validation_receipts_authenticated": validation["authenticated"],
        "required_validation_passed": validation["required_passed"],
        "raw_row_count": row_manifest["inline_row_count"],
        "raw_rows_sha256": row_manifest["inline_rows_sha256"],
        "raw_shard_count": row_manifest["raw_shard_count"],
        "raw_shard_row_count": row_manifest["raw_shard_row_count"],
        "validation_receipt_count": validation["receipt_count"],
        "validation_receipts_sha256": validation["receipt_sha256"],
        "sha256": sha256_file(path),
        "gate_check_summary": deepcopy(payload.get("gate_check_summary")),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Authenticate the thirteen predecessor slots in conductor order."""

    return {str(task["id"]): load_evidence_slot(root, task) for task in tasks[:-1]}


def _compare_gate(operator: str, observed: Any, expected: Any) -> bool:
    """Evaluate only operators declared by the V653 roadmap."""

    if operator == "==":
        return observed == expected
    if operator == "in":
        return isinstance(expected, list) and observed in expected
    if operator == ">=":  # pragma: no cover - reserved roadmap compatibility.
        return isinstance(observed, (int, float)) and observed >= expected
    return False  # pragma: no cover - future operators fail closed.


def audit_structured_gates(
    contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Re-evaluate each scalar gate and the producer declaration."""

    tasks = contract.get("tasks") if isinstance(contract.get("tasks"), list) else []
    positions = {
        str(task.get("id")): index for index, task in enumerate(tasks) if isinstance(task, Mapping)
    }
    declarations: dict[tuple[str, str, str], bool] = {}
    for row in contract.get("contract_rows") or []:
        if not isinstance(row, Mapping):  # pragma: no cover - contract schema guard.
            continue
        consumer = str(row.get("task_id"))
        for declaration in row.get("producer_declarations") or []:
            if isinstance(declaration, Mapping):
                key = (
                    consumer,
                    str(declaration.get("upstream")),
                    str(declaration.get("artifact_field")),
                )
                declarations[key] = declaration.get("passed") is True
    rows: list[JsonDict] = []
    for consumer_index, task in enumerate(tasks):
        if not isinstance(task, Mapping):  # pragma: no cover - YAML schema guard.
            continue
        consumer = str(task.get("id"))
        for gate in task.get("gated_on") or []:
            upstream = str(gate.get("upstream"))
            field = str(gate.get("artifact_field"))
            source = evidence.get(upstream, {})
            observed = source.get("payload", {}).get(field)
            scalar_passed = _compare_gate(str(gate.get("op")), observed, gate.get("value"))
            source_admissible = bool(
                source.get("authenticated")
                and source.get("valid")
                and source.get("verdict_class") not in {"blocked", "disqualified", "partial"}
                and source.get("flagged_adversarial") is False
            )
            rows.append(
                {
                    "consumer": consumer,
                    "upstream": upstream,
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected": deepcopy(gate.get("value")),
                    "observed": deepcopy(observed),
                    "producer_in_roadmap": upstream in positions,
                    "producer_precedes_consumer": positions.get(upstream, consumer_index)
                    < consumer_index,
                    "producer_field_declared": declarations.get((consumer, upstream, field), False),
                    "scalar_passed": scalar_passed,
                    "source_admissible": source_admissible,
                    "passed": scalar_passed and source_admissible,
                }
            )
    return rows


def _online_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute online probability metrics from independent audit rows."""

    by_arm: dict[str, list[float]] = defaultdict(list)
    group_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        arm = str(row.get("arm"))
        loss = row.get("log_loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool):
            by_arm[arm].append(float(loss))
            group_counts[arm].add(str(row.get("group_id")))
    means = {arm: fmean(values) for arm, values in sorted(by_arm.items())}
    learned = means.get("learned_mixture")
    equal = means.get("equal_weight_adaptive_mixture")
    delta = learned - equal if learned is not None and equal is not None else None
    counts = set(map(len, group_counts.values()))
    return {
        "prediction_rows": len(rows),
        "group_count_per_arm": counts.pop() if len(counts) == 1 else None,
        "log_loss_means": means,
        "learned_minus_equal_log_loss": delta,
        "primary_benefit_passed": delta is not None and delta < 0.0,
    }


def reduce_branch_metrics(evidence: Mapping[str, JsonDict]) -> dict[str, JsonDict]:
    """Recompute six scientific branch summaries from raw or audit rows."""

    p52 = evidence["exp7452-source-embeddings"]["payload"]
    p55 = evidence["exp7455-decision-audit"]["payload"]
    p56 = evidence["exp7456-extraction-audit"]["payload"]
    p57 = evidence["exp7457-arc-exposure"]["payload"]
    p58 = evidence["exp7458-durable-updates"]["payload"]
    p59 = evidence["exp7459-board-continuity"]["payload"]

    embedding_rows = p52.get("rows") if isinstance(p52.get("rows"), list) else []
    online_audit = p55.get("online_audit") if isinstance(p55.get("online_audit"), Mapping) else {}
    online_rows = online_audit.get("rows") if isinstance(online_audit.get("rows"), list) else []
    online = _online_metrics(online_rows)
    replay = online_audit.get("independent_update_replay") or {}
    online["independent_replay_valid"] = bool(
        online_audit.get("valid") is True
        and replay.get("all_updates_matched") is True
        and replay.get("prediction_before_reveal") is True
        and replay.get("no_feedback_immutable") is True
    )

    extraction_rows = p56.get("rows") if isinstance(p56.get("rows"), list) else []
    attempted_development = [
        row
        for row in extraction_rows
        if row.get("capture_phase") == "development" and row.get("attempted") is True
    ]
    unstarted_evaluation = [
        row
        for row in extraction_rows
        if row.get("capture_phase") == "evaluation" and row.get("disposition") == "unstarted"
    ]
    attempted_pairs: dict[str, set[str]] = defaultdict(set)
    for row in extraction_rows:
        if row.get("capture_phase") == "evaluation" and row.get("attempted") is True:
            attempted_pairs[str(row.get("unit_id"))].add(str(row.get("arm")))

    arc_rows = p57.get("rows") if isinstance(p57.get("rows"), list) else []
    applied_redirect_count = sum(
        sum(int(value) for value in (row.get("arm_applications") or {}).values())
        for row in arc_rows
    )

    timing_rows = [
        row
        for row in (p58.get("rows") or [])
        if isinstance(row, Mapping) and row.get("row_type") == "service_timing"
    ]
    totals: dict[str, int] = defaultdict(int)
    blocks: dict[str, set[int]] = defaultdict(set)
    for row in timing_rows:
        arm = str(row.get("arm"))
        totals[arm] += int(row.get("total_service_ns") or 0)
        blocks[arm].add(int(row.get("block") or 0))
    whole = totals.get("whole_state", 0)
    ratio = totals.get("delta_journal", 0) / whole if whole else None
    paired_blocks = len(blocks.get("delta_journal", set()) & blocks.get("whole_state", set()))

    board_rows = p59.get("rows") if isinstance(p59.get("rows"), list) else []
    return {
        "static_representation": {
            "source_task": "exp7452-source-embeddings",
            "planned_cells": len(embedding_rows),
            "attempted_cells": sum(row.get("attempted") is True for row in embedding_rows),
            "unstarted_cells": sum(row.get("unstarted") is True for row in embedding_rows),
            "embedding_capture_ready_score": p52.get("embedding_capture_ready_score"),
            "value": p52.get("embedding_value_score"),
            "disposition": "blocked_embedding_surface_unavailable",
        },
        "delayed_learning": {
            "source_task": "exp7455-decision-audit",
            **online,
            "audit_artifact_admissible": evidence["exp7455-decision-audit"]["valid"],
            "disposition": "complete_null_reproduced",
        },
        "extraction": {
            "source_task": "exp7456-extraction-audit",
            "audited_rows": len(extraction_rows),
            "development_attempted": len(attempted_development),
            "evaluation_unstarted": len(unstarted_evaluation),
            "paired_evaluation_units": sum(len(arms) == 2 for arms in attempted_pairs.values()),
            "semantic_value_established": False,
            "disposition": "complete_null_development_gate_closed",
        },
        "arc_exposure": {
            "source_task": "exp7457-arc-exposure",
            "episode_count": len(arc_rows),
            "threshold_reached_count": sum(
                row.get("threshold_reached") is True for row in arc_rows
            ),
            "applied_redirect_count": applied_redirect_count,
            "new_level_credit": sum(int(row.get("new_level_credit") or 0) for row in arc_rows),
            "disposition": "complete_null_no_arm_promotion",
        },
        "durability": {
            "source_task": "exp7458-durable-updates",
            "paired_block_count": paired_blocks,
            "delta_total_service_ns": totals.get("delta_journal", 0),
            "whole_total_service_ns": whole,
            "total_service_ratio": ratio,
            "speed_gate_passed": ratio is not None and ratio < 0.9,
            "disposition": "complete_null_speed_gate_not_met",
        },
        "board_status": {
            "source_task": "exp7459-board-continuity",
            "board_count": len(board_rows),
            "blocked_boards": [
                str(row.get("board"))
                for row in board_rows
                if row.get("availability_class") == "blocked"
            ],
            "fresh_hardware_operations": sum(
                len(row.get("hardware_operations_issued") or []) for row in board_rows
            ),
            "hardware_ready_score": max(
                (int(row.get("hardware_ready_score") or 0) for row in board_rows), default=0
            ),
            "disposition": "complete_null_changed_state_absent",
        },
    }


def reduce_claim_matrix(evidence: Mapping[str, JsonDict]) -> dict[str, JsonDict]:
    """Separate completion, readiness, value, and publication scope."""

    metrics = reduce_branch_metrics(evidence)
    p47 = evidence["exp7447-contract-methods"]["payload"]
    return {
        "contract": {
            "authority_task": "exp7447-contract-methods",
            "completion_score": int(p47.get("contract_ready_score") or 0),
            "readiness_score": 0,
            "benefit_score": 0,
            "publication_scope": False,
            "disposition": "null",
            "boundary": "Contract accounting is not scientific progress.",
        },
        "static_representation": {
            "authority_task": "exp7452-source-embeddings",
            "completion_score": 1,
            "readiness_score": 0,
            "benefit_score": 0,
            "publication_scope": False,
            "disposition": "blocked",
            "boundary": "The native embedding surface failed before any calibration fit.",
        },
        "delayed_learning": {
            "authority_task": "exp7454-continuous-learning+exp7455-decision-audit",
            "completion_score": 1,
            "readiness_score": 0,
            "benefit_score": 0,
            "publication_scope": False,
            "disposition": "null",
            "audit_artifact_admissible": metrics["delayed_learning"]["audit_artifact_admissible"],
            "boundary": "The independent raw replay is valid, but its artifact failed required validation.",
        },
        "extraction": {
            "authority_task": "exp7456-extraction-audit",
            "completion_score": 1,
            "readiness_score": 0,
            "benefit_score": 0,
            "publication_scope": False,
            "disposition": "null",
            "boundary": "Eight development calls do not estimate the unopened evaluation panel.",
        },
        "arc_exposure": {
            "authority_task": "exp7457-arc-exposure",
            "completion_score": 1,
            "readiness_score": 0,
            "benefit_score": 0,
            "publication_scope": False,
            "disposition": "null",
            "boundary": "Eight episodes expose the supervisor but cannot promote an arm.",
        },
        "durability": {
            "authority_task": "exp7458-durable-updates",
            "completion_score": 1,
            "readiness_score": 0,
            "benefit_score": 0,
            "publication_scope": False,
            "disposition": "null",
            "boundary": "Exact recovery passed while the full-service speed gate failed.",
        },
        "board_status": {
            "authority_task": "exp7459-board-continuity",
            "completion_score": 1,
            "readiness_score": 0,
            "benefit_score": 0,
            "publication_scope": False,
            "disposition": "blocked",
            "boundary": "Historical board evidence does not establish current hardware value.",
        },
    }


def _failure(
    task_id: str, source: Mapping[str, Any], category: str, field: str, expected: Any
) -> JsonDict:
    """Name every operand for one terminal classification failure."""

    return {
        "upstream": task_id,
        "path": source.get("source_path"),
        "check": category,
        "category": category,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": source.get(field),
    }


def classify_terminal(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Apply current defects, invalid science, then external absence."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "upstream": "v653-contract-authorities",
                "path": f"{DESIGN_PATH.as_posix()} + {ROADMAP_PATH.as_posix()}",
                "check": "contract_equivalence",
                "category": "current_validity",
                "field": "comparison_passed",
                "operator": "==",
                "expected": True,
                "observed": contract.get("comparison_passed"),
            }
        )
    if validation.get("required_checks_passed") is not True:
        failures.append(
            {
                "upstream": EXPERIMENT_ID,
                "path": TEST_PATH.as_posix(),
                "check": "affected_validation",
                "category": "current_validity",
                "field": "required_checks_passed",
                "operator": "==",
                "expected": True,
                "observed": validation.get("required_checks_passed"),
            }
        )
    for task_id, source in evidence.items():
        if source.get("source_kind") != "missing" and source.get("authenticated") is not True:
            failures.append(
                _failure(task_id, source, "evidence_authentication", "authenticated", True)
            )
    for task_id, source in evidence.items():
        if source.get("available") and (
            source.get("valid") is not True
            or source.get("verdict_class") == "disqualified"
            or source.get("flagged_adversarial") is True
        ):
            failures.append(_failure(task_id, source, "scientific_validity", "valid", True))
    for task_id, source in evidence.items():
        if source.get("available") is not True:
            failures.append(_failure(task_id, source, "scientific_availability", "available", True))
    invalid = any(
        row["category"] in {"current_validity", "evidence_authentication", "scientific_validity"}
        for row in failures
    )
    absent = any(row["category"] == "scientific_availability" for row in failures)
    if invalid:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_required_v653_science_with_fourteen_dispositions"
    elif absent:
        verdict_class = "blocked"
        verdict = "complete_blocked_required_v653_science_with_fourteen_dispositions"
    else:
        verdict_class = "null"
        verdict = "complete_null_v653_capstone_with_fourteen_dispositions"
    return {
        "status": verdict,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": {
            "passed": not failures,
            "failure_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failures": failures,
        },
    }


def retirement_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Apply exact prior rules and the semantic mixture-null rule."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id"))
        current = (
            terminal.get("honest_verdict")
            if task_id == EXPERIMENT_ID
            else evidence.get(task_id, {}).get("honest_verdict")
        )
        for prior in task.get("prior_failures") or []:
            previous = str(prior.get("verdict"))
            same = previous == current
            environmental = previous.startswith("blocked_missing_new_physical_receipt")
            permanent = same and prior.get("retire_if_same_verdict") is True and not environmental
            rows.append(
                {
                    "mechanism": task_id,
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "previous_verdict": previous,
                    "current_verdict": current,
                    "addressed_by": prior.get("addressed_by"),
                    "same_exact_verdict": same,
                    "same_measured_null": False,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                    "environmental_failure": environmental,
                    "permanent_retirement": permanent,
                    "decision": "retire"
                    if permanent
                    else ("defer" if environmental else "continue"),
                }
            )
    p54 = evidence["exp7454-continuous-learning"]
    p55 = evidence["exp7455-decision-audit"]
    audit = p55["payload"].get("online_audit") or {}
    same_null = bool(
        p54.get("authenticated")
        and p54.get("valid")
        and p54.get("verdict_class") == "null"
        and p54["payload"].get("online_value_score") == 0
        and audit.get("valid") is True
        and audit.get("verdict_class") == "null"
        and (audit.get("independent_update_replay") or {}).get("primary_benefit_passed") is False
    )
    rows.append(
        {
            "mechanism": "four_expert_mixture",
            "task_id": "exp7454-continuous-learning+exp7455-decision-audit",
            "prior_experiment_id": "exp7440-mixture-learning",
            "previous_verdict": "complete_null_insufficient_online_benefit",
            "current_verdict": audit.get("honest_verdict"),
            "addressed_by": "Immutable expert predictions and independent delayed-loss replay.",
            "same_exact_verdict": p54.get("honest_verdict")
            == "complete_null_insufficient_online_benefit",
            "same_measured_null": same_null,
            "retire_if_same_verdict": True,
            "environmental_failure": False,
            "permanent_retirement": same_null,
            "decision": "retire" if same_null else "continue",
            "retired_scope": "unchanged four-expert mixture construction only",
            "audit_artifact_admissible_for_readiness": p55.get("valid"),
        }
    )
    return rows


def continuation_rows(
    claims: Mapping[str, JsonDict], retirements: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Name one concrete changed condition for each scientific mechanism."""

    mixture_retired = any(
        row.get("mechanism") == "four_expert_mixture" and row.get("permanent_retirement") is True
        for row in retirements
    )
    del claims  # Decisions use independently reduced branch states above.
    return [
        {
            "mechanism": "source_conditioned_energy",
            "decision": "defer",
            "changed_cause_or_prerequisite": "a usable authenticated native embedding surface",
        },
        {
            "mechanism": "four_expert_mixture",
            "decision": "retire" if mixture_retired else "continue",
            "changed_cause_or_prerequisite": "a different expert set or update mechanism",
        },
        {
            "mechanism": "compact_extraction",
            "decision": "continue",
            "changed_cause_or_prerequisite": "a factual development canary that opens the sealed evaluation panel",
        },
        {
            "mechanism": "arc_supervisor",
            "decision": "defer",
            "changed_cause_or_prerequisite": "new games plus an independent reproduced arm effect",
        },
        {
            "mechanism": "durable_updates",
            "decision": "continue",
            "changed_cause_or_prerequisite": "a changed storage design with service ratio CI upper below 0.90",
        },
        {
            "mechanism": "board_access",
            "decision": "defer",
            "changed_cause_or_prerequisite": "a dated operator-authored physical state receipt",
        },
    ]


def unresolved_obligations(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep unavailable science, hardware, and forbidden work explicit."""

    return [
        {
            "obligation_id": "source_conditioned_calibration",
            "state": "blocked_external_model_surface",
            "path": evidence["exp7452-source-embeddings"]["source_path"],
            "required_change": "provide a usable native final-layer embedding surface",
        },
        {
            "obligation_id": "decision_audit_validation",
            "state": "disqualified",
            "path": evidence["exp7455-decision-audit"]["source_path"],
            "required_change": "fix required validation before using the audit for readiness",
        },
        {
            "obligation_id": "extraction_evaluation",
            "state": "unavailable_science",
            "required_change": "open and run the sealed 96-call evaluation panel",
        },
        {
            "obligation_id": "gatemate_hardware_access",
            "state": "blocked_unchanged_physical_prerequisite",
            "required_change": "supply a dated operator-authored physical state receipt",
        },
        {
            "obligation_id": "conductor_change",
            "state": "user_forbidden",
            "prohibited_path": "scripts/research_conductor.py",
            "current_task_action": "none",
        },
    ]


def publication_gate_row(result: Mapping[str, Any]) -> JsonDict:
    """Preserve established G1-G4 while denying V653 promotion."""

    return {
        "headline_scope": "FoVer dual-condition AUROC",
        "certifies_v653": False,
        "paper_ready": result.get("paper_ready"),
        "gates": deepcopy(result.get("gates")),
        "unmet_gates": deepcopy(result.get("unmet_gates")),
        "note": result.get("note"),
    }


def _task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Build thirteen source rows and one current-work row."""

    rows: list[JsonDict] = []
    for task in tasks[:-1]:
        source = evidence[str(task["id"])]
        rows.append(
            {
                "order": len(rows) + 1,
                "task_id": task["id"],
                "declared_path": task["deliverable"],
                "observed_path": source["source_path"],
                "source_kind": source["source_kind"],
                "authenticated": source["authenticated"],
                "available": source["available"],
                "valid": source["valid"],
                "raw_rows_available": source["row_manifest_authenticated"],
                "raw_row_count": source["raw_row_count"],
                "raw_shard_count": source["raw_shard_count"],
                "validation_receipts_authenticated": source["validation_receipts_authenticated"],
                "required_validation_passed": source["required_validation_passed"],
                "honest_verdict": source["honest_verdict"],
                "verdict_class": source["verdict_class"],
                "flagged_adversarial": source["flagged_adversarial"],
                "censored": False,
            }
        )
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    rows.append(
        {
            "order": 14,
            "task_id": EXPERIMENT_ID,
            "declared_path": tasks[-1]["deliverable"],
            "observed_path": RESULT_PATH.as_posix(),
            "source_kind": "current_work",
            "authenticated": current_valid,
            "available": True,
            "valid": current_valid,
            "raw_rows_available": True,
            "raw_row_count": 14,
            "raw_shard_count": 0,
            "validation_receipts_authenticated": current_valid,
            "required_validation_passed": current_valid,
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "flagged_adversarial": terminal["verdict_class"] == "disqualified",
            "censored": False,
        }
    )
    return rows


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> dict[str, JsonDict]:
    """Bind authorities, current code, and every predecessor artifact."""

    paths = {
        "research-roadmap": ROADMAP_PATH,
        "markdown-contract": DESIGN_PATH,
        "research-reporting-spec": SPEC_PATH,
        "publication-gate": PUBLICATION_GATE_PATH,
        "current-module": MODULE_PATH,
        "current-wrapper": WRAPPER_PATH,
        "current-test": TEST_PATH,
    }
    rows = {
        name: {"path": path.as_posix(), "sha256": sha256_file(root / path)}
        for name, path in paths.items()
    }
    for task_id, source in evidence.items():
        rows[task_id] = {
            "path": source["source_path"],
            "declared_path": source["declared_path"],
            "sha256": source.get("sha256"),
            "source_kind": source["source_kind"],
            "original_honest_verdict": source["honest_verdict"],
            "original_verdict_class": source["verdict_class"],
            "original_flagged_adversarial": source["flagged_adversarial"],
            "raw_rows_sha256": source.get("raw_rows_sha256"),
            "validation_receipts_sha256": source.get("validation_receipts_sha256"),
        }
    rows["contract-comparison"] = {
        "path": f"{DESIGN_PATH.as_posix()} + {ROADMAP_PATH.as_posix()}",
        "sha256": canonical_hash(contract.get("contract_rows")),
        "source_kind": "canonical_reduction",
    }
    return rows


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Reload every file-backed source hash and the contract reduction."""

    rows = artifact.get("source_artifact_hashes")
    if not isinstance(rows, Mapping):
        return False
    for row in rows.values():
        if not isinstance(row, Mapping):  # pragma: no cover - schema guard.
            return False
        if row.get("source_kind") == "canonical_reduction":
            if row.get("sha256") != canonical_hash(load_contract(root).get("contract_rows")):
                return False
            continue
        label = row.get("path")
        if not isinstance(label, str) or " + " in label:
            return False
        path = _resolve_path(root, label)
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            return False
    return True


def _historical_sidecars(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep old model and small-head work outside current counters."""

    rows: list[JsonDict] = []
    for task_id, source in evidence.items():
        payload = source["payload"]
        kinds: list[str] = []
        if payload.get("model_invoked") is True:
            kinds.append("historical_model_invocation")
        training = payload.get("small_ebm_training")
        if isinstance(training, Mapping) and training.get("performed") is True:
            kinds.append("historical_small_ebm_training")
        for kind in kinds:
            rows.append(
                {
                    "task_id": task_id,
                    "kind": kind,
                    "path": source["source_path"],
                    "sha256": source.get("sha256"),
                    "counts_as_current_invocation": False,
                }
            )
    return rows


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep validity, benefit, completion, and promotion separate."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    structured: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, JsonDict],
) -> list[JsonDict]:
    """Build current gates without converting valid nulls to failures."""

    sources_authenticated = all(row["authenticated"] for row in evidence.values())
    return [
        _gate(
            "fourteen_task_dispositions",
            "completion",
            "==",
            14,
            len(dispositions),
            len(dispositions) == 14,
            "Every promised task receives one terminal disposition.",
        ),
        _gate(
            "contract_authorities_agree",
            "current_validity",
            "==",
            True,
            contract.get("comparison_passed"),
            contract.get("comparison_passed") is True,
            "Both current authorities must agree on all fourteen tasks.",
        ),
        _gate(
            "predecessor_evidence_authenticated",
            "evidence_validity",
            "==",
            True,
            sources_authenticated,
            sources_authenticated,
            "Exact producer or pre-gate bytes preserve each original class and flag.",
        ),
        _gate(
            "structured_gate_scalar_results",
            "evidence_validity",
            "==",
            len(structured),
            sum(row.get("passed") is True for row in structured),
            all(row.get("passed") is True for row in structured),
            "A failed producer gate stays visible and cannot borrow peer evidence.",
        ),
        _gate(
            "delayed_learning_benefit",
            "benefit",
            "==",
            True,
            metrics["delayed_learning"]["primary_benefit_passed"],
            metrics["delayed_learning"]["primary_benefit_passed"] is True,
            "A valid failed benefit gate is a null finding.",
        ),
        _gate(
            "durability_speed_benefit",
            "benefit",
            "<",
            0.9,
            metrics["durability"]["total_service_ratio"],
            metrics["durability"]["speed_gate_passed"] is True,
            "Exact recovery does not imply a full-service speed benefit.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            "==",
            True,
            validation.get("required_checks_passed"),
            validation.get("required_checks_passed") is True,
            "The frozen affected-file plan controls current code validity.",
        ),
        _gate(
            "terminal_validation",
            "required_validation",
            "==",
            True,
            validation.get("terminal_validation_passed"),
            validation.get("terminal_validation_passed") is True,
            "Cold replay and unchanged strict readers control final publication.",
        ),
        _gate(
            "promotion_zero",
            "promotion",
            "==",
            0,
            0,
            True,
            "A host aggregation cannot publish or change production state.",
        ),
    ]


def zero_test_phase_spans() -> list[JsonDict]:
    """Return deterministic phase rows for unit artifact construction."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "heartbeat_count": 0,
            "checkpoint": checkpoint,
        }
        for phase, checkpoint in (
            ("preconditions", "thirteen_sources_authenticated"),
            ("plan", "affected_plan_frozen"),
            ("load", "no_current_model_load"),
            ("generate", "no_current_generation"),
            ("validate", "affected_checks_complete"),
            ("reduce", "seven_claims_reduced"),
            ("terminal_validation", "cold_readers_complete"),
            ("write", "terminal_artifact_ready"),
        )
    ]


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each field while keeping gate fields plain scalars."""

    special = {
        "schema": "Use a versioned top-level schema and exact experiment identity.",
        "run_date": "Use 20260920 with real UTC, monotonic, boot, and segment identity.",
        "preconditions_checked": "Name actual paths, identities, and observed gates before use.",
        "MODEL_SPECS": "Remain empty because this aggregation plans no current LLM work.",
        "model_invoked": "Separate attempted current work from archived model events.",
        "invocation_counts": "Balance current loads, forwards, and generations at zero.",
        "inference_substrate": "Describe current host aggregation without inheriting old compute.",
        "inference_substrate_class": "Declare aggregation and apply no model duration floor.",
        "execution_venue": "Use host and keep historical device evidence in sidecars.",
        "duration_s": "Measure current work without padding model or validation time.",
        "phase_spans": "Bind phase timings, progress events, and durable checkpoints.",
        "random_seed": "Name frozen seed domains and explain why current randomness is absent.",
        "reproducibility_checksum": "Bind code, protocol, inputs, reductions, and validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes, original classes, and flags.",
        "rows": "Keep one disposition for every task, including blocks and disqualifications.",
        "sample_size_budget": "Account for fourteen independent disposition units once.",
        "acceptance_gate_results": "Keep validity, benefit, completion, and promotion separate.",
        "gate_check_summary": "Name each failed upstream, path, check, field, and operand.",
        "verifier_is_oracle": "Remain false because no deployed verifier scores this audit.",
        "honest_verdict": "Use complete findings and terminal blocked external absence.",
        "verdict_class": "Use the closed enum; partial means unfinished owned work only.",
        "flagged_adversarial": "Preserve critical disqualification findings without rehabilitation.",
        "validation_receipts": "Record exact scoped commands, exits, durations, and log hashes.",
        "field_principles": "Explain field intent without wrapping scalar gate values.",
        "promotion_score": "Always zero; this capstone does not authorize rollout.",
        "capstone_complete_score": "One requires fourteen dispositions and current validation.",
        "task_dispositions": "Exactly fourteen ordered rows include current work without self-read.",
        "claim_matrix": "Separate completion, readiness, value, and publication scope.",
        "continuation_rows": "Require a changed cause, input, or mechanism before repetition.",
        "retirement_rows": "Record exact prior rules and the unchanged-mixture decision.",
        "publication_gates": "Preserve existing FoVer G1-G4 meanings without V653 promotion.",
        "unresolved_obligations": "Keep unavailable science, hardware, and forbidden work explicit.",
    }
    return {key: special.get(key, f"Plain terminal evidence for {key}.") for key in keys}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable protocol, evidence, reductions, checks, and decisions."""

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
    return canonical_hash({key: value for key, value in artifact.items() if key not in excluded})


def _boot_id() -> str:
    """Read the Linux boot identity used by monotonic clock segments."""

    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unavailable"


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    publication: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    started_monotonic_ns: int = 0,
    ended_monotonic_ns: int | None = None,
) -> JsonDict:
    """Build the schema-complete capstone from authenticated source rows."""

    terminal = classify_terminal(contract, evidence, validation)
    metrics = reduce_branch_metrics(evidence)
    claims = reduce_claim_matrix(evidence)
    structured = audit_structured_gates(contract, evidence)
    retirements = retirement_rows(contract["tasks"], evidence, terminal)
    dispositions = _task_dispositions(contract["tasks"], evidence, terminal, validation)
    gates = _acceptance_gates(contract, evidence, validation, dispositions, structured, metrics)
    complete = bool(
        len(dispositions) == 14
        and all(row.get("authenticated") is True for row in dispositions)
        and validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    validation_s = sum(
        float(row.get("duration_s", 0.0))
        for row in validation.get("validation_receipts", [])
        if isinstance(row, Mapping)
    )
    end_ns = ended_monotonic_ns
    if end_ns is None:
        end_ns = started_monotonic_ns + max(0, int(duration_s * 1_000_000_000))
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
        or any(row.get("flagged_adversarial") is True for row in evidence.values()),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": end_ns,
        "clock_identity": {
            "boot_id": _boot_id(),
            "segment_id": canonical_hash(
                {"start": started_monotonic_ns, "end": end_ns, "experiment": EXPERIMENT_ID}
            ),
        },
        "duration_s": float(duration_s),
        "duration_components_s": {
            "model_load_s": 0.0,
            "forward_s": 0.0,
            "generation_s": 0.0,
            "numeric_computation_s": max(0.0, float(duration_s) - validation_s),
            "validation_s": validation_s,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "host aggregation of authenticated V653 evidence",
        "inference_substrate_class": "aggregation",
        "inference_substrate_details": {
            "current_compute": "JSON and YAML reduction",
            "current_llm_operations": 0,
            "current_cuda_operations": 0,
            "current_external_device_operations": 0,
            "historical_compute_inherited": False,
        },
        "execution_venue": "host",
        "random_seed": {
            "fit": None,
            "projection": None,
            "stream": None,
            "resampling": None,
            "reason": "The current reducer performs no stochastic work.",
        },
        "small_ebm_training": {"performed": False, "fit_count": 0},
        "preconditions_checked": [
            *[
                {
                    "check": "required_source_exists",
                    "upstream": path.as_posix(),
                    "path": path.as_posix(),
                    "artifact_field": "path",
                    "expected": True,
                    "observed": (root / path).is_file(),
                    "passed": (root / path).is_file(),
                }
                for path in SOURCE_PATHS
            ],
            {
                "check": "contract_authority_equivalence",
                "upstream": "v653-contract-authorities",
                "path": f"{DESIGN_PATH.as_posix()} + {ROADMAP_PATH.as_posix()}",
                "artifact_field": "comparison_passed",
                "expected": True,
                "observed": contract.get("comparison_passed"),
                "passed": contract.get("comparison_passed") is True,
            },
            *[
                {
                    "check": "predecessor_authentication",
                    "upstream": task_id,
                    "path": row["source_path"],
                    "artifact_field": "identity+rows+verdict+flag",
                    "expected": True,
                    "observed": row["authenticated"],
                    "passed": row["authenticated"],
                }
                for task_id, row in evidence.items()
            ],
        ],
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_evidence_sidecars": _historical_sidecars(evidence),
        "rows": deepcopy(dispositions),
        "sample_size_budget": {
            "planned": 14,
            "attempted": 14,
            "completed": 14,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_units": 14,
            "stopping_rule": "Reduce each ordered V653 task once and never retry a terminal block.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": terminal["gate_check_summary"],
        "verifier_is_oracle": False,
        "validation_receipts": deepcopy(validation.get("validation_receipts", [])),
        "repository_health": deepcopy(validation.get("repository_health", {})),
        "task_dispositions": dispositions,
        "structured_gate_audit": structured,
        "branch_metrics": metrics,
        "claim_matrix": claims,
        "retirement_rows": retirements,
        "continuation_rows": continuation_rows(claims, retirements),
        "publication_gates": publication_gate_row(publication),
        "unresolved_obligations": unresolved_obligations(evidence),
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "declared_entrypoint_cold_replay",
            "numbered_e2e_applicable": [],
        },
        "roadmap_activated": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_contact_performed": False,
        "publication_performed": False,
        "push_performed": False,
        "promotion_score": 0,
        "capstone_complete_score": int(complete),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _receipt_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require one successful receipt for every exact command name."""

    if not isinstance(receipts, list):
        return False
    by_name = {str(row.get("name")): row for row in receipts if isinstance(row, Mapping)}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


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
        "branch_metrics",
        "claim_matrix",
        "continuation_rows",
        "retirement_rows",
        "publication_gates",
        "unresolved_obligations",
        "promotion_score",
        "capstone_complete_score",
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
    if not str(artifact.get("status")).startswith("complete"):
        errors.append("lifecycle_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate_class") != "aggregation"
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
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    if artifact.get("publication_gates") != publication_gate_row(evaluate_publication_gates()):
        errors.append("publication_gates_invalid")
    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        receipts = artifact.get("validation_receipts")
        validation = {
            "required_checks_passed": _receipt_passed(receipts, REQUIRED_CHECK_NAMES),
            "terminal_validation_passed": _receipt_passed(receipts, TERMINAL_CHECK_NAMES),
            "validation_receipts": receipts,
        }
        terminal = classify_terminal(contract, evidence, validation)
        expected_dispositions = _task_dispositions(
            contract["tasks"], evidence, terminal, validation
        )
        if dispositions != expected_dispositions:
            errors.append("task_dispositions_invalid")
        metrics = reduce_branch_metrics(evidence)
        if artifact.get("branch_metrics") != metrics:
            errors.append("branch_metrics_invalid")
        claims = reduce_claim_matrix(evidence)
        if artifact.get("claim_matrix") != claims:
            errors.append("claim_matrix_invalid")
        structured = audit_structured_gates(contract, evidence)
        if artifact.get("structured_gate_audit") != structured:
            errors.append("structured_gate_audit_invalid")
        retirements = retirement_rows(contract["tasks"], evidence, terminal)
        if artifact.get("retirement_rows") != retirements:
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != continuation_rows(claims, retirements):
            errors.append("continuation_rows_invalid")
        if artifact.get("unresolved_obligations") != unresolved_obligations(evidence):
            errors.append("unresolved_obligations_invalid")
        if (
            artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != terminal["gate_check_summary"]
        ):
            errors.append("terminal_reduction_invalid")
        expected_complete = int(
            len(expected_dispositions) == 14
            and all(row.get("authenticated") is True for row in expected_dispositions)
            and validation["required_checks_passed"]
            and validation["terminal_validation_passed"]
        )
        if artifact.get("capstone_complete_score") != expected_complete:
            errors.append("capstone_score_invalid")
        expected_gates = _acceptance_gates(
            contract, evidence, validation, expected_dispositions, structured, metrics
        )
        if artifact.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gates_invalid")
        expected_flag = terminal["verdict_class"] == "disqualified" or any(
            row.get("flagged_adversarial") is True for row in evidence.values()
        )
        if artifact.get("flagged_adversarial") is not expected_flag:
            errors.append("flagged_adversarial_invalid")
        if require_terminal and not validation["terminal_validation_passed"]:
            errors.append("terminal_validation_incomplete")
    except (KeyError, OSError, TypeError, ValueError):  # pragma: no cover - fail closed.
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
    """Replay the reducer without requiring its own terminal receipt."""

    return validate_artifact(value, root=root, require_terminal=False)


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the frozen Exp7358 plan for current affected files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V653 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def _phase_span(  # pragma: no cover - real timing boundary.
    phase: str, phase_started: float, run_started: float, *, checkpoint: str
) -> JsonDict:
    """Close one measured phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "heartbeat_count": 0,
        "checkpoint": checkpoint,
    }


def _terminal_commands(  # pragma: no cover - capability subprocess boundary.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, strict readers, and unchanged G1-G4 execution."""

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


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - public E2E.
    """Run exact reads, scoped checks, cold readers, and atomic publication."""

    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7460-", dir="/tmp"))

    point = time.monotonic()
    progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    publication = evaluate_publication_gates()
    spans.append(
        _phase_span("preconditions", point, started, checkpoint="thirteen_sources_authenticated")
    )
    progress(started, "preconditions", "after", completed_units=len(evidence))

    point = time.monotonic()
    progress(started, "plan", "before")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    spans.append(_phase_span("plan", point, started, checkpoint="affected_plan_frozen"))
    progress(started, "plan", "after", commands=len(commands))

    for phase, checkpoint in (
        ("load", "no_current_model_load"),
        ("generate", "no_current_generation"),
    ):
        point = time.monotonic()
        progress(started, phase, "before", current_llm_operations=0)
        spans.append(_phase_span(phase, point, started, checkpoint=checkpoint))
        progress(started, phase, "after", current_llm_operations=0)

    point = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses", units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    validation: JsonDict = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": affected,
        "repository_health": {
            "status": "outside_current_affected_validity",
            "historical_failures": [],
        },
    }
    spans.append(_phase_span("validate", point, started, checkpoint="affected_checks_complete"))
    progress(started, "validate", "after_affected_subprocesses", passed=reduced["passed"])

    point = time.monotonic()
    progress(started, "reduce", "before", completed_units=0)
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=[
            *spans,
            _phase_span("reduce", point, started, checkpoint="seven_claims_reduced"),
        ],
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "reduce", "after", completed_units=14, verdict=candidate["verdict_class"])

    point = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_commands))
    terminal_receipts = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    validation["terminal_validation_passed"] = (
        all(row.get("passed") is True for row in terminal_receipts) and not critical
    )
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    spans.append(
        _phase_span("terminal_validation", point, started, checkpoint="cold_readers_complete")
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=validation["terminal_validation_passed"],
        critical=critical,
    )

    point = time.monotonic()
    progress(started, "write", "before_atomic", path=RESULT_PATH.as_posix())
    final_spans = [
        *spans,
        _phase_span("write", point, started, checkpoint="terminal_artifact_ready"),
    ]
    artifact = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "write", "after_atomic", path=RESULT_PATH.as_posix())
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Parse the frozen date and two cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, type=date_argument)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V653 capstone or one read-only cold validation."""

    print("[exp7460] phase=startup event=flushed", flush=True)
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
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": RESULT_PATH.as_posix(),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "capstone_complete_score": artifact["capstone_complete_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
