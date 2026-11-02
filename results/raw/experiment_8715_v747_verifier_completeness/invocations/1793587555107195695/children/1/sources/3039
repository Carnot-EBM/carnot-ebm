"""Close V660 with fourteen dispositions and four separate conclusions.

The capstone reads producer bytes and conductor gate diagnostics. It loads no
model and changes no roadmap, model weight, production default, or publication.

Spec refs: REQ-REPORT-7559 and SCENARIO-REPORT-7559-*.
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
from carnot.experiment_7446_v652_capstone import (
    authenticate_row_manifest,
    authenticate_validation_receipts,
    load_json_object,
    numeric_experiment_id,
    terminal_status,
)
from carnot.experiment_7546_v660_contract_methods import (
    compare_contract_authorities,
    resolve_v660_roadmap,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7559-capstone"
SCHEMA = "carnot.exp7559.v660.capstone.v1"

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7559_v660_capstone.json")
RAW_DIR = Path("results/raw/experiment_7559_v660_capstone")
MODULE_PATH = Path("python/carnot/experiment_7559_v660_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7559_v660_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7559_v660_capstone.py")
NOTE_PATH = Path("docs/research-notes/v660-capstone.md")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7546, "contract-methods"),
        (7547, "count-stream"),
        (7548, "capture-runner"),
        (7549, "count-learning"),
        (7550, "count-audit"),
        (7551, "native-pilot"),
        (7552, "fit-capture"),
        (7553, "test-capture"),
        (7554, "energy-fit"),
        (7555, "source-evaluation"),
        (7556, "arc-corrected-custody"),
        (7557, "arc-generalization"),
        (7558, "service-boundary"),
        (7559, "capstone"),
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
    "sampling": None,
    "bootstrap": None,
    "audit": 7_559_660_01,
    "explanation": "Deterministic aggregation makes no fit, sample, or outcome-dependent draw.",
}
DIAGNOSTIC_PATHS = {
    f"exp{number}-{slug}": Path(f"results/experiment_{number}_{slug.replace('-', '_')}.json")
    for number, slug in (
        (7551, "native-pilot"),
        (7552, "fit-capture"),
        (7553, "test-capture"),
        (7554, "energy-fit"),
        (7555, "source-evaluation"),
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
    Path("python/carnot/experiment_7529_v658_capstone.py"),
    PUBLICATION_GATE_PATH,
    Path("scripts/recurring_blocker_ledger.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
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


def compare_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare the V660 public task fields through the qualified parser."""

    return compare_contract_authorities(markdown_text, roadmap)


def load_contract(root: Path) -> JsonDict:
    """Resolve the V660 YAML and compare it with the independent design."""

    selected, roadmap, candidates = resolve_v660_roadmap(root)
    comparison = compare_authorities((root / DESIGN_PATH).read_text(), roadmap)
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V660 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": deepcopy(candidates),
        "roadmap": deepcopy(roadmap),
        "tasks": deepcopy(tasks),
    }


def _ready_value_fields(payload: Mapping[str, Any]) -> JsonDict:
    """Retain scalar readiness values without interpreting them as benefit."""

    suffixes = ("_score", "_ready", "_complete")
    return {
        str(key): deepcopy(value)
        for key, value in sorted(payload.items())
        if str(key).endswith(suffixes)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _receipt_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Keep validation identity and hashes without copying large output tails."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list):
        return []
    fields = ("name", "required", "passed", "exit_code", "log_path", "log_sha256")
    return [
        {key: deepcopy(row.get(key)) for key in fields}
        for row in receipts
        if isinstance(row, Mapping)
    ]


def _required_receipts_pass(payload: Mapping[str, Any]) -> bool:
    """Require every producer receipt that the producer marked as required."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list) or not receipts:
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    required = [row for row in rows if row.get("required") is True] or rows
    return bool(required) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in required
    )


def _acceptance_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve producer gate operands without changing their conclusion."""

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


def _path_label(path: object, root: Path) -> str:
    """Use a stable repository path for diagnostics stored in this worktree."""

    candidate = Path(str(path))
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def parse_conductor_statuses(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Resolve the latest conductor row for each exact V660 title prefix."""

    path = root / CONDUCTOR_LOG_PATH
    if not path.is_file():
        return {}
    rows: dict[str, JsonDict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        for task in tasks:
            title = str(task.get("title") or "")
            shown = cells[1]
            if shown and title.startswith(shown):
                rows[str(task.get("id"))] = {
                    "timestamp": cells[0],
                    "title_prefix": shown,
                    "status": cells[2],
                    "detail": cells[3],
                    "path": CONDUCTOR_LOG_PATH.as_posix(),
                    "line": line_number,
                    "sha256": sha256_file(path),
                }
    return rows


def _base_missing(task: Mapping[str, Any]) -> JsonDict:
    """Create the shared shape for a task whose producer bytes are absent."""

    task_id = str(task.get("id") or "")
    expected = str(task.get("deliverable") or "")
    failure = {
        "check": "producer_artifact_exists",
        "upstream": task_id,
        "path": expected,
        "field": "path",
        "op": "exists",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    return {
        "task_id": task_id,
        "expected_path": expected,
        "artifact_path": None,
        "diagnostic_path": DIAGNOSTIC_PATHS.get(task_id, Path("")).as_posix() or None,
        "source_sha256": None,
        "source_size_bytes": 0,
        "evidence_state": "missing",
        "source_status": "MISSING",
        "authenticated": False,
        "custody_valid": False,
        "valid": False,
        "available": False,
        "advisory_only": task_id == "exp7546-contract-methods",
        "honest_verdict": "complete_blocked_missing_declared_producer_evidence",
        "verdict_class": "blocked",
        "original_verdict_class": None,
        "flagged_adversarial": False,
        "model_invoked": False,
        "inference_substrate": None,
        "inference_substrate_class": None,
        "ready_value_fields": {},
        "validation_receipts": [],
        "acceptance_gates": [],
        "row_receipt": {},
        "validation_receipt": {},
        "gate_check_summary": failure,
        "conductor_receipt": None,
        "payload": {},
    }


def _diagnostic_failure(payload: Mapping[str, Any], root: Path) -> JsonDict:
    """Normalize one conductor diagnostic without replacing absent values."""

    source = payload.get("blocked_diagnostic_contract")
    row = source if isinstance(source, Mapping) else payload
    return {
        "check": "conductor_pre_gate",
        "upstream": row.get("failed_upstream"),
        "path": _path_label(row.get("failed_evidence_path"), root),
        "field": row.get("failed_field"),
        "op": row.get("failed_operator"),
        "expected": deepcopy(row.get("failed_expected")),
        "observed": deepcopy(row.get("failed_observed")),
        "passed": False,
        "status": "GATE_BLOCK",
    }


def _gate_satisfied(observed: Any, op: object, expected: Any) -> bool:
    """Evaluate only the structured gate operators used by the V660 roadmap."""

    if op == "==":
        return observed == expected
    if op == "in":
        return isinstance(expected, list) and observed in expected
    return False


def _cascade_failure(
    task: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    prior: Mapping[str, JsonDict],
) -> JsonDict:
    """Name the first immediate gate that cannot read a producer field."""

    for gate in task.get("gated_on") or []:
        if not isinstance(gate, Mapping):
            continue
        upstream = str(gate.get("upstream") or "")
        field = str(gate.get("artifact_field") or "")
        source = prior.get(upstream, {})
        payload = source.get("payload") if isinstance(source, Mapping) else {}
        observed = payload.get(field, "absent") if isinstance(payload, Mapping) else "absent"
        expected = deepcopy(gate.get("value"))
        op = gate.get("op")
        if not _gate_satisfied(observed, op, expected):
            producer = tasks_by_id.get(upstream, {})
            return {
                "check": "conductor_pre_gate",
                "upstream": upstream,
                "path": str(producer.get("deliverable") or "absent_upstream_path"),
                "field": field,
                "op": op,
                "expected": expected,
                "observed": deepcopy(observed),
                "passed": False,
                "status": "GATE_BLOCK",
            }
    return {
        "check": "producer_artifact_exists",
        "upstream": str(task.get("id") or ""),
        "path": str(task.get("deliverable") or ""),
        "field": "path",
        "op": "exists",
        "expected": True,
        "observed": False,
        "passed": False,
        "status": "GATE_BLOCK",
    }


def _blocked_from_conductor(
    root: Path,
    task: Mapping[str, Any],
    conductor: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    prior: Mapping[str, JsonDict],
) -> JsonDict:
    """Use exact diagnostic bytes or the conductor row for an unstarted task."""

    row = _base_missing(task)
    diagnostic_relative = DIAGNOSTIC_PATHS.get(str(task.get("id") or ""))
    diagnostic = root / diagnostic_relative if diagnostic_relative else None
    payload: JsonDict = {}
    if diagnostic is not None and diagnostic.is_file():
        payload = load_json_object(diagnostic)
        failure = _diagnostic_failure(payload, root)
        source_path = diagnostic_relative.as_posix()
        source_hash = sha256_file(diagnostic)
        source_size = diagnostic.stat().st_size
    else:
        failure = _cascade_failure(task, tasks_by_id, prior)
        source_path = CONDUCTOR_LOG_PATH.as_posix()
        source_hash = conductor.get("sha256")
        source_size = (root / CONDUCTOR_LOG_PATH).stat().st_size
    row.update(
        artifact_path=None,
        diagnostic_path=diagnostic_relative.as_posix() if diagnostic_relative else None,
        source_sha256=source_hash,
        source_size_bytes=source_size,
        evidence_state="conductor_gate_blocked",
        source_status=str(conductor.get("status") or payload.get("status") or "GATE_BLOCK").upper(),
        authenticated=True,
        custody_valid=True,
        available=True,
        honest_verdict=str(
            payload.get("honest_verdict") or "GATE_BLOCK (conductor; no producer artifact)"
        ),
        verdict_class="blocked",
        gate_check_summary=failure,
        conductor_receipt=deepcopy(dict(conductor)),
        payload=payload,
        evidence_path=source_path,
    )
    return row


def _producer_failure(
    task_id: str,
    path: str,
    payload: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Name the first exact reason why present producer bytes are invalid."""

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
        "validation_failures": deepcopy(validation.get("failures") or []),
    }


def _invalid_producer(task: Mapping[str, Any], path: Path, error: str) -> JsonDict:
    """Keep malformed present bytes separate from an absent producer."""

    row = _base_missing(task)
    row.update(
        artifact_path=str(task.get("deliverable") or path),
        source_sha256=sha256_file(path),
        source_size_bytes=path.stat().st_size,
        evidence_state="invalid",
        source_status="INVALID",
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


def load_producer(
    root: Path,
    task: Mapping[str, Any],
    conductor: Mapping[str, Any],
    prior: Mapping[str, JsonDict],
    *,
    tasks_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Authenticate one producer or preserve its exact conductor gate block."""

    relative = Path(str(task.get("deliverable") or ""))
    path = root / relative
    if not path.is_file():
        if conductor.get("status") == "GATE_BLOCK":
            return _blocked_from_conductor(root, task, conductor, tasks_by_id or {}, prior)
        return _base_missing(task)
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
    task_id = str(task.get("id") or "")
    return {
        "task_id": task_id,
        "expected_path": relative.as_posix(),
        "artifact_path": relative.as_posix(),
        "diagnostic_path": DIAGNOSTIC_PATHS.get(task_id, Path("")).as_posix() or None,
        "source_sha256": sha256_file(path),
        "source_size_bytes": path.stat().st_size,
        "evidence_state": "terminal" if valid else "invalid",
        "source_status": "TERMINAL" if valid else "INVALID",
        "authenticated": authenticated,
        "custody_valid": authenticated,
        "valid": valid,
        "available": True,
        "advisory_only": task_id == "exp7546-contract-methods",
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "verdict_class": str(original_class) if valid else "disqualified",
        "original_verdict_class": original_class,
        "flagged_adversarial": payload.get("flagged_adversarial") is True,
        "model_invoked": payload.get("model_invoked") is True,
        "inference_substrate": payload.get("inference_substrate"),
        "inference_substrate_class": payload.get("inference_substrate_class"),
        "ready_value_fields": _ready_value_fields(payload),
        "validation_receipts": _receipt_rows(payload),
        "acceptance_gates": _acceptance_rows(payload),
        "row_receipt": rows,
        "validation_receipt": validation,
        "gate_check_summary": (
            deepcopy(payload.get("gate_check_summary"))
            if valid
            else _producer_failure(task_id, relative.as_posix(), payload, validation)
        ),
        "conductor_receipt": deepcopy(dict(conductor)) if conductor else None,
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Inventory all thirteen upstream tasks in declared conductor order."""

    conductor = parse_conductor_statuses(root, tasks)
    tasks_by_id = {str(task.get("id")): task for task in tasks}
    evidence: dict[str, JsonDict] = {}
    for task in tasks[:-1]:
        task_id = str(task.get("id"))
        evidence[task_id] = load_producer(
            root,
            task,
            conductor.get(task_id, {}),
            evidence,
            tasks_by_id=tasks_by_id,
        )
    return evidence


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, affected_complete: bool, terminal_complete: bool
) -> JsonDict:
    """Classify owned validation before producer invalidity and external absence."""

    if affected_complete and not terminal_complete:
        verdict = "partial"
        honest = "partial_retryable_current_capstone_validation_unfinished"
    elif not affected_complete:
        verdict = "disqualified"
        honest = "complete_disqualified_required_v660_capstone_validation"
    elif any(
        row.get("evidence_state") == "invalid" and row.get("advisory_only") is not True
        for row in rows
    ):
        verdict = "disqualified"
        honest = "complete_disqualified_required_v660_scientific_evidence"
    elif any(
        row.get("evidence_state") in {"missing", "conductor_gate_blocked"}
        or row.get("verdict_class") == "blocked"
        for row in rows
        if row.get("advisory_only") is not True
    ):
        verdict = "blocked"
        honest = "complete_blocked_required_v660_source_science_externally_gated"
    else:
        verdict = "null"
        honest = "complete_null_v660_accounting_without_aggregate_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _disposition(source: Mapping[str, Any], task: Mapping[str, Any], order: int) -> JsonDict:
    """Reduce one source while preserving its literal class, flags, and absence."""

    excluded = bool(
        source.get("advisory_only")
        or source.get("evidence_state") in {"missing", "invalid", "conductor_gate_blocked"}
        or source.get("verdict_class") in {"blocked", "disqualified"}
        or source.get("flagged_adversarial")
    )
    producer_started = source.get("evidence_state") not in {"missing", "conductor_gate_blocked"}
    evidence_path = source.get("artifact_path") or source.get("evidence_path")
    return {
        "order": order,
        "task_id": str(task.get("id")),
        "expected_artifact_path": source.get("expected_path"),
        "artifact_path": source.get("artifact_path"),
        "diagnostic_path": source.get("diagnostic_path"),
        "evidence_path": evidence_path,
        "artifact_sha256": source.get("source_sha256"),
        "evidence_state": source.get("evidence_state"),
        "source_status": source.get("source_status"),
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("verdict_class"),
        "original_verdict_class": source.get("original_verdict_class"),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
        "advisory_only": source.get("advisory_only") is True,
        "ready_value_fields": deepcopy(source.get("ready_value_fields") or {}),
        "row_receipt": deepcopy(source.get("row_receipt") or {}),
        "validation_receipt": deepcopy(source.get("validation_receipt") or {}),
        "validation_receipts": deepcopy(source.get("validation_receipts") or []),
        "acceptance_gates": deepcopy(source.get("acceptance_gates") or []),
        "gate_check_summary": deepcopy(source.get("gate_check_summary") or {}),
        "conductor_receipt": deepcopy(source.get("conductor_receipt")),
        "disposition_attempted": True,
        "disposition_completed": True,
        "attempted": True,
        "completed": True,
        "failed": source.get("evidence_state") == "invalid",
        "censored": False,
        "unstarted": not producer_started,
        "producer_started": producer_started,
        "producer_completed": source.get("evidence_state") == "terminal",
        "producer_failed": source.get("evidence_state") == "invalid",
        "producer_censored": False,
        "producer_unstarted": not producer_started,
        "excluded_from_positive_aggregate": excluded,
        "principle": "Literal source custody prevents a readiness field or absent result from becoming benefit.",
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
        _disposition(evidence[str(task.get("id"))], task, order)
        for order, task in enumerate(tasks[:-1], 1)
    ]
    rows.append(
        {
            "order": 14,
            "task_id": str(tasks[-1].get("id")),
            "expected_artifact_path": str(tasks[-1].get("deliverable")),
            "artifact_path": None,
            "diagnostic_path": None,
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
            "advisory_only": False,
            "ready_value_fields": {"capstone_complete_score": int(current_validation_complete)},
            "row_receipt": {},
            "validation_receipt": {"required_passed": current_validation_complete},
            "validation_receipts": [],
            "acceptance_gates": [],
            "gate_check_summary": {},
            "conductor_receipt": None,
            "disposition_attempted": True,
            "disposition_completed": current_validation_complete,
            "attempted": True,
            "completed": current_validation_complete,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "producer_started": True,
            "producer_completed": current_validation_complete,
            "producer_failed": False,
            "producer_censored": False,
            "producer_unstarted": False,
            "excluded_from_positive_aggregate": True,
            "principle": "Current work cannot authenticate itself through its future terminal path.",
        }
    )
    return rows


def _source_ref(source: Mapping[str, Any]) -> JsonDict:
    """Bind one branch input without embedding its large measurement rows."""

    return {
        "task_id": source.get("task_id"),
        "path": source.get("artifact_path") or source.get("expected_path"),
        "sha256": source.get("source_sha256"),
        "evidence_state": source.get("evidence_state"),
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("original_verdict_class") or source.get("verdict_class"),
        "flagged_adversarial": source.get("flagged_adversarial") is True,
    }


def build_branch_dispositions(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep four V660 conclusions independent from each other's prerequisites."""

    count_measure = evidence["exp7549-count-learning"]
    count_audit = evidence["exp7550-count-audit"]
    count_payload = count_audit.get("payload") or {}
    count_ready = count_payload.get("count_claims_qualified_score")
    count_benefit = count_payload.get("qualified_exploratory_effect_score")
    count_valid = bool(count_audit.get("valid") and count_ready == 1)

    source = evidence["exp7555-source-evaluation"]
    source_payload = source.get("payload") or {}
    source_valid = bool(source.get("valid"))
    source_complete = source_valid and source_payload.get("source_evaluation_complete_score") == 1
    source_benefit = (
        source_payload.get("qualified_source_benefit_score") if source_complete else None
    )

    custody = evidence["exp7556-arc-corrected-custody"]
    arc = evidence["exp7557-arc-generalization"]
    custody_payload = custody.get("payload") or {}
    arc_payload = arc.get("payload") or {}
    arc_valid = bool(custody.get("valid") and arc.get("valid"))
    custody_ready = custody_payload.get("corrected_arc_ready_score")
    analysis_complete = arc_payload.get("arc_analysis_complete_score")
    arc_benefit = int(
        bool(
            arc_valid
            and arc_payload.get("gate_ready_to_ship") is True
            and (arc_payload.get("endpoint_identifiability") or {}).get(
                "plan_linked_efficacy_identifiable"
            )
            is True
        )
    )

    service = evidence["exp7558-service-boundary"]
    service_payload = service.get("payload") or {}
    service_valid = bool(service.get("valid"))
    service_complete = service_payload.get("service_cost_complete_score")
    boards_complete = service_payload.get("board_continuity_complete_score")
    hardware_benefit = int(
        bool(
            service_valid
            and (service_payload.get("hardware_acceleration_bound") or {}).get(
                "measured_dominant_hardware_cost"
            )
            is True
        )
    )
    return [
        {
            "branch": "exploratory_cached_count_learning",
            "sources": [_source_ref(count_measure), _source_ref(count_audit)],
            "independent_audit_required": True,
            "independent_audit_complete": count_valid,
            "readiness": count_ready,
            "benefit": count_benefit,
            "confirmatory_benefit": count_payload.get("confirmatory_benefit_score"),
            "verdict_class": "null" if count_valid and count_benefit == 0 else "disqualified",
            "positive_claim": False,
            "exposure_scope": "exploratory_cached_previously_inspected_qwen_forecasts",
            "no_headroom": count_payload.get("no_headroom", False),
            "conclusion": "Independent arithmetic qualified the completed count measurement. The registered exploratory benefit failed.",
        },
        {
            "branch": "fresh_injected_tool_decisions",
            "sources": [_source_ref(source)],
            "independent_source_reduction_required": True,
            "independent_source_reduction_complete": source_complete,
            "readiness": int(source_complete),
            "benefit": source_benefit,
            "verdict_class": (str(source.get("verdict_class")) if source_complete else "blocked"),
            "positive_claim": False,
            "no_headroom": False,
            "conclusion": "Fresh source capture and its independent reduction did not run after the external GPU-capacity gate failed.",
        },
        {
            "branch": "corrected_live_agent_generalization",
            "sources": [_source_ref(custody), _source_ref(arc)],
            "custody_ready": custody_ready,
            "analysis_complete": analysis_complete,
            "benefit": arc_benefit,
            "gate_ready_to_ship": arc_payload.get("gate_ready_to_ship"),
            "verdict_class": "null" if arc_valid and analysis_complete == 1 else "disqualified",
            "positive_claim": False,
            "no_headroom": arc_payload.get("no_headroom", False),
            "conclusion": "Corrected bytes are qualified, but plan-linked efficacy is not identifiable and support floors failed.",
        },
        {
            "branch": "cpu_durability_and_board_continuity",
            "sources": [_source_ref(service)],
            "service_complete": service_complete,
            "board_continuity_complete": boards_complete,
            "hardware_benefit": hardware_benefit,
            "board_rows": deepcopy(service_payload.get("board_rows") or []),
            "verdict_class": (
                "null"
                if service_valid and service_complete == 1 and boards_complete == 1
                else "disqualified"
            ),
            "positive_claim": False,
            "no_headroom": False,
            "conclusion": "Durable CPU service and board accounting completed. No measured dominant cost supports hardware benefit.",
        },
    ]


def reduce_prior_failures(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Compare literal prior verdicts and retire only completed repeated science."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id"))
        priors = task.get("prior_failures") or []
        if not priors or any(not isinstance(prior, Mapping) for prior in priors):
            raise ValueError(f"at least one prior failure required for {task_id}")
        source = evidence.get(task_id)
        current_verdict = source.get("honest_verdict") if source else None
        external = bool(
            source
            and (
                source.get("evidence_state") in {"missing", "conductor_gate_blocked"}
                or source.get("verdict_class") == "blocked"
            )
        )
        completed_mechanism = bool(
            source
            and source.get("evidence_state") == "terminal"
            and source.get("valid") is True
            and not external
        )
        for prior in priors:
            exact = bool(source is not None and current_verdict == prior.get("verdict"))
            same_completed = bool(exact and completed_mechanism)
            triggered = bool(
                prior.get("retire_if_same_verdict") is True and same_completed and not external
            )
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment": prior.get("experiment_id"),
                    "prior_honest_verdict": prior.get("verdict"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "current_honest_verdict": current_verdict,
                    "current_verdict_class": source.get("verdict_class") if source else None,
                    "exact_text_match": exact,
                    "same_completed_mechanism": same_completed,
                    "external_absence": external,
                    "retirement_triggered": triggered,
                }
            )
    return rows


def retirement_rows(prior_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return only literal repeats of the same completed scientific mechanism."""

    return [
        {
            "task_id": row.get("task_id"),
            "prior_experiment": row.get("prior_experiment"),
            "prior_honest_verdict": row.get("prior_honest_verdict"),
            "current_honest_verdict": row.get("current_honest_verdict"),
            "changed_mechanism": row.get("addressed_by"),
            "scope": "same_completed_scientific_mechanism_only",
            "exclusion_workflow": "ops/exclusion_manifest.yaml",
        }
        for row in prior_rows
        if row.get("retirement_triggered") is True
    ]


def continuation_rows() -> list[JsonDict]:
    """Require changed evidence or mechanism before any branch reopens."""

    return [
        {
            "branch": "exploratory_cached_count_learning",
            "current_state": "qualified_exploratory_null",
            "changed_evidence": "Use a preregistered corpus whose labels were not inspected by prior Carnot research.",
            "changed_mechanism": "Use a source-aware or hierarchical learner that differs from the completed local eight-bin mechanism.",
            "reopen_when": "Both fresh exposure custody and one changed learner are frozen before outcomes.",
            "activation_authorized": False,
        },
        {
            "branch": "fresh_injected_tool_decisions",
            "current_state": "blocked_external_gpu_capacity",
            "changed_evidence": "Observe an admissible GPU and acquire a new exclusive lease before the native pilot starts.",
            "changed_mechanism": None,
            "reopen_when": "The exact Exp7548 capacity field changes to one and the pilot rechecks ownership.",
            "activation_authorized": False,
        },
        {
            "branch": "corrected_live_agent_generalization",
            "current_state": "qualified_null_missing_causal_endpoint",
            "changed_evidence": "Collect joined accepted-model, executed-plan, action, and attributable-progress outcomes with both classes across at least four games per class.",
            "changed_mechanism": "Bind each fired induction attempt to the executed plan instead of using frame change as efficacy.",
            "reopen_when": "At least 100 joined attempts satisfy the frozen support floor.",
            "activation_authorized": False,
        },
        {
            "branch": "cpu_durability_and_board_continuity",
            "current_state": "durability_complete_hardware_benefit_unmeasured",
            "changed_evidence": "Measure a dominant service cost or provide a dated operator-authored GateMate physical-change receipt.",
            "changed_mechanism": None,
            "reopen_when": "A measured denominator identifies an accelerable bottleneck or the physical prerequisite changes.",
            "activation_authorized": False,
        },
    ]


@lru_cache(maxsize=2)
def evaluate_publication_gates(root: Path) -> JsonDict:
    """Run the stable G1-G4 reader and recompute its conjunction."""

    command = (str(root / ".venv/bin/python"), "-u", PUBLICATION_GATE_PATH.as_posix(), "--json")
    completed = subprocess.run(  # noqa: S603 - fixed repository command.
        command, cwd=root, text=True, capture_output=True, timeout=120, check=False
    )
    try:
        parsed = json.loads(completed.stdout)
        if not isinstance(parsed, Mapping):
            raise json.JSONDecodeError("mapping required", completed.stdout, 0)
        raw_gates = parsed.get("gates")
        if not isinstance(raw_gates, Mapping):
            raise json.JSONDecodeError("gates required", completed.stdout, 0)
        gates = {
            name: deepcopy(raw_gates.get(name) or {"pass": False})
            for name in ("G1", "G2", "G3", "G4")
        }
        paper_ready = all(gates[name].get("pass") is True for name in gates)
        unmet = [name for name in gates if gates[name].get("pass") is not True]
        result: JsonDict = {
            **deepcopy(dict(parsed)),
            "gates": gates,
            "paper_ready": paper_ready,
            "unmet_gates": unmet,
        }
    except (json.JSONDecodeError, TypeError):
        result = {"paper_ready": False, "gates": {}, "unmet_gates": ["reader_failed"]}
    return {
        **result,
        "command_argv": list(command),
        "exit_code": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "stdout_sha256": "sha256:" + hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "publication_performed": False,
    }


def collect_preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record required paths, REQ identity, resources, and upstream states."""

    rows: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        exists = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if exists else "absent_or_empty",
                "passed": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "path": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "expected": "REQ-REPORT-7559",
            "observed": "REQ-REPORT-7559" if "REQ-REPORT-7559" in spec_text else None,
            "passed": "REQ-REPORT-7559" in spec_text,
        }
    )
    rows.append(
        {
            "check": "roadmap_authority",
            "upstream": "V660 authority resolution",
            "path": str(contract.get("selected_roadmap_path")),
            "field": "selected_roadmap_path",
            "expected": "milestone-matching staged or active YAML",
            "observed": contract.get("selected_roadmap_path"),
            "passed": contract.get("comparison_passed") is True,
            "candidates": deepcopy(contract.get("resolution_candidates") or []),
        }
    )
    disk = shutil.disk_usage(root)
    resource_passed = bool((os.cpu_count() or 0) > 0 and disk.free > 0)
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
            "passed": resource_passed,
        }
    )
    for task_id, source in evidence.items():
        rows.append(
            {
                "check": f"producer_state:{task_id}",
                "upstream": task_id,
                "path": source.get("artifact_path")
                or source.get("evidence_path")
                or source.get("expected_path"),
                "field": "evidence_state",
                "expected": "explicit terminal, invalid, missing, or conductor gate-blocked state",
                "observed": source.get("evidence_state"),
                "passed": source.get("evidence_state")
                in {"terminal", "invalid", "missing", "conductor_gate_blocked"},
                "sha256": source.get("source_sha256"),
            }
        )
    return rows


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Bind current code, authorities, notes, producer bytes, and diagnostics."""

    paths = [*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH]
    selected = Path(str(contract.get("selected_roadmap_path")))
    if selected not in paths:
        paths.append(selected)
    if (root / NOTE_PATH).is_file():
        paths.append(NOTE_PATH)
    rows = [
        {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "evidence_class": "current_source",
        }
        for relative in dict.fromkeys(paths)
    ]
    for task_id, source in evidence.items():
        path = (
            source.get("artifact_path")
            or source.get("evidence_path")
            or source.get("expected_path")
        )
        rows.append(
            {
                "path": path,
                "sha256": source.get("source_sha256"),
                "evidence_class": source.get("evidence_state"),
                "task_id": task_id,
                "original_honest_verdict": source.get("honest_verdict"),
                "original_verdict_class": source.get("original_verdict_class"),
                "original_flagged_adversarial": source.get("flagged_adversarial") is True,
            }
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
    """Store operands and a failure-prevention principle for one gate."""

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
    branches: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep current validity, accounting readiness, and benefit independent."""

    by_branch = {str(row.get("branch")): row for row in branches}
    custody_valid = all(
        source.get("custody_valid") is True
        for source in evidence.values()
        if source.get("evidence_state") != "missing" and source.get("advisory_only") is not True
    )
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    validity = "Invalid evidence cannot support science."
    readiness = "A valid null remains auditable because readiness is not benefit."
    benefit = "Insufficient support or an unavailable reducer cannot become promotion."
    count = by_branch["exploratory_cached_count_learning"]
    source = by_branch["fresh_injected_tool_decisions"]
    arc = by_branch["corrected_live_agent_generalization"]
    service = by_branch["cpu_durability_and_board_continuity"]
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v660_contract",
            str(contract.get("selected_roadmap_path")),
            "comparison_passed",
            True,
            contract.get("comparison_passed"),
            contract.get("comparison_passed") is True,
            validity,
        ),
        _gate(
            "producer_and_diagnostic_custody",
            "validity",
            "v660_upstreams",
            "task_dispositions",
            "custody_valid",
            True,
            custody_valid,
            custody_valid,
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
            "independent_count_audit",
            "readiness",
            "exp7550-count-audit",
            "branch_dispositions.exploratory_cached_count_learning",
            "independent_audit_complete",
            True,
            count.get("independent_audit_complete"),
            count.get("independent_audit_complete") is True,
            readiness,
        ),
        _gate(
            "count_exploratory_benefit",
            "benefit",
            "exp7550-count-audit",
            "branch_dispositions.exploratory_cached_count_learning",
            "benefit",
            1,
            count.get("benefit"),
            count.get("benefit") == 1,
            benefit,
        ),
        _gate(
            "independent_source_reduction",
            "support",
            "exp7555-source-evaluation",
            "branch_dispositions.fresh_injected_tool_decisions",
            "independent_source_reduction_complete",
            True,
            source.get("independent_source_reduction_complete"),
            source.get("independent_source_reduction_complete") is True,
            benefit,
        ),
        _gate(
            "corrected_arc_analysis",
            "readiness",
            "exp7557-arc-generalization",
            "branch_dispositions.corrected_live_agent_generalization",
            "analysis_complete",
            1,
            arc.get("analysis_complete"),
            arc.get("analysis_complete") == 1,
            readiness,
        ),
        _gate(
            "corrected_arc_benefit",
            "benefit",
            "exp7557-arc-generalization",
            "branch_dispositions.corrected_live_agent_generalization",
            "benefit",
            1,
            arc.get("benefit"),
            arc.get("benefit") == 1,
            benefit,
        ),
        _gate(
            "durable_service_and_board_accounting",
            "readiness",
            "exp7558-service-boundary",
            "branch_dispositions.cpu_durability_and_board_continuity",
            "service_and_board_complete",
            [1, 1],
            [service.get("service_complete"), service.get("board_continuity_complete")],
            service.get("service_complete") == service.get("board_continuity_complete") == 1,
            readiness,
        ),
        _gate(
            "measured_hardware_benefit",
            "benefit",
            "exp7558-service-boundary",
            "branch_dispositions.cpu_durability_and_board_continuity",
            "hardware_benefit",
            1,
            service.get("hardware_benefit"),
            service.get("hardware_benefit") == 1,
            benefit,
        ),
    ]


def _normalized_blocked_failure(source: Mapping[str, Any]) -> JsonDict:
    """Give every blocked source exact path, field, and operands."""

    summary = source.get("gate_check_summary")
    if isinstance(summary, Mapping):
        candidate = (
            summary.get("first_failure")
            or summary.get("failed_check")
            or (summary.get("failures") or [None])[0]
            or summary
        )
        if isinstance(candidate, Mapping):
            row = deepcopy(dict(candidate))
            row.setdefault("check", "producer_terminal_disposition")
            row.setdefault("upstream", source.get("task_id"))
            row.setdefault(
                "path",
                source.get("artifact_path")
                or source.get("evidence_path")
                or source.get("expected_path"),
            )
            row.setdefault("field", row.get("check") or "verdict_class")
            row.setdefault("op", "==")
            row.setdefault("expected", "unblocked_terminal_evidence")
            row.setdefault("observed", source.get("honest_verdict"))
            row["passed"] = False
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
    """Return only failures that determine invalid, blocked, or unfinished state."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "check": "contract_authorities_agree",
                "upstream": "v660_contract",
                "path": str(contract.get("selected_roadmap_path")),
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
    for source in evidence.values():
        if source.get("advisory_only") is True:
            continue
        if source.get("evidence_state") == "invalid":
            failures.append(deepcopy(dict(source.get("gate_check_summary") or {})))
        elif (
            source.get("evidence_state") in {"missing", "conductor_gate_blocked"}
            or source.get("verdict_class") == "blocked"
        ):
            failures.append(_normalized_blocked_failure(source))
    return failures


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every classification failure and its first exact cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


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


FIELD_PRINCIPLES = {
    "experiment_id": "The exact ID, milestone, and date prevent cross-run evidence drift.",
    "preconditions_checked": "Exact inputs and current resources prevent fabricated fallback.",
    "MODEL_SPECS": "An empty declaration prevents historical calls from becoming current calls.",
    "model_specs": "Both model-spec aliases stay empty for aggregation-only work.",
    "model_invoked": "False separates current aggregation from cited historical model work.",
    "invocation_counts": "Typed zero counters expose invented loads, forwards, or generations.",
    "inference_substrate_class": "The aggregation class prevents model-duration implications.",
    "inference_substrate": "The exact aggregation label distinguishes reduction from inference.",
    "execution_venue": "The legal host value stays separate from device and compute details.",
    "duration_s": "Monotonic current duration cannot absorb authoring or historical runtime.",
    "phase_spans": "Disjoint phase boundaries expose unfinished or hidden long work.",
    "random_seed": "Frozen applicable seeds prevent favorable rerun selection.",
    "reproducibility_checksum": "One digest binds code, authorities, source bytes, and raw rows.",
    "rows": "Fourteen ordered dispositions prevent missing work from becoming zero evidence.",
    "sample_size_budget": "Planned, completed, failed, censored, and unstarted units stay distinct.",
    "acceptance_gate_results": "Validity, readiness, support, and benefit cannot substitute.",
    "gate_check_summary": "Each block names exact upstream operands and observed absence.",
    "honest_verdict": "A complete prefix records terminal external blockage exactly once.",
    "verdict_class": "The closed enum reserves partial for unfinished owned work.",
    "verifier_is_oracle": "Probabilistic evidence and fixtures cannot become correctness proof.",
    "flagged_adversarial": "The current artifact flag stays separate from retained source flags.",
    "validation_receipts": "Exact commands, exits, scopes, and hashes make checks auditable.",
    "field_principles": "Each top-level field states the inference failure it prevents.",
    "capstone_complete_score": "A bare one requires fourteen honest dispositions, not positives.",
    "branch_dispositions": "Four scopes prevent a null, block, or readiness field from leaking.",
    "publication_gates": "Stable G1-G4 and exact unmet gates do not authorize publication.",
    "continuation_rows": "Changed evidence or mechanism is required before reopening a branch.",
}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give each top-level field one plain failure-prevention explanation."""

    return {
        field: FIELD_PRINCIPLES.get(
            field, f"Exact {field} evidence prevents a reader from inferring an unstated result."
        )
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks, process identity, and prose."""

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
        "device_compute",
    }
    return canonical_hash({key: item for key, item in value.items() if key not in excluded})


def _source_revision(root: Path) -> str:
    """Read the current Git revision without changing repository state."""

    completed = subprocess.run(  # noqa: S603 - fixed read-only Git command.
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _sample_size_budget(dispositions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate completed accounting from the underlying producer execution state."""

    return {
        "independent_unit": "ordered_v660_task_disposition",
        "planned": 14,
        "attempted": sum(int(row.get("disposition_attempted") is True) for row in dispositions),
        "completed": sum(int(row.get("disposition_completed") is True) for row in dispositions),
        "excluded": sum(
            int(row.get("excluded_from_positive_aggregate") is True) for row in dispositions
        ),
        "failed": sum(int(row.get("producer_failed") is True) for row in dispositions),
        "censored": sum(int(row.get("producer_censored") is True) for row in dispositions),
        "unstarted": sum(int(row.get("producer_unstarted") is True) for row in dispositions),
        "underlying_producer_work": {
            "planned_before_capstone": 13,
            "started": sum(int(row.get("producer_started") is True) for row in dispositions[:-1]),
            "completed_terminal_artifacts": sum(
                int(row.get("producer_completed") is True) for row in dispositions[:-1]
            ),
            "gate_blocked_without_run": sum(
                int(row.get("evidence_state") == "conductor_gate_blocked")
                for row in dispositions[:-1]
            ),
        },
    }


def capstone_markdown(value: Mapping[str, Any]) -> str:
    """Render ordered dispositions, four conclusions, and publication gates."""

    lines = [
        "# V660 capstone",
        "",
        "The milestone has fourteen honest dispositions. Scientific benefit remains branch-specific.",
        "",
        "## Fourteen dispositions",
        "",
        "| Task | Class | Literal disposition | Evidence |",
        "|---|---|---|---|",
    ]
    for row in value.get("task_dispositions") or []:
        path = (
            row.get("artifact_path")
            or row.get("evidence_path")
            or row.get("expected_artifact_path")
        )
        lines.append(
            f"| {row.get('task_id')} | {row.get('verdict_class')} | `{row.get('honest_verdict')}` | `{path}` |"
        )
    lines.extend(["", "## Independent branch conclusions", ""])
    for row in value.get("branch_dispositions") or []:
        lines.extend(
            [
                f"### {row.get('branch')}",
                "",
                f"Class: `{row.get('verdict_class')}`. {row.get('conclusion')}",
                "",
            ]
        )
    publication = value.get("publication_gates") or {}
    lines.extend(
        [
            "## Publication gates",
            "",
            f"Paper ready: `{publication.get('paper_ready')}`. Unmet gates: `{publication.get('unmet_gates')}`.",
            "",
            "No activation, publication, submission, external contact, or push occurred.",
            "",
        ]
    )
    return "\n".join(lines)


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
    """Build one compact terminal record from authenticated V660 evidence."""

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
    branches = build_branch_dispositions(evidence)
    priors = reduce_prior_failures(contract["tasks"], evidence)
    failures = failure_rows(contract, evidence, validation)
    receipts = deepcopy(validation.get("validation_receipts") or [])
    validation_s = sum(
        float(row.get("duration_s") or 0.0) for row in receipts if isinstance(row, Mapping)
    )
    historical_s = sum(
        float(source.get("payload", {}).get("duration_s") or 0.0)
        for source in evidence.values()
        if isinstance(source.get("payload"), Mapping)
    )
    complete_score = int(
        contract.get("comparison_passed") is True
        and len(dispositions) == 14
        and all(row.get("disposition_completed") is True for row in dispositions)
        and current_complete
    )
    contract_payload = evidence["exp7546-contract-methods"].get("payload") or {}
    arc_payload = evidence["exp7556-arc-corrected-custody"].get("payload") or {}
    publication = deepcopy(evaluate_publication_gates(root))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7559,
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
                "excluded_from_scientific_gate": True,
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
            "device": "host_cpu",
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
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
        "preserved_prior_boundaries": {
            "v659_prior_dispositions": deepcopy(contract_payload.get("prior_dispositions") or []),
            "v659_unissued_promises": deepcopy(contract_payload.get("unissued_promises") or []),
            "b2_historical_exp7531": deepcopy(arc_payload.get("historical_exp7531") or {}),
            "b2_historical_verdicts": deepcopy(arc_payload.get("historical_verdicts") or {}),
            "b2_positive_control_diagnostic": deepcopy(
                arc_payload.get("positive_control_diagnostic") or {}
            ),
            "exp7532_rehabilitated": False,
            "v657_static_audit_rehabilitated": False,
        },
    }
    artifact.update(
        {
            "rows": dispositions,
            "task_dispositions": dispositions,
            "sample_size_budget": _sample_size_budget(dispositions),
            "branch_dispositions": branches,
            "prior_failure_rows": priors,
            "retirement_rows": retirement_rows(priors),
            "permanent_exclusion_entries_added": [],
            "continuation_rows": continuation_rows(),
            "acceptance_gate_results": acceptance_gates(
                contract, evidence, dispositions, branches, validation
            ),
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
                validation.get("repository_health", {"status": "outside_current_affected_validity"})
            ),
            "capability_e2e": {
                "declared_entrypoint": "required",
                "fresh_process_cold_replay": "required",
                "independent_reduction": "required",
                "predict_release_update_persist_reload": {
                    "applicable_to_current_reporting_code": False,
                    "upstream_evidence_task": "exp7558-service-boundary",
                    "upstream_capability_e2e": deepcopy(
                        (evidence["exp7558-service-boundary"].get("payload") or {}).get(
                            "capability_e2e"
                        )
                        or {},
                    ),
                },
                "numbered_runtime_e2e": "not_applicable_read_only_reporting",
            },
            "no_headroom": False,
            "no_headroom_annotation": "No aggregate no-headroom claim is made. Count retention failed, source science is absent, ARC causal joins are absent, and hardware benefit is unmeasured.",
            "verifier_is_oracle": False,
            "roadmap_activation_performed": False,
            "roadmap_archive_performed": False,
            "publication_performed": False,
            "submission_performed": False,
            "external_contact_performed": False,
            "production_defaults_changed": False,
            "generator_weights_changed": False,
            "qwen_weights_preserved": True,
            "push_performed": False,
            "research_conductor_modified": False,
        }
    )
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
            ("terminal_validation", 4, "cold_readers_complete"),
            ("write", 1, "terminal_artifact_ready"),
        )
    ]


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
    """Cold-check identity, source bytes, reductions, principles, and checksum."""

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
        "branch_dispositions",
        "retirement_rows",
        "publication_gates",
        "continuation_rows",
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
            contract["tasks"], evidence, terminal, current_validation_complete=current_complete
        )
        branches = build_branch_dispositions(evidence)
        priors = reduce_prior_failures(contract["tasks"], evidence)
        failures = failure_rows(contract, evidence, validation)
        if dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        if artifact.get("branch_dispositions") != branches:
            errors.append("branch_dispositions_invalid")
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
        expected_gates = acceptance_gates(contract, evidence, expected_rows, branches, validation)
        if artifact.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gates_invalid")
        if artifact.get("sample_size_budget") != _sample_size_budget(expected_rows):
            errors.append("sample_size_budget_invalid")
        expected_complete = int(
            contract.get("comparison_passed") is True
            and len(expected_rows) == 14
            and all(row.get("disposition_completed") is True for row in expected_rows)
            and current_complete
        )
        score = artifact.get("capstone_complete_score")
        if not isinstance(score, int) or isinstance(score, bool) or score != expected_complete:
            errors.append("capstone_score_invalid")
        publication = evaluate_publication_gates(root)
        if (
            artifact.get("publication_gates") != publication
            or artifact.get("publication_gate_results") != publication
        ):
            errors.append("publication_gates_invalid")
        retained = [
            {
                "task_id": task_id,
                "flagged_adversarial": True,
                "excluded_from_scientific_gate": True,
            }
            for task_id, source in evidence.items()
            if source.get("flagged_adversarial") is True
        ]
        if artifact.get("retained_source_adversarial_flags") != retained:
            errors.append("source_adversarial_flags_invalid")
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
    """Replay every pure reduction without requiring final terminal receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the shared validation plan for only the affected Exp7559 files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V660 execution date."""

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
        f"[exp7559] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
    """Replace generated prose only after its bytes reach local storage."""

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

    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7559-", dir="/tmp"))

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in INPUT_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    preconditions = collect_preconditions(root, contract, evidence)
    required_preconditions = [
        row
        for row in preconditions
        if str(row.get("check")).startswith("source_bytes:")
        or row.get("check") in {"driving_requirement", "roadmap_authority", "aggregation_resource"}
    ]
    if any(row.get("passed") is not True for row in required_preconditions):
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
    progress(started, "note", "before_atomic", path=NOTE_PATH.as_posix())
    note_source = build_artifact(
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
    _atomic_text(root / NOTE_PATH, capstone_markdown(note_source))
    spans.append(
        _span(
            "note",
            phase_started,
            started,
            completed_units=1,
            checkpoint="v660_capstone_note_written",
        )
    )
    progress(started, "note", "after_atomic", path=NOTE_PATH.as_posix())

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

    print("[exp7559] phase=startup event=flushed", flush=True)
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
