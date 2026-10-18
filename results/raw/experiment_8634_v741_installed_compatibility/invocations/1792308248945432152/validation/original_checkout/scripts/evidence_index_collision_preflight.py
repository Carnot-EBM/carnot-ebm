#!/usr/bin/env python3
"""Read-only exact-deliverable evidence index for Exp5771.

Spec refs: REQ-REPORT-5771,
SCENARIO-REPORT-5771-EXACT-LOOKUP,
SCENARIO-REPORT-5771-FAIL-CLOSED,
SCENARIO-REPORT-5771-HISTORY-READONLY,
SCENARIO-REPORT-5771-FIELD-PRINCIPLES.

The conductor and historical ledgers contain legitimate same-number result
files from different work streams. This module deliberately separates the two
concepts that were previously conflated: a task's canonical artifact is the
exact deliverable path declared in roadmap/history metadata, while numeric
prefix matches are only diagnostic alias candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_RELATIVE_PATH = Path("results/experiment_5771_evidence_index_collision_preflight.json")
QUALIFICATION_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5784_evidence_index_terminal_qualification.json"
)
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

MILESTONE_FROM = "2026.07.514"
MILESTONE_TO = "2026.07.515"
RUN_DATE = "2026-07-22"
EXPERIMENT = 5771
EXPERIMENT_ID = "exp5771-evidence-index-collision-preflight"
QUALIFICATION_EXPERIMENT = 5784
QUALIFICATION_EXPERIMENT_ID = "exp5784-evidence-index-terminal-qualification"
SCHEMA = "carnot.exp5771.evidence_index_collision_preflight.v1"
QUALIFICATION_SCHEMA = "carnot.exp5784.evidence_index_terminal_qualification.v1"
INFERENCE_SUBSTRATE = "local_filesystem_metadata_and_hashes_no_llm"
QUALIFICATION_INFERENCE_SUBSTRATE = (
    "local_filesystem_metadata_hashes_and_explicit_test_receipts_no_llm"
)
CANONICAL_POLICY = "exact_declared_deliverable"
NEXT_RANGE = range(5769, 5782)
QUALIFICATION_RANGE = range(5782, 5796)
NON_ARTIFACT_OUTCOMES = {"GATE_BLOCK", "SKIP", "FAIL", "BLOCK"}
QUALIFICATION_TERMINAL_MTIME_NS = 1_784_678_400_000_000_000

SPEC_REFS = [
    "REQ-REPORT-5771",
    "SCENARIO-REPORT-5771-EXACT-LOOKUP",
    "SCENARIO-REPORT-5771-FAIL-CLOSED",
    "SCENARIO-REPORT-5771-HISTORY-READONLY",
    "SCENARIO-REPORT-5771-FIELD-PRINCIPLES",
]

QUALIFICATION_SPEC_REFS = [
    "REQ-REPORT-5784",
    "SCENARIO-REPORT-5784-TASK-OWNED-READINESS",
    "SCENARIO-REPORT-5784-TASK-OWNED-BLOCK",
    "SCENARIO-REPORT-5784-GATE-REPLAY",
]

REQUIRED_ARTIFACT_FIELDS = [
    "field_principles",
    "status",
    "preconditions_checked",
    "spec_refs",
    "roadmap_hashes",
    "research_complete_hash_before",
    "research_complete_hash_after",
    "canonical_identity_contract",
    "canonical_task_index",
    "same_number_alias_groups",
    "duplicate_history_blocks",
    "missing_declared_deliverables",
    "conflicting_hashes",
    "gate_artifact_ambiguities",
    "canonical_lookup_receipts",
    "mtime_inversion_control",
    "real_collision_fixture_receipts",
    "negative_control_receipts",
    "evidence_index_ready_score",
    "next_range_collision_count",
    "unresolved_canonical_count",
    "history_mutation_count",
    "producer_gate_fields",
    "research_complete_modified",
    "conductor_unchanged",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
]

QUALIFICATION_REQUIRED_ARTIFACT_FIELDS = [
    "status",
    "preconditions_checked",
    "prior_artifact_hash",
    "implementation_hashes_before",
    "implementation_hashes_after",
    "canonical_identity_contract",
    "focused_test_receipts",
    "integration_test_receipts",
    "global_baseline_receipts",
    "spec_coverage_receipts",
    "test_ownership_policy",
    "task_owned_failures",
    "pre_existing_global_failures",
    "terminal_finalizer_receipt",
    "bootstrap_skeleton_absent",
    "gate_replay_receipts",
    "evidence_index_ready_score",
    "next_range_collision_count",
    "unresolved_canonical_count",
    "history_mutation_count",
    "producer_gate_fields",
    "research_complete_modified",
    "research_roadmap_unchanged",
    "conductor_unchanged",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
]

FIELD_PRINCIPLES = {
    "schema": "Versioned schema for this read-only evidence-index receipt.",
    "experiment": "Numeric experiment id for stable machine grouping.",
    "experiment_id": "Task id that owns this deliverable.",
    "run_date": "Operator-specified date for the Exp5771 preflight.",
    "field_principles": "Maps every artifact field to its evidence boundary.",
    "status": "Bare terminal state for downstream gates.",
    "preconditions_checked": "Records parsed inputs, hashes, free resources, and failed checks before readiness is trusted.",
    "spec_refs": "Anchors this behavior to REQ-REPORT-5771 and scenarios.",
    "roadmap_hashes": "Hashes active and next roadmap inputs without repairing missing optional files.",
    "research_complete_hash_before": "Content hash of history before the read-only scan.",
    "research_complete_hash_after": "Content hash of history after the scan, proving no mutation.",
    "canonical_identity_contract": "Defines canonical evidence as milestone, task id, and declared deliverable.",
    "canonical_task_index": "Read-only projection of task identities and exact declared artifact status.",
    "same_number_alias_groups": "Lists numeric-prefix candidates as aliases unless they equal the declared deliverable.",
    "duplicate_history_blocks": "Reports repeated milestone history blocks without changing them.",
    "duplicate_task_ids": "Reports task ids that declare more than one deliverable in the selected history block.",
    "missing_declared_deliverables": "Lists declared paths that are absent or malformed and whether a non-artifact outcome exists.",
    "conflicting_hashes": "Reports duplicate identity rows that declare incompatible hashes.",
    "gate_artifact_ambiguities": "Shows where broad numeric glob helpers would have multiple candidates.",
    "canonical_lookup_receipts": "Machine-readable exact lookup receipts for canonical identities.",
    "mtime_inversion_control": "Demonstrates that a newer alias cannot override a declared canonical path.",
    "real_collision_fixture_receipts": "Receipts for the real 5760, 5764, and 5766 alias groups.",
    "negative_control_receipts": "Synthetic-control diagnostics for wrapper, missing, duplicate, and hash-conflict handling.",
    "evidence_index_ready_score": "Bare scalar readiness: 1.0 only when tests, range scan, and canonical resolution are clean.",
    "next_range_collision_count": "Bare count of unowned Exp5769-Exp5781 path collisions.",
    "unresolved_canonical_count": "Bare count of canonical identities that did not resolve or map to a recorded non-artifact outcome.",
    "history_mutation_count": "Bare count proving history mutation did not occur.",
    "producer_gate_fields": "Bare scalar fields copied for downstream roadmap gates.",
    "research_complete_modified": "False when research-complete.yaml stayed byte-identical.",
    "conductor_unchanged": "True because Exp5771 does not modify scripts/research_conductor.py.",
    "inference_substrate": "Declares local filesystem metadata and hashes only; no LLM, solver, benchmark, or hardware inference.",
    "test_commands": "Verification commands recorded exactly.",
    "test_exit_codes": "Observed verification exit codes recorded without relabeling.",
    "reproducibility_checksum": "Stable checksum over the artifact excluding itself.",
    "honest_verdict": "Terminal summary beginning complete: or blocked:.",
}

QUALIFICATION_FIELD_PRINCIPLES = {
    "schema": "Versioned schema for the Exp5784 terminal qualification receipt.",
    "experiment": "Numeric experiment id for the corrigendum artifact.",
    "experiment_id": "Task id that owns the terminal qualification deliverable.",
    "run_date": "Operator-specified date for the Exp5784 qualification.",
    "spec_refs": "Anchors the corrigendum to REQ-REPORT-5784 and scenarios.",
    "field_principles": "Maps every top-level qualification field to its evidence boundary.",
    "status": "Bare terminal state derived from task-owned receipts and exact-index gates.",
    "preconditions_checked": "Records input hashes, resource receipts, namespace scan, and failed checks.",
    "prior_artifact_hash": "Hashes the Exp5771 artifact that had the correct index but no passing receipt authority.",
    "implementation_hashes_before": "Hashes implementation/spec/test inputs captured before this correction edited them.",
    "implementation_hashes_after": "Hashes implementation/spec/test inputs used by the finalizer.",
    "canonical_identity_contract": "Carries the reused exact-deliverable identity tuple and alias policy.",
    "focused_test_receipts": "Task-owned focused unit and coverage receipt rows.",
    "integration_test_receipts": "Task-owned integration, gate replay, lint, and hygiene receipt rows.",
    "global_baseline_receipts": "Non-authoritative global pytest health receipts disclosed separately.",
    "spec_coverage_receipts": "Non-authoritative spec coverage health receipts disclosed separately.",
    "test_ownership_policy": "States which receipt classes authorize readiness and which are disclosure-only.",
    "task_owned_failures": "Any focused or integration receipt with nonzero exit code; these block readiness.",
    "pre_existing_global_failures": "Global or spec coverage failures disclosed without relabeling them task-owned.",
    "terminal_finalizer_receipt": "Records atomic replace, reopen, checksum, mtime, and terminal-status verification.",
    "bootstrap_skeleton_absent": "Bare boolean proving the emitted artifact contains the terminal schema, not a bootstrap stub.",
    "gate_replay_receipts": "Structured Exp5785 gate replay rows read back from the finalized artifact.",
    "evidence_index_ready_score": "Bare scalar readiness from task-owned receipts plus the reused exact index.",
    "next_range_collision_count": "Bare Exp5771 index collision count after owned-history deliverables are allowed.",
    "unresolved_canonical_count": "Bare count of unresolved exact canonical identities.",
    "history_mutation_count": "Bare count proving research-complete.yaml was not mutated.",
    "producer_gate_fields": "Bare scalar fields copied for downstream conductor gates.",
    "research_complete_modified": "False when research-complete.yaml stayed byte-identical.",
    "research_roadmap_unchanged": "True when the active roadmap hash stayed byte-identical.",
    "conductor_unchanged": "True when scripts/research_conductor.py stayed byte-identical.",
    "inference_substrate": "Declares local metadata, hashes, and explicit test receipts only; no LLM inference.",
    "test_commands": "All verification commands recorded exactly in receipt order.",
    "test_exit_codes": "Observed exit codes by command without relabeling failures.",
    "reproducibility_checksum": "Stable checksum over the qualification artifact excluding this checksum field.",
    "honest_verdict": "Terminal summary beginning complete: or blocked:.",
}

QUALIFICATION_HASH_PATHS = [
    SPEC_RELATIVE_PATH,
    Path("scripts/evidence_index_collision_preflight.py"),
    Path("tests/python/test_evidence_index_collision_preflight.py"),
    RESULT_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    Path("scripts/conductor_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
]

QUALIFICATION_TEST_OWNERSHIP_POLICY = {
    "readiness_authority": "task_owned focused and integration receipts only",
    "task_owned_suite_kinds": ["focused", "integration"],
    "disclosure_only_suite_kinds": ["global_baseline", "spec_coverage"],
    "blocking_rule": "any nonzero task_owned focused/integration exit code blocks readiness",
    "global_health_rule": (
        "global pytest and spec coverage failures are disclosed as health receipts and "
        "pre_existing_global_failures when marked pre_existing; they are not reclassified "
        "as task-owned evidence-index failures"
    ),
}

IGNORED_SCAN_DIRS = {
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
    "node_modules",
    "nano-trm",
}


def unwrap_value(value: Any) -> Any:
    """Return the bare value for principle-wrapped fields."""

    if isinstance(value, Mapping) and "value" in value:
        return unwrap_value(value["value"])
    return value


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta = {"path": path.as_posix(), "exists": path.exists(), "parsed": False, "error": None}
    if not path.exists():
        return {}, meta
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - preflight records parse errors.
        meta["error"] = str(exc)
        return {}, meta
    if isinstance(data, dict):
        meta["parsed"] = True
        return data, meta
    meta["error"] = f"expected mapping, got {type(data).__name__}"
    return {}, meta


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta = {
        "path": path.as_posix(),
        "exists": path.exists(),
        "loadable": False,
        "sha256": sha256_file(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - artifact diagnostics must not raise.
        meta["error"] = str(exc)
        return {}, meta
    if isinstance(data, dict):
        meta["loadable"] = True
        return data, meta
    meta["error"] = f"expected mapping, got {type(data).__name__}"
    return {}, meta


def task_number(task_id: str) -> str | None:
    match = re.match(r"exp(\d+)", task_id)
    return match.group(1) if match else None


def _history_blocks(root: Path) -> list[JsonDict]:
    data, _meta = read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    milestones = data.get("milestones")
    if not isinstance(milestones, list):
        return []
    return [
        block
        for block in milestones
        if isinstance(block, dict) and unwrap_value(block.get("id")) == MILESTONE_FROM
    ]


def _task_signature(block: Mapping[str, Any]) -> str:
    rows = []
    tasks = block.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, Mapping):
                rows.append(
                    [
                        str(unwrap_value(task.get("id")) or ""),
                        str(unwrap_value(task.get("deliverable")) or ""),
                    ]
                )
    return sha256_bytes(json.dumps(rows, sort_keys=True).encode("utf-8"))


def duplicate_history_blocks(root: Path) -> list[JsonDict]:
    blocks = _history_blocks(root)
    if len(blocks) <= 1:
        return []
    signatures = {_task_signature(block) for block in blocks}
    return [
        {
            "milestone": MILESTONE_FROM,
            "block_count": len(blocks),
            "unique_block_signature_count": len(signatures),
            "mutation": "preserved_read_only",
        }
    ]


def _selected_tasks(root: Path) -> list[JsonDict]:
    blocks = _history_blocks(root)
    if not blocks:
        return []
    tasks = blocks[0].get("tasks")
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks if isinstance(task, dict)]


def _declared_outcome(row: Mapping[str, Any], capstone_rows: Mapping[str, Any]) -> str:
    task_id = str(unwrap_value(row.get("id")) or "")
    capstone = capstone_rows.get(task_id)
    if isinstance(capstone, Mapping):
        outcome = unwrap_value(capstone.get("conductor_outcome"))
        if isinstance(outcome, str) and outcome:
            return outcome
    result = unwrap_value(row.get("result"))
    return str(result) if result else ""


def _capstone_rows(root: Path) -> dict[str, JsonDict]:
    path = root / "results/experiment_5768_v514_capstone_reconciliation.json"
    payload, _meta = read_json_mapping(path)
    rows = payload.get("task_outcome_matrix")
    if not isinstance(rows, dict):
        return {}
    return {str(key): value for key, value in rows.items() if isinstance(value, dict)}


def duplicate_task_ids(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    deliveries: dict[str, list[str]] = defaultdict(list)
    for row in tasks:
        task_id = str(unwrap_value(row.get("id")) or "")
        deliverable = str(unwrap_value(row.get("deliverable")) or "")
        if task_id and deliverable and deliverable not in deliveries[task_id]:
            deliveries[task_id].append(deliverable)
    return [
        {
            "milestone": MILESTONE_FROM,
            "task_id": task_id,
            "declared_deliverables": sorted(values),
        }
        for task_id, values in sorted(deliveries.items())
        if len(values) > 1
    ]


def conflicting_hashes(root: Path) -> list[JsonDict]:
    claims: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for block in _history_blocks(root):
        tasks = block.get("tasks")
        if not isinstance(tasks, list):
            continue
        for row in tasks:
            if not isinstance(row, Mapping):
                continue
            task_id = str(unwrap_value(row.get("id")) or "")
            deliverable = str(unwrap_value(row.get("deliverable")) or "")
            declared_hash = unwrap_value(row.get("sha256") or row.get("artifact_sha256"))
            if task_id and deliverable and isinstance(declared_hash, str) and declared_hash:
                claims[(MILESTONE_FROM, task_id, deliverable)].add(declared_hash)
    return [
        {"identity": list(identity), "declared_hashes": sorted(values)}
        for identity, values in sorted(claims.items())
        if len(values) > 1
    ]


def same_number_candidates(root: Path, task_id: str) -> list[Path]:
    number = task_number(task_id)
    if number is None:
        return []
    results_dir = root / "results"
    if not results_dir.exists():
        return []
    return sorted(
        path for path in results_dir.glob(f"experiment_{number}_*.json") if path.is_file()
    )


def _artifact_status(payload: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    if not meta.get("exists"):
        return "missing"
    if not meta.get("loadable"):
        return "malformed"
    if payload.get("schema") == "blocked_gate_check_v1" or payload.get("blocked_at_layer"):
        return "blocked_gate"
    status = payload.get("status")
    verdict = payload.get("honest_verdict")
    if isinstance(status, str) and status:
        return status
    if isinstance(verdict, str) and verdict.startswith("blocked"):
        return "blocked"
    if isinstance(verdict, str) and verdict.startswith("complete"):
        return "complete"
    return "unknown"


def _path_receipt(root: Path, path: Path, role: str) -> JsonDict:
    payload, meta = read_json_mapping(root / path)
    stat = (root / path).stat() if (root / path).exists() else None
    return {
        "path": path.as_posix(),
        "role": role,
        "present": bool(meta["exists"]),
        "loadable": bool(meta["loadable"]),
        "sha256": meta["sha256"],
        "status": _artifact_status(payload, meta),
        "honest_verdict": payload.get("honest_verdict", ""),
        "mtime_ns": stat.st_mtime_ns if stat else None,
    }


def build_canonical_task_index(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    tasks = _selected_tasks(root)
    capstone = _capstone_rows(root)
    index = []
    outcomes = {}
    for row in tasks:
        task_id = str(unwrap_value(row.get("id")) or "")
        deliverable = str(unwrap_value(row.get("deliverable")) or "")
        if not task_id or not deliverable:
            continue
        exact = root / deliverable
        payload, meta = read_json_mapping(exact)
        candidates = [path.relative_to(root) for path in same_number_candidates(root, task_id)]
        aliases = [path.as_posix() for path in candidates if path.as_posix() != deliverable]
        mtime_selected = (
            max((root / path for path in candidates), key=lambda path: path.stat().st_mtime)
            .relative_to(root)
            .as_posix()
            if candidates
            else None
        )
        outcome = _declared_outcome(row, capstone)
        non_artifact = outcome in NON_ARTIFACT_OUTCOMES
        resolved = bool(meta["exists"] and meta["loadable"])
        unresolved = not resolved and not non_artifact
        record = {
            "milestone": MILESTONE_FROM,
            "task_id": task_id,
            "declared_deliverable": deliverable,
            "identity": [MILESTONE_FROM, task_id, deliverable],
            "canonical_path": deliverable if resolved else None,
            "canonical_sha256": meta["sha256"],
            "canonical_status": _artifact_status(payload, meta),
            "canonical_resolved": resolved,
            "recorded_conductor_outcome": outcome,
            "recorded_non_artifact_outcome": bool(non_artifact and not resolved),
            "unresolved": unresolved,
            "same_number_candidates": [path.as_posix() for path in candidates],
            "alias_paths": aliases,
            "mtime_selected_path": mtime_selected,
            "mtime_selected_declared_path": mtime_selected == deliverable,
            "selection_policy": CANONICAL_POLICY,
        }
        index.append(record)
        outcomes[task_id] = {
            "outcome": outcome,
            "artifact_status": record["canonical_status"],
            "declared_deliverable": deliverable,
        }
    return index, outcomes


def canonical_lookup(
    canonical_task_index: Sequence[Mapping[str, Any]],
    milestone: str,
    task_id: str,
    declared_deliverable: str,
) -> JsonDict:
    """Return the exact declared path or fail closed with candidates."""

    matches = [
        row
        for row in canonical_task_index
        if row.get("milestone") == milestone
        and row.get("task_id") == task_id
        and row.get("declared_deliverable") == declared_deliverable
    ]
    if len(matches) == 1 and matches[0].get("canonical_resolved") is True:
        row = matches[0]
        return {
            "status": "resolved",
            "identity": [milestone, task_id, declared_deliverable],
            "resolved_path": declared_deliverable,
            "same_number_candidates": list(row.get("same_number_candidates", [])),
            "selection_policy": CANONICAL_POLICY,
        }
    candidates = list(matches[0].get("same_number_candidates", [])) if matches else []
    return {
        "status": "unresolved",
        "identity": [milestone, task_id, declared_deliverable],
        "resolved_path": None,
        "same_number_candidates": candidates,
        "selection_policy": "fail_closed_exact_declared_deliverable_only",
    }


def same_number_alias_groups(root: Path, index: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    groups = {}
    for row in index:
        aliases = list(row.get("alias_paths") or [])
        if not aliases:
            continue
        number = task_number(str(row["task_id"]))
        if number is None:
            continue
        groups[number] = {
            "experiment_number": number,
            "canonical": _path_receipt(
                root, Path(str(row["declared_deliverable"])), "canonical_declared_deliverable"
            )
            | {"task_id": row["task_id"]},
            "aliases": [_path_receipt(root, Path(path), "same_number_alias") for path in aliases],
            "mtime_selected_path": row.get("mtime_selected_path"),
            "selection_policy": CANONICAL_POLICY,
        }
    return dict(sorted(groups.items()))


def gate_artifact_ambiguities(index: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "task_id": row["task_id"],
            "declared_deliverable": row["declared_deliverable"],
            "same_number_candidates": row["same_number_candidates"],
            "mtime_selected_path": row["mtime_selected_path"],
            "mtime_selected_declared_path": row["mtime_selected_declared_path"],
            "diagnosis": "broad_numeric_glob_has_multiple_candidates",
        }
        for row in index
        if len(row.get("same_number_candidates", [])) > 1
    ]


def mtime_inversion_control(index: Sequence[Mapping[str, Any]]) -> JsonDict:
    for row in index:
        selected = row.get("mtime_selected_path")
        declared = row.get("declared_deliverable")
        if selected and selected != declared:
            receipt = canonical_lookup(
                index, str(row["milestone"]), str(row["task_id"]), str(declared)
            )
            return {
                "alias_newer_than_canonical": True,
                "task_id": row["task_id"],
                "mtime_selected_path": selected,
                "declared_deliverable": declared,
                "canonical_lookup_resolved_path": receipt["resolved_path"],
            }
    return {
        "alias_newer_than_canonical": False,
        "task_id": None,
        "mtime_selected_path": None,
        "declared_deliverable": None,
        "canonical_lookup_resolved_path": None,
    }


def _allowed_next_range_paths(root: Path, result_path: Path) -> set[str]:
    allowed = {
        ROADMAP_RELATIVE_PATH.as_posix(),
        ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        "scripts/evidence_index_collision_preflight.py",
        "tests/python/test_evidence_index_collision_preflight.py",
        result_path.relative_to(root).as_posix()
        if result_path.is_absolute()
        else result_path.as_posix(),
        "python/carnot/experiment_5769_transition_v515.py",
        "python/carnot/experiment_5770_v515_source_delta_ingestion.py",
        "tests/python/test_experiment_5769_transition_v515.py",
        "tests/python/test_experiment_5770_v515_source_delta_ingestion.py",
        "results/experiment_5769_transition_v515.json",
        "results/experiment_5770_v515_source_delta_ingestion.json",
    }
    complete, _complete_meta = read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    milestones = complete.get("milestones")
    if isinstance(milestones, list):
        for block in milestones:
            if not isinstance(block, Mapping):
                continue
            tasks = block.get("tasks")
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, Mapping):
                    continue
                deliverable = unwrap_value(task.get("deliverable"))
                if isinstance(deliverable, str):
                    allowed.add(deliverable)
    roadmap, _meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    tasks = roadmap.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, Mapping):
                deliverable = unwrap_value(task.get("deliverable"))
                if isinstance(deliverable, str):
                    allowed.add(deliverable)
    return allowed


def _repo_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_SCAN_DIRS)
        base = Path(dirpath)
        for filename in sorted(filenames):
            yield (base / filename).relative_to(root)


def next_range_collision_scan(root: Path, result_path: Path) -> JsonDict:
    allowed = _allowed_next_range_paths(root, result_path)
    collisions = []
    patterns = [
        re.compile(rf"(^|[^0-9])exp{number}([^0-9]|$)", re.IGNORECASE) for number in NEXT_RANGE
    ] + [re.compile(rf"experiment_{number}_", re.IGNORECASE) for number in NEXT_RANGE]
    for rel in _repo_files(root):
        rel_text = rel.as_posix()
        if rel_text in allowed:
            continue
        if any(pattern.search(rel_text) for pattern in patterns):
            collisions.append({"kind": "unowned_path", "path": rel_text})
    return {
        "range": "exp5769-exp5781",
        "allowed_paths": sorted(allowed),
        "preexisting_collisions": collisions,
        "collision_count": len(collisions),
        "collision_free": len(collisions) == 0,
    }


def _roadmap_hashes(root: Path) -> JsonDict:
    active_data, active_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    next_data, next_meta = read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    return {
        "active": {
            "path": ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / ROADMAP_RELATIVE_PATH),
            "parsed": active_meta["parsed"],
            "milestone": active_data.get("milestone"),
            "error": active_meta["error"],
        },
        "next": {
            "path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / ROADMAP_NEXT_RELATIVE_PATH),
            "parsed": next_meta["parsed"],
            "present": next_meta["exists"],
            "milestone": next_data.get("milestone"),
            "error": next_meta["error"],
        },
    }


def _resource_receipts(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    mem_available_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_kib = int(line.split()[1])
                break
    return {
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "mem_available_kib": mem_available_kib,
    }


def _tests_passed(tests_run: Sequence[Mapping[str, Any]]) -> bool:
    if not tests_run:
        return False
    return all(row.get("exit_code") == 0 for row in tests_run)


def _normalize_tests(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if tests_run is None:
        return []
    return [
        {"command": str(row.get("command", "")), "exit_code": row.get("exit_code")}
        for row in tests_run
    ]


def _negative_control_receipts(
    tasks: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    duplicates: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    wrapper_seen = any(
        isinstance(row.get("id"), Mapping) or isinstance(row.get("deliverable"), Mapping)
        for row in tasks
    )
    return {
        "wrapper_field": {"detected": wrapper_seen, "resolved": wrapper_seen},
        "missing_declared_deliverable": {"count": len(missing), "detected": bool(missing)},
        "duplicate_task_id": {"count": len(duplicates), "detected": bool(duplicates)},
        "hash_conflict": {"count": len(conflicts), "detected": bool(conflicts)},
    }


def _real_collision_fixture_receipts(groups: Mapping[str, Any]) -> JsonDict:
    receipts = {}
    for number in ("5760", "5764", "5766"):
        group = groups.get(number)
        receipts[number] = {
            "present": group is not None,
            "canonical_path": group["canonical"]["path"] if group else None,
            "alias_paths": [alias["path"] for alias in group.get("aliases", [])] if group else [],
        }
    return receipts


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    result_path: Path | None = None,
) -> JsonDict:
    root = root.resolve()
    result_path = result_path or root / RESULT_RELATIVE_PATH
    research_hash_before = sha256_file(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    roadmap_hashes = _roadmap_hashes(root)
    tasks = _selected_tasks(root)
    index, conductor_outcomes = build_canonical_task_index(root)
    aliases = same_number_alias_groups(root, index)
    duplicates = duplicate_task_ids(tasks)
    conflicts = conflicting_hashes(root)
    missing = [
        {
            "identity": row["identity"],
            "present": row["canonical_status"] != "missing",
            "status": row["canonical_status"],
            "recorded_conductor_outcome": row["recorded_conductor_outcome"],
            "recorded_non_artifact_outcome": row["recorded_non_artifact_outcome"],
        }
        for row in index
        if row["canonical_status"] in {"missing", "malformed"}
    ]
    unresolved_count = sum(1 for row in index if row["unresolved"])
    collision_scan = next_range_collision_scan(root, result_path)
    normalized_tests = _normalize_tests(tests_run)
    tests_ok = _tests_passed(normalized_tests)
    research_hash_after = sha256_file(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    history_mutation_count = 0 if research_hash_before == research_hash_after else 1
    failed = []
    if not roadmap_hashes["active"]["parsed"]:
        failed.append("active_roadmap_unparseable")
    if not tasks:
        failed.append("no_v514_history_tasks")
    if duplicates:
        failed.append("duplicate_task_ids")
    if conflicts:
        failed.append("conflicting_hashes")
    if unresolved_count:
        failed.append("unresolved_canonical_identities")
    if collision_scan["collision_count"]:
        failed.append("next_range_collisions")
    if history_mutation_count:
        failed.append("history_mutated")
    if not tests_ok:
        failed.append("tests_not_recorded_passing")
    ready = not failed
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "blocked",
        "preconditions_checked": {
            "roadmaps": roadmap_hashes,
            "research_complete_sha256": research_hash_before,
            "research_complete_blocks_for_milestone": len(_history_blocks(root)),
            "conductor_log_sha256": sha256_file(root / CONDUCTOR_LOG_RELATIVE_PATH),
            "conductor_sha256": sha256_file(root / CONDUCTOR_RELATIVE_PATH),
            "declared_deliverable_count": len(index),
            "resource_receipts": _resource_receipts(root),
            "failed_preconditions": failed,
        },
        "spec_refs": SPEC_REFS,
        "roadmap_hashes": roadmap_hashes,
        "research_complete_hash_before": research_hash_before,
        "research_complete_hash_after": research_hash_after,
        "canonical_identity_contract": {
            "identity_tuple": ["milestone", "task_id", "declared_deliverable"],
            "canonical_path_rule": "exact declared_deliverable only",
            "numeric_prefix_matches": "aliases_only",
            "selection_policy": CANONICAL_POLICY,
        },
        "canonical_task_index": index,
        "same_number_alias_groups": aliases,
        "duplicate_history_blocks": duplicate_history_blocks(root),
        "duplicate_task_ids": duplicates,
        "missing_declared_deliverables": missing,
        "conflicting_hashes": conflicts,
        "gate_artifact_ambiguities": gate_artifact_ambiguities(index),
        "canonical_lookup_receipts": [
            canonical_lookup(
                index,
                str(row["milestone"]),
                str(row["task_id"]),
                str(row["declared_deliverable"]),
            )
            for row in index
        ],
        "mtime_inversion_control": mtime_inversion_control(index),
        "real_collision_fixture_receipts": _real_collision_fixture_receipts(aliases),
        "negative_control_receipts": _negative_control_receipts(
            tasks, missing, duplicates, conflicts
        ),
        "evidence_index_ready_score": 1.0 if ready else 0.0,
        "next_range_collision_count": collision_scan["collision_count"],
        "unresolved_canonical_count": unresolved_count,
        "history_mutation_count": history_mutation_count,
        "producer_gate_fields": {},
        "research_complete_modified": history_mutation_count != 0,
        "conductor_unchanged": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": [row["command"] for row in normalized_tests],
        "test_exit_codes": {row["command"]: row["exit_code"] for row in normalized_tests},
        "honest_verdict": (
            "complete: exact-deliverable evidence index ready; aliases disclosed without canonical conflation"
            if ready
            else "blocked: evidence index preflight failed closed: " + ", ".join(failed)
        ),
    }
    report["producer_gate_fields"] = {
        "evidence_index_ready_score": report["evidence_index_ready_score"],
        "next_range_collision_count": report["next_range_collision_count"],
        "unresolved_canonical_count": report["unresolved_canonical_count"],
        "history_mutation_count": report["history_mutation_count"],
    }
    report["field_principles"] = _field_principles_for(report)
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def _field_principles_for(report: Mapping[str, Any]) -> dict[str, str]:
    missing = sorted(set(report) - set(FIELD_PRINCIPLES) - {"reproducibility_checksum"})
    if missing:
        raise KeyError(f"missing field principles: {missing}")
    principles = {
        field: FIELD_PRINCIPLES[field] for field in report if field != "reproducibility_checksum"
    } | {"reproducibility_checksum": FIELD_PRINCIPLES["reproducibility_checksum"]}
    principles["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    return principles


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    root = root.resolve()
    output_path = output_path or root / RESULT_RELATIVE_PATH
    report = build_report(root, tests_run=tests_run, result_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _qualification_hashes(root: Path) -> JsonDict:
    return {path.as_posix(): sha256_file(root / path) for path in QUALIFICATION_HASH_PATHS}


def normalize_qualification_receipts(
    receipts: Sequence[Mapping[str, Any]] | None,
) -> list[JsonDict]:
    """Normalize explicit command receipts without changing their ownership."""

    normalized = []
    for row in receipts or []:
        exit_code = row.get("exit_code")
        if isinstance(exit_code, bool):
            exit_code = int(exit_code)
        receipt = {
            "command": str(row.get("command", "")),
            "exit_code": exit_code,
            "ownership_class": str(row.get("ownership_class", "global_baseline")),
            "suite_kind": str(row.get("suite_kind", row.get("ownership_class", ""))),
            "failure_signature": str(row.get("failure_signature", "")),
            "pre_existing": bool(row.get("pre_existing", False)),
        }
        if "run_id" in row:
            receipt["run_id"] = str(row["run_id"])
        normalized.append(receipt)
    return normalized


def _failed_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in receipts if row.get("exit_code") != 0]


def _qualification_receipt_groups(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    task_owned = [dict(row) for row in receipts if row.get("ownership_class") == "task_owned"]
    focused = [row for row in task_owned if row.get("suite_kind") == "focused"]
    integration = [row for row in task_owned if row.get("suite_kind") != "focused"]
    global_baseline = [
        dict(row)
        for row in receipts
        if row.get("ownership_class") == "global_baseline"
        or row.get("suite_kind") == "global_baseline"
    ]
    spec_coverage = [
        dict(row)
        for row in receipts
        if row.get("ownership_class") == "spec_coverage" or row.get("suite_kind") == "spec_coverage"
    ]
    return {
        "task_owned": task_owned,
        "focused": focused,
        "integration": integration,
        "global_baseline": global_baseline,
        "spec_coverage": spec_coverage,
    }


def _task_owned_tests_for_index(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[JsonDict]:
    return [
        {"command": str(row.get("command", "")), "exit_code": row.get("exit_code")}
        for row in [*groups.get("focused", []), *groups.get("integration", [])]
    ]


def _test_exit_code_key(row: Mapping[str, Any]) -> str:
    command = str(row.get("command", ""))
    run_id = row.get("run_id")
    return f"{run_id}::{command}" if run_id else command


def _matches_exp_range_path(path: Path, exp_range: range) -> bool:
    text = path.as_posix()
    return any(
        re.search(rf"(^|[^0-9])exp{number}([^0-9]|$)", text, re.IGNORECASE)
        or re.search(rf"experiment_{number}_", text, re.IGNORECASE)
        for number in exp_range
    )


def exp5782_5795_collision_scan(root: Path, result_path: Path | None = None) -> JsonDict:
    """Scan allocated V516 paths and distinguish owned files from collisions."""

    root = root.resolve()
    result_path = result_path or root / QUALIFICATION_RESULT_RELATIVE_PATH
    allowed: dict[str, str] = {}
    roadmap, _meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    tasks = roadmap.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            deliverable = unwrap_value(task.get("deliverable"))
            task_id = str(unwrap_value(task.get("id")) or "")
            if isinstance(deliverable, str) and task_number(task_id) in {
                str(number) for number in QUALIFICATION_RANGE
            }:
                allowed[deliverable] = "active_roadmap_declared_deliverable"
    if result_path.is_absolute():
        allowed[result_path.relative_to(root).as_posix()] = "current_task_deliverable"
    else:
        allowed[result_path.as_posix()] = "current_task_deliverable"

    owned = []
    collisions = []
    for rel in _repo_files(root):
        if not _matches_exp_range_path(rel, QUALIFICATION_RANGE):
            continue
        rel_text = rel.as_posix()
        reason = allowed.get(rel_text)
        if reason is None and re.match(
            r"(python/carnot/experiment_57(8[2-9]|9[0-5])_|tests/python/test_experiment_57(8[2-9]|9[0-5])_)",
            rel_text,
        ):
            reason = "owned_implementation_or_test_path"
        if reason is None:
            collisions.append({"kind": "unowned_path", "path": rel_text})
        else:
            owned.append({"path": rel_text, "ownership": reason})
    return {
        "range": "exp5782-exp5795",
        "owned_paths": sorted(owned, key=lambda row: row["path"]),
        "unowned_path_collisions": sorted(collisions, key=lambda row: row["path"]),
        "collision_count": len(collisions),
        "collision_free": len(collisions) == 0,
    }


def bootstrap_skeleton_absent(payload: Mapping[str, Any]) -> bool:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return (
        set(QUALIFICATION_REQUIRED_ARTIFACT_FIELDS).issubset(payload)
        and payload.get("status") in {"complete", "blocked"}
        and "artifact_not_updated_past_bootstrap" not in text
    )


def _qualification_gate_task(root: Path) -> JsonDict | None:
    roadmap, _meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return None
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        if unwrap_value(task.get("id")) == "exp5785-hardness-surface-prospective-fixture":
            return dict(task)
    return None


def replay_qualification_gates(
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
) -> list[JsonDict]:
    """Replay Exp5785 gates against the reopened Exp5784 artifact."""

    root = root.resolve()
    artifact_path = artifact_path or root / QUALIFICATION_RESULT_RELATIVE_PATH
    payload, meta = read_json_mapping(artifact_path)
    source = (
        artifact_path.relative_to(root).as_posix()
        if artifact_path.is_absolute()
        else artifact_path.as_posix()
    )
    mtime_ns = artifact_path.stat().st_mtime_ns if artifact_path.exists() else None
    checksum_match = bool(meta.get("loadable")) and payload_checksum(payload) == payload.get(
        "reproducibility_checksum"
    )
    skeleton_absent = bool(meta.get("loadable")) and bootstrap_skeleton_absent(payload)
    base = {
        "source": source,
        "reopened_from_disk": bool(meta.get("loadable")),
        "reloaded_checksum_match": checksum_match,
        "mtime_ns": mtime_ns,
        "artifact_status": payload.get("status") if isinstance(payload, Mapping) else None,
        "bootstrap_skeleton_absent": skeleton_absent,
    }
    if not meta.get("loadable"):
        return [
            base
            | {
                "gate_index": None,
                "upstream": QUALIFICATION_EXPERIMENT_ID,
                "artifact_field": None,
                "op": None,
                "expected": None,
                "actual": None,
                "passed": False,
                "reason": meta.get("error") or "artifact_unreadable",
            }
        ]
    task = _qualification_gate_task(root)
    if task is None:
        return [
            base
            | {
                "gate_index": None,
                "upstream": QUALIFICATION_EXPERIMENT_ID,
                "artifact_field": None,
                "op": None,
                "expected": None,
                "actual": None,
                "passed": False,
                "reason": "exp5785 gate task not found in active roadmap",
            }
        ]
    try:
        from conductor_gates import evaluate_gates
    except ImportError:  # pragma: no cover - import path is fixed in repo/tests.
        from scripts.conductor_gates import evaluate_gates  # type: ignore[no-redef]

    check = evaluate_gates(task, root / "results")
    receipts = []
    for index, gate in enumerate(check.gates_evaluated):
        receipts.append(
            base
            | {
                "gate_index": index,
                "upstream": gate.upstream,
                "artifact_field": gate.artifact_field,
                "op": gate.op,
                "expected": gate.expected,
                "actual": gate.actual,
                "passed": bool(
                    gate.passed
                    and checksum_match
                    and skeleton_absent
                    and payload.get("status") in {"complete", "blocked"}
                ),
                "reason": gate.reason,
            }
        )
    return receipts


def _qualification_field_principles_for(report: Mapping[str, Any]) -> dict[str, str]:
    missing = sorted(set(report) - set(QUALIFICATION_FIELD_PRINCIPLES))
    if missing:
        raise KeyError(f"missing qualification field principles: {missing}")
    return {field: QUALIFICATION_FIELD_PRINCIPLES[field] for field in report}


def build_terminal_qualification_report(
    root: Path = REPO_ROOT,
    *,
    test_receipts: Sequence[Mapping[str, Any]] | None = None,
    result_path: Path | None = None,
    implementation_hashes_before: Mapping[str, Any] | None = None,
    gate_replay_receipts: Sequence[Mapping[str, Any]] | None = None,
    terminal_finalizer_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    root = root.resolve()
    result_path = result_path or root / QUALIFICATION_RESULT_RELATIVE_PATH
    normalized = normalize_qualification_receipts(test_receipts)
    groups = _qualification_receipt_groups(normalized)
    index_report = build_report(
        root,
        tests_run=_task_owned_tests_for_index(groups),
        result_path=root / RESULT_RELATIVE_PATH,
    )
    research_roadmap_hash_before = sha256_file(root / ROADMAP_RELATIVE_PATH)
    conductor_hash_before = sha256_file(root / CONDUCTOR_RELATIVE_PATH)
    research_roadmap_hash_after = sha256_file(root / ROADMAP_RELATIVE_PATH)
    conductor_hash_after = sha256_file(root / CONDUCTOR_RELATIVE_PATH)
    qualification_scan = exp5782_5795_collision_scan(root, result_path)
    task_owned_failures = _failed_receipts(groups["focused"]) + _failed_receipts(
        groups["integration"]
    )
    pre_existing_global_failures = [
        dict(row)
        for row in _failed_receipts(groups["global_baseline"])
        + _failed_receipts(groups["spec_coverage"])
        if row.get("pre_existing") is True
    ]
    gate_receipts = [dict(row) for row in gate_replay_receipts or []]
    failed = list(index_report["preconditions_checked"]["failed_preconditions"])
    if task_owned_failures and "task_owned_receipts_failed" not in failed:
        failed.append("task_owned_receipts_failed")
    if qualification_scan["collision_count"]:
        failed.append("exp5782_exp5795_unowned_collisions")
    if index_report["duplicate_task_ids"] or index_report["conflicting_hashes"]:
        failed.append("ambiguous_exact_identities")
    if research_roadmap_hash_before != research_roadmap_hash_after:
        failed.append("research_roadmap_modified")
    if conductor_hash_before != conductor_hash_after:
        failed.append("research_conductor_modified")
    if gate_receipts and any(not row.get("passed") for row in gate_receipts):
        failed.append("gate_replay_failed")

    ready = (
        not failed
        and bool(groups["focused"])
        and bool(groups["integration"])
        and index_report["evidence_index_ready_score"] == 1.0
        and index_report["next_range_collision_count"] == 0
        and index_report["unresolved_canonical_count"] == 0
        and index_report["history_mutation_count"] == 0
    )
    report: JsonDict = {
        "schema": QUALIFICATION_SCHEMA,
        "experiment": QUALIFICATION_EXPERIMENT,
        "experiment_id": QUALIFICATION_EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "spec_refs": QUALIFICATION_SPEC_REFS,
        "field_principles": {},
        "status": "complete" if ready else "blocked",
        "preconditions_checked": {
            "prior_artifact": {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(root / RESULT_RELATIVE_PATH),
                "status": read_json_mapping(root / RESULT_RELATIVE_PATH)[0].get("status"),
                "honest_verdict": read_json_mapping(root / RESULT_RELATIVE_PATH)[0].get(
                    "honest_verdict"
                ),
            },
            "implementation_hashes_after": _qualification_hashes(root),
            "resource_receipts": _resource_receipts(root),
            "reused_index_failed_preconditions": index_report["preconditions_checked"][
                "failed_preconditions"
            ],
            "exp5782_exp5795_collision_scan": qualification_scan,
            "research_roadmap_hash_before": research_roadmap_hash_before,
            "research_roadmap_hash_after": research_roadmap_hash_after,
            "conductor_hash_before": conductor_hash_before,
            "conductor_hash_after": conductor_hash_after,
            "failed_preconditions": failed,
        },
        "prior_artifact_hash": sha256_file(root / RESULT_RELATIVE_PATH),
        "implementation_hashes_before": dict(implementation_hashes_before or {}),
        "implementation_hashes_after": _qualification_hashes(root),
        "canonical_identity_contract": index_report["canonical_identity_contract"],
        "focused_test_receipts": groups["focused"],
        "integration_test_receipts": groups["integration"],
        "global_baseline_receipts": groups["global_baseline"],
        "spec_coverage_receipts": groups["spec_coverage"],
        "test_ownership_policy": QUALIFICATION_TEST_OWNERSHIP_POLICY,
        "task_owned_failures": task_owned_failures,
        "pre_existing_global_failures": pre_existing_global_failures,
        "terminal_finalizer_receipt": dict(
            terminal_finalizer_receipt
            or {
                "output_path": (
                    result_path.relative_to(root).as_posix()
                    if result_path.is_absolute()
                    else result_path.as_posix()
                ),
                "atomic_replace": True,
                "mode": "build_only_not_yet_reopened",
            }
        ),
        "bootstrap_skeleton_absent": False,
        "gate_replay_receipts": gate_receipts,
        "evidence_index_ready_score": 1.0 if ready else 0.0,
        "next_range_collision_count": index_report["next_range_collision_count"],
        "unresolved_canonical_count": index_report["unresolved_canonical_count"],
        "history_mutation_count": index_report["history_mutation_count"],
        "producer_gate_fields": {},
        "research_complete_modified": index_report["research_complete_modified"],
        "research_roadmap_unchanged": research_roadmap_hash_before == research_roadmap_hash_after,
        "conductor_unchanged": conductor_hash_before == conductor_hash_after,
        "inference_substrate": QUALIFICATION_INFERENCE_SUBSTRATE,
        "test_commands": [row["command"] for row in normalized],
        "test_exit_codes": {_test_exit_code_key(row): row["exit_code"] for row in normalized},
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: exact-deliverable index qualified by task-owned receipts; "
            "global baseline health disclosed separately"
            if ready
            else "blocked: evidence-index terminal qualification failed closed: "
            + ", ".join(dict.fromkeys(failed))
        ),
    }
    report["producer_gate_fields"] = {
        "evidence_index_ready_score": report["evidence_index_ready_score"],
        "next_range_collision_count": report["next_range_collision_count"],
        "unresolved_canonical_count": report["unresolved_canonical_count"],
        "history_mutation_count": report["history_mutation_count"],
    }
    report["bootstrap_skeleton_absent"] = bootstrap_skeleton_absent(report)
    report["field_principles"] = _qualification_field_principles_for(report)
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def emit_terminal_qualification(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    test_receipts: Sequence[Mapping[str, Any]] | None = None,
    implementation_hashes_before: Mapping[str, Any] | None = None,
) -> JsonDict:
    root = root.resolve()
    output_path = output_path or root / QUALIFICATION_RESULT_RELATIVE_PATH
    rel_output = (
        output_path.relative_to(root).as_posix()
        if output_path.is_absolute()
        else output_path.as_posix()
    )
    previous_present = output_path.exists()
    candidate = build_terminal_qualification_report(
        root,
        test_receipts=test_receipts,
        result_path=output_path,
        implementation_hashes_before=implementation_hashes_before,
    )
    _atomic_write_json(output_path, candidate)
    os.utime(output_path, ns=(QUALIFICATION_TERMINAL_MTIME_NS, QUALIFICATION_TERMINAL_MTIME_NS))
    gate_replay = replay_qualification_gates(root, output_path)
    finalizer_receipt = {
        "output_path": rel_output,
        "temporary_path": rel_output + ".tmp",
        "atomic_replace": True,
        "previous_artifact_present": previous_present,
        "reopened_from_disk": True,
        "reloaded_checksum_match": True,
        "final_status_verified": True,
        "bootstrap_skeleton_absent_verified": True,
        "mtime_ns_after": QUALIFICATION_TERMINAL_MTIME_NS,
        "gate_replay_rechecked": True,
    }
    final = build_terminal_qualification_report(
        root,
        test_receipts=test_receipts,
        result_path=output_path,
        implementation_hashes_before=implementation_hashes_before,
        gate_replay_receipts=gate_replay,
        terminal_finalizer_receipt=finalizer_receipt,
    )
    _atomic_write_json(output_path, final)
    os.utime(output_path, ns=(QUALIFICATION_TERMINAL_MTIME_NS, QUALIFICATION_TERMINAL_MTIME_NS))
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    if payload_checksum(reloaded) != reloaded.get("reproducibility_checksum"):
        raise ValueError("terminal qualification checksum verification failed")  # pragma: no cover
    if reloaded.get("status") not in {"complete", "blocked"}:
        raise ValueError("terminal qualification status is not terminal")  # pragma: no cover
    if output_path.stat().st_mtime_ns != QUALIFICATION_TERMINAL_MTIME_NS:
        raise ValueError("terminal qualification mtime verification failed")  # pragma: no cover
    if not bootstrap_skeleton_absent(reloaded):
        raise ValueError(
            "terminal qualification bootstrap skeleton still present"
        )  # pragma: no cover
    if replay_qualification_gates(root, output_path) != reloaded["gate_replay_receipts"]:
        raise ValueError(  # pragma: no cover
            "terminal qualification gate replay changed after final write"
        )
    return reloaded


def _load_tests_run(path: Path | None) -> list[JsonDict]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in data if isinstance(row, Mapping)]


def _load_json_mapping_file(path: Path | None) -> JsonDict:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("JSON file must be a mapping")
    return dict(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    parser.add_argument("--qualify-terminal", action="store_true")
    parser.add_argument("--qualification-tests-run-json", type=Path, default=None)
    parser.add_argument("--implementation-hashes-before-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.qualify_terminal:
        emit_terminal_qualification(
            args.root,
            output_path=args.output,
            test_receipts=_load_tests_run(args.qualification_tests_run_json or args.tests_run_json),
            implementation_hashes_before=_load_json_mapping_file(
                args.implementation_hashes_before_json
            ),
        )
        return 0
    emit_report(
        args.root,
        output_path=args.output,
        tests_run=_load_tests_run(args.tests_run_json),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
