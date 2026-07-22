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
SCHEMA = "carnot.exp5771.evidence_index_collision_preflight.v1"
INFERENCE_SUBSTRATE = "local_filesystem_metadata_and_hashes_no_llm"
CANONICAL_POLICY = "exact_declared_deliverable"
NEXT_RANGE = range(5769, 5782)
NON_ARTIFACT_OUTCOMES = {"GATE_BLOCK", "SKIP", "FAIL", "BLOCK"}

SPEC_REFS = [
    "REQ-REPORT-5771",
    "SCENARIO-REPORT-5771-EXACT-LOOKUP",
    "SCENARIO-REPORT-5771-FAIL-CLOSED",
    "SCENARIO-REPORT-5771-HISTORY-READONLY",
    "SCENARIO-REPORT-5771-FIELD-PRINCIPLES",
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


def _load_tests_run(path: Path | None) -> list[JsonDict]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in data if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(
        args.root,
        output_path=args.output,
        tests_run=_load_tests_run(args.tests_run_json),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
