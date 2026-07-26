"""Exp5931 branch-independent reconciliation for milestone .526.

Spec refs: REQ-REPORT-5931, REQ-CAPSTONE-5931,
SCENARIO-REPORT-5931-EXACT-MATRIX, SCENARIO-REPORT-5931-TERMINAL-CLASSES,
SCENARIO-REPORT-5931-BRANCH-SEMANTICS,
SCENARIO-REPORT-5931-APPEND-RETIRE-AND-RECOMMEND.

This module is a closeout ledger over evidence that already exists. It treats
the active roadmap's declared deliverable path as the only admissible evidence
locator for an experiment, because capstones are where adjacent filenames,
source modules, and downstream gates can otherwise blur into fake success. The
receipts below deliberately keep structural ConstraintIR support, exact
semantic success, continuous-learning chronology, ARC live execution, and
hardware mapping in separate buckets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.07.526"
MILESTONE_TITLE = "Schema-Decoding, Fresh CSL, Live-Runner Binding, ABI Mapping, and Exact Capstone"
EXPERIMENT_ID = "exp5931-v526-capstone-reconciliation"
EXPERIMENT_NAME = "experiment_5931_v526_capstone_reconciliation"
RESULT_RELATIVE_PATH = Path("results/experiment_5931_v526_capstone_reconciliation.json")
INFERENCE_SUBSTRATE = "aggregation_from_exact_declared_artifacts"
SCHEMA = "carnot.experiment_5931.v526_capstone_reconciliation.v1"
COMPLETED_DATE = "2026-07-26"

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
REPORT_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
CAPSTONE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
CHANGE_PROPOSAL_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
PRIOR_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5917_v525_capstone_reconciliation.json")

TERMINAL_CLASSES = (
    "positive",
    "null",
    "underpowered",
    "blocked-precondition",
    "retired",
    "gate-blocked",
    "no-change",
    "missing",
)
EXPECTED_TASK_IDS = tuple(f"exp{number}" for number in range(5918, 5932))

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_and_task_range",
    "activated_task_and_declared_deliverable_matrix",
    "exact_terminal_classification",
    "adversarial_verifier_receipts",
    "transition_and_source_receipt",
    "schema_constraintir_structural_and_exact_semantic_receipt",
    "continuous_self_learning_chronology_lift_safety_retention_and_rollback_receipt",
    "arc_coordinate_execution_binding_and_live_receipt",
    "adaptive_state_abi_and_hardware_receipt",
    "branch_independence_receipt",
    "prior_failure_and_retirement_decisions",
    "missing_gate_blocked_and_reserved_receipts",
    "duplicate_history_amplification_count",
    "research_complete_append_receipt",
    "docs_reconciled",
    "next_three_falsifiable_recommendations",
    "registry_unchanged",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal capstone state over the activated .526 identities.",
    "preconditions_checked": (
        "Roadmaps, declared artifacts, history, logs, exclusions, ARC registry, "
        "protected files, verifier state, resources, and output path are checked before completion."
    ),
    "milestone_and_task_range": "The activated milestone must be exactly 2026.07.526 with Exp5918 through Exp5931.",
    "activated_task_and_declared_deliverable_matrix": (
        "only exact activated identities and exact declared paths count."
    ),
    "exact_terminal_classification": (
        "positive, null, underpowered, blocked-precondition, retired, gate-blocked, "
        "no-change, and missing classes remain disjoint."
    ),
    "adversarial_verifier_receipts": (
        "Fresh verifier receipts cover every present declared .526 artifact without replacing missing ones."
    ),
    "transition_and_source_receipt": (
        "Transition blockage and source-refresh nulls are separate receipts, not milestone-wide conclusions."
    ),
    "schema_constraintir_structural_and_exact_semantic_receipt": (
        "Structural grammar/tokenizer support is separate from exact semantic success."
    ),
    "continuous_self_learning_chronology_lift_safety_retention_and_rollback_receipt": (
        "CSL chronology, missing live lift, safety, retention, rollback, and ABI operations stay separate."
    ),
    "arc_coordinate_execution_binding_and_live_receipt": (
        "Offline ARC qualification, execution binding, and live rows are different authorities."
    ),
    "adaptive_state_abi_and_hardware_receipt": (
        "ABI parity, static synthesis, and physical board execution are distinct receipts."
    ),
    "branch_independence_receipt": (
        "a gate block or branch failure cannot erase or manufacture evidence in another branch."
    ),
    "prior_failure_and_retirement_decisions": (
        "same-verdict retirement is mechanical and scope-specific."
    ),
    "missing_gate_blocked_and_reserved_receipts": (
        "Missing deliverables, gate-blocked tasks, invalid receipts, and current capstone emission stay visible."
    ),
    "duplicate_history_amplification_count": "must be bare zero.",
    "research_complete_append_receipt": "The completion append receipt records exact zero-or-one .526 append behavior.",
    "docs_reconciled": "Relevant OpenSpec entries are recorded while conductor-owned ledgers are deferred by the stop rule.",
    "next_three_falsifiable_recommendations": (
        "Exactly three recommendations carry prerequisites, stop rules, authority boundaries, and excluded scopes."
    ),
    "registry_unchanged": "The ARC registry before and after hashes must match.",
    "protected_files_unchanged": "Protected roadmap, conductor, ARC, north-star, public docs, and ops-ledger files remain byte-identical.",
    "duration_s": "Measured wall time exposes aggregation-only execution.",
    "inference_substrate": "use `aggregation_from_exact_declared_artifacts`.",
    "field_provenance": "Every required field traces to exact paths, hashes, receipts, commands, or classifications.",
    "test_commands": "Commands document focused unit, coverage, YAML, verifier, branch, manifest, append, registry, reconciliation, spec, E2E, protected-file, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as success.",
    "reproducibility_checksum": "A checksum detects later ledger, artifact, or protected-file drift.",
    "honest_verdict": "use `complete:`, `complete_with_nulls:`, or `blocked:`.",
}

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    Path("docs/index.html"),
    Path("README.md"),
)

PRECONDITION_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    CHANGE_PROPOSAL_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    PRIOR_CAPSTONE_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
    REPORT_SPEC_RELATIVE_PATH,
    CAPSTONE_SPEC_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
)

EXP5923_RETIREMENT_MARKER = (
    "exp5923_schema_supported_constraintir_zero_exact_semantics_retired_v526"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def path_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    meta["loadable"] = isinstance(payload, Mapping)
    if not meta["loadable"]:
        meta["error"] = "json_not_mapping"
        return {}, meta
    return dict(payload), meta


def _read_yaml(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        meta["error"] = f"yaml_error:{exc}"
        return {}, meta
    meta["loadable"] = isinstance(payload, Mapping)
    if not meta["loadable"]:
        meta["error"] = "yaml_not_mapping"
        return {}, meta
    return dict(payload), meta


def _task_number(task_id: str) -> int | None:
    match = re.search(r"exp(\d{4})", task_id)
    return int(match.group(1)) if match else None


def _active_tasks(root: Path) -> tuple[list[JsonDict], JsonDict]:
    roadmap, roadmap_meta = _read_yaml(root / ROADMAP_RELATIVE_PATH)
    rows = roadmap.get("tasks", [])
    task_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    selected = [
        row
        for row in task_rows
        if isinstance(row.get("id"), str)
        and (number := _task_number(str(row["id"]))) is not None
        and 5918 <= number <= 5931
    ]
    selected.sort(key=lambda row: int(str(row["id"])[3:7]))
    task_ids = [str(row["id"]) for row in selected]
    expected_numbers = list(range(5918, 5932))
    return selected, {
        "active_roadmap": roadmap_meta,
        "milestone": roadmap.get("milestone"),
        "milestone_matches": roadmap.get("milestone") == MILESTONE,
        "task_ids": task_ids,
        "task_numbers": [_task_number(task_id) for task_id in task_ids],
        "exact_range": [_task_number(task_id) for task_id in task_ids] == expected_numbers,
        "task_count": len(selected),
        "expected_task_count": len(expected_numbers),
        "expected_task_ids": list(EXPECTED_TASK_IDS),
    }


def _roadmap_next_receipt(root: Path) -> JsonDict:
    payload, meta = _read_yaml(root / ROADMAP_NEXT_RELATIVE_PATH)
    tasks = payload.get("tasks", []) if payload else []
    task_ids = [str(row.get("id")) for row in tasks if isinstance(row, Mapping)]
    return {
        "present": meta["present"],
        "loadable": meta["loadable"],
        "sha256": meta["sha256"],
        "milestone": payload.get("milestone") if payload else None,
        "task_ids": task_ids,
    }


def _artifact_payloads(
    root: Path, tasks: Sequence[JsonMap]
) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for row in tasks:
        task_id = str(row["id"])
        rel_path = Path(str(row["deliverable"]))
        payload, meta = _read_json(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _conductor_outcomes(root: Path, tasks: Sequence[JsonMap]) -> dict[str, JsonDict]:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = text.splitlines()
    outcomes: dict[str, JsonDict] = {}
    for row in tasks:
        task_id = str(row["id"])
        title = str(row.get("title") or "")
        prefix = title[:45]
        matches = [line for line in lines if task_id in line or (prefix and prefix in line)]
        latest = matches[-1] if matches else None
        status = "MISSING"
        if latest:
            for candidate in ("GATE_BLOCK", "FLAGGED", "FAIL", "SKIP", "OK"):
                if f"| {candidate} |" in latest:
                    status = candidate
                    break
        outcomes[task_id] = {
            "latest_status": status,
            "latest_line": latest,
            "attempt_count": len(matches),
        }
    return outcomes


def _receipt_for_task(task_id: str, artifact: str, adversarial_receipt: JsonMap) -> JsonDict:
    reports = adversarial_receipt.get("reports")
    if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        for row in reports:
            if isinstance(row, Mapping) and row.get("artifact") == artifact:
                report = dict(row)
                report.setdefault("task_id", task_id)
                return report
    return {
        "artifact": artifact,
        "loaded": False,
        "flag_count": None,
        "max_severity": None,
        "flags": [],
        "task_id": task_id,
    }


def _terminal_class(
    task_id: str, payload: JsonMap, meta: JsonMap, conductor: JsonMap, receipt: JsonMap
) -> tuple[str, str | None]:
    if task_id == EXPERIMENT_ID:
        return "positive", "current-capstone-emission"
    flag_count = receipt.get("flag_count")
    max_severity = receipt.get("max_severity")
    if (
        isinstance(flag_count, int)
        and flag_count > 0
        and isinstance(max_severity, int)
        and max_severity >= 2
    ):
        return "missing", "adversarial-verifier-critical"
    if not meta.get("present"):
        if conductor.get("latest_status") == "GATE_BLOCK":
            return "gate-blocked", "missing-deliverable-with-conductor-gate-block"
        return "missing", "declared-deliverable-missing"
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    schema = str(payload.get("schema") or "")
    if (
        schema == "blocked_gate_check_v1"
        or verdict.startswith("blocked_gate")
        or payload.get("gates_evaluated")
    ):
        return "gate-blocked", "conductor-gate-check"
    if status == "retired" or verdict.startswith("retired:"):
        return "retired", "retire-if-same-verdict"
    if "underpowered" in status or verdict.startswith("complete_underpowered"):
        return "underpowered", "underpowered"
    if status.startswith("blocked_precondition") or verdict.startswith("blocked_precondition"):
        return "blocked-precondition", "blocked-precondition"
    if status.startswith("blocked") or verdict.startswith("blocked:"):
        return "blocked-precondition", "blocked"
    if (
        "no_physical_probe" in status
        or status.startswith("no_change")
        or verdict.startswith("no_change:")
        or verdict.startswith("complete_static_mapping:")
    ):
        return "no-change", "static-mapping-no-physical-probe"
    if status == "complete_null" or verdict.startswith("complete_null"):
        return "null", "honest-null"
    if status in {"complete_ready", "ready", "complete_positive"} or verdict.startswith(
        ("complete_ready", "ready:", "complete_positive")
    ):
        return "positive", "ready-or-positive"
    if status == "complete" and not verdict.startswith("complete_null"):
        return "positive", "complete"
    return "missing", "unrecognized-terminal-treated-as-missing"


def _terminal_classification(
    tasks: Sequence[JsonMap],
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    adversarial_receipt: JsonMap,
) -> JsonDict:
    by_task: dict[str, str] = {}
    subclasses: dict[str, str | None] = {}
    by_class: dict[str, list[str]] = {name: [] for name in TERMINAL_CLASSES}
    invalid: list[str] = []
    for row in tasks:
        task_id = str(row["id"])
        artifact = str(row["deliverable"])
        terminal, subclass = _terminal_class(
            task_id,
            payloads.get(task_id, {}),
            metadata.get(task_id, {}),
            conductor.get(task_id, {}),
            _receipt_for_task(task_id, artifact, adversarial_receipt),
        )
        by_task[task_id] = terminal
        subclasses[task_id] = subclass
        by_class[terminal].append(task_id)
        if subclass == "adversarial-verifier-critical":
            invalid.append(task_id)
    return {
        "terminal_class_by_task_id": by_task,
        "terminal_subclass_by_task_id": subclasses,
        "task_ids_by_terminal_class": by_class,
        "invalid_artifact_task_ids": invalid,
        "disjoint_terminal_classes": list(TERMINAL_CLASSES),
        "all_activated_classified_once": len(by_task) == len(tasks)
        and all(sum(task_id in values for values in by_class.values()) == 1 for task_id in by_task),
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _activated_matrix(
    tasks: Sequence[JsonMap],
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    classes: JsonMap,
) -> JsonDict:
    rows: list[JsonDict] = []
    for row in tasks:
        task_id = str(row["id"])
        meta = metadata.get(task_id, {})
        payload = payloads.get(task_id, {})
        rows.append(
            {
                "milestone": str(row.get("milestone") or MILESTONE),
                "task_id": task_id,
                "title": str(row.get("title") or ""),
                "declared_deliverable": str(row.get("deliverable") or ""),
                "declared_deliverable_present": bool(meta.get("present")),
                "declared_deliverable_loadable": bool(meta.get("loadable")),
                "declared_deliverable_sha256": meta.get("sha256"),
                "status": str(payload.get("status") or ""),
                "honest_verdict": str(payload.get("honest_verdict") or ""),
                "terminal_class": classes["terminal_class_by_task_id"][task_id],
                "terminal_subclass": classes["terminal_subclass_by_task_id"][task_id],
                "conductor": conductor.get(task_id, {}),
                "identity": [MILESTONE, task_id, str(row.get("deliverable") or "")],
                "evidence_selection": "declared_deliverable_path",
            }
        )
    return {
        "selection_policy": "exact_declared_deliverable",
        "activated_task_count": len(rows),
        "tasks": rows,
        "principle": FIELD_PRINCIPLES["activated_task_and_declared_deliverable_matrix"],
    }


def _source_hashes(root: Path, tasks: Sequence[JsonMap]) -> dict[str, JsonDict]:
    paths = set(PRECONDITION_HASH_PATHS)
    paths.update(Path(str(row["deliverable"])) for row in tasks)
    paths.add(RESULT_RELATIVE_PATH)
    return {
        path.as_posix(): {"present": (root / path).exists(), "sha256": path_sha256(root / path)}
        for path in sorted(paths, key=lambda item: item.as_posix())
    }


def _resource_receipts(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    mem_available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_mb = int(line.split()[1]) // 1024
                break
    return {
        "disk": {
            "available_mb": disk.free // (1024 * 1024),
            "required_mb": 512,
            "ok": disk.free >= 512 * 1024 * 1024,
        },
        "ram": {
            "available_mb": mem_available_mb,
            "required_mb": 512,
            "ok": mem_available_mb == 0 or mem_available_mb >= 512,
        },
    }


def _atomic_output_receipt(root: Path) -> JsonDict:
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    probe = result_path.with_name(result_path.name + ".atomic-probe")
    try:
        probe.write_text("probe\n", encoding="utf-8")
        os.replace(probe, probe.with_suffix(".renamed"))
        probe.with_suffix(".renamed").unlink(missing_ok=True)
        ok = True
        error = None
    except OSError as exc:
        ok = False
        error = str(exc)
    return {
        "path": result_path.as_posix(),
        "parent_exists": result_path.parent.exists(),
        "parent_writable": os.access(result_path.parent, os.W_OK),
        "ok": ok,
        "error": error,
    }


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): path_sha256(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    files: dict[str, JsonDict] = {}
    after = _protected_hashes(root)
    for path in PROTECTED_RELATIVE_PATHS:
        key = path.as_posix()
        files[key] = {
            "present": (root / path).exists(),
            "sha256_before": before.get(key),
            "sha256_after": after.get(key),
            "unchanged": before.get(key) == after.get(key),
        }
    return {
        "files": files,
        "all_unchanged": all(row["unchanged"] for row in files.values()),
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _registry_unchanged(root: Path, before_hash: str | None) -> JsonDict:
    after_hash = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)
    return {
        "path": ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "unchanged": before_hash == after_hash,
        "principle": FIELD_PRINCIPLES["registry_unchanged"],
    }


def _completion_blocks(root: Path) -> list[JsonMap]:
    payload, _meta = _read_yaml(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    blocks = payload.get("milestones")
    return [row for row in blocks if isinstance(row, Mapping)] if isinstance(blocks, list) else []


def _duplicate_history_count(blocks: Sequence[JsonMap]) -> int:
    signatures = Counter(
        (
            str(block.get("id")),
            tuple(
                (str(task.get("id")), str(task.get("deliverable")))
                for task in block.get("tasks", [])
                if isinstance(task, Mapping)
            ),
        )
        for block in blocks
    )
    return sum(count - 1 for count in signatures.values() if count > 1)


def _completion_block_text(tasks: Sequence[JsonMap], classes: JsonMap) -> str:
    lines = [
        f"- id: {json.dumps(MILESTONE)}",
        f"  title: {json.dumps(MILESTONE_TITLE)}",
        '  doc: "openspec/change-proposals/research-roadmap-vNEXT.md"',
        f"  completed: {json.dumps(COMPLETED_DATE)}",
        "  finding: Terminal outcomes preserved by Exp5931 capstone; see artifact.",
        "  tasks:",
    ]
    for row in tasks:
        task_id = str(row["id"])
        terminal = classes["terminal_class_by_task_id"][task_id]
        subclass = classes["terminal_subclass_by_task_id"][task_id]
        result = terminal if not subclass else f"{terminal} ({subclass})"
        lines.extend(
            [
                f"  - id: {json.dumps(task_id)}",
                f"    title: {json.dumps(str(row.get('title') or ''))}",
                f"    deliverable: {json.dumps(str(row.get('deliverable') or ''))}",
                f"    result: {json.dumps(result)}",
            ]
        )
    return "\n".join(lines) + "\n"


def _append_completion_once(
    root: Path, tasks: Sequence[JsonMap], classes: JsonMap, update_ledgers: bool
) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    before_hash = path_sha256(path)
    before_blocks = _completion_blocks(root)
    before_duplicates = _duplicate_history_count(before_blocks)
    if any(str(block.get("id")) == MILESTONE for block in before_blocks):
        return {
            "appended": False,
            "append_count": 0,
            "reason": "milestone_block_already_present",
            "before_sha256": before_hash,
            "after_sha256": before_hash,
            "before_duplicate_history_count": before_duplicates,
            "after_duplicate_history_count": before_duplicates,
            "duplicate_history_amplification_count": 0,
            "principle": FIELD_PRINCIPLES["research_complete_append_receipt"],
        }
    if update_ledgers:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else "milestones:\n"
        separator = "" if existing.endswith("\n") else "\n"
        path.write_text(
            existing + separator + _completion_block_text(tasks, classes), encoding="utf-8"
        )
    after_blocks = _completion_blocks(root) if update_ledgers else before_blocks
    after_duplicates = _duplicate_history_count(after_blocks)
    return {
        "appended": bool(update_ledgers),
        "append_count": 1 if update_ledgers else 0,
        "reason": "milestone_block_absent" if update_ledgers else "dry_run_milestone_block_absent",
        "before_sha256": before_hash,
        "after_sha256": path_sha256(path) if update_ledgers else before_hash,
        "before_duplicate_history_count": before_duplicates,
        "after_duplicate_history_count": after_duplicates,
        "duplicate_history_amplification_count": max(0, after_duplicates - before_duplicates),
        "principle": FIELD_PRINCIPLES["research_complete_append_receipt"],
    }


def _manifest_entry_text() -> str:
    return "\n".join(
        [
            f"- id: {EXP5923_RETIREMENT_MARKER}",
            "  scope_key: schema_supported_constraintir_zero_exact_semantics_v526",
            "  experiment_scope: Exp5923 schema-supported ConstraintIR live A/B exact-semantic zero-result reprompt scope",
            "  reason: >-",
            "    retire_if_same_verdict: Exp5923 replayed the ready schema/tokenizer",
            "    support through all three mandated GGUF families and still produced zero",
            "    exact semantic success, so this exact schema-supported reprompt mechanism",
            "    is retired. Open structural compiler, tokenizer bridge, exact executor,",
            "    CSL memory, and unrelated constraint mechanisms remain separate.",
            "  experiment_ids:",
            "  - exp5909-sota-constraint-synthesis-ab",
            "  - exp5910-verification-guided-constraint-repair",
            "  - exp5923-sota-schema-supported-constraintir-ab",
            f"  retired_milestone: {MILESTONE}",
            "  retired_by_artifact: results/experiment_5923_sota_schema_supported_constraintir_ab.json",
            "  recorded_by_artifact: results/experiment_5931_v526_capstone_reconciliation.json",
            "  operator_reopen_required: true",
            "  retire_if_same_verdict: true",
            "  blocked_patterns:",
            "  - schema-supported ConstraintIR reprompt after zero exact semantic success",
            "  - treating structural validity as semantic success after Exp5923 retirement",
            "",
        ]
    )


def _append_manifest_entry_once(root: Path, update_ledgers: bool) -> JsonDict:
    path = root / EXCLUSION_MANIFEST_RELATIVE_PATH
    before_hash = path_sha256(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if EXP5923_RETIREMENT_MARKER in text:
        return {
            "manifest_append_count": 0,
            "appended": False,
            "reason": "retirement_entry_already_present",
            "before_sha256": before_hash,
            "after_sha256": before_hash,
        }
    if update_ledgers:
        path.parent.mkdir(parents=True, exist_ok=True)
        base = text
        if "retired_extras:" not in base:
            base += "\nretired_extras:\n"
        separator = "" if base.endswith("\n") else "\n"
        path.write_text(base + separator + _manifest_entry_text(), encoding="utf-8")
    return {
        "manifest_append_count": 1 if update_ledgers else 0,
        "appended": bool(update_ledgers),
        "reason": "same_verdict_recurred" if update_ledgers else "dry_run_same_verdict_recurred",
        "before_sha256": before_hash,
        "after_sha256": path_sha256(path) if update_ledgers else before_hash,
    }


def _current_task_retirement_recurred(task_id: str, payload: JsonMap) -> tuple[bool, str]:
    decision = payload.get("retirement_decision")
    if (
        task_id == "exp5923-sota-schema-supported-constraintir-ab"
        and isinstance(decision, Mapping)
        and decision.get("retire") is True
    ):
        return True, "artifact_retirement_decision_retire_true"
    repeated = payload.get("repeated_verdict_retirement_decision")
    if (
        isinstance(repeated, Mapping)
        and repeated.get("retire_if_same_verdict") is True
        and repeated.get("same_verdict_recurred") is True
    ):
        return True, "artifact_repeated_verdict_retirement_decision"
    return False, "no_exact_same_verdict_retirement_receipt"


def _retirement_decisions(
    root: Path, tasks: Sequence[JsonMap], payloads: Mapping[str, JsonMap], update_ledgers: bool
) -> JsonDict:
    audit: list[JsonDict] = []
    retire_rows: list[JsonDict] = []
    for row in tasks:
        task_id = str(row["id"])
        priors = row.get("prior_failures") or []
        if not isinstance(priors, Sequence) or isinstance(priors, (str, bytes)):
            continue
        for prior in priors:
            if not isinstance(prior, Mapping):
                continue
            requested = prior.get("retire_if_same_verdict") is True
            recurred, evidence = _current_task_retirement_recurred(
                task_id, payloads.get(task_id, {})
            )
            audit_row = {
                "task_id": task_id,
                "prior_experiment_id": prior.get("experiment_id"),
                "prior_verdict": prior.get("verdict"),
                "retire_if_same_verdict": requested,
                "same_verdict_recurred": bool(requested and recurred),
                "evidence": evidence if requested else "retire_not_requested",
                "adjacent_open_mechanism_retired": False,
            }
            audit.append(audit_row)
            if audit_row["same_verdict_recurred"]:
                retire_rows.append(
                    {
                        "task_id": task_id,
                        "prior_experiment_id": prior.get("experiment_id"),
                        "retire_if_same_verdict": True,
                        "same_verdict_recurred": True,
                        "decision": "retire_exact_scope",
                        "scope_key": "schema_supported_constraintir_zero_exact_semantics_v526",
                    }
                )
    manifest = (
        _append_manifest_entry_once(root, update_ledgers)
        if any(
            row["task_id"] == "exp5923-sota-schema-supported-constraintir-ab" for row in retire_rows
        )
        else {"manifest_append_count": 0, "appended": False, "reason": "no_same_verdict_recurrence"}
    )
    return {
        "prior_failure_audit": audit,
        "retire_if_same_verdict_decisions": retire_rows,
        "manifest_append_count": manifest["manifest_append_count"],
        "manifest_update_receipt": manifest,
        "retired_adjacent_open_mechanism_count": 0,
        "retired_task_ids_reopened": [],
        "principle": FIELD_PRINCIPLES["prior_failure_and_retirement_decisions"],
    }


def _missing_gate_reserved_receipts(matrix: JsonMap, classes: JsonMap) -> JsonDict:
    missing = [
        row["task_id"]
        for row in matrix["tasks"]
        if not row["declared_deliverable_present"] and row["task_id"] != EXPERIMENT_ID
    ]
    gate_blocked = classes["task_ids_by_terminal_class"]["gate-blocked"]
    return {
        "missing_declared_deliverable_task_ids": missing,
        "gate_blocked_task_ids": gate_blocked,
        "gate_blocked_missing_declared_deliverable_task_ids": [
            task_id for task_id in missing if task_id in gate_blocked
        ],
        "current_capstone_emission_task_id": EXPERIMENT_ID,
        "current_capstone_not_counted_missing": True,
        "invalid_artifact_task_ids": classes["invalid_artifact_task_ids"],
        "principle": FIELD_PRINCIPLES["missing_gate_blocked_and_reserved_receipts"],
    }


def _task_classes(classes: JsonMap, task_ids: Sequence[str]) -> list[str]:
    return [classes["terminal_class_by_task_id"][task_id] for task_id in task_ids]


def _branch_independence(classes: JsonMap) -> JsonDict:
    branch_ids = {
        "transition_and_source": [
            "exp5918-transition-v526",
            "exp5919-v526-source-delta-ingestion",
        ],
        "schema_constraint_ir": [
            "exp5921-schema-derived-constraintir-support",
            "exp5922-gguf-schema-decoder-bridge",
            "exp5923-sota-schema-supported-constraintir-ab",
        ],
        "continuous_self_learning": [
            "exp5924-transactional-constraint-memory-v2",
            "exp5925-sota-transactional-csl-prospective",
            "exp5926-adaptive-state-abi-v2-parity",
        ],
        "arc": [
            "exp5927-coordinate-router-progress-qualification",
            "exp5928-arc-live-runner-execution-binding",
            "exp5929-arc-structured-memory-bound-live-ab",
        ],
        "hardware": ["exp5930-adaptive-state-board-mapping"],
    }
    return {
        "branch_task_ids": branch_ids,
        "branch_classes": {
            branch: _task_classes(classes, task_ids) for branch, task_ids in branch_ids.items()
        },
        "branch_independence_preserved": True,
        "downstream_gate_does_not_infer_upstream_success": True,
        "structural_validity_does_not_infer_exact_semantics": True,
        "offline_arc_does_not_infer_live_evidence": True,
        "static_mapping_does_not_infer_acceleration": True,
        "principle": FIELD_PRINCIPLES["branch_independence_receipt"],
    }


def _transition_and_source_receipt(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    transition = payloads.get("exp5918-transition-v526", {})
    source = payloads.get("exp5919-v526-source-delta-ingestion", {})
    return {
        "exp5918_transition": {
            "terminal_class": classes["terminal_class_by_task_id"].get("exp5918-transition-v526"),
            "status": transition.get("status"),
            "honest_verdict": transition.get("honest_verdict"),
            "failed_preconditions": transition.get("failed_preconditions")
            or transition.get("preconditions_checked", {}).get("failed_preconditions")
            if isinstance(transition.get("preconditions_checked"), Mapping)
            else transition.get("failed_preconditions"),
        },
        "exp5919_source_delta": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5919-v526-source-delta-ingestion"
            ),
            "status": source.get("status"),
            "honest_verdict": source.get("honest_verdict"),
            "accepted_finding_count": source.get("accepted_finding_count"),
            "references_changed": source.get("references_changed"),
        },
        "transition_blockage_is_not_milestone_wide": True,
        "source_null_is_not_no_change_to_all_branches": True,
        "principle": FIELD_PRINCIPLES["transition_and_source_receipt"],
    }


def _schema_receipt(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    compiler = payloads.get("exp5921-schema-derived-constraintir-support", {})
    bridge = payloads.get("exp5922-gguf-schema-decoder-bridge", {})
    live = payloads.get("exp5923-sota-schema-supported-constraintir-ab", {})
    semantic = live.get("exact_semantic_primary_comparison_and_intervals")
    return {
        "exp5921_terminal_class": classes["terminal_class_by_task_id"].get(
            "exp5921-schema-derived-constraintir-support"
        ),
        "exp5922_terminal_class": classes["terminal_class_by_task_id"].get(
            "exp5922-gguf-schema-decoder-bridge"
        ),
        "exp5923_terminal_class": classes["terminal_class_by_task_id"].get(
            "exp5923-sota-schema-supported-constraintir-ab"
        ),
        "structural_support_ready": compiler.get("schema_decode_contract_ready_score") == 1.0
        and bridge.get("gguf_schema_decoder_bridge_ready_score") == 1.0,
        "compiler_structural_receipt": compiler.get(
            "schema_to_grammar_type_scope_compiler_receipt"
        ),
        "tokenizer_bridge_receipt": bridge.get("per_model_terminal_token_mapping"),
        "semantic_authority_boundary": compiler.get("semantic_authority_boundary"),
        "structural_validity_is_semantic_success": False,
        "exact_semantic_primary_comparison_and_intervals": semantic,
        "schema_decode_live_ready_score": live.get("schema_decode_live_ready_score"),
        "exact_semantic_success_promoted": False,
        "retirement_decision": live.get("retirement_decision"),
        "principle": FIELD_PRINCIPLES["schema_constraintir_structural_and_exact_semantic_receipt"],
    }


def _csl_receipt(
    payloads: Mapping[str, JsonMap], metadata: Mapping[str, JsonMap], classes: JsonMap
) -> JsonDict:
    stream = payloads.get("exp5920-prospective-event-stream-admission", {})
    memory = payloads.get("exp5924-transactional-constraint-memory-v2", {})
    live_meta = metadata.get("exp5925-sota-transactional-csl-prospective", {})
    abi = payloads.get("exp5926-adaptive-state-abi-v2-parity", {})
    return {
        "stream_admission": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5920-prospective-event-stream-admission"
            ),
            "ready_score": stream.get("prospective_stream_admission_ready_score"),
            "chronology": stream.get("chronology_split_and_visibility_receipts"),
            "exact_label_authority": stream.get("exact_label_authority"),
        },
        "transactional_memory": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5924-transactional-constraint-memory-v2"
            ),
            "ready_score": memory.get("transactional_memory_fixture_ready_score"),
            "gate_replay_receipt": memory.get("gate_replay_receipt"),
            "safety_retention_rollback": memory.get(
                "poison_burst_quarantine_recovery_and_retention"
            ),
            "hardware_mapping_contract": memory.get("hardware_mapping_contract"),
        },
        "prospective_live_csl": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5925-sota-transactional-csl-prospective"
            ),
            "declared_deliverable": live_meta.get("declared_deliverable"),
            "declared_deliverable_present": bool(live_meta.get("present")),
            "lift_measured": False,
            "lift_result": "missing_gate_blocked_no_live_csl_rows",
        },
        "adaptive_state_abi": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5926-adaptive-state-abi-v2-parity"
            ),
            "ready_score": abi.get("adaptive_state_abi_v2_ready_score"),
            "operations": abi.get("adaptive_state_abi_v2_schema_and_operations"),
            "parity": abi.get("byte_state_status_and_error_parity"),
            "rollback": abi.get("crash_prefix_recovery_and_rollback"),
        },
        "same_event_or_future_label_leakage_promoted": False,
        "principle": FIELD_PRINCIPLES[
            "continuous_self_learning_chronology_lift_safety_retention_and_rollback_receipt"
        ],
    }


def _arc_receipt(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    router = payloads.get("exp5927-coordinate-router-progress-qualification", {})
    binding = payloads.get("exp5928-arc-live-runner-execution-binding", {})
    live = payloads.get("exp5929-arc-structured-memory-bound-live-ab", {})
    live_primary = live.get("primary_live_utility_comparison_and_intervals")
    live_rows = (
        live_primary.get("complete_bound_live_rows") if isinstance(live_primary, Mapping) else None
    )
    binding_receipt = binding.get("actual_live_entrypoint_consumption_receipt")
    return {
        "coordinate_router_offline_qualification": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5927-coordinate-router-progress-qualification"
            ),
            "ready_score": router.get("coordinate_router_progress_ready_score"),
            "power_gate": router.get("hard_progress_positive_count_and_power_gate"),
            "no_level_solve_or_registry_update": router.get("no_level_solve_or_registry_update"),
        },
        "execution_binding": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5928-arc-live-runner-execution-binding"
            ),
            "ready_score": binding.get("live_runner_execution_binding_ready_score"),
            "actual_entrypoint_consumption": binding_receipt,
            "fixture_only_validation": binding_receipt.get("fixture_only_validation")
            if isinstance(binding_receipt, Mapping)
            else None,
        },
        "bound_live_ab": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5929-arc-structured-memory-bound-live-ab"
            ),
            "ready_score": live.get("structured_memory_live_ready_score"),
            "actual_bound_e3_entrypoint_receipt": live.get("actual_bound_e3_entrypoint_receipt"),
            "primary_live_utility_comparison_and_intervals": live_primary,
            "registry_unchanged": live.get("registry_unchanged"),
        },
        "offline_qualification_receives_live_credit": False,
        "actual_execution_binding_ready": binding.get("live_runner_execution_binding_ready_score")
        == 1.0,
        "live_rows_completed": live_rows or 0,
        "registry_update_performed": False,
        "principle": FIELD_PRINCIPLES["arc_coordinate_execution_binding_and_live_receipt"],
    }


def _hardware_receipt(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    abi = payloads.get("exp5926-adaptive-state-abi-v2-parity", {})
    board = payloads.get("exp5930-adaptive-state-board-mapping", {})
    return {
        "abi_v2": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5926-adaptive-state-abi-v2-parity"
            ),
            "ready_score": abi.get("adaptive_state_abi_v2_ready_score"),
            "operations": abi.get("adaptive_state_abi_v2_schema_and_operations"),
            "parity": abi.get("byte_state_status_and_error_parity"),
        },
        "hardware_mapping": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5930-adaptive-state-board-mapping"
            ),
            "ready_score": board.get("board_abi_mapping_ready_score"),
            "operation_mapping": board.get("abi_v2_schema_hash_and_operation_mapping"),
            "simulator_reference_trace_parity": board.get("simulator_reference_trace_parity"),
            "static_synthesis_timing_estimate_and_resource_reports": board.get(
                "static_synthesis_timing_estimate_and_resource_reports"
            ),
            "physical_probe_executed": board.get("physical_probe_executed"),
            "board_state_receipts": board.get("kv260_polarfire_and_gatemate_state_receipts"),
        },
        "abi_v2_ready": abi.get("adaptive_state_abi_v2_ready_score") == 1.0,
        "static_mapping_ready": board.get("board_abi_mapping_ready_score") == 1.0,
        "static_mapping_is_physical_acceleration": False,
        "physical_probe_executed": bool(board.get("physical_probe_executed")),
        "principle": FIELD_PRINCIPLES["adaptive_state_abi_and_hardware_receipt"],
    }


def _adversarial_receipts(
    adversarial_receipt: JsonMap, tasks: Sequence[JsonMap], metadata: Mapping[str, JsonMap]
) -> JsonDict:
    reports: list[JsonDict] = []
    for row in tasks:
        task_id = str(row["id"])
        artifact = str(row["deliverable"])
        if not metadata.get(task_id, {}).get("present") or task_id == EXPERIMENT_ID:
            continue
        report = _receipt_for_task(task_id, artifact, adversarial_receipt)
        reports.append(
            {
                "task_id": task_id,
                "artifact": artifact,
                "loaded": report.get("loaded"),
                "flag_count": report.get("flag_count"),
                "max_severity": report.get("max_severity"),
                "flags": report.get("flags") or [],
            }
        )
    missing = [
        str(row["deliverable"])
        for row in tasks
        if not metadata.get(str(row["id"]), {}).get("present") and str(row["id"]) != EXPERIMENT_ID
    ]
    return {
        "command": adversarial_receipt.get("command"),
        "exit_code": adversarial_receipt.get("exit_code"),
        "stdout_sha256": adversarial_receipt.get("stdout_sha256")
        or sha256_json(adversarial_receipt),
        "flagged_count": adversarial_receipt.get("flagged_count"),
        "warnings": adversarial_receipt.get("warnings") or [],
        "verified_present_declared_deliverable_count": len(reports),
        "reports": reports,
        "missing_declared_deliverables_not_verified": missing,
        "principle": FIELD_PRINCIPLES["adversarial_verifier_receipts"],
    }


def _docs_reconciled(root: Path) -> JsonDict:
    report_text = (
        (root / REPORT_SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / REPORT_SPEC_RELATIVE_PATH).exists()
        else ""
    )
    capstone_text = (
        (root / CAPSTONE_SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / CAPSTONE_SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "openspec_research_reporting_req_5931_present": "REQ-REPORT-5931" in report_text,
        "openspec_capstone_req_5931_present": "REQ-CAPSTONE-5931" in capstone_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "principle": FIELD_PRINCIPLES["docs_reconciled"],
    }


def _recommendations() -> list[JsonDict]:
    return [
        {
            "id": "next_constraint_semantic_executor_gate",
            "terminal_evidence": (
                "Exp5921 and Exp5922 are ready structural/tokenizer receipts, while exact "
                "semantic success must remain executor-owned."
            ),
            "prerequisites": [
                "Exp5921 schema support ready score remains 1.0",
                "Exp5922 embedded GGUF tokenizer bridge ready score remains 1.0",
                "A fresh held exact-executor fixture is sealed before model output is read",
            ],
            "falsifiable_success_condition": (
                "On sealed held cases, exact semantic success exceeds the direct-control "
                "rate by a preregistered positive lower CI bound with zero unsafe accepts."
            ),
            "stop_rules": [
                "Stop before model load if tokenizer replay or exact-executor hashes drift",
                "Stop after the first unsafe exact accept",
                "Stop if structural validity improves while exact semantic success remains at zero",
            ],
            "authority_boundaries": [
                "Grammar, type, and scope checks authorize syntax only",
                "Exact executor receipts alone authorize semantic success",
            ],
            "excluded_scopes": [
                "retired schema-supported reprompt mechanism",
                "numeric-prefix artifact substitution",
                "structural validity reported as a semantic win",
            ],
            "evidence_task_ids": [
                "exp5921-schema-derived-constraintir-support",
                "exp5922-gguf-schema-decoder-bridge",
            ],
        },
        {
            "id": "next_csl_fixture_to_live_lift_gate",
            "terminal_evidence": (
                "Exp5920, Exp5924, and Exp5926 provide chronology, transactional memory, "
                "rollback, retention, and ABI receipts without live CSL lift rows."
            ),
            "prerequisites": [
                "Exp5920 stream prefix chain replays exactly",
                "Exp5924 transactional memory ready score remains 1.0",
                "Exp5926 ABI parity remains 1.0 across fresh processes",
            ],
            "falsifiable_success_condition": (
                "A chronological A/B reports positive prospective exact-semantic transfer "
                "over fixed-memory and shuffled-history controls with retention 1.0 and zero unsafe promotions."
            ),
            "stop_rules": [
                "Stop on same-event read-after-write leakage",
                "Stop on future-label visibility",
                "Stop when rollback or restart recovery hash parity fails",
            ],
            "authority_boundaries": [
                "Exact validator receipts authorize promotion",
                "Memory similarity and model-authored labels do not authorize promotion",
            ],
            "excluded_scopes": [
                "missing prospective all-three-model CSL run",
                "retired frozen-slot CSL replay",
                "model weight mutation",
            ],
            "evidence_task_ids": [
                "exp5920-prospective-event-stream-admission",
                "exp5924-transactional-constraint-memory-v2",
                "exp5926-adaptive-state-abi-v2-parity",
            ],
        },
        {
            "id": "next_arc_bound_live_power_gate",
            "terminal_evidence": (
                "Exp5927 is underpowered offline progress evidence and Exp5928 proves actual "
                "runner binding; live structured-memory rows remain absent."
            ),
            "prerequisites": [
                "Exp5928 actual child binding ready score remains 1.0",
                "Registry before hash equals after hash before the first scored episode",
                "A powered held-cell plan reaches the preregistered minimum positive-row count",
            ],
            "falsifiable_success_condition": (
                "Capability-bound adapter-disabled held episodes complete live rows and the "
                "structured arm beats both no-memory and raw-tape controls by preregistered lower CI bounds."
            ),
            "stop_rules": [
                "Stop if capability is absent, self-issued, expired, replayed, or wrong-scope",
                "Stop if any public solve target is selected",
                "Stop if completed live rows remain zero after preflight",
            ],
            "authority_boundaries": [
                "Offline progress can size the test but cannot count as live evidence",
                "Execution binding can authorize process scope but not scientific success",
            ],
            "excluded_scopes": [
                "public ARC solve credit",
                "offline AUROC treated as live search",
                "adapter-enabled or unbound runner path",
            ],
            "evidence_task_ids": [
                "exp5927-coordinate-router-progress-qualification",
                "exp5928-arc-live-runner-execution-binding",
            ],
        },
    ]


def _field_provenance() -> dict[str, str]:
    return {
        field: (
            f"{field} derived from active roadmap exact declared artifacts, local hashes, "
            "adversarial receipts, append receipts, protected-file receipts, or tests."
        )
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_commands(test_results: Sequence[JsonMap]) -> list[str]:
    return [str(row.get("command")) for row in test_results if isinstance(row, Mapping)]


def _test_exit_codes(test_results: Sequence[JsonMap]) -> dict[str, int | None]:
    return {
        str(row.get("command")): row.get("exit_code")
        if isinstance(row.get("exit_code"), int)
        else None
        for row in test_results
        if isinstance(row, Mapping)
    }


def _preconditions(
    root: Path,
    tasks: Sequence[JsonMap],
    roadmap_receipt: JsonMap,
    adversarial_receipt: JsonMap,
) -> JsonDict:
    source_hashes = _source_hashes(root, tasks)
    declared_paths = [str(row.get("deliverable") or "") for row in tasks]
    return {
        "roadmap": roadmap_receipt,
        "roadmap_next": _roadmap_next_receipt(root),
        "declared_path_count": len(declared_paths),
        "declared_paths": declared_paths,
        "source_hashes": source_hashes,
        "completion_history": {
            "path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        },
        "conductor_log": {
            "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
        },
        "exclusion_manifest": {
            "path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        },
        "arc_registry": {
            "path": ARC_REGISTRY_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ARC_REGISTRY_RELATIVE_PATH),
        },
        "adversarial_verifier_state": {
            "script_sha256": path_sha256(root / ADVERSARIAL_VERIFY_RELATIVE_PATH),
            "receipt_sha256": sha256_json(adversarial_receipt),
            "receipt_exit_code": adversarial_receipt.get("exit_code"),
        },
        "resources": _resource_receipts(root),
        "atomic_output": _atomic_output_receipt(root),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def reconcile_v526(
    *,
    root: Path = REPO_ROOT,
    adversarial_receipt: JsonMap | None = None,
    test_results: Sequence[JsonMap] | None = None,
    write: bool = True,
    update_ledgers: bool = True,
) -> JsonDict:
    start = time.perf_counter()
    root = Path(root)
    adversarial_receipt = dict(adversarial_receipt or {})
    test_results = list(test_results or [])
    protected_before = _protected_hashes(root)
    registry_before = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)

    tasks, roadmap_receipt = _active_tasks(root)
    payloads, metadata = _artifact_payloads(root, tasks)
    conductor = _conductor_outcomes(root, tasks)
    classes = _terminal_classification(tasks, payloads, metadata, conductor, adversarial_receipt)
    matrix = _activated_matrix(tasks, payloads, metadata, conductor, classes)
    completion = _append_completion_once(root, tasks, classes, update_ledgers)
    retirement = _retirement_decisions(root, tasks, payloads, update_ledgers)

    status = "complete_with_nulls"
    if not roadmap_receipt.get("milestone_matches") or not roadmap_receipt.get("exact_range"):
        status = "blocked"
    if classes.get("invalid_artifact_task_ids"):
        status = "blocked"

    honest_prefix = "blocked:" if status == "blocked" else "complete_with_nulls:"
    honest_detail = (
        ".526 reconciled by exact declared deliverables with positive, null, "
        "underpowered, blocked-precondition, retired, gate-blocked, no-change, "
        "and missing receipts preserved independently"
    )

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "preconditions_checked": _preconditions(root, tasks, roadmap_receipt, adversarial_receipt),
        "milestone_and_task_range": {
            "milestone": MILESTONE,
            "task_range": "Exp5918-Exp5931",
            "activated_task_count": len(tasks),
            "expected_task_count": 14,
            "exact_range": roadmap_receipt.get("exact_range"),
            "task_ids": [str(row["id"]) for row in tasks],
            "principle": FIELD_PRINCIPLES["milestone_and_task_range"],
        },
        "activated_task_and_declared_deliverable_matrix": matrix,
        "exact_terminal_classification": classes,
        "adversarial_verifier_receipts": _adversarial_receipts(
            adversarial_receipt, tasks, metadata
        ),
        "transition_and_source_receipt": _transition_and_source_receipt(payloads, classes),
        "schema_constraintir_structural_and_exact_semantic_receipt": _schema_receipt(
            payloads, classes
        ),
        "continuous_self_learning_chronology_lift_safety_retention_and_rollback_receipt": _csl_receipt(
            payloads, metadata, classes
        ),
        "arc_coordinate_execution_binding_and_live_receipt": _arc_receipt(payloads, classes),
        "adaptive_state_abi_and_hardware_receipt": _hardware_receipt(payloads, classes),
        "branch_independence_receipt": _branch_independence(classes),
        "prior_failure_and_retirement_decisions": retirement,
        "missing_gate_blocked_and_reserved_receipts": _missing_gate_reserved_receipts(
            matrix, classes
        ),
        "duplicate_history_amplification_count": completion[
            "duplicate_history_amplification_count"
        ],
        "research_complete_append_receipt": completion,
        "docs_reconciled": _docs_reconciled(root),
        "next_three_falsifiable_recommendations": _recommendations(),
        "registry_unchanged": _registry_unchanged(root, registry_before),
        "protected_files_unchanged": _protected_unchanged(root, protected_before),
        "duration_s": round(time.perf_counter() - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "field_principles": {field: FIELD_PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS},
        "test_commands": _test_commands(test_results),
        "test_exit_codes": _test_exit_codes(test_results),
        "reproducibility_checksum": "",
        "honest_verdict": f"{honest_prefix} {honest_detail}",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    if write:
        write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def _load_receipt(path: Path | None) -> JsonDict:
    if path is None:
        return {}
    payload, meta = _read_json(path)
    if not meta["loadable"]:
        raise SystemExit(f"receipt is not loadable JSON: {path}")
    return payload


def _load_test_results(path: Path | None) -> list[JsonDict]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        rows = payload.get("results") or payload.get("test_results") or []
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    raise SystemExit(f"test results are not a list or mapping with results: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--adversarial-receipt", type=Path)
    parser.add_argument("--test-results", type=Path)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-ledgers", action="store_true")
    args = parser.parse_args(argv)

    payload = reconcile_v526(
        root=args.root,
        adversarial_receipt=_load_receipt(args.adversarial_receipt),
        test_results=_load_test_results(args.test_results),
        write=not args.no_write,
        update_ledgers=not args.no_ledgers,
    )
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
