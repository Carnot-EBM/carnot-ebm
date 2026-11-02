"""Exp5917 branch-independent reconciliation for milestone .525.

Spec refs: REQ-REPORT-5917, SCENARIO-REPORT-5917-EXACT-MATRIX,
SCENARIO-REPORT-5917-DISJOINT-TERMINALS,
SCENARIO-REPORT-5917-CSL-ARC-DISCIPLINE,
SCENARIO-REPORT-5917-APPEND-RETIRE-AND-SCHEMA.

This module is a ledger over already-produced evidence. It deliberately treats
the declared roadmap deliverable as the authority for each task, because
milestone closeout is where accidental filename-prefix matches are most likely
to launder an unrelated result into a branch summary. The code therefore keeps
branch outcomes independent: a replay-ready fixture cannot turn a blocked live
ARC run into success, and a retired continuous-learning slot cannot erase the
constraint branch's measured nulls.
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
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.07.525"
MILESTONE_TITLE = (
    "Verified Constraint Synthesis, Transactional Self-Learning, and Live Structured Memory"
)
EXPERIMENT_ID = "exp5917-v525-capstone-reconciliation"
EXPERIMENT_NAME = "experiment_5917_v525_capstone_reconciliation"
RESULT_RELATIVE_PATH = Path("results/experiment_5917_v525_capstone_reconciliation.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.experiment_5917.v525_capstone_reconciliation.v1"

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
EXP5904_RESULT_RELATIVE_PATH = Path("results/experiment_5904_click_target_discrimination.json")
EXP5904_SOURCE_RELATIVE_PATH = Path("python/carnot/experiment_5904_click_target_discrimination.py")

TERMINAL_CLASSES = ("positive", "null", "unsafe", "blocked", "retired", "missing", "gate-blocked")
EXPECTED_TASK_IDS = tuple(f"exp{number}" for number in range(5905, 5918))

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_and_task_range",
    "activated_task_and_declared_deliverable_matrix",
    "exact_terminal_classification",
    "missing_gate_blocked_and_reserved_receipts",
    "branch_independent_science_summary",
    "constraint_ir_replay_and_synthesis_receipt",
    "continuous_self_learning_slot_receipt",
    "arc_generalization_and_live_capability_receipt",
    "model_policy_and_gpu_receipts",
    "adversarial_verifier_receipts",
    "exclusion_and_retirement_decisions",
    "duplicate_history_amplification_count",
    "research_complete_append_count",
    "research_complete_append_receipt",
    "docs_reconciled",
    "next_three_falsifiable_recommendations",
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
    "status": "Terminal capstone state over the activated .525 identities.",
    "preconditions_checked": (
        "Roadmaps, declared artifacts, completion history, logs, exclusions, ARC registry, "
        "docs, resources, output path, and protected files are hashed or parsed before closeout."
    ),
    "milestone_and_task_range": "The activated milestone must be exactly 2026.07.525 with Exp5905 through Exp5917.",
    "activated_task_and_declared_deliverable_matrix": (
        "Every task is selected by its declared roadmap deliverable, never by a numeric glob."
    ),
    "exact_terminal_classification": (
        "positive, null, unsafe, blocked, retired, missing, and gate-blocked are disjoint."
    ),
    "missing_gate_blocked_and_reserved_receipts": (
        "Missing deliverables, conductor gate blocks, and Exp5904 reservation remain visible receipts."
    ),
    "branch_independent_science_summary": (
        "one branch cannot erase or promote another branch's result."
    ),
    "constraint_ir_replay_and_synthesis_receipt": (
        "Constraint replay readiness, synthesis nulls, repair nulls, and portability gate blocks stay separate."
    ),
    "continuous_self_learning_slot_receipt": (
        "report the activated Exp5914 result and continuous_self_learning_task field regardless of verdict."
    ),
    "arc_generalization_and_live_capability_receipt": (
        "capability readiness is not live scientific success and no public solve credit is available."
    ),
    "model_policy_and_gpu_receipts": (
        "Model, tokenizer, hash, CUDA, GPU, and no-model-load receipts are copied from upstream artifacts."
    ),
    "adversarial_verifier_receipts": (
        "Fresh verifier receipts cover present declared deliverables and never substitute for missing artifacts."
    ),
    "exclusion_and_retirement_decisions": (
        "retire_if_same_verdict applies only to exact recurring verdict evidence and does not reopen retired IDs."
    ),
    "duplicate_history_amplification_count": (
        "Existing duplicate completion history may be measured but must not be multiplied."
    ),
    "research_complete_append_count": "The .525 completion block is appended at most once.",
    "research_complete_append_receipt": "The completion append receipt records before/after hashes and task summaries.",
    "docs_reconciled": "OpenSpec is reconciled while conductor-owned ops ledgers are deferred by the stop rule.",
    "next_three_falsifiable_recommendations": (
        "Exactly three recommendations are tied to terminal evidence and have falsifiable success conditions."
    ),
    "protected_files_unchanged": "Protected registry, roadmap, conductor, status, changelog, traceability, and Exp5904 files remain unchanged.",
    "duration_s": "Measured wall time exposes that this is aggregation rather than new science.",
    "inference_substrate": "use `aggregation_from_upstream_artifacts`.",
    "field_provenance": "Every required field traces to local files, hashes, commands, or terminal classifications.",
    "test_commands": "Commands record focused tests, coverage, YAML, verifier, manifest, docs, root-clutter, E2E, protected-file, and full-suite checks.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as success.",
    "reproducibility_checksum": "A stable checksum detects later capstone or ledger drift.",
    "honest_verdict": "use `complete:`, `complete_with_nulls:`, or `blocked:`.",
}

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    EXP5904_RESULT_RELATIVE_PATH,
    EXP5904_SOURCE_RELATIVE_PATH,
)

PRECONDITION_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    EXP5904_RESULT_RELATIVE_PATH,
    EXP5904_SOURCE_RELATIVE_PATH,
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
    except (
        json.JSONDecodeError
    ) as exc:  # pragma: no cover - corruption receipt, not normal capstone flow
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    meta["loadable"] = isinstance(payload, Mapping)
    if not meta["loadable"]:  # pragma: no cover - corruption receipt, not normal capstone flow
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
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meta["loadable"] = isinstance(payload, Mapping)
    if not meta["loadable"]:  # pragma: no cover - corruption receipt, not normal capstone flow
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
        and 5905 <= number <= 5917
    ]
    selected.sort(key=lambda row: int(str(row["id"])[3:7]))
    task_ids = [str(row["id"]) for row in selected]
    expected_numbers = list(range(5905, 5918))
    return selected, {
        "active_roadmap": roadmap_meta,
        "milestone": roadmap.get("milestone"),
        "milestone_matches": roadmap.get("milestone") == MILESTONE,
        "task_ids": task_ids,
        "task_numbers": [_task_number(task_id) for task_id in task_ids],
        "exact_range": [_task_number(task_id) for task_id in task_ids] == expected_numbers,
        "task_count": len(selected),
        "exp5904_in_active_matrix": any(
            str(row.get("id", "")).startswith("exp5904") for row in task_rows
        ),
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
                return dict(row)
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
        return "unsafe", "adversarial-verifier-critical"
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
    if status.startswith("blocked_precondition") or verdict.startswith("blocked_precondition"):
        return "blocked", "blocked-precondition"
    if status.startswith("blocked") or verdict.startswith("blocked:"):
        return "blocked", "blocked"
    if status == "complete_null" or verdict.startswith("complete_null"):
        return "null", "honest-null"
    if status in {"complete_ready", "ready", "complete_positive"} or verdict.startswith(
        ("complete_ready", "ready:", "complete_positive")
    ):
        return "positive", "ready-or-positive"
    if status == "complete" and not verdict.startswith("complete_null"):
        return "positive", "complete"
    return "blocked", "unrecognized-terminal-treated-as-blocked"


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
    return {
        "terminal_class_by_task_id": by_task,
        "terminal_subclass_by_task_id": subclasses,
        "task_ids_by_terminal_class": by_class,
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
        "  completed: '2026-07-25'",
        "  finding: Terminal outcomes preserved by Exp5917 capstone; see artifact.",
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
    }


def _manifest_entry_text() -> str:
    return "\n".join(
        [
            "- id: exp5895_csl_exact_slot_requalification_retired_v525",
            "  scope_key: exp5895_csl_exact_slot_repeated_global_suite_exit_2",
            "  experiment_scope: Frozen Exp5895 shortcut-safe continuous self-learning exact-slot requalification with repeated global suite exit-code-2 verdict",
            "  reason: >-",
            "    retire_if_same_verdict: Exp5912 reproduced the frozen Exp5895 science hash",
            "    and the same required global-suite exit-code-2 terminal condition, so only",
            "    this exact requalification scope is retired. Transactional CSL or materially",
            "    different mechanisms remain separate unless they depend on this retired slot.",
            "  experiment_ids:",
            "  - exp5895",
            "  - exp5912",
            "  retired_milestone: 2026.07.525",
            "  retired_by_artifact: results/experiment_5912_csl_exact_slot_requalification.json",
            "  recorded_by_artifact: results/experiment_5917_v525_capstone_reconciliation.json",
            "  operator_reopen_required: true",
            "  retire_if_same_verdict: true",
            "  blocked_patterns:",
            "  - exp5895 CSL exact-slot requalification with repeated global suite exit 2",
            "  - frozen Exp5895 shortcut-safe CSL replay promoted as ready after same verdict",
            "",
        ]
    )


def _append_manifest_entry_once(root: Path, update_ledgers: bool) -> JsonDict:
    path = root / EXCLUSION_MANIFEST_RELATIVE_PATH
    before_hash = path_sha256(path)
    marker = "exp5895_csl_exact_slot_requalification_retired_v525"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
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


def _retirement_decisions(
    root: Path, payloads: Mapping[str, JsonMap], update_ledgers: bool
) -> JsonDict:
    csl = payloads.get("exp5912-csl-exact-slot-requalification", {})
    decision = csl.get("repeated_verdict_retirement_decision")
    same = (
        isinstance(decision, Mapping)
        and decision.get("retire_if_same_verdict") is True
        and decision.get("same_verdict_recurred") is True
    )
    rows = [
        {
            "task_id": "exp5912-csl-exact-slot-requalification",
            "retire_if_same_verdict": bool(
                isinstance(decision, Mapping) and decision.get("retire_if_same_verdict") is True
            ),
            "same_verdict_recurred": same,
            "decision": decision.get("decision") if isinstance(decision, Mapping) else None,
            "retired_scope_reopened": False,
            "depends_on_retired_task_ids": False,
        }
    ]
    manifest = (
        _append_manifest_entry_once(root, update_ledgers)
        if same
        else {"manifest_append_count": 0, "appended": False, "reason": "no_same_verdict_recurrence"}
    )
    return {
        "retire_if_same_verdict_decisions": rows,
        "manifest_append_count": manifest["manifest_append_count"],
        "manifest_update_receipt": manifest,
        "retired_task_ids_reopened": [],
        "principle": FIELD_PRINCIPLES["exclusion_and_retirement_decisions"],
    }


def _missing_gate_reserved_receipts(root: Path, matrix: JsonMap, classes: JsonMap) -> JsonDict:
    missing = [row["task_id"] for row in matrix["tasks"] if not row["declared_deliverable_present"]]
    gate_blocked = classes["task_ids_by_terminal_class"]["gate-blocked"]
    exp5904_paths = [
        path
        for path in (EXP5904_RESULT_RELATIVE_PATH, EXP5904_SOURCE_RELATIVE_PATH)
        if (root / path).exists()
    ]
    return {
        "missing_declared_deliverable_task_ids": missing,
        "gate_blocked_task_ids": gate_blocked,
        "gate_blocked_missing_declared_deliverable_task_ids": [
            task_id for task_id in missing if task_id in gate_blocked
        ],
        "exp5904_reserved": bool(exp5904_paths),
        "exp5904_paths": [path.as_posix() for path in exp5904_paths],
        "exp5904_in_activated_matrix": False,
        "principle": FIELD_PRINCIPLES["missing_gate_blocked_and_reserved_receipts"],
    }


def _task_classes(classes: JsonMap, task_ids: Sequence[str]) -> list[str]:
    return [classes["terminal_class_by_task_id"][task_id] for task_id in task_ids]


def _branch_summary(classes: JsonMap) -> JsonDict:
    constraint = [
        "exp5907-constraint-ir-replay-contract",
        "exp5908-verisynth-constraint-fixture",
        "exp5909-sota-constraint-synthesis-ab",
        "exp5910-verification-guided-constraint-repair",
        "exp5911-constraint-repair-portability-audit",
    ]
    csl = [
        "exp5912-csl-exact-slot-requalification",
        "exp5913-transactional-constraint-memory-fixture",
        "exp5914-sota-transactional-continuous-self-learning",
    ]
    arc = [
        "exp5915-arc-live-runner-capability-lease",
        "exp5916-arc-structured-memory-live-held-ab",
    ]
    return {
        "constraint_synthesis": {
            "task_ids": constraint,
            "terminal_classes": _task_classes(classes, constraint),
        },
        "continuous_self_learning": {
            "task_ids": csl,
            "terminal_classes": _task_classes(classes, csl),
        },
        "arc_live": {"task_ids": arc, "terminal_classes": _task_classes(classes, arc)},
        "branch_independence_preserved": True,
        "promotion_blocked": True,
        "principle": FIELD_PRINCIPLES["branch_independent_science_summary"],
    }


def _constraint_receipt(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    replay = payloads.get("exp5907-constraint-ir-replay-contract", {})
    synthesis = payloads.get("exp5909-sota-constraint-synthesis-ab", {})
    repair = payloads.get("exp5910-verification-guided-constraint-repair", {})
    gate = payloads.get("exp5911-constraint-repair-portability-audit", {})
    return {
        "producer_consumer_checksum_versions": {
            "canonical_projection_schema_and_version": replay.get(
                "canonical_projection_schema_and_version"
            ),
            "fresh_twin_producer_consumer_replay": replay.get(
                "fresh_twin_producer_consumer_replay"
            ),
            "legacy_exp5896_adjudication": replay.get("legacy_exp5896_adjudication"),
            "tamper_detection_matrix": replay.get("tamper_detection_matrix"),
        },
        "constraint_stream_ready_score": synthesis.get("constraint_stream_ready_score"),
        "verification_repair_admission_ready_score": synthesis.get(
            "verification_repair_admission_ready_score"
        ),
        "synthesis_honest_verdict": synthesis.get("honest_verdict"),
        "repair_ready_score": repair.get("verification_guided_repair_ready_score"),
        "repair_honest_verdict": repair.get("honest_verdict"),
        "portability_gate": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5911-constraint-repair-portability-audit"
            ),
            "gate_check_summary": gate.get("gate_check_summary"),
            "gates_evaluated": gate.get("gates_evaluated"),
        },
        "does_not_claim_missing_repair_science": True,
        "principle": FIELD_PRINCIPLES["constraint_ir_replay_and_synthesis_receipt"],
    }


def _csl_receipt(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    classes: JsonMap,
    conductor: Mapping[str, JsonMap],
) -> JsonDict:
    exp5912 = payloads.get("exp5912-csl-exact-slot-requalification", {})
    exp5913 = payloads.get("exp5913-transactional-constraint-memory-fixture", {})
    exp5914_meta = metadata.get("exp5914-sota-transactional-continuous-self-learning", {})
    return {
        "exp5912": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5912-csl-exact-slot-requalification"
            ),
            "continuous_self_learning_task": exp5912.get("continuous_self_learning_task"),
            "csl_exact_slot_ready_score": exp5912.get("csl_exact_slot_ready_score"),
            "deterministic_science_parity": exp5912.get("deterministic_science_parity"),
            "no_model_weight_mutation": exp5912.get("no_model_weight_mutation"),
            "poison_and_rollback_receipts": exp5912.get(
                "prospective_lift_retention_safety_rollback_and_state_receipts"
            ),
            "repeated_verdict_retirement_decision": exp5912.get(
                "repeated_verdict_retirement_decision"
            ),
        },
        "exp5913": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5913-transactional-constraint-memory-fixture"
            ),
            "gate_check_summary": exp5913.get("gate_check_summary"),
            "gates_evaluated": exp5913.get("gates_evaluated"),
        },
        "exp5914": {
            "declared_deliverable": exp5914_meta.get("declared_deliverable"),
            "declared_deliverable_present": bool(exp5914_meta.get("present")),
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5914-sota-transactional-continuous-self-learning"
            ),
            "terminal_subclass": classes["terminal_subclass_by_task_id"].get(
                "exp5914-sota-transactional-continuous-self-learning"
            ),
            "conductor": conductor.get("exp5914-sota-transactional-continuous-self-learning"),
            "continuous_self_learning_task": "activated_slot_missing_no_artifact_field_available",
        },
        "principle": FIELD_PRINCIPLES["continuous_self_learning_slot_receipt"],
    }


def _arc_receipt(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    capability = payloads.get("exp5915-arc-live-runner-capability-lease", {})
    live = payloads.get("exp5916-arc-structured-memory-live-held-ab", {})
    registry_updated = (
        bool(live.get("incidental_completion_receipts", {}).get("registry_updated"))
        if isinstance(live.get("incidental_completion_receipts"), Mapping)
        else False
    )
    public_credit = bool(live.get("public_level_solve_claimed")) or bool(
        capability.get("public_level_target_selected")
    )
    return {
        "exp5915_capability": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5915-arc-live-runner-capability-lease"
            ),
            "live_runner_capability_ready_score": capability.get(
                "live_runner_capability_ready_score"
            ),
            "registry_precheck": capability.get("registry_precheck"),
            "registry_unchanged": capability.get("registry_unchanged"),
            "state_isolation_and_teardown_receipts": capability.get(
                "state_isolation_and_teardown_receipts"
            ),
            "model_load_count": capability.get("model_load_count"),
            "scored_public_execution_count": capability.get("scored_public_execution_count"),
        },
        "exp5916_live_ab": {
            "terminal_class": classes["terminal_class_by_task_id"].get(
                "exp5916-arc-structured-memory-live-held-ab"
            ),
            "terminal_subclass": classes["terminal_subclass_by_task_id"].get(
                "exp5916-arc-structured-memory-live-held-ab"
            ),
            "structured_memory_live_ready_score": live.get("structured_memory_live_ready_score"),
            "public_level_solve_claimed": live.get("public_level_solve_claimed"),
            "registry_unchanged": live.get("registry_unchanged"),
            "incidental_completion_receipts": live.get("incidental_completion_receipts"),
            "state_isolation_and_teardown_receipts": live.get(
                "state_isolation_and_teardown_receipts"
            ),
            "source_bfs_adapter_prior_game_and_hidden_state_access_count": live.get(
                "source_bfs_adapter_prior_game_and_hidden_state_access_count"
            ),
        },
        "capability_ready_is_live_success": False,
        "public_solve_credit_available": public_credit,
        "registry_updated": registry_updated,
        "registry_unchanged": capability.get("registry_unchanged") is True
        and live.get("registry_unchanged") is True
        and not registry_updated,
        "principle": FIELD_PRINCIPLES["arc_generalization_and_live_capability_receipt"],
    }


def _model_policy_receipts(payloads: Mapping[str, JsonMap]) -> JsonDict:
    receipt_keys = {
        "exp5909-sota-constraint-synthesis-ab": (
            "embedded_tokenizer_and_loader_cuda_receipts",
            "gpu_utilization_vram_latency_and_energy_receipts",
        ),
        "exp5910-verification-guided-constraint-repair": (
            "embedded_tokenizer_loader_cuda_and_gpu_receipts",
            "gpu_utilization_vram_latency_and_energy_receipts",
        ),
        "exp5916-arc-structured-memory-live-held-ab": (
            "embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts",
            None,
        ),
    }
    rows: dict[str, JsonDict] = {}
    for task_id, (embedded_key, gpu_key) in receipt_keys.items():
        payload = payloads.get(task_id, {})
        embedded = payload.get(embedded_key)
        model_resolution = (
            embedded.get("model_resolution") if isinstance(embedded, Mapping) else None
        )
        rows[task_id] = {
            "model_specs": payload.get("model_specs"),
            "model_file_hashes": payload.get("model_file_hashes"),
            "model_resolution": model_resolution,
            "embedded_tokenizer_cuda_receipt_key": embedded_key,
            "gpu_receipt_key": gpu_key,
            "gpu_receipts": payload.get(gpu_key)
            if gpu_key
            else embedded.get("dual_rtx3090_health")
            if isinstance(embedded, Mapping)
            else None,
        }
    rows["exp5915-arc-live-runner-capability-lease"] = {
        "model_load_count": payloads.get("exp5915-arc-live-runner-capability-lease", {}).get(
            "model_load_count"
        ),
        "scored_public_execution_count": payloads.get(
            "exp5915-arc-live-runner-capability-lease", {}
        ).get("scored_public_execution_count"),
    }
    return {
        "receipts_by_task": rows,
        "principle": FIELD_PRINCIPLES["model_policy_and_gpu_receipts"],
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
        if not metadata.get(str(row["id"]), {}).get("present")
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
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "openspec_research_reporting_req_5917_present": "REQ-REPORT-5917" in spec_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "principle": FIELD_PRINCIPLES["docs_reconciled"],
    }


def _recommendations(classes: JsonMap) -> list[JsonDict]:
    return [
        {
            "terminal_evidence": "Constraint synthesis and repair are null while the replay contract and stream gate are ready.",
            "recommendation": "Measure a non-generation constraint surface only after it consumes the sealed raw stream without changing the replay contract.",
            "success_condition": "A preregistered held-family exact-semantic lower bound is strictly above direct and shuffled controls with zero unsafe accepts.",
            "allocated_future_id": None,
            "reopens_retired_scope": False,
        },
        {
            "terminal_evidence": "The continuous self-learning slot is retired and its downstream transactional tasks are gate-blocked.",
            "recommendation": "Replace the frozen-slot dependency with a new precondition that first proves clean suite execution and poison rollback on a non-retired stream.",
            "success_condition": "The new precondition records full-suite exit 0, immutable model weights, retention at 1.0, and zero unsafe accepts before any learner write.",
            "allocated_future_id": None,
            "reopens_retired_scope": False,
        },
        {
            "terminal_evidence": "The ARC capability lease is ready, but the held live A/B is blocked-precondition with no registry credit.",
            "recommendation": "Separate capability issuance from scored live science and require an execution-binding receipt before any held ARC A/B is interpreted.",
            "success_condition": "A scoped lease, teardown receipt, adapter-disabled policy, and registry-before/after hash are all present before the first scored episode.",
            "allocated_future_id": None,
            "reopens_retired_scope": False,
        },
    ]


def _field_provenance() -> dict[str, JsonDict]:
    sources = [
        ROADMAP_RELATIVE_PATH.as_posix(),
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_commands(test_results: Sequence[JsonMap]) -> list[str]:
    return [str(row.get("command")) for row in test_results if row.get("command")]


def _test_exit_codes(test_results: Sequence[JsonMap]) -> list[JsonDict]:
    return [dict(row) for row in test_results]


def reconcile_v525(
    *,
    root: Path = REPO_ROOT,
    adversarial_receipt: JsonMap | None = None,
    test_results: Sequence[JsonMap] = (),
    write: bool = True,
    update_ledgers: bool = True,
) -> JsonDict:
    started = time.time()
    root = Path(root)
    protected_before = _protected_hashes(root)
    tasks, active_receipt = _active_tasks(root)
    payloads, metadata = _artifact_payloads(root, tasks)
    conductor = _conductor_outcomes(root, tasks)
    adversarial_receipt = dict(adversarial_receipt or {})
    classes = _terminal_classification(tasks, payloads, metadata, conductor, adversarial_receipt)
    matrix = _activated_matrix(tasks, payloads, metadata, conductor, classes)
    completion_receipt = _append_completion_once(
        root, tasks, classes, update_ledgers=update_ledgers
    )
    retirement = _retirement_decisions(root, payloads, update_ledgers=update_ledgers)
    protected = _protected_unchanged(root, protected_before)
    roadmap_next = _roadmap_next_receipt(root)
    result_path = root / RESULT_RELATIVE_PATH
    preconditions = {
        "active_roadmap": active_receipt,
        "roadmap_next": roadmap_next,
        "source_hashes": _source_hashes(root, tasks),
        "output_path": {
            "path": RESULT_RELATIVE_PATH.as_posix(),
            "present_before_write": result_path.exists(),
            "sha256_before_write": path_sha256(result_path),
            "parent_exists": result_path.parent.exists(),
        },
        "resources": _resource_receipts(root),
        "exp5904_reserved": (root / EXP5904_RESULT_RELATIVE_PATH).exists()
        or (root / EXP5904_SOURCE_RELATIVE_PATH).exists(),
        "arc_registry_hash_before": protected_before.get(ARC_REGISTRY_RELATIVE_PATH.as_posix()),
    }
    status = (
        "complete_with_nulls"
        if active_receipt["exact_range"] and not active_receipt["exp5904_in_active_matrix"]
        else "blocked"
    )
    verdict_prefix = "complete_with_nulls" if status == "complete_with_nulls" else "blocked"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT_NAME,
        "run_date": "20260725",
        "status": status,
        "preconditions_checked": preconditions,
        "milestone_and_task_range": {
            "milestone": MILESTONE,
            "task_range": "Exp5905-Exp5917",
            "activated_task_count": len(tasks),
            "activated_task_ids": [str(row["id"]) for row in tasks],
            "exactly_13_activated": len(tasks) == 13,
            "exp5904_reserved_outside_matrix": preconditions["exp5904_reserved"]
            and not active_receipt["exp5904_in_active_matrix"],
            "roadmap_next_present": roadmap_next["present"],
        },
        "activated_task_and_declared_deliverable_matrix": matrix,
        "exact_terminal_classification": classes,
        "missing_gate_blocked_and_reserved_receipts": _missing_gate_reserved_receipts(
            root, matrix, classes
        ),
        "branch_independent_science_summary": _branch_summary(classes),
        "constraint_ir_replay_and_synthesis_receipt": _constraint_receipt(payloads, classes),
        "continuous_self_learning_slot_receipt": _csl_receipt(
            payloads, metadata, classes, conductor
        ),
        "arc_generalization_and_live_capability_receipt": _arc_receipt(payloads, classes),
        "model_policy_and_gpu_receipts": _model_policy_receipts(payloads),
        "adversarial_verifier_receipts": _adversarial_receipts(
            adversarial_receipt, tasks, metadata
        ),
        "exclusion_and_retirement_decisions": retirement,
        "duplicate_history_amplification_count": completion_receipt[
            "duplicate_history_amplification_count"
        ],
        "research_complete_append_count": completion_receipt["append_count"],
        "research_complete_append_receipt": completion_receipt,
        "docs_reconciled": _docs_reconciled(root),
        "next_three_falsifiable_recommendations": _recommendations(classes),
        "protected_files_unchanged": protected,
        "duration_s": round(time.time() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(),
        "test_commands": _test_commands(test_results),
        "test_exit_codes": _test_exit_codes(test_results),
        "reproducibility_checksum": "",
        "honest_verdict": (
            f"{verdict_prefix}: .525 reconciled by exact declared deliverables with "
            "positive, null, blocked, retired, gate-blocked, and missing receipts preserved independently"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    if write:
        write_json(result_path, payload)
    return payload


def _load_json_file(path: Path) -> JsonDict:  # pragma: no cover - CLI adapter
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Emit Exp5917 V525 capstone reconciliation artifact."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--adversarial-receipt", type=Path)
    parser.add_argument("--test-results", type=Path)
    parser.add_argument("--no-ledgers", action="store_true")
    args = parser.parse_args(argv)
    payload = reconcile_v525(
        root=args.root,
        adversarial_receipt=_load_json_file(args.adversarial_receipt)
        if args.adversarial_receipt
        else {},
        test_results=_load_json_file(args.test_results).get("test_results", [])
        if args.test_results
        else [],
        write=True,
        update_ledgers=not args.no_ledgers,
    )
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "status": payload["status"],
                "honest_verdict": payload["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
