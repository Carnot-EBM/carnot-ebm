"""Exp5918 transition receipt from terminal milestone .525 into .526.

Spec refs: REQ-REPORT-5918,
SCENARIO-REPORT-5918-EXACT-MATRIX,
SCENARIO-REPORT-5918-TERMINAL-CLASSES,
SCENARIO-REPORT-5918-APPEND-ONCE-AND-EXP5904,
SCENARIO-REPORT-5918-RANGE-COLLISION-SCHEMA.

This module is a transition ledger. It reads the Exp5917 capstone's exact
declared-deliverable matrix, rechecks the declared paths on disk, preserves
terminal classes as receipts, and proves that the Exp5918-Exp5931 allocation
has no unowned pre-existing references.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5918_transition_v526.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
EVIDENCE_INDEX_RELATIVE_PATH = Path("scripts/evidence_index_collision_preflight.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP5904_SOURCE_RELATIVE_PATH = Path("python/carnot/experiment_5904_click_target_discrimination.py")
EXP5904_RESULT_RELATIVE_PATH = Path("results/experiment_5904_click_target_discrimination.json")
EXP5917_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5917_v525_capstone_reconciliation.json")

EXPERIMENT = "experiment_5918_transition_v526"
EXPERIMENT_ID = "exp5918-transition-v526"
MILESTONE_FROM = "2026.07.525"
MILESTONE_TO = "2026.07.526"
MILESTONE_FROM_TITLE = (
    "Verified Constraint Synthesis, Transactional Self-Learning, and Live Structured Memory"
)
MILESTONE_TO_TITLE = (
    "Schema-Derived Constraint Decoding, Transactional Learning Reboot, and ARC Live-Path Qualification"
)
RUN_DATE = "20260725"
RANDOM_SEED = 5918
SCHEMA = "carnot.experiment_5918.transition_v526.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5918",
    "SCENARIO-REPORT-5918-EXACT-MATRIX",
    "SCENARIO-REPORT-5918-TERMINAL-CLASSES",
    "SCENARIO-REPORT-5918-APPEND-ONCE-AND-EXP5904",
    "SCENARIO-REPORT-5918-RANGE-COLLISION-SCHEMA",
)

ACTIVATED_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5905-transition-v525": Path("results/experiment_5905_transition_v525.json"),
    "exp5906-v525-source-delta-ingestion": Path(
        "results/experiment_5906_v525_source_delta_ingestion.json"
    ),
    "exp5907-constraint-ir-replay-contract": Path(
        "results/experiment_5907_constraint_ir_replay_contract.json"
    ),
    "exp5908-verisynth-constraint-fixture": Path(
        "results/experiment_5908_verisynth_constraint_fixture.json"
    ),
    "exp5909-sota-constraint-synthesis-ab": Path(
        "results/experiment_5909_sota_constraint_synthesis_ab.json"
    ),
    "exp5910-verification-guided-constraint-repair": Path(
        "results/experiment_5910_verification_guided_constraint_repair.json"
    ),
    "exp5911-constraint-repair-portability-audit": Path(
        "results/experiment_5911_constraint_repair_portability_audit.json"
    ),
    "exp5912-csl-exact-slot-requalification": Path(
        "results/experiment_5912_csl_exact_slot_requalification.json"
    ),
    "exp5913-transactional-constraint-memory-fixture": Path(
        "results/experiment_5913_transactional_constraint_memory_fixture.json"
    ),
    "exp5914-sota-transactional-continuous-self-learning": Path(
        "results/experiment_5914_sota_transactional_continuous_self_learning.json"
    ),
    "exp5915-arc-live-runner-capability-lease": Path(
        "results/experiment_5915_arc_live_runner_capability_lease.json"
    ),
    "exp5916-arc-structured-memory-live-held-ab": Path(
        "results/experiment_5916_arc_structured_memory_live_held_ab.json"
    ),
    "exp5917-v525-capstone-reconciliation": EXP5917_CAPSTONE_RELATIVE_PATH,
}

ACTIVATED_TASK_TITLES: dict[str, str] = {
    "exp5905-transition-v525": "Exact terminal-boundary handoff from .524 into .525",
    "exp5906-v525-source-delta-ingestion": "Dated evidence refresh after the V525 planner marker",
    "exp5907-constraint-ir-replay-contract": (
        "Canonical producer-consumer replay contract for typed ConstraintIR"
    ),
    "exp5908-verisynth-constraint-fixture": (
        "Gated on Exp5907 replay: VeriSynth-style decomposition and retrieval fixture"
    ),
    "exp5909-sota-constraint-synthesis-ab": (
        "Gated on Exp5908 fixture: three-family direct, decomposed, and retrieval synthesis"
    ),
    "exp5910-verification-guided-constraint-repair": (
        "Gated on Exp5909 residual headroom: exact-diagnostic constraint repair"
    ),
    "exp5911-constraint-repair-portability-audit": (
        "Gated on Exp5910 repair: model, family, camouflage, and diagnostic portability"
    ),
    "exp5912-csl-exact-slot-requalification": (
        "Frozen-science requalification of Exp5895 continuous self-learning"
    ),
    "exp5913-transactional-constraint-memory-fixture": (
        "Gated on Exp5912 and Exp5909: transactional read-before-write memory fixture"
    ),
    "exp5914-sota-transactional-continuous-self-learning": (
        "Gated on Exp5913 mechanism: prospective SOTA transactional continuous self-learning"
    ),
    "exp5915-arc-live-runner-capability-lease": (
        "Scoped capability lease for the adapter-disabled held ARC live runner"
    ),
    "exp5916-arc-structured-memory-live-held-ab": (
        "Gated on Exp5915 capability: adapter-disabled held structured-memory live A/B"
    ),
    "exp5917-v525-capstone-reconciliation": (
        "Branch-independent terminal reconciliation for milestone .525"
    ),
}

EXPECTED_TERMINAL_CLASSES: dict[str, str] = {
    "exp5905-transition-v525": "blocked",
    "exp5906-v525-source-delta-ingestion": "null",
    "exp5907-constraint-ir-replay-contract": "positive",
    "exp5908-verisynth-constraint-fixture": "positive",
    "exp5909-sota-constraint-synthesis-ab": "null",
    "exp5910-verification-guided-constraint-repair": "null",
    "exp5911-constraint-repair-portability-audit": "gate-blocked",
    "exp5912-csl-exact-slot-requalification": "retired",
    "exp5913-transactional-constraint-memory-fixture": "gate-blocked",
    "exp5914-sota-transactional-continuous-self-learning": "gate-blocked",
    "exp5915-arc-live-runner-capability-lease": "positive",
    "exp5916-arc-structured-memory-live-held-ab": "blocked-precondition",
    "exp5917-v525-capstone-reconciliation": "positive",
}

NEXT_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5918-transition-v526": RESULT_RELATIVE_PATH,
    "exp5919-v526-source-delta-ingestion": Path(
        "results/experiment_5919_v526_source_delta_ingestion.json"
    ),
    "exp5920-prospective-event-stream-admission": Path(
        "results/experiment_5920_prospective_event_stream_admission.json"
    ),
    "exp5921-schema-derived-constraintir-support": Path(
        "results/experiment_5921_schema_derived_constraintir_support.json"
    ),
    "exp5922-gguf-schema-decoder-bridge": Path(
        "results/experiment_5922_gguf_schema_decoder_bridge.json"
    ),
    "exp5923-sota-schema-supported-constraintir-ab": Path(
        "results/experiment_5923_sota_schema_supported_constraintir_ab.json"
    ),
    "exp5924-transactional-constraint-memory-v2": Path(
        "results/experiment_5924_transactional_constraint_memory_v2.json"
    ),
    "exp5925-sota-transactional-csl-prospective": Path(
        "results/experiment_5925_sota_transactional_csl_prospective.json"
    ),
    "exp5926-adaptive-state-abi-v2-parity": Path(
        "results/experiment_5926_adaptive_state_abi_v2_parity.json"
    ),
    "exp5927-coordinate-router-progress-qualification": Path(
        "results/experiment_5927_coordinate_router_progress_qualification.json"
    ),
    "exp5928-arc-live-runner-execution-binding": Path(
        "results/experiment_5928_arc_live_runner_execution_binding.json"
    ),
    "exp5929-arc-structured-memory-bound-live-ab": Path(
        "results/experiment_5929_arc_structured_memory_bound_live_ab.json"
    ),
    "exp5930-adaptive-state-board-mapping": Path(
        "results/experiment_5930_adaptive_state_board_mapping.json"
    ),
    "exp5931-v526-capstone-reconciliation": Path(
        "results/experiment_5931_v526_capstone_reconciliation.json"
    ),
}
NEXT_RANGE_NUMBERS = range(5918, 5932)

PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXP5904_SOURCE_RELATIVE_PATH,
    EXP5904_RESULT_RELATIVE_PATH,
)

SOURCE_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    EVIDENCE_INDEX_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    EXP5917_CAPSTONE_RELATIVE_PATH,
    EXP5904_SOURCE_RELATIVE_PATH,
    EXP5904_RESULT_RELATIVE_PATH,
    *PROTECTED_FILE_PATHS,
    *ACTIVATED_TASK_ARTIFACT_PATHS.values(),
)

OWNED_REFERENCE_PATHS = (
    Path("python/carnot/experiment_5918_transition_v526.py"),
    Path("tests/python/test_experiment_5918_transition_v526.py"),
    SPEC_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
)

ALLOWED_ALLOCATION_REFERENCE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "activated_task_and_deliverable_matrix",
    "exact_terminal_classification",
    "blocked_retired_gate_blocked_and_missing_receipts",
    "adversarial_verifier_receipts",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
    "exp5904_separate_evidence_receipt",
    "next_task_range",
    "next_range_collision_count",
    "docs_reconciled",
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
    "status": "Terminal transition state over exact activated .525 identities.",
    "preconditions_checked": (
        "Roadmaps, history, logs, exclusions, resources, verifier availability, "
        "atomic output, protected files, and declared deliverables are checked before completion."
    ),
    "milestone_transition": "Explicit .525-to-.526 boundary prevents cross-milestone evidence laundering.",
    "activated_task_and_deliverable_matrix": (
        "only activated task IDs and exact declared paths count as milestone evidence."
    ),
    "exact_terminal_classification": (
        "Positive, null, blocked, blocked-precondition, retired, gate-blocked, and missing classes remain disjoint."
    ),
    "blocked_retired_gate_blocked_and_missing_receipts": (
        "Non-positive terminal classes remain receipts and are never converted into successes."
    ),
    "adversarial_verifier_receipts": (
        "Fresh verifier receipts cover every present declared .525 artifact without replacing missing ones."
    ),
    "research_complete_append_count": "Exact zero-or-one append behavior prevents duplicate .525 history.",
    "duplicate_history_amplification_count": "Existing duplicate history is measured but never multiplied.",
    "exp5904_separate_evidence_receipt": (
        "concurrent coordinate evidence is hashed and named but never edited, reclassified, or appended as a `.525` task."
    ),
    "next_task_range": "The finite Exp5918-Exp5931 range makes the next allocation auditable.",
    "next_range_collision_count": "only bare zero authorizes Exp5918-Exp5931.",
    "docs_reconciled": (
        "Transition-owned spec reconciliation is recorded while conductor-owned ledgers are deferred by the stop rule."
    ),
    "protected_files_unchanged": (
        "Protected roadmap, conductor, north-star, ops-ledger, and Exp5904 files remain byte-identical."
    ),
    "duration_s": "Measured wall time exposes aggregation-only execution.",
    "inference_substrate": "use `aggregation_from_upstream_artifacts`.",
    "field_provenance": "Every required field traces to exact paths, hashes, receipts, commands, or classifications.",
    "test_commands": (
        "Commands document focused unit, coverage, YAML parse, exact-path/hash, duplicate-history, "
        "verifier, exclusion-manifest, range-collision, protected-file, spec, E2E applicability, "
        "root-clutter, and full-suite checks."
    ),
    "test_exit_codes": "Exit codes prevent failed checks from being reported as success.",
    "reproducibility_checksum": "A checksum detects later ledger, artifact, or allocation drift.",
    "honest_verdict": "use a `complete:` or `blocked:` prefix.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); p=pathlib.Path('research-roadmap-next.yaml'); yaml.safe_load(p.read_text()) if p.exists() else None; yaml.safe_load(pathlib.Path('research-complete.yaml').read_text()); yaml.safe_load(pathlib.Path('ops/exclusion_manifest.yaml').read_text())\"",
    ".venv/bin/pytest tests/python/test_experiment_5918_transition_v526.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5918_transition_v526.py -m pytest tests/python/test_experiment_5918_transition_v526.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5918_transition_v526.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json <present .525 declared deliverables>",
    ".venv/bin/python scripts/check_exclusion_manifest.py",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py --check",
    ".venv/bin/pytest tests/python -q",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def path_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
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
    if not isinstance(payload, dict):
        meta["error"] = "json_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
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
        meta["error"] = f"yaml_error:{exc.__class__.__name__}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "yaml_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _history_blocks(root: Path) -> list[JsonMap]:
    payload, _meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    blocks = payload.get("milestones")
    return [block for block in blocks if isinstance(block, Mapping)] if isinstance(blocks, list) else []


def _task_signature(block: JsonMap) -> tuple[tuple[str, str], ...]:
    tasks = block.get("tasks")
    if not isinstance(tasks, list):
        return ()
    return tuple(
        (str(row.get("id")), str(row.get("deliverable") or ""))
        for row in tasks
        if isinstance(row, Mapping)
    )


def _duplicate_history_block_count(blocks: Sequence[JsonMap]) -> int:
    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
    for block in blocks:
        grouped[(str(block.get("id")), _task_signature(block))] += 1
    return sum(count - 1 for count in grouped.values() if count > 1)


def _completion_block_text() -> str:
    def q(value: str) -> str:
        return json.dumps(value, ensure_ascii=True)

    task_lines: list[str] = []
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        task_lines.extend(
            [
                f"  - id: {q(task_id)}",
                f"    title: {q(ACTIVATED_TASK_TITLES[task_id])}",
                f"    deliverable: {q(rel_path.as_posix())}",
                f"    result: {q(EXPECTED_TERMINAL_CLASSES[task_id])}",
            ]
        )
    return "\n".join(
        [
            f"- id: {q(MILESTONE_FROM)}",
            f"  title: {q(MILESTONE_FROM_TITLE)}",
            f"  doc: {q(ROADMAP_DOC_RELATIVE_PATH.as_posix())}",
            "  completed: '2026-07-25'",
            "  finding: Terminal outcomes preserved by transition artifact.",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _append_completion_if_absent(root: Path, terminal: bool) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    before_meta = _read_yaml_mapping(path)[1]
    before_blocks = _history_blocks(root)
    before_duplicate_count = _duplicate_history_block_count(before_blocks)
    if not terminal:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "nonterminal_identity_present",
            "before_sha256": before_meta["sha256"],
            "after_sha256": before_meta["sha256"],
            "before_duplicate_block_count": before_duplicate_count,
            "after_duplicate_block_count": before_duplicate_count,
            "duplicate_history_amplification_count": 0,
        }
    if [block for block in before_blocks if block.get("id") == MILESTONE_FROM]:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "exact_milestone_block_present",
            "before_sha256": before_meta["sha256"],
            "after_sha256": before_meta["sha256"],
            "before_duplicate_block_count": before_duplicate_count,
            "after_duplicate_block_count": before_duplicate_count,
            "duplicate_history_amplification_count": 0,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        separator = "" if existing.endswith("\n") else "\n"
        path.write_text(existing + separator + _completion_block_text(), encoding="utf-8")
    else:
        path.write_text("milestones:\n" + _completion_block_text(), encoding="utf-8")
    after_blocks = _history_blocks(root)
    after_duplicate_count = _duplicate_history_block_count(after_blocks)
    return {
        "append_count": 1,
        "appended": True,
        "reason": "exact_milestone_block_absent",
        "before_sha256": before_meta["sha256"],
        "after_sha256": path_sha256(path),
        "before_duplicate_block_count": before_duplicate_count,
        "after_duplicate_block_count": after_duplicate_count,
        "duplicate_history_amplification_count": max(0, after_duplicate_count - before_duplicate_count),
    }


def _capstone_payload(root: Path) -> tuple[JsonDict, JsonDict]:
    return _read_json_mapping(root / EXP5917_CAPSTONE_RELATIVE_PATH)


def _capstone_task_rows(capstone: JsonMap) -> list[JsonMap]:
    matrix = capstone.get("activated_task_and_declared_deliverable_matrix")
    if not isinstance(matrix, Mapping):
        return []
    tasks = matrix.get("tasks")
    return [row for row in tasks if isinstance(row, Mapping)] if isinstance(tasks, list) else []


def _capstone_terminal_classes(capstone: JsonMap) -> JsonDict:
    classes = capstone.get("exact_terminal_classification")
    if not isinstance(classes, Mapping):
        return {}
    return {
        "terminal_class_by_task_id": {
            str(key): str(value)
            for key, value in (classes.get("terminal_class_by_task_id") or {}).items()
        }
        if isinstance(classes.get("terminal_class_by_task_id"), Mapping)
        else {},
        "terminal_subclass_by_task_id": {
            str(key): str(value)
            for key, value in (classes.get("terminal_subclass_by_task_id") or {}).items()
        }
        if isinstance(classes.get("terminal_subclass_by_task_id"), Mapping)
        else {},
    }


def _artifact_payloads(
    root: Path, capstone_rows: Sequence[JsonMap]
) -> tuple[dict[str, JsonDict], dict[str, JsonDict], dict[str, JsonDict]]:
    by_task = {str(row.get("task_id") or row.get("id")): row for row in capstone_rows}
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    row_info: dict[str, JsonDict] = {}
    for task_id, expected_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = by_task.get(task_id, {})
        declared = row.get("declared_deliverable") or row.get("deliverable")
        rel_path = Path(str(declared)) if isinstance(declared, str) else expected_path
        payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        meta["expected_deliverable"] = expected_path.as_posix()
        meta["declared_path_matches_expected"] = rel_path == expected_path
        payloads[task_id] = payload
        metadata[task_id] = meta
        row_info[task_id] = {
            "title": str(row.get("title") or ACTIVATED_TASK_TITLES[task_id]),
            "capstone_declared_present": bool(row.get("declared_deliverable_present", False)),
            "capstone_declared_loadable": bool(row.get("declared_deliverable_loadable", False)),
            "capstone_status": str(row.get("status") or ""),
            "capstone_honest_verdict": str(row.get("honest_verdict") or ""),
            "conductor": dict(row.get("conductor") or {}) if isinstance(row.get("conductor"), Mapping) else {},
        }
    return payloads, metadata, row_info


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    stdout_json = receipt.get("stdout_json")
    if not isinstance(stdout_json, Mapping):
        return []
    reports = stdout_json.get("reports")
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        flags = reports[0].get("flags")
        return [dict(flag) for flag in flags if isinstance(flag, Mapping)] if isinstance(flags, list) else []
    return []


def _receipt_flag_count(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("flag_count") or 0)
        return int(stdout_json.get("flagged_count") or 0)
    return int(receipt.get("flag_count") or 0)


def _receipt_max_severity(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("max_severity", -1))
    return int(receipt.get("max_severity", -1))


def _complete_receipt(row: JsonMap) -> JsonDict:
    receipt = dict(row)
    receipt["flag_count"] = _receipt_flag_count(receipt)
    receipt["max_severity"] = _receipt_max_severity(receipt)
    receipt["flags"] = _receipt_flags(receipt)
    receipt.setdefault("receipt_hash", sha256_json(receipt.get("stdout_json", {})))
    return receipt


def _normalize_adversarial_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None,
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    if receipts is None:
        return {}
    source = receipts.values() if isinstance(receipts, Mapping) else receipts
    rows: dict[str, JsonDict] = {}
    for row in source:
        if isinstance(row, Mapping) and row.get("task_id"):
            task_id = str(row["task_id"])
            if metadata.get(task_id, {}).get("present"):
                rows[task_id] = _complete_receipt(row)
    return rows


def run_live_adversarial_receipts(
    root: Path, metadata: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:  # pragma: no cover
    executable = (
        (root / ".venv/bin/python").as_posix()
        if (root / ".venv/bin/python").exists()
        else sys.executable
    )
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if not metadata.get(task_id, {}).get("present"):
            continue
        command = [
            executable,
            ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
            "--json",
            rel_path.as_posix(),
        ]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: Any = json.loads(result.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": "stdout_not_json", "stdout": result.stdout}
        receipts[task_id] = _complete_receipt(
            {
                "task_id": task_id,
                "artifact_path": rel_path.as_posix(),
                "command": " ".join(command),
                "exit_code": result.returncode,
                "stdout_json": stdout_json,
                "stderr": result.stderr,
                "receipt_hash": sha256_json(stdout_json),
            }
        )
    return receipts


def _fallback_terminal_class(task_id: str, payload: JsonMap, meta: JsonMap) -> str:
    if not meta.get("present"):
        return "missing"
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    schema = str(payload.get("schema") or "")
    if status == "blocked_precondition" or verdict.startswith("blocked_precondition"):
        return "blocked-precondition"
    if schema == "blocked_gate_check_v1" or verdict.startswith("blocked_gate") or payload.get("gates_evaluated"):
        return "gate-blocked"
    if status == "retired" or verdict.startswith("retired:"):
        return "retired"
    if status == "complete_null" or verdict.startswith("complete_null"):
        return "null"
    if status.startswith("blocked") or verdict.startswith("blocked:"):
        return "blocked"
    if status.startswith(("complete", "ready")) or verdict.startswith(("complete:", "ready:")):
        return "positive"
    return "missing"


def _terminal_class_from_capstone(task_id: str, capstone_classes: JsonMap) -> str | None:
    raw = capstone_classes.get("terminal_class_by_task_id")
    subclasses = capstone_classes.get("terminal_subclass_by_task_id")
    raw_class = raw.get(task_id) if isinstance(raw, Mapping) else None
    subclass = subclasses.get(task_id) if isinstance(subclasses, Mapping) else None
    if subclass == "blocked-precondition":
        return "blocked-precondition"
    return str(raw_class) if isinstance(raw_class, str) and raw_class else None


def _exact_terminal_classification(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    capstone_classes: JsonMap,
) -> JsonDict:
    allowed = (
        "positive",
        "null",
        "blocked",
        "blocked-precondition",
        "retired",
        "gate-blocked",
        "missing",
        "unsafe",
    )
    by_task: dict[str, str] = {}
    by_class: dict[str, list[str]] = {name: [] for name in allowed}
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        terminal = _terminal_class_from_capstone(task_id, capstone_classes)
        terminal = terminal or _fallback_terminal_class(
            task_id, payloads.get(task_id, {}), metadata.get(task_id, {})
        )
        by_task[task_id] = terminal
        by_class.setdefault(terminal, []).append(task_id)
    nonterminal = [task_id for task_id, terminal in by_task.items() if terminal not in allowed]
    return {
        "terminal_class_by_task_id": by_task,
        "task_ids_by_terminal_class": by_class,
        "expected_terminal_class_by_task_id": dict(EXPECTED_TERMINAL_CLASSES),
        "disjoint_terminal_class_count": len(by_task),
        "all_activated_terminal": not nonterminal and len(by_task) == len(ACTIVATED_TASK_ARTIFACT_PATHS),
        "nonterminal_task_ids": nonterminal,
        "classification_source": EXP5917_CAPSTONE_RELATIVE_PATH.as_posix(),
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _activated_matrix(
    metadata: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    row_info: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    matrix: dict[str, JsonDict] = {}
    for task_id, expected_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        meta = metadata[task_id]
        payload = payloads[task_id]
        matrix[task_id] = {
            "identity": [MILESTONE_FROM, task_id, meta["declared_deliverable"]],
            "milestone": MILESTONE_FROM,
            "task_id": task_id,
            "title": row_info[task_id]["title"],
            "declared_deliverable": meta["declared_deliverable"],
            "expected_deliverable": expected_path.as_posix(),
            "declared_path_matches_expected": bool(meta["declared_path_matches_expected"]),
            "selection_policy": ARTIFACT_SELECTION_POLICY,
            "activated": True,
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "status": str(payload.get("status") or row_info[task_id]["capstone_status"]),
            "honest_verdict": str(
                payload.get("honest_verdict") or row_info[task_id]["capstone_honest_verdict"]
            ),
            "capstone_declared_present": row_info[task_id]["capstone_declared_present"],
            "capstone_declared_loadable": row_info[task_id]["capstone_declared_loadable"],
            "conductor": row_info[task_id]["conductor"],
            "terminal_evidence_source": "declared_deliverable_path_plus_exp5917_capstone_class",
        }
    return matrix


def _blocked_retired_gate_missing_receipts(matrix: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    by_task = classes["terminal_class_by_task_id"]
    blocked = [task_id for task_id, terminal in by_task.items() if terminal == "blocked"]
    blocked_pre = [
        task_id for task_id, terminal in by_task.items() if terminal == "blocked-precondition"
    ]
    retired = [task_id for task_id, terminal in by_task.items() if terminal == "retired"]
    gate_blocked = [task_id for task_id, terminal in by_task.items() if terminal == "gate-blocked"]
    missing = [task_id for task_id, row in matrix.items() if not row["present"]]
    receipts: list[JsonDict] = []
    for task_id in [*blocked, *blocked_pre, *retired, *gate_blocked, *missing]:
        if any(row.get("task_id") == task_id for row in receipts):
            continue
        row = matrix[task_id]
        receipts.append(
            {
                "task_id": task_id,
                "declared_deliverable": row["declared_deliverable"],
                "terminal_class": by_task.get(task_id),
                "present": row["present"],
                "honest_verdict": row["honest_verdict"],
                "conductor": row["conductor"],
                "treated_as_success": False,
            }
        )
    return {
        "blocked_task_ids": blocked,
        "blocked_precondition_task_ids": blocked_pre,
        "retired_task_ids": retired,
        "gate_blocked_task_ids": gate_blocked,
        "missing_declared_deliverable_task_ids": missing,
        "receipts": receipts,
        "principle": FIELD_PRINCIPLES["blocked_retired_gate_blocked_and_missing_receipts"],
    }


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    paths = sorted(set(SOURCE_HASH_PATHS), key=lambda value: value.as_posix())
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in paths
    }


def _resource_receipts(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    memory_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                memory_mb = int(line.split()[1]) // 1024
                break
    return {
        "disk": {
            "available_mb": disk.free // (1024 * 1024),
            "required_mb": 512,
            "ok": disk.free >= 512 * 1024 * 1024,
        },
        "memory": {
            "available_mb": memory_mb,
            "required_mb": 512,
            "ok": memory_mb == 0 or memory_mb >= 512,
        },
    }


def _atomic_output_receipt(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".tmp-probe")
    probe.write_text("atomic-probe\n", encoding="utf-8")
    ok = probe.read_text(encoding="utf-8") == "atomic-probe\n"
    probe.unlink()
    return {
        "declared_path": path.as_posix(),
        "parent_exists": path.parent.exists(),
        "parent_writable": path.parent.exists() and path.parent.is_dir(),
        "atomic_probe_write_ok": ok,
        "ok": ok,
    }


def _protected_file_hashes(root: Path) -> dict[str, str | None]:
    return {rel_path.as_posix(): path_sha256(root / rel_path) for rel_path in PROTECTED_FILE_PATHS}


def _protected_files_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    files: dict[str, JsonDict] = {}
    after = _protected_file_hashes(root)
    for rel_path in PROTECTED_FILE_PATHS:
        key = rel_path.as_posix()
        files[key] = {
            "present": (root / rel_path).exists(),
            "sha256_before": before.get(key),
            "sha256_after": after.get(key),
            "unchanged": before.get(key) == after.get(key),
        }
    return {
        "files": files,
        "all_unchanged": all(row["unchanged"] for row in files.values()),
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _exp5904_separate_evidence_receipt(root: Path) -> JsonDict:
    work_paths = (EXP5904_SOURCE_RELATIVE_PATH, EXP5904_RESULT_RELATIVE_PATH)
    existing_paths = [path.as_posix() for path in work_paths if (root / path).exists()]
    return {
        "exp5904_separate": bool(existing_paths),
        "separation_reason": "concurrent_coordinate_router_evidence_outside_activated_525_range",
        "existing_paths": existing_paths,
        "path_hashes": {
            path.as_posix(): path_sha256(root / path)
            for path in work_paths
            if (root / path).exists()
        },
        "edited_by_exp5918": False,
        "required_by_exp5918": False,
        "classified_by_exp5918": False,
        "appended_as_v525_task": False,
        "included_in_activated_matrix": False,
        "included_in_next_range": False,
        "principle": FIELD_PRINCIPLES["exp5904_separate_evidence_receipt"],
    }


def _next_range_numbers_in_text(text: str) -> set[int]:
    lowered = text.lower()
    if not any(marker in lowered for marker in ("exp591", "exp592", "exp593", "experiment_591", "experiment_592", "experiment_593")):
        return set()
    numbers: set[int] = set()
    for number in NEXT_RANGE_NUMBERS:
        if (
            re.search(rf"(?<![a-z0-9_])exp{number}(?![a-z0-9])", lowered)
            or re.search(rf"(?<![a-z0-9_])experiment_{number}(?![a-z0-9])", lowered)
        ):
            numbers.add(number)
    return numbers


def _scan_candidate_paths(root: Path) -> list[Path]:
    candidates = [
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        ROADMAP_DOC_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    ]
    for folder in ("python", "tests", "scripts", "openspec/change-proposals", "ops"):
        base = root / folder
        if base.exists():
            candidates.extend(
                path.relative_to(root)
                for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    results = root / "results"
    if results.exists():
        candidates.extend(path.relative_to(root) for path in results.glob("experiment_*") if path.is_file())
    return sorted(set(candidates), key=lambda value: value.as_posix())


def _allowed_range_reference_kind(rel_path: Path, numbers: set[int]) -> str | None:
    if rel_path in OWNED_REFERENCE_PATHS:
        return "transition_owned_reference"
    if rel_path in ALLOWED_ALLOCATION_REFERENCE_PATHS:
        return "allowed_allocation_reference"
    if rel_path == CONDUCTOR_LOG_RELATIVE_PATH and numbers and numbers <= {5918}:
        return "transition_owned_conductor_attempt_reference"
    return None


def _range_collision_scan(root: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed: list[JsonDict] = []
    for rel_path in _scan_candidate_paths(root):
        path = root / rel_path
        text = rel_path.as_posix()
        if path.exists() and path.stat().st_size < 2_000_000:
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
        numbers = _next_range_numbers_in_text(text)
        if not numbers:
            continue
        kind = _allowed_range_reference_kind(rel_path, numbers)
        row = {"path": rel_path.as_posix(), "kind": kind or "unexpected_next_range_reference"}
        if kind:
            row["numbers"] = sorted(numbers)
            allowed.append(row)
        else:
            collisions.append(row)
    return {
        "range": {"start": 5918, "end": 5931},
        "collision_count": len(collisions),
        "collisions": collisions,
        "allowed_references": allowed,
        "principle": FIELD_PRINCIPLES["next_range_collision_count"],
    }


def _docs_reconciled(root: Path) -> JsonDict:
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "openspec_research_reporting_req_5918_present": "REQ-REPORT-5918" in spec_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
    }


def _field_provenance() -> dict[str, JsonDict]:
    base_sources = [
        EXP5917_CAPSTONE_RELATIVE_PATH.as_posix(),
        ROADMAP_RELATIVE_PATH.as_posix(),
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": base_sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_rows(tests_run: Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [
            {"command": command, "exit_code": None, "status": "not_recorded"}
            for command in DEFAULT_TEST_COMMANDS
        ]
    return [dict(row) for row in tests_run]


def _failed_required_test_commands(rows: Sequence[JsonMap]) -> list[str]:
    return [
        str(row.get("command"))
        for row in rows
        if row.get("blocking", True) is not False
        and isinstance(row.get("exit_code"), int)
        and int(row["exit_code"]) != 0
    ]


def _adversarial_receipts_group(
    receipts: Mapping[str, JsonMap], matrix: Mapping[str, JsonMap]
) -> JsonDict:
    reports: list[JsonDict] = []
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        row = matrix[task_id]
        if not row["present"]:
            continue
        receipt = receipts.get(task_id)
        if not isinstance(receipt, Mapping):
            continue
        reports.append(
            {
                "task_id": task_id,
                "artifact": row["declared_deliverable"],
                "command": str(receipt.get("command") or ""),
                "exit_code": receipt.get("exit_code"),
                "loaded": True,
                "flag_count": _receipt_flag_count(receipt),
                "max_severity": _receipt_max_severity(receipt),
                "flags": _receipt_flags(receipt),
                "receipt_hash": str(receipt.get("receipt_hash") or ""),
            }
        )
    return {
        "reports": reports,
        "verified_present_declared_deliverable_count": len(reports),
        "missing_declared_deliverables_not_verified": [
            row["declared_deliverable"] for row in matrix.values() if not row["present"]
        ],
        "failed_receipt_task_ids": [
            row["task_id"]
            for row in reports
            if not isinstance(row.get("exit_code"), int) or row.get("exit_code") != 0
        ],
        "flagged_count": sum(int(row["flag_count"]) for row in reports),
        "principle": FIELD_PRINCIPLES["adversarial_verifier_receipts"],
    }


def _status_and_verdict(failed_preconditions: Sequence[str], classes: JsonMap) -> tuple[str, str]:
    if failed_preconditions:
        return "blocked", "blocked: Exp5918 transition preconditions failed"
    by_task = classes.get("terminal_class_by_task_id", {})
    if any(value == "unsafe" for value in by_task.values()):
        return "blocked", "blocked: unsafe .525 identity present"
    if any(
        value in {"null", "blocked", "blocked-precondition", "retired", "gate-blocked", "missing"}
        for value in by_task.values()
    ):
        return (
            "complete_with_terminal_receipts",
            "complete: archived terminal .525 identities into .526 without outcome laundering; next_range_collision_count=0",
        )
    return "complete", "complete: archived terminal .525 identities into .526 with collision-free allocation"


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.monotonic()
    root = root.resolve()
    protected_before = _protected_file_hashes(root)
    active_roadmap, active_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_next, roadmap_next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    complete_meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)[1]
    exclusion_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)[1]
    capstone, capstone_meta = _capstone_payload(root)
    capstone_rows = _capstone_task_rows(capstone)
    capstone_classes = _capstone_terminal_classes(capstone)
    payloads, metadata, row_info = _artifact_payloads(root, capstone_rows)
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = run_live_adversarial_receipts(root, metadata)
    matrix = _activated_matrix(metadata, payloads, row_info)
    classes = _exact_terminal_classification(payloads, metadata, capstone_classes)
    append_receipt = _append_completion_if_absent(root, bool(classes["all_activated_terminal"]))
    range_scan = _range_collision_scan(root)
    protected = _protected_files_unchanged(root, protected_before)
    exp5904 = _exp5904_separate_evidence_receipt(root)
    test_rows = _tests_run_rows(tests_run)
    failed_tests = _failed_required_test_commands(test_rows)
    verifier_group = _adversarial_receipts_group(receipts, matrix)
    present_task_ids = [task_id for task_id, row in matrix.items() if row["present"]]
    receipt_task_ids = {row["task_id"] for row in verifier_group["reports"]}
    missing_receipts = [task_id for task_id in present_task_ids if task_id not in receipt_task_ids]
    resources = _resource_receipts(root)
    atomic = _atomic_output_receipt(root / RESULT_RELATIVE_PATH)
    failed_preconditions: list[str] = []
    if active_meta["present"] and not active_meta["loadable"]:
        failed_preconditions.append("active_roadmap_unloadable")
    if active_meta["loadable"] and active_roadmap.get("milestone") != MILESTONE_TO:
        failed_preconditions.append("active_roadmap_milestone_mismatch")
    if roadmap_next_meta["present"] and not roadmap_next_meta["loadable"]:
        failed_preconditions.append("roadmap_next_unloadable")
    if complete_meta["present"] and not complete_meta["loadable"]:
        failed_preconditions.append("research_complete_unparseable")
    if exclusion_meta["present"] and not exclusion_meta["loadable"]:
        failed_preconditions.append("exclusion_manifest_unparseable")
    if not capstone_meta["loadable"]:
        failed_preconditions.append("exp5917_capstone_unreadable")
    if not (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists():
        failed_preconditions.append("live_verifier_missing")
    if range_scan["collision_count"] != 0:
        failed_preconditions.append("next_range_collision")
    if append_receipt["duplicate_history_amplification_count"] != 0:
        failed_preconditions.append("duplicate_history_amplified")
    if not exp5904["exp5904_separate"]:
        failed_preconditions.append("exp5904_not_separate")
    if classes["terminal_class_by_task_id"] != EXPECTED_TERMINAL_CLASSES:
        failed_preconditions.append("terminal_outcomes_not_preserved")
    if missing_receipts:
        failed_preconditions.append("missing_adversarial_receipts")
    if verifier_group["failed_receipt_task_ids"]:
        failed_preconditions.append("adversarial_verifier_failed")
    if failed_tests:
        failed_preconditions.append("required_tests_failed")
    if not protected["all_unchanged"]:
        failed_preconditions.append("protected_file_modified")
    if not resources["disk"]["ok"] or not resources["memory"]["ok"]:
        failed_preconditions.append("insufficient_resources")
    if not atomic["ok"]:
        failed_preconditions.append("atomic_output_unavailable")
    status, verdict = _status_and_verdict(failed_preconditions, classes)
    result_duration = duration_s if duration_s is not None else round(time.monotonic() - start, 6)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE_TO,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": {
            "active_roadmap": {
                **active_meta,
                "milestone": active_roadmap.get("milestone") if active_meta["loadable"] else None,
                "task_count": len(active_roadmap.get("tasks", []))
                if isinstance(active_roadmap.get("tasks"), list)
                else 0,
            },
            "roadmap_next": {
                **roadmap_next_meta,
                "milestone": roadmap_next.get("milestone") if roadmap_next_meta["loadable"] else None,
            },
            "research_complete": complete_meta,
            "conductor_log": {
                "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "present": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
            },
            "exclusion_manifest": exclusion_meta,
            "exp5917_capstone": capstone_meta,
            "source_hashes": _source_hashes(root),
            "declared_present_deliverable_hashes": {
                task_id: row["sha256"] for task_id, row in matrix.items() if row["present"]
            },
            "resource_receipts": resources,
            "atomic_output": atomic,
            "adversarial_verifier_available": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
            "range_collision_scan": range_scan,
            "protected_file_hashes_before": protected_before,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_transition": {
            "source_milestone": MILESTONE_FROM,
            "destination_milestone": MILESTONE_TO,
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "activated_task_and_deliverable_matrix": matrix,
        "exact_terminal_classification": classes,
        "blocked_retired_gate_blocked_and_missing_receipts": _blocked_retired_gate_missing_receipts(
            matrix, classes
        ),
        "adversarial_verifier_receipts": verifier_group,
        "research_complete_append_count": append_receipt["append_count"],
        "duplicate_history_amplification_count": append_receipt[
            "duplicate_history_amplification_count"
        ],
        "research_complete_append_receipt": append_receipt,
        "exp5904_separate_evidence_receipt": exp5904,
        "next_task_range": {
            "start": "exp5918",
            "end": "exp5931",
            "count": len(NEXT_TASK_ARTIFACT_PATHS),
            "task_ids": list(NEXT_TASK_ARTIFACT_PATHS),
        },
        "next_range_collision_count": range_scan["collision_count"],
        "docs_reconciled": _docs_reconciled(root),
        "protected_files_unchanged": protected,
        "duration_s": result_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": [str(row.get("command", "")) for row in test_rows],
        "test_exit_codes": {str(row.get("command", "")): row.get("exit_code") for row in test_rows},
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required field: {missing[0]}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if not isinstance(payload.get("next_range_collision_count"), int):
        raise ValueError("next_range_collision_count must be a bare integer")
    if payload["next_range_collision_count"] != 0:
        raise ValueError("next_range_collision_count must be zero")
    if payload.get("research_complete_append_count") not in {0, 1}:
        raise ValueError("research_complete_append_count must be zero or one")
    if payload.get("duplicate_history_amplification_count") != 0:
        raise ValueError("duplicate_history_amplification_count must be zero")
    matrix = payload.get("activated_task_and_deliverable_matrix")
    if isinstance(matrix, Mapping) and "exp5904-click-target-discrimination" in matrix:
        raise ValueError("Exp5904 must not be in activated matrix")
    if not isinstance(matrix, Mapping) or len(matrix) != 13:
        raise ValueError("activated matrix must contain exactly thirteen .525 identities")
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = matrix.get(task_id)
        if not isinstance(row, Mapping):
            raise ValueError("activated matrix must contain exactly thirteen .525 identities")
        if row.get("identity") != [MILESTONE_FROM, task_id, rel_path.as_posix()]:
            raise ValueError("activated identity mismatch")
    classes = payload.get("exact_terminal_classification")
    if not isinstance(classes, Mapping):
        raise ValueError("terminal classes missing")
    if classes.get("terminal_class_by_task_id") != EXPECTED_TERMINAL_CLASSES:
        raise ValueError("terminal classes do not preserve .525 outcomes")
    receipts = payload.get("blocked_retired_gate_blocked_and_missing_receipts")
    if not isinstance(receipts, Mapping):
        raise ValueError("terminal receipt missing")
    for row in receipts.get("receipts", []):
        if not isinstance(row, Mapping) or row.get("treated_as_success") is not False:
            raise ValueError("terminal receipt treated as success")
    verifier = payload.get("adversarial_verifier_receipts")
    if not isinstance(verifier, Mapping):
        raise ValueError("adversarial verifier receipts missing")
    present_count = sum(1 for row in matrix.values() if isinstance(row, Mapping) and row.get("present"))
    if verifier.get("verified_present_declared_deliverable_count") != present_count:
        raise ValueError("missing adversarial verifier receipt")
    for row in verifier.get("reports", []):
        if not isinstance(row, Mapping) or not row.get("receipt_hash"):
            raise ValueError("missing adversarial verifier receipt fields")
        if "scripts/adversarial_verify.py" not in str(row.get("command") or ""):
            raise ValueError("adversarial verifier receipt command must run scripts/adversarial_verify.py")
    exp5904 = payload.get("exp5904_separate_evidence_receipt")
    if not isinstance(exp5904, Mapping):
        raise ValueError("Exp5904 separate evidence missing")
    if (
        exp5904.get("exp5904_separate") is not True
        or exp5904.get("edited_by_exp5918") is not False
        or exp5904.get("classified_by_exp5918") is not False
        or exp5904.get("appended_as_v525_task") is not False
        or exp5904.get("included_in_activated_matrix") is not False
    ):
        raise ValueError("Exp5904 separate evidence was laundered")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected file changed")
    for row in protected.get("files", {}).values():
        if isinstance(row, Mapping) and row.get("unchanged") is not True:
            raise ValueError("protected file changed")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field provenance missing")
    for field in REQUIRED_ARTIFACT_FIELDS:
        row = provenance.get(field)
        if not isinstance(row, Mapping) or row.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field provenance missing for {field}")
    if payload_checksum(payload) != payload.get("reproducibility_checksum"):
        raise ValueError("checksum mismatch")


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
) -> JsonDict:
    root = root.resolve()
    output_path = output_path or root / RESULT_RELATIVE_PATH
    report = build_report(root, adversarial_receipts=adversarial_receipts, tests_run=tests_run)
    write_json(output_path, report)
    return report


def _load_tests_run(path: Path | None) -> list[JsonDict]:  # pragma: no cover
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in data if isinstance(row, Mapping)]


def _load_receipts(path: Path | None) -> list[JsonDict] | None:  # pragma: no cover
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("adversarial receipts JSON must be a list")
    return [dict(row) for row in data if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    parser.add_argument("--adversarial-receipts-json", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(
        args.root,
        output_path=args.output,
        adversarial_receipts=_load_receipts(args.adversarial_receipts_json),
        tests_run=_load_tests_run(args.tests_run_json),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
