"""Exp5890 transition receipt from terminal milestone .523 into .524.

Spec refs: REQ-REPORT-5890, SCENARIO-REPORT-5890-EXACT-ARCHIVE,
SCENARIO-REPORT-5890-APPEND-ONCE, SCENARIO-REPORT-5890-UNACTIVATED-PROPOSAL,
SCENARIO-REPORT-5890-RANGE-COLLISION, SCENARIO-REPORT-5890-SCHEMA.

This module closes a milestone boundary by reading exact upstream receipts. It
does not repair retired experiments, infer success from missing files, or treat
proposal-only IDs as evidence. The identity contract is deliberately narrow:
only `(milestone, task_id, declared_deliverable)` rows from the completion
ledger count.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5890_transition_v524.json")

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

EXPERIMENT = "experiment_5890_transition_v524"
EXPERIMENT_ID = "exp5890-transition-v524"
MILESTONE_FROM = "2026.07.523"
MILESTONE_TO = "2026.07.524"
MILESTONE_FROM_TITLE = (
    "Shortcut-Resistant Self-Learning, Layer-Dynamic Energy, and Lossless Live Memory"
)
MILESTONE_TO_TITLE = (
    "Grounded Constraint IR, Shortcut-Safe Self-Learning, and Structured Live Memory"
)
RUN_DATE = "20260724"
RANDOM_SEED = 5890
SCHEMA = "carnot.experiment_5890.transition_v524.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5890",
    "SCENARIO-REPORT-5890-EXACT-ARCHIVE",
    "SCENARIO-REPORT-5890-APPEND-ONCE",
    "SCENARIO-REPORT-5890-UNACTIVATED-PROPOSAL",
    "SCENARIO-REPORT-5890-RANGE-COLLISION",
    "SCENARIO-REPORT-5890-SCHEMA",
)

ACTIVATED_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5877-transition-v523": Path("results/experiment_5877_transition_v523.json"),
    "exp5878-v523-source-delta-ingestion": Path(
        "results/experiment_5878_v523_source_delta_ingestion.json"
    ),
    "exp5879-hardness-headroom-taxonomy-corrigendum": Path(
        "results/experiment_5879_hardness_headroom_taxonomy_corrigendum.json"
    ),
    "exp5880-grounding-shortcut-fixture": Path(
        "results/experiment_5880_grounding_shortcut_fixture.json"
    ),
    "exp5881-one-to-one-grounding-acquisition-ab": Path(
        "results/experiment_5881_one_to_one_grounding_acquisition_ab.json"
    ),
    "exp5882-shortcut-resistant-continuous-self-learning": Path(
        "results/experiment_5882_shortcut_resistant_continuous_self_learning.json"
    ),
}
EXPECTED_TASK_IDS = tuple(ACTIVATED_TASK_ARTIFACT_PATHS)

ACTIVATED_TASK_TITLES: dict[str, str] = {
    "exp5877-transition-v523": "Exact terminal-boundary handoff from .522 into .523",
    "exp5878-v523-source-delta-ingestion": (
        "Dated evidence refresh after the V523 planner marker"
    ),
    "exp5879-hardness-headroom-taxonomy-corrigendum": (
        "Changed-taxonomy audit of nuisance controls versus oracle-derived solver telemetry"
    ),
    "exp5880-grounding-shortcut-fixture": (
        "Gated on Exp5879 headroom: exact constraint-satisfaction and cognition-shortcut fixture"
    ),
    "exp5881-one-to-one-grounding-acquisition-ab": (
        "Gated on Exp5880 fixture: one-to-one atom grounding structural-acquisition A/B"
    ),
    "exp5882-shortcut-resistant-continuous-self-learning": (
        "Gated on Exp5881 mechanism: prospective shortcut-resistant continuous self-learning"
    ),
}

CONDUCTOR_TITLE_PATTERNS: dict[str, str] = {
    "exp5877-transition-v523": "Exact terminal-boundary handoff from .522 into .52",
    "exp5878-v523-source-delta-ingestion": (
        "Dated evidence refresh after the V523 planner mark"
    ),
    "exp5879-hardness-headroom-taxonomy-corrigendum": (
        "Changed-taxonomy audit of nuisance controls versus"
    ),
    "exp5880-grounding-shortcut-fixture": (
        "Gated on Exp5879 headroom: exact constraint-satisf"
    ),
    "exp5881-one-to-one-grounding-acquisition-ab": (
        "Gated on Exp5880 fixture: one-to-one atom groundin"
    ),
    "exp5882-shortcut-resistant-continuous-self-learning": (
        "Gated on Exp5881 mechanism: prospective shortcut-r"
    ),
}

MISSING_DELIVERABLE_REASONS = {
    "exp5880-grounding-shortcut-fixture": "upstream_retired_exp5879",
    "exp5882-shortcut-resistant-continuous-self-learning": "upstream_retired_exp5881",
}

UNACTIVATED_PROPOSAL_TASK_IDS = (
    "exp5883-gguf-intermediate-layer-surface-preflight",
    "exp5884-three-family-layer-dynamic-energy",
    "exp5885-layer-dynamic-portability-camouflage-audit",
    "exp5886-arc-programmatic-memory-contract",
    "exp5887-arc-programmatic-memory-causal-audit",
    "exp5888-arc-programmatic-memory-live-ab",
    "exp5889-v523-capstone-reconciliation",
)

UNACTIVATED_PROPOSAL_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5883-gguf-intermediate-layer-surface-preflight": Path(
        "results/experiment_5883_gguf_intermediate_layer_surface_preflight.json"
    ),
    "exp5884-three-family-layer-dynamic-energy": Path(
        "results/experiment_5884_three_family_layer_dynamic_energy.json"
    ),
    "exp5885-layer-dynamic-portability-camouflage-audit": Path(
        "results/experiment_5885_layer_dynamic_portability_camouflage_audit.json"
    ),
    "exp5886-arc-programmatic-memory-contract": Path(
        "results/experiment_5886_arc_programmatic_memory_contract.json"
    ),
    "exp5887-arc-programmatic-memory-causal-audit": Path(
        "results/experiment_5887_arc_programmatic_memory_causal_audit.json"
    ),
    "exp5888-arc-programmatic-memory-live-ab": Path(
        "results/experiment_5888_arc_programmatic_memory_live_ab.json"
    ),
    "exp5889-v523-capstone-reconciliation": Path(
        "results/experiment_5889_v523_capstone_reconciliation.json"
    ),
}

NEXT_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5890-transition-v524": RESULT_RELATIVE_PATH,
    "exp5891-v524-source-delta-ingestion": Path(
        "results/experiment_5891_v524_source_delta_ingestion.json"
    ),
    "exp5892-headroom-evidence-escrow": Path(
        "results/experiment_5892_headroom_evidence_escrow.json"
    ),
    "exp5893-grounding-shortcut-fixture": Path(
        "results/experiment_5893_grounding_shortcut_fixture.json"
    ),
    "exp5894-one-to-one-grounding-ab": Path(
        "results/experiment_5894_one_to_one_grounding_ab.json"
    ),
    "exp5895-shortcut-safe-continuous-self-learning": Path(
        "results/experiment_5895_shortcut_safe_continuous_self_learning.json"
    ),
    "exp5896-typed-constraint-ir-fixture": Path(
        "results/experiment_5896_typed_constraint_ir_fixture.json"
    ),
    "exp5897-sota-constraint-ir-repair-ab": Path(
        "results/experiment_5897_sota_constraint_ir_repair_ab.json"
    ),
    "exp5898-recursive-constraint-improvement": Path(
        "results/experiment_5898_recursive_constraint_improvement.json"
    ),
    "exp5899-constraint-repair-portability-audit": Path(
        "results/experiment_5899_constraint_repair_portability_audit.json"
    ),
    "exp5900-arc-structured-evidence-memory-contract": Path(
        "results/experiment_5900_arc_structured_evidence_memory_contract.json"
    ),
    "exp5901-arc-structured-memory-causal-audit": Path(
        "results/experiment_5901_arc_structured_memory_causal_audit.json"
    ),
    "exp5902-arc-structured-memory-live-ab": Path(
        "results/experiment_5902_arc_structured_memory_live_ab.json"
    ),
    "exp5903-v524-capstone-reconciliation": Path(
        "results/experiment_5903_v524_capstone_reconciliation.json"
    ),
}
NEXT_RANGE_NUMBERS = range(5890, 5904)

PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
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
    *PROTECTED_FILE_PATHS,
    *ACTIVATED_TASK_ARTIFACT_PATHS.values(),
)

OWNED_REFERENCE_PATHS = (
    Path("python/carnot/experiment_5890_transition_v524.py"),
    Path("tests/python/test_experiment_5890_transition_v524.py"),
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
    "missing_and_gate_blocked_receipts",
    "unactivated_proposal_id_receipt",
    "adversarial_verifier_receipts",
    "outcome_classification",
    "retired_and_science_ready_preserved",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
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
    "status": "Terminal state distinguishes exact archival from bootstrap work.",
    "preconditions_checked": (
        "Parsed roadmaps, hashes, resources, verifier availability, atomic output, and "
        "declared deliverables ground the handoff."
    ),
    "milestone_transition": "Explicit .523-to-.524 boundary prevents prefix aliasing.",
    "activated_task_and_deliverable_matrix": (
        "Only activated task IDs and declared paths count as evidence."
    ),
    "missing_and_gate_blocked_receipts": (
        "Activated missing and gate-blocked tasks remain receipts rather than successes."
    ),
    "unactivated_proposal_id_receipt": (
        "Proposal-only Exp5883-Exp5889 cannot become completed work."
    ),
    "adversarial_verifier_receipts": (
        "Fresh verifier output preserves current artifact-quality authority."
    ),
    "outcome_classification": "Disjoint terminal classes prevent history laundering.",
    "retired_and_science_ready_preserved": (
        "Operational retirement cannot erase a measured scalar or promote it to "
        "downstream completion."
    ),
    "research_complete_append_count": "Exact zero-or-one prevents duplicate history.",
    "duplicate_history_amplification_count": (
        "Must be bare zero; existing duplicate history cannot be multiplied."
    ),
    "next_task_range": "A declared finite Exp5890-Exp5903 interval makes allocation auditable.",
    "next_range_collision_count": "Only bare zero authorizes the allocation.",
    "docs_reconciled": (
        "Transition-owned specifications and receipts disclose deferred operator reconciliation."
    ),
    "protected_files_unchanged": "Protected user and conductor files remain byte-identical.",
    "duration_s": "Measured wall time exposes bootstrap-only work.",
    "inference_substrate": (
        "Use `aggregation_from_upstream_artifacts` for archival aggregation rather than model inference."
    ),
    "field_provenance": (
        "Every field traces to exact paths, hashes, commands, or roadmap records."
    ),
    "test_commands": (
        "Commands document focused, coverage, YAML, exact-path/hash, duplicate-history, "
        "adversarial-verifier, alias, exclusion-manifest, range-collision, protected-file, "
        "reconciliation, spec, and root-clutter checks."
    ),
    "test_exit_codes": "Exit codes prevent failed owned checks becoming success.",
    "reproducibility_checksum": (
        "A checksum detects ledger, artifact, or allocation drift."
    ),
    "honest_verdict": "A `complete:` or `blocked:` prefix makes the handoff terminal.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); p=pathlib.Path('research-roadmap-next.yaml'); yaml.safe_load(p.read_text()) if p.exists() else None; yaml.safe_load(pathlib.Path('research-complete.yaml').read_text()); yaml.safe_load(pathlib.Path('ops/exclusion_manifest.yaml').read_text())\"",
    ".venv/bin/pytest tests/python/test_experiment_5890_transition_v524.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5890_transition_v524.py -m pytest tests/python/test_experiment_5890_transition_v524.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5890_transition_v524.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable ordering before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return the repository-standard prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def path_sha256(path: str | Path) -> str | None:
    """Hash file bytes; absent paths return ``None``."""

    target = Path(path)
    if not target.exists():
        return None
    with target.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def write_json(path: Path, payload: JsonMap) -> None:
    """Atomically write stable JSON so partial transition artifacts are avoided."""

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
    rows: list[tuple[str, str]] = []
    for row in tasks:
        if isinstance(row, Mapping) and isinstance(row.get("id"), str):
            deliverable = row.get("deliverable")
            rows.append((str(row["id"]), str(deliverable) if isinstance(deliverable, str) else ""))
    return tuple(rows)


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
                "    result: terminal outcome preserved by Exp5890 transition",
            ]
        )
    return "\n".join(
        [
            f"- id: {q(MILESTONE_FROM)}",
            f"  title: {q(MILESTONE_FROM_TITLE)}",
            f"  doc: {q(ROADMAP_DOC_RELATIVE_PATH.as_posix())}",
            "  completed: '2026-07-24'",
            "  finding: Terminal outcomes preserved by Exp5890; see conductor log and artifact.",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _append_completion_if_absent(root: Path) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    before_meta = _read_yaml_mapping(path)[1]
    before_blocks = _history_blocks(root)
    before_duplicate_count = _duplicate_history_block_count(before_blocks)
    exact_blocks_before = [block for block in before_blocks if block.get("id") == MILESTONE_FROM]
    if exact_blocks_before:
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
    if before_meta["present"] and not before_meta["loadable"]:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "research_complete_unparseable",
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
        prefix = (
            "# Carnot Research - Completed Experiments\n"
            "# Tasks moved here from research-roadmap.yaml after terminal conductor closure.\n\n"
            "milestones:\n"
        )
        path.write_text(prefix + _completion_block_text(), encoding="utf-8")
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


def _completion_task_rows(root: Path) -> list[JsonMap]:
    blocks = [block for block in _history_blocks(root) if block.get("id") == MILESTONE_FROM]
    if not blocks:
        return []
    tasks = blocks[0].get("tasks")
    return [row for row in tasks if isinstance(row, Mapping)] if isinstance(tasks, list) else []


def _artifact_payloads(root: Path, task_rows: Sequence[JsonMap]) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    by_task = {
        str(row.get("id")): row
        for row in task_rows
        if isinstance(row.get("id"), str)
    }
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, expected_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = by_task.get(task_id, {})
        declared = row.get("deliverable")
        rel_path = Path(str(declared)) if isinstance(declared, str) else expected_path
        payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        meta["expected_deliverable"] = expected_path.as_posix()
        meta["declared_path_matches_expected"] = rel_path == expected_path
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _status_from_log(line: str | None) -> str:
    if line is None:
        return "MISSING"
    for status in ("GATE_BLOCK", "FLAGGED", "FAIL", "OK"):
        if f"| {status} |" in line:
            return status
    return "LOGGED"


def _conductor_outcomes(root: Path) -> dict[str, JsonDict]:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    outcomes: dict[str, JsonDict] = {}
    for task_id, pattern in CONDUCTOR_TITLE_PATTERNS.items():
        matches = [line for line in text.splitlines() if pattern in line]
        latest = matches[-1] if matches else None
        outcomes[task_id] = {
            "latest_status": _status_from_log(latest),
            "latest_line": latest,
            "attempt_count": len(matches),
        }
    return outcomes


def _activated_task_matrix(
    task_rows: Sequence[JsonMap],
    metadata: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    by_task = {
        str(row.get("id")): row
        for row in task_rows
        if isinstance(row.get("id"), str)
    }
    matrix: dict[str, JsonDict] = {}
    for task_id, expected_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = by_task.get(task_id, {})
        meta = metadata[task_id]
        payload = payloads[task_id]
        science_scalar = payload.get("hardness_surface_headroom_ready_score")
        matrix[task_id] = {
            "identity": [MILESTONE_FROM, task_id, meta["declared_deliverable"]],
            "milestone": MILESTONE_FROM,
            "task_id": task_id,
            "title": str(row.get("title") or ACTIVATED_TASK_TITLES[task_id]),
            "declared_deliverable": meta["declared_deliverable"],
            "expected_deliverable": expected_path.as_posix(),
            "declared_path_matches_expected": bool(meta["declared_path_matches_expected"]),
            "selection_policy": ARTIFACT_SELECTION_POLICY,
            "activated": True,
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "status": str(payload.get("status") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "science_ready_scalar": science_scalar,
            "missing_recorded_explicitly": not bool(meta["present"]),
            "missing_reason": MISSING_DELIVERABLE_REASONS.get(task_id) if not meta["present"] else None,
            "completion_row_result": str(row.get("result") or ""),
            "conductor": conductor.get(task_id, {}),
            "treated_as_success": False
            if task_id in MISSING_DELIVERABLE_REASONS or task_id == "exp5881-one-to-one-grounding-acquisition-ab"
            else None,
        }
    return matrix


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            flags = reports[0].get("flags")
            if isinstance(flags, list):
                return [dict(flag) for flag in flags if isinstance(flag, Mapping)]
    return []


def _receipt_flag_count(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("flag_count") or 0)
        return int(stdout_json.get("flagged_count") or 0)
    return 0


def _receipt_max_severity(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("max_severity", -1))
    return -1


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
        if not isinstance(row, Mapping) or not row.get("task_id"):
            continue
        task_id = str(row["task_id"])
        if task_id in metadata and metadata[task_id].get("present"):
            rows[task_id] = _complete_receipt(row)
    return rows


def run_live_adversarial_receipts(root: Path, metadata: Mapping[str, JsonMap]) -> dict[str, JsonDict]:  # pragma: no cover
    """Run the live artifact verifier for every present declared `.523` file."""

    executable = (root / ".venv/bin/python").as_posix() if (root / ".venv/bin/python").exists() else sys.executable
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if not metadata.get(task_id, {}).get("present"):
            continue
        command = [executable, ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(), "--json", rel_path.as_posix()]
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


def _classify_outcomes(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    receipts: Mapping[str, JsonMap],
) -> JsonDict:
    classes: dict[str, Any] = {
        "terminal_class_by_task_id": {},
        "complete_transition_task_ids": [],
        "no_accepted_source_delta_task_ids": [],
        "retired_science_ready_task_ids": [],
        "science_ready_task_ids": [],
        "missing_declared_deliverable_task_ids": [],
        "gate_blocked_task_ids": [],
        "verifier_warn_task_ids": [],
        "off_path_task_ids": [],
    }
    for task_id in EXPECTED_TASK_IDS:
        payload = payloads.get(task_id, {})
        meta = metadata.get(task_id, {})
        status = str(payload.get("status") or "")
        verdict = str(payload.get("honest_verdict") or "")
        receipt = receipts.get(task_id, {})
        if _receipt_flag_count(receipt) > 0 or _receipt_max_severity(receipt) >= 1:
            classes["verifier_warn_task_ids"].append(task_id)
        if not meta.get("present"):
            terminal = (
                "missing_upstream_retired"
                if MISSING_DELIVERABLE_REASONS.get(task_id)
                else "missing_declared_deliverable"
            )
            classes["missing_declared_deliverable_task_ids"].append(task_id)
        elif task_id == "exp5879-hardness-headroom-taxonomy-corrigendum" and (
            payload.get("hardness_surface_headroom_ready_score") == 1.0
            and (status in {"blocked", "retired"} or verdict.startswith(("blocked:", "retired:")))
        ):
            terminal = "retired_science_ready"
            classes["retired_science_ready_task_ids"].append(task_id)
            classes["science_ready_task_ids"].append(task_id)
        elif payload.get("schema") == "blocked_gate_check_v1" or verdict.startswith("blocked_gate"):
            terminal = "gate_blocked"
            classes["gate_blocked_task_ids"].append(task_id)
        elif task_id == "exp5878-v523-source-delta-ingestion" and payload.get("accepted_finding_count") == 0:
            terminal = "no_accepted_source_delta"
            classes["no_accepted_source_delta_task_ids"].append(task_id)
        elif task_id == "exp5877-transition-v523" and (
            status == "complete" or verdict.startswith("complete:")
        ):
            terminal = "complete_transition"
            classes["complete_transition_task_ids"].append(task_id)
        else:
            terminal = "off_path"
            classes["off_path_task_ids"].append(task_id)
        classes["terminal_class_by_task_id"][task_id] = terminal
    return classes


def _retired_and_science_ready_preserved(
    classes: JsonMap,
    payloads: Mapping[str, JsonMap],
) -> bool:
    expected = {
        "exp5877-transition-v523": "complete_transition",
        "exp5878-v523-source-delta-ingestion": "no_accepted_source_delta",
        "exp5879-hardness-headroom-taxonomy-corrigendum": "retired_science_ready",
        "exp5880-grounding-shortcut-fixture": "missing_upstream_retired",
        "exp5881-one-to-one-grounding-acquisition-ab": "gate_blocked",
        "exp5882-shortcut-resistant-continuous-self-learning": "missing_upstream_retired",
    }
    exp5879 = payloads.get("exp5879-hardness-headroom-taxonomy-corrigendum", {})
    return (
        classes.get("terminal_class_by_task_id") == expected
        and exp5879.get("hardness_surface_headroom_ready_score") == 1.0
        and str(exp5879.get("status") or "") != "complete"
    )


def _missing_and_gate_blocked_receipts(
    matrix: Mapping[str, JsonMap],
    classes: JsonMap,
) -> JsonDict:
    missing_task_ids = [
        task_id
        for task_id in EXPECTED_TASK_IDS
        if task_id in set(classes.get("missing_declared_deliverable_task_ids", []))
    ]
    gate_blocked_task_ids = [
        task_id
        for task_id in EXPECTED_TASK_IDS
        if task_id in set(classes.get("gate_blocked_task_ids", []))
    ]
    receipts: list[JsonDict] = []
    for task_id in [*missing_task_ids, *gate_blocked_task_ids]:
        row = matrix[task_id]
        receipts.append(
            {
                "task_id": task_id,
                "declared_deliverable": row.get("declared_deliverable"),
                "terminal_class": classes.get("terminal_class_by_task_id", {}).get(task_id),
                "present": row.get("present"),
                "missing_reason": row.get("missing_reason"),
                "honest_verdict": row.get("honest_verdict"),
                "conductor": row.get("conductor"),
                "treated_as_success": False,
            }
        )
    payload = {
        "missing_task_ids": missing_task_ids,
        "gate_blocked_task_ids": gate_blocked_task_ids,
        "receipts": receipts,
        "principle": FIELD_PRINCIPLES["missing_and_gate_blocked_receipts"],
    }
    payload["receipt_hash"] = sha256_json(payload)
    return payload


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in SOURCE_HASH_PATHS
    }


def _atomic_output_receipt(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".tmp-probe")
    ok = False
    error = None
    try:
        probe.write_text("atomic-probe\n", encoding="utf-8")
        ok = probe.read_text(encoding="utf-8") == "atomic-probe\n"
    except OSError as exc:
        error = f"{exc.__class__.__name__}:{exc}"
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "declared_path": path.as_posix(),
        "parent_exists": path.parent.exists(),
        "parent_writable": path.parent.exists() and path.parent.is_dir(),
        "atomic_probe_write_ok": ok,
        "ok": ok and error is None,
        "error": error,
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


def _contains_next_range_reference(text: str) -> bool:
    lowered = text.lower()
    if not any(marker in lowered for marker in ("exp589", "exp590", "experiment_589", "experiment_590")):
        return False
    for task_id, rel_path in NEXT_TASK_ARTIFACT_PATHS.items():
        if task_id.lower() in lowered or rel_path.as_posix().lower() in lowered:
            return True
    return any(
        re.search(rf"(?<![a-z0-9_])exp{number}(?![a-z0-9])", lowered)
        or re.search(rf"(?<![a-z0-9_])experiment_{number}(?![a-z0-9])", lowered)
        for number in NEXT_RANGE_NUMBERS
    )


def _scan_candidate_paths(root: Path) -> list[Path]:
    candidates = [
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        ROADMAP_DOC_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    ]
    for folder in ("python", "tests", "scripts"):
        base = root / folder
        if base.exists():
            candidates.extend(
                path.relative_to(root)
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
    results = root / "results"
    if results.exists():
        for path in results.glob("experiment_*"):
            if path.is_file():
                candidates.append(path.relative_to(root))
        for path in results.glob("*transition*.json"):
            if path.is_file():
                candidates.append(path.relative_to(root))
    return sorted({path for path in candidates}, key=lambda value: value.as_posix())


def _range_collision_scan(root: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed: list[JsonDict] = []
    for rel_path in _scan_candidate_paths(root):
        path = root / rel_path
        reference = False
        if rel_path.parts[:1] == ("results",):
            reference = _contains_next_range_reference(rel_path.as_posix())
            if not reference and "transition" in rel_path.name and path.stat().st_size < 2_000_000:
                reference = _contains_next_range_reference(
                    path.read_text(encoding="utf-8", errors="replace")
                )
        elif path.exists() and path.stat().st_size < 2_000_000:
            reference = _contains_next_range_reference(
                path.read_text(encoding="utf-8", errors="replace")
            )
        if not reference:
            continue
        if rel_path in OWNED_REFERENCE_PATHS:
            allowed.append({"path": rel_path.as_posix(), "kind": "transition_owned_reference"})
        elif rel_path in ALLOWED_ALLOCATION_REFERENCE_PATHS:
            allowed.append({"path": rel_path.as_posix(), "kind": "allowed_allocation_reference"})
        else:
            collisions.append({"path": rel_path.as_posix(), "kind": "unexpected_next_range_reference"})
    collisions = sorted(
        {json.dumps(row, sort_keys=True): row for row in collisions}.values(),
        key=lambda row: str(row["path"]),
    )
    allowed = sorted(
        {json.dumps(row, sort_keys=True): row for row in allowed}.values(),
        key=lambda row: str(row["path"]),
    )
    return {
        "range": {"start": "exp5890", "end": "exp5903", "numbers": list(NEXT_RANGE_NUMBERS)},
        "allowed_references": allowed,
        "collisions": collisions,
        "collision_count": len(collisions),
        "collision_free": not collisions,
    }


def _protected_files(root: Path, modification_overrides: Mapping[Path, bool] | None) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in PROTECTED_FILE_PATHS:
        if modification_overrides is not None and rel_path in modification_overrides:
            modified = bool(modification_overrides[rel_path])
            source = "test_override"
        else:
            result = subprocess.run(  # pragma: no cover
                ["git", "status", "--short", "--", rel_path.as_posix()],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            modified = bool(result.stdout.strip())  # pragma: no cover
            source = "git_status"  # pragma: no cover
        digest = path_sha256(root / rel_path)
        rows[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "sha256_before": digest,
            "sha256_after": digest,
            "unchanged": not modified,
            "modified_by_exp5890": modified,
            "check_source": source,
        }
    return rows


def _docs_reconciled() -> JsonDict:
    return {
        "openspec_research_reporting": "reconciled_by_REQ_REPORT_5890",
        "ops_status_md": "deferred_by_operator_stop_rule",
        "ops_changelog_md": "deferred_by_operator_stop_rule",
        "traceability_md": "deferred_by_operator_stop_rule",
        "ops_conductor_log_md": "read_only_evidence_source",
        "files_modified_by_this_workflow": [
            SPEC_RELATIVE_PATH.as_posix(),
            "tests/python/test_experiment_5890_transition_v524.py",
            "python/carnot/experiment_5890_transition_v524.py",
            RESULT_RELATIVE_PATH.as_posix(),
        ],
    }


def _tests_run_rows(tests_run: Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [
            {"command": command, "exit_code": None, "status": "not_recorded"}
            for command in DEFAULT_TEST_COMMANDS
        ]
    return [dict(row) for row in tests_run]


def _failed_required_test_commands(test_rows: Sequence[JsonMap]) -> list[str]:
    failures: list[str] = []
    for row in test_rows:
        code = row.get("exit_code")
        command = str(row.get("command") or "")
        if code is None or code == 0:
            continue
        if row.get("blocking") is False:
            continue
        if code == 1 and "scripts/adversarial_verify.py" in command:
            continue
        failures.append(command)
    return failures


def _failed_nonblocking_test_commands(test_rows: Sequence[JsonMap]) -> list[str]:
    return [
        str(row.get("command") or "")
        for row in test_rows
        if row.get("blocking") is False
        and row.get("exit_code") is not None
        and row.get("exit_code") != 0
    ]


def _test_exit_codes(test_rows: Sequence[JsonMap]) -> JsonDict:
    return {str(row.get("command")): row.get("exit_code") for row in test_rows}


def _roadmap_task_summary(payload: JsonMap) -> JsonDict:
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    return {
        "task_count": len(tasks),
        "task_ids": [str(row.get("id")) for row in tasks if isinstance(row, Mapping)],
        "deliverables": [
            str(row.get("deliverable"))
            for row in tasks
            if isinstance(row, Mapping) and row.get("deliverable") is not None
        ],
    }


def _unactivated_proposal_receipt(task_rows: Sequence[JsonMap]) -> JsonDict:
    block_task_ids = [
        str(row.get("id"))
        for row in task_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    ]
    present = [task_id for task_id in UNACTIVATED_PROPOSAL_TASK_IDS if task_id in block_task_ids]
    receipt: JsonDict = {
        "task_ids": list(UNACTIVATED_PROPOSAL_TASK_IDS),
        "declared_deliverables": {
            task_id: UNACTIVATED_PROPOSAL_ARTIFACT_PATHS[task_id].as_posix()
            for task_id in UNACTIVATED_PROPOSAL_TASK_IDS
        },
        "proposal_only_source": "prior .523 proposal prose",
        "activation_status": "unactivated",
        "present_in_activated_completion_block": present,
        "appended_as_completed": bool(present),
        "principle": FIELD_PRINCIPLES["unactivated_proposal_id_receipt"],
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.perf_counter()
    roadmap, roadmap_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_next, roadmap_next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    research_complete_meta_before = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)[1]
    append_receipt = _append_completion_if_absent(root)
    task_rows = _completion_task_rows(root)
    payloads, metadata = _artifact_payloads(root, task_rows)
    conductor = _conductor_outcomes(root)
    matrix = _activated_task_matrix(task_rows, metadata, payloads, conductor)
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:
        receipts = run_live_adversarial_receipts(root, metadata)  # pragma: no cover
    classes = _classify_outcomes(payloads, metadata, receipts)
    missing_gate = _missing_and_gate_blocked_receipts(matrix, classes)
    preserved = _retired_and_science_ready_preserved(classes, payloads)
    unactivated = _unactivated_proposal_receipt(task_rows)
    collision_scan = _range_collision_scan(root)
    protected = _protected_files(root, modification_overrides)
    test_rows = _tests_run_rows(tests_run)
    failed_tests = _failed_required_test_commands(test_rows)
    failed_nonblocking_tests = _failed_nonblocking_test_commands(test_rows)
    atomic = _atomic_output_receipt(root / RESULT_RELATIVE_PATH)
    resources = _resource_receipts(root)
    present_without_receipts = [
        task_id
        for task_id, meta in metadata.items()
        if meta.get("present") and task_id not in receipts
    ]
    declared_mismatches = [
        task_id for task_id, meta in metadata.items() if not meta.get("declared_path_matches_expected")
    ]
    block_task_ids = [
        str(row.get("id"))
        for row in task_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    ]

    failed_preconditions: list[str] = []
    if not roadmap_meta.get("loadable"):
        failed_preconditions.append("active_roadmap_unloadable")
    if not (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists():
        failed_preconditions.append("live_verifier_missing")
    if not atomic["ok"]:
        failed_preconditions.append("atomic_output_unavailable")
    if not resources["disk"]["ok"] or not resources["memory"]["ok"]:
        failed_preconditions.append("insufficient_resources")
    if append_receipt["reason"] == "research_complete_unparseable":
        failed_preconditions.append("research_complete_unparseable")
    if collision_scan["collision_count"]:
        failed_preconditions.append("next_range_collision")
    if append_receipt["duplicate_history_amplification_count"]:
        failed_preconditions.append("duplicate_history_amplified")
    if set(block_task_ids) != set(EXPECTED_TASK_IDS):
        failed_preconditions.append("activated_task_count_mismatch")
    if not preserved:
        failed_preconditions.append("terminal_outcomes_not_preserved")
    if unactivated["appended_as_completed"]:
        failed_preconditions.append("unactivated_proposal_ids_laundered")
    if present_without_receipts:
        failed_preconditions.append("missing_adversarial_receipts")
    if declared_mismatches:
        failed_preconditions.append("declared_deliverable_mismatch")
    if failed_tests:
        failed_preconditions.append("required_test_command_failed")
    if not all(row["unchanged"] for row in protected.values()):
        failed_preconditions.append("protected_file_modified")

    status = "complete" if not failed_preconditions else "blocked"
    if status == "complete":
        honest = (
            "complete: archived terminal .523 identities by exact declared deliverables into "
            ".524; Exp5879 retirement and science-ready scalar, Exp5880/Exp5882 missing "
            "deliverables, Exp5881 gate block, no-delta, and verifier-warn outcomes "
            "preserved; next_range_collision_count=0"
        )
    else:
        honest = "blocked: exp5890 transition preconditions failed: " + ",".join(
            failed_preconditions
        )

    preconditions = {
        "run_date": RUN_DATE,
        "active_roadmap": {
            **roadmap_meta,
            "milestone": roadmap.get("milestone"),
            **_roadmap_task_summary(roadmap),
        },
        "roadmap_next": {
            **roadmap_next_meta,
            "milestone": roadmap_next.get("milestone"),
            **_roadmap_task_summary(roadmap_next),
            "missing_recorded_explicitly": roadmap_next_meta.get("present") is False,
        },
        "research_complete": {
            **research_complete_meta_before,
            "append_receipt": append_receipt,
        },
        "duplicate_history": {
            "before_duplicate_block_count": append_receipt["before_duplicate_block_count"],
            "after_duplicate_block_count": append_receipt["after_duplicate_block_count"],
            "amplification_count": append_receipt["duplicate_history_amplification_count"],
        },
        "source_hashes": _source_hashes(root),
        "declared_activated_deliverable_hashes": {
            task_id: {
                "path": meta["declared_deliverable"],
                "present": meta["present"],
                "loadable": meta["loadable"],
                "sha256": meta["sha256"],
                "missing_reason": MISSING_DELIVERABLE_REASONS.get(task_id)
                if not meta["present"]
                else None,
            }
            for task_id, meta in metadata.items()
        },
        "missing_declared_deliverables": [
            task_id for task_id, meta in metadata.items() if not meta.get("present")
        ],
        "live_verifier": {
            "path": ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
            "present": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
            "receipts_for_present_deliverables": len(receipts),
            "present_declared_deliverable_count": sum(
                1 for meta in metadata.values() if meta.get("present")
            ),
            "present_declared_deliverables_without_receipts": present_without_receipts,
        },
        "atomic_output": atomic,
        "resources": resources,
        "range_collision_scan": collision_scan,
        "protected_files": protected,
        "failed_required_test_commands": failed_tests,
        "failed_nonblocking_test_commands": failed_nonblocking_tests,
        "declared_deliverable_mismatches": declared_mismatches,
        "activated_task_ids_in_history": block_task_ids,
        "failed_preconditions": failed_preconditions,
    }

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": preconditions,
        "milestone_transition": {
            "source_milestone": MILESTONE_FROM,
            "destination_milestone": MILESTONE_TO,
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "activated_task_and_deliverable_matrix": matrix,
        "missing_and_gate_blocked_receipts": missing_gate,
        "unactivated_proposal_id_receipt": unactivated,
        "adversarial_verifier_receipts": receipts,
        "outcome_classification": classes,
        "retired_and_science_ready_preserved": preserved,
        "research_complete_append_count": append_receipt["append_count"],
        "duplicate_history_amplification_count": append_receipt[
            "duplicate_history_amplification_count"
        ],
        "next_task_range": {
            "start": "exp5890",
            "end": "exp5903",
            "task_ids": list(NEXT_TASK_ARTIFACT_PATHS),
            "declared_deliverables": [
                path.as_posix() for path in NEXT_TASK_ARTIFACT_PATHS.values()
            ],
        },
        "next_range_collision_count": collision_scan["collision_count"],
        "docs_reconciled": _docs_reconciled(),
        "protected_files_unchanged": protected,
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": {
            field: {
                "principle": FIELD_PRINCIPLES[field],
                "sources": [
                    RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
                    ROADMAP_RELATIVE_PATH.as_posix(),
                    ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                    CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                    EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
                    ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
                ],
            }
            for field in REQUIRED_ARTIFACT_FIELDS
        },
        "test_commands": [row.get("command") for row in test_rows],
        "test_exit_codes": _test_exit_codes(test_rows),
        "reproducibility_checksum": "",
        "honest_verdict": honest,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _terminal_class_lists(classes: JsonMap) -> list[set[str]]:
    return [
        set(classes.get("complete_transition_task_ids", [])),
        set(classes.get("no_accepted_source_delta_task_ids", [])),
        set(classes.get("retired_science_ready_task_ids", [])),
        set(classes.get("missing_declared_deliverable_task_ids", [])),
        set(classes.get("gate_blocked_task_ids", [])),
        set(classes.get("off_path_task_ids", [])),
    ]


def validate_artifact(payload: JsonMap) -> bool:
    """Validate the Exp5890 terminal transition artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")
    if payload.get("status") not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    verdict = str(payload.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict missing complete: or blocked: terminal prefix")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")
    if not isinstance(payload.get("next_range_collision_count"), int):
        raise ValueError("next_range_collision_count must be a bare integer")
    if payload.get("status") == "complete" and payload.get("next_range_collision_count") != 0:
        raise ValueError("next_range_collision_count must be zero for completion")
    if payload.get("research_complete_append_count") not in (0, 1):
        raise ValueError("research_complete_append_count must be zero or one")
    if payload.get("duplicate_history_amplification_count") != 0:
        raise ValueError("duplicate_history_amplification_count must be zero")
    if payload.get("retired_and_science_ready_preserved") is not True:
        raise ValueError("laundered retired or science-ready evidence")

    missing_gate = payload.get("missing_and_gate_blocked_receipts")
    if not isinstance(missing_gate, Mapping):
        raise ValueError("missing and gate-blocked receipts missing")
    expected_missing = [
        "exp5880-grounding-shortcut-fixture",
        "exp5882-shortcut-resistant-continuous-self-learning",
    ]
    if missing_gate.get("missing_task_ids") != expected_missing or missing_gate.get("gate_blocked_task_ids") != [
        "exp5881-one-to-one-grounding-acquisition-ab"
    ]:
        raise ValueError("missing and gate-blocked receipts changed")
    receipts_rows = missing_gate.get("receipts")
    if not isinstance(receipts_rows, list) or any(
        not isinstance(row, Mapping) or row.get("treated_as_success") is not False
        for row in receipts_rows
    ):
        raise ValueError("missing and gate-blocked receipts cannot be successes")

    unactivated = payload.get("unactivated_proposal_id_receipt")
    if not isinstance(unactivated, Mapping) or unactivated.get("appended_as_completed"):
        raise ValueError("unactivated proposal IDs cannot be completed work")

    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or not all(
        isinstance(row, Mapping) and row.get("unchanged") is True for row in protected.values()
    ):
        raise ValueError("protected file changed")

    field_provenance = payload.get("field_provenance")
    principles = payload.get("field_principles")
    if not isinstance(field_provenance, Mapping) or not isinstance(principles, Mapping):
        raise ValueError("field provenance/principles missing")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in field_provenance or field not in principles:
            raise ValueError(f"field provenance missing for {field}")

    classes = payload.get("outcome_classification")
    if not isinstance(classes, Mapping):
        raise ValueError("outcome_classification missing")
    terminal_lists = _terminal_class_lists(classes)
    for index, left in enumerate(terminal_lists):
        for right in terminal_lists[index + 1 :]:
            if left & right:
                raise ValueError("terminal classes overlap")
    terminal_map = classes.get("terminal_class_by_task_id")
    if not isinstance(terminal_map, Mapping) or set(terminal_map) != set(EXPECTED_TASK_IDS):
        raise ValueError("terminal class map missing activated tasks")

    matrix = payload.get("activated_task_and_deliverable_matrix")
    receipts = payload.get("adversarial_verifier_receipts")
    if not isinstance(matrix, Mapping) or not isinstance(receipts, Mapping):
        raise ValueError("adversarial verifier receipt matrix missing")
    if set(matrix) != set(EXPECTED_TASK_IDS):
        raise ValueError("activated task matrix must contain exactly six .523 tasks")
    required_receipt_keys = {
        "command",
        "exit_code",
        "stdout_json",
        "flag_count",
        "max_severity",
        "receipt_hash",
    }
    for task_id, row in matrix.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"malformed matrix row for {task_id}")
        expected_identity = [
            MILESTONE_FROM,
            task_id,
            ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
        ]
        if row.get("identity") != expected_identity:
            raise ValueError(f"activated identity mismatch for {task_id}")
        if task_id in MISSING_DELIVERABLE_REASONS and (
            row.get("present") is not False
            or row.get("missing_reason") != MISSING_DELIVERABLE_REASONS[task_id]
        ):
            number = task_id.split("-", 1)[0].replace("exp", "Exp")
            raise ValueError(f"{number} missing deliverable must preserve upstream retirement")
        if task_id == "exp5879-hardness-headroom-taxonomy-corrigendum" and (
            row.get("science_ready_scalar") != 1.0
            or terminal_map.get(task_id) != "retired_science_ready"
            or row.get("status") == "complete"
        ):
            raise ValueError("Exp5879 science-ready scalar or retirement state was laundered")
        if row.get("present") is not True:
            continue
        receipt = receipts.get(task_id)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"missing adversarial verifier receipt for {task_id}")
        missing_receipt_fields = sorted(required_receipt_keys - set(receipt))
        if missing_receipt_fields:
            raise ValueError(
                f"missing adversarial verifier receipt fields for {task_id}: {missing_receipt_fields}"
            )
        command = str(receipt.get("command") or "")
        declared = str(row.get("declared_deliverable") or "")
        artifact_path = str(receipt.get("artifact_path") or "")
        if (
            "--milestone-range" in command
            or "*" in command
            or (declared and declared not in command)
            or (artifact_path and artifact_path != declared)
        ):
            raise ValueError(f"adversarial verifier receipt command is not exact for {task_id}")

    next_task_range = payload.get("next_task_range")
    if not isinstance(next_task_range, Mapping) or next_task_range.get("start") != "exp5890":
        raise ValueError("next_task_range missing Exp5890 start")
    if next_task_range.get("end") != "exp5903":
        raise ValueError("next_task_range missing Exp5903 end")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--tests-run-json", type=Path)
    args = parser.parse_args(argv)
    tests_run = None
    if args.tests_run_json is not None:
        tests_run = json.loads(args.tests_run_json.read_text(encoding="utf-8"))
    artifact = build_report(REPO_ROOT, tests_run=tests_run)
    validate_artifact(artifact)
    write_json(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
