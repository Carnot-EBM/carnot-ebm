"""Exp5905 transition receipt from terminal milestone .524 into .525.

Spec refs: REQ-REPORT-5905, SCENARIO-REPORT-5905-EXACT-ARCHIVE,
SCENARIO-REPORT-5905-EXP5895-MIXED-RECEIPT,
SCENARIO-REPORT-5905-RESERVATION-AND-RANGE,
SCENARIO-REPORT-5905-APPEND-ONCE, SCENARIO-REPORT-5905-SCHEMA.

This module is an evidence ledger, not a scientific rerun. It selects prior
work only through `(milestone, task_id, declared_deliverable)` rows, preserves
terminal outcomes as they landed, reserves the concurrently occupied Exp5904
identity, and proves the Exp5905-Exp5917 allocation is collision-free.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5905_transition_v525.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
E2E_TEST_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
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

EXPERIMENT = "experiment_5905_transition_v525"
EXPERIMENT_ID = "exp5905-transition-v525"
MILESTONE_FROM = "2026.07.524"
MILESTONE_TO = "2026.07.525"
MILESTONE_FROM_TITLE = "Grounded Constraint IR, Shortcut-Safe Self-Learning, and Structured Live Memory"
MILESTONE_TO_TITLE = (
    "Verified Constraint Synthesis, Transactional Self-Learning, and Live Structured Memory"
)
RUN_DATE = "20260725"
RANDOM_SEED = 5905
SCHEMA = "carnot.experiment_5905.transition_v525.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5905",
    "SCENARIO-REPORT-5905-EXACT-ARCHIVE",
    "SCENARIO-REPORT-5905-EXP5895-MIXED-RECEIPT",
    "SCENARIO-REPORT-5905-RESERVATION-AND-RANGE",
    "SCENARIO-REPORT-5905-APPEND-ONCE",
    "SCENARIO-REPORT-5905-SCHEMA",
)

ACTIVATED_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5890-transition-v524": Path("results/experiment_5890_transition_v524.json"),
    "exp5891-v524-source-delta-ingestion": Path(
        "results/experiment_5891_v524_source_delta_ingestion.json"
    ),
    "exp5892-headroom-evidence-escrow": Path(
        "results/experiment_5892_headroom_evidence_escrow.json"
    ),
    "exp5893-grounding-shortcut-fixture": Path(
        "results/experiment_5893_grounding_shortcut_fixture.json"
    ),
    "exp5894-one-to-one-grounding-ab": Path("results/experiment_5894_one_to_one_grounding_ab.json"),
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

ACTIVATED_TASK_TITLES: dict[str, str] = {
    "exp5890-transition-v524": "Exact terminal-boundary handoff from .523 into .524",
    "exp5891-v524-source-delta-ingestion": "Dated evidence refresh after the V524 planner marker",
    "exp5892-headroom-evidence-escrow": "Immutable hardness-headroom evidence escrow and clean admission",
    "exp5893-grounding-shortcut-fixture": "Gated on Exp5892 admission: exact grounding-shortcut fixture",
    "exp5894-one-to-one-grounding-ab": "Gated on Exp5893 fixture: one-to-one atom-grounding acquisition A/B",
    "exp5895-shortcut-safe-continuous-self-learning": (
        "Gated on Exp5894 mechanism: prospective shortcut-safe continuous self-learning"
    ),
    "exp5896-typed-constraint-ir-fixture": "Engine-neutral typed ConstraintIR fixture with exact certificates",
    "exp5897-sota-constraint-ir-repair-ab": (
        "Gated on Exp5896 fixture: three-family translate-run-inspect-repair A/B"
    ),
    "exp5898-recursive-constraint-improvement": (
        "Gated on Exp5897 trace lift: constraint-wise recursive improvement"
    ),
    "exp5899-constraint-repair-portability-audit": (
        "Gated on Exp5898 recursion: portability, leakage, and camouflage audit"
    ),
    "exp5900-arc-structured-evidence-memory-contract": (
        "Agent-owned ARC event tape and structured evidence-index contract"
    ),
    "exp5901-arc-structured-memory-causal-audit": (
        "Gated on Exp5900 contract: ARC retrieval fidelity and causal necessity"
    ),
    "exp5902-arc-structured-memory-live-ab": (
        "Gated on Exp5901 causality: adapter-disabled live E3 structured-memory A/B"
    ),
    "exp5903-v524-capstone-reconciliation": "Branch-independent terminal reconciliation for milestone .524",
}

EXPECTED_TERMINAL_CLASSES: dict[str, str] = {
    "exp5890-transition-v524": "ready/positive",
    "exp5891-v524-source-delta-ingestion": "null",
    "exp5892-headroom-evidence-escrow": "ready/positive",
    "exp5893-grounding-shortcut-fixture": "ready/positive",
    "exp5894-one-to-one-grounding-ab": "ready/positive",
    "exp5895-shortcut-safe-continuous-self-learning": "null",
    "exp5896-typed-constraint-ir-fixture": "ready/positive",
    "exp5897-sota-constraint-ir-repair-ab": "blocked-precondition",
    "exp5898-recursive-constraint-improvement": "gate-blocked",
    "exp5899-constraint-repair-portability-audit": "gate-blocked",
    "exp5900-arc-structured-evidence-memory-contract": "ready/positive",
    "exp5901-arc-structured-memory-causal-audit": "ready/positive",
    "exp5902-arc-structured-memory-live-ab": "blocked-precondition",
    "exp5903-v524-capstone-reconciliation": "ready/positive",
}

NEXT_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5905-transition-v525": RESULT_RELATIVE_PATH,
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
    "exp5917-v525-capstone-reconciliation": Path(
        "results/experiment_5917_v525_capstone_reconciliation.json"
    ),
}
NEXT_RANGE_NUMBERS = range(5905, 5918)

CONDUCTOR_TITLE_PATTERNS: dict[str, str] = {
    "exp5890-transition-v524": "Exact terminal-boundary handoff from .523 into .52",
    "exp5891-v524-source-delta-ingestion": "Dated evidence refresh after the V524 planner mark",
    "exp5892-headroom-evidence-escrow": "Immutable hardness-headroom evidence escrow and cl",
    "exp5893-grounding-shortcut-fixture": "Gated on Exp5892 admission: exact grounding-shortc",
    "exp5894-one-to-one-grounding-ab": "Gated on Exp5893 fixture: one-to-one atom-groundin",
    "exp5895-shortcut-safe-continuous-self-learning": (
        "Gated on Exp5894 mechanism: prospective shortcut-s"
    ),
    "exp5896-typed-constraint-ir-fixture": "Engine-neutral typed ConstraintIR fixture with exa",
    "exp5897-sota-constraint-ir-repair-ab": "Gated on Exp5896 fixture: three-family translate-r",
    "exp5898-recursive-constraint-improvement": (
        "Gated on Exp5897 trace lift: constraint-wise recur"
    ),
    "exp5899-constraint-repair-portability-audit": (
        "Gated on Exp5898 recursion: portability, leakage,"
    ),
    "exp5900-arc-structured-evidence-memory-contract": (
        "Agent-owned ARC event tape and structured evidence"
    ),
    "exp5901-arc-structured-memory-causal-audit": "Gated on Exp5900 contract: ARC retrieval fidelity",
    "exp5902-arc-structured-memory-live-ab": "Gated on Exp5901 causality: adapter-disabled live",
    "exp5903-v524-capstone-reconciliation": "Branch-independent terminal reconciliation for mil",
}

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
    E2E_TEST_PLAN_RELATIVE_PATH,
    EVIDENCE_INDEX_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *PROTECTED_FILE_PATHS,
    *ACTIVATED_TASK_ARTIFACT_PATHS.values(),
)

OWNED_REFERENCE_PATHS = (
    Path("python/carnot/experiment_5905_transition_v525.py"),
    Path("tests/python/test_experiment_5905_transition_v525.py"),
    SPEC_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
)

ALLOWED_ALLOCATION_REFERENCE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    Path("ops/metrics.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "activated_task_and_deliverable_matrix",
    "exact_terminal_classification",
    "exp5895_science_and_operational_receipt",
    "blocked_and_gate_blocked_receipts",
    "adversarial_verifier_receipts",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
    "exp5904_reservation_receipt",
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
    "status": "Terminal transition state over exact activated .524 identities.",
    "preconditions_checked": (
        "Parsed roadmaps, hashes, resources, atomic output, verifier availability, "
        "and declared deliverables ground the handoff."
    ),
    "milestone_transition": (
        "Explicit .524-to-.525 boundary prevents prefix aliasing and outcome laundering."
    ),
    "activated_task_and_deliverable_matrix": "Only activated task IDs and declared paths count as evidence.",
    "exact_terminal_classification": (
        "Positive, null, blocked-precondition, and gate-blocked classes remain disjoint."
    ),
    "exp5895_science_and_operational_receipt": (
        "A positive submetric cannot erase a nonzero required test, and a test failure cannot erase measured science."
    ),
    "blocked_and_gate_blocked_receipts": (
        "Blocked-precondition and gate-blocked outcomes remain receipts rather than successes."
    ),
    "adversarial_verifier_receipts": (
        "Fresh verifier receipts cover every present declared .524 artifact without replacing missing ones."
    ),
    "research_complete_append_count": "Exact zero-or-one append behavior prevents duplicate completion history.",
    "duplicate_history_amplification_count": (
        "Existing duplicate history is measured but never multiplied."
    ),
    "exp5904_reservation_receipt": (
        "Concurrent work is recorded but never edited, required, or classified by this milestone."
    ),
    "next_task_range": "A declared finite Exp5905-Exp5917 interval makes allocation auditable.",
    "next_range_collision_count": "Only bare zero authorizes Exp5905-Exp5917.",
    "docs_reconciled": (
        "Transition-owned spec reconciliation is recorded while operator-delegated ledgers remain untouched."
    ),
    "protected_files_unchanged": (
        "Protected roadmap, conductor, north-star, Exp5904, and operator-ledger files remain byte-identical during this task."
    ),
    "duration_s": "Measured wall time exposes aggregation-only execution.",
    "inference_substrate": "Use `aggregation_from_upstream_artifacts`.",
    "field_provenance": (
        "Every required field traces to exact paths, hashes, receipts, commands, or classifications."
    ),
    "test_commands": (
        "Commands document focused unit, coverage, YAML parse, exact-path/hash, duplicate-history, "
        "verifier, exclusion-manifest, range-collision, protected-file, reconciliation, spec, "
        "root-clutter, and full-suite checks."
    ),
    "test_exit_codes": "Exit codes prevent failed checks becoming success.",
    "reproducibility_checksum": "A checksum detects later ledger, artifact, or allocation drift.",
    "honest_verdict": "Use a `complete:` or `blocked:` prefix.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); p=pathlib.Path('research-roadmap-next.yaml'); yaml.safe_load(p.read_text()) if p.exists() else None; yaml.safe_load(pathlib.Path('research-complete.yaml').read_text()); yaml.safe_load(pathlib.Path('ops/exclusion_manifest.yaml').read_text())\"",
    ".venv/bin/pytest tests/python/test_experiment_5905_transition_v525.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5905_transition_v525.py -m pytest tests/python/test_experiment_5905_transition_v525.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5905_transition_v525.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json results/experiment_5890_transition_v524.json results/experiment_5891_v524_source_delta_ingestion.json results/experiment_5892_headroom_evidence_escrow.json results/experiment_5893_grounding_shortcut_fixture.json results/experiment_5894_one_to_one_grounding_ab.json results/experiment_5895_shortcut_safe_continuous_self_learning.json results/experiment_5896_typed_constraint_ir_fixture.json results/experiment_5897_sota_constraint_ir_repair_ab.json results/experiment_5898_recursive_constraint_improvement.json results/experiment_5900_arc_structured_evidence_memory_contract.json results/experiment_5901_arc_structured_memory_causal_audit.json results/experiment_5902_arc_structured_memory_live_ab.json results/experiment_5903_v524_capstone_reconciliation.json",
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
                "    result: terminal outcome preserved by transition receipt",
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


def _completion_task_rows(root: Path) -> list[JsonMap]:
    blocks = [block for block in _history_blocks(root) if block.get("id") == MILESTONE_FROM]
    if not blocks:
        return []
    tasks = blocks[-1].get("tasks")
    return [row for row in tasks if isinstance(row, Mapping)] if isinstance(tasks, list) else []


def _artifact_payloads(root: Path, task_rows: Sequence[JsonMap]) -> tuple[dict[str, JsonDict], dict[str, JsonDict], dict[str, JsonDict]]:
    by_task = {str(row.get("id")): row for row in task_rows if isinstance(row.get("id"), str)}
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    row_info: dict[str, JsonDict] = {}
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
        row_info[task_id] = {
            "title": str(row.get("title") or ACTIVATED_TASK_TITLES[task_id]),
            "completion_row_result": str(row.get("result") or ""),
        }
    return payloads, metadata, row_info


def _status_from_log(line: str | None) -> str:
    if line is None:
        return "MISSING"
    for status in ("GATE_BLOCK", "FLAGGED", "FAIL", "OK", "SKIP"):
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


def _terminal_class(
    task_id: str,
    payload: JsonMap,
    meta: JsonMap,
    conductor: JsonMap,
    receipt: JsonMap,
) -> str:
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    schema = str(payload.get("schema") or "")
    if _receipt_flag_count(receipt) > 0 and _receipt_max_severity(receipt) >= 2:
        return "unsafe/disqualified"
    if not meta.get("present"):
        return "gate-blocked" if conductor.get("latest_status") == "GATE_BLOCK" else "missing"
    if task_id == "exp5903-v524-capstone-reconciliation" and status.startswith("complete"):
        return "ready/positive"
    if status == "blocked_precondition" or verdict.startswith("blocked_precondition"):
        return "blocked-precondition"
    if schema == "blocked_gate_check_v1" or verdict.startswith("blocked_gate") or payload.get("gates_evaluated"):
        return "gate-blocked"
    if status == "retired" or verdict.startswith("retired:"):
        return "retired"
    if (
        status == "complete_null"
        or verdict.startswith("complete_null")
        or (
            task_id == "exp5891-v524-source-delta-ingestion"
            and payload.get("accepted_finding_count") == 0
        )
    ):
        return "null"
    if status.startswith("blocked") or verdict.startswith("blocked:"):
        return "blocked"
    if status in {"complete", "complete_ready", "complete_positive", "ready"} or verdict.startswith(
        ("complete:", "complete_ready", "complete_positive", "ready:")
    ):
        return "ready/positive"
    return "blocked"


def _activated_matrix(
    metadata: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    row_info: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
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
            "status": str(payload.get("status") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "completion_row_result": row_info[task_id]["completion_row_result"],
            "conductor": conductor.get(task_id, {}),
            "terminal_evidence_source": "declared_deliverable",
        }
    return matrix


def _exact_terminal_classification(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    receipts: Mapping[str, JsonMap],
) -> JsonDict:
    allowed = (
        "ready/positive",
        "null",
        "unsafe/disqualified",
        "blocked-precondition",
        "blocked",
        "retired",
        "gate-blocked",
        "missing",
    )
    by_task: dict[str, str] = {}
    by_class: dict[str, list[str]] = {name: [] for name in allowed}
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        terminal = _terminal_class(
            task_id,
            payloads.get(task_id, {}),
            metadata.get(task_id, {}),
            conductor.get(task_id, {}),
            receipts.get(task_id, {}),
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
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _blocked_and_gate_receipts(matrix: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    by_task = classes["terminal_class_by_task_id"]
    blocked_pre = [task_id for task_id, terminal in by_task.items() if terminal == "blocked-precondition"]
    gate_blocked = [task_id for task_id, terminal in by_task.items() if terminal == "gate-blocked"]
    missing_but_gate = [task_id for task_id in gate_blocked if not matrix[task_id]["present"]]
    receipts: list[JsonDict] = []
    for task_id in [*blocked_pre, *gate_blocked]:
        row = matrix[task_id]
        receipts.append(
            {
                "task_id": task_id,
                "declared_deliverable": row["declared_deliverable"],
                "terminal_class": by_task[task_id],
                "present": row["present"],
                "honest_verdict": row["honest_verdict"],
                "conductor": row["conductor"],
                "treated_as_success": False,
            }
        )
    return {
        "blocked_precondition_task_ids": blocked_pre,
        "gate_blocked_task_ids": gate_blocked,
        "declared_deliverable_missing_but_gate_blocked_task_ids": missing_but_gate,
        "receipts": receipts,
        "principle": FIELD_PRINCIPLES["blocked_and_gate_blocked_receipts"],
    }


def _test_exit_code(payload: JsonMap, command: str) -> int | None:
    tests = payload.get("test_exit_codes")
    if isinstance(tests, Mapping):
        value = tests.get(command)
        return int(value) if isinstance(value, int) else None
    if isinstance(tests, list):
        for row in tests:
            if isinstance(row, Mapping) and row.get("command") == command:
                value = row.get("exit_code")
                return int(value) if isinstance(value, int) else None
    return None


def _exp5895_receipt(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    task_id = "exp5895-shortcut-safe-continuous-self-learning"
    payload = payloads.get(task_id, {})
    metrics = payload.get("prospective_semantic_and_constraint_metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    lift = metrics.get("primary_minus_best_shortcut_control")
    lift = lift if isinstance(lift, Mapping) else {}
    transfer = payload.get("forward_transfer_recurrence_retention_and_regret")
    transfer = transfer if isinstance(transfer, Mapping) else {}
    retention = transfer.get("retention")
    retention = retention if isinstance(retention, Mapping) else {}
    shortcut = payload.get("shortcut_false_accept_metrics")
    shortcut = shortcut if isinstance(shortcut, Mapping) else {}
    rollback = payload.get("rollback_restart_and_state_hashes")
    rollback = rollback if isinstance(rollback, Mapping) else {}
    weights = payload.get("no_model_weight_mutation")
    weights = weights if isinstance(weights, Mapping) else {}
    full_suite_command = ".venv/bin/pytest tests/python -q"
    full_suite_exit = _test_exit_code(payload, full_suite_command)
    ready_score = payload.get("shortcut_resistant_csl_ready_score")
    terminal_class = classes["terminal_class_by_task_id"].get(task_id)
    positive = {
        "prospective_semantic_lift_mean_delta": lift.get("mean_delta"),
        "prospective_semantic_lift_ci95": lift.get("ci95"),
        "protected_prefix_retention": retention.get("protected_prefix_retention"),
        "unsafe_accept_count": shortcut.get("unsafe_accept_count"),
        "restart_equivalence": rollback.get("restart_equivalence"),
        "rollback_hash_mismatch_count": rollback.get("rollback_hash_mismatch_count"),
        "no_model_weight_mutation_all_unchanged": weights.get("all_unchanged") is True,
    }
    operational = {
        "shortcut_resistant_csl_ready_score": ready_score,
        "required_full_suite_command": full_suite_command,
        "required_full_suite_exit_code": full_suite_exit,
        "promoted_as_ready": terminal_class == "ready/positive",
    }
    science_preserved = (
        isinstance(positive["prospective_semantic_lift_mean_delta"], (int, float))
        and positive["prospective_semantic_lift_mean_delta"] > 0
        and positive["protected_prefix_retention"] == 1.0
        and positive["unsafe_accept_count"] == 0
        and positive["restart_equivalence"] == 1.0
        and positive["rollback_hash_mismatch_count"] == 0
        and positive["no_model_weight_mutation_all_unchanged"] is True
    )
    operational_null = ready_score == 0.0 and isinstance(full_suite_exit, int) and full_suite_exit != 0
    return {
        "task_id": task_id,
        "declared_deliverable": ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
        "terminal_class": terminal_class,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "positive_scientific_submetrics": positive,
        "operational_null_receipt": operational,
        "science_preserved": science_preserved,
        "operational_null_preserved": operational_null,
        "laundering_detected": not (science_preserved and operational_null and not operational["promoted_as_ready"]),
        "principle": FIELD_PRINCIPLES["exp5895_science_and_operational_receipt"],
    }


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    paths = sorted({path for path in SOURCE_HASH_PATHS}, key=lambda value: value.as_posix())
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in paths
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


def _exp5904_reservation_receipt(root: Path) -> JsonDict:
    work_paths = (EXP5904_SOURCE_RELATIVE_PATH, EXP5904_RESULT_RELATIVE_PATH)
    existing_paths = [path.as_posix() for path in work_paths if (root / path).exists()]
    return {
        "exp5904_reserved": bool(existing_paths),
        "reservation_reason": "concurrent_click_target_work_occupies_identity",
        "existing_paths": existing_paths,
        "path_hashes": {
            path.as_posix(): path_sha256(root / path)
            for path in work_paths
            if (root / path).exists()
        },
        "edited_by_exp5905": False,
        "required_by_exp5905": False,
        "classified_by_exp5905": False,
        "included_in_activated_matrix": False,
        "included_in_next_range": False,
        "principle": FIELD_PRINCIPLES["exp5904_reservation_receipt"],
    }


def _next_range_numbers_in_text(text: str) -> set[int]:
    lowered = text.lower()
    if not any(marker in lowered for marker in ("exp590", "exp591", "experiment_590", "experiment_591")):
        return set()
    numbers: set[int] = set()
    for task_id, rel_path in NEXT_TASK_ARTIFACT_PATHS.items():
        if task_id.lower() in lowered or rel_path.as_posix().lower() in lowered:
            numbers.add(int(task_id[3:7]))
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
        Path("ops/metrics.md"),
        SPEC_RELATIVE_PATH,
    ]
    for folder in ("python", "tests", "scripts"):
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
    return sorted({path for path in candidates}, key=lambda value: value.as_posix())


def _allowed_range_reference_kind(rel_path: Path, numbers: set[int]) -> str | None:
    if rel_path in OWNED_REFERENCE_PATHS:
        return "transition_owned_reference"
    if rel_path in ALLOWED_ALLOCATION_REFERENCE_PATHS:
        return "allowed_allocation_reference"
    if rel_path == CONDUCTOR_LOG_RELATIVE_PATH and numbers and numbers <= {5905}:
        return "transition_owned_conductor_attempt_reference"
    return None


def _range_collision_scan(root: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed: list[JsonDict] = []
    for rel_path in _scan_candidate_paths(root):
        path = root / rel_path
        text = rel_path.as_posix()
        if path.exists() and rel_path.parts[:1] != ("results",) and path.stat().st_size < 2_000_000:
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
        "range": {"start": 5905, "end": 5917},
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
        "openspec_research_reporting_updated": "REQ-REPORT-5905" in spec_text,
        "ops_status_deferred_to_conductor": True,
        "ops_changelog_deferred_to_conductor": True,
        "traceability_deferred_to_conductor": True,
        "ops_conductor_log_deferred_to_conductor": True,
    }


def _field_provenance() -> dict[str, JsonDict]:
    base_sources = [
        ROADMAP_RELATIVE_PATH.as_posix(),
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": base_sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_rows(tests_run: Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [{"command": command, "exit_code": None, "status": "not_recorded"} for command in DEFAULT_TEST_COMMANDS]
    return [dict(row) for row in tests_run]


def _failed_required_test_commands(rows: Sequence[JsonMap]) -> list[str]:
    return [
        str(row.get("command"))
        for row in rows
        if row.get("blocking", True) is not False
        and isinstance(row.get("exit_code"), int)
        and int(row["exit_code"]) != 0
    ]


def _status_and_verdict(failed_preconditions: Sequence[str], classes: JsonMap) -> tuple[str, str]:
    if failed_preconditions:
        return "blocked", "blocked: Exp5905 transition preconditions failed"
    by_task = classes.get("terminal_class_by_task_id", {})
    if any(value == "unsafe/disqualified" for value in by_task.values()):
        return "blocked", "blocked: unsafe/disqualified .524 identity present"
    if any(value in {"null", "blocked-precondition", "gate-blocked", "missing"} for value in by_task.values()):
        return (
            "complete_with_nulls",
            "complete: archived terminal .524 identities into .525 with Exp5895 null, blocked-precondition, and gate-blocked receipts preserved; next_range_collision_count=0",
        )
    return "complete", "complete: archived terminal .524 identities into .525 with collision-free allocation"


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.monotonic()
    protected_before = _protected_file_hashes(root)
    active_roadmap, active_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_next, roadmap_next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    complete_meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)[1]
    task_rows = _completion_task_rows(root)
    payloads, metadata, row_info = _artifact_payloads(root, task_rows)
    conductor = _conductor_outcomes(root)
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = run_live_adversarial_receipts(root, metadata)
    matrix = _activated_matrix(metadata, payloads, row_info, conductor)
    classes = _exact_terminal_classification(payloads, metadata, conductor, receipts)
    append_receipt = _append_completion_if_absent(root, bool(classes["all_activated_terminal"]))
    range_scan = _range_collision_scan(root)
    protected = _protected_files_unchanged(root, protected_before)
    exp5895 = _exp5895_receipt(payloads, classes)
    exp5904 = _exp5904_reservation_receipt(root)
    test_rows = _tests_run_rows(tests_run)
    failed_tests = _failed_required_test_commands(test_rows)
    present_task_ids = [task_id for task_id, row in matrix.items() if row["present"]]
    missing_receipts = [task_id for task_id in present_task_ids if task_id not in receipts]
    failed_preconditions: list[str] = []
    if active_meta["present"] and not active_meta["loadable"]:
        failed_preconditions.append("active_roadmap_unloadable")
    if complete_meta["present"] and not complete_meta["loadable"]:
        failed_preconditions.append("research_complete_unparseable")
    if not (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists():
        failed_preconditions.append("live_verifier_missing")
    if range_scan["collision_count"] != 0:
        failed_preconditions.append("next_range_collision")
    if append_receipt["duplicate_history_amplification_count"] != 0:
        failed_preconditions.append("duplicate_history_amplified")
    if not exp5904["exp5904_reserved"]:
        failed_preconditions.append("exp5904_not_reserved")
    if not exp5895["science_preserved"] or not exp5895["operational_null_preserved"]:
        failed_preconditions.append("exp5895_science_or_null_not_preserved")
    if classes["terminal_class_by_task_id"] != EXPECTED_TERMINAL_CLASSES:
        failed_preconditions.append("terminal_outcomes_not_preserved")
    if missing_receipts:
        failed_preconditions.append("missing_adversarial_receipts")
    if failed_tests:
        failed_preconditions.append("required_tests_failed")
    if not protected["all_unchanged"]:
        failed_preconditions.append("protected_file_modified")
    resources = _resource_receipts(root)
    if not resources["disk"]["ok"] or not resources["memory"]["ok"]:
        failed_preconditions.append("insufficient_resources")
    atomic = _atomic_output_receipt(root / RESULT_RELATIVE_PATH)
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
                "task_count": len(active_roadmap.get("tasks", [])) if isinstance(active_roadmap.get("tasks"), list) else 0,
            },
            "roadmap_next": {
                **roadmap_next_meta,
                "milestone": roadmap_next.get("milestone") if roadmap_next_meta["loadable"] else None,
            },
            "research_complete": complete_meta,
            "source_hashes": _source_hashes(root),
            "declared_present_deliverable_hashes": {
                task_id: row["sha256"] for task_id, row in matrix.items() if row["present"]
            },
            "resource_receipts": resources,
            "atomic_output": atomic,
            "adversarial_verifier_available": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
            "range_collision_scan": range_scan,
            "duplicate_history": {
                "before_duplicate_block_count": append_receipt["before_duplicate_block_count"],
                "after_duplicate_block_count": append_receipt["after_duplicate_block_count"],
            },
            "missing_adversarial_receipt_task_ids": missing_receipts,
            "failed_required_test_commands": failed_tests,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_transition": {
            "source_milestone": MILESTONE_FROM,
            "destination_milestone": MILESTONE_TO,
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "activated_task_and_deliverable_matrix": matrix,
        "exact_terminal_classification": classes,
        "exp5895_science_and_operational_receipt": exp5895,
        "blocked_and_gate_blocked_receipts": _blocked_and_gate_receipts(matrix, classes),
        "adversarial_verifier_receipts": {
            task_id: receipts[task_id] for task_id in ACTIVATED_TASK_ARTIFACT_PATHS if task_id in receipts
        },
        "research_complete_append_count": append_receipt["append_count"],
        "research_complete_append_receipt": append_receipt,
        "duplicate_history_amplification_count": append_receipt["duplicate_history_amplification_count"],
        "exp5904_reservation_receipt": exp5904,
        "next_task_range": {
            "start": "exp5905",
            "end": "exp5917",
            "count": len(NEXT_TASK_ARTIFACT_PATHS),
            "task_ids": list(NEXT_TASK_ARTIFACT_PATHS),
            "reserved_predecessor": "exp5904",
        },
        "next_range_collision_count": range_scan["collision_count"],
        "docs_reconciled": _docs_reconciled(root),
        "protected_files_unchanged": protected,
        "duration_s": result_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": [row.get("command") for row in test_rows],
        "test_exit_codes": test_rows,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_artifact(report: JsonMap) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            raise ValueError(f"missing required field: {field}")
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field provenance missing")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in provenance or provenance[field].get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field provenance missing: {field}")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(report.get("honest_verdict") or "").startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    collision_count = report.get("next_range_collision_count")
    if not isinstance(collision_count, int):
        raise ValueError("next_range_collision_count must be a bare integer")
    if report.get("status") != "blocked" and collision_count != 0:
        raise ValueError("next_range_collision_count must be zero for completion")
    if report.get("research_complete_append_count") not in {0, 1}:
        raise ValueError("research_complete_append_count must be 0 or 1")
    if report.get("duplicate_history_amplification_count") != 0:
        raise ValueError("duplicate_history_amplification_count must be 0")
    matrix = report.get("activated_task_and_deliverable_matrix")
    if not isinstance(matrix, Mapping) or set(matrix) != set(ACTIVATED_TASK_ARTIFACT_PATHS):
        raise ValueError("activated matrix must contain exactly fourteen .524 task ids")
    for task_id, expected_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = matrix[task_id]
        if not isinstance(row, Mapping):
            raise ValueError("malformed matrix row")
        expected_identity = [MILESTONE_FROM, task_id, expected_path.as_posix()]
        if row.get("identity") != expected_identity:
            raise ValueError("activated identity mismatch")
    classes = report.get("exact_terminal_classification")
    by_task = classes.get("terminal_class_by_task_id") if isinstance(classes, Mapping) else None
    if by_task != EXPECTED_TERMINAL_CLASSES:
        raise ValueError("terminal classes do not preserve .524 outcomes")
    exp5904 = report.get("exp5904_reservation_receipt")
    if (
        not isinstance(exp5904, Mapping)
        or exp5904.get("exp5904_reserved") is not True
        or exp5904.get("edited_by_exp5905") is not False
        or exp5904.get("classified_by_exp5905") is not False
    ):
        raise ValueError("Exp5904 reservation is not preserved")
    exp5895 = report.get("exp5895_science_and_operational_receipt")
    if not isinstance(exp5895, Mapping) or exp5895.get("laundering_detected") is True:
        raise ValueError("Exp5895 laundering detected")
    operational = exp5895.get("operational_null_receipt")
    if not isinstance(operational, Mapping) or operational.get("required_full_suite_exit_code") != 2:
        raise ValueError("Exp5895 full-suite exit must remain 2")
    if operational.get("promoted_as_ready") is not False:
        raise ValueError("Exp5895 laundering promoted null as ready")
    blocked = report.get("blocked_and_gate_blocked_receipts")
    if not isinstance(blocked, Mapping):
        raise ValueError("blocked/gate receipt missing")
    for row in blocked.get("receipts", []):
        if isinstance(row, Mapping) and row.get("treated_as_success") is not False:
            raise ValueError("blocked/gate receipt promoted as success")
    receipts = report.get("adversarial_verifier_receipts")
    if not isinstance(receipts, Mapping):
        raise ValueError("adversarial verifier receipts must be keyed by task id")
    for task_id, row in matrix.items():
        if row.get("present") is not True:
            continue
        receipt = receipts.get(task_id)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"missing adversarial verifier receipt: {task_id}")
        if not {"task_id", "artifact_path", "command", "exit_code", "receipt_hash"} <= set(receipt):
            raise ValueError("missing adversarial verifier receipt fields")
        command = str(receipt.get("command"))
        if "adversarial_verify.py" not in command or "--json" not in command:
            raise ValueError("adversarial verifier receipt command must run --json")
    protected = report.get("protected_files_unchanged")
    files = protected.get("files") if isinstance(protected, Mapping) else None
    if not isinstance(files, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected file changed")
    if any(isinstance(row, Mapping) and row.get("unchanged") is not True for row in files.values()):
        raise ValueError("protected file changed")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        raise ValueError("checksum mismatch")


def write_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
) -> JsonDict:  # pragma: no cover
    report = build_report(root, adversarial_receipts=adversarial_receipts, tests_run=tests_run)
    write_json(root / RESULT_RELATIVE_PATH, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    report = write_report(args.repo_root)
    print(json.dumps({"result_path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
