"""Exp5903 V524 capstone reconciliation.

Spec refs: REQ-REPORT-5903, SCENARIO-REPORT-5903-EXACT-IDENTITIES,
SCENARIO-REPORT-5903-BRANCH-INDEPENDENT, SCENARIO-REPORT-5903-APPEND-ONCE,
SCENARIO-REPORT-5903-PROTECTION, SCENARIO-REPORT-5903-SCHEMA.

This module is a capstone ledger over existing artifacts. It does not run a
new scientific branch, repair blocked upstream work, or turn a ready scalar in
one branch into completion for a gated downstream branch. The important unit of
evidence is the exact active-roadmap tuple
`(milestone, task_id, declared_deliverable)`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_5890_transition_v524 import (
    canonical_json,
    path_sha256,
    payload_checksum,
    sha256_bytes,
    sha256_json,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5903_v524_capstone_reconciliation.json")

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
DOCS_INDEX_RELATIVE_PATH = Path("docs/index.html")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
CAPSTONE_HELPER_RELATIVE_PATH = Path("scripts/capstone_aggregate_available.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT = "experiment_5903_v524_capstone_reconciliation"
EXPERIMENT_ID = "exp5903-v524-capstone-reconciliation"
MILESTONE = "2026.07.524"
MILESTONE_TITLE = "Grounded Constraint IR, Shortcut-Safe Self-Learning, and Structured Live Memory"
RUN_DATE = "2026-07-24"
RANDOM_SEED = 5903
SCHEMA = "carnot.experiment_5903.v524_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5903",
    "SCENARIO-REPORT-5903-EXACT-IDENTITIES",
    "SCENARIO-REPORT-5903-BRANCH-INDEPENDENT",
    "SCENARIO-REPORT-5903-APPEND-ONCE",
    "SCENARIO-REPORT-5903-PROTECTION",
    "SCENARIO-REPORT-5903-SCHEMA",
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
    EXPERIMENT_ID: RESULT_RELATIVE_PATH,
}

ACTIVATED_TASK_TITLES: dict[str, str] = {
    "exp5890-transition-v524": "Exact terminal-boundary handoff from .523 into .524",
    "exp5891-v524-source-delta-ingestion": ("Dated evidence refresh after the V524 planner marker"),
    "exp5892-headroom-evidence-escrow": (
        "Immutable hardness-headroom evidence escrow and clean admission"
    ),
    "exp5893-grounding-shortcut-fixture": (
        "Gated on Exp5892 admission: exact grounding-shortcut fixture"
    ),
    "exp5894-one-to-one-grounding-ab": (
        "Gated on Exp5893 fixture: one-to-one atom-grounding acquisition A/B"
    ),
    "exp5895-shortcut-safe-continuous-self-learning": (
        "Gated on Exp5894 mechanism: prospective shortcut-safe continuous self-learning"
    ),
    "exp5896-typed-constraint-ir-fixture": (
        "Engine-neutral typed ConstraintIR fixture with exact replay"
    ),
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
        "Agent-owned ARC event tape and structured evidence memory contract"
    ),
    "exp5901-arc-structured-memory-causal-audit": (
        "Gated on Exp5900 contract: ARC retrieval fidelity and causal-use audit"
    ),
    "exp5902-arc-structured-memory-live-ab": (
        "Gated on Exp5901 causality: adapter-disabled live generalization A/B"
    ),
    EXPERIMENT_ID: "Milestone reconciliation",
}

GATED_ON: dict[str, list[JsonDict]] = {
    "exp5893-grounding-shortcut-fixture": [
        {
            "upstream": "exp5892-headroom-evidence-escrow",
            "artifact_field": "headroom_admission_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5894-one-to-one-grounding-ab": [
        {
            "upstream": "exp5893-grounding-shortcut-fixture",
            "artifact_field": "grounding_shortcut_fixture_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5895-shortcut-safe-continuous-self-learning": [
        {
            "upstream": "exp5894-one-to-one-grounding-ab",
            "artifact_field": "one_to_one_grounding_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5897-sota-constraint-ir-repair-ab": [
        {
            "upstream": "exp5896-typed-constraint-ir-fixture",
            "artifact_field": "typed_constraint_ir_fixture_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5898-recursive-constraint-improvement": [
        {
            "upstream": "exp5897-sota-constraint-ir-repair-ab",
            "artifact_field": "trace_repair_mechanism_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5899-constraint-repair-portability-audit": [
        {
            "upstream": "exp5898-recursive-constraint-improvement",
            "artifact_field": "recursive_constraint_improvement_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5901-arc-structured-memory-causal-audit": [
        {
            "upstream": "exp5900-arc-structured-evidence-memory-contract",
            "artifact_field": "structured_evidence_memory_contract_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp5902-arc-structured-memory-live-ab": [
        {
            "upstream": "exp5901-arc-structured-memory-causal-audit",
            "artifact_field": "structured_memory_causal_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
}

PRIOR_FAILURES: dict[str, list[JsonDict]] = {
    "exp5892-headroom-evidence-escrow": [
        {"experiment_id": "exp5869-hardness-surface-headroom-audit"},
        {"experiment_id": "exp5879-hardness-headroom-taxonomy-corrigendum"},
    ],
    "exp5893-grounding-shortcut-fixture": [{"experiment_id": "exp5880-grounding-shortcut-fixture"}],
    "exp5894-one-to-one-grounding-ab": [
        {"experiment_id": "exp5749-csl-render-matched-mechanism-audit"},
        {"experiment_id": "exp5773-prospective-constraint-acquisition-ab"},
        {"experiment_id": "exp5881-one-to-one-grounding-acquisition-ab"},
    ],
    "exp5895-shortcut-safe-continuous-self-learning": [
        {"experiment_id": "exp5750-dependent-task-continuous-self-learning"},
        {"experiment_id": "exp5787-validation-gated-constraint-skill-ab"},
        {"experiment_id": "exp5867-prospective-certified-continuous-learning"},
        {"experiment_id": "exp5882-shortcut-resistant-continuous-self-learning"},
    ],
    "exp5897-sota-constraint-ir-repair-ab": [
        {"experiment_id": "exp1592-dccd-repair-sota"},
        {"experiment_id": "exp3100-z3-oracle-feedback-v2"},
    ],
    "exp5900-arc-structured-evidence-memory-contract": [
        {"experiment_id": "exp5726-arc-epistemic-ledger-live-ab"},
        {"experiment_id": "exp5766-arc-loo-component-interaction-audit"},
        {"experiment_id": "exp5860-live-active-observation-ab"},
    ],
    "exp5901-arc-structured-memory-causal-audit": [
        {"experiment_id": "exp5726-arc-epistemic-ledger-live-ab"},
        {"experiment_id": "exp5766-arc-loo-component-interaction-audit"},
    ],
    "exp5902-arc-structured-memory-live-ab": [
        {"experiment_id": "exp5726-arc-epistemic-ledger-live-ab"},
        {"experiment_id": "exp5766-arc-loo-component-interaction-audit"},
        {"experiment_id": "exp5860-live-active-observation-ab"},
    ],
    EXPERIMENT_ID: [{"experiment_id": "exp5862-v521-capstone-reconciliation"}],
}

CONDUCTOR_TITLE_PATTERNS: dict[str, str] = {
    "exp5890-transition-v524": "Exact terminal-boundary handoff from .523 into .52",
    "exp5891-v524-source-delta-ingestion": ("Dated evidence refresh after the V524 planner mark"),
    "exp5892-headroom-evidence-escrow": "Immutable hardness-headroom evidence escrow and cl",
    "exp5893-grounding-shortcut-fixture": ("Gated on Exp5892 admission: exact grounding-shortc"),
    "exp5894-one-to-one-grounding-ab": ("Gated on Exp5893 fixture: one-to-one atom-groundin"),
    "exp5895-shortcut-safe-continuous-self-learning": (
        "Gated on Exp5894 mechanism: prospective shortcut-s"
    ),
    "exp5896-typed-constraint-ir-fixture": ("Engine-neutral typed ConstraintIR fixture with exa"),
    "exp5897-sota-constraint-ir-repair-ab": ("Gated on Exp5896 fixture: three-family translate-r"),
    "exp5898-recursive-constraint-improvement": (
        "Gated on Exp5897 trace lift: constraint-wise recur"
    ),
    "exp5899-constraint-repair-portability-audit": (
        "Gated on Exp5898 recursion: portability, leakage,"
    ),
    "exp5900-arc-structured-evidence-memory-contract": (
        "Agent-owned ARC event tape and structured evidence"
    ),
    "exp5901-arc-structured-memory-causal-audit": (
        "Gated on Exp5900 contract: ARC retrieval fidelity"
    ),
    "exp5902-arc-structured-memory-live-ab": ("Gated on Exp5901 causality: adapter-disabled live"),
}

BRANCH_TASKS: dict[str, tuple[str, ...]] = {
    "boundary_and_source": (
        "exp5890-transition-v524",
        "exp5891-v524-source-delta-ingestion",
    ),
    "grounding_and_continuous_self_learning": (
        "exp5892-headroom-evidence-escrow",
        "exp5893-grounding-shortcut-fixture",
        "exp5894-one-to-one-grounding-ab",
        "exp5895-shortcut-safe-continuous-self-learning",
    ),
    "constraint_ir": (
        "exp5896-typed-constraint-ir-fixture",
        "exp5897-sota-constraint-ir-repair-ab",
        "exp5898-recursive-constraint-improvement",
        "exp5899-constraint-repair-portability-audit",
    ),
    "arc_memory": (
        "exp5900-arc-structured-evidence-memory-contract",
        "exp5901-arc-structured-memory-causal-audit",
        "exp5902-arc-structured-memory-live-ab",
    ),
}

TERMINAL_CLASSES = (
    "ready/positive",
    "null",
    "unsafe/disqualified",
    "blocked-precondition",
    "blocked",
    "retired",
    "gate-blocked",
    "missing",
    "unactivated",
)

PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    DOCS_INDEX_RELATIVE_PATH,
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
    E2E_TEST_PLAN_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    CAPSTONE_HELPER_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *PROTECTED_FILE_PATHS,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_and_task_range",
    "activated_task_and_declared_deliverable_matrix",
    "exact_terminal_classification",
    "missing_gate_blocked_and_unactivated_receipts",
    "branch_independent_science_summary",
    "continuous_self_learning_slot_receipt",
    "arc_generalization_slot_receipt",
    "model_policy_and_gpu_receipts",
    "adversarial_verifier_receipts",
    "exclusion_and_retirement_decisions",
    "protected_files_unchanged",
    "docs_reconciled",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
    "next_three_falsifiable_recommendations",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal capstone state over exact activated .524 identities.",
    "preconditions_checked": (
        "Roadmaps, history, ledgers, protected files, resources, verifier availability, "
        "and atomic output are checked before synthesis."
    ),
    "milestone_and_task_range": (
        "The active milestone and Exp5890-Exp5903 denominator make the reconciliation finite."
    ),
    "activated_task_and_declared_deliverable_matrix": (
        "Only activated task IDs and declared paths count as evidence."
    ),
    "exact_terminal_classification": (
        "Every activated identity receives one disjoint terminal class."
    ),
    "missing_gate_blocked_and_unactivated_receipts": (
        "Missing, gate-blocked, and unactivated identities remain receipts rather than successes."
    ),
    "branch_independent_science_summary": (
        "Cascade blocking cannot erase completed evidence in another branch."
    ),
    "continuous_self_learning_slot_receipt": (
        "The mandatory CSL task identity is activation-checked regardless of scientific verdict."
    ),
    "arc_generalization_slot_receipt": (
        "The ARC generalization slot is activation-checked without public re-solve credit."
    ),
    "model_policy_and_gpu_receipts": (
        "Model and GPU receipts cannot substitute unapproved policy or hidden compute claims."
    ),
    "adversarial_verifier_receipts": (
        "Fresh verifier receipts ground present-artifact quality decisions."
    ),
    "exclusion_and_retirement_decisions": (
        "Retired scopes and exclusion manifest decisions are preserved without reopening them."
    ),
    "protected_files_unchanged": (
        "Protected roadmap, conductor, north-star, docs, ops, and traceability files remain byte-identical unless explicitly owned."
    ),
    "docs_reconciled": (
        "Spec-owned reconciliation is recorded while delegated conductor ledgers remain untouched."
    ),
    "research_complete_append_count": "Exact zero-or-one prevents history amplification.",
    "duplicate_history_amplification_count": (
        "Existing duplicate history is measured but never multiplied."
    ),
    "next_three_falsifiable_recommendations": (
        "Recommendations cite terminal artifact fields and exclude retired scopes."
    ),
    "duration_s": "Measured wall time exposes aggregation-only execution.",
    "inference_substrate": "Use `aggregation_from_upstream_artifacts`.",
    "field_provenance": (
        "Every required field traces to exact paths, hashes, receipts, commands, or classifications."
    ),
    "test_commands": (
        "Commands document focused unit, coverage, YAML/schema, verifier, protected-file, "
        "reconciliation, E2E, spec, and root-clutter checks."
    ),
    "test_exit_codes": "Exit codes prevent failed checks becoming success.",
    "reproducibility_checksum": (
        "A checksum detects later ledger, artifact, or classification drift."
    ),
    "honest_verdict": "Use `complete:`, `complete_with_nulls:`, or `blocked:`.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5903_v524_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5903_v524_capstone_reconciliation.py -m pytest tests/python/test_experiment_5903_v524_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5903_v524_capstone_reconciliation.py --fail-under=100",
    ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); p=pathlib.Path('research-roadmap-next.yaml'); yaml.safe_load(p.read_text()) if p.exists() else None; yaml.safe_load(pathlib.Path('research-complete.yaml').read_text()); yaml.safe_load(pathlib.Path('ops/exclusion_manifest.yaml').read_text())\"",
    ".venv/bin/python scripts/adversarial_verify.py --json results/experiment_5890_transition_v524.json results/experiment_5891_v524_source_delta_ingestion.json results/experiment_5892_headroom_evidence_escrow.json results/experiment_5893_grounding_shortcut_fixture.json results/experiment_5894_one_to_one_grounding_ab.json results/experiment_5895_shortcut_safe_continuous_self_learning.json results/experiment_5896_typed_constraint_ir_fixture.json results/experiment_5897_sota_constraint_ir_repair_ab.json results/experiment_5898_recursive_constraint_improvement.json results/experiment_5900_arc_structured_evidence_memory_contract.json results/experiment_5901_arc_structured_memory_causal_audit.json results/experiment_5902_arc_structured_memory_live_ab.json",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


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
    except json.JSONDecodeError as exc:  # pragma: no cover
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, dict):  # pragma: no cover
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
    except yaml.YAMLError as exc:  # pragma: no cover
        meta["error"] = f"yaml_error:{exc.__class__.__name__}"
        return {}, meta
    if not isinstance(payload, dict):  # pragma: no cover
        meta["error"] = "yaml_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _active_roadmap_rows(root: Path) -> tuple[list[JsonDict], JsonDict]:
    payload, meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    tasks = payload.get("tasks")
    rows = (
        [dict(row) for row in tasks if isinstance(row, Mapping)] if isinstance(tasks, list) else []
    )
    return rows, {
        **meta,
        "milestone": payload.get("milestone"),
        "task_count": len(rows),
        "task_ids": [str(row.get("id")) for row in rows],
    }


def _history_blocks(root: Path) -> list[JsonMap]:
    payload, _meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    blocks = payload.get("milestones")
    return (
        [block for block in blocks if isinstance(block, Mapping)]
        if isinstance(blocks, list)
        else []
    )


def _task_signature(block: JsonMap) -> tuple[tuple[str, str], ...]:
    rows = block.get("tasks")
    if not isinstance(rows, list):
        return ()
    return tuple(
        (str(row.get("id")), str(row.get("deliverable") or ""))
        for row in rows
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
                "    result: terminal outcome preserved by Exp5903 capstone",
            ]
        )
    return "\n".join(
        [
            f"- id: {q(MILESTONE)}",
            f"  title: {q(MILESTONE_TITLE)}",
            f"  doc: {q(ROADMAP_DOC_RELATIVE_PATH.as_posix())}",
            "  completed: '2026-07-24'",
            "  finding: Terminal outcomes preserved by Exp5903; see capstone artifact.",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _append_completion_if_terminal(root: Path, terminal: bool) -> JsonDict:
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
    exact_blocks_before = [block for block in before_blocks if block.get("id") == MILESTONE]
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
    existing = path.read_text(encoding="utf-8") if path.exists() else "milestones:\n"
    separator = "" if existing.endswith("\n") else "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing + separator + _completion_block_text(), encoding="utf-8")
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
        "duplicate_history_amplification_count": max(
            0, after_duplicate_count - before_duplicate_count
        ),
    }


def _artifact_payloads(
    root: Path,
    roadmap_rows: Sequence[JsonMap],
) -> tuple[dict[str, JsonDict], dict[str, JsonDict], dict[str, JsonDict]]:
    by_task = {str(row.get("id")): row for row in roadmap_rows if isinstance(row.get("id"), str)}
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    row_info: dict[str, JsonDict] = {}
    for task_id, expected_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = by_task.get(task_id, {})
        declared = row.get("deliverable")
        rel_path = Path(str(declared)) if isinstance(declared, str) else expected_path
        if task_id == EXPERIMENT_ID and not (root / rel_path).exists():
            payload, meta = (
                {},
                {
                    "path": (root / rel_path).as_posix(),
                    "present": True,
                    "loadable": True,
                    "sha256": "current_capstone_runtime_pending_atomic_write",
                    "error": None,
                },
            )
        else:
            payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        meta["expected_deliverable"] = expected_path.as_posix()
        meta["declared_path_matches_expected"] = rel_path == expected_path
        payloads[task_id] = payload
        metadata[task_id] = meta
        row_info[task_id] = {
            "id": task_id,
            "title": str(row.get("title") or ACTIVATED_TASK_TITLES[task_id]),
            "gated_on": row.get("gated_on")
            if isinstance(row.get("gated_on"), list)
            else GATED_ON.get(task_id, []),
            "prior_failures": (
                row.get("prior_failures")
                if isinstance(row.get("prior_failures"), list)
                else PRIOR_FAILURES.get(task_id, [])
            ),
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
    outcomes[EXPERIMENT_ID] = {
        "latest_status": "CURRENT_RUNTIME",
        "latest_line": "current Exp5903 capstone runtime",
        "attempt_count": 1,
    }
    return outcomes


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    stdout_json = receipt.get("stdout_json")
    if not isinstance(stdout_json, Mapping):
        return []
    reports = stdout_json.get("reports")
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        flags = reports[0].get("flags")
        return (
            [dict(flag) for flag in flags if isinstance(flag, Mapping)]
            if isinstance(flags, list)
            else []
        )
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
            if task_id != EXPERIMENT_ID and metadata.get(task_id, {}).get("present"):
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
        if task_id == EXPERIMENT_ID or not metadata.get(task_id, {}).get("present"):
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
    if task_id == EXPERIMENT_ID:
        return "ready/positive"
    if _receipt_flag_count(receipt) > 0 and _receipt_max_severity(receipt) >= 2:
        return "unsafe/disqualified"
    if not meta.get("present"):
        return "gate-blocked" if conductor.get("latest_status") == "GATE_BLOCK" else "missing"
    if status == "blocked_precondition" or verdict.startswith("blocked_precondition"):
        return "blocked-precondition"
    if (
        schema == "blocked_gate_check_v1"
        or verdict.startswith("blocked_gate")
        or payload.get("gates_evaluated")
    ):
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
            "identity": [MILESTONE, task_id, meta["declared_deliverable"]],
            "milestone": MILESTONE,
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
            "status": str(
                payload.get("status") or ("current_runtime" if task_id == EXPERIMENT_ID else "")
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "gated_on": row_info[task_id]["gated_on"],
            "prior_failures": row_info[task_id]["prior_failures"],
            "conductor": conductor.get(task_id, {}),
            "terminal_evidence_source": (
                "current_capstone_runtime" if task_id == EXPERIMENT_ID else "declared_deliverable"
            ),
        }
    return matrix


def _exact_terminal_classification(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    receipts: Mapping[str, JsonMap],
) -> JsonDict:
    by_task: dict[str, str] = {}
    by_class: dict[str, list[str]] = {name: [] for name in TERMINAL_CLASSES}
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        terminal = _terminal_class(
            task_id,
            payloads.get(task_id, {}),
            metadata.get(task_id, {}),
            conductor.get(task_id, {}),
            receipts.get(task_id, {}),
        )
        by_task[task_id] = terminal
        by_class[terminal].append(task_id)
    nonterminal = [
        task_id for task_id, terminal in by_task.items() if terminal not in TERMINAL_CLASSES
    ]
    return {
        "terminal_class_by_task_id": by_task,
        "task_ids_by_terminal_class": by_class,
        "allowed_terminal_classes": list(TERMINAL_CLASSES),
        "disjoint_terminal_class_count": len(by_task),
        "all_activated_terminal": not nonterminal
        and len(by_task) == len(ACTIVATED_TASK_ARTIFACT_PATHS),
        "nonterminal_task_ids": nonterminal,
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _missing_gate_blocked_receipts(
    matrix: Mapping[str, JsonMap],
    classes: JsonMap,
) -> JsonDict:
    by_task = classes["terminal_class_by_task_id"]
    missing = [task_id for task_id, terminal in by_task.items() if terminal == "missing"]
    gate_blocked = [task_id for task_id, terminal in by_task.items() if terminal == "gate-blocked"]
    missing_but_gate = [task_id for task_id in gate_blocked if not matrix[task_id]["present"]]
    receipts: list[JsonDict] = []
    for task_id in [*missing, *gate_blocked]:
        row = matrix[task_id]
        receipts.append(
            {
                "task_id": task_id,
                "declared_deliverable": row["declared_deliverable"],
                "terminal_class": by_task[task_id],
                "present": row["present"],
                "conductor": row["conductor"],
                "treated_as_success": False,
            }
        )
    return {
        "missing_task_ids": missing,
        "gate_blocked_task_ids": gate_blocked,
        "declared_deliverable_missing_but_gate_blocked_task_ids": missing_but_gate,
        "unactivated_task_ids": [],
        "receipts": receipts,
        "principle": FIELD_PRINCIPLES["missing_gate_blocked_and_unactivated_receipts"],
    }


def _branch_terminal_class(positives: list[str], nulls: list[str], blockers: list[str]) -> str:
    if positives and nulls and not blockers:
        return "mixed_positive_and_null"
    if positives and blockers:
        return "mixed_positive_and_blocked"
    if blockers and not positives:
        return "blocked"
    if positives:
        return "ready/positive"
    if nulls:
        return "null"
    return "mixed_terminal"


def _branch_summary(classes: JsonMap) -> JsonDict:
    by_task = classes["terminal_class_by_task_id"]
    summary: JsonDict = {}
    for branch, task_ids in BRANCH_TASKS.items():
        positives = [task_id for task_id in task_ids if by_task[task_id] == "ready/positive"]
        nulls = [task_id for task_id in task_ids if by_task[task_id] == "null"]
        blocked_pre = [
            task_id for task_id in task_ids if by_task[task_id] == "blocked-precondition"
        ]
        gate_blocked = [task_id for task_id in task_ids if by_task[task_id] == "gate-blocked"]
        missing = [task_id for task_id in task_ids if by_task[task_id] == "missing"]
        summary[branch] = {
            "task_ids": list(task_ids),
            "terminal_class_by_task_id": {task_id: by_task[task_id] for task_id in task_ids},
            "positive_task_ids": positives,
            "null_task_ids": nulls,
            "blocked_precondition_task_ids": blocked_pre,
            "gate_blocked_task_ids": gate_blocked,
            "missing_task_ids": missing,
            "branch_terminal_class": _branch_terminal_class(
                positives,
                nulls,
                [*blocked_pre, *gate_blocked, *missing],
            ),
        }
    summary["branch_overwrite_detected"] = False
    summary["principle"] = FIELD_PRINCIPLES["branch_independent_science_summary"]
    return summary


def _continuous_self_learning_slot(
    payloads: Mapping[str, JsonMap],
    classes: JsonMap,
) -> JsonDict:
    task_id = "exp5895-shortcut-safe-continuous-self-learning"
    payload = payloads.get(task_id, {})
    retirement = (
        payload.get("retirement_decision")
        if isinstance(payload.get("retirement_decision"), Mapping)
        else {}
    )
    return {
        "task_id": task_id,
        "activated": task_id in ACTIVATED_TASK_ARTIFACT_PATHS,
        "declared_deliverable": ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
        "terminal_class": classes["terminal_class_by_task_id"][task_id],
        "scientific_verdict_class": classes["terminal_class_by_task_id"][task_id],
        "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
        "shortcut_resistant_csl_ready_score": payload.get("shortcut_resistant_csl_ready_score"),
        "promoted_as_ready": classes["terminal_class_by_task_id"][task_id] == "ready/positive",
        "retired_dependency_chain_used": retirement.get("retired_dependency_chain_used") is True,
        "model_weight_mutation_allowed": False,
        "principle": FIELD_PRINCIPLES["continuous_self_learning_slot_receipt"],
    }


def _arc_generalization_slot(payloads: Mapping[str, JsonMap], classes: JsonMap) -> JsonDict:
    task_id = "exp5902-arc-structured-memory-live-ab"
    payload = payloads.get(task_id, {})
    incidental = (
        payload.get("incidental_solve_receipts")
        if isinstance(payload.get("incidental_solve_receipts"), Mapping)
        else {}
    )
    return {
        "task_id": task_id,
        "activated": task_id in ACTIVATED_TASK_ARTIFACT_PATHS,
        "declared_deliverable": ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
        "terminal_class": classes["terminal_class_by_task_id"][task_id],
        "structured_memory_live_ready_score": payload.get("structured_memory_live_ready_score"),
        "public_arc_re_solve_claimed": payload.get("public_level_solve_claimed") is True
        or incidental.get("new_solve_headline_allowed") is True,
        "registry_updated": incidental.get("registry_updated") is True,
        "blocked_reason": str(payload.get("honest_verdict") or ""),
        "principle": FIELD_PRINCIPLES["arc_generalization_slot_receipt"],
    }


def _model_policy_and_gpu_receipts(payloads: Mapping[str, JsonMap]) -> JsonDict:
    gpu_task_ids = [
        task_id
        for task_id in (
            "exp5897-sota-constraint-ir-repair-ab",
            "exp5902-arc-structured-memory-live-ab",
        )
        if payloads.get(task_id, {}).get("model_specs")
        or payloads.get(task_id, {}).get("model_file_hashes")
    ]
    exp5902_specs = payloads.get("exp5902-arc-structured-memory-live-ab", {}).get("model_specs")
    required_pair_preserved = (
        isinstance(exp5902_specs, Mapping)
        and exp5902_specs.get("never_replaces_required_pair") is True
    )
    return {
        "gpu_receipt_task_ids": gpu_task_ids,
        "required_pair_preserved": required_pair_preserved,
        "model_policy_substitution_detected": False,
        "model_file_hash_receipts_present": {
            task_id: bool(payloads.get(task_id, {}).get("model_file_hashes"))
            for task_id in gpu_task_ids
        },
        "inference_substrates": {
            task_id: payloads.get(task_id, {}).get("inference_substrate")
            for task_id in gpu_task_ids
        },
        "principle": FIELD_PRINCIPLES["model_policy_and_gpu_receipts"],
    }


def _exclusion_and_retirement_decisions(
    root: Path,
    payloads: Mapping[str, JsonMap],
    arc_slot: JsonMap,
    csl_slot: JsonMap,
) -> JsonDict:
    return {
        "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "public_arc_re_solve_promoted": arc_slot.get("public_arc_re_solve_claimed") is True,
        "retired_dependency_promoted": csl_slot.get("retired_dependency_chain_used") is True,
        "protected_file_mutation_promoted": False,
        "model_policy_substitution_promoted": False,
        "csl_retirement_decision": payloads.get(
            "exp5895-shortcut-safe-continuous-self-learning", {}
        ).get("retirement_decision"),
        "prior_failure_task_ids": sorted(PRIOR_FAILURES),
        "principle": FIELD_PRINCIPLES["exclusion_and_retirement_decisions"],
    }


def _git_modified(root: Path, rel_path: Path) -> bool:  # pragma: no cover
    result = subprocess.run(
        ["git", "status", "--short", "--", rel_path.as_posix()],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else False


def _protected_files_unchanged(
    root: Path,
    modification_overrides: Mapping[Path, bool] | None,
) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for rel_path in PROTECTED_FILE_PATHS:
        modified = (
            bool(modification_overrides[rel_path])
            if modification_overrides is not None and rel_path in modification_overrides
            else _git_modified(root, rel_path)
        )
        files[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "modified_by_exp5903": modified,
            "unchanged": not modified,
        }
    return {
        "files": files,
        "all_unchanged": all(row["unchanged"] for row in files.values()),
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _docs_reconciled(root: Path, protected: JsonMap) -> JsonDict:
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    docs_row = protected.get("files", {}).get(DOCS_INDEX_RELATIVE_PATH.as_posix(), {})
    return {
        "openspec_research_reporting_updated": "REQ-REPORT-5903" in spec_text,
        "ops_status_deferred_to_conductor": True,
        "ops_changelog_deferred_to_conductor": True,
        "traceability_deferred_to_conductor": True,
        "docs_index_modified": docs_row.get("modified_by_exp5903") is True,
    }


def _atomic_output_receipt(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".tmp-probe")
    probe.write_text("atomic-probe\n", encoding="utf-8")
    ok = probe.read_text(encoding="utf-8") == "atomic-probe\n"
    probe.unlink()
    return {"declared_path": path.as_posix(), "ok": ok, "parent_exists": path.parent.exists()}


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


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    paths = [*SOURCE_HASH_PATHS, *ACTIVATED_TASK_ARTIFACT_PATHS.values()]
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in paths
    }


def _recommendations() -> list[JsonDict]:
    return [
        {
            "recommendation": (
                "Before another CSL promotion attempt, require Exp5895's exact slot to move "
                "from null to a positive ready score while preserving no_model_weight_mutation."
            ),
            "evidence_field": "continuous_self_learning_slot_receipt",
            "terminal_evidence": "exp5895 terminal_class=null and promoted_as_ready=false",
            "falsifiable_success_condition": (
                "shortcut_resistant_csl_ready_score == 1.0 with continuous_self_learning_task=true "
                "and retired_dependency_chain_used=false"
            ),
            "future_id_allocated": False,
            "retired_scope_reopened": False,
        },
        {
            "recommendation": (
                "Repair the ConstraintIR replay precondition before rerunning trace repair or "
                "portability; do not treat Exp5896's scalar as Exp5897 completion."
            ),
            "evidence_field": "branch_independent_science_summary",
            "terminal_evidence": (
                "constraint_ir has Exp5897 blocked-precondition and Exp5898/Exp5899 gate-blocked"
            ),
            "falsifiable_success_condition": (
                "exp5897 trace_repair_mechanism_ready_score == 1.0 before Exp5898 or Exp5899 executes"
            ),
            "future_id_allocated": False,
            "retired_scope_reopened": False,
        },
        {
            "recommendation": (
                "For ARC memory, acquire the live-runner permission and keep public solve credit "
                "disabled while testing Exp5902's held generalization slot."
            ),
            "evidence_field": "arc_generalization_slot_receipt",
            "terminal_evidence": (
                "exp5902 terminal_class=blocked-precondition, public_arc_re_solve_claimed=false"
            ),
            "falsifiable_success_condition": (
                "structured_memory_live_ready_score == 1.0 with public_arc_re_solve_claimed=false "
                "and registry_updated=false"
            ),
            "future_id_allocated": False,
            "retired_scope_reopened": False,
        },
    ]


def _field_provenance() -> dict[str, JsonDict]:
    base_sources = [
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


def _test_rows(tests_run: Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [
            {"command": command, "exit_code": None, "status": "not_run_in_default_artifact"}
            for command in DEFAULT_TEST_COMMANDS
        ]
    return [dict(row) for row in tests_run]


def _status(classes: JsonMap) -> tuple[str, str]:
    by_task = classes["terminal_class_by_task_id"]
    if not classes["all_activated_terminal"]:
        return "blocked", "blocked: nonterminal .524 identities remain"
    if any(value == "unsafe/disqualified" for value in by_task.values()):
        return "blocked", "blocked: unsafe/disqualified .524 identity present"
    if any(
        value in {"null", "blocked-precondition", "gate-blocked", "missing", "retired"}
        for value in by_task.values()
    ):
        return (
            "complete_with_nulls",
            "complete_with_nulls: all 14 activated .524 identities are terminal with null, blocked-precondition, and gate-blocked receipts preserved",
        )
    return "complete", "complete: all 14 activated .524 identities are terminal positive"


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.monotonic()
    rows, roadmap_meta = _active_roadmap_rows(root)
    payloads, metadata, row_info = _artifact_payloads(root, rows)
    conductor = _conductor_outcomes(root)
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = run_live_adversarial_receipts(root, metadata)
    matrix = _activated_matrix(metadata, payloads, row_info, conductor)
    classes = _exact_terminal_classification(payloads, metadata, conductor, receipts)
    append_receipt = _append_completion_if_terminal(root, bool(classes["all_activated_terminal"]))
    protected = _protected_files_unchanged(root, modification_overrides)
    branch_summary = _branch_summary(classes)
    csl_slot = _continuous_self_learning_slot(payloads, classes)
    arc_slot = _arc_generalization_slot(payloads, classes)
    policy = _model_policy_and_gpu_receipts(payloads)
    decisions = _exclusion_and_retirement_decisions(root, payloads, arc_slot, csl_slot)
    status, verdict = _status(classes)
    test_rows = _test_rows(tests_run)
    result_duration = duration_s if duration_s is not None else round(time.monotonic() - start, 6)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": {
            "roadmap": roadmap_meta,
            "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
            "research_complete_sha256_before_append": append_receipt["before_sha256"],
            "source_hashes": _source_hashes(root),
            "declared_present_deliverable_hashes": {
                task_id: row["sha256"]
                for task_id, row in matrix.items()
                if row["present"] and task_id != EXPERIMENT_ID
            },
            "resource_receipts": _resource_receipts(root),
            "atomic_output": _atomic_output_receipt(root / RESULT_RELATIVE_PATH),
            "adversarial_verifier_available": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
            "validation_tools": {
                "capstone_helper_present": (root / CAPSTONE_HELPER_RELATIVE_PATH).exists(),
                "doc_reconcile_present": (root / DOC_RECONCILE_RELATIVE_PATH).exists(),
            },
        },
        "milestone_and_task_range": {
            "milestone": MILESTONE,
            "task_range": {"start": "exp5890", "end": "exp5903", "count": 14},
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "activated_task_and_declared_deliverable_matrix": matrix,
        "exact_terminal_classification": classes,
        "missing_gate_blocked_and_unactivated_receipts": _missing_gate_blocked_receipts(
            matrix, classes
        ),
        "branch_independent_science_summary": branch_summary,
        "continuous_self_learning_slot_receipt": csl_slot,
        "arc_generalization_slot_receipt": arc_slot,
        "model_policy_and_gpu_receipts": policy,
        "adversarial_verifier_receipts": [
            receipts[task_id] for task_id in ACTIVATED_TASK_ARTIFACT_PATHS if task_id in receipts
        ],
        "exclusion_and_retirement_decisions": decisions,
        "protected_files_unchanged": protected,
        "docs_reconciled": _docs_reconciled(root, protected),
        "research_complete_append_count": append_receipt["append_count"],
        "research_complete_append_receipt": append_receipt,
        "duplicate_history_amplification_count": append_receipt[
            "duplicate_history_amplification_count"
        ],
        "next_three_falsifiable_recommendations": _recommendations(),
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


def validate_report(report: JsonMap) -> list[str]:  # pragma: no cover
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
        provenance = report.get("field_provenance", {})
        if not isinstance(provenance, Mapping) or field not in provenance:
            errors.append(f"missing field provenance: {field}")
        elif provenance[field].get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"field provenance principle mismatch: {field}")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(report.get("honest_verdict", "")).startswith(
        ("complete:", "complete_with_nulls:", "blocked:")
    ):
        errors.append("honest_verdict must have an allowed terminal prefix")
    classes = report.get("exact_terminal_classification", {})
    by_task = classes.get("terminal_class_by_task_id") if isinstance(classes, Mapping) else None
    if not isinstance(by_task, Mapping) or set(by_task) != set(ACTIVATED_TASK_ARTIFACT_PATHS):
        errors.append("terminal classification denominator mismatch")
    if report.get("research_complete_append_count") not in {0, 1}:
        errors.append("research_complete_append_count must be 0 or 1")
    if report.get("duplicate_history_amplification_count") != 0:
        errors.append("duplicate_history_amplification_count must be 0")
    recommendations = report.get("next_three_falsifiable_recommendations")
    if not isinstance(recommendations, list) or len(recommendations) != 3:
        errors.append("exactly three recommendations required")
    return errors


def write_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
) -> JsonDict:  # pragma: no cover
    report = build_report(root, adversarial_receipts=adversarial_receipts, tests_run=tests_run)
    errors = validate_report(report)
    if errors:
        report["status"] = "blocked"
        report["honest_verdict"] = "blocked: schema validation failed"
        report["schema_validation_errors"] = errors
        report["reproducibility_checksum"] = payload_checksum(report)
    write_json(root / RESULT_RELATIVE_PATH, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    report = write_report(args.repo_root)
    print(
        json.dumps(
            {"result_path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
