"""Exp5837 transition receipt from terminal milestone .519 into .520.

Spec refs: REQ-REPORT-5837, SCENARIO-REPORT-5837,
SCENARIO-REPORT-5837-VERIFIER-RECEIPTS,
SCENARIO-REPORT-5837-COLLISION-BLOCK,
SCENARIO-REPORT-5837-FIELD-PROVENANCE.

This module archives a terminal milestone by exact declared identity. Its main
risk is evidence laundering: Exp5828 looks positive in its own verdict text but
is adversarially stamped, and Exp5829 depends on that flagged lifecycle
artifact. The transition therefore records both as completed evidence without
letting either become clean or headline-eligible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

import yaml

from carnot.experiment_5754_v513_capstone_reconciliation import (
    _read_json_any,
    path_sha256,
    payload_checksum,
    sha256_bytes,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5837_transition_v520.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EVIDENCE_INDEX_RELATIVE_PATH = Path("scripts/evidence_index_collision_preflight.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")

EXPERIMENT = "experiment_5837_transition_v520"
EXPERIMENT_ID = "exp5837-transition-v520"
MILESTONE_FROM = "2026.07.519"
MILESTONE_TO = "2026.07.520"
NEXT_TASK_RANGE = "exp5837-exp5848"
RESERVED_TASK_RANGE = "exp5830-exp5836"
RUN_DATE = "2026-07-23"
RANDOM_SEED = 5837
SCHEMA = "carnot.experiment_5837.transition_v520.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5837",
    "SCENARIO-REPORT-5837",
    "SCENARIO-REPORT-5837-VERIFIER-RECEIPTS",
    "SCENARIO-REPORT-5837-COLLISION-BLOCK",
    "SCENARIO-REPORT-5837-FIELD-PROVENANCE",
)

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5823-transition-v519": Path("results/experiment_5823_transition_v519.json"),
    "exp5824-v519-source-delta-ingestion": Path(
        "results/experiment_5824_v519_source_delta_ingestion.json"
    ),
    "exp5825-certified-adaptive-memory-contract": Path(
        "results/experiment_5825_certified_adaptive_memory_contract.json"
    ),
    "exp5826-out-of-template-constraint-stream": Path(
        "results/experiment_5826_out_of_template_constraint_stream.json"
    ),
    "exp5827-minimal-core-structural-acquisition-ab": Path(
        "results/experiment_5827_minimal_core_structural_acquisition_ab.json"
    ),
    "exp5828-future-validated-structural-memory": Path(
        "results/experiment_5828_future_validated_structural_memory.json"
    ),
    "exp5829-transfer-selective-replay-audit": Path(
        "results/experiment_5829_transfer_selective_replay_audit.json"
    ),
}
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

TASK_TITLES: dict[str, str] = {
    "exp5823-transition-v519": (
        "Archive terminal .518 evidence, retire finite-ID answer transport, and allocate .519"
    ),
    "exp5824-v519-source-delta-ingestion": (
        "Time-windowed post-V519 literature and implementation freshness receipt"
    ),
    "exp5825-certified-adaptive-memory-contract": (
        "Canonical certified-event and adaptive-memory preflight contract"
    ),
    "exp5826-out-of-template-constraint-stream": (
        "Gated on Exp5825 contract: chronological out-of-template constraint structure stream"
    ),
    "exp5827-minimal-core-structural-acquisition-ab": (
        "Gated on Exp5826 stream: minimal-core and active-query structural acquisition A/B"
    ),
    "exp5828-future-validated-structural-memory": (
        "Gated on Exp5827 structural lift: future-validated write-protected continuous memory"
    ),
    "exp5829-transfer-selective-replay-audit": (
        "Gated on Exp5828 durable memory: transfer-selective replay and recurrence audit"
    ),
}

CONDUCTOR_TITLE_PATTERNS: dict[str, str] = {
    "exp5823-transition-v519": "Archive terminal .518 evidence, retire finite-ID a",
    "exp5824-v519-source-delta-ingestion": "Time-windowed post-V519 literature and implementat",
    "exp5825-certified-adaptive-memory-contract": (
        "Canonical certified-event and adaptive-memory pref"
    ),
    "exp5826-out-of-template-constraint-stream": (
        "Gated on Exp5825 contract: chronological out-of-te"
    ),
    "exp5827-minimal-core-structural-acquisition-ab": (
        "Gated on Exp5826 stream: minimal-core and active-q"
    ),
    "exp5828-future-validated-structural-memory": (
        "Gated on Exp5827 structural lift: future-validated"
    ),
    "exp5829-transfer-selective-replay-audit": (
        "Gated on Exp5828 durable memory: transfer-selectiv"
    ),
}

VERIFIER_TASK_IDS = (
    "exp5825-certified-adaptive-memory-contract",
    "exp5826-out-of-template-constraint-stream",
    "exp5827-minimal-core-structural-acquisition-ab",
    "exp5828-future-validated-structural-memory",
    "exp5829-transfer-selective-replay-audit",
)

RESERVED_UNACTIVATED_TASK_IDS = (
    "exp5830-sota-paired-embedding-corpus",
    "exp5831-cross-family-embedding-energy-verifier",
    "exp5832-arc-write-protected-world-fact-tape",
    "exp5833-arc-world-feedback-probe-ab",
    "exp5834-bounded-adaptive-memory-microkernel",
    "exp5835-attached-board-adaptive-memory-receipts",
    "exp5836-capstone-v519",
)

NEXT_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5837-transition-v520": RESULT_RELATIVE_PATH,
    "exp5838-v520-source-delta-ingestion": Path(
        "results/experiment_5838_v520_source_delta_ingestion.json"
    ),
    "exp5839-v519-evidence-qualification": Path(
        "results/experiment_5839_v519_evidence_qualification.json"
    ),
    "exp5840-exact-counterfactual-embedding-fixture": Path(
        "results/experiment_5840_exact_counterfactual_embedding_fixture.json"
    ),
    "exp5841-sota-comparative-embedding-corpus": Path(
        "results/experiment_5841_sota_comparative_embedding_corpus.json"
    ),
    "exp5842-cross-family-comparative-energy-verifier": Path(
        "results/experiment_5842_cross_family_comparative_energy_verifier.json"
    ),
    "exp5843-sparse-oracle-continuous-learning": Path(
        "results/experiment_5843_sparse_oracle_continuous_learning.json"
    ),
    "exp5844-arc-write-protected-fact-tape": Path(
        "results/experiment_5844_arc_write_protected_fact_tape.json"
    ),
    "exp5845-arc-world-feedback-acquisition-ab": Path(
        "results/experiment_5845_arc_world_feedback_acquisition_ab.json"
    ),
    "exp5846-bounded-adaptive-memory-microkernel": Path(
        "results/experiment_5846_bounded_adaptive_memory_microkernel.json"
    ),
    "exp5847-attached-board-memory-receipts": Path(
        "results/experiment_5847_attached_board_memory_receipts.json"
    ),
    "exp5848-capstone-v520": Path("results/experiment_5848_capstone_v520.json"),
}
NEXT_EXTRA_DELIVERABLE_PATHS = (
    Path("results/experiment_5840_exact_counterfactual_embedding_fixture.rows.jsonl"),
    Path("results/experiment_5841_sota_comparative_embedding_corpus.npz"),
)
NEXT_TASK_IDS = tuple(NEXT_TASK_ARTIFACT_PATHS)
NEXT_RANGE_NUMBERS = range(5837, 5849)

PROTECTED_FILE_PATHS = (ROADMAP_RELATIVE_PATH, CONDUCTOR_RELATIVE_PATH)
ALLOWED_ALLOCATION_TEXT_PATHS = (ROADMAP_RELATIVE_PATH, VNEXT_RELATIVE_PATH)
COLLISION_TEXT_PATHS = (
    RESEARCH_COMPLETE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": (
            '.venv/bin/python -c "import pathlib, yaml; '
            "yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); "
            "p=pathlib.Path('research-roadmap-next.yaml'); "
            "yaml.safe_load(p.read_text()) if p.exists() else None; "
            "yaml.safe_load(pathlib.Path('research-complete.yaml').read_text()); "
            "yaml.safe_load(pathlib.Path('ops/exclusion_manifest.yaml').read_text())\""
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/pytest tests/python/test_experiment_5837_transition_v520.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage run --rcfile=/dev/null "
            "--include=python/carnot/experiment_5837_transition_v520.py "
            "-m pytest tests/python/test_experiment_5837_transition_v520.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage report --rcfile=/dev/null "
            "--include=python/carnot/experiment_5837_transition_v520.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/root_clutter_sweep.py",
        "exit_code": None,
        "status": "not_run",
    },
)

REQUIRED_PRINCIPLE_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "declared_deliverable_matrix",
    "adversarial_verifier_receipts",
    "outcome_classification",
    "flagged_evidence_preserved",
    "reserved_unactivated_task_ids",
    "research_complete_append_count",
    "next_task_range",
    "next_range_collision_count",
    "docs_reconciled",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Identifies the versioned Exp5837 transition artifact schema.",
    "experiment": "Names the local experiment slug without relying on paths.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "run_date": "Records the operator-specified transition date as a fixed value.",
    "random_seed": "Deterministic metadata for checksum stability; no stochastic run occurs.",
    "spec_refs": "Anchors the artifact to REQ-REPORT-5837 and its scenarios.",
    "result_path": "Names the emitted deliverable path.",
    "field_principles": "Maps every top-level artifact field to its evidence boundary.",
    "status": (
        "A normalized terminal state distinguishes a complete transition from a bootstrap artifact."
    ),
    "preconditions_checked": (
        "Exact hashes and resource checks prevent archival against missing or ambiguous evidence."
    ),
    "milestone_transition": "Explicit source and destination milestones prevent numeric-prefix aliasing.",
    "canonical_identity_contract": (
        "Defines canonical evidence as milestone, task id, and declared deliverable."
    ),
    "declared_deliverable_matrix": (
        "Declared paths plus conductor outcomes are the authority for activated task identity."
    ),
    "same_number_alias_groups": "Records same-number files as aliases, never canonical evidence.",
    "adversarial_verifier_receipts": (
        "Fresh command and exit-code receipts preserve the live quality authority."
    ),
    "outcome_classification": (
        "Disjoint classes prevent flagged or provisional evidence from becoming a clean success."
    ),
    "flagged_evidence_preserved": (
        "True proves the transition did not launder Exp5828 or its downstream taint."
    ),
    "reserved_unactivated_task_ids": (
        "Proposal-only identities remain tombstoned and cannot silently collide."
    ),
    "reserved_unactivated_range": "Names the proposal-only `.519` identity interval.",
    "research_complete_append_count": "An exact append count prevents duplicate milestone history.",
    "duplicate_history_diagnostics": "Reports duplicate history without rewriting it.",
    "collision_scan": "Shows the Exp5837-Exp5848 namespace scan and collision sources.",
    "next_task_range": "A declared interval makes downstream allocation auditable.",
    "next_range_collision_count": "Only a bare zero authorizes Exp5837-Exp5848.",
    "docs_reconciled": "Specs, traceability, and ops summaries must match archived evidence classes.",
    "research_roadmap_unchanged": "Bare boolean must remain true because active roadmap mutation is forbidden.",
    "conductor_unchanged": "Bare boolean must remain true by operator instruction.",
    "duration_s": "Measured wall time exposes bootstrap-only execution.",
    "inference_substrate": (
        "`aggregation_from_upstream_artifacts` prevents archival from masquerading as inference."
    ),
    "field_provenance": "Per-field paths and hashes make the transition independently auditable.",
    "test_commands": "Recorded commands show which identity, verifier, and collision checks ran.",
    "test_exit_codes": "Exit codes prevent failed checks from being narrated as passing.",
    "reproducibility_checksum": "A content hash detects later ledger or allocation drift.",
    "honest_verdict": (
        "A `complete:` or `blocked:` prefix provides a mechanically terminal outcome."
    ),
}


@dataclass(frozen=True)
class ProcessResult:
    """Small testable substitute for subprocess.CompletedProcess."""

    returncode: int
    stdout: str
    stderr: str


def _read_yaml_with_meta(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "parsed": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        return {}, meta
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        meta["error"] = str(exc)
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = f"expected mapping, got {type(payload).__name__}"
        return {}, meta
    meta["parsed"] = True
    return payload, meta


def _roadmap_summary(path: Path) -> JsonDict:
    payload, meta = _read_yaml_with_meta(path)
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    task_ids = [
        str(row["id"])
        for row in tasks
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    ]
    deliverables = [
        str(row["deliverable"])
        for row in tasks
        if isinstance(row, Mapping) and isinstance(row.get("deliverable"), str)
    ]
    milestone = payload.get("milestone") if isinstance(payload.get("milestone"), str) else None
    return {**meta, "milestone": milestone, "task_ids": task_ids, "deliverables": deliverables}


def _research_complete_payload(root: Path) -> JsonDict:
    payload, _meta = _read_yaml_with_meta(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    return payload


def _research_complete_blocks(root: Path, milestone: str = MILESTONE_FROM) -> list[JsonDict]:
    milestones = _research_complete_payload(root).get("milestones")
    if not isinstance(milestones, list):
        return []
    return [
        block for block in milestones if isinstance(block, dict) and block.get("id") == milestone
    ]


def _task_signature(block: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    tasks = block.get("tasks")
    if not isinstance(tasks, list):
        return ()
    rows: list[tuple[str, str]] = []
    for row in tasks:
        if isinstance(row, Mapping) and isinstance(row.get("id"), str):
            deliverable = row.get("deliverable")
            rows.append((str(row["id"]), str(deliverable) if isinstance(deliverable, str) else ""))
    return tuple(rows)


def _completion_block_text() -> str:
    def q(value: str) -> str:
        return json.dumps(value, ensure_ascii=True)

    task_lines: list[str] = []
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        task_lines.extend(
            [
                f"  - id: {q(task_id)}",
                f"    title: {q(TASK_TITLES[task_id])}",
                f"    deliverable: {q(rel_path.as_posix())}",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(
        [
            f"- id: {q(MILESTONE_FROM)}",
            "  title: "
            + q("Certified Adaptive Memory, Internal Energy, and World-Feedback Induction"),
            f"  doc: {q(VNEXT_RELATIVE_PATH.as_posix())}",
            "  completed: '2026-07-23'",
            "  finding: See conductor log for per-experiment results.",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _append_research_complete_if_absent(root: Path) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    _payload, meta = _read_yaml_with_meta(path)
    if meta["present"] and not meta["parsed"]:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "research_complete_unparseable",
            "sha256_before_append": meta["sha256"],
            "sha256_after_append": meta["sha256"],
        }
    if _research_complete_blocks(root):
        return {
            "append_count": 0,
            "appended": False,
            "reason": "exact_milestone_block_present",
            "sha256_before_append": meta["sha256"],
            "sha256_after_append": meta["sha256"],
        }

    before = meta["sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        prefix = (
            "# Carnot Research - Completed Experiments\n"
            "# Tasks moved here from research-roadmap.yaml after successful completion.\n\n"
            "milestones:\n"
        )
        path.write_text(prefix + _completion_block_text(), encoding="utf-8")
    else:
        existing = path.read_text(encoding="utf-8")
        separator = "" if existing.endswith("\n") else "\n"
        path.write_text(existing + separator + _completion_block_text(), encoding="utf-8")
    return {
        "append_count": 1,
        "appended": True,
        "reason": "exact_milestone_block_absent",
        "sha256_before_append": before,
        "sha256_after_append": path_sha256(path),
    }


def _duplicate_task_conflicts(tasks: Sequence[Any]) -> list[str]:
    deliverables_by_task: dict[str, set[str]] = defaultdict(set)
    for row in tasks:
        if isinstance(row, Mapping) and isinstance(row.get("id"), str):
            deliverable = row.get("deliverable")
            deliverables_by_task[str(row["id"])].add(
                str(deliverable) if isinstance(deliverable, str) else ""
            )
    return sorted(task_id for task_id, values in deliverables_by_task.items() if len(values) > 1)


def _declared_deliverable_matrix(
    root: Path, append_count: int
) -> tuple[list[JsonDict], JsonDict, list[str]]:
    blocks = _research_complete_blocks(root)
    signatures = [_task_signature(block) for block in blocks]
    unique_signatures = set(signatures)
    stats: JsonDict = {
        "research_complete_milestone_from_block_count": len(blocks),
        "unique_declared_deliverable_block_count": len(unique_signatures),
        "declared_deliverables_unambiguous": len(unique_signatures) <= 1,
        "completion_block_source": "research_complete" if blocks else "unavailable",
        "research_complete_append_count": append_count,
    }
    failures: list[str] = []
    if len(unique_signatures) > 1:
        failures.append("ambiguous_research_complete_declared_task_blocks")

    selected_tasks = blocks[0].get("tasks") if blocks else []
    task_rows = selected_tasks if isinstance(selected_tasks, list) else []
    conflicts = _duplicate_task_conflicts(task_rows)
    if conflicts:
        failures.append(f"duplicate_task_id_conflicts={conflicts}")
    by_task: dict[str, JsonMap] = {
        str(row["id"]): row
        for row in task_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    declared_ids = tuple(
        str(row["id"])
        for row in task_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    )
    if declared_ids != EXPECTED_TASK_IDS:
        failures.append(f"declared_task_ids_mismatch={list(declared_ids)}")

    matrix: list[JsonDict] = []
    mismatches: list[str] = []
    for task_id in EXPECTED_TASK_IDS:
        row = by_task.get(task_id, {})
        declared = row.get("deliverable")
        expected = TASK_ARTIFACT_PATHS[task_id].as_posix()
        declared_path = declared if isinstance(declared, str) else ""
        if declared_path != expected:
            mismatches.append(f"{task_id}:{declared_path or '<missing>'}!={expected}")
        title = row.get("title") if isinstance(row.get("title"), str) else TASK_TITLES[task_id]
        result = row.get("result") if isinstance(row.get("result"), str) else ""
        matrix.append(
            {
                "identity": [MILESTONE_FROM, task_id, declared_path or expected],
                "milestone": MILESTONE_FROM,
                "task_id": task_id,
                "title": title,
                "declared_deliverable": declared_path or expected,
                "research_complete_result": result,
                "selection_policy": ARTIFACT_SELECTION_POLICY,
            }
        )
    if mismatches:
        failures.append(f"declared_deliverable_mismatch={mismatches}")
    return matrix, stats, failures


def _artifact_terminal_status(payload: JsonMap, metadata: JsonMap) -> str:
    if metadata.get("exists") is False:
        return "missing"
    if metadata.get("loadable") is False:
        return "malformed"
    status = payload.get("status")
    verdict = payload.get("honest_verdict")
    if status == "blocked" or (isinstance(verdict, str) and verdict.startswith("blocked:")):
        return "blocked"
    if status == "complete" or (isinstance(verdict, str) and verdict.startswith("complete:")):
        return "complete"
    if isinstance(verdict, str) and verdict.startswith("positive:"):
        return "complete"
    return str(status) if isinstance(status, str) and status else "unknown"


def _task_number(task_id: str) -> str | None:
    match = re.match(r"exp(\d+)", task_id)
    return match.group(1) if match else None


def _parse_conductor_log(path_or_root: Path) -> list[JsonDict]:
    path = path_or_root / CONDUCTOR_LOG_RELATIVE_PATH if path_or_root.is_dir() else path_or_root
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "timestamp": parts[0],
                "title": parts[1],
                "outcome": parts[2],
                "detail": parts[3],
                "line": line,
            }
        )
    return rows


def _conductor_outcomes(root: Path) -> tuple[dict[str, JsonDict], list[str]]:
    rows = _parse_conductor_log(root)
    outcomes: dict[str, JsonDict] = {}
    missing: list[str] = []
    for task_id, pattern in CONDUCTOR_TITLE_PATTERNS.items():
        matches = [row for row in rows if pattern in str(row["title"])]
        if not matches:
            missing.append(task_id)
        latest = matches[-1] if matches else {}
        outcomes[task_id] = {
            "outcome": latest.get("outcome", "UNKNOWN"),
            "detail": latest.get("detail", ""),
            "source": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "latest_evidence_line": latest.get("line", ""),
            "evidence_lines": [str(row["line"]) for row in matches],
            "attempt_count": len(matches),
            "failure_count": sum(1 for row in matches if row.get("outcome") in {"FAIL", "SKIP"}),
            "flagged_count": sum(1 for row in matches if row.get("outcome") == "FLAGGED"),
        }
    return outcomes, missing


def _corrigendum_kinds(payload: JsonMap, severity: str) -> list[str]:
    rows = payload.get("corrigendum_pending")
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("kind"))
        for row in rows
        if isinstance(row, Mapping) and str(row.get("severity", "")).lower() == severity
    ]


def _canonical_artifacts(
    root: Path,
    matrix: Sequence[JsonMap],
    conductor_outcomes: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for item in matrix:
        task_id = str(item["task_id"])
        rel_path = Path(str(item["declared_deliverable"]))
        payload, metadata = _read_json_any(root / rel_path)
        rows[task_id] = {
            "identity": [MILESTONE_FROM, task_id, rel_path.as_posix()],
            "path": rel_path.as_posix(),
            "present": bool(metadata.get("exists")),
            "loadable": bool(metadata.get("loadable")),
            "sha256": metadata.get("sha256"),
            "status": _artifact_terminal_status(payload, metadata),
            "honest_verdict": payload.get("honest_verdict")
            if isinstance(payload.get("honest_verdict"), str)
            else "",
            "accepted_finding_count": payload.get("accepted_finding_count"),
            "artifact_flagged_adversarial": payload.get("flagged_adversarial") is True,
            "artifact_critical_findings": _corrigendum_kinds(payload, "critical"),
            "artifact_warn_findings": _corrigendum_kinds(payload, "warn"),
            "upstream_artifact_hashes": payload.get("upstream_artifact_hashes")
            if isinstance(payload.get("upstream_artifact_hashes"), Mapping)
            else {},
            "duration_s": payload.get("duration_s"),
            "inference_substrate": payload.get("inference_substrate"),
            "conductor_outcome": conductor_outcomes.get(task_id, {}).get("outcome", "UNKNOWN"),
            "conductor_detail": conductor_outcomes.get(task_id, {}).get("detail", ""),
            "selected_by": ARTIFACT_SELECTION_POLICY,
            "error": metadata.get("error"),
        }
    return rows


def _same_number_alias_groups(
    root: Path, canonical_artifacts: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:
    groups: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS:
        number = _task_number(task_id)
        if number is None or task_id not in canonical_artifacts:
            continue
        canonical = canonical_artifacts[task_id]
        canonical_path = Path(str(canonical["path"]))
        candidates = sorted((root / "results").glob(f"experiment_{number}*.json"))
        aliases: list[JsonDict] = []
        for candidate in candidates:
            rel_path = candidate.relative_to(root)
            if rel_path == canonical_path:
                continue
            payload, metadata = _read_json_any(candidate)
            aliases.append(
                {
                    "path": rel_path.as_posix(),
                    "present": bool(metadata.get("exists")),
                    "loadable": bool(metadata.get("loadable")),
                    "sha256": metadata.get("sha256"),
                    "status": _artifact_terminal_status(payload, metadata),
                    "role": "same_number_alias",
                }
            )
        groups[number] = {
            "experiment_number": number,
            "canonical": {
                "task_id": task_id,
                "path": canonical["path"],
                "present": canonical["present"],
                "sha256": canonical["sha256"],
                "status": canonical["status"],
                "role": "canonical_declared_deliverable",
            },
            "aliases": aliases,
            "selection_policy": ARTIFACT_SELECTION_POLICY,
        }
    return groups


def _resource_receipts(root: Path, result_path: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    mem_available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])
                break
    writable_candidates = (
        root,
        root / "results",
        result_path.parent if result_path.is_absolute() else root / result_path.parent,
        Path("/tmp"),
    )
    return {
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "mem_available_kib": mem_available,
        "writable_paths": {
            path.as_posix(): {"present": path.exists(), "writable": os.access(path, os.W_OK)}
            for path in writable_candidates
        },
    }


def _duplicate_history_diagnostics(root: Path) -> JsonDict:
    milestones = _research_complete_payload(root).get("milestones")
    blocks = milestones if isinstance(milestones, list) else []
    by_milestone: dict[str, list[JsonMap]] = defaultdict(list)
    for block in blocks:
        if isinstance(block, Mapping) and isinstance(block.get("id"), str):
            by_milestone[str(block["id"])].append(block)
    duplicates: list[JsonDict] = []
    for milestone, milestone_blocks in sorted(by_milestone.items()):
        if len(milestone_blocks) <= 1:
            continue
        duplicates.append(
            {
                "milestone": milestone,
                "block_count": len(milestone_blocks),
                "unique_block_signature_count": len(
                    {_task_signature(block) for block in milestone_blocks}
                ),
                "mutation": "preserved_read_only",
            }
        )
    from_blocks = by_milestone.get(MILESTONE_FROM, [])
    return {
        "history_mutation_policy": "append_once_if_absent_no_dedup_sort_or_rewrite",
        "milestone_from_block_count": len(from_blocks),
        "milestone_from_unique_signature_count": len(
            {_task_signature(block) for block in from_blocks}
        ),
        "duplicate_milestone_blocks": duplicates,
        "duplicate_history_block_count": sum(row["block_count"] - 1 for row in duplicates),
    }


def _next_range_tokens() -> tuple[str, ...]:
    tokens: list[str] = []
    for task_id, rel_path in NEXT_TASK_ARTIFACT_PATHS.items():
        number = _task_number(task_id)
        if number:
            tokens.append(f"exp{number}")
            tokens.append(f"experiment_{number}")
        tokens.append(task_id)
        tokens.append(rel_path.as_posix())
    tokens.extend(path.as_posix() for path in NEXT_EXTRA_DELIVERABLE_PATHS)
    return tuple(dict.fromkeys(tokens))


def _text_has_next_range(text: str) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in _next_range_tokens())


def _collision_scan(root: Path, result_path: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed_references: list[JsonDict] = []
    for rel_path in ALLOWED_ALLOCATION_TEXT_PATHS:
        path = root / rel_path
        if path.exists() and _text_has_next_range(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            allowed_references.append(
                {"path": rel_path.as_posix(), "kind": "allowed_allocation_reference"}
            )
    for rel_path in COLLISION_TEXT_PATHS:
        path = root / rel_path
        if path.exists() and _text_has_next_range(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            collisions.append(
                {"path": rel_path.as_posix(), "kind": "preexisting_content_reference"}
            )

    rel_result_path = result_path.relative_to(root) if result_path.is_absolute() else result_path
    results_dir = root / "results"
    for number in NEXT_RANGE_NUMBERS:
        for candidate in sorted(results_dir.glob(f"experiment_{number}*")):
            if not candidate.is_file():
                continue
            rel_path = candidate.relative_to(root)
            if rel_path == rel_result_path:
                continue
            collisions.append({"path": rel_path.as_posix(), "kind": "preexisting_result_file"})

    for candidate in sorted(results_dir.glob("experiment_*transition*.json")):
        rel_path = candidate.relative_to(root)
        if rel_path == rel_result_path:
            continue
        payload, metadata = _read_json_any(candidate)
        if not metadata.get("loadable"):
            continue
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        if payload.get("next_task_range") == NEXT_TASK_RANGE or _text_has_next_range(text):
            collisions.append({"path": rel_path.as_posix(), "kind": "prior_transition_allocation"})

    collisions = sorted(
        {json.dumps(row, sort_keys=True): row for row in collisions}.values(),
        key=lambda row: (str(row["path"]), str(row["kind"])),
    )
    return {
        "next_task_ids": list(NEXT_TASK_IDS),
        "next_declared_deliverables": [
            NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix() for task_id in NEXT_TASK_IDS
        ],
        "next_extra_deliverables": [path.as_posix() for path in NEXT_EXTRA_DELIVERABLE_PATHS],
        "allowed_allocation_references": allowed_references,
        "preexisting_collisions": collisions,
        "preexisting_collision_count": len(collisions),
        "collision_free": not collisions,
    }


def _git_modified(root: Path, rel_path: Path) -> bool:  # pragma: no cover - live repo check
    result = subprocess.run(
        ["git", "status", "--short", "--", rel_path.as_posix()],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


def _protected_files(
    root: Path,
    modification_overrides: Mapping[Path, bool] | None,
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in PROTECTED_FILE_PATHS:
        if modification_overrides is not None and rel_path in modification_overrides:
            modified = bool(modification_overrides[rel_path])
            source = "test_override"
        else:  # pragma: no cover - live artifact generation uses git status
            modified = _git_modified(root, rel_path)
            source = "git_status"
        rows[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "modified_by_exp5837": modified,
            "check_source": source,
        }
    return rows


def _input_hashes(root: Path, canonical_artifacts: Mapping[str, JsonMap]) -> JsonDict:
    source_paths = (
        RESEARCH_COMPLETE_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        VNEXT_RELATIVE_PATH,
        EVIDENCE_INDEX_RELATIVE_PATH,
        DOC_RECONCILE_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    )
    return {
        "source_files": {
            rel_path.as_posix(): {
                "present": (root / rel_path).exists(),
                "sha256": path_sha256(root / rel_path),
            }
            for rel_path in source_paths
        },
        "declared_deliverables": {
            task_id: {"path": row["path"], "sha256": row["sha256"], "present": row["present"]}
            for task_id, row in canonical_artifacts.items()
        },
    }


def _default_verifier_command(path: Path) -> str:
    return f".venv/bin/python scripts/adversarial_verify.py --json {path.as_posix()}"


def _task_id_for_artifact_path(artifact_path: Path) -> str:
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        if rel_path == artifact_path:
            return task_id
    return ""


def _flag_rows_from_stdout(stdout_json: Any) -> list[JsonDict]:
    if not isinstance(stdout_json, Mapping):
        return []
    reports = stdout_json.get("reports")
    if not isinstance(reports, list) or not reports:
        return []
    first = reports[0]
    if not isinstance(first, Mapping):
        return []
    flags = first.get("flags")
    return (
        [dict(row) for row in flags if isinstance(row, Mapping)] if isinstance(flags, list) else []
    )


def _receipt_hash_payload(row: Mapping[str, Any]) -> bytes:
    receipt = {
        "command": row.get("command"),
        "exit_code": row.get("exit_code"),
        "stdout_json": row.get("stdout_json"),
        "stderr": row.get("stderr", ""),
        "stdout_parse_error": row.get("stdout_parse_error"),
    }
    return json.dumps(receipt, sort_keys=True, ensure_ascii=True).encode("utf-8")


def normalize_adversarial_verifier_receipts(
    receipts: Sequence[Mapping[str, Any]] | None,
) -> list[JsonDict]:
    by_task = {
        str(row.get("task_id")): dict(row)
        for row in receipts or []
        if isinstance(row, Mapping) and row.get("task_id")
    }
    normalized: list[JsonDict] = []
    for task_id in VERIFIER_TASK_IDS:
        rel_path = TASK_ARTIFACT_PATHS.get(task_id, Path(""))
        row = by_task.get(task_id)
        if row is None:
            normalized.append(
                {
                    "task_id": task_id,
                    "artifact_path": rel_path.as_posix(),
                    "command": _default_verifier_command(rel_path),
                    "exit_code": None,
                    "receipt_hash": None,
                    "critical_findings": [],
                    "warn_findings": [],
                    "headline_eligible": False,
                    "headline_ineligible_reason": "missing_live_receipt",
                    "present": False,
                }
            )
            continue
        stdout_json = row.get("stdout_json")
        flags = _flag_rows_from_stdout(stdout_json)
        if not flags:
            flags = [dict(flag) for flag in row.get("flags", []) if isinstance(flag, Mapping)]
        critical = [flag for flag in flags if str(flag.get("severity", "")).lower() == "critical"]
        warn = [flag for flag in flags if str(flag.get("severity", "")).lower() == "warn"]
        exit_code = row.get("exit_code")
        headline_reason = ""
        headline_eligible = True
        if critical:
            headline_eligible = False
            headline_reason = "fresh_critical_findings"
        elif exit_code != 0:
            headline_eligible = False
            headline_reason = "verifier_exit_nonzero"
        elif task_id == "exp5829-transfer-selective-replay-audit":
            headline_eligible = False
            headline_reason = "upstream_flagged_exp5828"
        if headline_eligible:
            headline_reason = "clean_live_receipt"
        normalized.append(
            {
                "task_id": task_id,
                "artifact_path": str(row.get("artifact_path") or rel_path.as_posix()),
                "command": str(row.get("command") or _default_verifier_command(rel_path)),
                "exit_code": exit_code,
                "receipt_hash": row.get("receipt_hash") or sha256_bytes(_receipt_hash_payload(row)),
                "critical_findings": critical,
                "warn_findings": warn,
                "headline_eligible": headline_eligible,
                "headline_ineligible_reason": headline_reason,
                "present": True,
                "stdout_parse_error": row.get("stdout_parse_error"),
            }
        )
    return normalized


def run_adversarial_verifier(
    root: Path,
    artifact_path: Path,
    *,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> JsonDict:
    command = [
        ".venv/bin/python",
        "scripts/adversarial_verify.py",
        "--json",
        artifact_path.as_posix(),
    ]
    completed = subprocess_run(
        command,
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    stdout = str(completed.stdout)
    stderr = str(completed.stderr)
    stdout_json: JsonDict | None = None
    parse_error = None
    try:
        parsed = json.loads(stdout)
        stdout_json = parsed if isinstance(parsed, dict) else {"_non_mapping_stdout": parsed}
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
    row: JsonDict = {
        "task_id": _task_id_for_artifact_path(artifact_path),
        "artifact_path": artifact_path.as_posix(),
        "command": " ".join(command),
        "exit_code": int(completed.returncode),
        "stdout_json": stdout_json,
        "stderr": stderr,
        "stdout_parse_error": parse_error,
    }
    row["receipt_hash"] = sha256_bytes(_receipt_hash_payload(row))
    return row


def run_adversarial_verifiers(
    root: Path = REPO_ROOT,
) -> list[JsonDict]:  # pragma: no cover - CLI path
    return [
        run_adversarial_verifier(root, TASK_ARTIFACT_PATHS[task_id])
        for task_id in VERIFIER_TASK_IDS
    ]


def _receipt_by_task(receipts: Sequence[JsonMap]) -> dict[str, JsonMap]:
    return {str(row["task_id"]): row for row in receipts}


def _receipt_kinds(receipt: JsonMap, key: str) -> list[str]:
    rows = receipt.get(key)
    if not isinstance(rows, list):
        return []
    return [str(row.get("kind")) for row in rows if isinstance(row, Mapping)]


def _exp5828_stamp_preserved(
    canonical_artifacts: Mapping[str, JsonMap], receipts: Mapping[str, JsonMap]
) -> bool:
    artifact = canonical_artifacts.get("exp5828-future-validated-structural-memory", {})
    receipt = receipts.get("exp5828-future-validated-structural-memory", {})
    return (
        artifact.get("artifact_flagged_adversarial") is True
        and "DURATION_TOO_SHORT" in artifact.get("artifact_critical_findings", [])
        and "METHODOLOGY_MISSING" in artifact.get("artifact_warn_findings", [])
        and "DURATION_TOO_SHORT" in _receipt_kinds(receipt, "critical_findings")
        and "METHODOLOGY_MISSING" in _receipt_kinds(receipt, "warn_findings")
    )


def _exp5829_upstream_tainted(canonical_artifacts: Mapping[str, JsonMap]) -> bool:
    exp5828 = canonical_artifacts.get("exp5828-future-validated-structural-memory", {})
    exp5829 = canonical_artifacts.get("exp5829-transfer-selective-replay-audit", {})
    upstream = exp5829.get("upstream_artifact_hashes")
    return (
        isinstance(upstream, Mapping)
        and exp5828.get("artifact_flagged_adversarial") is True
        and upstream.get("exp5828_lifecycle_artifact") == exp5828.get("sha256")
    )


def _classify_outcomes(
    canonical_artifacts: Mapping[str, JsonMap],
    conductor_rows: Mapping[str, JsonMap],
    receipts: Mapping[str, JsonMap],
) -> JsonDict:
    clean_positive: list[str] = []
    clean_null: list[str] = []
    clean_negative: list[str] = []
    blocked_skipped: list[str] = []
    flagged: list[str] = []
    flagged_upstream: list[str] = []
    missing: list[str] = []
    by_task: dict[str, str] = {}
    exp5828_preserved = _exp5828_stamp_preserved(canonical_artifacts, receipts)
    exp5829_tainted = _exp5829_upstream_tainted(canonical_artifacts)
    for task_id in EXPECTED_TASK_IDS:
        artifact = canonical_artifacts.get(task_id, {})
        status = artifact.get("status")
        outcome = conductor_rows.get(task_id, {}).get("outcome")
        if status in {"missing", "malformed"}:
            missing.append(task_id)
            by_task[task_id] = "missing"
        elif task_id == "exp5828-future-validated-structural-memory" and exp5828_preserved:
            flagged.append(task_id)
            by_task[task_id] = "flagged"
        elif task_id == "exp5829-transfer-selective-replay-audit":
            if exp5829_tainted:
                flagged_upstream.append(task_id)
                by_task[task_id] = "flagged-upstream/provisional"
            else:
                missing.append(task_id)
                by_task[task_id] = "missing"
        elif outcome in {"GATE_BLOCK", "SKIP"} or status == "blocked":
            blocked_skipped.append(task_id)
            by_task[task_id] = "blocked/skipped"
        elif (
            task_id == "exp5824-v519-source-delta-ingestion"
            and artifact.get("accepted_finding_count") == 0
        ):
            clean_null.append(task_id)
            by_task[task_id] = "clean-null"
        elif (
            task_id
            in {
                "exp5823-transition-v519",
                "exp5825-certified-adaptive-memory-contract",
                "exp5826-out-of-template-constraint-stream",
                "exp5827-minimal-core-structural-acquisition-ab",
            }
            and status == "complete"
        ):
            clean_positive.append(task_id)
            by_task[task_id] = "clean-positive"
        else:
            missing.append(task_id)
            by_task[task_id] = "missing"
    return {
        "clean_positive_task_ids": clean_positive,
        "clean_null_task_ids": clean_null,
        "clean_negative_task_ids": clean_negative,
        "blocked_skipped_task_ids": blocked_skipped,
        "flagged_task_ids": flagged,
        "flagged_upstream_provisional_task_ids": flagged_upstream,
        "missing_task_ids": missing,
        "proposal_only_task_ids": list(RESERVED_UNACTIVATED_TASK_IDS),
        "clean_success_task_ids": list(clean_positive),
        "headline_eligible_task_ids": list(clean_positive),
        "classification_by_task_id": by_task,
        "classification_policy": (
            "clean-positive, clean-null, clean-negative, blocked/skipped, flagged, "
            "flagged-upstream/provisional, missing, and proposal-only classes are disjoint"
        ),
    }


def _adversarial_receipt_failures(receipts: Mapping[str, JsonMap]) -> list[str]:
    failures: list[str] = []
    missing = [
        task_id for task_id in VERIFIER_TASK_IDS if not receipts.get(task_id, {}).get("present")
    ]
    if missing:
        failures.append(f"adversarial_receipts_missing={missing}")
    for task_id in VERIFIER_TASK_IDS:
        receipt = receipts.get(task_id, {})
        critical = _receipt_kinds(receipt, "critical_findings")
        warn = _receipt_kinds(receipt, "warn_findings")
        exit_code = receipt.get("exit_code")
        if task_id == "exp5828-future-validated-structural-memory":
            if "DURATION_TOO_SHORT" not in critical or "METHODOLOGY_MISSING" not in warn:
                failures.append("exp5828_live_stamp_not_preserved")
            if exit_code == 0:
                failures.append("exp5828_verifier_exit_unexpectedly_clean")
        elif critical:
            failures.append(f"{task_id}_unexpected_critical_findings={critical}")
        elif exit_code != 0:
            failures.append(f"{task_id}_verifier_exit_code={exit_code}")
    return failures


def _tests_failed(tests_run: Sequence[JsonMap]) -> list[str]:
    return [
        str(row.get("command"))
        for row in tests_run
        if row.get("exit_code") != 0 and row.get("blocking", True) is not False
    ]


def _field_principles_for(artifact: Mapping[str, Any]) -> dict[str, str]:
    missing = [field for field in artifact if field not in FIELD_PRINCIPLES]
    if missing:
        raise KeyError(f"missing field principles: {missing}")
    return {field: FIELD_PRINCIPLES[field] for field in artifact}


def _hashes_for_sources(root: Path, sources: Sequence[str]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for source in sources:
        path = root / source
        hashes[source] = path_sha256(path) if path.is_file() else None
    return hashes


def _field_provenance_for(artifact: Mapping[str, Any], root: Path) -> dict[str, JsonDict]:
    default_sources = ["python/carnot/experiment_5837_transition_v520.py"]
    task_artifacts = [path.as_posix() for path in TASK_ARTIFACT_PATHS.values()]
    source_files = [
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ROADMAP_RELATIVE_PATH.as_posix(),
        ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
        VNEXT_RELATIVE_PATH.as_posix(),
        EVIDENCE_INDEX_RELATIVE_PATH.as_posix(),
        DOC_RECONCILE_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
    ]
    sources_by_field: dict[str, list[str]] = {
        "status": default_sources,
        "preconditions_checked": [*source_files, *task_artifacts],
        "milestone_transition": [
            ROADMAP_RELATIVE_PATH.as_posix(),
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        ],
        "declared_deliverable_matrix": [
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            *task_artifacts,
        ],
        "adversarial_verifier_receipts": [
            "scripts/adversarial_verify.py",
            *[TASK_ARTIFACT_PATHS[task_id].as_posix() for task_id in VERIFIER_TASK_IDS],
        ],
        "outcome_classification": [
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            *task_artifacts,
        ],
        "flagged_evidence_preserved": [
            TASK_ARTIFACT_PATHS["exp5828-future-validated-structural-memory"].as_posix(),
            TASK_ARTIFACT_PATHS["exp5829-transfer-selective-replay-audit"].as_posix(),
            "scripts/adversarial_verify.py",
        ],
        "reserved_unactivated_task_ids": [
            TASK_ARTIFACT_PATHS["exp5823-transition-v519"].as_posix(),
            VNEXT_RELATIVE_PATH.as_posix(),
        ],
        "research_complete_append_count": [RESEARCH_COMPLETE_RELATIVE_PATH.as_posix()],
        "next_task_range": [ROADMAP_RELATIVE_PATH.as_posix(), VNEXT_RELATIVE_PATH.as_posix()],
        "next_range_collision_count": [
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            ROADMAP_RELATIVE_PATH.as_posix(),
            ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            VNEXT_RELATIVE_PATH.as_posix(),
            "results/",
        ],
        "docs_reconciled": [
            SPEC_RELATIVE_PATH.as_posix(),
            "tests/python/test_experiment_5837_transition_v520.py",
            "python/carnot/experiment_5837_transition_v520.py",
        ],
        "duration_s": default_sources,
        "inference_substrate": default_sources,
        "field_provenance": [default_sources[0], SPEC_RELATIVE_PATH.as_posix()],
        "test_commands": ["operator_test_receipts", "scripts/adversarial_verify.py"],
        "test_exit_codes": ["operator_test_receipts", "scripts/adversarial_verify.py"],
        "reproducibility_checksum": ["artifact_payload_excluding_checksum"],
        "honest_verdict": default_sources,
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": sources_by_field.get(field, default_sources),
            "sha256_by_source": _hashes_for_sources(
                root, sources_by_field.get(field, default_sources)
            ),
            "derivation": "direct_read_or_deterministic_reconciliation",
        }
        for field in artifact
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
    duration_s: float | None = None,
    result_path: Path | None = None,
) -> JsonDict:
    started = time.perf_counter()
    root = root.resolve()
    result_path = result_path or root / RESULT_RELATIVE_PATH
    append_receipt = _append_research_complete_if_absent(root)
    roadmap_active = _roadmap_summary(root / ROADMAP_RELATIVE_PATH)
    roadmap_next = _roadmap_summary(root / ROADMAP_NEXT_RELATIVE_PATH)
    declared_matrix, complete_stats, matrix_failures = _declared_deliverable_matrix(
        root, int(append_receipt["append_count"])
    )
    conductor_rows, missing_conductor_task_ids = _conductor_outcomes(root)
    canonical_artifacts = _canonical_artifacts(root, declared_matrix, conductor_rows)
    normalized_receipts = normalize_adversarial_verifier_receipts(adversarial_receipts)
    receipts_by_task = _receipt_by_task(normalized_receipts)
    alias_groups = _same_number_alias_groups(root, canonical_artifacts)
    outcome_classification = _classify_outcomes(
        canonical_artifacts, conductor_rows, receipts_by_task
    )
    collision_scan = _collision_scan(root, result_path)
    duplicate_history = _duplicate_history_diagnostics(root)
    protected_files = _protected_files(root, modification_overrides)
    input_hashes = _input_hashes(root, canonical_artifacts)
    run_rows = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]

    exp5829_tainted = _exp5829_upstream_tainted(canonical_artifacts)
    for row in declared_matrix:
        task_id = str(row["task_id"])
        artifact = canonical_artifacts[task_id]
        conductor = conductor_rows[task_id]
        receipt = receipts_by_task.get(task_id, {})
        row["canonical_artifact_path"] = artifact["path"]
        row["canonical_artifact_present"] = artifact["present"]
        row["canonical_artifact_loadable"] = artifact["loadable"]
        row["canonical_artifact_sha256"] = artifact["sha256"]
        row["canonical_artifact_status"] = artifact["status"]
        row["canonical_artifact_honest_verdict"] = artifact["honest_verdict"]
        row["artifact_flagged_adversarial"] = artifact["artifact_flagged_adversarial"]
        row["artifact_critical_findings"] = artifact["artifact_critical_findings"]
        row["artifact_warn_findings"] = artifact["artifact_warn_findings"]
        row["conductor_outcome"] = conductor["outcome"]
        row["conductor_latest_evidence_line"] = conductor["latest_evidence_line"]
        row["outcome_class"] = outcome_classification["classification_by_task_id"].get(task_id)
        if task_id in VERIFIER_TASK_IDS:
            row["adversarial_exit_code"] = receipt.get("exit_code")
            row["adversarial_receipt_hash"] = receipt.get("receipt_hash")
            row["adversarial_critical_findings"] = _receipt_kinds(receipt, "critical_findings")
            row["adversarial_warn_findings"] = _receipt_kinds(receipt, "warn_findings")
            row["headline_eligible"] = receipt.get("headline_eligible")
        if task_id == "exp5829-transfer-selective-replay-audit":
            row["upstream_taint_source_task_id"] = (
                "exp5828-future-validated-structural-memory" if exp5829_tainted else None
            )

    research_roadmap_unchanged = not protected_files[ROADMAP_RELATIVE_PATH.as_posix()][
        "modified_by_exp5837"
    ]
    conductor_unchanged = not protected_files[CONDUCTOR_RELATIVE_PATH.as_posix()][
        "modified_by_exp5837"
    ]
    missing_or_malformed = [
        task_id
        for task_id, row in canonical_artifacts.items()
        if row["status"] in {"missing", "malformed"}
    ]
    exp5828_artifact_preserved = (
        canonical_artifacts["exp5828-future-validated-structural-memory"].get(
            "artifact_flagged_adversarial"
        )
        is True
        and "DURATION_TOO_SHORT"
        in canonical_artifacts["exp5828-future-validated-structural-memory"].get(
            "artifact_critical_findings", []
        )
        and "METHODOLOGY_MISSING"
        in canonical_artifacts["exp5828-future-validated-structural-memory"].get(
            "artifact_warn_findings", []
        )
    )
    flagged_evidence_preserved = (
        exp5828_artifact_preserved
        and _exp5828_stamp_preserved(canonical_artifacts, receipts_by_task)
        and exp5829_tainted
        and not set(outcome_classification["flagged_task_ids"]).intersection(
            outcome_classification["clean_success_task_ids"]
        )
        and not set(outcome_classification["flagged_upstream_provisional_task_ids"]).intersection(
            outcome_classification["clean_success_task_ids"]
        )
    )
    failed_tests = _tests_failed(run_rows)

    failed_preconditions = list(matrix_failures)
    if append_receipt["reason"] == "research_complete_unparseable":
        failed_preconditions.append("research_complete_unparseable")
    if not roadmap_active["parsed"]:
        failed_preconditions.append("active_roadmap_unparseable")
    if roadmap_active["milestone"] != MILESTONE_TO:
        failed_preconditions.append(f"active_roadmap_milestone={roadmap_active['milestone']!r}")
    if not set(roadmap_active["task_ids"]).issubset(set(NEXT_TASK_IDS)):
        failed_preconditions.append(f"active_roadmap_task_ids={roadmap_active['task_ids']}")
    if roadmap_next["present"] and not roadmap_next["parsed"]:
        failed_preconditions.append("next_roadmap_unparseable")
    if missing_or_malformed:
        failed_preconditions.append(
            f"missing_or_malformed_declared_deliverables={missing_or_malformed}"
        )
    if missing_conductor_task_ids:
        failed_preconditions.append(f"missing_conductor_outcomes={missing_conductor_task_ids}")
    if not exp5828_artifact_preserved:
        failed_preconditions.append("exp5828_artifact_stamp_not_preserved")
    if not exp5829_tainted:
        failed_preconditions.append("exp5829_upstream_taint_not_preserved")
    failed_preconditions.extend(_adversarial_receipt_failures(receipts_by_task))
    if collision_scan["preexisting_collision_count"]:
        failed_preconditions.append(
            f"next_range_collision_count={collision_scan['preexisting_collision_count']}"
        )
    if not research_roadmap_unchanged:
        failed_preconditions.append("research_roadmap_modified")
    if not conductor_unchanged:
        failed_preconditions.append("research_conductor_modified")
    if failed_tests:
        failed_preconditions.append(f"test_failures={failed_tests}")

    status = "blocked" if failed_preconditions else "complete"
    measured_duration = duration_s
    if measured_duration is None:
        measured_duration = round(max(time.perf_counter() - started, 0.000001), 6)
    verifier_commands = [str(row["command"]) for row in normalized_receipts]
    test_commands = [*verifier_commands, *[str(row.get("command")) for row in run_rows]]
    test_exit_codes = {
        **{str(row["command"]): row.get("exit_code") for row in normalized_receipts},
        **{str(row.get("command")): row.get("exit_code") for row in run_rows},
    }

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": {},
        "status": status,
        "preconditions_checked": {
            "roadmaps": {"active": roadmap_active, "next": roadmap_next},
            "input_hashes": input_hashes,
            "declared_deliverable_count": len(declared_matrix),
            "canonical_artifact_count": len(canonical_artifacts),
            "canonical_hash_count": sum(1 for row in canonical_artifacts.values() if row["sha256"]),
            "same_number_alias_group_count": len(alias_groups),
            "adversarial_verifier_receipt_count": sum(
                1 for row in normalized_receipts if row.get("present")
            ),
            "next_range_collision_count": collision_scan["preexisting_collision_count"],
            "resource_receipts": _resource_receipts(root, result_path),
            "research_complete_append_receipt": append_receipt,
            "research_roadmap_unchanged": research_roadmap_unchanged,
            "conductor_unchanged": conductor_unchanged,
            **complete_stats,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_transition": {
            "source_milestone": MILESTONE_FROM,
            "destination_milestone": MILESTONE_TO,
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "canonical_identity_contract": {
            "identity_tuple": ["milestone", "task_id", "declared_deliverable"],
            "canonical_path_rule": "exact declared_deliverable only",
            "numeric_prefix_matches": "aliases_only",
            "proposal_only_tasks": "reserved_unactivated_not_execution_evidence",
            "selection_policy": ARTIFACT_SELECTION_POLICY,
        },
        "declared_deliverable_matrix": declared_matrix,
        "same_number_alias_groups": alias_groups,
        "adversarial_verifier_receipts": normalized_receipts,
        "outcome_classification": outcome_classification,
        "flagged_evidence_preserved": flagged_evidence_preserved,
        "reserved_unactivated_task_ids": list(RESERVED_UNACTIVATED_TASK_IDS),
        "reserved_unactivated_range": RESERVED_TASK_RANGE,
        "research_complete_append_count": int(complete_stats["research_complete_append_count"]),
        "duplicate_history_diagnostics": duplicate_history,
        "collision_scan": collision_scan,
        "next_task_range": NEXT_TASK_RANGE,
        "next_range_collision_count": collision_scan["preexisting_collision_count"],
        "docs_reconciled": {
            "mode": "transition_owned_spec_tests_module_result_only",
            "operator_owned_docs_deferred": True,
            "deferred_files": [
                "_bmad/traceability.md",
                "ops/status.md",
                "ops/changelog.md",
            ],
            "transition_owned_files": [
                SPEC_RELATIVE_PATH.as_posix(),
                "python/carnot/experiment_5837_transition_v520.py",
                "tests/python/test_experiment_5837_transition_v520.py",
                RESULT_RELATIVE_PATH.as_posix(),
            ],
            "deferred_reason": "operator stop-when-done rule delegates ops/status/traceability updates",
        },
        "research_roadmap_unchanged": research_roadmap_unchanged,
        "conductor_unchanged": conductor_unchanged,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": {},
        "test_commands": test_commands,
        "test_exit_codes": test_exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "blocked: exp5837 transition preconditions failed: " + "; ".join(failed_preconditions)
            if failed_preconditions
            else (
                "complete: archived terminal .519 evidence by exact declared deliverables "
                "into .520; Exp5828 flagged stamp preserved; Exp5829 upstream taint "
                "preserved as provisional; Exp5830-Exp5836 tombstoned; "
                "next_range_collision_count=0; research_complete_append_count="
                f"{complete_stats['research_complete_append_count']}"
            )
        ),
    }
    artifact["field_principles"] = _field_principles_for(artifact)
    artifact["field_provenance"] = _field_provenance_for(artifact, root)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    adversarial_receipts: Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    output_path = output_path or root / RESULT_RELATIVE_PATH
    artifact = build_report(
        root,
        adversarial_receipts=adversarial_receipts,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
        duration_s=duration_s,
        result_path=output_path,
    )
    write_json(output_path, artifact)
    return artifact


def _load_json_list(path: Path | None) -> list[JsonDict]:  # pragma: no cover - CLI convenience
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    parser.add_argument("--adversarial-receipts-json", type=Path, default=None)
    parser.add_argument("--run-adversarial", action="store_true")
    args = parser.parse_args(argv)
    receipts = (
        run_adversarial_verifiers(args.root)
        if args.run_adversarial
        else _load_json_list(args.adversarial_receipts_json)
    )
    emit_report(
        args.root,
        output_path=args.output,
        adversarial_receipts=receipts,
        tests_run=_load_json_list(args.tests_run_json),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
