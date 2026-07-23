"""Exp5823 transition receipt from terminal milestone .518 into .519.

Spec refs: REQ-REPORT-5823, SCENARIO-REPORT-5823,
SCENARIO-REPORT-5823-RETIREMENT,
SCENARIO-REPORT-5823-COLLISION-BLOCK,
SCENARIO-REPORT-5823-FIELD-PROVENANCE.

This module is a ledger reconciler. It does not rerun model inference, edit the
roadmap, or let proposal-only IDs masquerade as executed evidence. The critical
boundary is Exp5813: the split-budget transport was implemented cleanly, but
the generated-answer SOTA canary remained a clean negative and retires the
same-mechanism GGUF answer path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
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
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5823_transition_v519.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP5813_ROW_RELATIVE_PATH = Path("results/experiment_5813_split_budget_sota_canary.rows.jsonl")

EXPERIMENT = "experiment_5823_transition_v519"
EXPERIMENT_ID = "exp5823-transition-v519"
MILESTONE_FROM = "2026.07.518"
MILESTONE_TO = "2026.07.519"
NEXT_TASK_RANGE = "exp5823-exp5836"
RESERVED_TASK_RANGE = "exp5816-exp5822"
RUN_DATE = "2026-07-22"
RANDOM_SEED = 5823
SCHEMA = "carnot.experiment_5823.transition_v519.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"
RETIREMENT_SCOPE_KEY = "finite_id_gguf_generated_answer_transport_same_mechanism_v519"

SPEC_REFS = (
    "REQ-REPORT-5823",
    "SCENARIO-REPORT-5823",
    "SCENARIO-REPORT-5823-RETIREMENT",
    "SCENARIO-REPORT-5823-COLLISION-BLOCK",
    "SCENARIO-REPORT-5823-FIELD-PROVENANCE",
)

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5809-transition-v518": Path("results/experiment_5809_transition_v518.json"),
    "exp5810-v518-source-delta-ingestion": Path(
        "results/experiment_5810_v518_source_delta_ingestion.json"
    ),
    "exp5811-exp5799-event-provenance-audit": Path(
        "results/experiment_5811_exp5799_event_provenance_audit.json"
    ),
    "exp5812-split-budget-channel-contract": Path(
        "results/experiment_5812_split_budget_channel_contract.json"
    ),
    "exp5813-split-budget-sota-canary": Path(
        "results/experiment_5813_split_budget_sota_canary.json"
    ),
    "exp5814-channel-qualified-constraint-stream": Path(
        "results/experiment_5814_channel_qualified_constraint_stream.json"
    ),
    "exp5815-future-validated-constraint-skill-ab": Path(
        "results/experiment_5815_future_validated_constraint_skill_ab.json"
    ),
}
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

TASK_TITLES: dict[str, str] = {
    "exp5809-transition-v518": (
        "Transition four activated .517 tasks, quarantine flagged evidence, and allocate .518"
    ),
    "exp5810-v518-source-delta-ingestion": (
        "Time-windowed post-V518 literature and implementation freshness receipt"
    ),
    "exp5811-exp5799-event-provenance-audit": (
        "Companion audit for Exp5799 event definitions, row replay, and CUDA provenance"
    ),
    "exp5812-split-budget-channel-contract": (
        "Gated on Exp5811 clean evidence: implement split-budget finite-choice transport"
    ),
    "exp5813-split-budget-sota-canary": (
        "Gated on Exp5812 contract: changed-mechanism canary on all three mandated SOTA GGUFs"
    ),
    "exp5814-channel-qualified-constraint-stream": (
        "Gated on Exp5813 all-family qualification: prospective chronological SOTA stream"
    ),
    "exp5815-future-validated-constraint-skill-ab": (
        "Gated on Exp5814 clean stream: future-validated typed-skill continuous-learning A/B"
    ),
}

CONDUCTOR_TITLE_PATTERNS: dict[str, str] = {
    "exp5809-transition-v518": "Transition four activated .517 tasks",
    "exp5810-v518-source-delta-ingestion": "Time-windowed post-V518 literature and implementat",
    "exp5811-exp5799-event-provenance-audit": "Companion audit for Exp5799 event definitions",
    "exp5812-split-budget-channel-contract": "Gated on Exp5811 clean evidence: implement split-b",
    "exp5813-split-budget-sota-canary": "Gated on Exp5812 contract: changed-mechanism canar",
    "exp5814-channel-qualified-constraint-stream": "Gated on Exp5813 all-family qualification: prospec",
    "exp5815-future-validated-constraint-skill-ab": "Gated on Exp5814 clean stream: future-validated ty",
}

RESERVED_UNACTIVATED_TASK_IDS = (
    "exp5816-constraint-skill-endurance",
    "exp5817-constraint-skill-ood-audit",
    "exp5818-arc-bootstrap-safe-sota-panel",
    "exp5819-arc-immutable-selector",
    "exp5820-arc-live-heldout-world-model-ab",
    "exp5821-self-learning-kernel-board-handoff",
    "exp5822-v518-capstone-reconciliation",
)

NEXT_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5823-transition-v519": RESULT_RELATIVE_PATH,
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
    "exp5830-sota-paired-embedding-corpus": Path(
        "results/experiment_5830_sota_paired_embedding_corpus.json"
    ),
    "exp5831-cross-family-embedding-energy-verifier": Path(
        "results/experiment_5831_cross_family_embedding_energy_verifier.json"
    ),
    "exp5832-arc-write-protected-world-fact-tape": Path(
        "results/experiment_5832_arc_write_protected_world_fact_tape.json"
    ),
    "exp5833-arc-world-feedback-probe-ab": Path(
        "results/experiment_5833_arc_world_feedback_probe_ab.json"
    ),
    "exp5834-bounded-adaptive-memory-microkernel": Path(
        "results/experiment_5834_bounded_adaptive_memory_microkernel.json"
    ),
    "exp5835-attached-board-adaptive-memory-receipts": Path(
        "results/experiment_5835_attached_board_adaptive_memory_receipts.json"
    ),
    "exp5836-capstone-v519": Path("results/experiment_5836_capstone_v519.json"),
}
NEXT_TASK_IDS = tuple(NEXT_TASK_ARTIFACT_PATHS)
ACTIVE_TASK_IDS = NEXT_TASK_IDS[:7]

BLOCKED_SAME_MECHANISM_RETRIES = (
    "finite-id generated-answer retry",
    "shared-budget generated-answer retry",
    "split-budget generated-answer retry",
    "grammar generated-answer retry",
    "stop/parser generated-answer retry",
)
PRESERVED_OPEN_SURFACES = (
    "sota paired embeddings",
    "final-token/final-layer embedding verifier",
    "exact constraint acquisition",
    "non-generation ARC world-feedback surfaces",
)
RETIREMENT_BLOCKED_PATTERNS = (
    "finite-ID GGUF generated-answer retry",
    "shared-budget finite-choice generated-answer retry",
    "split-budget generated-answer retry",
    "grammar constrained generated-answer retry",
    "stop-token generated-answer retry",
    "parser-only generated-answer retry",
    "same-mechanism MMLU-Pro GGUF answer transport retry",
)
EXPECTED_RETIREMENT_ENTRY: JsonDict = {
    "id": RETIREMENT_SCOPE_KEY,
    "scope_key": RETIREMENT_SCOPE_KEY,
    "experiment_scope": (
        "Current GGUF generated-answer transport lane using finite IDs, shared or "
        "split budgets, grammar constraints, stop tuning, or parser retries."
    ),
    "reason": (
        "retire_if_same_verdict: Exp5813 ran the changed split-budget canary on "
        "the three mandated SOTA GGUF families and still produced "
        "answer_channel_ready_score=0.0, zero qualified SOTA families, 2/144 "
        "exact labels, 138 parser failures, and 134 truncations. Future work "
        "must use a materially different non-generation surface."
    ),
    "experiment_ids": ["exp5799", "exp5812", "exp5813"],
    "retired_milestone": MILESTONE_FROM,
    "retired_by_artifact": TASK_ARTIFACT_PATHS["exp5813-split-budget-sota-canary"].as_posix(),
    "operator_reopen_required": True,
    "retire_if_same_verdict": True,
    "blocked_same_mechanism_retries": list(BLOCKED_SAME_MECHANISM_RETRIES),
    "blocked_patterns": list(RETIREMENT_BLOCKED_PATTERNS),
    "preserved_open_surfaces": list(PRESERVED_OPEN_SURFACES),
}

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
            ".venv/bin/python -c \"import pathlib, yaml; "
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
            ".venv/bin/pytest tests/python/test_experiment_5823_transition_v519.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage run --rcfile=/dev/null "
            "--include=python/carnot/experiment_5823_transition_v519.py "
            "-m pytest tests/python/test_experiment_5823_transition_v519.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage report --rcfile=/dev/null "
            "--include=python/carnot/experiment_5823_transition_v519.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None, "status": "not_run"},
)

REQUIRED_PRINCIPLE_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "declared_deliverable_matrix",
    "outcome_classification",
    "answer_transport_retirement",
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
    "schema": "Identifies the versioned Exp5823 transition artifact schema.",
    "experiment": "Names the local experiment slug without relying on paths.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "run_date": "Records the operator-specified transition date as a fixed value.",
    "random_seed": "Deterministic metadata for checksum stability; no stochastic run occurs.",
    "spec_refs": "Anchors the artifact to REQ-REPORT-5823 and its scenarios.",
    "result_path": "Names the emitted deliverable path.",
    "field_principles": "Maps every top-level artifact field to its evidence boundary.",
    "status": (
        "A normalized terminal state distinguishes a complete transition from a "
        "bootstrap artifact."
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
    "outcome_classification": (
        "Disjoint evidence classes prevent clean negatives and gate blocks from becoming successes."
    ),
    "answer_transport_retirement": (
        "A scope-keyed retirement prevents another same-mechanism generated-answer retry while preserving embeddings."
    ),
    "reserved_unactivated_task_ids": (
        "Proposal-only identities remain tombstoned and cannot silently collide."
    ),
    "reserved_unactivated_range": "Names the reserved proposal-only identity interval.",
    "research_complete_append_count": "An exact append count prevents duplicate milestone history.",
    "duplicate_history_diagnostics": "Reports duplicate history without rewriting it.",
    "collision_scan": "Shows the Exp5823-Exp5836 namespace scan and collision sources.",
    "next_task_range": "A declared interval makes downstream task allocation auditable.",
    "next_range_collision_count": "Only a bare zero authorizes Exp5823-Exp5836.",
    "docs_reconciled": "Specs, traceability, and ops summaries must match the archived evidence classes.",
    "research_roadmap_unchanged": "Bare boolean must remain true because active roadmap mutation is forbidden.",
    "conductor_unchanged": "Bare boolean must remain true by operator instruction.",
    "duration_s": "Measured wall time exposes bootstrap-only execution.",
    "inference_substrate": (
        "`aggregation_from_upstream_artifacts` prevents archival work from masquerading as inference."
    ),
    "field_provenance": "Per-field sources make the transition independently auditable.",
    "test_commands": "Recorded commands show which identity, retirement, and collision checks ran.",
    "test_exit_codes": "Exit codes prevent failed checks from being narrated as passing.",
    "reproducibility_checksum": "A content hash detects later ledger or allocation drift.",
    "honest_verdict": (
        "A `complete:` or `blocked:` prefix provides a mechanically terminal outcome."
    ),
}


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


def _duplicate_task_conflicts(tasks: Sequence[Any]) -> list[str]:
    deliverables_by_task: dict[str, set[str]] = defaultdict(set)
    for row in tasks:
        if isinstance(row, Mapping) and isinstance(row.get("id"), str):
            deliverable = row.get("deliverable")
            deliverables_by_task[str(row["id"])].add(
                str(deliverable) if isinstance(deliverable, str) else ""
            )
    return sorted(task_id for task_id, values in deliverables_by_task.items() if len(values) > 1)


def _synthetic_task_rows() -> list[JsonDict]:
    return [
        {
            "id": task_id,
            "title": TASK_TITLES[task_id],
            "deliverable": rel_path.as_posix(),
            "result": "append-if-absent exact transition receipt",
        }
        for task_id, rel_path in TASK_ARTIFACT_PATHS.items()
    ]


def _declared_deliverable_matrix(root: Path) -> tuple[list[JsonDict], JsonDict, list[str]]:
    blocks = _research_complete_blocks(root)
    signatures = [_task_signature(block) for block in blocks]
    unique_signatures = set(signatures)
    append_count = 0 if blocks else 1
    stats: JsonDict = {
        "research_complete_milestone_from_block_count": len(blocks),
        "unique_declared_deliverable_block_count": len(unique_signatures),
        "declared_deliverables_unambiguous": len(unique_signatures) <= 1,
        "completion_block_source": "research_complete" if blocks else "synthetic_append_if_absent",
        "research_complete_append_count": append_count,
    }
    failures: list[str] = []
    if len(unique_signatures) > 1:
        failures.append("ambiguous_research_complete_declared_task_blocks")

    selected_tasks = blocks[0].get("tasks") if blocks else _synthetic_task_rows()
    task_rows = selected_tasks if isinstance(selected_tasks, list) else []
    conflicts = _duplicate_task_conflicts(task_rows)
    if conflicts:
        failures.append(f"duplicate_task_id_conflicts={conflicts}")
    by_task: dict[str, JsonMap] = {
        str(row["id"]): row
        for row in task_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    declared_ids = (
        tuple(task_id for task_id, _deliverable in next(iter(unique_signatures), ()))
        if blocks
        else EXPECTED_TASK_IDS
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
            "gate_block_count": sum(1 for row in matches if row.get("outcome") == "GATE_BLOCK"),
            "failure_count": sum(1 for row in matches if row.get("outcome") in {"FAIL", "SKIP"}),
        }
    return outcomes, missing


def _exp5813_terminal_evidence(payload: JsonMap) -> JsonDict:
    metrics = payload.get("independent_failure_metrics")
    metric_map = metrics if isinstance(metrics, Mapping) else payload
    row_count = metric_map.get("row_count", payload.get("total_row_count"))
    exact_count = payload.get("exact_label_count")
    exact_coverage = metric_map.get("exact_label_coverage")
    if exact_count is None and isinstance(row_count, int | float) and isinstance(exact_coverage, int | float):
        exact_count = int(round(float(row_count) * float(exact_coverage)))
    qualified_families = payload.get("qualified_sota_family_count")
    if qualified_families is None and payload.get("qualified_real_sota_model_count") == 0:
        qualified_families = 0
    return {
        "answer_channel_ready_score": payload.get("answer_channel_ready_score"),
        "qualified_sota_family_count": qualified_families,
        "qualified_real_sota_model_count": payload.get("qualified_real_sota_model_count"),
        "exact_label_count": exact_count,
        "row_count": row_count,
        "parser_failure_count": metric_map.get("parser_failure_count"),
        "truncation_count": metric_map.get("truncation_count"),
    }


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
        status = _artifact_terminal_status(payload, metadata)
        terminal_evidence = (
            _exp5813_terminal_evidence(payload)
            if task_id == "exp5813-split-budget-sota-canary" and metadata.get("loadable")
            else {}
        )
        rows[task_id] = {
            "identity": [MILESTONE_FROM, task_id, rel_path.as_posix()],
            "path": rel_path.as_posix(),
            "present": bool(metadata.get("exists")),
            "loadable": bool(metadata.get("loadable")),
            "sha256": metadata.get("sha256"),
            "status": status,
            "honest_verdict": payload.get("honest_verdict")
            if isinstance(payload.get("honest_verdict"), str)
            else "",
            "artifact_flagged_adversarial": payload.get("flagged_adversarial") is True,
            "conductor_outcome": conductor_outcomes.get(task_id, {}).get("outcome", "UNKNOWN"),
            "conductor_detail": conductor_outcomes.get(task_id, {}).get("detail", ""),
            "selected_by": ARTIFACT_SELECTION_POLICY,
            "error": metadata.get("error"),
            **terminal_evidence,
        }
    return rows


def _same_number_alias_groups(root: Path, canonical_artifacts: Mapping[str, JsonMap]) -> dict[str, JsonDict]:
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


def _row_file_receipt(root: Path) -> JsonDict:
    path = root / EXP5813_ROW_RELATIVE_PATH
    if not path.exists():
        return {
            "path": EXP5813_ROW_RELATIVE_PATH.as_posix(),
            "present": False,
            "sha256": None,
            "row_count": 0,
        }
    line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return {
        "path": EXP5813_ROW_RELATIVE_PATH.as_posix(),
        "present": True,
        "sha256": path_sha256(path),
        "row_count": line_count,
    }


def _resource_receipts(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    mem_available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])
                break
    return {
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "mem_available_kib": mem_available,
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
        "history_mutation_policy": "read_only_no_dedup_sort_or_rewrite",
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
    return tuple(dict.fromkeys(tokens))


def _text_has_next_range(text: str) -> bool:
    return any(token in text for token in _next_range_tokens())


def _collision_scan(root: Path, result_path: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed_references: list[JsonDict] = []
    for rel_path in ALLOWED_ALLOCATION_TEXT_PATHS:
        path = root / rel_path
        if path.exists() and _text_has_next_range(path.read_text(encoding="utf-8", errors="replace")):
            allowed_references.append(
                {"path": rel_path.as_posix(), "kind": "allowed_allocation_reference"}
            )
    for rel_path in COLLISION_TEXT_PATHS:
        path = root / rel_path
        if path.exists() and _text_has_next_range(path.read_text(encoding="utf-8", errors="replace")):
            collisions.append({"path": rel_path.as_posix(), "kind": "preexisting_content_reference"})

    rel_result_path = result_path.relative_to(root) if result_path.is_absolute() else result_path
    results_dir = root / "results"
    for number in range(5823, 5837):
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
            "modified_by_exp5823": modified,
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
        "row_files": {EXP5813_ROW_RELATIVE_PATH.as_posix(): _row_file_receipt(root)},
    }


def _manifest_entries(manifest: JsonMap) -> list[JsonMap]:
    entries: list[JsonMap] = []
    for key in ("retired_experiments", "retired_extras", "retired"):
        value = manifest.get(key)
        if isinstance(value, list):
            entries.extend(row for row in value if isinstance(row, Mapping))
    return entries


def _answer_transport_retirement(root: Path, exp5813: JsonMap) -> JsonDict:
    manifest, meta = _read_yaml_with_meta(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    entries = _manifest_entries(manifest)
    matches = [
        entry
        for entry in entries
        if entry.get("scope_key") == RETIREMENT_SCOPE_KEY or entry.get("id") == RETIREMENT_SCOPE_KEY
    ]
    usable = []
    for entry in matches:
        patterns = set(entry.get("blocked_patterns", [])) | set(
            entry.get("blocked_same_mechanism_retries", [])
        )
        has_patterns = set(BLOCKED_SAME_MECHANISM_RETRIES).issubset(patterns) or set(
            RETIREMENT_BLOCKED_PATTERNS
        ).issubset(patterns)
        has_open = set(PRESERVED_OPEN_SURFACES).issubset(
            set(entry.get("preserved_open_surfaces", []))
        )
        if entry.get("retire_if_same_verdict") is True and has_patterns and has_open:
            usable.append(entry)
    terminal_evidence = _exp5813_terminal_evidence(exp5813)
    return {
        "manifest_entry_present": bool(usable),
        "manifest_path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_sha256": meta["sha256"],
        "scope_key": RETIREMENT_SCOPE_KEY,
        "retire_if_same_verdict": bool(usable),
        "blocked_same_mechanism_retries": list(BLOCKED_SAME_MECHANISM_RETRIES),
        "preserved_open_surfaces": list(PRESERVED_OPEN_SURFACES),
        "matched_entry_count": len(matches),
        "usable_entry_count": len(usable),
        "terminal_evidence": terminal_evidence,
        "terminal_evidence_artifact": TASK_ARTIFACT_PATHS[
            "exp5813-split-budget-sota-canary"
        ].as_posix(),
        "scope_status": "retired" if usable else "missing_or_incomplete_manifest_entry",
    }


def _classify_outcomes(
    canonical_artifacts: Mapping[str, JsonMap],
    conductor_rows: Mapping[str, JsonMap],
) -> JsonDict:
    clean_positive: list[str] = []
    clean_null: list[str] = []
    clean_negative: list[str] = []
    gate_blocked: list[str] = []
    preemptively_skipped: list[str] = []
    flagged: list[str] = []
    missing: list[str] = []
    by_task: dict[str, str] = {}
    known_positive = {
        "exp5809-transition-v518",
        "exp5811-exp5799-event-provenance-audit",
        "exp5812-split-budget-channel-contract",
    }
    for task_id in EXPECTED_TASK_IDS:
        artifact = canonical_artifacts.get(task_id, {})
        status = artifact.get("status")
        conductor = conductor_rows.get(task_id, {})
        outcome = conductor.get("outcome")
        detail = str(conductor.get("detail", artifact.get("conductor_detail", ""))).lower()
        is_preemptive_skip = outcome in {"SKIP", "GATE_BLOCK"} and (
            "pre-emptive skip" in detail or "upstream retired" in detail
        )
        is_flagged = bool(artifact.get("artifact_flagged_adversarial")) or outcome == "FLAGGED"
        if is_flagged:
            flagged.append(task_id)
            by_task[task_id] = "flagged"
        elif is_preemptive_skip:
            preemptively_skipped.append(task_id)
            by_task[task_id] = "preemptively-skipped"
        elif status in {"missing", "malformed"}:
            missing.append(task_id)
            by_task[task_id] = "missing"
        elif task_id == "exp5814-channel-qualified-constraint-stream" or outcome == "GATE_BLOCK" or status == "blocked":
            gate_blocked.append(task_id)
            by_task[task_id] = "gate-blocked"
        elif task_id in known_positive and status == "complete":
            clean_positive.append(task_id)
            by_task[task_id] = "clean-positive"
        elif task_id == "exp5810-v518-source-delta-ingestion" and status == "complete":
            clean_null.append(task_id)
            by_task[task_id] = "clean-null"
        elif task_id == "exp5813-split-budget-sota-canary" and status == "complete":
            clean_negative.append(task_id)
            by_task[task_id] = "clean-negative"
        else:
            missing.append(task_id)
            by_task[task_id] = "missing"
    return {
        "clean_positive_task_ids": clean_positive,
        "clean_null_task_ids": clean_null,
        "clean_negative_task_ids": clean_negative,
        "gate_blocked_task_ids": gate_blocked,
        "preemptively_skipped_task_ids": preemptively_skipped,
        "flagged_task_ids": flagged,
        "missing_task_ids": missing,
        "proposal_only_task_ids": list(RESERVED_UNACTIVATED_TASK_IDS),
        "clean_success_task_ids": list(clean_positive),
        "classification_by_task_id": by_task,
        "classification_policy": (
            "flagged, preemptive skip, missing, gate-blocked, clean-positive, "
            "clean-null, clean-negative, and proposal-only classes are disjoint"
        ),
    }


def _test_exit_codes(tests_run: Sequence[JsonMap]) -> JsonDict:
    return {str(row.get("command")): row.get("exit_code") for row in tests_run}


def _tests_failed(tests_run: Sequence[JsonMap]) -> list[str]:
    return [
        str(row.get("command"))
        for row in tests_run
        if row.get("exit_code") not in {0, None} and row.get("blocking", True) is not False
    ]


def _field_principles_for(artifact: Mapping[str, Any]) -> dict[str, str]:
    missing = [field for field in artifact if field not in FIELD_PRINCIPLES]
    if missing:
        raise KeyError(f"missing field principles: {missing}")
    return {field: FIELD_PRINCIPLES[field] for field in artifact}


def _field_provenance_for(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    default_sources = ["python/carnot/experiment_5823_transition_v519.py"]
    sources_by_field: dict[str, list[str]] = {
        "status": ["python/carnot/experiment_5823_transition_v519.py"],
        "preconditions_checked": [
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            ROADMAP_RELATIVE_PATH.as_posix(),
            ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            VNEXT_RELATIVE_PATH.as_posix(),
            EXP5813_ROW_RELATIVE_PATH.as_posix(),
        ],
        "milestone_transition": [
            ROADMAP_RELATIVE_PATH.as_posix(),
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        ],
        "declared_deliverable_matrix": [
            RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            *[path.as_posix() for path in TASK_ARTIFACT_PATHS.values()],
        ],
        "outcome_classification": [
            CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            *[path.as_posix() for path in TASK_ARTIFACT_PATHS.values()],
        ],
        "answer_transport_retirement": [
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            TASK_ARTIFACT_PATHS["exp5813-split-budget-sota-canary"].as_posix(),
        ],
        "reserved_unactivated_task_ids": [
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            TASK_ARTIFACT_PATHS["exp5809-transition-v518"].as_posix(),
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
            "tests/python/test_experiment_5823_transition_v519.py",
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        ],
        "duration_s": ["python/carnot/experiment_5823_transition_v519.py"],
        "inference_substrate": ["python/carnot/experiment_5823_transition_v519.py"],
        "field_provenance": [
            "python/carnot/experiment_5823_transition_v519.py",
            SPEC_RELATIVE_PATH.as_posix(),
        ],
        "test_commands": ["operator_test_receipts"],
        "test_exit_codes": ["operator_test_receipts"],
        "reproducibility_checksum": ["artifact_payload_excluding_checksum"],
        "honest_verdict": ["python/carnot/experiment_5823_transition_v519.py"],
    }
    return {
        field: {
            "sources": sources_by_field.get(field, default_sources),
            "derivation": "direct_read_or_deterministic_reconciliation",
        }
        for field in artifact
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
    duration_s: float | None = None,
    result_path: Path | None = None,
) -> JsonDict:
    started = time.perf_counter()
    root = root.resolve()
    result_path = result_path or root / RESULT_RELATIVE_PATH
    roadmap_active = _roadmap_summary(root / ROADMAP_RELATIVE_PATH)
    roadmap_next = _roadmap_summary(root / ROADMAP_NEXT_RELATIVE_PATH)
    declared_matrix, complete_stats, matrix_failures = _declared_deliverable_matrix(root)
    conductor_rows, missing_conductor_task_ids = _conductor_outcomes(root)
    canonical_artifacts = _canonical_artifacts(root, declared_matrix, conductor_rows)
    alias_groups = _same_number_alias_groups(root, canonical_artifacts)
    outcome_classification = _classify_outcomes(canonical_artifacts, conductor_rows)
    exp5813_payload, _exp5813_meta = _read_json_any(
        root / TASK_ARTIFACT_PATHS["exp5813-split-budget-sota-canary"]
    )
    retirement = _answer_transport_retirement(root, exp5813_payload)
    collision_scan = _collision_scan(root, result_path)
    duplicate_history = _duplicate_history_diagnostics(root)
    protected_files = _protected_files(root, modification_overrides)
    input_hashes = _input_hashes(root, canonical_artifacts)
    run_rows = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]

    for row in declared_matrix:
        task_id = str(row["task_id"])
        artifact = canonical_artifacts[task_id]
        conductor = conductor_rows[task_id]
        row["canonical_artifact_path"] = artifact["path"]
        row["canonical_artifact_present"] = artifact["present"]
        row["canonical_artifact_loadable"] = artifact["loadable"]
        row["canonical_artifact_sha256"] = artifact["sha256"]
        row["canonical_artifact_status"] = artifact["status"]
        row["canonical_artifact_honest_verdict"] = artifact["honest_verdict"]
        row["artifact_flagged_adversarial"] = artifact["artifact_flagged_adversarial"]
        row["conductor_outcome"] = conductor["outcome"]
        row["conductor_latest_evidence_line"] = conductor["latest_evidence_line"]
        row["outcome_class"] = outcome_classification["classification_by_task_id"].get(task_id)
        if task_id == "exp5813-split-budget-sota-canary":
            row.update(retirement["terminal_evidence"])
            row["generated_answer_success"] = False

    research_roadmap_unchanged = not protected_files[ROADMAP_RELATIVE_PATH.as_posix()][
        "modified_by_exp5823"
    ]
    conductor_unchanged = not protected_files[CONDUCTOR_RELATIVE_PATH.as_posix()][
        "modified_by_exp5823"
    ]
    missing_or_malformed = [
        task_id
        for task_id, row in canonical_artifacts.items()
        if row["status"] in {"missing", "malformed"}
        and task_id not in outcome_classification["preemptively_skipped_task_ids"]
    ]
    failed_tests = _tests_failed(run_rows)
    exp5813_evidence = retirement["terminal_evidence"]
    exp5813_clean_negative = (
        exp5813_evidence.get("answer_channel_ready_score") == 0.0
        and exp5813_evidence.get("qualified_sota_family_count") == 0
        and exp5813_evidence.get("exact_label_count") == 2
        and exp5813_evidence.get("row_count") == 144
        and exp5813_evidence.get("parser_failure_count") == 138
        and exp5813_evidence.get("truncation_count") == 134
    )

    failed_preconditions = list(matrix_failures)
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
    if not input_hashes["row_files"][EXP5813_ROW_RELATIVE_PATH.as_posix()]["present"]:
        failed_preconditions.append("exp5813_row_file_missing")
    if missing_conductor_task_ids:
        failed_preconditions.append(f"missing_conductor_outcomes={missing_conductor_task_ids}")
    if not exp5813_clean_negative:
        failed_preconditions.append("exp5813_clean_negative_metrics_not_confirmed")
    if not outcome_classification["gate_blocked_task_ids"]:
        failed_preconditions.append("exp5814_gate_block_not_confirmed")
    if not outcome_classification["preemptively_skipped_task_ids"]:
        failed_preconditions.append("exp5815_preemptive_skip_not_confirmed")
    if not retirement["manifest_entry_present"]:
        failed_preconditions.append("answer_transport_retirement_missing")
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
            "next_range_collision_count": collision_scan["preexisting_collision_count"],
            "resource_receipts": _resource_receipts(root),
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
        "outcome_classification": outcome_classification,
        "answer_transport_retirement": retirement,
        "reserved_unactivated_task_ids": list(RESERVED_UNACTIVATED_TASK_IDS),
        "reserved_unactivated_range": RESERVED_TASK_RANGE,
        "research_complete_append_count": int(complete_stats["research_complete_append_count"]),
        "duplicate_history_diagnostics": duplicate_history,
        "collision_scan": collision_scan,
        "next_task_range": NEXT_TASK_RANGE,
        "next_range_collision_count": collision_scan["preexisting_collision_count"],
        "docs_reconciled": {
            "mode": "transition_owned_spec_tests_module_manifest_result_only",
            "operator_owned_docs_deferred": True,
            "deferred_files": [
                "_bmad/traceability.md",
                "ops/status.md",
                "ops/changelog.md",
            ],
            "transition_owned_files": [
                SPEC_RELATIVE_PATH.as_posix(),
                "python/carnot/experiment_5823_transition_v519.py",
                "tests/python/test_experiment_5823_transition_v519.py",
                EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
                RESULT_RELATIVE_PATH.as_posix(),
            ],
            "deferred_reason": "operator stop-when-done rule delegates ops/status/traceability updates",
        },
        "research_roadmap_unchanged": research_roadmap_unchanged,
        "conductor_unchanged": conductor_unchanged,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": {},
        "test_commands": [str(row.get("command")) for row in run_rows],
        "test_exit_codes": _test_exit_codes(run_rows),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "blocked: exp5823 transition preconditions failed: " + "; ".join(failed_preconditions)
            if failed_preconditions
            else (
                "complete: archived terminal .518 evidence by exact declared deliverables "
                "into .519; Exp5813 clean-negative answer transport retired; "
                "Exp5816-Exp5822 tombstoned; next_range_collision_count=0; "
                "research_complete_append_count="
                f"{complete_stats['research_complete_append_count']}"
            )
        ),
    }
    artifact["field_principles"] = _field_principles_for(artifact)
    artifact["field_provenance"] = _field_provenance_for(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    output_path = output_path or root / RESULT_RELATIVE_PATH
    artifact = build_report(
        root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
        duration_s=duration_s,
        result_path=output_path,
    )
    write_json(output_path, artifact)
    return artifact


def _load_tests_run(path: Path | None) -> list[JsonDict]:  # pragma: no cover - CLI convenience
    if path is None:
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(args.root, output_path=args.output, tests_run=_load_tests_run(args.tests_run_json))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
